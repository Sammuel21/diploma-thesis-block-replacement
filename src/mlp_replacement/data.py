import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class TokenSequenceDataset(Dataset):
    """Expose fixed-length token sequences in the format expected by the model."""

    def __init__(self, sequences):
        self._sequences = tuple(sequence.detach().clone().long() for sequence in sequences)

    def __len__(self):
        return len(self._sequences)

    def __getitem__(self, index):
        input_ids = self._sequences[index]
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


@dataclass(frozen=True)
class DataLoaders:
    """Keep training, validation, recovery, and final-test roles separated."""

    calibration: DataLoader
    operator_validation: DataLoader
    recovery: DataLoader | None
    recovery_validation: DataLoader | None
    model_validation: DataLoader
    test: DataLoader | None


def tokenize_text(tokenizer, text):
    """Tokenize one text without padding or truncating it."""

    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return encoded.input_ids.squeeze(0)


def sample_partitioned_windows(records, tokenizer, partition_sizes, sequence_length, seed, text_column="text"):
    """Sample document-disjoint token windows and divide them by pipeline purpose."""

    total = sum(partition_sizes.values())
    if total == 0:
        return {name: [] for name in partition_sizes}
    if len(records) == 0:
        raise ValueError("Cannot sample calibration windows from an empty dataset")

    rng = random.Random(seed)
    sampled = []
    used_records = set()
    max_attempts = max(1_000, total * 200)
    attempts = 0

    while len(sampled) < total and attempts < max_attempts:
        attempts += 1
        record_index = rng.randrange(len(records))
        if record_index in used_records:
            continue
        text = str(records[record_index].get(text_column) or "")
        if not text.strip():
            continue

        ids = tokenize_text(tokenizer, text)
        if ids.numel() < sequence_length:
            continue
        start = rng.randint(0, ids.numel() - sequence_length)
        end = start + sequence_length
        used_records.add(record_index)
        sampled.append(ids[start:end].clone())

    if len(sampled) != total:
        raise RuntimeError(
            f"Only sampled {len(sampled)} of {total} requested token windows after "
            f"{attempts} attempts"
        )

    partitions = {}
    offset = 0
    for name, size in partition_sizes.items():
        partitions[name] = sampled[offset : offset + size]
        offset += size
    return partitions


def contiguous_token_windows(records, tokenizer, count, sequence_length, text_column="text"):
    """Split a continuous evaluation corpus into ordered fixed-length sequences."""

    text = "\n\n".join(str(record.get(text_column) or "") for record in records)
    ids = tokenize_text(tokenizer, text)
    available = ids.numel() // sequence_length
    selected_count = available if count is None else count
    if available < selected_count:
        raise ValueError(f"Evaluation corpus provides {available} sequences, but {count} were requested")
    return [
        ids[index * sequence_length : (index + 1) * sequence_length].clone()
        for index in range(selected_count)
    ]


def load_text_dataset(spec):
    """Load the dataset split described by a DatasetSpec."""

    from datasets import load_dataset

    kwargs = {
        "path": spec.path,
        "name": spec.name,
        "split": spec.split,
        "revision": spec.revision,
        "streaming": spec.streaming,
    }
    if spec.data_file is not None:
        kwargs["data_files"] = {spec.split: spec.data_file}
    return load_dataset(**kwargs)


def make_token_loader(sequences, batch_size):
    """Create a deterministic loader over fixed token sequences."""

    return DataLoader(TokenSequenceDataset(sequences), batch_size=batch_size, shuffle=False)


def build_data_loaders(tokenizer, config, include_recovery=True):
    """Build non-overlapping loaders for every configured experiment stage."""

    if config.calibration_source.streaming:
        raise ValueError("Random calibration sampling currently requires a non-streaming dataset")

    calibration_data = load_text_dataset(config.calibration_source)
    batch_size = config.batch_size
    partition_batches = {
        "calibration": config.num_calibration_batches,
        "operator_validation": config.num_operator_validation_batches,
        "recovery": config.num_recovery_batches if include_recovery else 0,
        "recovery_validation": config.num_recovery_validation_batches if include_recovery else 0,
    }
    partition_sizes = {name: batches * batch_size for name, batches in partition_batches.items()}
    windows = sample_partitioned_windows(
        calibration_data,
        tokenizer,
        partition_sizes,
        config.sequence_length,
        config.seed,
        config.calibration_source.text_column,
    )

    model_validation_data = load_text_dataset(config.model_validation_source)
    model_validation_sequences = contiguous_token_windows(
        model_validation_data,
        tokenizer,
        (
            config.num_model_validation_batches * batch_size
            if config.num_model_validation_batches is not None else None
        ),
        config.sequence_length,
        config.model_validation_source.text_column,
    )

    test_loader = None
    if config.num_test_batches is None or config.num_test_batches > 0:
        test_data = load_text_dataset(config.test_source)
        test_sequences = contiguous_token_windows(
            test_data,
            tokenizer,
            config.num_test_batches * batch_size if config.num_test_batches is not None else None,
            config.sequence_length,
            config.test_source.text_column,
        )
        test_loader = make_token_loader(test_sequences, batch_size)

    return DataLoaders(
        calibration=make_token_loader(windows["calibration"], batch_size),
        operator_validation=make_token_loader(windows["operator_validation"], batch_size),
        recovery=make_token_loader(windows["recovery"], batch_size) if include_recovery else None,
        recovery_validation=(
            make_token_loader(windows["recovery_validation"], batch_size) if include_recovery else None
        ),
        model_validation=make_token_loader(model_validation_sequences, batch_size),
        test=test_loader,
    )


@dataclass(frozen=True)
class PackedTokenCache:
    """Memory-map one finite, non-repeating packed token stream."""

    path: Path
    token_count: int
    sequence_length: int
    fingerprint: str

    def batch(self, token_offset, token_count, batch_size):
        """Read consecutive complete sequences without repeating cache content."""

        import numpy as np

        token_offset = int(token_offset)
        token_count = int(token_count)
        batch_size = int(batch_size)
        if token_offset < 0 or token_count < 1:
            raise ValueError("Packed-token offsets and counts must be positive")
        if token_count % self.sequence_length:
            raise ValueError("Packed-token reads must contain complete sequences")
        if token_offset + token_count > self.token_count:
            raise ValueError("Packed-token read exceeds the finite cache")
        sequence_count = token_count // self.sequence_length
        if sequence_count > batch_size:
            raise ValueError("Packed-token read exceeds the configured microbatch")
        tokens = np.memmap(self.path, mode="r", dtype=np.int32)
        selected = np.asarray(
            tokens[token_offset : token_offset + token_count], dtype=np.int64
        ).copy()
        input_ids = torch.from_numpy(selected).reshape(
            sequence_count, self.sequence_length
        )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


def packed_token_fingerprint(source, tokenizer_identity, token_count, sequence_length):
    """Fingerprint the provenance and exact extent of a packed token cache."""

    payload = {
        "source": source,
        "tokenizer": tokenizer_identity,
        "token_count": int(token_count),
        "sequence_length": int(sequence_length),
        "format": "signed-int32-packed-documents-with-eos-v1",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_or_open_packed_token_cache(
    records,
    tokenizer,
    path,
    token_count,
    sequence_length,
    source,
    tokenizer_identity,
    text_column="text",
):
    """Build or validate a compact token stream from fresh documents.

    Each source document is tokenized once and separated by EOS.  The stream is
    truncated exactly once at ``token_count`` and is never wrapped or repeated.
    A matching manifest makes subsequent sparsity runs and resumes reuse the same
    bytes without holding millions of Python tensors in memory.
    """

    import numpy as np

    path = Path(path)
    manifest_path = path.with_suffix(path.suffix + ".json")
    token_count = int(token_count)
    sequence_length = int(sequence_length)
    if token_count < 1 or token_count % sequence_length:
        raise ValueError("Packed-token count must be a positive number of sequences")
    fingerprint = packed_token_fingerprint(
        source, tokenizer_identity, token_count, sequence_length
    )
    if path.exists() or manifest_path.exists():
        if not path.is_file() or not manifest_path.is_file():
            raise ValueError("Packed-token cache and manifest must either both exist or both be absent")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_bytes = token_count * np.dtype(np.int32).itemsize
        if (
            manifest.get("fingerprint") != fingerprint
            or int(manifest.get("token_count", -1)) != token_count
            or path.stat().st_size != expected_bytes
        ):
            raise ValueError("Existing packed-token cache does not match this run")
        return PackedTokenCache(path, token_count, sequence_length, fingerprint)

    if tokenizer.eos_token_id is None:
        raise ValueError("Packed recovery data requires an EOS token")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    written = 0
    documents = 0
    try:
        with temporary.open("wb") as output:
            for record in records:
                text = str(record.get(text_column) or "")
                if not text.strip():
                    continue
                token_ids = tokenizer(
                    text,
                    add_special_tokens=False,
                    return_attention_mask=False,
                ).input_ids
                if not token_ids:
                    continue
                document = np.asarray(
                    [*token_ids, tokenizer.eos_token_id], dtype=np.int32
                )
                remaining = token_count - written
                selected = document[:remaining]
                selected.tofile(output)
                written += int(selected.size)
                documents += 1
                if written == token_count:
                    break
        if written != token_count:
            raise RuntimeError(
                f"Fresh recovery source supplied {written:,} of {token_count:,} "
                "tokens; the stream will not be repeated"
            )
        temporary.replace(path)
        manifest = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "token_count": token_count,
            "sequence_length": sequence_length,
            "dtype": "int32",
            "documents_consumed": documents,
            "source": source,
            "tokenizer": tokenizer_identity,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return PackedTokenCache(path, token_count, sequence_length, fingerprint)
