import random
from dataclasses import dataclass

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
