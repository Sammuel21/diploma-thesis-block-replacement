from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig, DatasetSpec


class TokenSequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[torch.Tensor]):
        self._sequences = tuple(sequence.detach().clone().long() for sequence in sequences)

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids = self._sequences[index]
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


@dataclass(frozen=True)
class DataLoaders:
    calibration: DataLoader
    operator_validation: DataLoader
    recovery: DataLoader | None
    recovery_validation: DataLoader | None
    evaluation: DataLoader


def _token_ids(tokenizer, text: str) -> torch.Tensor:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return encoded.input_ids.squeeze(0)


def sample_partitioned_windows(
    records: Sequence[Mapping[str, object]],
    tokenizer,
    partition_sizes: Mapping[str, int],
    sequence_length: int,
    seed: int,
    text_column: str = "text",
) -> dict[str, list[torch.Tensor]]:
    """Sample non-identical token windows and divide them by pipeline purpose."""

    total = sum(partition_sizes.values())
    if total == 0:
        return {name: [] for name in partition_sizes}
    if len(records) == 0:
        raise ValueError("Cannot sample calibration windows from an empty dataset")

    rng = random.Random(seed)
    sampled: list[torch.Tensor] = []
    used_windows: set[tuple[int, int]] = set()
    max_attempts = max(1_000, total * 200)
    attempts = 0

    while len(sampled) < total and attempts < max_attempts:
        attempts += 1
        record_index = rng.randrange(len(records))
        text = str(records[record_index].get(text_column) or "")
        if not text.strip():
            continue

        ids = _token_ids(tokenizer, text)
        if ids.numel() < sequence_length:
            continue
        start = rng.randint(0, ids.numel() - sequence_length)
        key = (record_index, start)
        if key in used_windows:
            continue

        used_windows.add(key)
        sampled.append(ids[start : start + sequence_length].clone())

    if len(sampled) != total:
        raise RuntimeError(
            f"Only sampled {len(sampled)} of {total} requested token windows after "
            f"{attempts} attempts"
        )

    partitions: dict[str, list[torch.Tensor]] = {}
    offset = 0
    for name, size in partition_sizes.items():
        partitions[name] = sampled[offset : offset + size]
        offset += size
    return partitions


def contiguous_token_windows(
    records: Iterable[Mapping[str, object]],
    tokenizer,
    count: int,
    sequence_length: int,
    text_column: str = "text",
) -> list[torch.Tensor]:
    text = "\n\n".join(str(record.get(text_column) or "") for record in records)
    ids = _token_ids(tokenizer, text)
    available = ids.numel() // sequence_length
    if available < count:
        raise ValueError(f"Evaluation corpus provides {available} sequences, but {count} were requested")
    return [
        ids[index * sequence_length : (index + 1) * sequence_length].clone()
        for index in range(count)
    ]


def _load_dataset(spec: DatasetSpec):
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


def _loader(sequences: Sequence[torch.Tensor], batch_size: int) -> DataLoader:
    return DataLoader(TokenSequenceDataset(sequences), batch_size=batch_size, shuffle=False)


def build_data_loaders(tokenizer, config: DataConfig, include_recovery: bool = True) -> DataLoaders:
    if config.calibration_source.streaming:
        raise ValueError("Random calibration sampling currently requires a non-streaming dataset")

    calibration_data = _load_dataset(config.calibration_source)
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

    evaluation_data = _load_dataset(config.evaluation_source)
    evaluation_sequences = contiguous_token_windows(
        evaluation_data,
        tokenizer,
        config.num_evaluation_batches * batch_size,
        config.sequence_length,
        config.evaluation_source.text_column,
    )

    return DataLoaders(
        calibration=_loader(windows["calibration"], batch_size),
        operator_validation=_loader(windows["operator_validation"], batch_size),
        recovery=_loader(windows["recovery"], batch_size) if include_recovery else None,
        recovery_validation=(
            _loader(windows["recovery_validation"], batch_size) if include_recovery else None
        ),
        evaluation=_loader(evaluation_sequences, batch_size),
    )

