from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from .config import SelectionConfig


@dataclass(frozen=True)
class LayerSelection:
    strategy: str
    indices: tuple[int, ...]
    eligible_indices: tuple[int, ...]


def eligible_layer_indices(
    available_indices: Sequence[int], protected_prefix: int, protected_suffix: int
) -> tuple[int, ...]:
    ordered = tuple(sorted(int(index) for index in available_indices))
    stop = len(ordered) - protected_suffix if protected_suffix else len(ordered)
    eligible = ordered[protected_prefix:stop]
    if not eligible:
        raise ValueError("Boundary protection leaves no eligible MLP blocks")
    return eligible


def select_layers(
    available_indices: Sequence[int],
    config: SelectionConfig,
    bi_scores: Mapping[int, float] | None = None,
) -> LayerSelection:
    eligible = eligible_layer_indices(
        available_indices, config.protected_prefix, config.protected_suffix
    )
    eligible_set = set(eligible)

    if config.strategy == "manual":
        if not config.manual_indices:
            raise ValueError("Manual selection requires at least one layer index")
        invalid = set(config.manual_indices) - eligible_set
        if invalid:
            raise ValueError(f"Manual indices are unavailable or protected: {sorted(invalid)}")
        selected = list(dict.fromkeys(int(index) for index in config.manual_indices))
    else:
        if config.k > len(eligible):
            raise ValueError(f"Requested k={config.k}, but only {len(eligible)} layers are eligible")
        if config.strategy == "first_k":
            selected = list(eligible[: config.k])
        elif config.strategy == "random_k":
            selected = random.Random(config.seed).sample(list(eligible), config.k)
        elif config.strategy == "top_k_bi":
            if bi_scores is None:
                raise ValueError("top_k_bi selection requires BI scores")
            missing = [index for index in eligible if index not in bi_scores]
            if missing:
                raise ValueError(f"BI scores are missing eligible layers: {missing}")
            reverse = config.bi_order == "desc"
            selected = sorted(eligible, key=lambda index: bi_scores[index], reverse=reverse)[: config.k]
        else:
            raise ValueError(f"Unsupported selection strategy: {config.strategy}")

    if config.application_order == "layer":
        selected.sort()
    return LayerSelection(config.strategy, tuple(selected), eligible)

