"""Whole-neuron width allocation for variable-width SwiGLU replacements."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AllocationBudget:
    """Describe a requested budget and its nearest whole-neuron realization."""

    requested_mlp_removal: float
    requested_removed_parameters: int
    realized_removed_parameters: int
    realized_mlp_removal: float
    retained_width: int
    parameter_step: int


def rank_normalize(values):
    """Return ascending average-rank percentiles without a pandas dependency."""

    items = [(int(layer), float(value)) for layer, value in values.items()]
    if not items:
        raise ValueError("At least one score is required")
    if len(items) == 1:
        return {items[0][0]: 0.0}
    ordered = sorted(items, key=lambda item: (item[1], item[0]))
    ranks = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_zero_based_rank = (start + end - 1) / 2
        for layer, _ in ordered[start:end]:
            ranks[layer] = average_zero_based_rank / (len(ordered) - 1)
        start = end
    return ranks


def allocate_ranked_swiglu_widths(
    original_widths,
    importance_scores,
    target_mlp_removal,
    hidden_size,
    temperature=1.0,
):
    """Allocate widths at the nearest representable aggregate parameter budget.

    This project-defined allocator converts ascending importance ranks to removal
    propensities with ``exp(-rank / temperature)``.  It rounds only the aggregate
    retained width, then reconciles per-layer floors by largest remainder.  The
    returned budget therefore states both the requested and realized removal.
    """

    widths = {int(layer): int(width) for layer, width in original_widths.items()}
    scores = {int(layer): float(score) for layer, score in importance_scores.items()}
    if set(widths) != set(scores):
        raise ValueError("Original widths and importance scores must cover the same layers")
    if not widths or any(width < 1 for width in widths.values()):
        raise ValueError("Every original width must be positive")
    if not 0.0 <= target_mlp_removal < 1.0:
        raise ValueError("target_mlp_removal must lie in [0, 1)")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if hidden_size < 1:
        raise ValueError("hidden_size must be positive")

    layers = tuple(sorted(widths))
    normalized = rank_normalize(scores)
    propensities = {
        layer: math.exp(-normalized[layer] / temperature) for layer in layers
    }
    parameter_step = 3 * int(hidden_size)
    original_total_width = sum(widths.values())
    original_parameters = original_total_width * parameter_step
    requested_removed = round(original_parameters * target_mlp_removal)
    requested_retained = original_parameters - requested_removed
    target_total_width = round(requested_retained / parameter_step)
    target_total_width = min(
        original_total_width,
        max(len(layers), target_total_width),
    )

    removal_width = original_total_width - target_total_width
    denominator = sum(widths[layer] * propensities[layer] for layer in layers)
    continuous = {
        layer: widths[layer]
        - removal_width * widths[layer] * propensities[layer] / denominator
        for layer in layers
    }
    allocated = {
        layer: min(widths[layer], max(1, math.floor(continuous[layer])))
        for layer in layers
    }
    current = sum(allocated.values())
    while current < target_total_width:
        choices = [layer for layer in layers if allocated[layer] < widths[layer]]
        if not choices:
            raise RuntimeError("Could not reconcile the retained-width budget")
        layer = max(choices, key=lambda item: continuous[item] - allocated[item])
        allocated[layer] += 1
        current += 1
    while current > target_total_width:
        choices = [layer for layer in layers if allocated[layer] > 1]
        if not choices:
            raise RuntimeError("Could not reconcile the retained-width budget")
        layer = max(choices, key=lambda item: allocated[item] - continuous[item])
        allocated[layer] -= 1
        current -= 1

    realized_removed = (original_total_width - target_total_width) * parameter_step
    budget = AllocationBudget(
        requested_mlp_removal=float(target_mlp_removal),
        requested_removed_parameters=requested_removed,
        realized_removed_parameters=realized_removed,
        realized_mlp_removal=realized_removed / original_parameters,
        retained_width=target_total_width,
        parameter_step=parameter_step,
    )
    rows = tuple(
        {
            "layer": layer,
            "raw_importance": scores[layer],
            "normalized_importance": normalized[layer],
            "removal_propensity": propensities[layer],
            "continuous_width": continuous[layer],
            "original_width": widths[layer],
            "replacement_width": allocated[layer],
            "replacement_width_ratio": allocated[layer] / widths[layer],
            "replacement_parameters": allocated[layer] * parameter_step,
            "realized_layer_removal": 1.0 - allocated[layer] / widths[layer],
        }
        for layer in layers
    )
    return rows, budget
