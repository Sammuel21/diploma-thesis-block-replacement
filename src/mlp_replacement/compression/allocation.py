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


@dataclass(frozen=True)
class SwiGLUWidthCurvePoint:
    """Describe one observed and monotonicized layer-width cost."""

    layer: int
    original_width: int
    replacement_width: int
    replacement_width_ratio: float
    replacement_parameters: int
    raw_teacher_kl: float
    monotone_teacher_kl: float
    retains_dense_module: bool
    has_output_bias: bool


@dataclass(frozen=True)
class DiscreteSwiGLUAllocation:
    """Return one curve-based whole-model allocation and its budget."""

    rows: tuple[dict, ...]
    requested_mlp_removal: float
    requested_retained_parameters: int
    realized_retained_parameters: int
    realized_mlp_removal: float
    predicted_teacher_kl: float
    boundary_layer: int | None


def nonincreasing_isotonic(values):
    """Pool adjacent violations so cost cannot rise with additional width."""

    blocks = []
    for index, value in enumerate(values):
        blocks.append([index, index + 1, float(value), 1])
        while len(blocks) >= 2 and blocks[-2][2] < blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            count = left[3] + right[3]
            mean = (left[2] * left[3] + right[2] * right[3]) / count
            blocks.append([left[0], right[1], mean, count])
    result = [0.0] * len(values)
    for start, end, mean, unused_count in blocks:
        result[start:end] = [mean] * (end - start)
    return tuple(result)


def build_swiglu_width_curve(
    layer,
    original_width,
    evaluations,
    hidden_size,
):
    """Build one monotone singleton-KL curve from explicit fitted widths.

    ``evaluations`` maps widths to dictionaries containing ``teacher_kl`` and
    optional ``has_output_bias``. A full-width point represents the retained
    dense module and therefore has the original parameter count and no bias.
    """

    layer = int(layer)
    original_width = int(original_width)
    hidden_size = int(hidden_size)
    if original_width < 1 or hidden_size < 1 or not evaluations:
        raise ValueError("Width curves require positive dimensions and points")
    normalized = []
    for width, values in evaluations.items():
        width = int(width)
        if not 1 <= width <= original_width:
            raise ValueError("Curve widths must lie within the dense MLP width")
        retains_dense = width == original_width
        has_bias = bool(values.get("has_output_bias", False)) and not retains_dense
        parameters = (
            3 * hidden_size * original_width
            if retains_dense
            else 3 * hidden_size * width + (hidden_size if has_bias else 0)
        )
        normalized.append(
            {
                "width": width,
                "teacher_kl": float(values["teacher_kl"]),
                "parameters": parameters,
                "retains_dense": retains_dense,
                "has_bias": has_bias,
            }
        )
    normalized.sort(key=lambda row: row["width"])
    if len({row["width"] for row in normalized}) != len(normalized):
        raise ValueError("Width-curve points must have unique widths")
    monotone = nonincreasing_isotonic(
        [row["teacher_kl"] for row in normalized]
    )
    return tuple(
        SwiGLUWidthCurvePoint(
            layer=layer,
            original_width=original_width,
            replacement_width=row["width"],
            replacement_width_ratio=row["width"] / original_width,
            replacement_parameters=row["parameters"],
            raw_teacher_kl=row["teacher_kl"],
            monotone_teacher_kl=monotone[index],
            retains_dense_module=row["retains_dense"],
            has_output_bias=row["has_bias"],
        )
        for index, row in enumerate(normalized)
    )


def allocate_discrete_swiglu_widths(
    curves,
    target_mlp_removal,
    hidden_size,
):
    """Allocate fitted widths with a multiple-choice knapsack."""

    hidden_size = int(hidden_size)
    if hidden_size < 1:
        raise ValueError("hidden_size must be positive")
    if not 0.0 <= float(target_mlp_removal) < 1.0:
        raise ValueError("target_mlp_removal must lie in [0, 1)")
    by_layer = {
        int(layer): tuple(
            sorted(points, key=lambda point: point.replacement_width)
        )
        for layer, points in curves.items()
    }
    if not by_layer or any(not points for points in by_layer.values()):
        raise ValueError("Every allocated layer requires a width curve")
    for layer, points in by_layer.items():
        if any(point.layer != layer for point in points):
            raise ValueError("Width-curve layer keys and points differ")
        if any(point.replacement_parameters % hidden_size for point in points):
            raise ValueError("Curve parameter counts must align to hidden size")

    original_parameters = sum(
        3 * hidden_size * points[0].original_width
        for points in by_layer.values()
    )
    requested_retained = original_parameters - round(
        original_parameters * float(target_mlp_removal)
    )
    budget_units = requested_retained // hidden_size
    states = {0: (0.0, ())}
    for layer in sorted(by_layer):
        next_states = {}
        for used_units, (cost, choices) in states.items():
            for point in by_layer[layer]:
                units = point.replacement_parameters // hidden_size
                total = used_units + units
                if total > budget_units:
                    continue
                candidate = (
                    cost + point.monotone_teacher_kl,
                    choices + (point,),
                )
                existing = next_states.get(total)
                candidate_widths = tuple(
                    item.replacement_width for item in candidate[1]
                )
                existing_widths = (
                    tuple(item.replacement_width for item in existing[1])
                    if existing is not None
                    else ()
                )
                if existing is None or (
                    candidate[0], candidate_widths
                ) < (existing[0], existing_widths):
                    next_states[total] = candidate
        if not next_states:
            raise ValueError("No discrete allocation fits the requested budget")
        states = next_states

    used_units, (unused_objective, selected) = min(
        states.items(),
        key=lambda item: (
            item[1][0],
            budget_units - item[0],
            tuple(point.replacement_width for point in item[1][1]),
        ),
    )
    remaining_parameters = requested_retained - used_units * hidden_size
    parameter_step = 3 * hidden_size
    extra_neurons = max(0, remaining_parameters // parameter_step)
    boundary_layer = None
    selected_by_layer = {point.layer: point for point in selected}
    if extra_neurons:
        choices = []
        for layer, point in selected_by_layer.items():
            wider = [
                candidate
                for candidate in by_layer[layer]
                if candidate.replacement_width > point.replacement_width
            ]
            if not wider:
                continue
            next_point = wider[0]
            capacity = next_point.replacement_width - point.replacement_width
            if capacity < extra_neurons:
                continue
            improvement = (
                point.monotone_teacher_kl - next_point.monotone_teacher_kl
            ) / capacity
            choices.append((-improvement, layer, point, next_point))
        if choices:
            unused_improvement, boundary_layer, point, next_point = min(choices)
            width = point.replacement_width + extra_neurons
            fraction = extra_neurons / (
                next_point.replacement_width - point.replacement_width
            )
            boundary_cost = point.monotone_teacher_kl + fraction * (
                next_point.monotone_teacher_kl - point.monotone_teacher_kl
            )
            has_bias = point.has_output_bias or next_point.has_output_bias
            selected_by_layer[boundary_layer] = SwiGLUWidthCurvePoint(
                layer=boundary_layer,
                original_width=point.original_width,
                replacement_width=width,
                replacement_width_ratio=width / point.original_width,
                replacement_parameters=(
                    3 * hidden_size * width + (hidden_size if has_bias else 0)
                ),
                raw_teacher_kl=boundary_cost,
                monotone_teacher_kl=boundary_cost,
                retains_dense_module=False,
                has_output_bias=has_bias,
            )

    rows = tuple(
        {
            "layer": layer,
            "original_width": point.original_width,
            "replacement_width": point.replacement_width,
            "replacement_width_ratio": point.replacement_width_ratio,
            "replacement_parameters": point.replacement_parameters,
            "predicted_teacher_kl": point.monotone_teacher_kl,
            "raw_teacher_kl": point.raw_teacher_kl,
            "retains_dense_module": point.retains_dense_module,
            "has_output_bias": point.has_output_bias,
            "is_boundary_width": layer == boundary_layer,
        }
        for layer, point in sorted(selected_by_layer.items())
    )
    realized_retained = sum(row["replacement_parameters"] for row in rows)
    if realized_retained > requested_retained:
        raise RuntimeError("Discrete allocator exceeded its parameter budget")
    return DiscreteSwiGLUAllocation(
        rows=rows,
        requested_mlp_removal=float(target_mlp_removal),
        requested_retained_parameters=requested_retained,
        realized_retained_parameters=realized_retained,
        realized_mlp_removal=1.0 - realized_retained / original_parameters,
        predicted_teacher_kl=sum(
            row["predicted_teacher_kl"] for row in rows
        ),
        boundary_layer=boundary_layer,
    )


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
        for layer, unused_score in ordered[start:end]:
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
