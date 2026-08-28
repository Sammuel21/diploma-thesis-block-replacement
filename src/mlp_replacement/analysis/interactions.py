"""Selection and interaction helpers for controlled degradation studies."""

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class DegradationSelection:
    """Identify one exact subset evaluated in a degradation experiment."""

    strategy: str
    k: int
    indices: tuple[int, ...]
    seed: int | None = None


def bi_prefix(scores, eligible_indices, k, ascending=True):
    """Select an ordered low- or high-BI prefix from eligible layers."""

    eligible = tuple(sorted(int(index) for index in eligible_indices))
    if not 1 <= k <= len(eligible):
        raise ValueError(f"k must lie within [1, {len(eligible)}]")
    missing = [index for index in eligible if index not in scores]
    if missing:
        raise ValueError(f"BI scores are missing eligible layers: {missing}")
    ranked = sorted(eligible, key=lambda index: scores[index], reverse=not ascending)
    return tuple(sorted(ranked[:k]))


def random_subset(eligible_indices, k, seed):
    """Return a deterministic prefix of one seeded random layer ordering.

    Reusing a seed for increasing values of ``k`` therefore produces a nested
    random trajectory, matching the prefix interpretation used by BI ordering.
    """

    eligible = list(sorted(int(index) for index in eligible_indices))
    if not 1 <= k <= len(eligible):
        raise ValueError(f"k must lie within [1, {len(eligible)}]")
    random.Random(seed).shuffle(eligible)
    return tuple(sorted(eligible[:k]))


def build_degradation_selections(
    canonical_scores,
    adapted_scores,
    eligible_indices,
    k_values,
    random_seeds,
):
    """Build canonical, adapted, and random subset specifications."""

    selections = []
    for k in k_values:
        selections.extend(
            (
                DegradationSelection(
                    "canonical_bi_asc",
                    k,
                    bi_prefix(canonical_scores, eligible_indices, k, ascending=True),
                ),
                DegradationSelection(
                    "canonical_bi_desc",
                    k,
                    bi_prefix(canonical_scores, eligible_indices, k, ascending=False),
                ),
                DegradationSelection(
                    "mlp_bi_asc",
                    k,
                    bi_prefix(adapted_scores, eligible_indices, k, ascending=True),
                ),
                DegradationSelection(
                    "mlp_bi_desc",
                    k,
                    bi_prefix(adapted_scores, eligible_indices, k, ascending=False),
                ),
            )
        )
        selections.extend(
            DegradationSelection(
                "random",
                k,
                random_subset(eligible_indices, k, seed),
                seed,
            )
            for seed in random_seeds
        )
    return tuple(selections)


def degradation_interaction(observed_delta, indices, singleton_deltas):
    """Subtract the additive singleton prediction from observed degradation."""

    missing = [index for index in indices if index not in singleton_deltas]
    if missing:
        raise ValueError(f"Singleton deltas are missing layers: {missing}")
    additive_prediction = sum(float(singleton_deltas[index]) for index in indices)
    return float(observed_delta) - additive_prediction
