"""Replacement operator families and local fitting helpers."""

from .modules import (
    BottleneckMLPReplacement,
    GatedMLPReplacement,
    HybridReplacement,
    LinearReplacement,
    LowRankLinearReplacement,
    MeanReplacement,
    ZeroReplacement,
    initialize_low_rank_from_linear,
    initialize_low_rank_from_svd,
    linear_svd,
)
from .training import fit_operator, fit_replacement_operator, fit_ridge_linear

__all__ = [
    "BottleneckMLPReplacement",
    "GatedMLPReplacement",
    "HybridReplacement",
    "LinearReplacement",
    "LowRankLinearReplacement",
    "MeanReplacement",
    "ZeroReplacement",
    "fit_operator",
    "fit_replacement_operator",
    "fit_ridge_linear",
    "initialize_low_rank_from_linear",
    "initialize_low_rank_from_svd",
    "linear_svd",
]
