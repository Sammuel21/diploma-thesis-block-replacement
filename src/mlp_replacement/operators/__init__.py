"""Replacement operator families and local fitting helpers."""

from .modules import (
    BottleneckMLPReplacement,
    GatedMLPReplacement,
    HybridReplacement,
    LinearReplacement,
    LowRankLinearReplacement,
    MeanReplacement,
    ZeroReplacement,
    initialize_gated_mlp_from_teacher,
    initialize_low_rank_from_linear,
    initialize_low_rank_from_svd,
    linear_svd,
    swiglu_neuron_importance_scores,
)
from .training import (
    GatedReconstructionResult,
    fit_operator,
    fit_operator_fp32_detailed,
    fit_replacement_operator,
    fit_ridge_linear,
    initialize_gated_mlp_with_output_reconstruction,
)

__all__ = [
    "BottleneckMLPReplacement",
    "GatedMLPReplacement",
    "GatedReconstructionResult",
    "HybridReplacement",
    "LinearReplacement",
    "LowRankLinearReplacement",
    "MeanReplacement",
    "ZeroReplacement",
    "fit_operator",
    "fit_operator_fp32_detailed",
    "fit_replacement_operator",
    "fit_ridge_linear",
    "initialize_gated_mlp_from_teacher",
    "initialize_gated_mlp_with_output_reconstruction",
    "initialize_low_rank_from_linear",
    "initialize_low_rank_from_svd",
    "linear_svd",
    "swiglu_neuron_importance_scores",
]
