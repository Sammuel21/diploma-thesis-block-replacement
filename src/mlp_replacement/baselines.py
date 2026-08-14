"""Reusable baseline calculations for one MLP block."""

from dataclasses import dataclass

import torch

from .evaluation.footprint import parameter_footprint
from .evaluation.language_model import LanguageModelMetrics, evaluate_language_model
from .evaluation.operator import evaluate_operator
from .operators import (
    GatedMLPReplacement,
    MeanReplacement,
    ZeroReplacement,
    fit_operator,
    fit_ridge_linear,
)
from .operators.training import OperatorTrainingEpoch
from .surgery import count_state_elements, temporary_replacement


@dataclass(frozen=True)
class SingleBlockBaselineResult:
    """Collect baseline rows, fit histories, and shared reference information."""

    rows: tuple[dict, ...]
    training_histories: dict[str, tuple[OperatorTrainingEpoch, ...]]
    baseline_language_model: LanguageModelMetrics
    original_intermediate_width: int
    replacement_intermediate_width: int


def run_single_block_baselines(
    model,
    block,
    training_pairs,
    validation_pairs,
    model_validation_loader,
    operator_training,
    intermediate_width_ratio=0.5,
    linear_ridge=1e-4,
    max_model_validation_batches=None,
):
    """Fit and evaluate the fixed baseline suite for one prepared MLP block."""

    if not 0.0 < intermediate_width_ratio <= 1.0:
        raise ValueError("intermediate_width_ratio must lie within (0, 1]")
    if linear_ridge < 0:
        raise ValueError("linear_ridge must be non-negative")
    if training_pairs.hidden_size != validation_pairs.hidden_size:
        raise ValueError("Training and validation hidden sizes differ")
    if not hasattr(block.module, "up_proj"):
        raise ValueError("The baseline suite requires an MLP with an up_proj module")

    device = next(model.parameters()).device
    hidden_size = training_pairs.hidden_size
    original_intermediate_width = int(block.module.up_proj.out_features)
    replacement_intermediate_width = max(
        1,
        round(intermediate_width_ratio * original_intermediate_width),
    )

    original_footprint = parameter_footprint(block.module)
    original_parameters = original_footprint.parameters
    original_state_elements = count_state_elements(block.module)
    baseline_language_model = evaluate_language_model(
        model,
        model_validation_loader,
        device,
        max_model_validation_batches,
    )

    rows = [{
        "name": "original_mlp",
        "family": "teacher",
        "intermediate_width": original_intermediate_width,
        "parameters": original_parameters,
        "state_elements": original_state_elements,
        "theoretical_weight_bytes": original_footprint.theoretical_weight_bytes,
        "relative_block_parameters": 1.0,
        "parameter_reduction_percent": 0.0,
        "removed_parameters": 0,
        "local_mse": 0.0,
        "local_relative_mse": 0.0,
        "local_r2": 1.0,
        "local_cosine": 1.0,
        "local_norm_ratio": 1.0,
        "median_token_relative_error": 0.0,
        "p95_token_relative_error": 0.0,
        "loss": baseline_language_model.loss,
        "perplexity": baseline_language_model.perplexity,
        "delta_loss": 0.0,
        "delta_perplexity": 0.0,
        "recovery": False,
    }]
    training_histories = {}

    def append_candidate(name, family, module, history=(), intermediate_width=None):
        module = module.to(device)
        local = evaluate_operator(
            module,
            validation_pairs,
            device,
            operator_training.batch_size,
        )
        with temporary_replacement(model, block.index, module) as record:
            footprint = parameter_footprint(module)
            state_elements = count_state_elements(module)
            integrated = evaluate_language_model(
                model,
                model_validation_loader,
                device,
                max_model_validation_batches,
            )

        rows.append({
            "name": name,
            "family": family,
            "intermediate_width": intermediate_width,
            "parameters": footprint.parameters,
            "state_elements": state_elements,
            "theoretical_weight_bytes": footprint.theoretical_weight_bytes,
            "relative_block_parameters": footprint.parameters / original_parameters,
            "parameter_reduction_percent": 100.0 * (
                1.0 - footprint.parameters / original_parameters
            ),
            "removed_parameters": record.original_parameters - footprint.parameters,
            "local_mse": local.mse,
            "local_relative_mse": local.relative_mse,
            "local_r2": local.r2,
            "local_cosine": local.cosine_similarity,
            "local_norm_ratio": local.norm_ratio,
            "median_token_relative_error": local.median_token_relative_error,
            "p95_token_relative_error": local.p95_token_relative_error,
            "loss": integrated.loss,
            "perplexity": integrated.perplexity,
            "delta_loss": integrated.loss - baseline_language_model.loss,
            "delta_perplexity": (
                integrated.perplexity - baseline_language_model.perplexity
            ),
            "recovery": False,
        })
        training_histories[name] = tuple(history)

    append_candidate("zero", "constant_control", ZeroReplacement())
    append_candidate(
        "mean",
        "constant_control",
        MeanReplacement(training_pairs.targets.mean(dim=0)),
    )

    dense_linear = fit_ridge_linear(
        training_pairs,
        ridge=linear_ridge,
        bias=False,
        device=device,
    )
    append_candidate("dense_linear", "linear", dense_linear)

    dense_affine = fit_ridge_linear(
        training_pairs,
        ridge=linear_ridge,
        bias=True,
        device=device,
    )
    append_candidate("dense_affine", "affine", dense_affine)

    torch.manual_seed(operator_training.seed)
    narrow_swiglu = GatedMLPReplacement(
        hidden_size,
        replacement_intermediate_width,
        bias=operator_training.bias,
    )
    narrow_swiglu_fit = fit_operator(
        narrow_swiglu,
        training_pairs,
        validation_pairs,
        operator_training,
        device,
    )
    append_candidate(
        "narrow_swiglu",
        "gated_mlp",
        narrow_swiglu_fit.module,
        narrow_swiglu_fit.history,
        replacement_intermediate_width,
    )

    return SingleBlockBaselineResult(
        rows=tuple(rows),
        training_histories=training_histories,
        baseline_language_model=baseline_language_model,
        original_intermediate_width=original_intermediate_width,
        replacement_intermediate_width=replacement_intermediate_width,
    )
