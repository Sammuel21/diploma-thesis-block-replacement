"""Headless migration of ``notebooks/model/swiglu.ipynb``."""

from __future__ import annotations

import argparse
import math
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from workflows.runs.model._common import (
    default_artifact_path,
    json_records,
    load_artifact,
    load_workflow_config,
    release_cuda,
    report_memory,
    require_new_path,
    resolve_path,
    start_run_log,
    write_artifact,
)

from mlp_replacement.analysis.screening import compute_bi_scores
from mlp_replacement.capture import collect_modules_io
from mlp_replacement.compression.recovery import (
    cache_teacher_logits,
    mean_cache_loss,
    recover_replacements,
)
from mlp_replacement.compression.surgery import temporary_replacements
from mlp_replacement.config import (
    CaptureConfig,
    DataConfig,
    DatasetSpec,
    ModelConfig,
    OperatorConfig,
    RecoveryConfig,
)
from mlp_replacement.data import (
    DataLoaders,
    build_data_loaders,
    contiguous_token_windows,
    load_text_dataset,
    make_token_loader,
    sample_partitioned_windows,
)
from mlp_replacement.evaluation.footprint import parameter_footprint
from mlp_replacement.evaluation.language_model import evaluate_language_model
from mlp_replacement.evaluation.operator import evaluate_operator
from mlp_replacement.model import discover_mlp_blocks, load_model_and_tokenizer
from mlp_replacement.operators import (
    GatedMLPReplacement,
    fit_operator,
    initialize_gated_mlp_from_teacher,
    swiglu_neuron_importance_scores,
)
from mlp_replacement.runlog import environment_record


WORKFLOW = "swiglu"
HISTORICAL_SCHEMA = 2
OPTIMIZED_SCHEMA = 3
PINNED_REVISION = "effd688a12921b4cc83e3312b6feb579f70f9c71"
DEFAULT_HISTORICAL_REFERENCE = Path(
    "data/results/notebook-model-study/swiglu-compression.json"
)
DEFAULT_CONFIG = Path("workflows/configs/model/swiglu.json")

SEED = 21
PROTECTED_PREFIX = 1
PROTECTED_SUFFIX = 1
TARGET_MLP_SPARSITY = 0.50
ALLOCATION_TEMPERATURE = 1.0
SPARSITY_RATIOS = (0.20, 0.30, 0.40, 0.50)
POLICIES = ("uniform", "canonical_bi", "residual_aware_mlp_bi")
OPTIMIZED_CALIBRATION_BATCHES = 384
OPTIMIZED_CAPTURE_GROUP_SIZE = 6
OPTIMIZED_METHODS = {
    "new_data_random": "random_weights",
    "complete_method": "importance_teacher_subset",
}
SEQUENCE_LENGTH = 128
BATCH_SIZE = 2
HISTORICAL_CALIBRATION_BATCHES = 48
OPERATOR_VALIDATION_BATCHES = 24
RECOVERY_BATCHES = 64
RECOVERY_VALIDATION_BATCHES = 24
MODEL_VALIDATION_BATCHES = 24
TEST_BATCHES = 0
OPERATOR_EPOCHS = 64
OPERATOR_LEARNING_RATE = 1e-3
OPERATOR_BATCH_SIZE = 2048
OPERATOR_WEIGHT_DECAY = 0.0
OPERATOR_SCHEDULER = "constant"
OPERATOR_EARLY_STOPPING_PATIENCE = 3
RECOVERY_EPOCHS = 1
RECOVERY_ENABLED = True
RECOVERY_LEARNING_RATE = 1e-5
RECOVERY_WEIGHT_DECAY = 0.0
RECOVERY_TEMPERATURE = 1.0
RECOVERY_CACHE_DTYPE = "float16"
RECOVERY_EARLY_STOPPING_PATIENCE = None
CAPTURE_DEVICE = "cpu"
HISTORICAL_CAPTURE_DTYPE = "float32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the model-wide variable-width SwiGLU allocation study"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Explicit notebook-equivalent workflow configuration",
    )
    parser.add_argument(
        "--stage",
        choices=("historical", "optimized", "all"),
        default="optimized",
        help="Notebook section to execute (default: optimized)",
    )
    parser.add_argument(
        "--historical-artifact",
        type=Path,
        default=DEFAULT_HISTORICAL_REFERENCE,
        help="Schema-2 control used by the optimized stage",
    )
    parser.add_argument("--historical-output", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def configure(settings: dict) -> None:
    """Make the checked-in JSON configuration authoritative for this process."""

    global PINNED_REVISION, SEED, PROTECTED_PREFIX, PROTECTED_SUFFIX
    global TARGET_MLP_SPARSITY, ALLOCATION_TEMPERATURE, SPARSITY_RATIOS
    global POLICIES, OPTIMIZED_CALIBRATION_BATCHES, OPTIMIZED_CAPTURE_GROUP_SIZE
    global OPTIMIZED_METHODS, SEQUENCE_LENGTH, BATCH_SIZE
    global HISTORICAL_CALIBRATION_BATCHES, OPERATOR_VALIDATION_BATCHES
    global RECOVERY_BATCHES, RECOVERY_VALIDATION_BATCHES
    global MODEL_VALIDATION_BATCHES, TEST_BATCHES, OPERATOR_EPOCHS
    global OPERATOR_LEARNING_RATE, OPERATOR_BATCH_SIZE, OPERATOR_WEIGHT_DECAY
    global OPERATOR_SCHEDULER, OPERATOR_EARLY_STOPPING_PATIENCE
    global RECOVERY_EPOCHS, RECOVERY_LEARNING_RATE, RECOVERY_WEIGHT_DECAY
    global RECOVERY_TEMPERATURE, RECOVERY_CACHE_DTYPE
    global RECOVERY_EARLY_STOPPING_PATIENCE, CAPTURE_DEVICE
    global HISTORICAL_CAPTURE_DTYPE
    global RECOVERY_ENABLED

    data = settings["data"]
    layers = settings["layers"]
    allocation = settings["allocation"]
    operator = settings["operator"]
    recovery = settings["recovery"]
    capture = settings["capture"]
    configured_policies = tuple(allocation["policies"])
    supported_policies = {
        "uniform",
        "canonical_bi",
        "residual_aware_mlp_bi",
    }
    if set(configured_policies) != supported_policies:
        raise ValueError(
            "SwiGLU policy configuration must contain uniform and both BI policies"
        )
    if capture["optimized_storage_dtype"] != "model_native":
        raise ValueError("Optimized SwiGLU capture must use model_native storage")

    PINNED_REVISION = settings["model_revision"]
    SEED = int(settings["seed"])
    SEQUENCE_LENGTH = int(data["sequence_length"])
    BATCH_SIZE = int(data["batch_size"])
    HISTORICAL_CALIBRATION_BATCHES = int(data["historical_calibration_batches"])
    OPTIMIZED_CALIBRATION_BATCHES = int(data["optimized_calibration_batches"])
    OPERATOR_VALIDATION_BATCHES = int(data["operator_validation_batches"])
    RECOVERY_BATCHES = int(data["recovery_batches"])
    RECOVERY_VALIDATION_BATCHES = int(data["recovery_validation_batches"])
    MODEL_VALIDATION_BATCHES = int(data["model_validation_batches"])
    TEST_BATCHES = int(data["test_batches"])
    PROTECTED_PREFIX = int(layers["protected_prefix"])
    PROTECTED_SUFFIX = int(layers["protected_suffix"])
    TARGET_MLP_SPARSITY = float(allocation["target_mlp_sparsity"])
    ALLOCATION_TEMPERATURE = float(allocation["temperature"])
    POLICIES = configured_policies
    SPARSITY_RATIOS = tuple(float(value) for value in allocation["historical_sparsity_ratios"])
    OPERATOR_EPOCHS = int(operator["epochs"])
    OPERATOR_LEARNING_RATE = float(operator["learning_rate"])
    OPERATOR_BATCH_SIZE = int(operator["batch_size"])
    OPERATOR_WEIGHT_DECAY = float(operator["weight_decay"])
    OPERATOR_SCHEDULER = operator["scheduler"]
    OPERATOR_EARLY_STOPPING_PATIENCE = operator["early_stopping_patience"]
    OPTIMIZED_METHODS = dict(operator["optimized_initializations"])
    RECOVERY_EPOCHS = int(recovery["epochs"])
    RECOVERY_ENABLED = bool(recovery["enabled"])
    RECOVERY_LEARNING_RATE = float(recovery["learning_rate"])
    RECOVERY_WEIGHT_DECAY = float(recovery["weight_decay"])
    RECOVERY_TEMPERATURE = float(recovery["temperature"])
    RECOVERY_CACHE_DTYPE = recovery["cache_dtype"]
    RECOVERY_EARLY_STOPPING_PATIENCE = recovery["early_stopping_patience"]
    CAPTURE_DEVICE = capture["storage_device"]
    HISTORICAL_CAPTURE_DTYPE = capture["historical_storage_dtype"]
    OPTIMIZED_CAPTURE_GROUP_SIZE = int(capture["optimized_group_size"])


def study_configs(revision: str, calibration_batches: int):
    model = ModelConfig(
        model_id="HuggingFaceTB/SmolLM2-1.7B",
        revision=revision,
        tokenizer_revision=revision,
        device="auto",
        dtype="auto",
    )
    data = DataConfig(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        num_calibration_batches=calibration_batches,
        num_operator_validation_batches=OPERATOR_VALIDATION_BATCHES,
        num_recovery_batches=RECOVERY_BATCHES,
        num_recovery_validation_batches=RECOVERY_VALIDATION_BATCHES,
        num_model_validation_batches=MODEL_VALIDATION_BATCHES,
        num_test_batches=TEST_BATCHES,
        seed=SEED,
    )
    capture = CaptureConfig(
        storage_device=CAPTURE_DEVICE,
        storage_dtype=HISTORICAL_CAPTURE_DTYPE,
    )
    training = OperatorConfig(
        kind="swiglu",
        epochs=OPERATOR_EPOCHS,
        learning_rate=OPERATOR_LEARNING_RATE,
        batch_size=OPERATOR_BATCH_SIZE,
        weight_decay=OPERATOR_WEIGHT_DECAY,
        scheduler=OPERATOR_SCHEDULER,
        early_stopping_patience=OPERATOR_EARLY_STOPPING_PATIENCE,
        seed=SEED,
    )
    recovery = RecoveryConfig(
        enabled=RECOVERY_ENABLED,
        epochs=RECOVERY_EPOCHS,
        learning_rate=RECOVERY_LEARNING_RATE,
        weight_decay=RECOVERY_WEIGHT_DECAY,
        temperature=RECOVERY_TEMPERATURE,
        cache_dtype=RECOVERY_CACHE_DTYPE,
        early_stopping_patience=RECOVERY_EARLY_STOPPING_PATIENCE,
    )
    return model, data, capture, training, recovery


def rank_normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    values = pd.Series(scores, dtype=float)
    if len(values) == 1:
        return {int(values.index[0]): 0.0}
    return ((values.rank(method="average") - 1) / (len(values) - 1)).to_dict()


def allocate_widths(
    policy: str,
    layers,
    blocks_by_layer,
    original_parameters,
    eligible_parameters: int,
    target_sparsity: float,
    scores: dict[int, float] | None,
    temperature: float,
) -> pd.DataFrame:
    """Apply the notebook's exact whole-neuron budget reconciliation."""

    target_removed = round(eligible_parameters * target_sparsity)
    target_retained = eligible_parameters - target_removed
    normalized = rank_normalize_scores(scores) if scores is not None else None
    propensity = {
        layer: math.exp(-normalized[layer] / temperature)
        if normalized is not None
        else 1.0
        for layer in layers
    }
    denominator = sum(
        original_parameters[layer] * propensity[layer] for layer in layers
    )
    rows = []
    for layer in layers:
        block = blocks_by_layer[layer].module
        hidden_size = block.up_proj.in_features
        original_width = block.up_proj.out_features
        parameter_step = 3 * hidden_size
        removal_share = original_parameters[layer] * propensity[layer] / denominator
        allocated_removal = target_removed * removal_share
        parameter_cap = original_parameters[layer] - allocated_removal
        continuous_width = parameter_cap / parameter_step
        rows.append(
            {
                "policy": policy,
                "layer": layer,
                "raw_importance": None if scores is None else scores[layer],
                "normalized_importance": (
                    None if normalized is None else normalized[layer]
                ),
                "allocated_sparsity": allocated_removal
                / original_parameters[layer],
                "allocated_retention": parameter_cap / original_parameters[layer],
                "original_width": original_width,
                "continuous_width": continuous_width,
                "replacement_width": int(continuous_width),
                "parameter_step": parameter_step,
                "original_parameters": original_parameters[layer],
            }
        )
    remaining = target_retained - sum(
        row["replacement_width"] * row["parameter_step"] for row in rows
    )
    for row in sorted(
        rows,
        key=lambda item: item["continuous_width"] - item["replacement_width"],
        reverse=True,
    ):
        if row["parameter_step"] <= remaining:
            row["replacement_width"] += 1
            remaining -= row["parameter_step"]
    for row in rows:
        row["replacement_width_ratio"] = (
            row["replacement_width"] / row["original_width"]
        )
        row["replacement_parameters"] = (
            row["replacement_width"] * row["parameter_step"]
        )
        row["realized_sparsity"] = 1 - (
            row["replacement_parameters"] / row["original_parameters"]
        )
    return pd.DataFrame(rows).drop(columns=["continuous_width", "parameter_step"])


def summarize_allocation(
    allocation: pd.DataFrame, fixed_parameters: int, dense_parameters: int
) -> pd.DataFrame:
    summary = (
        allocation.groupby("policy", sort=False, as_index=False)
        .agg(
            original_mlp_parameters=("original_parameters", "sum"),
            replacement_mlp_parameters=("replacement_parameters", "sum"),
            minimum_width_ratio=("replacement_width_ratio", "min"),
            maximum_width_ratio=("replacement_width_ratio", "max"),
        )
    )
    summary["mlp_sparsity"] = 1 - (
        summary["replacement_mlp_parameters"]
        / summary["original_mlp_parameters"]
    )
    summary["model_parameters"] = (
        fixed_parameters + summary["replacement_mlp_parameters"]
    )
    summary["model_sparsity"] = 1 - (
        summary["model_parameters"] / dense_parameters
    )
    return summary


def fit_grouped_allocations(
    model,
    loaders,
    data_config,
    capture_config,
    blocks_by_layer,
    layers,
    allocation_df,
    training_configs,
    device,
    model_dtype,
    group_size,
    seed_random_by_layer=True,
):
    """Capture bounded layer groups, fit each method/policy, then release inputs."""

    fitted = {
        method: {policy: {} for policy in POLICIES} for method in training_configs
    }
    fitting_rows = []
    history_rows = []
    rankings: dict[int, torch.Tensor] = {}
    importance_rows = []
    groups = [layers[start : start + group_size] for start in range(0, len(layers), group_size)]
    for group_index, group in enumerate(groups, start=1):
        print(f"Capture group {group_index}/{len(groups)} | {group}", flush=True)
        paths = [blocks_by_layer[layer].path for layer in group]
        training_pairs_by_path = collect_modules_io(
            model,
            paths,
            loaders.calibration,
            data_config.num_calibration_batches,
            device,
            storage_device=capture_config.storage_device,
            storage_dtype=model_dtype,
        )
        validation_pairs_by_path = collect_modules_io(
            model,
            paths,
            loaders.operator_validation,
            data_config.num_operator_validation_batches,
            device,
            storage_device=capture_config.storage_device,
            storage_dtype=model_dtype,
        )
        report_memory(f"After capture group {group_index}")
        teacher_config = next(
            (
                config
                for config in training_configs.values()
                if config.initialization == "importance_teacher_subset"
            ),
            None,
        )
        if teacher_config is not None:
            for layer in group:
                block = blocks_by_layer[layer]
                scores = swiglu_neuron_importance_scores(
                    block.module,
                    training_pairs_by_path[block.path].inputs,
                    teacher_config.batch_size,
                )
                rankings[layer] = torch.argsort(
                    scores, descending=True, stable=True
                )
                importance_rows.append(
                    {
                        "layer": layer,
                        "minimum_score": float(scores.min().item()),
                        "median_score": float(scores.median().item()),
                        "maximum_score": float(scores.max().item()),
                    }
                )
        group_allocations = allocation_df[allocation_df["layer"].isin(group)]
        for methodology, training_config in training_configs.items():
            for policy, policy_rows in group_allocations.groupby("policy", sort=False):
                print(f"Fitting {methodology} | {policy}", flush=True)
                for allocation in policy_rows.itertuples(index=False):
                    layer = int(allocation.layer)
                    block = blocks_by_layer[layer]
                    training_pairs = training_pairs_by_path[block.path]
                    validation_pairs = validation_pairs_by_path[block.path]
                    width = int(allocation.replacement_width)
                    if (
                        seed_random_by_layer
                        and training_config.initialization == "random_weights"
                    ):
                        torch.manual_seed(training_config.seed + layer)
                    module = GatedMLPReplacement(training_pairs.hidden_size, width).to(device)
                    if training_config.initialization == "importance_teacher_subset":
                        selected = rankings[layer][:width].sort().values
                        initialize_gated_mlp_from_teacher(
                            module, block.module, selected
                        )
                    initial = evaluate_operator(
                        module, validation_pairs, device, training_config.batch_size
                    )
                    fit = fit_operator(
                        module,
                        training_pairs,
                        validation_pairs,
                        training_config,
                        device,
                    )
                    metrics = evaluate_operator(
                        fit.module,
                        validation_pairs,
                        device,
                        training_config.batch_size,
                    )
                    fitted[methodology][policy][layer] = fit.module.to(
                        device="cpu", dtype=model_dtype
                    )
                    fitting_rows.append(
                        {
                            "methodology": methodology,
                            "initialization": training_config.initialization,
                            "policy": policy,
                            "layer": layer,
                            "replacement_width": width,
                            "replacement_width_ratio": allocation.replacement_width_ratio,
                            "replacement_parameters": allocation.replacement_parameters,
                            "initial_local_mse": initial.mse,
                            "initial_local_relative_mse": initial.relative_mse,
                            "initial_local_cosine": initial.cosine_similarity,
                            "best_epoch": fit.best_epoch,
                            "local_mse": metrics.mse,
                            "local_relative_mse": metrics.relative_mse,
                            "local_cosine": metrics.cosine_similarity,
                            "nmse_improvement": initial.relative_mse
                            - metrics.relative_mse,
                        }
                    )
                    history_rows.extend(
                        {
                            "methodology": methodology,
                            "policy": policy,
                            "layer": layer,
                            "epoch": epoch.epoch,
                            "train_mse": epoch.train_mse,
                            "validation_mse": epoch.validation_mse,
                            "learning_rate": epoch.learning_rate,
                        }
                        for epoch in fit.history
                    )
        training_pairs_by_path.clear()
        validation_pairs_by_path.clear()
        release_cuda(torch)
        report_memory(f"After fit group {group_index}")
    return (
        fitted,
        pd.DataFrame(fitting_rows),
        pd.DataFrame(history_rows),
        rankings,
        pd.DataFrame(importance_rows).drop_duplicates("layer"),
    )


def evaluate_fitted_models(
    model,
    fitted,
    methodologies,
    policies,
    layers,
    allocation_df,
    loaders,
    data_config,
    recovery_config,
    dense_footprint,
    dense_metrics,
    device,
    model_dtype,
):
    validation_cache = cache_teacher_logits(
        model,
        loaders.model_validation,
        data_config.num_model_validation_batches,
        device,
        recovery_config.cache_dtype,
    )
    recovery_cache = cache_teacher_logits(
        model,
        loaders.recovery,
        data_config.num_recovery_batches,
        device,
        recovery_config.cache_dtype,
    )
    recovery_validation_cache = cache_teacher_logits(
        model,
        loaders.recovery_validation,
        data_config.num_recovery_validation_batches,
        device,
        recovery_config.cache_dtype,
    )
    evaluation_rows = []
    recovery_rows = []
    for methodology in methodologies:
        for policy in policies:
            print(f"Evaluating {methodology} | {policy}", flush=True)
            replacements = {
                layer: deepcopy(fitted[methodology][policy][layer]) for layer in layers
            }
            with temporary_replacements(model, replacements) as manifest:
                footprint = parameter_footprint(model)
                pre_metrics = evaluate_language_model(
                    model,
                    loaders.model_validation,
                    device,
                    data_config.num_model_validation_batches,
                )
                pre_kl = mean_cache_loss(
                    model, validation_cache, recovery_config.temperature, device
                )
                allocation = allocation_df.query("policy == @policy")
                realized = 1 - (
                    allocation["replacement_parameters"].sum()
                    / allocation["original_parameters"].sum()
                )
                common = {
                    "methodology": methodology,
                    "policy": policy,
                    "parameters": footprint.parameters,
                    "realized_mlp_sparsity": realized,
                    "model_sparsity": 1
                    - footprint.parameters / dense_footprint.parameters,
                    "theoretical_weight_bytes": footprint.theoretical_weight_bytes,
                }
                evaluation_rows.append(
                    {
                        **common,
                        "phase": "pre_recovery",
                        "teacher_kl": pre_kl,
                        "loss": pre_metrics.loss,
                        "loss_delta": pre_metrics.loss - dense_metrics.loss,
                        "perplexity": pre_metrics.perplexity,
                        "perplexity_delta": pre_metrics.perplexity
                        - dense_metrics.perplexity,
                    }
                )
                recovery = recover_replacements(
                    model,
                    recovery_cache,
                    recovery_validation_cache,
                    [record.path for record in manifest.records],
                    recovery_config,
                    device,
                )
                post_metrics = evaluate_language_model(
                    model,
                    loaders.model_validation,
                    device,
                    data_config.num_model_validation_batches,
                )
                post_kl = mean_cache_loss(
                    model, validation_cache, recovery_config.temperature, device
                )
                evaluation_rows.append(
                    {
                        **common,
                        "phase": "post_recovery",
                        "teacher_kl": post_kl,
                        "loss": post_metrics.loss,
                        "loss_delta": post_metrics.loss - dense_metrics.loss,
                        "perplexity": post_metrics.perplexity,
                        "perplexity_delta": post_metrics.perplexity
                        - dense_metrics.perplexity,
                    }
                )
                recovery_rows.extend(
                    {
                        "methodology": methodology,
                        "policy": policy,
                        "epoch": epoch.epoch,
                        "train_kl": epoch.train_kl,
                        "validation_kl": epoch.validation_kl,
                        "best_epoch": recovery.best_epoch,
                    }
                    for epoch in recovery.history
                )
            for replacement in replacements.values():
                replacement.to(device="cpu", dtype=model_dtype)
            del replacements
            release_cuda(torch)
    del validation_cache, recovery_cache, recovery_validation_cache
    return pd.DataFrame(evaluation_rows), pd.DataFrame(recovery_rows)


def historical_sweep(
    model,
    loaders,
    data_config,
    capture_config,
    training_config,
    recovery_config,
    blocks_by_layer,
    layers,
    score_maps,
    original_parameters,
    eligible_parameters,
    fixed_parameters,
    dense_footprint,
    dense_metrics,
    validation_cache,
    recovery_cache,
    recovery_validation_cache,
    device,
):
    paths = [blocks_by_layer[layer].path for layer in layers]
    training_by_path = collect_modules_io(
        model,
        paths,
        loaders.calibration,
        data_config.num_calibration_batches,
        device,
        storage_device=capture_config.storage_device,
        storage_dtype=torch.float32,
    )
    validation_by_path = collect_modules_io(
        model,
        paths,
        loaders.operator_validation,
        data_config.num_operator_validation_batches,
        device,
        storage_device=capture_config.storage_device,
        storage_dtype=torch.float32,
    )
    allocation_frames = []
    fitting_rows = []
    history_rows = []
    evaluation_rows = []
    recovery_rows = []
    for target_sparsity in SPARSITY_RATIOS:
        allocation = pd.concat(
            [
                allocate_widths(
                    policy,
                    layers,
                    blocks_by_layer,
                    original_parameters,
                    eligible_parameters,
                    target_sparsity,
                    score_maps.get(policy),
                    ALLOCATION_TEMPERATURE,
                )
                for policy in POLICIES
            ],
            ignore_index=True,
        )
        allocation.insert(0, "target_mlp_sparsity", target_sparsity)
        allocation_frames.append(allocation)
        for policy in POLICIES:
            print(f"Historical sweep {target_sparsity:.0%} | {policy}", flush=True)
            policy_allocation = allocation.query("policy == @policy")
            replacements = {}
            for item in policy_allocation.itertuples(index=False):
                layer = int(item.layer)
                block = blocks_by_layer[layer]
                training_pairs = training_by_path[block.path]
                validation_pairs = validation_by_path[block.path]
                module = GatedMLPReplacement(
                    training_pairs.hidden_size, int(item.replacement_width)
                )
                fit = fit_operator(
                    module,
                    training_pairs,
                    validation_pairs,
                    training_config,
                    device,
                )
                metrics = evaluate_operator(
                    fit.module,
                    validation_pairs,
                    device,
                    training_config.batch_size,
                )
                replacements[layer] = fit.module.to("cpu")
                fitting_rows.append(
                    {
                        "target_mlp_sparsity": target_sparsity,
                        "policy": policy,
                        "layer": layer,
                        "replacement_width": int(item.replacement_width),
                        "replacement_width_ratio": item.replacement_width_ratio,
                        "replacement_parameters": item.replacement_parameters,
                        "best_epoch": fit.best_epoch,
                        "local_mse": metrics.mse,
                        "local_relative_mse": metrics.relative_mse,
                        "local_cosine": metrics.cosine_similarity,
                    }
                )
                history_rows.extend(
                    {
                        "target_mlp_sparsity": target_sparsity,
                        "policy": policy,
                        "layer": layer,
                        "epoch": epoch.epoch,
                        "train_mse": epoch.train_mse,
                        "validation_mse": epoch.validation_mse,
                        "learning_rate": epoch.learning_rate,
                    }
                    for epoch in fit.history
                )
            realized = 1 - (
                policy_allocation["replacement_parameters"].sum()
                / policy_allocation["original_parameters"].sum()
            )
            with temporary_replacements(model, replacements) as manifest:
                footprint = parameter_footprint(model)
                pre = evaluate_language_model(
                    model,
                    loaders.model_validation,
                    device,
                    data_config.num_model_validation_batches,
                )
                pre_kl = mean_cache_loss(
                    model, validation_cache, recovery_config.temperature, device
                )
                common = {
                    "target_mlp_sparsity": target_sparsity,
                    "realized_mlp_sparsity": realized,
                    "policy": policy,
                    "parameters": footprint.parameters,
                    "model_sparsity": 1
                    - footprint.parameters / dense_footprint.parameters,
                    "theoretical_weight_bytes": footprint.theoretical_weight_bytes,
                }
                evaluation_rows.append(
                    {
                        **common,
                        "phase": "pre_recovery",
                        "teacher_kl": pre_kl,
                        "loss": pre.loss,
                        "loss_delta": pre.loss - dense_metrics.loss,
                        "perplexity": pre.perplexity,
                        "perplexity_delta": pre.perplexity
                        - dense_metrics.perplexity,
                    }
                )
                recovery = recover_replacements(
                    model,
                    recovery_cache,
                    recovery_validation_cache,
                    [record.path for record in manifest.records],
                    recovery_config,
                    device,
                )
                post = evaluate_language_model(
                    model,
                    loaders.model_validation,
                    device,
                    data_config.num_model_validation_batches,
                )
                post_kl = mean_cache_loss(
                    model, validation_cache, recovery_config.temperature, device
                )
                evaluation_rows.append(
                    {
                        **common,
                        "phase": "post_recovery",
                        "teacher_kl": post_kl,
                        "loss": post.loss,
                        "loss_delta": post.loss - dense_metrics.loss,
                        "perplexity": post.perplexity,
                        "perplexity_delta": post.perplexity
                        - dense_metrics.perplexity,
                    }
                )
                recovery_rows.extend(
                    {
                        "target_mlp_sparsity": target_sparsity,
                        "policy": policy,
                        "epoch": epoch.epoch,
                        "train_kl": epoch.train_kl,
                        "validation_kl": epoch.validation_kl,
                        "best_epoch": recovery.best_epoch,
                    }
                    for epoch in recovery.history
                )
            del replacements
            release_cuda(torch)
    training_by_path.clear()
    validation_by_path.clear()
    allocation_df = pd.concat(allocation_frames, ignore_index=True)
    allocation_summary = (
        allocation_df.groupby(
            ["target_mlp_sparsity", "policy"], sort=False, as_index=False
        )
        .agg(
            original_mlp_parameters=("original_parameters", "sum"),
            replacement_mlp_parameters=("replacement_parameters", "sum"),
            minimum_width_ratio=("replacement_width_ratio", "min"),
            maximum_width_ratio=("replacement_width_ratio", "max"),
        )
    )
    allocation_summary["realized_mlp_sparsity"] = 1 - (
        allocation_summary["replacement_mlp_parameters"]
        / allocation_summary["original_mlp_parameters"]
    )
    allocation_summary["model_parameters"] = (
        fixed_parameters + allocation_summary["replacement_mlp_parameters"]
    )
    allocation_summary["model_sparsity"] = 1 - (
        allocation_summary["model_parameters"] / dense_footprint.parameters
    )
    fitting_df = pd.DataFrame(fitting_rows)
    evaluation_df = pd.DataFrame(evaluation_rows)
    local = fitting_df.groupby(
        ["target_mlp_sparsity", "policy"], sort=False, as_index=False
    ).agg(
        mean_local_relative_mse=("local_relative_mse", "mean"),
        mean_local_cosine=("local_cosine", "mean"),
    )
    pre = evaluation_df.query("phase == 'pre_recovery'").rename(
        columns={
            "teacher_kl": "pre_recovery_kl",
            "loss": "pre_recovery_loss",
            "loss_delta": "pre_recovery_loss_delta",
            "perplexity": "pre_recovery_perplexity",
            "perplexity_delta": "pre_recovery_perplexity_delta",
        }
    )
    post = evaluation_df.query("phase == 'post_recovery'").rename(
        columns={
            "teacher_kl": "post_recovery_kl",
            "loss": "post_recovery_loss",
            "loss_delta": "post_recovery_loss_delta",
            "perplexity": "post_recovery_perplexity",
            "perplexity_delta": "post_recovery_perplexity_delta",
        }
    )
    keys = ["target_mlp_sparsity", "policy"]
    comparison = (
        allocation_summary.merge(local, on=keys)
        .merge(
            pre[
                keys
                + [
                    "pre_recovery_kl",
                    "pre_recovery_loss",
                    "pre_recovery_loss_delta",
                    "pre_recovery_perplexity",
                    "pre_recovery_perplexity_delta",
                ]
            ],
            on=keys,
        )
        .merge(
            post[
                keys
                + [
                    "post_recovery_kl",
                    "post_recovery_loss",
                    "post_recovery_loss_delta",
                    "post_recovery_perplexity",
                    "post_recovery_perplexity_delta",
                ]
            ],
            on=keys,
        )
    )
    comparison["kl_reduction"] = (
        comparison["pre_recovery_kl"] - comparison["post_recovery_kl"]
    )
    comparison["perplexity_reduction"] = (
        comparison["pre_recovery_perplexity"]
        - comparison["post_recovery_perplexity"]
    )
    return {
        "allocation": allocation_df,
        "allocation_summary": allocation_summary,
        "fitting": fitting_df,
        "history": pd.DataFrame(history_rows),
        "evaluation": evaluation_df,
        "recovery": pd.DataFrame(recovery_rows),
        "comparison": comparison,
    }


def run_historical(output: Path, settings: dict) -> dict:
    output = require_new_path(output)
    model_config, data_config, capture_config, training_config, recovery_config = (
        study_configs(PINNED_REVISION, HISTORICAL_CALIBRATION_BATCHES)
    )
    run_log = start_run_log(
        output,
        WORKFLOW,
        {"stage": "historical", "workflow_config": settings},
    )
    try:
        torch.manual_seed(SEED)
        run_log.begin("load_model_and_data")
        model, tokenizer = load_model_and_tokenizer(model_config)
        device = next(model.parameters()).device
        loaders = build_data_loaders(tokenizer, data_config, include_recovery=True)
        blocks = discover_mlp_blocks(model)
        blocks_by_layer = {block.index: block for block in blocks}
        available = [block.index for block in blocks]
        stop = len(available) - PROTECTED_SUFFIX
        layers = tuple(available[PROTECTED_PREFIX:stop])
        protected = tuple(available[:PROTECTED_PREFIX] + available[stop:])
        dense_footprint = parameter_footprint(model)
        original_parameters = {
            layer: sum(
                parameter.numel()
                for parameter in blocks_by_layer[layer].module.parameters()
            )
            for layer in available
        }
        eligible_parameters = sum(original_parameters[layer] for layer in layers)
        fixed_parameters = dense_footprint.parameters - eligible_parameters
        dense_metrics = evaluate_language_model(
            model,
            loaders.model_validation,
            device,
            data_config.num_model_validation_batches,
        )
        dense_reference = pd.DataFrame(
            [
                {
                    "variant": "dense_reference",
                    "layers": len(available),
                    "eligible_layers": len(layers),
                    "parameters": dense_footprint.parameters,
                    "eligible_mlp_parameters": eligible_parameters,
                    "fixed_model_parameters": fixed_parameters,
                    "theoretical_weight_bytes": dense_footprint.theoretical_weight_bytes,
                    "teacher_kl": 0.0,
                    "loss": dense_metrics.loss,
                    "perplexity": dense_metrics.perplexity,
                    "predicted_tokens": dense_metrics.predicted_tokens,
                    "batches": dense_metrics.batches,
                }
            ]
        )
        run_log.record("load_model_and_data", {"eligible_layers": list(layers)})

        run_log.begin("importance")
        canonical = compute_bi_scores(
            model,
            loaders.calibration,
            data_config.num_calibration_batches,
            device,
            scope="transformer_layer",
            layer_indices=layers,
        )
        residual = compute_bi_scores(
            model,
            loaders.calibration,
            data_config.num_calibration_batches,
            device,
            scope="mlp_sublayer",
            layer_indices=layers,
        )
        importance = pd.DataFrame(
            {
                "layer": layers,
                "depth_fraction": [layer / (len(available) - 1) for layer in layers],
                "canonical_bi": [canonical.scores[layer] for layer in layers],
                "residual_aware_mlp_bi": [residual.scores[layer] for layer in layers],
            }
        )
        score_maps = {
            "canonical_bi": canonical.scores,
            "residual_aware_mlp_bi": residual.scores,
        }
        run_log.record("importance", json_records(importance))

        target_removed = round(eligible_parameters * TARGET_MLP_SPARSITY)
        target_retained = eligible_parameters - target_removed
        target_model = fixed_parameters + target_retained
        global_budget = pd.DataFrame(
            [
                {
                    "target_scope": "eligible_mlp",
                    "target_reduction": TARGET_MLP_SPARSITY,
                    "target_retention": 1 - TARGET_MLP_SPARSITY,
                    "original_eligible_mlp_parameters": eligible_parameters,
                    "target_removed_parameters": target_removed,
                    "target_retained_mlp_parameters": target_retained,
                    "fixed_model_parameters": fixed_parameters,
                    "target_model_parameters": target_model,
                    "target_model_reduction": 1
                    - target_model / dense_footprint.parameters,
                }
            ]
        )
        allocation = pd.concat(
            [
                allocate_widths(
                    policy,
                    layers,
                    blocks_by_layer,
                    original_parameters,
                    eligible_parameters,
                    TARGET_MLP_SPARSITY,
                    score_maps.get(policy),
                    ALLOCATION_TEMPERATURE,
                )
                for policy in POLICIES
            ],
            ignore_index=True,
        )
        allocation_summary = summarize_allocation(
            allocation, fixed_parameters, dense_footprint.parameters
        )
        allocation_summary["retained_budget_difference"] = (
            allocation_summary["replacement_mlp_parameters"] - target_retained
        )

        run_log.begin("main_operator_fitting")
        fitted, fitting, history, _, _ = fit_grouped_allocations(
            model,
            loaders,
            data_config,
            capture_config,
            blocks_by_layer,
            layers,
            allocation,
            {"historical_random": training_config},
            device,
            torch.float32,
            len(layers),
            seed_random_by_layer=False,
        )
        fitting = fitting.drop(
            columns=[
                "methodology",
                "initialization",
                "initial_local_mse",
                "initial_local_relative_mse",
                "initial_local_cosine",
                "nmse_improvement",
            ]
        )
        history = history.drop(columns=["methodology"])
        run_log.record("main_operator_fitting", {"operators": len(fitting)})

        validation_cache = cache_teacher_logits(
            model,
            loaders.model_validation,
            data_config.num_model_validation_batches,
            device,
            recovery_config.cache_dtype,
        )
        recovery_cache = cache_teacher_logits(
            model,
            loaders.recovery,
            data_config.num_recovery_batches,
            device,
            recovery_config.cache_dtype,
        )
        recovery_validation_cache = cache_teacher_logits(
            model,
            loaders.recovery_validation,
            data_config.num_recovery_validation_batches,
            device,
            recovery_config.cache_dtype,
        )
        evaluation_rows = []
        recovery_rows = []
        for policy in POLICIES:
            replacements = {
                layer: deepcopy(fitted["historical_random"][policy][layer])
                for layer in layers
            }
            with temporary_replacements(model, replacements) as manifest:
                footprint = parameter_footprint(model)
                pre = evaluate_language_model(
                    model,
                    loaders.model_validation,
                    device,
                    data_config.num_model_validation_batches,
                )
                pre_kl = mean_cache_loss(
                    model, validation_cache, recovery_config.temperature, device
                )
                common = {
                    "policy": policy,
                    "parameters": footprint.parameters,
                    "model_sparsity": 1
                    - footprint.parameters / dense_footprint.parameters,
                    "theoretical_weight_bytes": footprint.theoretical_weight_bytes,
                }
                evaluation_rows.append(
                    {
                        **common,
                        "phase": "pre_recovery",
                        "teacher_kl": pre_kl,
                        "loss": pre.loss,
                        "loss_delta": pre.loss - dense_metrics.loss,
                        "perplexity": pre.perplexity,
                        "perplexity_delta": pre.perplexity
                        - dense_metrics.perplexity,
                    }
                )
                recovery = recover_replacements(
                    model,
                    recovery_cache,
                    recovery_validation_cache,
                    [record.path for record in manifest.records],
                    recovery_config,
                    device,
                )
                post = evaluate_language_model(
                    model,
                    loaders.model_validation,
                    device,
                    data_config.num_model_validation_batches,
                )
                post_kl = mean_cache_loss(
                    model, validation_cache, recovery_config.temperature, device
                )
                evaluation_rows.append(
                    {
                        **common,
                        "phase": "post_recovery",
                        "teacher_kl": post_kl,
                        "loss": post.loss,
                        "loss_delta": post.loss - dense_metrics.loss,
                        "perplexity": post.perplexity,
                        "perplexity_delta": post.perplexity
                        - dense_metrics.perplexity,
                    }
                )
                recovery_rows.extend(
                    {
                        "policy": policy,
                        "epoch": epoch.epoch,
                        "train_kl": epoch.train_kl,
                        "validation_kl": epoch.validation_kl,
                        "best_epoch": recovery.best_epoch,
                    }
                    for epoch in recovery.history
                )
            del replacements
            release_cuda(torch)
        evaluation = pd.DataFrame(evaluation_rows)
        recovery_history = pd.DataFrame(recovery_rows)
        local = fitting.groupby("policy", sort=False, as_index=False).agg(
            mean_local_relative_mse=("local_relative_mse", "mean"),
            mean_local_cosine=("local_cosine", "mean"),
        )
        pre = evaluation.query("phase == 'pre_recovery'")
        post = evaluation.query("phase == 'post_recovery'")
        comparison = allocation_summary.merge(local, on="policy")
        for prefix, frame in (("pre_recovery", pre), ("post_recovery", post)):
            comparison = comparison.merge(
                frame[
                    [
                        "policy",
                        "teacher_kl",
                        "loss",
                        "loss_delta",
                        "perplexity",
                        "perplexity_delta",
                    ]
                ].rename(
                    columns={
                        column: f"{prefix}_{'kl' if column == 'teacher_kl' else column}"
                        for column in (
                            "teacher_kl",
                            "loss",
                            "loss_delta",
                            "perplexity",
                            "perplexity_delta",
                        )
                    }
                ),
                on="policy",
            )
        comparison["kl_reduction"] = (
            comparison["pre_recovery_kl"]
            - comparison["post_recovery_kl"]
        )
        comparison["perplexity_reduction"] = (
            comparison["pre_recovery_perplexity"]
            - comparison["post_recovery_perplexity"]
        )
        fitted.clear()
        release_cuda(torch)

        run_log.begin("historical_sparsity_sweep")
        sweep = historical_sweep(
            model,
            loaders,
            data_config,
            capture_config,
            training_config,
            recovery_config,
            blocks_by_layer,
            layers,
            score_maps,
            original_parameters,
            eligible_parameters,
            fixed_parameters,
            dense_footprint,
            dense_metrics,
            validation_cache,
            recovery_cache,
            recovery_validation_cache,
            device,
        )
        run_log.record(
            "historical_sparsity_sweep", {"experiments": len(SPARSITY_RATIOS) * len(POLICIES)}
        )
        artifact = {
            "schema_version": HISTORICAL_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": {
                "model": {
                    **asdict(model_config),
                    "resolved_revision": getattr(model.config, "_commit_hash", None),
                },
                "data": asdict(data_config),
                "capture": asdict(capture_config),
                "operator_training": asdict(training_config),
                "recovery": {
                    **asdict(recovery_config),
                    "trainable_scope": "replacement_only",
                },
                "target_mlp_sparsity": TARGET_MLP_SPARSITY,
                "sparsity_sweep_ratios": list(SPARSITY_RATIOS),
                "sparsity_sweep_policies": list(POLICIES),
                "allocation_temperature": ALLOCATION_TEMPERATURE,
                "eligible_layers": list(layers),
                "protected_layers": list(protected),
            },
            "results": {
                "dense_reference": json_records(dense_reference),
                "importance": json_records(importance),
                "global_budget": json_records(global_budget),
                "allocation": json_records(allocation),
                "allocation_summary": json_records(allocation_summary),
                "operator_fitting": json_records(fitting),
                "operator_training_history": json_records(history),
                "model_evaluation": json_records(evaluation),
                "recovery_history": json_records(recovery_history),
                "comparison": json_records(comparison),
                "sparsity_sweep_allocation": json_records(sweep["allocation"]),
                "sparsity_sweep_allocation_summary": json_records(
                    sweep["allocation_summary"]
                ),
                "sparsity_sweep_operator_fitting": json_records(sweep["fitting"]),
                "sparsity_sweep_operator_training_history": json_records(
                    sweep["history"]
                ),
                "sparsity_sweep_model_evaluation": json_records(
                    sweep["evaluation"]
                ),
                "sparsity_sweep_recovery_history": json_records(sweep["recovery"]),
                "sparsity_sweep_comparison": json_records(sweep["comparison"]),
            },
        }
        write_artifact(output, artifact)
        run_log.complete({"artifact_path": str(output), "schema_version": 2})
        print(f"Saved historical SwiGLU artifact to {output}", flush=True)
        return artifact
    except BaseException as error:
        run_log.fail(error)
        raise


def configs_from_historical(historical: dict):
    model_values = historical["configuration"]["model"]
    revision = model_values.get("resolved_revision") or model_values.get("revision")
    if not revision:
        raise ValueError("Historical artifact does not record a model revision")
    model_config = ModelConfig(
        model_id=model_values["model_id"],
        revision=revision,
        tokenizer_revision=model_values.get("tokenizer_revision") or revision,
        device=model_values["device"],
        dtype=model_values["dtype"],
        trust_remote_code=model_values["trust_remote_code"],
    )
    data_values = dict(historical["configuration"]["data"])
    for name in ("calibration_source", "model_validation_source", "test_source"):
        data_values[name] = DatasetSpec(**data_values[name])
    base_batches = int(data_values["num_calibration_batches"])
    data_values["num_calibration_batches"] = OPTIMIZED_CALIBRATION_BATCHES
    data_config = DataConfig(**data_values)
    capture_config = CaptureConfig(**historical["configuration"]["capture"])
    recovery_values = dict(historical["configuration"]["recovery"])
    recovery_values.pop("trainable_scope", None)
    recovery_config = RecoveryConfig(**recovery_values)
    training_configs = {
        method: OperatorConfig(
            kind="swiglu",
            initialization=initialization,
            epochs=OPERATOR_EPOCHS,
            learning_rate=OPERATOR_LEARNING_RATE,
            batch_size=OPERATOR_BATCH_SIZE,
            weight_decay=OPERATOR_WEIGHT_DECAY,
            scheduler=OPERATOR_SCHEDULER,
            early_stopping_patience=OPERATOR_EARLY_STOPPING_PATIENCE,
            seed=int(data_config.seed),
        )
        for method, initialization in OPTIMIZED_METHODS.items()
    }
    return (
        model_config,
        data_config,
        capture_config,
        recovery_config,
        training_configs,
        base_batches,
    )


def build_optimized_loaders(tokenizer, data_config, base_batches):
    partitions = {
        "calibration": base_batches,
        "operator_validation": data_config.num_operator_validation_batches,
        "recovery": data_config.num_recovery_batches,
        "recovery_validation": data_config.num_recovery_validation_batches,
        "additional_calibration": data_config.num_calibration_batches - base_batches,
    }
    data = load_text_dataset(data_config.calibration_source)
    windows = sample_partitioned_windows(
        data,
        tokenizer,
        {name: count * data_config.batch_size for name, count in partitions.items()},
        data_config.sequence_length,
        data_config.seed,
        data_config.calibration_source.text_column,
    )
    calibration = windows["calibration"] + windows["additional_calibration"]
    validation_data = load_text_dataset(data_config.model_validation_source)
    validation = contiguous_token_windows(
        validation_data,
        tokenizer,
        data_config.num_model_validation_batches * data_config.batch_size,
        data_config.sequence_length,
        data_config.model_validation_source.text_column,
    )
    loaders = DataLoaders(
        calibration=make_token_loader(calibration, data_config.batch_size),
        operator_validation=make_token_loader(
            windows["operator_validation"], data_config.batch_size
        ),
        recovery=make_token_loader(windows["recovery"], data_config.batch_size),
        recovery_validation=make_token_loader(
            windows["recovery_validation"], data_config.batch_size
        ),
        model_validation=make_token_loader(validation, data_config.batch_size),
        test=None,
    )
    del data, validation_data
    return loaders, partitions


def run_optimized(
    output: Path,
    historical_path: Path,
    settings: dict,
    historical: dict | None = None,
) -> dict:
    output = require_new_path(output)
    historical_path = resolve_path(historical_path)
    historical = historical or load_artifact(
        historical_path, HISTORICAL_SCHEMA, "historical SwiGLU artifact"
    )
    (
        model_config,
        data_config,
        capture_config,
        recovery_config,
        training_configs,
        base_batches,
    ) = configs_from_historical(historical)
    layers = tuple(historical["configuration"]["eligible_layers"])
    protected = tuple(historical["configuration"]["protected_layers"])
    target_sparsity = float(historical["configuration"]["target_mlp_sparsity"])
    temperature = float(historical["configuration"]["allocation_temperature"])
    if not math.isclose(target_sparsity, TARGET_MLP_SPARSITY):
        raise ValueError("Historical artifact target sparsity differs from the config")
    if not math.isclose(temperature, ALLOCATION_TEMPERATURE):
        raise ValueError("Historical artifact allocation temperature differs from the config")
    run_log = start_run_log(
        output,
        WORKFLOW,
        {
            "stage": "optimized",
            "historical_artifact": str(historical_path),
            "workflow_config": settings,
        },
    )
    try:
        torch.manual_seed(int(data_config.seed))
        run_log.begin("load_model_and_data")
        model, tokenizer = load_model_and_tokenizer(model_config)
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        blocks = discover_mlp_blocks(model)
        blocks_by_layer = {block.index: block for block in blocks}
        if not set(layers).issubset(blocks_by_layer):
            raise ValueError("Historical eligible layers do not match the model")
        loaders, partitions = build_optimized_loaders(
            tokenizer, data_config, base_batches
        )
        dense_footprint = parameter_footprint(model)
        dense_metrics = evaluate_language_model(
            model,
            loaders.model_validation,
            device,
            data_config.num_model_validation_batches,
        )
        dense_reference = pd.DataFrame(
            [
                {
                    "variant": "dense_reference",
                    "parameters": dense_footprint.parameters,
                    "teacher_kl": 0.0,
                    "loss": dense_metrics.loss,
                    "perplexity": dense_metrics.perplexity,
                    "predicted_tokens": dense_metrics.predicted_tokens,
                    "batches": dense_metrics.batches,
                }
            ]
        )
        original_parameters = {
            layer: sum(
                parameter.numel()
                for parameter in blocks_by_layer[layer].module.parameters()
            )
            for layer in layers
        }
        eligible_parameters = sum(original_parameters.values())
        fixed_parameters = dense_footprint.parameters - eligible_parameters
        run_log.record(
            "load_model_and_data",
            {"partition_batches": partitions, "eligible_layers": list(layers)},
        )

        run_log.begin("importance_and_allocation")
        canonical = compute_bi_scores(
            model,
            loaders.calibration,
            data_config.num_calibration_batches,
            device,
            scope="transformer_layer",
            layer_indices=layers,
        )
        residual = compute_bi_scores(
            model,
            loaders.calibration,
            data_config.num_calibration_batches,
            device,
            scope="mlp_sublayer",
            layer_indices=layers,
        )
        importance = pd.DataFrame(
            {
                "layer": layers,
                "canonical_bi": [canonical.scores[layer] for layer in layers],
                "residual_aware_mlp_bi": [residual.scores[layer] for layer in layers],
            }
        )
        score_maps = {
            "canonical_bi": canonical.scores,
            "residual_aware_mlp_bi": residual.scores,
        }
        allocation = pd.concat(
            [
                allocate_widths(
                    policy,
                    layers,
                    blocks_by_layer,
                    original_parameters,
                    eligible_parameters,
                    target_sparsity,
                    score_maps.get(policy),
                    temperature,
                )
                for policy in POLICIES
            ],
            ignore_index=True,
        )
        allocation_summary = summarize_allocation(
            allocation, fixed_parameters, dense_footprint.parameters
        )
        run_log.record(
            "importance_and_allocation", {"policies": list(POLICIES)}
        )

        run_log.begin("grouped_operator_fitting")
        fitted, fitting, history, rankings, teacher_importance = (
            fit_grouped_allocations(
                model,
                loaders,
                data_config,
                capture_config,
                blocks_by_layer,
                layers,
                allocation,
                training_configs,
                device,
                model_dtype,
                OPTIMIZED_CAPTURE_GROUP_SIZE,
            )
        )
        run_log.record("grouped_operator_fitting", {"operators": len(fitting)})
        run_log.begin("model_evaluation_and_recovery")
        evaluation, recovery_history = evaluate_fitted_models(
            model,
            fitted,
            tuple(training_configs),
            POLICIES,
            layers,
            allocation,
            loaders,
            data_config,
            recovery_config,
            dense_footprint,
            dense_metrics,
            device,
            model_dtype,
        )
        run_log.record(
            "model_evaluation_and_recovery", {"evaluations": len(evaluation)}
        )

        historical_importance = pd.DataFrame(
            historical["results"]["importance"]
        ).assign(data_budget="Historical")
        importance_comparison = pd.concat(
            [historical_importance, importance.assign(data_budget="Optimized data")],
            ignore_index=True,
        )
        historical_allocation = pd.DataFrame(
            historical["results"]["allocation"]
        ).assign(method="Historical")
        allocation_comparison = pd.concat(
            [historical_allocation, allocation.assign(method="Optimized data")],
            ignore_index=True,
        )
        labels = {
            "historical_random": "Historical",
            "new_data_random": "New data only",
            "complete_method": "Complete method",
        }
        historical_fitting = pd.DataFrame(
            historical["results"]["operator_fitting"]
        ).assign(
            methodology="historical_random",
            initialization="random_weights",
            initial_local_mse=float("nan"),
            initial_local_relative_mse=float("nan"),
            initial_local_cosine=float("nan"),
            nmse_improvement=float("nan"),
        )
        fitting_comparison = pd.concat(
            [historical_fitting, fitting], ignore_index=True
        )
        fitting_comparison["method"] = fitting_comparison["methodology"].map(labels)
        fitting_summary = fitting_comparison.groupby(
            ["methodology", "method", "policy"], sort=False, as_index=False
        ).agg(
            fitted_blocks=("layer", "count"),
            mean_width_ratio=("replacement_width_ratio", "mean"),
            mean_initial_nmse=("initial_local_relative_mse", "mean"),
            mean_fitted_nmse=("local_relative_mse", "mean"),
            worst_fitted_nmse=("local_relative_mse", "max"),
            mean_fitted_cosine=("local_cosine", "mean"),
            mean_best_epoch=("best_epoch", "mean"),
        )
        historical_evaluation = pd.DataFrame(
            historical["results"]["model_evaluation"]
        ).assign(methodology="historical_random")
        evaluation_comparison = pd.concat(
            [historical_evaluation, evaluation], ignore_index=True
        )
        evaluation_comparison["method"] = evaluation_comparison["methodology"].map(
            labels
        )
        optimized_pairs = (
            data_config.num_calibration_batches
            * data_config.batch_size
            * data_config.sequence_length
        )
        artifact = {
            "schema_version": OPTIMIZED_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": environment_record(),
            "configuration": {
                "model": {
                    **asdict(model_config),
                    "resolved_revision": getattr(model.config, "_commit_hash", None),
                },
                "data": asdict(data_config),
                "capture": {
                    **asdict(capture_config),
                    "storage_dtype": str(model_dtype).removeprefix("torch."),
                    "module_group_size": OPTIMIZED_CAPTURE_GROUP_SIZE,
                },
                "operator_training": {
                    method: asdict(config)
                    for method, config in training_configs.items()
                },
                "recovery": {
                    **asdict(recovery_config),
                    "trainable_scope": "replacement_only",
                },
                "methodologies": {
                    "historical_random": {
                        "calibration_pairs": base_batches
                        * data_config.batch_size
                        * data_config.sequence_length,
                        "initialization": "random_weights",
                    },
                    "new_data_random": {
                        "calibration_pairs": optimized_pairs,
                        "initialization": "random_weights",
                    },
                    "complete_method": {
                        "calibration_pairs": optimized_pairs,
                        "initialization": "importance_teacher_subset",
                        "importance_metric": (
                            "rms_intermediate_activation_times_"
                            "down_projection_column_l2"
                        ),
                    },
                },
                "historical_control": {
                    "artifact_path": str(historical_path),
                    "schema_version": historical["schema_version"],
                },
                "partition_order": list(partitions),
                "target_mlp_sparsity": target_sparsity,
                "allocation_temperature": temperature,
                "allocation_policies": list(POLICIES),
                "eligible_layers": list(layers),
                "protected_layers": list(protected),
            },
            "results": {
                "dense_reference": json_records(dense_reference),
                "importance": json_records(importance),
                "importance_comparison": json_records(importance_comparison),
                "allocation": json_records(allocation),
                "allocation_summary": json_records(allocation_summary),
                "allocation_comparison": json_records(allocation_comparison),
                "teacher_neuron_importance": json_records(teacher_importance),
                "teacher_neuron_rankings": {
                    str(layer): ranking.tolist() for layer, ranking in rankings.items()
                },
                "operator_fitting": json_records(fitting),
                "operator_training_history": json_records(history),
                "block_fitting_comparison": json_records(fitting_comparison),
                "block_fitting_summary": json_records(fitting_summary),
                "model_evaluation": json_records(evaluation),
                "recovery_history": json_records(recovery_history),
                "model_evaluation_comparison": json_records(evaluation_comparison),
            },
        }
        write_artifact(output, artifact)
        run_log.complete({"artifact_path": str(output), "schema_version": 3})
        print(f"Saved optimized SwiGLU artifact to {output}", flush=True)
        return artifact
    except BaseException as error:
        run_log.fail(error)
        raise


def main() -> None:
    args = parse_args()
    settings = load_workflow_config(args.config, WORKFLOW)
    configure(settings)
    historical = None
    historical_path = args.historical_artifact
    if args.stage in {"historical", "all"}:
        historical_path = (
            args.historical_output
            or (args.output if args.stage == "historical" else None)
            or default_artifact_path(WORKFLOW, "historical")
        )
        historical = run_historical(historical_path, settings)
    if args.stage in {"optimized", "all"}:
        output = args.output or default_artifact_path(WORKFLOW, "optimized")
        run_optimized(output, historical_path, settings, historical)


if __name__ == "__main__":
    main()
