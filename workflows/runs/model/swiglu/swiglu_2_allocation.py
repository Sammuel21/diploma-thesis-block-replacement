"""Headless migration of ``notebooks/model/swiglu/swiglu-2.ipynb``."""

from __future__ import annotations

import argparse
import math
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

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
    synchronize_cuda,
    write_artifact,
)

from mlp_replacement.capture import collect_modules_io
from mlp_replacement.compression.recovery import (
    cache_teacher_logits,
    mean_cache_loss,
    recover_replacements,
)
from mlp_replacement.compression.surgery import (
    temporary_replacement,
    temporary_replacements,
)
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
)
from mlp_replacement.runlog import environment_record


WORKFLOW = "swiglu-2"
REFERENCE_SCHEMA = 3
ARTIFACT_SCHEMA = 1
DEFAULT_CONFIG = Path("workflows/configs/model/swiglu/swiglu-2-allocation.json")

PROBE_WIDTH_RATIOS = (0.25, 0.5)
REFERENCE_WIDTH_RATIO = 0.5
WIDTH_SWEEP_RATIOS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
ALLOCATION_TEMPERATURES = (1.0, 2.0, 4.0)
BOUNDARY_MINIMUM_RETENTIONS = (0.3, 0.4)
ALLOCATION_SELECTION_BATCHES = 12
SCREENING_CALIBRATION_BATCHES = 96
ALLOCATION_SCORE_NAMES = (
    "canonical_bi",
    "residual_aware_mlp_bi",
    "singleton_kl_w25",
    "singleton_kl_w50",
    "singleton_loss_delta_w25",
    "singleton_loss_delta_w50",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the second-generation model-wide SwiGLU allocation search"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Explicit notebook-equivalent allocation-search configuration",
    )
    parser.add_argument(
        "--reference-artifact",
        type=Path,
        default=None,
        help="Schema-3 optimized SwiGLU artifact containing the matched control",
    )
    parser.add_argument("--output", type=Path, help="Schema-1 result path")
    return parser.parse_args()


def configure(settings: dict) -> Path:
    """Make the checked-in search configuration authoritative for this process."""

    global PROBE_WIDTH_RATIOS, REFERENCE_WIDTH_RATIO, WIDTH_SWEEP_RATIOS
    global ALLOCATION_TEMPERATURES, BOUNDARY_MINIMUM_RETENTIONS
    global ALLOCATION_SELECTION_BATCHES, SCREENING_CALIBRATION_BATCHES
    global ALLOCATION_SCORE_NAMES

    PROBE_WIDTH_RATIOS = tuple(float(value) for value in settings["probe_width_ratios"])
    REFERENCE_WIDTH_RATIO = float(settings["reference_width_ratio"])
    WIDTH_SWEEP_RATIOS = tuple(float(value) for value in settings["width_sweep_ratios"])
    ALLOCATION_TEMPERATURES = tuple(
        float(value) for value in settings["allocation_temperatures"]
    )
    BOUNDARY_MINIMUM_RETENTIONS = tuple(
        float(value) for value in settings["boundary_minimum_retentions"]
    )
    ALLOCATION_SELECTION_BATCHES = int(settings["allocation_selection_batches"])
    SCREENING_CALIBRATION_BATCHES = int(settings["screening_calibration_batches"])
    ALLOCATION_SCORE_NAMES = tuple(settings["allocation_score_names"])
    if PROBE_WIDTH_RATIOS != (0.25, 0.5) or REFERENCE_WIDTH_RATIO != 0.5:
        raise ValueError(
            "Schema-1 SwiGLU-2 artifacts require 0.25 and 0.5 probe widths"
        )
    required_scores = {
        "canonical_bi",
        "residual_aware_mlp_bi",
        "singleton_kl_w25",
        "singleton_kl_w50",
        "singleton_loss_delta_w25",
        "singleton_loss_delta_w50",
    }
    if set(ALLOCATION_SCORE_NAMES) != required_scores:
        raise ValueError("SwiGLU-2 config must contain the six notebook score families")
    return Path(settings["reference_artifact"])


def configs_from_reference(reference: dict):
    configuration = reference["configuration"]
    model_values = configuration["model"]
    revision = model_values.get("resolved_revision") or model_values.get("revision")
    if not revision:
        raise ValueError("Reference artifact does not record a model revision")
    model_config = ModelConfig(
        model_id=model_values["model_id"],
        revision=revision,
        tokenizer_revision=model_values.get("tokenizer_revision") or revision,
        device=model_values["device"],
        dtype=model_values["dtype"],
        trust_remote_code=model_values["trust_remote_code"],
    )
    data_values = dict(configuration["data"])
    for name in ("calibration_source", "model_validation_source", "test_source"):
        data_values[name] = DatasetSpec(**data_values[name])
    data_config = DataConfig(**data_values)
    capture_values = dict(configuration["capture"])
    capture_group_size = int(capture_values.pop("module_group_size"))
    capture_config = CaptureConfig(**capture_values)
    training_config = OperatorConfig(
        **configuration["operator_training"]["complete_method"]
    )
    recovery_values = dict(configuration["recovery"])
    recovery_values.pop("trainable_scope", None)
    recovery_config = RecoveryConfig(**recovery_values)
    layers = tuple(configuration["eligible_layers"])
    protected = tuple(configuration["protected_layers"])
    target_sparsity = float(configuration["target_mlp_sparsity"])
    pairs_per_batch = data_config.batch_size * data_config.sequence_length
    historical_pairs = int(
        configuration["methodologies"]["historical_random"]["calibration_pairs"]
    )
    if historical_pairs % pairs_per_batch:
        raise ValueError("Historical calibration pairs do not form full batches")
    base_batches = historical_pairs // pairs_per_batch
    additional_batches = data_config.num_calibration_batches - base_batches
    if additional_batches < 0:
        raise ValueError("Optimized calibration budget is smaller than the baseline")
    partitions = {
        "calibration": base_batches,
        "operator_validation": data_config.num_operator_validation_batches,
        "recovery": data_config.num_recovery_batches,
        "recovery_validation": data_config.num_recovery_validation_batches,
        "additional_calibration": additional_batches,
        "allocation_selection": ALLOCATION_SELECTION_BATCHES,
    }
    return (
        model_config,
        data_config,
        capture_config,
        capture_group_size,
        training_config,
        recovery_config,
        layers,
        protected,
        target_sparsity,
        partitions,
    )


def build_loaders(tokenizer, data_config, partitions):
    calibration_data = load_text_dataset(data_config.calibration_source)
    windows = sample_partitioned_windows(
        calibration_data,
        tokenizer,
        {name: batches * data_config.batch_size for name, batches in partitions.items()},
        data_config.sequence_length,
        data_config.seed,
        data_config.calibration_source.text_column,
    )
    calibration_sequences = windows["calibration"] + windows["additional_calibration"]
    validation_data = load_text_dataset(data_config.model_validation_source)
    validation_sequences = contiguous_token_windows(
        validation_data,
        tokenizer,
        data_config.num_model_validation_batches * data_config.batch_size,
        data_config.sequence_length,
        data_config.model_validation_source.text_column,
    )
    loaders = DataLoaders(
        calibration=make_token_loader(calibration_sequences, data_config.batch_size),
        operator_validation=make_token_loader(
            windows["operator_validation"], data_config.batch_size
        ),
        recovery=make_token_loader(windows["recovery"], data_config.batch_size),
        recovery_validation=make_token_loader(
            windows["recovery_validation"], data_config.batch_size
        ),
        model_validation=make_token_loader(validation_sequences, data_config.batch_size),
        test=None,
    )
    allocation_loader = make_token_loader(
        windows["allocation_selection"], data_config.batch_size
    )
    screening_loader = make_token_loader(
        calibration_sequences[: SCREENING_CALIBRATION_BATCHES * data_config.batch_size],
        data_config.batch_size,
    )
    del calibration_data, validation_data
    return loaders, allocation_loader, screening_loader


def capture_pairs(
    model,
    blocks_by_layer,
    group,
    training_loader,
    training_batches,
    validation_loader,
    validation_batches,
    capture_config,
    storage_dtype,
    device,
):
    paths = [blocks_by_layer[layer].path for layer in group]
    training = collect_modules_io(
        model,
        paths,
        training_loader,
        training_batches,
        device,
        storage_device=capture_config.storage_device,
        storage_dtype=storage_dtype,
    )
    validation = collect_modules_io(
        model,
        paths,
        validation_loader,
        validation_batches,
        device,
        storage_device=capture_config.storage_device,
        storage_dtype=storage_dtype,
    )
    return training, validation


def fit_teacher_initialized(
    block,
    training_pairs,
    validation_pairs,
    width,
    ranking,
    training_config,
    device,
):
    selected = ranking[:width].sort().values
    module = GatedMLPReplacement(training_pairs.hidden_size, width).to(device)
    initialize_gated_mlp_from_teacher(module, block.module, selected)
    initial = evaluate_operator(
        module, validation_pairs, device, training_config.batch_size
    )
    synchronize_cuda(torch, device)
    started = perf_counter()
    fit = fit_operator(
        module, training_pairs, validation_pairs, training_config, device
    )
    synchronize_cuda(torch, device)
    fit_seconds = perf_counter() - started
    metrics = evaluate_operator(
        fit.module, validation_pairs, device, training_config.batch_size
    )
    return fit, initial, metrics, fit_seconds


def run_probes(
    model,
    blocks_by_layer,
    layer_groups,
    layers,
    loaders,
    allocation_loader,
    data_config,
    capture_config,
    training_config,
    recovery_config,
    model_dtype,
    device,
    reference,
    runtime_rows,
):
    rankings = {
        int(layer): torch.tensor(values, dtype=torch.long)
        for layer, values in reference["results"]["teacher_neuron_rankings"].items()
    }
    teacher_cache = cache_teacher_logits(
        model,
        allocation_loader,
        ALLOCATION_SELECTION_BATCHES,
        device,
        recovery_config.cache_dtype,
    )
    dense_metrics = evaluate_language_model(
        model, allocation_loader, device, ALLOCATION_SELECTION_BATCHES
    )
    reference_operators = {}
    reference_fitting = []
    reference_history = []
    aggressive_fitting = []
    aggressive_history = []
    aggressive_impact = []
    phase_started = perf_counter()
    for group_index, group in enumerate(layer_groups, start=1):
        print(f"Probe capture {group_index}/{len(layer_groups)} | {group}", flush=True)
        capture_started = perf_counter()
        training_by_path, validation_by_path = capture_pairs(
            model,
            blocks_by_layer,
            group,
            loaders.calibration,
            data_config.num_calibration_batches,
            loaders.operator_validation,
            data_config.num_operator_validation_batches,
            capture_config,
            model_dtype,
            device,
        )
        runtime_rows.append(
            {
                "stage": "probe_capture",
                "group": group_index,
                "seconds": perf_counter() - capture_started,
            }
        )
        report_memory(f"After probe capture {group_index}")
        for layer in group:
            block = blocks_by_layer[layer]
            training_pairs = training_by_path[block.path]
            validation_pairs = validation_by_path[block.path]
            original_width = block.module.up_proj.out_features
            reference_width = round(original_width * REFERENCE_WIDTH_RATIO)
            print(f"Fitting width-0.5 reference | layer {layer}", flush=True)
            fit, initial, metrics, fit_seconds = fit_teacher_initialized(
                block,
                training_pairs,
                validation_pairs,
                reference_width,
                rankings[layer],
                training_config,
                device,
            )
            reference_operators[layer] = fit.module.to(
                device="cpu", dtype=model_dtype
            )
            reference_fitting.append(
                {
                    "policy": "uniform",
                    "layer": layer,
                    "probe_width_ratio": REFERENCE_WIDTH_RATIO,
                    "replacement_width": reference_width,
                    "replacement_width_ratio": reference_width / original_width,
                    "replacement_parameters": sum(
                        parameter.numel() for parameter in fit.module.parameters()
                    ),
                    "initial_local_mse": initial.mse,
                    "initial_local_relative_mse": initial.relative_mse,
                    "initial_local_cosine": initial.cosine_similarity,
                    "best_epoch": fit.best_epoch,
                    "epochs_completed": len(fit.history),
                    "local_mse": metrics.mse,
                    "local_relative_mse": metrics.relative_mse,
                    "local_cosine": metrics.cosine_similarity,
                    "fit_seconds": fit_seconds,
                }
            )
            reference_history.extend(
                {
                    "policy": "uniform",
                    "layer": layer,
                    "probe_width_ratio": REFERENCE_WIDTH_RATIO,
                    "epoch": epoch.epoch,
                    "train_mse": epoch.train_mse,
                    "validation_mse": epoch.validation_mse,
                    "learning_rate": epoch.learning_rate,
                }
                for epoch in fit.history
            )

            aggressive_ratio = PROBE_WIDTH_RATIOS[0]
            aggressive_width = max(1, round(original_width * aggressive_ratio))
            print(f"Fitting width-0.25 probe | layer {layer}", flush=True)
            aggressive_fit, aggressive_initial, aggressive_metrics, seconds = (
                fit_teacher_initialized(
                    block,
                    training_pairs,
                    validation_pairs,
                    aggressive_width,
                    rankings[layer],
                    training_config,
                    device,
                )
            )
            synchronize_cuda(torch, device)
            evaluation_started = perf_counter()
            with temporary_replacement(model, layer, aggressive_fit.module):
                model_metrics = evaluate_language_model(
                    model, allocation_loader, device, ALLOCATION_SELECTION_BATCHES
                )
                teacher_kl = mean_cache_loss(
                    model, teacher_cache, recovery_config.temperature, device
                )
            synchronize_cuda(torch, device)
            aggressive_fitting.append(
                {
                    "layer": layer,
                    "probe_width_ratio": aggressive_ratio,
                    "replacement_width": aggressive_width,
                    "replacement_width_ratio": aggressive_width / original_width,
                    "replacement_parameters": sum(
                        parameter.numel()
                        for parameter in aggressive_fit.module.parameters()
                    ),
                    "initial_local_mse": aggressive_initial.mse,
                    "initial_local_relative_mse": aggressive_initial.relative_mse,
                    "initial_local_cosine": aggressive_initial.cosine_similarity,
                    "best_epoch": aggressive_fit.best_epoch,
                    "epochs_completed": len(aggressive_fit.history),
                    "local_mse": aggressive_metrics.mse,
                    "local_relative_mse": aggressive_metrics.relative_mse,
                    "local_cosine": aggressive_metrics.cosine_similarity,
                    "fit_seconds": seconds,
                }
            )
            aggressive_impact.append(
                {
                    "layer": layer,
                    "probe_width_ratio": aggressive_ratio,
                    "singleton_kl": teacher_kl,
                    "singleton_loss": model_metrics.loss,
                    "singleton_loss_delta": model_metrics.loss - dense_metrics.loss,
                    "singleton_perplexity": model_metrics.perplexity,
                    "singleton_perplexity_delta": model_metrics.perplexity
                    - dense_metrics.perplexity,
                    "evaluation_seconds": perf_counter() - evaluation_started,
                }
            )
            aggressive_history.extend(
                {
                    "layer": layer,
                    "probe_width_ratio": aggressive_ratio,
                    "epoch": epoch.epoch,
                    "train_mse": epoch.train_mse,
                    "validation_mse": epoch.validation_mse,
                    "learning_rate": epoch.learning_rate,
                }
                for epoch in aggressive_fit.history
            )
            aggressive_fit.module.to(device="cpu", dtype=model_dtype)
        training_by_path.clear()
        validation_by_path.clear()
        release_cuda(torch)
        report_memory(f"After probe group {group_index}")

    reference_fitting_df = pd.DataFrame(reference_fitting)
    reference_history_df = pd.DataFrame(reference_history)
    aggressive_fitting_df = pd.DataFrame(aggressive_fitting)
    aggressive_history_df = pd.DataFrame(aggressive_history)
    runtime_rows.append(
        {
            "stage": "dual_width_probes_total",
            "group": None,
            "seconds": perf_counter() - phase_started,
        }
    )
    score_started = perf_counter()
    reference_impact = []
    for layer in layers:
        print(f"Singleton model-impact scoring | layer {layer}", flush=True)
        started = perf_counter()
        with temporary_replacement(model, layer, reference_operators[layer]):
            metrics = evaluate_language_model(
                model, allocation_loader, device, ALLOCATION_SELECTION_BATCHES
            )
            teacher_kl = mean_cache_loss(
                model, teacher_cache, recovery_config.temperature, device
            )
        reference_operators[layer].to(device="cpu", dtype=model_dtype)
        reference_impact.append(
            {
                "layer": layer,
                "probe_width_ratio": REFERENCE_WIDTH_RATIO,
                "singleton_kl": teacher_kl,
                "singleton_loss": metrics.loss,
                "singleton_loss_delta": metrics.loss - dense_metrics.loss,
                "singleton_perplexity": metrics.perplexity,
                "singleton_perplexity_delta": metrics.perplexity
                - dense_metrics.perplexity,
                "evaluation_seconds": perf_counter() - started,
            }
        )
    reference_impact_df = pd.DataFrame(reference_impact)
    aggressive_impact_df = pd.DataFrame(aggressive_impact)
    fitting_df = pd.concat(
        [aggressive_fitting_df, reference_fitting_df], ignore_index=True
    )
    history_df = pd.concat(
        [aggressive_history_df, reference_history_df], ignore_index=True
    )
    impact_df = pd.concat(
        [aggressive_impact_df, reference_impact_df], ignore_index=True
    )
    candidates = pd.DataFrame(reference["results"]["importance"])
    for ratio, suffix in ((0.25, "w25"), (0.5, "w50")):
        local = fitting_df[fitting_df["probe_width_ratio"] == ratio][
            [
                "layer",
                "initial_local_relative_mse",
                "local_relative_mse",
                "local_cosine",
            ]
        ].rename(
            columns={
                "initial_local_relative_mse": f"initial_local_relative_mse_{suffix}",
                "local_relative_mse": f"local_relative_mse_{suffix}",
                "local_cosine": f"local_cosine_{suffix}",
            }
        )
        impact = impact_df[impact_df["probe_width_ratio"] == ratio][
            [
                "layer",
                "singleton_kl",
                "singleton_loss",
                "singleton_loss_delta",
                "singleton_perplexity",
                "singleton_perplexity_delta",
            ]
        ].rename(
            columns={
                name: f"{name}_{suffix}"
                for name in (
                    "singleton_kl",
                    "singleton_loss",
                    "singleton_loss_delta",
                    "singleton_perplexity",
                    "singleton_perplexity_delta",
                )
            }
        )
        candidates = candidates.merge(local, on="layer", validate="one_to_one").merge(
            impact, on="layer", validate="one_to_one"
        )
    for score_name in ALLOCATION_SCORE_NAMES:
        candidates[f"{score_name}_rank"] = (
            candidates[score_name]
            .rank(method="min", ascending=False)
            .astype(int)
        )
    runtime_rows.append(
        {
            "stage": "candidate_scores_total",
            "group": None,
            "seconds": perf_counter() - score_started,
        }
    )
    return {
        "reference_operators": reference_operators,
        "rankings": rankings,
        "teacher_cache": teacher_cache,
        "selection_dense_metrics": dense_metrics,
        "reference_fitting": reference_fitting_df,
        "reference_history": reference_history_df,
        "reference_impact": reference_impact_df,
        "fitting": fitting_df,
        "history": history_df,
        "impact": impact_df,
        "candidates": candidates,
    }


def run_width_study(
    model,
    blocks_by_layer,
    loaders,
    allocation_loader,
    data_config,
    capture_config,
    training_config,
    recovery_config,
    model_dtype,
    device,
    probe,
    runtime_rows,
):
    candidates = probe["candidates"]
    representative_blocks = {}
    membership_rows = []
    for basis, score_name in (("kl50", "singleton_kl_w50"), ("kl25", "singleton_kl_w25")):
        ordered = candidates.sort_values(score_name).reset_index(drop=True)
        cohort = {
            "best": int(ordered.iloc[0]["layer"]),
            "middle": int(ordered.iloc[len(ordered) // 2]["layer"]),
            "worst": int(ordered.iloc[-1]["layer"]),
        }
        representative_blocks[basis] = cohort
        membership_rows.extend(
            {
                "selection_basis": basis,
                "impact_group": impact_group,
                "layer": layer,
                "selection_score": float(
                    candidates.loc[candidates["layer"] == layer, score_name].iloc[0]
                ),
            }
            for impact_group, layer in cohort.items()
        )
    membership = pd.DataFrame(membership_rows)
    study_layers = tuple(dict.fromkeys(membership["layer"].tolist()))
    capture_started = perf_counter()
    training_by_path, validation_by_path = capture_pairs(
        model,
        blocks_by_layer,
        study_layers,
        loaders.calibration,
        data_config.num_calibration_batches,
        loaders.operator_validation,
        data_config.num_operator_validation_batches,
        capture_config,
        model_dtype,
        device,
    )
    runtime_rows.append(
        {
            "stage": "width_study_capture",
            "group": None,
            "seconds": perf_counter() - capture_started,
        }
    )
    rows = []
    history_rows = []
    for layer in study_layers:
        block = blocks_by_layer[layer]
        training_pairs = training_by_path[block.path]
        validation_pairs = validation_by_path[block.path]
        original_width = block.module.up_proj.out_features
        for ratio in WIDTH_SWEEP_RATIOS:
            width = max(1, round(original_width * ratio))
            print(f"Width study | layer {layer} | width {ratio:.1f}", flush=True)
            if math.isclose(ratio, REFERENCE_WIDTH_RATIO):
                fit_row = probe["reference_fitting"].query("layer == @layer").iloc[0]
                impact_row = probe["reference_impact"].query("layer == @layer").iloc[0]
                rows.append(
                    {
                        "layer": layer,
                        "width_ratio": ratio,
                        "replacement_width": int(fit_row["replacement_width"]),
                        "replacement_parameters": int(
                            fit_row["replacement_parameters"]
                        ),
                        "initial_local_relative_mse": fit_row[
                            "initial_local_relative_mse"
                        ],
                        "local_relative_mse": fit_row["local_relative_mse"],
                        "local_cosine": fit_row["local_cosine"],
                        "singleton_kl": impact_row["singleton_kl"],
                        "singleton_loss": impact_row["singleton_loss"],
                        "singleton_loss_delta": impact_row["singleton_loss_delta"],
                        "singleton_perplexity": impact_row["singleton_perplexity"],
                        "best_epoch": int(fit_row["best_epoch"]),
                        "epochs_completed": int(fit_row["epochs_completed"]),
                        "fit_seconds": fit_row["fit_seconds"],
                        "evaluation_seconds": impact_row["evaluation_seconds"],
                        "reused_probe": True,
                    }
                )
                history_rows.extend(
                    {
                        "layer": layer,
                        "width_ratio": ratio,
                        "epoch": epoch.epoch,
                        "train_mse": epoch.train_mse,
                        "validation_mse": epoch.validation_mse,
                        "learning_rate": epoch.learning_rate,
                    }
                    for epoch in probe["reference_history"].query(
                        "layer == @layer"
                    ).itertuples(index=False)
                )
                continue
            fit, initial, metrics, fit_seconds = fit_teacher_initialized(
                block,
                training_pairs,
                validation_pairs,
                width,
                probe["rankings"][layer],
                training_config,
                device,
            )
            evaluation_started = perf_counter()
            with temporary_replacement(model, layer, fit.module):
                model_metrics = evaluate_language_model(
                    model, allocation_loader, device, ALLOCATION_SELECTION_BATCHES
                )
                teacher_kl = mean_cache_loss(
                    model,
                    probe["teacher_cache"],
                    recovery_config.temperature,
                    device,
                )
            rows.append(
                {
                    "layer": layer,
                    "width_ratio": ratio,
                    "replacement_width": width,
                    "replacement_parameters": sum(
                        parameter.numel() for parameter in fit.module.parameters()
                    ),
                    "initial_local_relative_mse": initial.relative_mse,
                    "local_relative_mse": metrics.relative_mse,
                    "local_cosine": metrics.cosine_similarity,
                    "singleton_kl": teacher_kl,
                    "singleton_loss": model_metrics.loss,
                    "singleton_loss_delta": model_metrics.loss
                    - probe["selection_dense_metrics"].loss,
                    "singleton_perplexity": model_metrics.perplexity,
                    "best_epoch": fit.best_epoch,
                    "epochs_completed": len(fit.history),
                    "fit_seconds": fit_seconds,
                    "evaluation_seconds": perf_counter() - evaluation_started,
                    "reused_probe": False,
                }
            )
            history_rows.extend(
                {
                    "layer": layer,
                    "width_ratio": ratio,
                    "epoch": epoch.epoch,
                    "train_mse": epoch.train_mse,
                    "validation_mse": epoch.validation_mse,
                    "learning_rate": epoch.learning_rate,
                }
                for epoch in fit.history
            )
            fit.module.to(device="cpu", dtype=model_dtype)
            release_cuda(torch)
    training_by_path.clear()
    validation_by_path.clear()
    report_memory("After representative-block study")
    return representative_blocks, membership, pd.DataFrame(rows), pd.DataFrame(history_rows)


def rank_normalize_scores(scores):
    values = pd.Series(scores, dtype=float)
    if len(values) == 1:
        return {int(values.index[0]): 0.0}
    return ((values.rank(method="average") - 1) / (len(values) - 1)).to_dict()


def allocate_widths(
    spec,
    candidates,
    layers,
    blocks_by_layer,
    original_parameters,
    eligible_parameters,
    target_sparsity,
):
    score_name = spec["score_name"]
    temperature = spec["temperature"]
    minimum_retention = spec["minimum_retention"]
    scores = (
        None
        if score_name is None
        else dict(zip(candidates["layer"], candidates[score_name]))
    )
    normalized = None if scores is None else rank_normalize_scores(scores)
    propensities = {
        layer: 1.0
        if normalized is None
        else math.exp(-normalized[layer] / temperature)
        for layer in layers
    }
    target_removed = round(eligible_parameters * target_sparsity)
    limits = {
        layer: original_parameters[layer]
        if minimum_retention is None
        else original_parameters[layer] * (1 - minimum_retention)
        for layer in layers
    }
    allocated = {layer: 0.0 for layer in layers}
    remaining = float(target_removed)
    active = set(layers)
    while active:
        denominator = sum(
            original_parameters[layer] * propensities[layer] for layer in active
        )
        proposed = {
            layer: remaining
            * original_parameters[layer]
            * propensities[layer]
            / denominator
            for layer in active
        }
        capped = [layer for layer in active if proposed[layer] > limits[layer]]
        if not capped:
            for layer in active:
                allocated[layer] = proposed[layer]
            remaining = 0.0
            break
        for layer in capped:
            allocated[layer] = limits[layer]
            remaining -= limits[layer]
            active.remove(layer)
    if remaining > 1e-3:
        raise ValueError("Minimum retained width makes the budget infeasible")
    rows = []
    for layer in layers:
        block = blocks_by_layer[layer].module
        hidden_size = block.up_proj.in_features
        original_width = block.up_proj.out_features
        parameter_step = 3 * hidden_size
        retained = original_parameters[layer] - allocated[layer]
        continuous_width = retained / parameter_step
        minimum_width = (
            1
            if minimum_retention is None
            else math.ceil(original_width * minimum_retention)
        )
        rows.append(
            {
                "policy": spec["policy"],
                "score_name": score_name or "uniform",
                "temperature": temperature,
                "bounded": minimum_retention is not None,
                "minimum_retention": minimum_retention,
                "layer": layer,
                "raw_importance": None if scores is None else scores[layer],
                "normalized_importance": (
                    None if normalized is None else normalized[layer]
                ),
                "continuous_width": continuous_width,
                "replacement_width": max(
                    minimum_width, math.floor(continuous_width)
                ),
                "minimum_width": minimum_width,
                "original_width": original_width,
                "parameter_step": parameter_step,
                "original_parameters": original_parameters[layer],
            }
        )
    parameter_steps = {row["parameter_step"] for row in rows}
    if len(parameter_steps) != 1:
        raise ValueError("Allocation requires equal per-neuron parameter cost")
    parameter_step = parameter_steps.pop()
    target_retained = eligible_parameters - target_removed
    if target_retained % parameter_step:
        raise ValueError("Target budget is not representable by whole neurons")
    target_width = target_retained // parameter_step
    current_width = sum(row["replacement_width"] for row in rows)
    while current_width < target_width:
        choices = [
            row for row in rows if row["replacement_width"] < row["original_width"]
        ]
        row = max(
            choices,
            key=lambda item: item["continuous_width"] - item["replacement_width"],
        )
        row["replacement_width"] += 1
        current_width += 1
    while current_width > target_width:
        choices = [
            row for row in rows if row["replacement_width"] > row["minimum_width"]
        ]
        if not choices:
            raise ValueError("Rounded minimum widths exceed the budget")
        row = max(
            choices,
            key=lambda item: item["replacement_width"] - item["continuous_width"],
        )
        row["replacement_width"] -= 1
        current_width -= 1
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
        for key in ("continuous_width", "minimum_width", "parameter_step"):
            row.pop(key)
    return pd.DataFrame(rows)


def allocation_specs():
    specs = [
        {
            "policy": "uniform",
            "score_name": None,
            "temperature": None,
            "minimum_retention": None,
        }
    ]
    specs.extend(
        {
            "policy": f"{score_name}_t{temperature:g}",
            "score_name": score_name,
            "temperature": temperature,
            "minimum_retention": None,
        }
        for score_name in ALLOCATION_SCORE_NAMES
        for temperature in ALLOCATION_TEMPERATURES
    )
    return specs


def allocation_summary(allocation, fixed_parameters, dense_parameters):
    summary = (
        allocation.groupby(
            ["policy", "score_name", "temperature", "bounded"],
            dropna=False,
            sort=False,
            as_index=False,
        )
        .agg(
            original_mlp_parameters=("original_parameters", "sum"),
            replacement_mlp_parameters=("replacement_parameters", "sum"),
            minimum_width_ratio=("replacement_width_ratio", "min"),
            maximum_width_ratio=("replacement_width_ratio", "max"),
        )
    )
    summary["mlp_parameter_reduction_pct"] = 100 * (
        1
        - summary["replacement_mlp_parameters"]
        / summary["original_mlp_parameters"]
    )
    summary["model_parameters"] = (
        fixed_parameters + summary["replacement_mlp_parameters"]
    )
    summary["model_parameter_reduction_pct"] = 100 * (
        1 - summary["model_parameters"] / dense_parameters
    )
    return summary


def fit_policy_set(
    model,
    blocks_by_layer,
    layer_groups,
    allocation,
    policies,
    training_loader,
    training_batches,
    loaders,
    data_config,
    capture_config,
    training_config,
    rankings,
    model_dtype,
    device,
    runtime_stage,
    runtime_rows,
):
    fitted = {policy: {} for policy in policies}
    fitting_rows = []
    history_rows = []
    phase_started = perf_counter()
    for group_index, group in enumerate(layer_groups, start=1):
        print(
            f"{runtime_stage} capture {group_index}/{len(layer_groups)} | {group}",
            flush=True,
        )
        capture_started = perf_counter()
        training_by_path, validation_by_path = capture_pairs(
            model,
            blocks_by_layer,
            group,
            training_loader,
            training_batches,
            loaders.operator_validation,
            data_config.num_operator_validation_batches,
            capture_config,
            model_dtype,
            device,
        )
        runtime_rows.append(
            {
                "stage": f"{runtime_stage}_capture",
                "group": group_index,
                "seconds": perf_counter() - capture_started,
            }
        )
        group_allocation = allocation[
            allocation["layer"].isin(group) & allocation["policy"].isin(policies)
        ]
        for policy, rows in group_allocation.groupby("policy", sort=False):
            print(f"{runtime_stage} fitting | {policy}", flush=True)
            for item in rows.itertuples(index=False):
                layer = int(item.layer)
                block = blocks_by_layer[layer]
                training_pairs = training_by_path[block.path]
                validation_pairs = validation_by_path[block.path]
                width = int(item.replacement_width)
                fit, initial, metrics, fit_seconds = fit_teacher_initialized(
                    block,
                    training_pairs,
                    validation_pairs,
                    width,
                    rankings[layer],
                    training_config,
                    device,
                )
                fitted[policy][layer] = fit.module.to(
                    device="cpu", dtype=model_dtype
                )
                fitting_rows.append(
                    {
                        "policy": policy,
                        "score_name": item.score_name,
                        "temperature": item.temperature,
                        "bounded": bool(item.bounded),
                        "minimum_retention": item.minimum_retention,
                        "layer": layer,
                        "replacement_width": width,
                        "replacement_width_ratio": item.replacement_width_ratio,
                        "replacement_parameters": item.replacement_parameters,
                        "initial_local_relative_mse": initial.relative_mse,
                        "local_relative_mse": metrics.relative_mse,
                        "local_cosine": metrics.cosine_similarity,
                        "best_epoch": fit.best_epoch,
                        "epochs_completed": len(fit.history),
                        "fit_seconds": fit_seconds,
                    }
                )
                history_rows.extend(
                    {
                        "policy": policy,
                        "layer": layer,
                        "epoch": epoch.epoch,
                        "train_mse": epoch.train_mse,
                        "validation_mse": epoch.validation_mse,
                        "learning_rate": epoch.learning_rate,
                    }
                    for epoch in fit.history
                )
        training_by_path.clear()
        validation_by_path.clear()
        release_cuda(torch)
        report_memory(f"After {runtime_stage} group {group_index}")
    runtime_rows.append(
        {
            "stage": f"{runtime_stage}_fitting_total",
            "group": None,
            "seconds": perf_counter() - phase_started,
        }
    )
    return fitted, pd.DataFrame(fitting_rows), pd.DataFrame(history_rows)


def evaluate_screening(
    model,
    fitted,
    specs,
    allocation_loader,
    teacher_cache,
    dense_metrics,
    recovery_config,
    dense_parameters,
    model_dtype,
    device,
):
    rows = []
    for spec in specs:
        policy = spec["policy"]
        print(f"Reduced-budget model evaluation | {policy}", flush=True)
        started = perf_counter()
        replacements = fitted[policy]
        with temporary_replacements(model, replacements):
            footprint = parameter_footprint(model)
            metrics = evaluate_language_model(
                model, allocation_loader, device, ALLOCATION_SELECTION_BATCHES
            )
            teacher_kl = mean_cache_loss(
                model, teacher_cache, recovery_config.temperature, device
            )
        rows.append(
            {
                "policy": policy,
                "score_name": spec["score_name"] or "uniform",
                "temperature": spec["temperature"],
                "bounded": spec["minimum_retention"] is not None,
                "minimum_retention": spec["minimum_retention"],
                "parameters": footprint.parameters,
                "model_parameter_reduction_pct": 100
                * (1 - footprint.parameters / dense_parameters),
                "teacher_kl": teacher_kl,
                "loss": metrics.loss,
                "loss_delta": metrics.loss - dense_metrics.loss,
                "perplexity": metrics.perplexity,
                "perplexity_delta": metrics.perplexity - dense_metrics.perplexity,
                "evaluation_seconds": perf_counter() - started,
            }
        )
        for replacement in replacements.values():
            replacement.to(device="cpu", dtype=model_dtype)
        release_cuda(torch)
    return pd.DataFrame(rows)


def choose_finalists(screening):
    bi = screening[
        screening["score_name"].isin(["canonical_bi", "residual_aware_mlp_bi"])
        & ~screening["bounded"]
    ]
    kl = screening[screening["score_name"].str.startswith("singleton_kl_")]
    loss = screening[
        screening["score_name"].str.startswith("singleton_loss_delta_")
    ]
    nonuniform = screening[screening["policy"] != "uniform"]
    best_unbounded = str(nonuniform.nsmallest(1, "teacher_kl").iloc[0]["policy"])
    finalists = list(
        dict.fromkeys(
            [
                "uniform",
                str(bi.nsmallest(1, "teacher_kl").iloc[0]["policy"]),
                str(kl.nsmallest(1, "teacher_kl").iloc[0]["policy"]),
                str(loss.nsmallest(1, "loss_delta").iloc[0]["policy"]),
                best_unbounded,
            ]
        )
    )
    return best_unbounded, finalists


def run_workflow(output: Path, reference_path: Path, settings: dict) -> dict:
    output = require_new_path(output)
    reference_path = resolve_path(reference_path)
    reference = load_artifact(reference_path, REFERENCE_SCHEMA, "optimized SwiGLU reference")
    (
        model_config,
        data_config,
        capture_config,
        group_size,
        training_config,
        recovery_config,
        layers,
        protected,
        target_sparsity,
        partitions,
    ) = configs_from_reference(reference)
    run_log = start_run_log(
        output,
        WORKFLOW,
        {
            "reference_artifact": str(reference_path),
            "workflow_config": settings,
        },
    )
    started = perf_counter()
    runtime_rows = []
    try:
        torch.manual_seed(int(data_config.seed))
        run_log.begin("load_model_and_data")
        model, tokenizer = load_model_and_tokenizer(model_config)
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        blocks = discover_mlp_blocks(model)
        blocks_by_layer = {block.index: block for block in blocks}
        if not set(layers).issubset(blocks_by_layer):
            raise ValueError("Reference eligible layers do not match the model")
        loaders, allocation_loader, screening_loader = build_loaders(
            tokenizer, data_config, partitions
        )
        dense_reference = pd.DataFrame(reference["results"]["dense_reference"])
        dense_footprint = parameter_footprint(model)
        if dense_footprint.parameters != int(dense_reference.iloc[0]["parameters"]):
            raise ValueError("Loaded model footprint differs from the reference")
        original_parameters = {
            layer: sum(
                parameter.numel()
                for parameter in blocks_by_layer[layer].module.parameters()
            )
            for layer in layers
        }
        eligible_parameters = sum(original_parameters.values())
        fixed_parameters = dense_footprint.parameters - eligible_parameters
        layer_groups = [
            layers[start_index : start_index + group_size]
            for start_index in range(0, len(layers), group_size)
        ]
        run_log.record(
            "load_model_and_data",
            {
                "eligible_layers": list(layers),
                "partition_batches": partitions,
                "capture_groups": [list(group) for group in layer_groups],
            },
        )

        run_log.begin("dual_width_probes")
        probe = run_probes(
            model,
            blocks_by_layer,
            layer_groups,
            layers,
            loaders,
            allocation_loader,
            data_config,
            capture_config,
            training_config,
            recovery_config,
            model_dtype,
            device,
            reference,
            runtime_rows,
        )
        run_log.record(
            "dual_width_probes", {"operators": len(probe["fitting"])}
        )

        run_log.begin("representative_width_study")
        representatives, membership, width_study, width_history = run_width_study(
            model,
            blocks_by_layer,
            loaders,
            allocation_loader,
            data_config,
            capture_config,
            training_config,
            recovery_config,
            model_dtype,
            device,
            probe,
            runtime_rows,
        )
        run_log.record(
            "representative_width_study", {"fits": len(width_study)}
        )

        specs = allocation_specs()
        allocation = pd.concat(
            [
                allocate_widths(
                    spec,
                    probe["candidates"],
                    layers,
                    blocks_by_layer,
                    original_parameters,
                    eligible_parameters,
                    target_sparsity,
                )
                for spec in specs
            ],
            ignore_index=True,
        )
        allocation_summary_df = allocation_summary(
            allocation, fixed_parameters, dense_footprint.parameters
        )

        run_log.begin("reduced_budget_screening")
        screening_fitted, screening_fitting, screening_history = fit_policy_set(
            model,
            blocks_by_layer,
            layer_groups,
            allocation,
            [spec["policy"] for spec in specs],
            screening_loader,
            SCREENING_CALIBRATION_BATCHES,
            loaders,
            data_config,
            capture_config,
            training_config,
            probe["rankings"],
            model_dtype,
            device,
            "screening",
            runtime_rows,
        )
        screening_fitting = screening_fitting.drop(
            columns=["minimum_retention"]
        )
        screening_started = perf_counter()
        screening_model = evaluate_screening(
            model,
            screening_fitted,
            specs,
            allocation_loader,
            probe["teacher_cache"],
            probe["selection_dense_metrics"],
            recovery_config,
            dense_footprint.parameters,
            model_dtype,
            device,
        )
        screening_model = screening_model.drop(columns=["minimum_retention"])
        runtime_rows.append(
            {
                "stage": "screening_model_evaluation_total",
                "group": None,
                "seconds": perf_counter() - screening_started,
            }
        )
        best_unbounded, finalists = choose_finalists(screening_model)
        screening_fitted.clear()
        release_cuda(torch)
        run_log.record(
            "reduced_budget_screening",
            {"best_unbounded_policy": best_unbounded, "finalists": finalists},
        )

        run_log.begin("minimum_width_ablation")
        boundary_started = perf_counter()
        best_spec = next(spec for spec in specs if spec["policy"] == best_unbounded)
        base_boundary = allocation[allocation["policy"] == best_unbounded].copy()
        boundary_specs = []
        boundary_frames = []
        for minimum in BOUNDARY_MINIMUM_RETENTIONS:
            spec = {
                **best_spec,
                "policy": f"{best_unbounded}_floor{int(minimum * 100)}",
                "minimum_retention": minimum,
            }
            frame = allocate_widths(
                spec,
                probe["candidates"],
                layers,
                blocks_by_layer,
                original_parameters,
                eligible_parameters,
                target_sparsity,
            )
            existing = [base_boundary["replacement_width"].tolist()] + [
                item["replacement_width"].tolist() for item in boundary_frames
            ]
            if frame["replacement_width"].tolist() not in existing:
                boundary_specs.append(spec)
                boundary_frames.append(frame)
        new_boundary = (
            pd.concat(boundary_frames, ignore_index=True)
            if boundary_frames
            else allocation.iloc[0:0].copy()
        )
        boundary_allocation = pd.concat(
            [base_boundary, new_boundary], ignore_index=True
        )
        final_allocation = pd.concat([allocation, new_boundary], ignore_index=True)
        if boundary_specs:
            boundary_fitted, boundary_fitting, boundary_history = fit_policy_set(
                model,
                blocks_by_layer,
                layer_groups,
                new_boundary,
                [spec["policy"] for spec in boundary_specs],
                screening_loader,
                SCREENING_CALIBRATION_BATCHES,
                loaders,
                data_config,
                capture_config,
                training_config,
                probe["rankings"],
                model_dtype,
                device,
                "minimum_width",
                runtime_rows,
            )
            boundary_model = evaluate_screening(
                model,
                boundary_fitted,
                boundary_specs,
                allocation_loader,
                probe["teacher_cache"],
                probe["selection_dense_metrics"],
                recovery_config,
                dense_footprint.parameters,
                model_dtype,
                device,
            )
            boundary_fitted.clear()
            boundary_fitting = boundary_fitting[
                [
                    "policy",
                    "layer",
                    "minimum_retention",
                    "replacement_width",
                    "replacement_width_ratio",
                    "replacement_parameters",
                    "initial_local_relative_mse",
                    "local_relative_mse",
                    "local_cosine",
                    "best_epoch",
                    "epochs_completed",
                    "fit_seconds",
                ]
            ]
        else:
            boundary_fitting = screening_fitting.iloc[0:0].copy()
            boundary_history = screening_history.iloc[0:0].copy()
            boundary_model = screening_model.iloc[0:0].copy()
        base_boundary_model = screening_model[
            screening_model["policy"] == best_unbounded
        ].copy()
        base_boundary_model["minimum_retention"] = None
        base_local = screening_fitting[
            screening_fitting["policy"] == best_unbounded
        ]
        boundary_local = pd.concat(
            [base_local, boundary_fitting], ignore_index=True
        ).groupby("policy", as_index=False).agg(
            mean_nmse=("local_relative_mse", "mean"),
            worst_nmse=("local_relative_mse", "max"),
        )
        boundary_widths = boundary_allocation.groupby("policy", as_index=False).agg(
            minimum_width_ratio=("replacement_width_ratio", "min"),
            maximum_width_ratio=("replacement_width_ratio", "max"),
        )
        boundary_screening = (
            pd.concat([base_boundary_model, boundary_model], ignore_index=True)
            .merge(boundary_local, on="policy", validate="one_to_one")
            .merge(boundary_widths, on="policy", validate="one_to_one")
        )
        best_bounded = (
            None
            if boundary_model.empty
            else str(boundary_model.nsmallest(1, "teacher_kl").iloc[0]["policy"])
        )
        if best_bounded is not None:
            finalists = list(dict.fromkeys([*finalists, best_bounded]))
        runtime_rows.append(
            {
                "stage": "minimum_width_ablation_total",
                "group": None,
                "seconds": perf_counter() - boundary_started,
            }
        )
        del probe["teacher_cache"]
        probe["teacher_cache"] = None
        release_cuda(torch)
        run_log.record(
            "minimum_width_ablation",
            {"best_bounded_policy": best_bounded, "finalists": finalists},
        )

        run_log.begin("full_budget_finalists")
        nonuniform = [policy for policy in finalists if policy != "uniform"]
        fitted_nonuniform, full_fitting, full_history = fit_policy_set(
            model,
            blocks_by_layer,
            layer_groups,
            final_allocation,
            nonuniform,
            loaders.calibration,
            data_config.num_calibration_batches,
            loaders,
            data_config,
            capture_config,
            training_config,
            probe["rankings"],
            model_dtype,
            device,
            "finalist",
            runtime_rows,
        )
        full_fitted = {"uniform": probe["reference_operators"], **fitted_nonuniform}
        full_fitting = full_fitting.drop(columns=["minimum_retention"])
        uniform_rows = probe["reference_fitting"].copy()
        uniform_rows["score_name"] = "uniform"
        uniform_rows["temperature"] = None
        uniform_rows["bounded"] = False
        full_fitting = pd.concat([uniform_rows, full_fitting], ignore_index=True)
        full_history = pd.concat(
            [probe["reference_history"], full_history], ignore_index=True
        )
        run_log.record("full_budget_finalists", {"finalists": finalists})

        run_log.begin("finalist_evaluation_and_recovery")
        validation_cache = cache_teacher_logits(
            model,
            loaders.model_validation,
            data_config.num_model_validation_batches,
            device,
            recovery_config.cache_dtype,
        )
        dense_loss = float(dense_reference.iloc[0]["loss"])
        dense_perplexity = float(dense_reference.iloc[0]["perplexity"])
        pre_rows = []
        pre_started = perf_counter()
        for policy in finalists:
            print(f"Full validation before recovery | {policy}", flush=True)
            evaluation_started = perf_counter()
            replacements = full_fitted[policy]
            with temporary_replacements(model, replacements):
                footprint = parameter_footprint(model)
                metrics = evaluate_language_model(
                    model,
                    loaders.model_validation,
                    device,
                    data_config.num_model_validation_batches,
                )
                teacher_kl = mean_cache_loss(
                    model, validation_cache, recovery_config.temperature, device
                )
            policy_allocation = final_allocation[
                final_allocation["policy"] == policy
            ]
            pre_rows.append(
                {
                    "policy": policy,
                    "phase": "pre_recovery",
                    "parameters": footprint.parameters,
                    "model_parameter_reduction_pct": 100
                    * (1 - footprint.parameters / dense_footprint.parameters),
                    "mlp_parameter_reduction_pct": 100
                    * (
                        1
                        - policy_allocation["replacement_parameters"].sum()
                        / policy_allocation["original_parameters"].sum()
                    ),
                    "teacher_kl": teacher_kl,
                    "loss": metrics.loss,
                    "loss_delta": metrics.loss - dense_loss,
                    "perplexity": metrics.perplexity,
                    "perplexity_delta": metrics.perplexity - dense_perplexity,
                    "evaluation_seconds": perf_counter() - evaluation_started,
                }
            )
            for replacement in replacements.values():
                replacement.to(device="cpu", dtype=model_dtype)
            release_cuda(torch)
        pre_recovery = pd.DataFrame(pre_rows)
        runtime_rows.append(
            {
                "stage": "finalist_pre_recovery_evaluation_total",
                "group": None,
                "seconds": perf_counter() - pre_started,
            }
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
        post_rows = []
        recovery_rows = []
        recovery_started_total = perf_counter()
        for policy in finalists:
            print(f"Model-wide recovery | {policy}", flush=True)
            replacements = {
                layer: deepcopy(full_fitted[policy][layer]) for layer in layers
            }
            with temporary_replacements(model, replacements) as manifest:
                recovery_started = perf_counter()
                recovery = recover_replacements(
                    model,
                    recovery_cache,
                    recovery_validation_cache,
                    [record.path for record in manifest.records],
                    recovery_config,
                    device,
                )
                recovery_seconds = perf_counter() - recovery_started
                footprint = parameter_footprint(model)
                evaluation_started = perf_counter()
                metrics = evaluate_language_model(
                    model,
                    loaders.model_validation,
                    device,
                    data_config.num_model_validation_batches,
                )
                teacher_kl = mean_cache_loss(
                    model, validation_cache, recovery_config.temperature, device
                )
                allocation_rows = final_allocation[
                    final_allocation["policy"] == policy
                ]
                post_rows.append(
                    {
                        "policy": policy,
                        "phase": "post_recovery",
                        "parameters": footprint.parameters,
                        "model_parameter_reduction_pct": 100
                        * (1 - footprint.parameters / dense_footprint.parameters),
                        "mlp_parameter_reduction_pct": 100
                        * (
                            1
                            - allocation_rows["replacement_parameters"].sum()
                            / allocation_rows["original_parameters"].sum()
                        ),
                        "teacher_kl": teacher_kl,
                        "loss": metrics.loss,
                        "loss_delta": metrics.loss - dense_loss,
                        "perplexity": metrics.perplexity,
                        "perplexity_delta": metrics.perplexity - dense_perplexity,
                        "evaluation_seconds": perf_counter() - evaluation_started,
                        "recovery_seconds": recovery_seconds,
                    }
                )
                recovery_rows.extend(
                    {
                        "policy": policy,
                        "epoch": epoch.epoch,
                        "train_kl": epoch.train_kl,
                        "validation_kl": epoch.validation_kl,
                        "best_epoch": recovery.best_epoch,
                        "recovery_seconds": recovery_seconds,
                    }
                    for epoch in recovery.history
                )
            del replacements
            release_cuda(torch)
        evaluation = pd.concat(
            [pre_recovery, pd.DataFrame(post_rows)], ignore_index=True
        )
        recovery_history = pd.DataFrame(recovery_rows)
        runtime_rows.append(
            {
                "stage": "model_wide_recovery_total",
                "group": None,
                "seconds": perf_counter() - recovery_started_total,
            }
        )
        del validation_cache, recovery_cache, recovery_validation_cache
        full_fitted.clear()
        fitted_nonuniform.clear()
        probe["reference_operators"].clear()
        release_cuda(torch)
        run_log.record(
            "finalist_evaluation_and_recovery", {"evaluations": len(evaluation)}
        )

        local_summary = full_fitting.groupby("policy", as_index=False).agg(
            mean_nmse=("local_relative_mse", "mean"),
            worst_nmse=("local_relative_mse", "max"),
            total_fit_minutes=("fit_seconds", lambda values: values.sum() / 60),
        )
        pre = evaluation.query("phase == 'pre_recovery'")[
            [
                "policy",
                "parameters",
                "model_parameter_reduction_pct",
                "mlp_parameter_reduction_pct",
                "teacher_kl",
                "loss",
                "loss_delta",
                "perplexity",
                "perplexity_delta",
                "evaluation_seconds",
            ]
        ].rename(
            columns={
                "teacher_kl": "pre_teacher_kl",
                "loss": "pre_loss",
                "loss_delta": "pre_loss_delta",
                "perplexity": "pre_perplexity",
                "perplexity_delta": "pre_perplexity_delta",
                "evaluation_seconds": "pre_evaluation_seconds",
            }
        )
        post = evaluation.query("phase == 'post_recovery'")[
            [
                "policy",
                "teacher_kl",
                "loss",
                "loss_delta",
                "perplexity",
                "perplexity_delta",
                "evaluation_seconds",
                "recovery_seconds",
            ]
        ].rename(
            columns={
                "teacher_kl": "post_teacher_kl",
                "loss": "post_loss",
                "loss_delta": "post_loss_delta",
                "perplexity": "post_perplexity",
                "perplexity_delta": "post_perplexity_delta",
                "evaluation_seconds": "post_evaluation_seconds",
            }
        )
        final_allocation_summary = (
            final_allocation[final_allocation["policy"].isin(finalists)]
            .groupby("policy", as_index=False)
            .agg(
                score_name=("score_name", "first"),
                temperature=("temperature", "first"),
                bounded=("bounded", "first"),
                minimum_width_ratio=("replacement_width_ratio", "min"),
                maximum_width_ratio=("replacement_width_ratio", "max"),
            )
        )
        final_summary = (
            pre.merge(post, on="policy", validate="one_to_one")
            .merge(local_summary, on="policy", validate="one_to_one")
            .merge(final_allocation_summary, on="policy", validate="one_to_one")
        )
        final_summary["kl_recovered_pct"] = 100 * (
            1 - final_summary["post_teacher_kl"] / final_summary["pre_teacher_kl"]
        )
        final_summary["loss_gap_recovered_pct"] = 100 * (
            final_summary["pre_loss_delta"] - final_summary["post_loss_delta"]
        ) / final_summary["pre_loss_delta"]
        final_summary = final_summary.sort_values("post_teacher_kl").reset_index(
            drop=True
        )
        winner = str(final_summary.iloc[0]["policy"])
        runtime_rows.append(
            {
                "stage": "notebook_total",
                "group": None,
                "seconds": perf_counter() - started,
            }
        )
        runtime = pd.DataFrame(runtime_rows)
        artifact = {
            "schema_version": ARTIFACT_SCHEMA,
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
                    "module_group_size": group_size,
                    "disk_io": False,
                },
                "operator_training": asdict(training_config),
                "recovery": {
                    **asdict(recovery_config),
                    "trainable_scope": "replacement_only",
                },
                "reference_artifact": {
                    "path": str(reference_path),
                    "schema_version": reference["schema_version"],
                },
                "partition_batches": partitions,
                "partition_order": list(partitions),
                "screening_calibration_batches": SCREENING_CALIBRATION_BATCHES,
                "probe_width_ratios": list(PROBE_WIDTH_RATIOS),
                "reference_width_ratio": REFERENCE_WIDTH_RATIO,
                "width_sweep_ratios": list(WIDTH_SWEEP_RATIOS),
                "allocation_score_names": list(ALLOCATION_SCORE_NAMES),
                "allocation_temperatures": list(ALLOCATION_TEMPERATURES),
                "boundary_minimum_retentions": list(
                    BOUNDARY_MINIMUM_RETENTIONS
                ),
                "target_mlp_sparsity": target_sparsity,
                "eligible_layers": list(layers),
                "protected_layers": list(protected),
                "representative_blocks": representatives,
                "allocation_specs": specs,
                "boundary_specs": boundary_specs,
                "best_unbounded_policy": best_unbounded,
                "best_bounded_policy": best_bounded,
                "finalist_policies": finalists,
                "winning_policy": winner,
            },
            "results": {
                "dense_reference": json_records(dense_reference),
                "probe_operator_fitting": json_records(probe["fitting"]),
                "probe_operator_training_history": json_records(probe["history"]),
                "probe_model_impact": json_records(probe["impact"]),
                "candidate_scores": json_records(probe["candidates"]),
                "representative_block_membership": json_records(membership),
                "width_study": json_records(width_study),
                "width_study_history": json_records(width_history),
                "allocation": json_records(allocation),
                "allocation_summary": json_records(allocation_summary_df),
                "screening_operator_fitting": json_records(screening_fitting),
                "screening_operator_training_history": json_records(
                    screening_history
                ),
                "screening_model_evaluation": json_records(screening_model),
                "boundary_allocation": json_records(boundary_allocation),
                "boundary_operator_fitting": json_records(boundary_fitting),
                "boundary_operator_training_history": json_records(
                    boundary_history
                ),
                "boundary_model_evaluation": json_records(boundary_model),
                "boundary_screening_comparison": json_records(
                    boundary_screening
                ),
                "finalist_operator_fitting": json_records(full_fitting),
                "finalist_operator_training_history": json_records(full_history),
                "model_evaluation": json_records(evaluation),
                "recovery_history": json_records(recovery_history),
                "final_summary": json_records(final_summary),
                "runtime": json_records(runtime),
            },
        }
        write_artifact(output, artifact)
        run_log.complete(
            {
                "artifact_path": str(output),
                "schema_version": ARTIFACT_SCHEMA,
                "winning_policy": winner,
            }
        )
        print(f"Saved SwiGLU-2 artifact to {output}", flush=True)
        return artifact
    except BaseException as error:
        run_log.fail(error)
        raise


def main() -> None:
    args = parse_args()
    settings = load_workflow_config(args.config, WORKFLOW)
    configured_reference = configure(settings)
    output = args.output or default_artifact_path(WORKFLOW)
    run_workflow(output, args.reference_artifact or configured_reference, settings)


if __name__ == "__main__":
    main()
