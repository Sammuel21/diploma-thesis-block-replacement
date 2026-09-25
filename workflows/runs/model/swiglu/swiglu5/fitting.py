"""SwiGLU-5 fitting; extracted with the established protocol unchanged."""

from __future__ import annotations

import gc
import math
from dataclasses import asdict
from time import perf_counter

import torch

from workflows.runs.model.common import relative_to_root, release_cuda, report_memory

from mlp_replacement.artifacts import atomic_torch_save, sha256_file
from mlp_replacement.capture import collect_modules_io
from mlp_replacement.compression.allocation import (
    allocate_discrete_swiglu_widths,
    build_swiglu_width_curve,
)
from mlp_replacement.compression.reconstruction import load_operator
from mlp_replacement.compression.surgery import temporary_fp32_replacements
from mlp_replacement.config import OperatorConfig
from mlp_replacement.data import make_token_loader
from mlp_replacement.evaluation.mixed_precision import evaluate_teacher_cache_mixed
from mlp_replacement.evaluation.operator import evaluate_operator
from mlp_replacement.operators import (
    GatedMLPReplacement,
    fit_operator_fp32_detailed,
    initialize_gated_mlp_from_teacher,
    initialize_gated_mlp_with_output_reconstruction,
    swiglu_neuron_importance_scores,
)

from ..shared import resolve_source_asset


def operator_config(context):
    values = context.settings["local_fitting"]
    return OperatorConfig(
        kind="swiglu",
        initialization="importance_teacher_subset",
        intermediate_ratio=1.0,
        bias=False,
        epochs=int(values["max_epochs"]),
        learning_rate=float(values["learning_rate"]),
        batch_size=int(values["batch_size"]),
        weight_decay=float(values["weight_decay"]),
        scheduler=str(values["scheduler"]),
        early_stopping_patience=int(values["early_stopping_patience"]),
        early_stopping_min_delta=float(values["early_stopping_min_delta"]),
        seed=int(context.settings["seed"]),
    )


def fit_key(initialization, layer, width, context_kind="dense"):
    return f"{initialization}:{context_kind}:layer-{int(layer):02d}:width-{int(width):05d}"


def operator_state_path(context, initialization, layer, width, context_kind="dense"):
    return (
        context.asset_dir
        / "operators"
        / initialization
        / context_kind
        / f"layer-{int(layer):02d}-width-{int(width):05d}.pt"
    )


def history_rows(fit_key, phase, history):
    return [
        {"fit_key": fit_key, "phase": phase, **asdict(epoch)}
        for epoch in history
    ]


def fit_operator(
    context,
    layer,
    width,
    initialization,
    training_pairs,
    validation_pairs,
    neuron_indices,
    context_kind="dense",
):
    fit_identifier = fit_key(initialization, layer, width, context_kind)
    existing = {
        row["fit_key"]: row for row in context.artifact["results"]["local_fitting"]
    }
    if fit_identifier in existing:
        row = existing[fit_identifier]
        state_path = resolve_source_asset(row["state_path"], context.output)
        if not state_path.is_file() or sha256_file(state_path) != row["state_sha256"]:
            raise ValueError(f"Persisted local-fit state changed: {state_path}")
        return row

    config = operator_config(context)
    teacher = context.blocks[layer].module
    down_bias = initialization in {"output_aware", "composition_aware"}
    module = GatedMLPReplacement(
        int(context.settings["model"]["hidden_size"]),
        int(width),
        down_bias=down_bias,
    ).to(context.device, dtype=torch.float32)
    initial_metrics = None
    reconstruction_metrics = None
    down_history = []
    started = perf_counter()
    if initialization == "legacy_subset":
        initialize_gated_mlp_from_teacher(module, teacher, neuron_indices)
        initial_metrics = evaluate_operator(
            module, validation_pairs, context.device, config.batch_size
        )
        fit = fit_operator_fp32_detailed(
            module, training_pairs, validation_pairs, config, context.device
        )
    elif initialization in {"output_aware", "composition_aware"}:
        reconstruction = initialize_gated_mlp_with_output_reconstruction(
            module,
            teacher,
            (training_pairs, validation_pairs),
            neuron_indices,
            config,
        )
        fit = reconstruction.full_fit
        reconstruction_metrics = {
            "mse": reconstruction.down_only_validation_mse,
            "nmse": reconstruction.down_only_validation_nmse,
            "cosine": reconstruction.down_only_validation_cosine,
            "best_epoch": int(reconstruction.down_only_fit.best_epoch),
            "epochs_completed": len(reconstruction.down_only_fit.history),
            "updates": int(reconstruction.down_only_fit.updates),
            "mean_residual_l2": math.sqrt(
                sum(value * value for value in reconstruction.initial_mean_residual)
            ),
            "mean_residual_mean": sum(reconstruction.initial_mean_residual)
            / len(reconstruction.initial_mean_residual),
            "mean_residual_max_abs": max(
                abs(value) for value in reconstruction.initial_mean_residual
            ),
        }
        down_history = history_rows(
            fit_identifier, "down_only", reconstruction.down_only_fit.history
        )
    else:
        raise ValueError(f"Unsupported SwiGLU-5 initialization: {initialization}")
    final_metrics = evaluate_operator(
        fit.module, validation_pairs, context.device, config.batch_size
    )
    state_path = operator_state_path(
        context, initialization, layer, width, context_kind
    )
    state = {
        name: tensor.detach().cpu() for name, tensor in fit.module.state_dict().items()
    }
    atomic_torch_save(state_path, state)
    row = {
        "fit_key": fit_identifier,
        "initialization": initialization,
        "capture_context": context_kind,
        "layer": int(layer),
        "replacement_width": int(width),
        "has_output_bias": down_bias,
        "calibration_pairs": int(training_pairs.num_tokens),
        "best_epoch": int(fit.best_epoch),
        "epochs_completed": len(fit.history),
        "updates": int(fit.updates),
        "initial_local_mse": float(
            initial_metrics.mse
            if initial_metrics is not None
            else reconstruction_metrics["mse"]
        ),
        "initial_local_nmse": float(
            initial_metrics.relative_mse
            if initial_metrics is not None
            else reconstruction_metrics["nmse"]
        ),
        "initial_local_cosine": float(
            initial_metrics.cosine_similarity
            if initial_metrics is not None
            else reconstruction_metrics["cosine"]
        ),
        "output_reconstruction": reconstruction_metrics,
        "local_mse": float(final_metrics.mse),
        "local_nmse": float(final_metrics.relative_mse),
        "local_cosine": float(final_metrics.cosine_similarity),
        "fit_seconds": perf_counter() - started,
        "state_path": relative_to_root(state_path),
        "state_sha256": sha256_file(state_path),
        "parameter_count": sum(value.numel() for value in state.values()),
        "history": down_history
        + history_rows(fit_identifier, "full", fit.history),
    }
    context.artifact["results"]["local_fitting"].append(row)
    context.persist("local_fitting")
    fit.module.to("cpu")
    del fit, module, state
    release_cuda(torch)
    return row


def load_fit_operator(context, row, device=None):
    return load_operator(
        resolve_source_asset(row["state_path"], context.output),
        int(context.settings["model"]["hidden_size"]),
        int(row["replacement_width"]),
        device=context.device if device is None else device,
        down_bias=bool(row.get("has_output_bias", False)),
    )


def singleton_kl(context, row, selection_cache):
    layer = int(row["layer"])
    module = load_fit_operator(context, row)
    try:
        with temporary_fp32_replacements(
            context.model, context.blocks, {layer: module}
        ):
            metrics = evaluate_teacher_cache_mixed(
                context.model,
                selection_cache,
                float(context.settings["recovery"]["temperature"]),
                context.device,
            )
    finally:
        module.to("cpu")
        release_cuda(torch)
    return metrics


def curve_lookup(context, initialization, layer, width):
    rows = context.artifact["results"]["width_curves"][initialization]
    return next(
        (
            row
            for row in rows
            if int(row["layer"]) == int(layer)
            and int(row["replacement_width"]) == int(width)
        ),
        None,
    )


def capture_dense_pairs(context, layers):
    selected_pairs = int(context.settings["references"]["selected_calibration_pairs"])
    batch_size = int(context.data["batch_size"])
    sequence_length = int(context.data["sequence_length"])
    pair_batches = selected_pairs // (batch_size * sequence_length)
    sequences = context.data["calibration_sequences"][: pair_batches * batch_size]
    loader = make_token_loader(sequences, batch_size)
    paths = [context.blocks[layer].path for layer in layers]
    training = collect_modules_io(
        context.model,
        paths,
        loader,
        pair_batches,
        context.device,
        storage_device="cpu",
        storage_dtype=context.model_dtype,
    )
    validation = collect_modules_io(
        context.model,
        paths,
        context.data["operator_validation"],
        int(context.data["partition_batches"]["operator_validation"]),
        context.device,
        storage_device="cpu",
        storage_dtype=context.model_dtype,
    )
    return training, validation


def selected_neurons(context, layer, pairs, width):
    scores = swiglu_neuron_importance_scores(
        context.blocks[layer].module,
        pairs.inputs,
        int(context.settings["local_fitting"]["batch_size"]),
    )
    return torch.argsort(scores, descending=True)[: int(width)].sort().values


def build_width_curves(
    context,
    selection_cache,
    initializations=("legacy_subset", "output_aware"),
):
    layers = tuple(
        int(value) for value in context.settings["compatibility"]["eligible_layers"]
    )
    initializations = tuple(str(value) for value in initializations)
    if not initializations or any(
        value not in {"legacy_subset", "output_aware"}
        for value in initializations
    ):
        raise ValueError("Width curves require a supported initialization")
    group_size = int(context.settings["local_fitting"]["capture_group_size"])
    original_width = int(context.settings["model"]["intermediate_size"])
    ratios = tuple(float(value) for value in context.settings["allocation"]["width_ratios"])
    widths = tuple(sorted({min(original_width, max(1, round(original_width * ratio))) for ratio in ratios}))
    for offset in range(0, len(layers), group_size):
        group = layers[offset : offset + group_size]
        needed = any(
            curve_lookup(context, initialization, layer, width) is None
            for initialization in initializations
            for layer in group
            for width in widths
        )
        if not needed:
            continue
        training_by_path, validation_by_path = capture_dense_pairs(context, group)
        for layer in group:
            path = context.blocks[layer].path
            training_pairs = training_by_path[path]
            validation_pairs = validation_by_path[path]
            scores = swiglu_neuron_importance_scores(
                context.blocks[layer].module,
                training_pairs.inputs,
                int(context.settings["local_fitting"]["batch_size"]),
            )
            ranking = torch.argsort(scores, descending=True)
            for initialization in initializations:
                for width in widths:
                    if curve_lookup(context, initialization, layer, width) is not None:
                        continue
                    if width == original_width:
                        raw_kl = 0.0
                        state_path = None
                        fit_key = None
                        parameter_count = 3 * int(context.settings["model"]["hidden_size"]) * original_width
                    else:
                        selected = ranking[:width].sort().values
                        fit_row = fit_operator(
                            context,
                            layer,
                            width,
                            initialization,
                            training_pairs,
                            validation_pairs,
                            selected,
                        )
                        singleton = singleton_kl(context, fit_row, selection_cache)
                        raw_kl = float(singleton["teacher_kl"])
                        state_path = fit_row["state_path"]
                        fit_key = fit_row["fit_key"]
                        parameter_count = int(fit_row["parameter_count"])
                    row = {
                        "layer": int(layer),
                        "original_width": original_width,
                        "replacement_width": int(width),
                        "replacement_width_ratio": width / original_width,
                        "raw_teacher_kl": raw_kl,
                        "has_output_bias": initialization == "output_aware" and width < original_width,
                        "retains_dense_module": width == original_width,
                        "replacement_parameters": parameter_count,
                        "fit_key": fit_key,
                        "state_path": state_path,
                    }
                    context.artifact["results"]["width_curves"][initialization].append(row)
                    context.persist("width_curves")
        training_by_path.clear()
        validation_by_path.clear()
        gc.collect()
        release_cuda(torch)
        report_memory(f"After SwiGLU-5 width-curve group {group}")

    for initialization in initializations:
        rows = context.artifact["results"]["width_curves"][initialization]
        for layer in layers:
            evaluations = {
                int(row["replacement_width"]): {
                    "teacher_kl": float(row["raw_teacher_kl"]),
                    "has_output_bias": bool(row["has_output_bias"]),
                }
                for row in rows
                if int(row["layer"]) == layer
            }
            curve = build_swiglu_width_curve(
                layer,
                original_width,
                evaluations,
                int(context.settings["model"]["hidden_size"]),
            )
            monotone = {
                point.replacement_width: point.monotone_teacher_kl for point in curve
            }
            for row in rows:
                if int(row["layer"]) == layer:
                    row["monotone_teacher_kl"] = monotone[int(row["replacement_width"])]
    context.persist("width_curves")


def discrete_allocation(context, initialization, target):
    curves = {}
    for layer in context.settings["compatibility"]["eligible_layers"]:
        rows = [
            row
            for row in context.artifact["results"]["width_curves"][initialization]
            if int(row["layer"]) == int(layer)
        ]
        curves[int(layer)] = build_swiglu_width_curve(
            int(layer),
            int(context.settings["model"]["intermediate_size"]),
            {
                int(row["replacement_width"]): {
                    "teacher_kl": float(row["raw_teacher_kl"]),
                    "has_output_bias": bool(row["has_output_bias"]),
                }
                for row in rows
            },
            int(context.settings["model"]["hidden_size"]),
        )
    return allocate_discrete_swiglu_widths(
        curves,
        float(target),
        int(context.settings["model"]["hidden_size"]),
    )


def ensure_fit_for_width(
    context,
    initialization,
    layer,
    width,
    training_pairs,
    validation_pairs,
    context_kind="dense",
    neuron_ranking=None,
):
    original_width = int(context.settings["model"]["intermediate_size"])
    if int(width) == original_width:
        return None
    existing = next(
        (
            row
            for row in context.artifact["results"]["local_fitting"]
            if row["fit_key"] == fit_key(initialization, layer, width, context_kind)
        ),
        None,
    )
    if existing is not None:
        return existing
    selected = (
        selected_neurons(context, layer, training_pairs, width)
        if neuron_ranking is None
        else neuron_ranking[: int(width)].sort().values
    )
    return fit_operator(
        context,
        layer,
        width,
        initialization,
        training_pairs,
        validation_pairs,
        selected,
        context_kind=context_kind,
    )
