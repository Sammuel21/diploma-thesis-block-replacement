"""Shared implementation for SwiGLU-5 search and confirmation.

The two public runners are intentionally thin.  This module owns the exact
SwiGLU-3 adaptation, candidate construction, token-budget recovery, selection,
and continuation contracts used by both processes.
"""

from __future__ import annotations

import gc
import json
import math
import os
import shutil
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch

from workflows.runs.model._common import (
    load_artifact,
    release_cuda,
    report_memory,
    resolve_path,
)

from mlp_replacement.capture import ActivationPairs, collect_modules_io
from mlp_replacement.compression.allocation import (
    allocate_discrete_swiglu_widths,
    build_swiglu_width_curve,
)
from mlp_replacement.compression.recovery import (
    build_teacher_final_hidden_cache,
    cache_teacher_logits,
    next_optimizer_boundary,
    recover_trainable_by_tokens,
    validate_teacher_final_hidden_cache,
)
from mlp_replacement.compression.surgery import replace_submodule
from mlp_replacement.config import OperatorConfig
from mlp_replacement.data import PackedTokenCache, make_token_loader
from mlp_replacement.evaluation.operator import evaluate_operator
from mlp_replacement.model import discover_mlp_blocks, load_model_and_tokenizer
from mlp_replacement.operators import (
    GatedMLPReplacement,
    fit_operator_fp32_detailed,
    initialize_gated_mlp_from_teacher,
    initialize_gated_mlp_with_output_reconstruction,
    swiglu_neuron_importance_scores,
)
from mlp_replacement.runlog import environment_record

from ._shared import (
    atomic_json,
    atomic_torch_save,
    build_local_data,
    deep_merge,
    evaluate_lm_mixed,
    evaluate_teacher_cache_mixed,
    evaluate_validation_kl_mixed,
    fingerprint,
    load_operator,
    make_model_config,
    recovery_memory_record,
    relative_to_root,
    resolve_source_asset,
    sha256_file,
    source_allocation,
    source_milestone,
    source_operator_rows,
    source_pre_recovery,
    validate_swiglu3_contract,
)


SEARCH_WORKFLOW = "swiglu-5-search"
CONFIRMATION_WORKFLOW = "swiglu-5-confirmation"
SCHEMA_VERSION = 1


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def default_output(workflow, target=None):
    leaf = "search" if workflow == SEARCH_WORKFLOW else "confirmation"
    suffix = f"-target-{target}" if target is not None else ""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path("data/results/workflows/model/swiglu-5") / leaf / (
        f"{workflow}{suffix}-{stamp}.json"
    )


def _asset_directory(output):
    output = Path(output)
    return output.with_suffix("").with_name(output.stem + ".assets")


def _link_or_copy_checkpoint(source, destination):
    """Retain one immutable checkpoint without serializing it twice."""

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise ValueError(f"Existing checkpoint differs: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _content_provenance(value):
    """Remove relocatable path fields while retaining content identity."""

    if isinstance(value, dict):
        return {
            key: _content_provenance(item)
            for key, item in value.items()
            if key not in {"path", "resolved_path"}
        }
    if isinstance(value, list):
        return [_content_provenance(item) for item in value]
    return value


def _checkpoint_rng():
    return {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_rng(checkpoint):
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_states") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])


def _replacement_state(model, paths):
    return {
        path: {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.get_submodule(path).state_dict().items()
        }
        for path in paths
    }


def _load_replacement_state(model, state):
    for path, values in state.items():
        model.get_submodule(path).load_state_dict(values)


@contextmanager
def _temporary_fp32_replacements(model, blocks, replacements):
    originals = {layer: blocks[layer].module for layer in replacements}
    try:
        for layer, replacement in replacements.items():
            replace_submodule(model, blocks[layer].path, replacement)
        yield
    finally:
        for layer, original in originals.items():
            replace_submodule(model, blocks[layer].path, original)


@dataclass
class SwiGLU5Context:
    workflow: str
    output: Path
    asset_dir: Path
    settings: dict
    run_fingerprint: str
    artifact: dict
    source_path: Path | None = None
    source: dict | None = None
    search_path: Path | None = None
    search: dict | None = None
    model: object | None = None
    tokenizer: object | None = None
    device: object | None = None
    model_dtype: object | None = None
    blocks: dict | None = None
    data: dict | None = None

    @property
    def sidecar(self):
        return self.output.with_suffix(".run.json")

    def persist(self, stage=None):
        self.artifact["updated_at_utc"] = utc_now()
        atomic_json(self.output, self.artifact)
        atomic_json(
            self.sidecar,
            {
                "schema_version": 1,
                "workflow": self.workflow,
                "status": self.artifact["status"],
                "current_stage": stage,
                "artifact_path": relative_to_root(self.output),
                "asset_directory": relative_to_root(self.asset_dir),
                "run_fingerprint": self.run_fingerprint,
                "updated_at_utc": self.artifact["updated_at_utc"],
                "error": self.artifact.get("error"),
            },
        )


def _strict_source_assets(source, source_path, calibration_pairs, targets):
    assets = {}
    packed = source["results"]["recovery"]["packed_token_cache"]
    token_path = resolve_source_asset(packed["path"], source_path)
    assets["packed_tokens"] = {
        "path": relative_to_root(token_path),
        "sha256": sha256_file(token_path),
        "fingerprint": packed["fingerprint"],
    }
    manifest_path = token_path.with_suffix(token_path.suffix + ".json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"SwiGLU-3 token manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("fingerprint") != packed["fingerprint"]:
        raise ValueError("SwiGLU-3 packed-token manifest fingerprint differs")
    assets["packed_token_manifest"] = {
        "path": relative_to_root(manifest_path),
        "sha256": sha256_file(manifest_path),
    }
    assets["operators"] = {}
    for target in targets:
        key = str(float(target))
        rows, _ = source_operator_rows(source, calibration_pairs, key)
        assets["operators"][key] = {}
        for layer, row in sorted(rows.items()):
            path = resolve_source_asset(row["state_path"], source_path)
            assets["operators"][key][str(layer)] = {
                "path": relative_to_root(path),
                "sha256": sha256_file(path),
            }
    return assets, token_path


def prepare_search_context(settings, config_path, source_path, output, resume):
    source_path = resolve_path(Path(source_path or settings["references"]["swiglu_3_artifact"]))
    source = load_artifact(source_path, 1, "completed SwiGLU-3 source")
    effective = deep_merge(source["configuration"], settings)
    effective["workflow"] = SEARCH_WORKFLOW
    contract = validate_swiglu3_contract(effective, source)
    targets = effective["compatibility"]["target_mlp_removals"]
    source_assets, _ = _strict_source_assets(
        source,
        source_path,
        int(effective["references"]["selected_calibration_pairs"]),
        targets,
    )
    provenance = {
        "configuration": {
            "path": relative_to_root(resolve_path(Path(config_path))),
            "sha256": sha256_file(resolve_path(Path(config_path))),
        },
        "swiglu_3": {
            "path": relative_to_root(source_path),
            "sha256": sha256_file(source_path),
            "contract": contract,
            "assets": source_assets,
        },
    }
    run_fingerprint = fingerprint(
        {"configuration": effective, "source_sha256": provenance["swiglu_3"]["sha256"]}
    )
    output = resolve_path(Path(output or default_output(SEARCH_WORKFLOW)))
    asset_dir = _asset_directory(output)
    if resume:
        if not output.is_file():
            raise FileNotFoundError(f"Resume artifact does not exist: {output}")
        artifact = json.loads(output.read_text(encoding="utf-8"))
        if artifact.get("run_fingerprint") != run_fingerprint:
            raise ValueError("SwiGLU-5 resume fingerprint differs")
        if _content_provenance(artifact.get("provenance")) != _content_provenance(
            provenance
        ):
            raise ValueError("SwiGLU-5 source/configuration assets changed")
        if artifact.get("status") == "completed":
            raise ValueError("Completed SwiGLU-5 search cannot be resumed")
    else:
        if output.exists() or asset_dir.exists():
            raise FileExistsError(f"SwiGLU-5 output already exists: {output}")
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "workflow": SEARCH_WORKFLOW,
            "experiment_family": "swiglu-5",
            "experiment_class": "homogeneous-swiglu-global-recovery",
            "status": "running",
            "created_at_utc": utc_now(),
            "run_fingerprint": run_fingerprint,
            "environment": environment_record(),
            "configuration": effective,
            "provenance": provenance,
            "results": {
                "published_swiglu_3": {},
                "dense_baseline": None,
                "data": {},
                "teacher_hidden_cache": {},
                "kernel_calibration": {},
                "width_curves": {"legacy_subset": [], "output_aware": []},
                "local_fitting": [],
                "candidates": {},
                "selection": {},
                "runtime": [],
            },
            "error": None,
        }
    context = SwiGLU5Context(
        SEARCH_WORKFLOW,
        output,
        asset_dir,
        effective,
        run_fingerprint,
        artifact,
        source_path=source_path,
        source=source,
    )
    context.persist("prepare")
    return context


def _load_search_resources(context):
    model_config = make_model_config(context.settings["model"])
    context.model, context.tokenizer = load_model_and_tokenizer(model_config)
    context.device = next(context.model.parameters()).device
    context.model_dtype = next(context.model.parameters()).dtype
    context.blocks = {block.index: block for block in discover_mlp_blocks(context.model)}
    expected = set(int(value) for value in context.settings["compatibility"]["eligible_layers"])
    if set(context.blocks) != set(range(int(context.settings["model"]["num_layers"]))) or not expected <= set(context.blocks):
        raise ValueError("Loaded model topology differs from the SwiGLU-3 contract")
    if any(
        int(context.blocks[layer].module.up_proj.out_features)
        != int(context.settings["model"]["intermediate_size"])
        for layer in context.blocks
    ):
        raise ValueError("Loaded model MLP widths differ from the pinned contract")
    input_embeddings = context.model.get_input_embeddings()
    output_embeddings = context.model.get_output_embeddings()
    if (
        bool(context.settings["model"].get("tie_word_embeddings"))
        and input_embeddings.weight.data_ptr() != output_embeddings.weight.data_ptr()
    ):
        raise ValueError("Loaded model does not preserve the required tied vocabulary table")
    context.data = build_local_data(context)
    torch.manual_seed(int(context.settings["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(context.settings["seed"]))
    return model_config


def _operator_config(context):
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


def _packed_source_cache(context):
    record = context.source["results"]["recovery"]["packed_token_cache"]
    path = resolve_source_asset(record["path"], context.source_path)
    expected_bytes = int(record["token_count"]) * 4
    if path.stat().st_size != expected_bytes:
        raise ValueError("SwiGLU-3 packed-token byte extent differs")
    return PackedTokenCache(
        path=path,
        token_count=int(record["token_count"]),
        sequence_length=int(record["sequence_length"]),
        fingerprint=str(record["fingerprint"]),
    )


def _import_swiglu3_evidence(context):
    imported = {}
    for target in context.settings["compatibility"]["target_mlp_removals"]:
        key = str(float(target))
        trajectory = context.source["results"]["recovery"]["trajectories"][key]
        imported[key] = {
            "configuration": {
                "learning_rate": context.source["configuration"]["recovery"]["learning_rate"],
                "optimizer": context.source["configuration"]["recovery"]["optimizer"],
                "scope": "replacement_only",
            },
            "allocation": source_allocation(context.source, key),
            "pre_recovery": source_pre_recovery(context.source, key),
            "validation_history": deepcopy(trajectory["validation_history"]),
            "milestones": [
                source_milestone(context.source, key, 10_000_000),
                source_milestone(context.source, key, 100_000_000),
            ],
            "selected_checkpoint": deepcopy(trajectory.get("selected_checkpoint")),
            "post_recovery": deepcopy(trajectory.get("post_recovery_model_evaluation")),
            "elapsed_seconds": trajectory.get("elapsed_seconds"),
            "memory": deepcopy(trajectory.get("memory")),
        }
    context.artifact["results"]["published_swiglu_3"] = imported
    context.persist("import_swiglu_3")


def _storage_preflight(context):
    """Estimate peak run storage before any expensive local fitting."""

    context.asset_dir.mkdir(parents=True, exist_ok=True)
    probe = context.asset_dir / ".hardlink-probe"
    linked_probe = context.asset_dir / ".hardlink-probe-link"
    hardlinks_supported = False
    try:
        probe.write_bytes(b"swiglu-5")
        os.link(probe, linked_probe)
        hardlinks_supported = True
    except OSError:
        hardlinks_supported = False
    finally:
        linked_probe.unlink(missing_ok=True)
        probe.unlink(missing_ok=True)

    hidden = int(context.settings["model"]["hidden_size"])
    dense_width = int(context.settings["model"]["intermediate_size"])
    layers = len(context.settings["compatibility"]["eligible_layers"])
    widths = {
        min(dense_width, max(1, round(dense_width * float(ratio))))
        for ratio in context.settings["allocation"]["width_ratios"]
        if float(ratio) < 1.0
    }
    curve_state_parameters = layers * sum(
        (3 * hidden * width) + (3 * hidden * width + hidden)
        for width in widths
    )
    targets = [
        float(value)
        for value in context.settings["compatibility"]["target_mlp_removals"]
    ]
    c1_state_parameters = 0
    per_target_recovery_parameters = {}
    original_eligible = layers * 3 * hidden * dense_width
    for target in targets:
        allocation = source_allocation(context.source, str(float(target)))
        source_output_aware = sum(
            3 * hidden * int(row["replacement_width"]) + hidden
            for row in allocation
        )
        c1_state_parameters += source_output_aware
        requested_retained = original_eligible - round(
            original_eligible * target
        )
        per_target_recovery_parameters[str(target)] = max(
            requested_retained,
            source_output_aware,
        )
    composition_state_parameters = sum(per_target_recovery_parameters.values())
    boundary_state_parameters = (
        2 * len(targets) * (3 * hidden * dense_width + hidden)
    )
    local_state_bytes = 4 * (
        curve_state_parameters
        + c1_state_parameters
        + composition_state_parameters
        + boundary_state_parameters
    )
    checkpoint_multiplier = 1 if hardlinks_supported else 2
    checkpoint_bytes = checkpoint_multiplier * 12 * sum(
        5 * parameters
        for parameters in per_target_recovery_parameters.values()
    )
    hidden_cache_bytes = (
        int(context.settings["teacher_hidden_cache"]["target_tokens"])
        * hidden
        * 2
    )
    estimated_peak_bytes = (
        local_state_bytes + checkpoint_bytes + hidden_cache_bytes
    )
    reserve_fraction = float(
        context.settings["storage_preflight"]["reserve_fraction"]
    )
    required_bytes = math.ceil(estimated_peak_bytes * (1.0 + reserve_fraction))
    free_bytes = shutil.disk_usage(context.asset_dir.parent).free
    record = {
        "hardlinks_supported": hardlinks_supported,
        "hidden_cache_bytes": hidden_cache_bytes,
        "local_fit_state_bytes_upper_bound": local_state_bytes,
        "qualifier_checkpoint_bytes_upper_bound": checkpoint_bytes,
        "estimated_peak_bytes": estimated_peak_bytes,
        "reserve_fraction": reserve_fraction,
        "required_free_bytes": required_bytes,
        "observed_free_bytes": free_bytes,
        "passed": free_bytes >= required_bytes,
    }
    context.artifact["results"]["storage_preflight"] = record
    context.persist("storage_preflight")
    if not record["passed"]:
        raise OSError(
            "SwiGLU-5 storage preflight requires "
            f"{required_bytes / 1024**3:.1f} GiB free but found "
            f"{free_bytes / 1024**3:.1f} GiB"
        )
    return record


def _fit_key(initialization, layer, width, context_kind="dense"):
    return f"{initialization}:{context_kind}:layer-{int(layer):02d}:width-{int(width):05d}"


def _operator_state_path(context, initialization, layer, width, context_kind="dense"):
    return (
        context.asset_dir
        / "operators"
        / initialization
        / context_kind
        / f"layer-{int(layer):02d}-width-{int(width):05d}.pt"
    )


def _history_rows(fit_key, phase, history):
    return [
        {"fit_key": fit_key, "phase": phase, **asdict(epoch)}
        for epoch in history
    ]


def _fit_operator(
    context,
    layer,
    width,
    initialization,
    training_pairs,
    validation_pairs,
    neuron_indices,
    context_kind="dense",
):
    fit_key = _fit_key(initialization, layer, width, context_kind)
    existing = {
        row["fit_key"]: row for row in context.artifact["results"]["local_fitting"]
    }
    if fit_key in existing:
        row = existing[fit_key]
        state_path = resolve_source_asset(row["state_path"], context.output)
        if not state_path.is_file() or sha256_file(state_path) != row["state_sha256"]:
            raise ValueError(f"Persisted local-fit state changed: {state_path}")
        return row

    config = _operator_config(context)
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
        down_history = _history_rows(
            fit_key, "down_only", reconstruction.down_only_fit.history
        )
    else:
        raise ValueError(f"Unsupported SwiGLU-5 initialization: {initialization}")
    final_metrics = evaluate_operator(
        fit.module, validation_pairs, context.device, config.batch_size
    )
    state_path = _operator_state_path(
        context, initialization, layer, width, context_kind
    )
    state = {
        name: tensor.detach().cpu() for name, tensor in fit.module.state_dict().items()
    }
    atomic_torch_save(state_path, state)
    row = {
        "fit_key": fit_key,
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
        "history": down_history + _history_rows(fit_key, "full", fit.history),
    }
    context.artifact["results"]["local_fitting"].append(row)
    context.persist("local_fitting")
    fit.module.to("cpu")
    del fit, module, state
    release_cuda(torch)
    return row


def _load_fit_operator(context, row, device=None):
    return load_operator(
        resolve_source_asset(row["state_path"], context.output),
        int(context.settings["model"]["hidden_size"]),
        int(row["replacement_width"]),
        device=context.device if device is None else device,
        down_bias=bool(row.get("has_output_bias", False)),
    )


def _singleton_kl(context, row, selection_cache):
    layer = int(row["layer"])
    module = _load_fit_operator(context, row)
    try:
        with _temporary_fp32_replacements(
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


def _curve_lookup(context, initialization, layer, width):
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


def _capture_dense_pairs(context, layers):
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


def _selected_neurons(context, layer, pairs, width):
    scores = swiglu_neuron_importance_scores(
        context.blocks[layer].module,
        pairs.inputs,
        int(context.settings["local_fitting"]["batch_size"]),
    )
    return torch.argsort(scores, descending=True)[: int(width)].sort().values


def _build_width_curves(context, selection_cache):
    layers = tuple(
        int(value) for value in context.settings["compatibility"]["eligible_layers"]
    )
    group_size = int(context.settings["local_fitting"]["capture_group_size"])
    original_width = int(context.settings["model"]["intermediate_size"])
    ratios = tuple(float(value) for value in context.settings["allocation"]["width_ratios"])
    widths = tuple(sorted({min(original_width, max(1, round(original_width * ratio))) for ratio in ratios}))
    for offset in range(0, len(layers), group_size):
        group = layers[offset : offset + group_size]
        needed = any(
            _curve_lookup(context, initialization, layer, width) is None
            for initialization in ("legacy_subset", "output_aware")
            for layer in group
            for width in widths
        )
        if not needed:
            continue
        training_by_path, validation_by_path = _capture_dense_pairs(context, group)
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
            for initialization in ("legacy_subset", "output_aware"):
                for width in widths:
                    if _curve_lookup(context, initialization, layer, width) is not None:
                        continue
                    if width == original_width:
                        raw_kl = 0.0
                        state_path = None
                        fit_key = None
                        parameter_count = 3 * int(context.settings["model"]["hidden_size"]) * original_width
                    else:
                        selected = ranking[:width].sort().values
                        fit_row = _fit_operator(
                            context,
                            layer,
                            width,
                            initialization,
                            training_pairs,
                            validation_pairs,
                            selected,
                        )
                        singleton = _singleton_kl(context, fit_row, selection_cache)
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

    for initialization in ("legacy_subset", "output_aware"):
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


def _discrete_allocation(context, initialization, target):
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


def _ensure_fit_for_width(
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
            if row["fit_key"] == _fit_key(initialization, layer, width, context_kind)
        ),
        None,
    )
    if existing is not None:
        return existing
    selected = (
        _selected_neurons(context, layer, training_pairs, width)
        if neuron_ranking is None
        else neuron_ranking[: int(width)].sort().values
    )
    return _fit_operator(
        context,
        layer,
        width,
        initialization,
        training_pairs,
        validation_pairs,
        selected,
        context_kind=context_kind,
    )


def _candidate_parameter_summary(context, allocation_rows):
    hidden = int(context.settings["model"]["hidden_size"])
    original_width = int(context.settings["model"]["intermediate_size"])
    original = len(allocation_rows) * 3 * hidden * original_width
    retained = sum(int(row["replacement_parameters"]) for row in allocation_rows)
    return {
        "original_eligible_parameters": original,
        "retained_eligible_parameters": retained,
        "removed_eligible_parameters": original - retained,
        "realized_eligible_mlp_removal": 1.0 - retained / original,
    }


def _legacy_candidate(context, target):
    key = str(float(target))
    state_rows, allocation = source_operator_rows(
        context.source,
        int(context.settings["references"]["selected_calibration_pairs"]),
        key,
    )
    rows = []
    for layer, allocation_row in sorted(allocation.items()):
        path = resolve_source_asset(state_rows[layer]["state_path"], context.source_path)
        rows.append(
            {
                "layer": layer,
                "original_width": int(allocation_row["original_width"]),
                "replacement_width": int(allocation_row["replacement_width"]),
                "replacement_parameters": int(allocation_row["replacement_parameters"]),
                "retains_dense_module": False,
                "has_output_bias": False,
                "state_path": relative_to_root(path),
                "state_sha256": sha256_file(path),
                "source": "swiglu-3-exact-state",
            }
        )
    return {
        "candidate_id": "S5-C0",
        "target": float(target),
        "initialization": "exact_swiglu_3",
        "allocation_method": "exact_swiglu_3_ranked_widths",
        "allocation": rows,
        **_candidate_parameter_summary(context, rows),
    }


def _candidate_modules(context, candidate, device):
    modules = {}
    for row in candidate["allocation"]:
        if row.get("retains_dense_module"):
            continue
        owner_artifact = (
            context.source_path
            if row.get("source") == "swiglu-3-exact-state"
            else context.output
        )
        path = resolve_source_asset(row["state_path"], owner_artifact)
        if sha256_file(path) != row["state_sha256"]:
            raise ValueError(f"Candidate state changed: {path}")
        modules[int(row["layer"])] = load_operator(
            path,
            int(context.settings["model"]["hidden_size"]),
            int(row["replacement_width"]),
            device=device,
            down_bias=bool(row.get("has_output_bias", False)),
        )
    return modules


def _evaluate_candidate(context, candidate, selection_cache, validation_cache):
    modules = _candidate_modules(context, candidate, context.device)
    try:
        with _temporary_fp32_replacements(context.model, context.blocks, modules):
            recovery_kl = evaluate_validation_kl_mixed(
                context.model,
                validation_cache,
                float(context.settings["recovery"]["temperature"]),
                context.device,
            )
            selection = evaluate_teacher_cache_mixed(
                context.model,
                selection_cache,
                float(context.settings["recovery"]["temperature"]),
                context.device,
            )
            wiki = evaluate_lm_mixed(
                context.model,
                context.data["model_validation"],
                context.device,
                int(context.settings["data"]["model_validation_batches"]),
            )
    finally:
        for module in modules.values():
            module.to("cpu")
        modules.clear()
        release_cuda(torch)
    return {
        "recovery_validation_kl": recovery_kl,
        "allocation_selection": selection,
        "wikitext_validation": wiki,
    }


def _build_c1_c3_candidates(context, selection_cache, validation_cache):
    results = context.artifact["results"]["candidates"]
    layers = tuple(int(value) for value in context.settings["compatibility"]["eligible_layers"])
    group_size = int(context.settings["local_fitting"]["capture_group_size"])
    targets = tuple(float(value) for value in context.settings["compatibility"]["target_mlp_removals"])
    allocations = {
        (initialization, target): _discrete_allocation(context, initialization, target)
        for initialization in ("legacy_subset", "output_aware")
        for target in targets
    }
    descriptors = {}
    for target in targets:
        key = str(target)
        descriptors[(key, "S5-C0")] = _legacy_candidate(context, target)
        legacy_widths = {
            int(row["layer"]): int(row["replacement_width"])
            for row in descriptors[(key, "S5-C0")]["allocation"]
        }
        recipes = {
            "S5-C1": (
                "output_aware",
                legacy_widths,
                "swiglu_3_ranked_widths",
                None,
            ),
            "S5-C2": (
                "legacy_subset",
                {int(row["layer"]): int(row["replacement_width"]) for row in allocations[("legacy_subset", target)].rows},
                "discrete_width_curve",
                allocations[("legacy_subset", target)],
            ),
            "S5-C3": (
                "output_aware",
                {int(row["layer"]): int(row["replacement_width"]) for row in allocations[("output_aware", target)].rows},
                "discrete_width_curve",
                allocations[("output_aware", target)],
            ),
        }
        for candidate_id, (
            initialization,
            widths,
            allocation_method,
            allocation_result,
        ) in recipes.items():
            descriptor = {
                "candidate_id": candidate_id,
                "target": target,
                "initialization": initialization,
                "allocation_method": allocation_method,
                "widths": widths,
                "allocation": [],
            }
            if allocation_result is not None:
                descriptor["allocation_solver"] = {
                    key: value
                    for key, value in asdict(allocation_result).items()
                    if key != "rows"
                }
            descriptors[(key, candidate_id)] = descriptor

    for offset in range(0, len(layers), group_size):
        group = layers[offset : offset + group_size]
        training_by_path, validation_by_path = _capture_dense_pairs(context, group)
        rankings = {}
        for layer in group:
            path = context.blocks[layer].path
            scores = swiglu_neuron_importance_scores(
                context.blocks[layer].module,
                training_by_path[path].inputs,
                int(context.settings["local_fitting"]["batch_size"]),
            )
            rankings[layer] = torch.argsort(scores, descending=True)
        for (key, candidate_id), descriptor in descriptors.items():
            if candidate_id == "S5-C0":
                continue
            for layer in group:
                width = descriptor["widths"][layer]
                path = context.blocks[layer].path
                fit_row = _ensure_fit_for_width(
                    context,
                    descriptor["initialization"],
                    layer,
                    width,
                    training_by_path[path],
                    validation_by_path[path],
                    neuron_ranking=rankings[layer],
                )
                if fit_row is None:
                    parameter_count = 3 * int(context.settings["model"]["hidden_size"]) * int(context.settings["model"]["intermediate_size"])
                    descriptor["allocation"].append(
                        {
                            "layer": layer,
                            "original_width": int(context.settings["model"]["intermediate_size"]),
                            "replacement_width": width,
                            "replacement_parameters": parameter_count,
                            "retains_dense_module": True,
                            "has_output_bias": False,
                            "state_path": None,
                            "state_sha256": None,
                            "is_boundary_width": bool(
                                descriptor.get("allocation_solver", {}).get(
                                    "boundary_layer"
                                )
                                == layer
                            ),
                        }
                    )
                else:
                    descriptor["allocation"].append(
                        {
                            "layer": layer,
                            "original_width": int(context.settings["model"]["intermediate_size"]),
                            "replacement_width": width,
                            "replacement_parameters": int(fit_row["parameter_count"]),
                            "retains_dense_module": False,
                            "has_output_bias": bool(fit_row["has_output_bias"]),
                            "state_path": fit_row["state_path"],
                            "state_sha256": fit_row["state_sha256"],
                            "fit_key": fit_row["fit_key"],
                            "is_boundary_width": bool(
                                descriptor.get("allocation_solver", {}).get(
                                    "boundary_layer"
                                )
                                == layer
                            ),
                        }
                    )
        training_by_path.clear()
        validation_by_path.clear()
        rankings.clear()
        gc.collect()
        release_cuda(torch)

    for (key, candidate_id), descriptor in descriptors.items():
        if candidate_id != "S5-C0":
            descriptor.pop("widths", None)
            descriptor["allocation"].sort(key=lambda row: int(row["layer"]))
            descriptor.update(_candidate_parameter_summary(context, descriptor["allocation"]))
        target_results = results.setdefault(key, {})
        if candidate_id not in target_results:
            descriptor["pre_recovery"] = _evaluate_candidate(
                context, descriptor, selection_cache, validation_cache
            )
            descriptor["recovery"] = {
                "status": "pending",
                "tokens_seen": 0,
                "optimizer_updates": 0,
                "validation_history": [],
                "full_evaluations": [],
            }
            target_results[candidate_id] = descriptor
            context.persist("candidate_assembly")


def _dense_targets(module, inputs, batch_size, device):
    chunks = []
    module.eval()
    with torch.no_grad():
        parameter = next(module.parameters())
        for start in range(0, inputs.shape[0], batch_size):
            batch = inputs[start : start + batch_size].to(
                device=device, dtype=parameter.dtype
            )
            chunks.append(module(batch).detach().to("cpu"))
    return torch.cat(chunks, dim=0)


def _build_composition_candidates(context, selection_cache, validation_cache):
    results = context.artifact["results"]["candidates"]
    layers = tuple(int(value) for value in context.settings["compatibility"]["eligible_layers"])
    group_size = int(context.settings["local_fitting"]["capture_group_size"])
    selected_pairs = int(context.settings["references"]["selected_calibration_pairs"])
    batch_size = int(context.data["batch_size"])
    sequence_length = int(context.data["sequence_length"])
    pair_batches = selected_pairs // (batch_size * sequence_length)
    training_loader = make_token_loader(
        context.data["calibration_sequences"][: pair_batches * batch_size], batch_size
    )
    for key, target_results in results.items():
        if "S5-C4" in target_results:
            continue
        parent = min(
            (target_results[candidate_id] for candidate_id in ("S5-C0", "S5-C1", "S5-C2", "S5-C3")),
            key=lambda row: (
                float(row["pre_recovery"]["recovery_validation_kl"]),
                row["candidate_id"],
            ),
        )
        parent_modules = _candidate_modules(context, parent, context.device)
        composition_rows = []
        try:
            with _temporary_fp32_replacements(context.model, context.blocks, parent_modules):
                for offset in range(0, len(layers), group_size):
                    group = layers[offset : offset + group_size]
                    paths = [context.blocks[layer].path for layer in group]
                    captured_training = collect_modules_io(
                        context.model,
                        paths,
                        training_loader,
                        pair_batches,
                        context.device,
                        storage_device="cpu",
                        storage_dtype=context.model_dtype,
                    )
                    captured_validation = collect_modules_io(
                        context.model,
                        paths,
                        context.data["operator_validation"],
                        int(context.data["partition_batches"]["operator_validation"]),
                        context.device,
                        storage_device="cpu",
                        storage_dtype=context.model_dtype,
                    )
                    for layer in group:
                        parent_row = next(row for row in parent["allocation"] if int(row["layer"]) == layer)
                        if parent_row.get("retains_dense_module"):
                            composition_rows.append(deepcopy(parent_row))
                            continue
                        width = int(parent_row["replacement_width"])
                        path = context.blocks[layer].path
                        dense = context.blocks[layer].module
                        train_inputs = captured_training[path].inputs
                        valid_inputs = captured_validation[path].inputs
                        training_pairs = ActivationPairs(
                            train_inputs,
                            _dense_targets(
                                dense,
                                train_inputs,
                                int(context.settings["local_fitting"]["batch_size"]),
                                context.device,
                            ),
                        )
                        validation_pairs = ActivationPairs(
                            valid_inputs,
                            _dense_targets(
                                dense,
                                valid_inputs,
                                int(context.settings["local_fitting"]["batch_size"]),
                                context.device,
                            ),
                        )
                        selected = _selected_neurons(context, layer, training_pairs, width)
                        fit_row = _fit_operator(
                            context,
                            layer,
                            width,
                            "composition_aware",
                            training_pairs,
                            validation_pairs,
                            selected,
                            context_kind=f"student-{key}-{parent['candidate_id']}",
                        )
                        composition_rows.append(
                            {
                                "layer": layer,
                                "original_width": int(context.settings["model"]["intermediate_size"]),
                                "replacement_width": width,
                                "replacement_parameters": int(fit_row["parameter_count"]),
                                "retains_dense_module": False,
                                "has_output_bias": True,
                                "state_path": fit_row["state_path"],
                                "state_sha256": fit_row["state_sha256"],
                                "fit_key": fit_row["fit_key"],
                                "is_boundary_width": bool(
                                    parent_row.get("is_boundary_width", False)
                                ),
                            }
                        )
                    captured_training.clear()
                    captured_validation.clear()
                    gc.collect()
                    release_cuda(torch)
        finally:
            for module in parent_modules.values():
                module.to("cpu")
            parent_modules.clear()
            release_cuda(torch)
        candidate = {
            "candidate_id": "S5-C4",
            "target": float(key),
            "initialization": "composition_aware",
            "allocation_method": "best_pre_recovery_parent_widths",
            "parent_candidate_id": parent["candidate_id"],
            "capture_context_frozen": True,
            "recapture_rounds": 1,
            "allocation": sorted(composition_rows, key=lambda row: int(row["layer"])),
        }
        candidate.update(_candidate_parameter_summary(context, candidate["allocation"]))
        candidate["pre_recovery"] = _evaluate_candidate(
            context, candidate, selection_cache, validation_cache
        )
        candidate["recovery"] = {
            "status": "pending",
            "tokens_seen": 0,
            "optimizer_updates": 0,
            "validation_history": [],
            "full_evaluations": [],
        }
        target_results["S5-C4"] = candidate
        context.persist("composition_aware")


def _load_candidate_student(context, candidate, model_config):
    student, tokenizer = load_model_and_tokenizer(model_config)
    blocks = {block.index: block for block in discover_mlp_blocks(student)}
    modules = _candidate_modules(context, candidate, next(student.parameters()).device)
    for layer, module in modules.items():
        replace_submodule(student, blocks[layer].path, module)
    target_paths = [blocks[layer].path for layer in sorted(modules)]
    train_modules = [student.get_submodule(path) for path in target_paths]
    if not target_paths:
        raise ValueError("A compressed candidate must contain replacement modules")
    for module in train_modules:
        for parameter in module.parameters():
            if parameter.dtype != torch.float32:
                raise ValueError("SwiGLU-5 replacements must retain FP32 master weights")
    del tokenizer, modules
    return student, target_paths, train_modules


def _clone_teacher_head(model, device):
    head = deepcopy(model.get_output_embeddings()).to(device)
    for parameter in head.parameters():
        parameter.requires_grad = False
    head.eval()
    return head


def _profile_trial(
    context,
    model_config,
    candidate,
    cache,
    teacher_head,
    geometry,
    compile_model=False,
):
    if torch.device(context.device).type != "cuda":
        raise RuntimeError("SwiGLU-5 kernel calibration requires CUDA")
    torch.manual_seed(int(context.settings["seed"]))
    torch.cuda.manual_seed_all(int(context.settings["seed"]))
    torch.cuda.reset_peak_memory_stats(context.device)
    student, _, train_modules = _load_candidate_student(
        context, candidate, model_config
    )
    execution_model = student
    compile_error = None
    if compile_model:
        try:
            execution_model = torch.compile(student, mode="reduce-overhead")
        except BaseException as error:
            compile_error = f"{type(error).__name__}: {error}"
    events = []
    microbatch_sequences = int(geometry["microbatch_sequences"])
    microbatch_tokens = microbatch_sequences * int(
        context.settings["compatibility"]["sequence_length"]
    )
    effective = int(context.settings["recovery"]["effective_batch_tokens"])
    profile_updates = int(context.settings["recovery"]["profile_optimizer_updates"])
    if profile_updates < 1:
        raise ValueError("Recovery profiling requires at least one optimizer update")
    profile_tokens = effective * profile_updates
    if microbatch_tokens * int(geometry["gradient_accumulation_steps"]) != effective:
        raise ValueError("Profile geometry does not preserve the effective batch")
    started = perf_counter()
    try:
        result = recover_trainable_by_tokens(
            student=execution_model,
            teacher=None,
            parameter_groups=[
                {
                    "name": "replacements",
                    "parameters": [
                        parameter
                        for module in train_modules
                        for parameter in module.parameters()
                    ],
                    "learning_rate": float(context.settings["recovery"]["learning_rate"]),
                    "weight_decay": float(context.settings["recovery"]["weight_decay"]),
                }
            ],
            train_modules=train_modules,
            batch_at=lambda offset, count: _packed_source_cache(context).batch(
                offset, count, microbatch_sequences
            ),
            target_tokens=profile_tokens,
            schedule_tokens=profile_tokens,
            microbatch_tokens=microbatch_tokens,
            accumulation_steps=int(geometry["gradient_accumulation_steps"]),
            temperature=float(context.settings["recovery"]["temperature"]),
            ce_weight=float(context.settings["recovery"]["ce_weight"]),
            scheduler="constant",
            warmup_fraction=0.0,
            final_lr_ratio=1.0,
            device=context.device,
            autocast_dtype=torch.bfloat16,
            checkpoint_schedule=((profile_tokens, (profile_tokens,)),),
            on_checkpoint=lambda event, _optimizer, _first: events.append(event),
            teacher_hidden_at=cache.batch,
            teacher_head=teacher_head,
            optimizer_backend=str(context.settings["recovery"]["optimizer_backend"]),
        )
        wall_seconds = perf_counter() - started
        peak = torch.cuda.max_memory_allocated(context.device) / 1024**3
        event = events[-1]
        return {
            **geometry,
            "execution": "compiled" if compile_model else "eager",
            "compile_error": compile_error,
            "tokens": result.tokens_seen,
            "training_seconds": result.elapsed_seconds,
            "wall_seconds": wall_seconds,
            "tokens_per_second": result.tokens_seen / result.elapsed_seconds,
            "mean_train_loss": event.mean_train_loss,
            "mean_train_kl": event.mean_train_kl,
            "peak_vram_gib": peak,
            "fits_memory": peak
            <= float(context.settings["recovery"]["maximum_peak_vram_gib"]),
        }
    finally:
        del execution_model, student, train_modules
        gc.collect()
        release_cuda(torch)


def _calibrate_recovery(context, model_config, cache, teacher_head):
    existing = context.artifact["results"]["kernel_calibration"]
    if existing.get("selected_geometry"):
        return existing
    if (
        context.settings["recovery"]["optimizer"] != "AdamW"
        or context.settings["recovery"]["optimizer_backend"] != "fused"
    ):
        raise ValueError("SwiGLU-5 search requires fused CUDA AdamW")
    pre_profile_rng = _checkpoint_rng()
    control = context.artifact["results"]["candidates"]["0.2"]["S5-C0"]
    profiles = []
    for geometry in context.settings["recovery"]["microbatch_candidates"]:
        try:
            profiles.append(
                _profile_trial(
                    context,
                    model_config,
                    control,
                    cache,
                    teacher_head,
                    geometry,
                )
            )
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            torch.cuda.empty_cache()
            profiles.append(
                {
                    **geometry,
                    "execution": "eager",
                    "fits_memory": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    viable = [row for row in profiles if row.get("fits_memory")]
    if not viable:
        raise RuntimeError("No profiled recovery geometry fits the VRAM envelope")
    selected = max(viable, key=lambda row: int(row["microbatch_sequences"]))
    compiled = None
    compile_selected = False
    try:
        compiled = _profile_trial(
            context,
            model_config,
            control,
            cache,
            teacher_head,
            {
                "microbatch_sequences": selected["microbatch_sequences"],
                "gradient_accumulation_steps": selected["gradient_accumulation_steps"],
            },
            compile_model=True,
        )
        speedup = compiled["tokens_per_second"] / selected["tokens_per_second"] - 1.0
        loss_difference = abs(
            float(compiled["mean_train_loss"]) - float(selected["mean_train_loss"])
        )
        compile_selected = (
            compiled.get("compile_error") is None
            and compiled["fits_memory"]
            and speedup >= float(context.settings["recovery"]["compile_minimum_speedup"])
            and loss_difference <= float(context.settings["recovery"]["compile_maximum_loss_difference"])
        )
        compiled["speedup_over_eager"] = speedup
        compiled["loss_difference_from_eager"] = loss_difference
    except BaseException as error:
        compiled = {"error": f"{type(error).__name__}: {error}"}
    record = {
        "disposable_state_restored_between_trials": True,
        "rng_restored_before_experiment": True,
        "profile_optimizer_updates": int(
            context.settings["recovery"]["profile_optimizer_updates"]
        ),
        "profiles": profiles,
        "selected_geometry": {
            "microbatch_sequences": int(selected["microbatch_sequences"]),
            "gradient_accumulation_steps": int(selected["gradient_accumulation_steps"]),
            "effective_batch_tokens": int(context.settings["recovery"]["effective_batch_tokens"]),
        },
        "compile_profile": compiled,
        "execution_mode": "compiled_reduce_overhead" if compile_selected else "eager",
        "observed_tokens_per_second": (
            float(compiled["tokens_per_second"])
            if compile_selected
            else float(selected["tokens_per_second"])
        ),
    }
    context.artifact["results"]["kernel_calibration"] = record
    context.persist("kernel_calibration")
    _restore_rng(pre_profile_rng)
    return record


def _requested_actual_map(requested_values, effective_batch, limit):
    mapped = {}
    for requested in requested_values:
        requested = int(requested)
        actual = next_optimizer_boundary(requested, effective_batch, limit)
        mapped.setdefault(actual, []).append(requested)
    return {actual: tuple(values) for actual, values in sorted(mapped.items())}


def _find_full_evaluation(trajectory, requested_tokens):
    for row in trajectory["full_evaluations"]:
        if int(requested_tokens) in [int(value) for value in row["requested_tokens"]]:
            return row
    raise ValueError(f"Candidate has no full evaluation at {requested_tokens:,} tokens")


def _recover_candidate(
    context,
    candidate,
    model_config,
    token_cache,
    hidden_cache,
    teacher_head,
    validation_cache,
    selection_cache,
    requested_target,
    kernel,
):
    recovery = candidate["recovery"]
    effective = int(context.settings["recovery"]["effective_batch_tokens"])
    actual_target = next_optimizer_boundary(
        int(requested_target), effective, int(hidden_cache.token_count)
    )
    if int(recovery.get("tokens_seen", 0)) >= actual_target:
        return
    if torch.device(context.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)
    student, target_paths, train_modules = _load_candidate_student(
        context, candidate, model_config
    )
    execution_model = student
    if kernel["execution_mode"] == "compiled_reduce_overhead":
        execution_model = torch.compile(student, mode="reduce-overhead")
    target_key = str(float(candidate["target"]))
    candidate_id = candidate["candidate_id"]
    recovery_dir = context.asset_dir / "recovery" / target_key / candidate_id
    current_path = recovery_dir / "current.pt"
    optimizer_state = None
    start_tokens = int(recovery.get("tokens_seen", 0))
    start_updates = int(recovery.get("optimizer_updates", 0))
    elapsed = float(recovery.get("training_seconds", 0.0))
    if current_path.is_file():
        checkpoint = torch.load(current_path, map_location="cpu", weights_only=False)
        if checkpoint.get("run_fingerprint") != context.run_fingerprint:
            raise ValueError("Candidate recovery checkpoint fingerprint differs")
        if checkpoint.get("candidate_fingerprint") != fingerprint(
            {key: value for key, value in candidate.items() if key != "recovery"}
        ):
            raise ValueError("Candidate recovery checkpoint recipe differs")
        if checkpoint.get("packed_token_fingerprint") != token_cache.fingerprint:
            raise ValueError("Candidate recovery token stream differs")
        _load_replacement_state(student, checkpoint["replacement_state"])
        optimizer_state = checkpoint["optimizer_state"]
        recovery.clear()
        recovery.update(checkpoint["recovery"])
        start_tokens = int(checkpoint["tokens_seen"])
        start_updates = int(checkpoint["optimizer_updates"])
        elapsed = float(checkpoint["training_seconds"])
        _restore_rng(checkpoint)
        retained_boundaries = {
            int(row["actual_tokens"])
            for row in recovery.get("durable_checkpoints", [])
        }
        for requested in context.settings["recovery"]["durable_checkpoint_tokens"]:
            boundary = next_optimizer_boundary(
                int(requested), effective, actual_target
            )
            durable_path = recovery_dir / f"endpoint-{boundary:012d}.pt"
            if (
                boundary <= start_tokens
                and boundary not in retained_boundaries
                and durable_path.is_file()
            ):
                recovery.setdefault("durable_checkpoints", []).append(
                    {
                        "requested_tokens": [int(requested)],
                        "actual_tokens": boundary,
                        "path": relative_to_root(durable_path),
                        "sha256": sha256_file(durable_path),
                        "retained": True,
                        "recovered_after_interruption": True,
                    }
                )
    elif start_tokens:
        raise FileNotFoundError(f"Candidate resume checkpoint is missing: {current_path}")
    else:
        recovery["validation_history"].append(
            {
                "requested_tokens": [0],
                "actual_tokens": 0,
                "optimizer_updates": 0,
                "recovery_validation_kl": candidate["pre_recovery"]["recovery_validation_kl"],
            }
        )
        recovery["full_evaluations"].append(
            {
                "requested_tokens": [0],
                "actual_tokens": 0,
                "optimizer_updates": 0,
                "recovery_validation_kl": candidate["pre_recovery"]["recovery_validation_kl"],
                "allocation_selection": deepcopy(candidate["pre_recovery"]["allocation_selection"]),
                "wikitext_validation": deepcopy(candidate["pre_recovery"]["wikitext_validation"]),
            }
        )
    recovery["status"] = "running"
    context.persist("recovery")

    validation_requested = list(
        range(
            int(context.settings["recovery"]["validation_interval_tokens"]),
            int(requested_target) + 1,
            int(context.settings["recovery"]["validation_interval_tokens"]),
        )
    )
    if int(requested_target) not in validation_requested:
        validation_requested.append(int(requested_target))
    full_requested = [
        int(value)
        for value in context.settings["recovery"]["full_evaluation_tokens"]
        if start_tokens < next_optimizer_boundary(int(value), effective, actual_target) <= actual_target
    ]
    durable_requested = [
        int(value)
        for value in context.settings["recovery"]["durable_checkpoint_tokens"]
        if start_tokens < next_optimizer_boundary(int(value), effective, actual_target) <= actual_target
    ]
    requested_union = sorted(
        {
            value
            for value in validation_requested + full_requested + durable_requested
            if start_tokens < next_optimizer_boundary(value, effective, actual_target) <= actual_target
        }
    )
    schedule_map = _requested_actual_map(requested_union, effective, actual_target)
    validation_map = _requested_actual_map(validation_requested, effective, actual_target)
    full_map = _requested_actual_map(full_requested, effective, actual_target)
    durable_map = _requested_actual_map(durable_requested, effective, actual_target)
    geometry = kernel["selected_geometry"]
    microbatch_sequences = int(geometry["microbatch_sequences"])
    microbatch_tokens = microbatch_sequences * int(token_cache.sequence_length)

    def persist_checkpoint(event, optimizer, first_step):
        evaluation_seconds = 0.0
        checkpoint_seconds = 0.0
        validation_kl = None
        if event.tokens_seen in validation_map:
            evaluation_started = perf_counter()
            validation_kl = evaluate_validation_kl_mixed(
                student,
                validation_cache,
                float(context.settings["recovery"]["temperature"]),
                context.device,
            )
            evaluation_seconds += perf_counter() - evaluation_started
            recovery["validation_history"].append(
                {
                    "requested_tokens": list(validation_map[event.tokens_seen]),
                    "actual_tokens": event.tokens_seen,
                    "optimizer_updates": event.optimizer_updates,
                    "recovery_validation_kl": validation_kl,
                    "mean_train_kl_since_resume": event.mean_train_kl,
                }
            )
        if event.tokens_seen in full_map:
            evaluation_started = perf_counter()
            if validation_kl is None:
                validation_kl = evaluate_validation_kl_mixed(
                    student,
                    validation_cache,
                    float(context.settings["recovery"]["temperature"]),
                    context.device,
                )
            selection = evaluate_teacher_cache_mixed(
                student,
                selection_cache,
                float(context.settings["recovery"]["temperature"]),
                context.device,
            )
            wiki = evaluate_lm_mixed(
                student,
                context.data["model_validation"],
                context.device,
                int(context.settings["data"]["model_validation_batches"]),
            )
            evaluation_seconds += perf_counter() - evaluation_started
            recovery["full_evaluations"].append(
                {
                    "requested_tokens": list(full_map[event.tokens_seen]),
                    "actual_tokens": event.tokens_seen,
                    "optimizer_updates": event.optimizer_updates,
                    "recovery_validation_kl": validation_kl,
                    "allocation_selection": selection,
                    "wikitext_validation": wiki,
                    "memory": recovery_memory_record(context.device),
                }
            )
        recovery["tokens_seen"] = event.tokens_seen
        recovery["optimizer_updates"] = event.optimizer_updates
        recovery["training_seconds"] = event.elapsed_seconds
        recovery["evaluation_seconds"] = float(recovery.get("evaluation_seconds", 0.0)) + evaluation_seconds
        recovery["first_step"] = recovery.get("first_step") or first_step
        checkpoint_started = perf_counter()
        state = _replacement_state(student, target_paths)
        checkpoint = {
            "schema_version": 1,
            "workflow": SEARCH_WORKFLOW,
            "run_fingerprint": context.run_fingerprint,
            "candidate_fingerprint": fingerprint(
                {key: value for key, value in candidate.items() if key != "recovery"}
            ),
            "target": candidate["target"],
            "candidate_id": candidate_id,
            "tokens_seen": event.tokens_seen,
            "optimizer_updates": event.optimizer_updates,
            "training_seconds": event.elapsed_seconds,
            "packed_token_fingerprint": token_cache.fingerprint,
            "replacement_state": state,
            "optimizer_state": optimizer.state_dict(),
            "recovery": deepcopy(recovery),
            **_checkpoint_rng(),
        }
        atomic_torch_save(current_path, checkpoint)
        durable_path = None
        durable_sha256 = None
        if event.tokens_seen in durable_map:
            durable_path = recovery_dir / f"endpoint-{event.tokens_seen:012d}.pt"
            _link_or_copy_checkpoint(current_path, durable_path)
            durable_sha256 = sha256_file(durable_path)
        checkpoint_seconds += perf_counter() - checkpoint_started
        recovery["checkpoint_seconds"] = float(recovery.get("checkpoint_seconds", 0.0)) + checkpoint_seconds
        if durable_path is not None:
            recovery.setdefault("durable_checkpoints", []).append(
                {
                    "requested_tokens": list(durable_map[event.tokens_seen]),
                    "actual_tokens": event.tokens_seen,
                    "path": relative_to_root(durable_path),
                    "sha256": durable_sha256,
                    "retained": True,
                }
            )
        context.persist("recovery")

    try:
        result = recover_trainable_by_tokens(
            student=execution_model,
            teacher=None,
            parameter_groups=[
                {
                    "name": "replacements",
                    "parameters": [
                        parameter
                        for module in train_modules
                        for parameter in module.parameters()
                    ],
                    "learning_rate": float(context.settings["recovery"]["learning_rate"]),
                    "weight_decay": float(context.settings["recovery"]["weight_decay"]),
                }
            ],
            train_modules=train_modules,
            batch_at=lambda offset, count: token_cache.batch(
                offset, count, microbatch_sequences
            ),
            target_tokens=actual_target,
            schedule_tokens=next_optimizer_boundary(
                int(context.settings["recovery"]["finalist_tokens"]),
                effective,
                hidden_cache.token_count,
            ),
            microbatch_tokens=microbatch_tokens,
            accumulation_steps=int(geometry["gradient_accumulation_steps"]),
            temperature=float(context.settings["recovery"]["temperature"]),
            ce_weight=float(context.settings["recovery"]["ce_weight"]),
            scheduler=str(context.settings["recovery"]["scheduler"]),
            warmup_fraction=float(context.settings["recovery"]["warmup_fraction"]),
            final_lr_ratio=float(context.settings["recovery"]["final_lr_ratio"]),
            device=context.device,
            autocast_dtype=torch.bfloat16,
            start_tokens=start_tokens,
            start_updates=start_updates,
            elapsed_seconds=elapsed,
            optimizer_state=optimizer_state,
            checkpoint_schedule=tuple(schedule_map.items()),
            on_checkpoint=persist_checkpoint,
            teacher_hidden_at=hidden_cache.batch,
            teacher_head=teacher_head,
            optimizer_backend=str(context.settings["recovery"]["optimizer_backend"]),
        )
        if result.tokens_seen != actual_target:
            raise RuntimeError("Candidate recovery ended before its token target")
        recovery["tokens_seen"] = result.tokens_seen
        recovery["optimizer_updates"] = result.optimizer_updates
        recovery["training_seconds"] = result.elapsed_seconds
        recovery["first_step"] = recovery.get("first_step") or result.first_step
        recovery["status"] = (
            "completed_5m"
            if int(requested_target) == int(context.settings["recovery"]["finalist_tokens"])
            else "completed_2m"
        )
        recovery["memory"] = recovery_memory_record(context.device)
        context.persist("recovery")
    finally:
        del execution_model, student, train_modules
        gc.collect()
        release_cuda(torch)


def _prune_search_candidate_checkpoints(
    context,
    candidate,
    keep_requested=(),
):
    """Prune non-selected recovery state while preserving its manifest."""

    keep_requested = {int(value) for value in keep_requested}
    recovery = candidate["recovery"]
    changed = False
    asset_root = context.asset_dir.resolve()
    for row in recovery.get("durable_checkpoints", []):
        if not bool(row.get("retained", True)):
            continue
        requested = {int(value) for value in row["requested_tokens"]}
        if requested & keep_requested:
            row["retained"] = True
            continue
        path = (
            context.asset_dir
            / "recovery"
            / str(float(candidate["target"]))
            / candidate["candidate_id"]
            / f"endpoint-{int(row['actual_tokens']):012d}.pt"
        ).resolve()
        try:
            path.relative_to(asset_root)
        except ValueError as error:
            raise ValueError(
                f"Refusing to prune checkpoint outside the run assets: {path}"
            ) from error
        if path.is_file():
            path.unlink()
        row["retained"] = False
        row["pruned_after_selection"] = True
        changed = True
    current_path = (
        context.asset_dir
        / "recovery"
        / str(float(candidate["target"]))
        / candidate["candidate_id"]
        / "current.pt"
    )
    if current_path.is_file() and (
        not keep_requested
        or int(recovery.get("tokens_seen", 0))
        >= next_optimizer_boundary(
            max(keep_requested),
            int(context.settings["recovery"]["effective_batch_tokens"]),
        )
    ):
        current_path.resolve().relative_to(asset_root)
        current_path.unlink()
        changed = True
    if changed:
        context.persist("checkpoint_pruning")


def _prune_search_local_fit_states(context):
    """Retain fit evidence in JSON after winner checkpoints supersede states."""

    asset_root = context.asset_dir.resolve()
    removed = 0
    removed_bytes = 0
    for row in context.artifact["results"]["local_fitting"]:
        if not bool(row.get("state_retained", True)):
            continue
        path = _operator_state_path(
            context,
            row["initialization"],
            row["layer"],
            row["replacement_width"],
            row["capture_context"],
        ).resolve()
        try:
            path.relative_to(asset_root)
        except ValueError as error:
            raise ValueError(
                f"Refusing to prune local fit outside the run assets: {path}"
            ) from error
        if path.is_file():
            removed_bytes += path.stat().st_size
            path.unlink()
            removed += 1
        row["state_retained"] = False
        row["pruned_after_search"] = True
    for rows in context.artifact["results"]["width_curves"].values():
        for row in rows:
            if row.get("state_path") is not None:
                row["state_retained"] = False
    return {
        "state_files_removed": removed,
        "bytes_removed": removed_bytes,
        "retained_evidence": "fit histories, metrics, hashes, and winner endpoints",
    }


def _runtime_guard(context, observed_tokens_per_second):
    completed_seconds = sum(
        float(row["seconds"]) for row in context.artifact["results"]["runtime"]
    )
    candidate_tokens = 38_000_000
    raw_projection = completed_seconds + candidate_tokens / float(observed_tokens_per_second)
    projected = raw_projection * (
        1.0 + float(context.settings["runtime_guard"]["reserve_fraction"])
    )
    record = {
        "completed_preparation_seconds": completed_seconds,
        "candidate_token_positions": candidate_tokens,
        "observed_tokens_per_second": float(observed_tokens_per_second),
        "raw_projected_seconds": raw_projection,
        "reserve_fraction": float(context.settings["runtime_guard"]["reserve_fraction"]),
        "projected_seconds": projected,
        "maximum_projected_seconds": int(context.settings["runtime_guard"]["maximum_projected_seconds"]),
        "passed": projected
        <= int(context.settings["runtime_guard"]["maximum_projected_seconds"]),
    }
    context.artifact["results"]["runtime_guard"] = record
    context.persist("runtime_guard")
    return record


def _select_finalists(context, target_key):
    candidates = context.artifact["results"]["candidates"][target_key]
    qualifier_requested = int(context.settings["recovery"]["qualifier_tokens"])
    challengers = []
    for candidate_id in ("S5-C1", "S5-C2", "S5-C3", "S5-C4"):
        trajectory = candidates[candidate_id]["recovery"]
        endpoint = next(
            row
            for row in trajectory["validation_history"]
            if qualifier_requested in [int(value) for value in row["requested_tokens"]]
        )
        challengers.append((float(endpoint["recovery_validation_kl"]), candidate_id))
    count = int(context.settings["selection"]["challenger_finalists"])
    selected = [candidate_id for _, candidate_id in sorted(challengers)[:count]]
    return ["S5-C0", *selected]


def _select_winner(context, target_key, finalists):
    candidates = context.artifact["results"]["candidates"][target_key]
    requested = int(context.settings["recovery"]["finalist_tokens"])
    control_eval = _find_full_evaluation(candidates["S5-C0"]["recovery"], requested)
    control_ppl = float(control_eval["wikitext_validation"]["perplexity"])
    eligible = []
    decisions = []
    for candidate_id in finalists:
        evaluation = _find_full_evaluation(candidates[candidate_id]["recovery"], requested)
        ppl = float(evaluation["wikitext_validation"]["perplexity"])
        passes = candidate_id == "S5-C0" or ppl <= control_ppl
        row = {
            "candidate_id": candidate_id,
            "recovery_validation_kl": float(evaluation["recovery_validation_kl"]),
            "wikitext_validation_perplexity": ppl,
            "control_perplexity": control_ppl,
            "passes_ppl_guardrail": passes,
        }
        decisions.append(row)
        if passes:
            eligible.append(row)
    pool = eligible
    minimum_kl = min(row["recovery_validation_kl"] for row in pool)
    tolerance = float(context.settings["selection"]["kl_tie_tolerance"])
    tied = [row for row in pool if row["recovery_validation_kl"] <= minimum_kl + tolerance]
    winner = min(
        tied,
        key=lambda row: (
            row["wikitext_validation_perplexity"], row["candidate_id"]
        ),
    )
    winner_candidate = candidates[winner["candidate_id"]]
    endpoint = next(
        row
        for row in winner_candidate["recovery"].get("durable_checkpoints", [])
        if requested in [int(value) for value in row["requested_tokens"]]
    )
    source_history = context.artifact["results"]["published_swiglu_3"][target_key]["validation_history"]
    source_nearest = min(
        source_history,
        key=lambda row: abs(
            int(row["tokens_seen"])
            - int(
                next_optimizer_boundary(
                    requested,
                    int(context.settings["recovery"]["effective_batch_tokens"]),
                )
            )
        ),
    )
    dense = context.artifact["results"]["dense_baseline"]
    control = next(row for row in decisions if row["candidate_id"] == "S5-C0")
    winner_eval = _find_full_evaluation(winner_candidate["recovery"], requested)
    return {
        "finalists": finalists,
        "guardrail_decisions": decisions,
        "winner_candidate_id": winner["candidate_id"],
        "selection_rule": (
            "challenger PPL no worse than S5-C0; lowest fixed T=1 KL; "
            "KL differences <=1e-6 tie-break by PPL then candidate ID; "
            "fall back to S5-C0 when no challenger passes"
        ),
        "winner_endpoint": endpoint,
        "gaps": {
            "to_dense": {
                "recovery_validation_kl": float(winner_eval["recovery_validation_kl"]),
                "wikitext_perplexity": float(winner_eval["wikitext_validation"]["perplexity"])
                - float(dense["wikitext_validation"]["perplexity"]),
            },
            "to_s5_c0": {
                "recovery_validation_kl": float(winner_eval["recovery_validation_kl"])
                - float(control["recovery_validation_kl"]),
                "wikitext_perplexity": float(winner_eval["wikitext_validation"]["perplexity"])
                - float(control["wikitext_validation_perplexity"]),
            },
            "to_swiglu_3_nearest_5m_validation": {
                "source_tokens": int(source_nearest["tokens_seen"]),
                "recovery_validation_kl": float(winner_eval["recovery_validation_kl"])
                - float(source_nearest["recovery_validation_kl"]),
                "ppl_comparison_available": False,
            },
        },
    }


def _record_stage_runtime(context, stage, started):
    context.artifact["results"]["runtime"].append(
        {"stage": stage, "seconds": perf_counter() - started}
    )
    context.persist(stage)


def run_search(context):
    """Execute the complete compute-bounded SwiGLU-5 tournament."""

    started = perf_counter()
    _import_swiglu3_evidence(context)
    _storage_preflight(context)
    model_config = _load_search_resources(context)
    context.artifact["results"]["data"] = {
        "partition_order": list(context.data["partition_batches"]),
        "partition_batches": deepcopy(context.data["partition_batches"]),
        "selected_calibration_pairs": int(
            context.settings["references"]["selected_calibration_pairs"]
        ),
        "sequence_length": int(context.data["sequence_length"]),
    }
    _record_stage_runtime(context, "load_model_and_data", started)

    started = perf_counter()
    cache_dtype = context.source["configuration"]["recovery"].get(
        "validation_cache_dtype", "float16"
    )
    validation_cache = cache_teacher_logits(
        context.model,
        context.data["recovery_validation"],
        int(context.data["partition_batches"]["recovery_validation"]),
        context.device,
        cache_dtype,
    )
    selection_cache = cache_teacher_logits(
        context.model,
        context.data["allocation_selection"],
        int(context.data["partition_batches"]["allocation_selection"]),
        context.device,
        cache_dtype,
    )
    context.artifact["results"]["dense_baseline"] = {
        "recovery_validation_kl": evaluate_validation_kl_mixed(
            context.model,
            validation_cache,
            float(context.settings["recovery"]["temperature"]),
            context.device,
        ),
        "allocation_selection": evaluate_teacher_cache_mixed(
            context.model,
            selection_cache,
            float(context.settings["recovery"]["temperature"]),
            context.device,
        ),
        "wikitext_validation": evaluate_lm_mixed(
            context.model,
            context.data["model_validation"],
            context.device,
            int(context.settings["data"]["model_validation_batches"]),
        ),
    }
    _record_stage_runtime(context, "dense_and_fixed_evaluation_caches", started)

    started = perf_counter()
    _build_width_curves(context, selection_cache)
    _record_stage_runtime(context, "width_curves", started)

    started = perf_counter()
    _build_c1_c3_candidates(context, selection_cache, validation_cache)
    _build_composition_candidates(context, selection_cache, validation_cache)
    _record_stage_runtime(context, "candidate_assembly", started)

    started = perf_counter()
    token_cache = _packed_source_cache(context)
    hidden_settings = context.settings["teacher_hidden_cache"]
    hidden_cache = build_teacher_final_hidden_cache(
        context.model,
        token_cache,
        context.asset_dir / "teacher-final-hidden",
        int(hidden_settings["target_tokens"]),
        int(hidden_settings["capture_batch_sequences"]),
        context.device,
        {
            "model_id": context.settings["model"]["model_id"],
            "revision": context.settings["model"]["revision"],
        },
        cache_dtype=torch.bfloat16,
        shard_tokens=int(hidden_settings["shard_tokens"]),
        minimum_free_gib=float(hidden_settings["minimum_free_gib"]),
    )
    equivalence = validate_teacher_final_hidden_cache(
        context.model,
        hidden_cache,
        token_cache,
        context.device,
        hidden_settings["validation_sample_offsets"],
        temperature=float(context.settings["recovery"]["temperature"]),
        maximum_mean_kl=float(hidden_settings["maximum_mean_kl"]),
    )
    context.artifact["results"]["teacher_hidden_cache"] = {
        "root": relative_to_root(hidden_cache.root),
        "manifest": deepcopy(hidden_cache.manifest),
        "equivalence_validation": equivalence,
    }
    teacher_head = _clone_teacher_head(context.model, context.device)
    _record_stage_runtime(context, "teacher_final_hidden_cache", started)

    del context.model, context.blocks
    context.model = None
    context.blocks = None
    gc.collect()
    release_cuda(torch)

    started = perf_counter()
    kernel = _calibrate_recovery(context, model_config, hidden_cache, teacher_head)
    _record_stage_runtime(context, "kernel_calibration", started)
    guard = _runtime_guard(context, kernel["observed_tokens_per_second"])
    if not guard["passed"]:
        context.artifact["status"] = "budget_guard_rejected"
        context.artifact["completed_at_utc"] = utc_now()
        context.persist(None)
        return

    started = perf_counter()
    qualifier = int(context.settings["recovery"]["qualifier_tokens"])
    for target_key in ("0.2", "0.5"):
        for candidate_id in ("S5-C0", "S5-C1", "S5-C2", "S5-C3", "S5-C4"):
            _recover_candidate(
                context,
                context.artifact["results"]["candidates"][target_key][candidate_id],
                model_config,
                token_cache,
                hidden_cache,
                teacher_head,
                validation_cache,
                selection_cache,
                qualifier,
                kernel,
            )
    context.artifact["results"]["selection"]["qualifier"] = {}
    finalists_by_target = {}
    for target_key in ("0.2", "0.5"):
        finalists = _select_finalists(context, target_key)
        finalists_by_target[target_key] = finalists
        context.artifact["results"]["selection"]["qualifier"][target_key] = {
            "requested_tokens": qualifier,
            "finalists": finalists,
        }
    context.persist("qualifier_selection")
    for target_key, candidates in context.artifact["results"]["candidates"].items():
        finalists = set(finalists_by_target[target_key])
        for candidate_id, candidate in candidates.items():
            if candidate_id not in finalists:
                _prune_search_candidate_checkpoints(context, candidate)

    finalist_target = int(context.settings["recovery"]["finalist_tokens"])
    for target_key, finalists in finalists_by_target.items():
        for candidate_id in finalists:
            _recover_candidate(
                context,
                context.artifact["results"]["candidates"][target_key][candidate_id],
                model_config,
                token_cache,
                hidden_cache,
                teacher_head,
                validation_cache,
                selection_cache,
                finalist_target,
                kernel,
            )
            _prune_search_candidate_checkpoints(
                context,
                context.artifact["results"]["candidates"][target_key][candidate_id],
                keep_requested=(finalist_target,),
            )
    for target_key, finalists in finalists_by_target.items():
        context.artifact["results"]["selection"][target_key] = _select_winner(
            context, target_key, finalists
        )
        winner = context.artifact["results"]["selection"][target_key][
            "winner_candidate_id"
        ]
        for candidate_id in finalists:
            _prune_search_candidate_checkpoints(
                context,
                context.artifact["results"]["candidates"][target_key][candidate_id],
                keep_requested=(finalist_target,) if candidate_id == winner else (),
            )
    _record_stage_runtime(context, "candidate_recovery_and_selection", started)
    context.artifact["results"]["recovery_work"] = {
        "qualifier_candidate_token_positions": 20_000_000,
        "continuation_candidate_token_positions": 18_000_000,
        "total_candidate_token_positions": 38_000_000,
        "effective_batch_tokens": int(context.settings["recovery"]["effective_batch_tokens"]),
        "checkpoint_storage_policy": (
            "hard-link identical current/milestone states when supported; "
            "prune rejected states; retain one 5M winner endpoint per target"
        ),
    }
    context.artifact["results"]["local_fit_state_pruning"] = (
        _prune_search_local_fit_states(context)
    )
    context.artifact["status"] = "completed"
    context.artifact["completed_at_utc"] = utc_now()
    context.persist(None)


def prepare_confirmation_context(
    settings,
    config_path,
    search_path,
    target,
    output,
    resume,
):
    target = float(target)
    if target not in [float(value) for value in settings["compatibility"]["allowed_targets"]]:
        raise ValueError("Confirmation --target must be exactly 0.2 or 0.5")
    target_key = str(target)
    search_path = resolve_path(Path(search_path))
    search = load_artifact(search_path, 1, "completed SwiGLU-5 search")
    if search.get("workflow") != SEARCH_WORKFLOW or search.get("status") != "completed":
        raise ValueError("Confirmation requires a completed SwiGLU-5 search artifact")
    selection = search["results"]["selection"].get(target_key)
    if not selection or not selection.get("winner_candidate_id"):
        raise ValueError(f"Search artifact has no selected winner for target {target_key}")
    source_record = search["provenance"]["swiglu_3"]
    source_path = resolve_path(Path(source_record["path"]))
    if not source_path.is_file():
        source_path = resolve_source_asset(source_record["path"], search_path)
    if sha256_file(source_path) != source_record["sha256"]:
        raise ValueError("Original SwiGLU-3 source artifact changed")
    source = load_artifact(source_path, 1, "original completed SwiGLU-3 source")
    validate_swiglu3_contract(search["configuration"], source)
    source_assets = source_record["assets"]
    for asset_name in ("packed_tokens", "packed_token_manifest"):
        asset_record = source_assets[asset_name]
        asset_path = resolve_source_asset(asset_record["path"], source_path)
        if sha256_file(asset_path) != asset_record["sha256"]:
            raise ValueError(
                f"Original SwiGLU-3 {asset_name.replace('_', ' ')} changed"
            )
    endpoint_record = selection["winner_endpoint"]
    endpoint_path = resolve_source_asset(endpoint_record["path"], search_path)
    if sha256_file(endpoint_path) != endpoint_record["sha256"]:
        raise ValueError("Selected SwiGLU-5 endpoint checkpoint changed")
    provenance = {
        "configuration": {
            "path": relative_to_root(resolve_path(Path(config_path))),
            "sha256": sha256_file(resolve_path(Path(config_path))),
        },
        "search": {
            "path": relative_to_root(search_path),
            "sha256": sha256_file(search_path),
            "run_fingerprint": search["run_fingerprint"],
        },
        "swiglu_3": deepcopy(source_record),
        "selected_endpoint": {
            **deepcopy(endpoint_record),
            "resolved_path": relative_to_root(endpoint_path),
        },
    }
    effective = deep_merge(search["configuration"], settings)
    effective["workflow"] = CONFIRMATION_WORKFLOW
    effective["target"] = target
    effective["selected_candidate_id"] = selection["winner_candidate_id"]
    run_fingerprint = fingerprint(
        {
            "configuration": effective,
            "search_sha256": provenance["search"]["sha256"],
            "endpoint_sha256": endpoint_record["sha256"],
        }
    )
    output = resolve_path(Path(output or default_output(CONFIRMATION_WORKFLOW, target_key)))
    asset_dir = _asset_directory(output)
    if resume:
        if not output.is_file():
            raise FileNotFoundError(f"Resume artifact does not exist: {output}")
        artifact = json.loads(output.read_text(encoding="utf-8"))
        if artifact.get("run_fingerprint") != run_fingerprint:
            raise ValueError("Confirmation resume fingerprint differs")
        if artifact.get("status") == "completed":
            raise ValueError("Completed confirmation cannot be resumed")
    else:
        if output.exists() or asset_dir.exists():
            raise FileExistsError(f"Confirmation output already exists: {output}")
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "workflow": CONFIRMATION_WORKFLOW,
            "experiment_family": "swiglu-5",
            "experiment_class": "homogeneous-swiglu-global-recovery",
            "status": "running",
            "created_at_utc": utc_now(),
            "run_fingerprint": run_fingerprint,
            "environment": environment_record(),
            "configuration": effective,
            "provenance": provenance,
            "results": {
                "target": target,
                "selected_candidate_id": selection["winner_candidate_id"],
                "kernel_calibration": {},
                "trajectory": {
                    "status": "pending",
                    "tokens_seen": 0,
                    "optimizer_updates": 0,
                    "validation_history": [],
                    "full_evaluations": [],
                    "scientific_checkpoints": [],
                },
                "paired_swiglu_3_comparison": [],
                "runtime": [],
            },
            "error": None,
        }
    context = SwiGLU5Context(
        CONFIRMATION_WORKFLOW,
        output,
        asset_dir,
        effective,
        run_fingerprint,
        artifact,
        source_path=source_path,
        source=source,
        search_path=search_path,
        search=search,
    )
    context.persist("prepare")
    return context


def _confirmation_candidate(context):
    key = str(float(context.settings["target"]))
    candidate_id = context.settings["selected_candidate_id"]
    return context.search["results"]["candidates"][key][candidate_id]


def _blank_candidate_student(context, model_config, candidate):
    student, tokenizer = load_model_and_tokenizer(model_config)
    blocks = {block.index: block for block in discover_mlp_blocks(student)}
    hidden = int(context.settings["model"]["hidden_size"])
    target_paths = []
    train_modules = []
    for row in candidate["allocation"]:
        if row.get("retains_dense_module"):
            continue
        layer = int(row["layer"])
        module = GatedMLPReplacement(
            hidden,
            int(row["replacement_width"]),
            down_bias=bool(row.get("has_output_bias", False)),
        ).to(next(student.parameters()).device, dtype=torch.float32)
        replace_submodule(student, blocks[layer].path, module)
        target_paths.append(blocks[layer].path)
        train_modules.append(module)
    del tokenizer
    return student, target_paths, train_modules


def _load_selected_search_checkpoint(context):
    record = context.artifact["provenance"]
    path = resolve_path(Path(record["selected_endpoint"]["resolved_path"]))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("run_fingerprint") != context.search["run_fingerprint"]:
        raise ValueError("Selected endpoint belongs to a different search run")
    candidate = _confirmation_candidate(context)
    expected = fingerprint(
        {key: value for key, value in candidate.items() if key != "recovery"}
    )
    if checkpoint.get("candidate_fingerprint") != expected:
        raise ValueError("Selected endpoint candidate recipe differs")
    return checkpoint


def _online_profile_trial(context, model_config, candidate, teacher, geometry):
    checkpoint = _load_selected_search_checkpoint(context)
    student, _, train_modules = _blank_candidate_student(
        context, model_config, candidate
    )
    _load_replacement_state(student, checkpoint["replacement_state"])
    _restore_rng(checkpoint)
    token_cache = _packed_source_cache(context)
    microbatch_sequences = int(geometry["microbatch_sequences"])
    microbatch_tokens = microbatch_sequences * token_cache.sequence_length
    effective = int(context.settings["recovery"]["effective_batch_tokens"])
    if microbatch_tokens * int(geometry["gradient_accumulation_steps"]) != effective:
        raise ValueError("Confirmation profile geometry changes effective batch size")
    start_tokens = int(checkpoint["tokens_seen"])
    target_tokens = start_tokens + effective
    events = []
    torch.cuda.reset_peak_memory_stats(context.device)
    try:
        result = recover_trainable_by_tokens(
            student=student,
            teacher=teacher,
            parameter_groups=[
                {
                    "name": "replacements",
                    "parameters": [
                        parameter for module in train_modules for parameter in module.parameters()
                    ],
                    "learning_rate": float(context.settings["recovery"]["learning_rate"]),
                    "weight_decay": float(context.settings["recovery"]["weight_decay"]),
                }
            ],
            train_modules=train_modules,
            batch_at=lambda offset, count: token_cache.batch(
                offset, count, microbatch_sequences
            ),
            target_tokens=target_tokens,
            schedule_tokens=int(context.settings["recovery"]["target_tokens"]),
            microbatch_tokens=microbatch_tokens,
            accumulation_steps=int(geometry["gradient_accumulation_steps"]),
            temperature=float(context.settings["recovery"]["temperature"]),
            ce_weight=float(context.settings["recovery"]["ce_weight"]),
            scheduler=str(context.settings["recovery"]["scheduler"]),
            warmup_fraction=float(context.settings["recovery"]["warmup_fraction"]),
            final_lr_ratio=float(context.settings["recovery"]["final_lr_ratio"]),
            device=context.device,
            autocast_dtype=torch.bfloat16,
            start_tokens=start_tokens,
            start_updates=int(checkpoint["optimizer_updates"]),
            elapsed_seconds=0.0,
            optimizer_state=checkpoint["optimizer_state"],
            checkpoint_schedule=((target_tokens, (target_tokens,)),),
            on_checkpoint=lambda event, _optimizer, _first: events.append(event),
            optimizer_backend=str(context.settings["recovery"]["optimizer_backend"]),
        )
        peak = torch.cuda.max_memory_allocated(context.device) / 1024**3
        event = events[-1]
        return {
            **geometry,
            "effective_batch_tokens": effective,
            "tokens_per_second": effective / result.elapsed_seconds,
            "training_seconds": result.elapsed_seconds,
            "mean_train_kl": event.mean_train_kl,
            "peak_vram_gib": peak,
            "fits_memory": peak <= float(context.settings["recovery"]["maximum_peak_vram_gib"]),
        }
    finally:
        del checkpoint, student, train_modules
        gc.collect()
        release_cuda(torch)


def _calibrate_online_confirmation(context, model_config, candidate, teacher):
    existing = context.artifact["results"]["kernel_calibration"]
    if existing.get("selected_geometry"):
        return existing
    if (
        context.settings["recovery"]["optimizer"] != "AdamW"
        or context.settings["recovery"]["optimizer_backend"] != "fused"
    ):
        raise ValueError("SwiGLU-5 confirmation requires fused CUDA AdamW")
    profiles = []
    for geometry in context.settings["recovery"]["microbatch_candidates"]:
        try:
            profiles.append(
                _online_profile_trial(
                    context, model_config, candidate, teacher, geometry
                )
            )
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            torch.cuda.empty_cache()
            profiles.append(
                {**geometry, "fits_memory": False, "error": f"{type(error).__name__}: {error}"}
            )
    viable = [row for row in profiles if row.get("fits_memory")]
    if not viable:
        raise RuntimeError("No online-teacher geometry fits the VRAM envelope")
    selected = max(viable, key=lambda row: int(row["microbatch_sequences"]))
    record = {
        "profiles": profiles,
        "disposable_state_restored_between_trials": True,
        "teacher_mode": "online_dense",
        "execution_mode": "eager",
        "selected_geometry": {
            "microbatch_sequences": int(selected["microbatch_sequences"]),
            "gradient_accumulation_steps": int(selected["gradient_accumulation_steps"]),
            "effective_batch_tokens": int(selected["effective_batch_tokens"]),
        },
    }
    context.artifact["results"]["kernel_calibration"] = record
    context.persist("kernel_calibration")
    return record


def _source_comparison_row(context, milestone, current, trajectory):
    source_current = milestone["current"]
    requested = int(milestone["requested_tokens"][0])
    source_elapsed = None
    source_throughput = None
    if requested == 100_000_000:
        source_trajectory = context.source["results"]["recovery"]["trajectories"][
            str(float(context.settings["target"]))
        ]
        source_elapsed = float(source_trajectory["elapsed_seconds"])
        source_throughput = int(milestone["actual_tokens"]) / source_elapsed
    current_training_seconds = float(
        current.get("cumulative_training_seconds", trajectory["training_seconds"])
    )
    current_elapsed = float(
        current.get(
            "cumulative_elapsed_seconds",
            current_training_seconds
            + float(trajectory.get("evaluation_seconds", 0.0))
            + float(trajectory.get("checkpoint_seconds", 0.0)),
        )
    )
    current_throughput = int(current["actual_tokens"]) / current_elapsed
    source_memory = (
        context.source["results"]["recovery"]["trajectories"][
            str(float(context.settings["target"]))
        ].get("memory", {})
        if requested == 100_000_000
        else {}
    )
    current_memory = current.get("memory", trajectory.get("memory", {}))
    return {
        "requested_tokens": requested,
        "swiglu_3_actual_tokens": int(milestone["actual_tokens"]),
        "swiglu_5_actual_tokens": int(current["actual_tokens"]),
        "same_actual_boundary": int(milestone["actual_tokens"]) == int(current["actual_tokens"]),
        "swiglu_3": {
            "recovery_validation_kl": float(source_current["recovery_validation_kl"]),
            "wikitext_validation_loss": float(source_current["wikitext_validation"]["loss"]),
            "wikitext_validation_perplexity": float(source_current["wikitext_validation"]["perplexity"]),
            "optimizer_updates": int(milestone["optimizer_updates"]),
            "elapsed_seconds": source_elapsed,
            "tokens_per_second": source_throughput,
            "memory": deepcopy(source_memory),
        },
        "swiglu_5": {
            "recovery_validation_kl": float(current["recovery_validation_kl"]),
            "wikitext_validation_loss": float(current["wikitext_validation"]["loss"]),
            "wikitext_validation_perplexity": float(current["wikitext_validation"]["perplexity"]),
            "optimizer_updates": int(current["optimizer_updates"]),
            "training_seconds": current_training_seconds,
            "elapsed_seconds": current_elapsed,
            "tokens_per_second": current_throughput,
            "memory": deepcopy(current_memory),
        },
        "difference_swiglu_5_minus_swiglu_3": {
            "recovery_validation_kl": float(current["recovery_validation_kl"])
            - float(source_current["recovery_validation_kl"]),
            "wikitext_validation_loss": float(current["wikitext_validation"]["loss"])
            - float(source_current["wikitext_validation"]["loss"]),
            "wikitext_validation_perplexity": float(current["wikitext_validation"]["perplexity"])
            - float(source_current["wikitext_validation"]["perplexity"]),
            "optimizer_updates": int(current["optimizer_updates"])
            - int(milestone["optimizer_updates"]),
            "elapsed_seconds": (
                current_elapsed - source_elapsed if source_elapsed is not None else None
            ),
            "tokens_per_second": (
                current_throughput - source_throughput
                if source_throughput is not None
                else None
            ),
            "peak_ram_gib": (
                float(current_memory["peak_ram_gib"]) - float(source_memory["peak_ram_gib"])
                if "peak_ram_gib" in current_memory and "peak_ram_gib" in source_memory
                else None
            ),
            "peak_vram_gib": (
                float(current_memory["peak_vram_gib"]) - float(source_memory["peak_vram_gib"])
                if "peak_vram_gib" in current_memory and "peak_vram_gib" in source_memory
                else None
            ),
        },
        "runtime_comparison_note": (
            None
            if source_elapsed is not None
            else (
                "SwiGLU-3 did not persist observed 10M elapsed-time or "
                "milestone-memory records"
            )
        ),
        "intentional_method_differences": [
            "selected SwiGLU-5 initialization/allocation recipe versus exact SwiGLU-3 recipe",
            "constant 3e-5 versus constant 1e-5 learning rate",
            "fused CUDA AdamW versus recorded SwiGLU-3 AdamW implementation",
            "profiled microbatch geometry versus SwiGLU-3 2-sequence x 8 accumulation geometry",
        ],
    }


def run_confirmation(context):
    """Continue one selected 5M search endpoint to the 100M boundary."""

    stage_started = perf_counter()
    model_config = make_model_config(context.settings["model"])
    teacher, tokenizer = load_model_and_tokenizer(model_config)
    context.model = teacher
    context.tokenizer = tokenizer
    context.device = next(teacher.parameters()).device
    context.model_dtype = next(teacher.parameters()).dtype
    teacher_blocks = {block.index: block for block in discover_mlp_blocks(teacher)}
    if set(teacher_blocks) != set(range(int(context.settings["model"]["num_layers"]))):
        raise ValueError("Confirmation teacher topology differs from the pinned model")
    if teacher.get_input_embeddings().weight.data_ptr() != teacher.get_output_embeddings().weight.data_ptr():
        raise ValueError("Confirmation teacher must retain tied embeddings")
    del teacher_blocks
    context.data = build_local_data(context)
    token_cache = _packed_source_cache(context)
    endpoint = _load_selected_search_checkpoint(context)
    if endpoint.get("packed_token_fingerprint") != token_cache.fingerprint:
        raise ValueError("Search endpoint and SwiGLU-3 token stream differ")
    start_requested = int(context.settings["recovery"]["start_tokens"])
    effective = int(context.settings["recovery"]["effective_batch_tokens"])
    expected_start = next_optimizer_boundary(
        start_requested, effective, token_cache.token_count
    )
    if int(endpoint["tokens_seen"]) != expected_start:
        raise ValueError("Selected search endpoint is not the exact 5M boundary")
    del endpoint
    cache_dtype = context.source["configuration"]["recovery"].get(
        "validation_cache_dtype", "float16"
    )
    validation_cache = cache_teacher_logits(
        teacher,
        context.data["recovery_validation"],
        int(context.data["partition_batches"]["recovery_validation"]),
        context.device,
        cache_dtype,
    )
    selection_cache = cache_teacher_logits(
        teacher,
        context.data["allocation_selection"],
        int(context.data["partition_batches"]["allocation_selection"]),
        context.device,
        cache_dtype,
    )
    context.artifact["results"]["runtime"].append(
        {"stage": "load_and_validate", "seconds": perf_counter() - stage_started}
    )
    context.persist("load_and_validate")

    candidate = _confirmation_candidate(context)
    stage_started = perf_counter()
    kernel = _calibrate_online_confirmation(
        context, model_config, candidate, teacher
    )
    context.artifact["results"]["runtime"].append(
        {"stage": "kernel_calibration", "seconds": perf_counter() - stage_started}
    )
    context.persist("kernel_calibration")

    student, target_paths, train_modules = _blank_candidate_student(
        context, model_config, candidate
    )
    endpoint = _load_selected_search_checkpoint(context)
    trajectory = context.artifact["results"]["trajectory"]
    current_path = context.asset_dir / "recovery" / "current.pt"
    best_path = context.asset_dir / "recovery" / "best.pt"
    final_path = context.asset_dir / "recovery" / "final.pt"
    optimizer_state = None
    if current_path.is_file():
        current = torch.load(current_path, map_location="cpu", weights_only=False)
        if current.get("run_fingerprint") != context.run_fingerprint:
            raise ValueError("Confirmation current checkpoint fingerprint differs")
        if current.get("packed_token_fingerprint") != token_cache.fingerprint:
            raise ValueError("Confirmation current checkpoint token stream differs")
        _load_replacement_state(student, current["replacement_state"])
        optimizer_state = current["optimizer_state"]
        trajectory.clear()
        trajectory.update(current["trajectory"])
        _restore_rng(current)
        start_tokens = int(current["tokens_seen"])
        start_updates = int(current["optimizer_updates"])
        training_seconds = float(current["training_seconds"])
        retained_boundaries = {
            int(row["actual_tokens"])
            for row in trajectory.get("scientific_checkpoints", [])
        }
        for requested in context.settings["recovery"]["full_evaluation_tokens"]:
            boundary = next_optimizer_boundary(
                int(requested),
                int(context.settings["recovery"]["effective_batch_tokens"]),
                int(context.settings["recovery"]["target_tokens"]),
            )
            scientific_path = (
                context.asset_dir
                / "recovery"
                / f"milestone-{boundary:012d}.pt"
            )
            if (
                boundary <= start_tokens
                and boundary not in retained_boundaries
                and scientific_path.is_file()
            ):
                trajectory.setdefault("scientific_checkpoints", []).append(
                    {
                        "requested_tokens": [int(requested)],
                        "actual_tokens": boundary,
                        "path": relative_to_root(scientific_path),
                        "sha256": sha256_file(scientific_path),
                        "recovered_after_interruption": True,
                    }
                )
            if boundary <= start_tokens:
                matching = [
                    row
                    for row in trajectory.get("full_evaluations", [])
                    if int(requested)
                    in [int(value) for value in row["requested_tokens"]]
                ]
                if matching and "cumulative_elapsed_seconds" not in matching[0]:
                    matching[0]["cumulative_training_seconds"] = training_seconds
                    matching[0]["cumulative_evaluation_seconds"] = float(
                        trajectory.get("evaluation_seconds", 0.0)
                    )
                    matching[0]["cumulative_checkpoint_seconds"] = float(
                        trajectory.get("checkpoint_seconds", 0.0)
                    )
                    matching[0]["cumulative_elapsed_seconds"] = (
                        matching[0]["cumulative_training_seconds"]
                        + matching[0]["cumulative_evaluation_seconds"]
                        + matching[0]["cumulative_checkpoint_seconds"]
                    )
    elif int(trajectory.get("tokens_seen", 0)):
        raise FileNotFoundError(f"Confirmation current checkpoint is missing: {current_path}")
    else:
        _load_replacement_state(student, endpoint["replacement_state"])
        optimizer_state = endpoint["optimizer_state"]
        _restore_rng(endpoint)
        start_tokens = int(endpoint["tokens_seen"])
        start_updates = int(endpoint["optimizer_updates"])
        training_seconds = float(endpoint["training_seconds"])
        search_full = _find_full_evaluation(
            candidate["recovery"], start_requested
        )
        trajectory.update(
            {
                "status": "running",
                "tokens_seen": start_tokens,
                "optimizer_updates": start_updates,
                "training_seconds": training_seconds,
                "evaluation_seconds": float(
                    candidate["recovery"].get("evaluation_seconds", 0.0)
                ),
                "checkpoint_seconds": float(
                    candidate["recovery"].get("checkpoint_seconds", 0.0)
                ),
                "validation_history": [
                    {
                        "requested_tokens": [start_requested],
                        "actual_tokens": start_tokens,
                        "optimizer_updates": start_updates,
                        "recovery_validation_kl": search_full["recovery_validation_kl"],
                        "source": "swiglu-5-search-endpoint",
                    }
                ],
                "full_evaluations": [deepcopy(search_full)],
                "scientific_checkpoints": [],
                "best_validation_kl": float(search_full["recovery_validation_kl"]),
                "best_checkpoint_tokens": start_tokens,
                "best_checkpoint_updates": start_updates,
            }
        )
        atomic_torch_save(
            best_path,
            {
                "schema_version": 1,
                "workflow": CONFIRMATION_WORKFLOW,
                "run_fingerprint": context.run_fingerprint,
                "tokens_seen": start_tokens,
                "optimizer_updates": start_updates,
                "replacement_state": _replacement_state(student, target_paths),
            },
        )
        context.persist("recovery")

    target_tokens = int(context.settings["recovery"]["target_tokens"])
    if target_tokens > token_cache.token_count:
        raise ValueError("Confirmation target exceeds the finite SwiGLU-3 token stream")
    interval = int(context.settings["recovery"]["validation_interval_tokens"])
    validation_requested = list(range(interval, target_tokens + 1, interval))
    validation_requested = [value for value in validation_requested if value > start_requested]
    full_requested = [
        int(value)
        for value in context.settings["recovery"]["full_evaluation_tokens"]
        if value > start_requested
    ]
    durable_requested = validation_requested
    requested_union = sorted(set(validation_requested + full_requested))
    schedule_map = _requested_actual_map(requested_union, effective, target_tokens)
    validation_map = _requested_actual_map(validation_requested, effective, target_tokens)
    full_map = _requested_actual_map(full_requested, effective, target_tokens)
    durable_map = _requested_actual_map(durable_requested, effective, target_tokens)
    geometry = kernel["selected_geometry"]
    microbatch_sequences = int(geometry["microbatch_sequences"])
    microbatch_tokens = microbatch_sequences * token_cache.sequence_length
    trajectory["status"] = "running"
    context.persist("recovery")
    if torch.device(context.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)

    def on_checkpoint(event, optimizer, first_step):
        evaluation_started = perf_counter()
        validation_kl = evaluate_validation_kl_mixed(
            student,
            validation_cache,
            float(context.settings["recovery"]["temperature"]),
            context.device,
        )
        trajectory["validation_history"].append(
            {
                "requested_tokens": list(validation_map[event.tokens_seen]),
                "actual_tokens": event.tokens_seen,
                "optimizer_updates": event.optimizer_updates,
                "recovery_validation_kl": validation_kl,
                "mean_train_kl_since_resume": event.mean_train_kl,
            }
        )
        if event.tokens_seen in full_map:
            trajectory["full_evaluations"].append(
                {
                    "requested_tokens": list(full_map[event.tokens_seen]),
                    "actual_tokens": event.tokens_seen,
                    "optimizer_updates": event.optimizer_updates,
                    "recovery_validation_kl": validation_kl,
                    "allocation_selection": evaluate_teacher_cache_mixed(
                        student,
                        selection_cache,
                        float(context.settings["recovery"]["temperature"]),
                        context.device,
                    ),
                    "wikitext_validation": evaluate_lm_mixed(
                        student,
                        context.data["model_validation"],
                        context.device,
                        int(context.settings["data"]["model_validation_batches"]),
                    ),
                    "memory": recovery_memory_record(context.device),
                }
            )
        trajectory["evaluation_seconds"] = float(
            trajectory.get("evaluation_seconds", 0.0)
        ) + perf_counter() - evaluation_started
        improved = validation_kl < float(trajectory["best_validation_kl"])
        if improved:
            trajectory["best_validation_kl"] = validation_kl
            trajectory["best_checkpoint_tokens"] = event.tokens_seen
            trajectory["best_checkpoint_updates"] = event.optimizer_updates
        trajectory["tokens_seen"] = event.tokens_seen
        trajectory["optimizer_updates"] = event.optimizer_updates
        trajectory["training_seconds"] = event.elapsed_seconds
        trajectory["first_step"] = trajectory.get("first_step") or first_step
        checkpoint_started = perf_counter()
        state = _replacement_state(student, target_paths)
        payload = {
            "schema_version": 1,
            "workflow": CONFIRMATION_WORKFLOW,
            "run_fingerprint": context.run_fingerprint,
            "tokens_seen": event.tokens_seen,
            "optimizer_updates": event.optimizer_updates,
            "training_seconds": event.elapsed_seconds,
            "packed_token_fingerprint": token_cache.fingerprint,
            "replacement_state": state,
            "optimizer_state": optimizer.state_dict(),
            "trajectory": deepcopy(trajectory),
            **_checkpoint_rng(),
        }
        atomic_torch_save(current_path, payload)
        scientific_path = None
        scientific_sha256 = None
        if event.tokens_seen in full_map:
            scientific_path = (
                context.asset_dir
                / "recovery"
                / f"milestone-{event.tokens_seen:012d}.pt"
            )
            _link_or_copy_checkpoint(current_path, scientific_path)
            scientific_sha256 = sha256_file(scientific_path)
        if improved:
            atomic_torch_save(
                best_path,
                {
                    "schema_version": 1,
                    "workflow": CONFIRMATION_WORKFLOW,
                    "run_fingerprint": context.run_fingerprint,
                    "tokens_seen": event.tokens_seen,
                    "optimizer_updates": event.optimizer_updates,
                    "replacement_state": state,
                },
            )
        trajectory["checkpoint_seconds"] = float(
            trajectory.get("checkpoint_seconds", 0.0)
        ) + perf_counter() - checkpoint_started
        if event.tokens_seen in full_map:
            full_row = trajectory["full_evaluations"][-1]
            full_row["cumulative_training_seconds"] = event.elapsed_seconds
            full_row["cumulative_evaluation_seconds"] = float(
                trajectory["evaluation_seconds"]
            )
            full_row["cumulative_checkpoint_seconds"] = float(
                trajectory["checkpoint_seconds"]
            )
            full_row["cumulative_elapsed_seconds"] = (
                full_row["cumulative_training_seconds"]
                + full_row["cumulative_evaluation_seconds"]
                + full_row["cumulative_checkpoint_seconds"]
            )
        if scientific_path is not None:
            trajectory["scientific_checkpoints"].append(
                {
                    "requested_tokens": list(full_map[event.tokens_seen]),
                    "actual_tokens": event.tokens_seen,
                    "path": relative_to_root(scientific_path),
                    "sha256": scientific_sha256,
                }
            )
        context.persist("recovery")

    stage_started = perf_counter()
    try:
        result = recover_trainable_by_tokens(
            student=student,
            teacher=teacher,
            parameter_groups=[
                {
                    "name": "replacements",
                    "parameters": [
                        parameter for module in train_modules for parameter in module.parameters()
                    ],
                    "learning_rate": float(context.settings["recovery"]["learning_rate"]),
                    "weight_decay": float(context.settings["recovery"]["weight_decay"]),
                }
            ],
            train_modules=train_modules,
            batch_at=lambda offset, count: token_cache.batch(
                offset, count, microbatch_sequences
            ),
            target_tokens=target_tokens,
            schedule_tokens=target_tokens,
            microbatch_tokens=microbatch_tokens,
            accumulation_steps=int(geometry["gradient_accumulation_steps"]),
            temperature=float(context.settings["recovery"]["temperature"]),
            ce_weight=float(context.settings["recovery"]["ce_weight"]),
            scheduler=str(context.settings["recovery"]["scheduler"]),
            warmup_fraction=float(context.settings["recovery"]["warmup_fraction"]),
            final_lr_ratio=float(context.settings["recovery"]["final_lr_ratio"]),
            device=context.device,
            autocast_dtype=torch.bfloat16,
            start_tokens=start_tokens,
            start_updates=start_updates,
            elapsed_seconds=training_seconds,
            optimizer_state=optimizer_state,
            checkpoint_schedule=tuple(schedule_map.items()),
            on_checkpoint=on_checkpoint,
            optimizer_backend=str(context.settings["recovery"]["optimizer_backend"]),
        )
        if result.tokens_seen != target_tokens:
            raise RuntimeError("Confirmation ended before the 100M boundary")
        trajectory["tokens_seen"] = result.tokens_seen
        trajectory["optimizer_updates"] = result.optimizer_updates
        trajectory["training_seconds"] = result.elapsed_seconds
        trajectory["memory"] = recovery_memory_record(context.device)
        trajectory["status"] = "completed"
        _link_or_copy_checkpoint(current_path, final_path)
        trajectory["final_checkpoint"] = {
            "path": relative_to_root(final_path),
            "sha256": sha256_file(final_path),
            "tokens_seen": result.tokens_seen,
        }
        trajectory["best_checkpoint"] = {
            "path": relative_to_root(best_path),
            "sha256": sha256_file(best_path),
            "tokens_seen": int(trajectory["best_checkpoint_tokens"]),
            "selection_rule": "lowest fixed validation KL at or below 100M",
        }
        context.artifact["results"]["runtime"].append(
            {"stage": "confirmation_recovery", "seconds": perf_counter() - stage_started}
        )
        comparisons = []
        target_key = str(float(context.settings["target"]))
        for requested in context.settings["comparison"]["swiglu_3_equal_token_milestones"]:
            source_m = source_milestone(context.source, target_key, int(requested))
            current_m = _find_full_evaluation(trajectory, int(requested))
            comparisons.append(
                _source_comparison_row(context, source_m, current_m, trajectory)
            )
        context.artifact["results"]["paired_swiglu_3_comparison"] = comparisons
        context.artifact["status"] = "completed"
        context.artifact["completed_at_utc"] = utc_now()
        context.persist(None)
        current_path.unlink(missing_ok=True)
    finally:
        del endpoint, student, teacher, train_modules
        gc.collect()
        release_cuda(torch)
