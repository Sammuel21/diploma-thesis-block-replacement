"""Runnable third-generation SwiGLU calibration, sparsity, and recovery study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from workflows.runs.model._common import (
    default_artifact_path,
    load_artifact,
    load_workflow_config,
    release_cuda,
    report_memory,
    resolve_path,
)

from mlp_replacement.capture import ActivationPairs, collect_modules_io
from mlp_replacement.compression.allocation import allocate_ranked_swiglu_widths
from mlp_replacement.compression.recovery import (
    cache_teacher_logits,
    next_optimizer_boundary,
    recover_replacements_by_tokens,
    token_checkpoint_schedule,
)
from mlp_replacement.compression.surgery import replace_submodule
from mlp_replacement.config import DatasetSpec, ModelConfig, OperatorConfig
from mlp_replacement.data import (
    build_or_open_packed_token_cache,
    contiguous_token_windows,
    load_text_dataset,
    make_token_loader,
    sample_partitioned_windows,
)
from mlp_replacement.evaluation.footprint import parameter_footprint
from mlp_replacement.evaluation.operator import evaluate_operator
from mlp_replacement.model import discover_mlp_blocks, load_model_and_tokenizer
from mlp_replacement.operators import (
    GatedMLPReplacement,
    fit_operator_fp32_detailed,
    initialize_gated_mlp_from_teacher,
)
from mlp_replacement.runlog import environment_record, json_value


WORKFLOW = "swiglu-3"
ARTIFACT_SCHEMA = 1
ALLOCATION_SCHEMA = 1
RANKING_SCHEMA = 3
WINNING_POLICY = "singleton_kl_w25_t1"
DEFAULT_CONFIG = Path("workflows/configs/model/swiglu-3.json")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run nested-calibration and 100M-token SwiGLU recovery studies"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the matching output and current recovery checkpoint",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Apply the checked-in reduced budgets without changing scientific defaults",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    encoded = json.dumps(
        json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deep_merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_value(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def relative_to_root(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(resolve_path(Path("."))))
    except ValueError:
        return str(path)


class WorkflowContext:
    """Hold live resources and incrementally persisted workflow state."""

    def __init__(
        self,
        output,
        asset_dir,
        settings,
        allocation_artifact,
        ranking_artifact,
        source_hashes,
        run_fingerprint,
        artifact,
    ):
        self.output = Path(output)
        self.asset_dir = Path(asset_dir)
        self.settings = settings
        self.allocation_artifact = allocation_artifact
        self.ranking_artifact = ranking_artifact
        self.source_hashes = source_hashes
        self.run_fingerprint = run_fingerprint
        self.artifact = artifact
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_dtype = None
        self.blocks_by_layer = None
        self.data = None

    @property
    def sidecar(self):
        return self.output.with_suffix(".run.json")

    def persist(self, current_stage=None):
        self.artifact["updated_at_utc"] = utc_now()
        atomic_json(self.output, self.artifact)
        atomic_json(
            self.sidecar,
            {
                "schema_version": 1,
                "workflow": WORKFLOW,
                "status": self.artifact["status"],
                "current_stage": current_stage,
                "artifact_path": relative_to_root(self.output),
                "asset_directory": relative_to_root(self.asset_dir),
                "run_fingerprint": self.run_fingerprint,
                "updated_at_utc": self.artifact["updated_at_utc"],
                "error": self.artifact.get("error"),
            },
        )


def prepare_context(settings, output, resume, smoke):
    allocation_path = resolve_path(Path(settings["references"]["allocation_artifact"]))
    ranking_path = resolve_path(Path(settings["references"]["ranking_artifact"]))
    allocation_artifact = load_artifact(
        allocation_path, ALLOCATION_SCHEMA, "completed SwiGLU-2 allocation"
    )
    ranking_artifact = load_artifact(
        ranking_path, RANKING_SCHEMA, "optimized SwiGLU ranking"
    )
    effective = deep_merge(settings, settings.get("smoke_overrides", {})) if smoke else settings
    effective.pop("smoke_overrides", None)
    effective["execution_mode"] = "smoke" if smoke else "scientific"
    source_hashes = {
        "allocation_artifact": sha256_file(allocation_path),
        "ranking_artifact": sha256_file(ranking_path),
    }
    run_fingerprint = fingerprint(
        {"settings": effective, "source_hashes": source_hashes}
    )
    if output is None:
        if resume:
            raise ValueError("--resume requires the original --output path")
        output = default_artifact_path(WORKFLOW)
    output = resolve_path(output)
    asset_dir = output.with_suffix("")
    asset_dir = asset_dir.with_name(asset_dir.name + ".assets")

    if resume:
        if not output.is_file():
            raise FileNotFoundError(f"Resume artifact does not exist: {output}")
        artifact = json.loads(output.read_text(encoding="utf-8"))
        if artifact.get("run_fingerprint") != run_fingerprint:
            raise ValueError("Resume configuration or reference hashes do not match")
        if not asset_dir.is_dir():
            raise FileNotFoundError(f"Resume asset directory is missing: {asset_dir}")
        if artifact.get("status") == "completed":
            raise ValueError("The requested workflow artifact is already complete")
        artifact["status"] = "running"
        artifact["error"] = None
    else:
        if output.exists() or output.with_suffix(".run.json").exists():
            raise FileExistsError(f"Workflow output already exists: {output}")
        if asset_dir.exists():
            raise FileExistsError(f"Workflow asset directory already exists: {asset_dir}")
        asset_dir.mkdir(parents=True)
        artifact = {
            "schema_version": ARTIFACT_SCHEMA,
            "workflow": WORKFLOW,
            "status": "running",
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "run_fingerprint": run_fingerprint,
            "environment": environment_record(),
            "configuration": effective,
            "provenance": {
                "source_paths": {
                    "allocation_artifact": relative_to_root(allocation_path),
                    "ranking_artifact": relative_to_root(ranking_path),
                },
                "source_sha256": source_hashes,
            },
            "results": {
                "calibration": {
                    "operator_fitting": [],
                    "operator_training_history": [],
                    "model_evaluation": [],
                    "selected_calibration_pairs": None,
                },
                "sparsity": {
                    "allocation": [],
                    "allocation_summary": [],
                    "operator_fitting": [],
                    "operator_training_history": [],
                    "model_evaluation": [],
                },
                "recovery": {"trajectories": {}},
                "runtime": [],
            },
            "error": None,
        }
    context = WorkflowContext(
        output,
        asset_dir,
        effective,
        allocation_artifact,
        ranking_artifact,
        source_hashes,
        run_fingerprint,
        artifact,
    )
    context.persist("initialization")
    return context


def reference_configs(context):
    reference = context.ranking_artifact["configuration"]
    model_values = reference["model"]
    revision = model_values.get("resolved_revision") or model_values["revision"]
    expected = context.settings["model"]
    if model_values["model_id"] != expected["model_id"] or revision != expected["revision"]:
        raise ValueError("Pinned model identity differs from the ranking artifact")
    model_config = ModelConfig(
        model_id=expected["model_id"],
        revision=expected["revision"],
        tokenizer_revision=expected["tokenizer_revision"],
        device=expected["device"],
        dtype=expected["dtype"],
        trust_remote_code=bool(expected["trust_remote_code"]),
    )
    operator_values = context.settings["local_fitting"]
    operator_config = OperatorConfig(
        kind="swiglu",
        initialization="importance_teacher_subset",
        bottleneck_ratio=0.25,
        intermediate_ratio=0.5,
        activation="silu",
        bias=False,
        epochs=int(operator_values["max_epochs"]),
        learning_rate=float(operator_values["learning_rate"]),
        batch_size=int(operator_values["batch_size"]),
        weight_decay=float(operator_values["weight_decay"]),
        scheduler="constant",
        gradient_clip_norm=None,
        early_stopping_patience=int(operator_values["early_stopping_patience"]),
        early_stopping_min_delta=float(operator_values["early_stopping_min_delta"]),
        seed=int(context.settings["seed"]),
    )
    return model_config, operator_config


def build_local_data(context):
    values = context.settings["data"]
    sequence_length = int(values["sequence_length"])
    batch_size = int(values["capture_batch_size"])
    partitions = {name: int(count) for name, count in values["partition_batches"].items()}
    expected_order = [
        "calibration",
        "operator_validation",
        "old_recovery",
        "recovery_validation",
        "additional_calibration",
        "allocation_selection",
        "appended_calibration",
    ]
    if list(partitions) != expected_order:
        raise ValueError("Local-data partitions must preserve the declared frozen order")
    maximum_pairs = max(int(value) for value in context.settings["calibration"]["pair_counts"])
    original_pairs = (
        partitions["calibration"] + partitions["additional_calibration"]
    ) * batch_size * sequence_length
    appended_pairs = partitions["appended_calibration"] * batch_size * sequence_length
    if original_pairs + appended_pairs != maximum_pairs:
        raise ValueError("Appended calibration partition does not reach the maximum pair budget")
    source = DatasetSpec(**values["local_source"])
    records = load_text_dataset(source)
    windows = sample_partitioned_windows(
        records,
        context.tokenizer,
        {name: count * batch_size for name, count in partitions.items()},
        sequence_length,
        int(context.settings["seed"]),
        source.text_column,
    )
    calibration_sequences = (
        windows["calibration"]
        + windows["additional_calibration"]
        + windows["appended_calibration"]
    )
    if len(calibration_sequences) * sequence_length != maximum_pairs:
        raise RuntimeError("Nested calibration sequence count is inconsistent")
    validation_source = DatasetSpec(**values["model_validation_source"])
    validation_records = load_text_dataset(validation_source)
    validation_batches = int(values["model_validation_batches"])
    validation_sequences = contiguous_token_windows(
        validation_records,
        context.tokenizer,
        validation_batches * batch_size,
        sequence_length,
        validation_source.text_column,
    )
    return {
        "calibration_sequences": calibration_sequences,
        "operator_validation": make_token_loader(
            windows["operator_validation"], batch_size
        ),
        "allocation_selection": make_token_loader(
            windows["allocation_selection"], batch_size
        ),
        "recovery_validation": make_token_loader(
            windows["recovery_validation"], batch_size
        ),
        "model_validation": make_token_loader(validation_sequences, batch_size),
        "partition_batches": partitions,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
    }


def load_live_resources(context):
    model_config, operator_config = reference_configs(context)
    torch.manual_seed(int(context.settings["seed"]))
    context.model, context.tokenizer = load_model_and_tokenizer(model_config)
    context.device = next(context.model.parameters()).device
    context.model_dtype = next(context.model.parameters()).dtype
    blocks = discover_mlp_blocks(context.model)
    context.blocks_by_layer = {block.index: block for block in blocks}
    expected = context.settings["model"]
    layers = tuple(int(layer) for layer in context.settings["allocation"]["eligible_layers"])
    if len(blocks) != int(expected["num_layers"]):
        raise ValueError("Loaded model layer count differs from the frozen configuration")
    for layer in layers:
        block = context.blocks_by_layer[layer].module
        if (
            block.up_proj.in_features != int(expected["hidden_size"])
            or block.up_proj.out_features != int(expected["intermediate_size"])
        ):
            raise ValueError("Loaded SwiGLU dimensions differ from the frozen configuration")
    context.data = build_local_data(context)
    return operator_config


def autocast_context(device):
    device = torch.device(device)
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def evaluate_lm_mixed(model, loader, device, max_batches):
    was_training = model.training
    model.eval()
    total_nll = 0.0
    predicted_tokens = 0
    batches = 0
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if batch_index >= max_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with autocast_context(device):
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    ).logits
                labels = input_ids[:, 1:].contiguous()
                mask = attention_mask[:, 1:].bool()
                labels = labels.masked_fill(~mask, -100)
                nll = F.cross_entropy(
                    logits[:, :-1, :].float().contiguous().view(-1, logits.shape[-1]),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                total_nll += float(nll.item())
                predicted_tokens += int(mask.sum().item())
                batches += 1
    finally:
        model.train(was_training)
    if predicted_tokens == 0:
        raise ValueError("Language-model evaluation contained no predicted tokens")
    loss = total_nll / predicted_tokens
    return {
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 709 else float("inf"),
        "predicted_tokens": predicted_tokens,
        "batches": batches,
    }


def evaluate_teacher_cache_mixed(model, teacher_cache, temperature, device):
    losses = []
    total_nll = 0.0
    predicted_tokens = 0
    model.eval()
    with torch.no_grad():
        for batch in teacher_cache.batches:
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            with autocast_context(device):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
            mask = attention_mask.bool()
            teacher_probabilities = torch.softmax(
                batch.logits.to(device=device, dtype=torch.float32)[mask] / temperature,
                dim=-1,
            )
            student_log_probabilities = torch.log_softmax(
                logits.float()[mask] / temperature, dim=-1
            )
            kl = F.kl_div(
                student_log_probabilities,
                teacher_probabilities,
                reduction="batchmean",
            ) * (temperature**2)
            losses.append(float(kl.item()))
            labels = input_ids[:, 1:].contiguous()
            valid = attention_mask[:, 1:].bool()
            labels = labels.masked_fill(~valid, -100)
            nll = F.cross_entropy(
                logits[:, :-1, :].float().contiguous().view(-1, logits.shape[-1]),
                labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_nll += float(nll.item())
            predicted_tokens += int(valid.sum().item())
    loss = total_nll / predicted_tokens
    return {
        "teacher_kl": sum(losses) / len(losses),
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 709 else float("inf"),
        "predicted_tokens": predicted_tokens,
        "batches": len(losses),
    }


def evaluate_validation_kl_mixed(model, teacher_cache, temperature, device):
    """Measure fixed-cache teacher KL for a mixed-dtype recovery model."""

    losses = []
    model.eval()
    with torch.no_grad():
        for batch in teacher_cache.batches:
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            with autocast_context(device):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
            mask = attention_mask.bool()
            probabilities = torch.softmax(
                batch.logits.to(device=device, dtype=torch.float32)[mask] / temperature,
                dim=-1,
            )
            log_probabilities = torch.log_softmax(
                logits.float()[mask] / temperature, dim=-1
            )
            losses.append(
                float(
                    (
                        F.kl_div(
                            log_probabilities,
                            probabilities,
                            reduction="batchmean",
                        )
                        * (temperature**2)
                    ).item()
                )
            )
    if not losses:
        raise ValueError("Recovery-validation cache contained no batches")
    return sum(losses) / len(losses)


def operator_path(context, stage, label, layer):
    return context.asset_dir / "operators" / stage / str(label) / f"layer-{layer:02d}.pt"


def load_operator(path, hidden_size, width, device="cpu"):
    module = GatedMLPReplacement(hidden_size, width).to(dtype=torch.float32)
    state = torch.load(path, map_location="cpu")
    module.load_state_dict(state)
    module.to(device=device, dtype=torch.float32)
    module.eval()
    return module


@contextmanager
def temporary_fp32_replacements(model, blocks_by_layer, replacements):
    originals = {layer: blocks_by_layer[layer].module for layer in replacements}
    try:
        for layer, replacement in replacements.items():
            replace_submodule(model, blocks_by_layer[layer].path, replacement)
        yield
    finally:
        for layer, original in originals.items():
            replace_submodule(model, blocks_by_layer[layer].path, original)


def save_fit_progress(context, section, fit_row, history_rows):
    results = context.artifact["results"][section]
    keys = ("fit_key",)
    existing = {tuple(row[key] for key in keys) for row in results["operator_fitting"]}
    if tuple(fit_row[key] for key in keys) not in existing:
        results["operator_fitting"].append(fit_row)
        results["operator_training_history"].extend(history_rows)
    context.persist(section)


def fit_local_operator(
    context,
    operator_config,
    layer,
    width,
    pair_count,
    training_pairs,
    validation_pairs,
    state_path,
    fit_key,
    section,
    labels,
):
    block = context.blocks_by_layer[layer].module
    ranking = torch.tensor(
        context.ranking_artifact["results"]["teacher_neuron_rankings"][str(layer)],
        dtype=torch.long,
    )
    selected = ranking[:width].sort().values
    module = GatedMLPReplacement(training_pairs.hidden_size, width).to(
        context.device, dtype=torch.float32
    )
    initialize_gated_mlp_from_teacher(module, block, selected)
    train_subset = ActivationPairs(
        training_pairs.inputs[:pair_count], training_pairs.targets[:pair_count]
    )
    initial = evaluate_operator(
        module, validation_pairs, context.device, operator_config.batch_size
    )
    started = perf_counter()
    fit = fit_operator_fp32_detailed(
        module, train_subset, validation_pairs, operator_config, context.device
    )
    fit_seconds = perf_counter() - started
    final = evaluate_operator(
        fit.module, validation_pairs, context.device, operator_config.batch_size
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        state_path,
        {name: tensor.detach().cpu() for name, tensor in fit.module.state_dict().items()},
    )
    fit_row = {
        "fit_key": fit_key,
        **labels,
        "layer": layer,
        "calibration_pairs": pair_count,
        "replacement_width": width,
        "replacement_width_ratio": width / block.up_proj.out_features,
        "initial_local_mse": initial.mse,
        "initial_local_nmse": initial.relative_mse,
        "initial_local_cosine": initial.cosine_similarity,
        "best_epoch": fit.best_epoch,
        "epochs_completed": len(fit.history),
        "updates": fit.updates,
        "local_mse": final.mse,
        "local_nmse": final.relative_mse,
        "local_cosine": final.cosine_similarity,
        "fit_seconds": fit_seconds,
        "parameter_dtype": "torch.float32",
        "state_path": relative_to_root(state_path),
    }
    history_rows = [
        {
            "fit_key": fit_key,
            **labels,
            "layer": layer,
            "calibration_pairs": pair_count,
            **asdict(epoch),
        }
        for epoch in fit.history
    ]
    save_fit_progress(context, section, fit_row, history_rows)
    fit.module.to("cpu")
    del fit, module, train_subset
    release_cuda(torch)


def evaluate_operator_set(context, state_paths, widths, selection_cache):
    hidden_size = int(context.settings["model"]["hidden_size"])
    operators = {
        layer: load_operator(state_paths[layer], hidden_size, widths[layer], context.device)
        for layer in sorted(widths)
    }
    try:
        with temporary_fp32_replacements(
            context.model, context.blocks_by_layer, operators
        ):
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
        for module in operators.values():
            module.to("cpu")
        operators.clear()
        release_cuda(torch)
    return selection, wiki


def winning_widths(context):
    rows = [
        row
        for row in context.allocation_artifact["results"]["allocation"]
        if row["policy"] == WINNING_POLICY
    ]
    layers = tuple(int(layer) for layer in context.settings["allocation"]["eligible_layers"])
    widths = {int(row["layer"]): int(row["replacement_width"]) for row in rows}
    if set(widths) != set(layers):
        raise ValueError("Completed allocation artifact does not contain the winning widths")
    return widths


def ensure_completed_allocation(context):
    final_rows = [
        row
        for row in context.allocation_artifact["results"]["final_summary"]
        if row["policy"] == WINNING_POLICY
    ]
    if not final_rows:
        raise ValueError("SwiGLU-2 artifact does not contain the winning final summary")
    expected = context.settings["references"]["winning_pre_recovery"]
    row = final_rows[0]
    if (
        abs(float(row["pre_teacher_kl"]) - float(expected["teacher_kl"])) > 1e-12
        or abs(float(row["pre_perplexity"]) - float(expected["perplexity"])) > 1e-12
    ):
        raise ValueError("Configured winning metrics differ from the completed artifact")


def run_calibration_stage(context, operator_config):
    """Fit and select among nested 98k, 196k, and 393k calibration budgets."""

    stage = context.artifact["results"]["calibration"]
    if stage.get("completed"):
        return int(stage["selected_calibration_pairs"])
    context.persist("calibration")
    started = perf_counter()
    pair_counts = tuple(int(value) for value in context.settings["calibration"]["pair_counts"])
    if tuple(sorted(pair_counts)) != pair_counts:
        raise ValueError("Calibration pair counts must be increasing")
    widths = winning_widths(context)
    layers = tuple(sorted(widths))
    group_size = int(context.settings["calibration"]["capture_group_size"])
    fitting_keys = {
        row["fit_key"] for row in stage["operator_fitting"]
    }
    max_batches = pair_counts[-1] // (
        context.data["batch_size"] * context.data["sequence_length"]
    )
    training_loader = make_token_loader(
        context.data["calibration_sequences"], context.data["batch_size"]
    )
    for offset in range(0, len(layers), group_size):
        group = layers[offset : offset + group_size]
        missing = [
            (layer, pair_count)
            for layer in group
            for pair_count in pair_counts
            if f"calibration:{pair_count}:{layer}" not in fitting_keys
        ]
        if not missing:
            continue
        print(f"Calibration capture layers {group}", flush=True)
        paths = [context.blocks_by_layer[layer].path for layer in group]
        training_by_path = collect_modules_io(
            context.model,
            paths,
            training_loader,
            max_batches,
            context.device,
            storage_device="cpu",
            storage_dtype=context.model_dtype,
        )
        validation_by_path = collect_modules_io(
            context.model,
            paths,
            context.data["operator_validation"],
            int(context.data["partition_batches"]["operator_validation"]),
            context.device,
            storage_device="cpu",
            storage_dtype=context.model_dtype,
        )
        for layer in group:
            path = context.blocks_by_layer[layer].path
            for pair_count in pair_counts:
                fit_key = f"calibration:{pair_count}:{layer}"
                if fit_key in fitting_keys:
                    continue
                print(f"FP32 local fit N={pair_count:,} layer={layer}", flush=True)
                fit_local_operator(
                    context,
                    operator_config,
                    layer,
                    widths[layer],
                    pair_count,
                    training_by_path[path],
                    validation_by_path[path],
                    operator_path(context, "calibration", pair_count, layer),
                    fit_key,
                    "calibration",
                    {"candidate": f"n{pair_count}"},
                )
                fitting_keys.add(fit_key)
        training_by_path.clear()
        validation_by_path.clear()
        release_cuda(torch)
        report_memory(f"After calibration capture group {group}")

    selection_cache = cache_teacher_logits(
        context.model,
        context.data["allocation_selection"],
        int(context.data["partition_batches"]["allocation_selection"]),
        context.device,
        context.settings["recovery"]["validation_cache_dtype"],
    )
    existing_evaluations = {int(row["calibration_pairs"]) for row in stage["model_evaluation"]}
    for pair_count in pair_counts:
        if pair_count in existing_evaluations:
            continue
        paths = {
            layer: operator_path(context, "calibration", pair_count, layer)
            for layer in layers
        }
        selection, wiki = evaluate_operator_set(context, paths, widths, selection_cache)
        stage["model_evaluation"].append(
            {
                "candidate": f"n{pair_count}",
                "calibration_pairs": pair_count,
                "allocation_selection": selection,
                "wikitext_validation": wiki,
            }
        )
        context.persist("calibration")
    winner = min(
        stage["model_evaluation"],
        key=lambda row: (
            float(row["allocation_selection"]["teacher_kl"]),
            int(row["calibration_pairs"]),
        ),
    )
    stage["selected_calibration_pairs"] = int(winner["calibration_pairs"])
    stage["selection_rule"] = (
        "lowest allocation-selection teacher KL; ties choose fewer calibration pairs"
    )
    stage["completed"] = True
    context.artifact["results"]["runtime"].append(
        {"stage": "calibration", "seconds": perf_counter() - started}
    )
    context.persist("calibration")
    del selection_cache
    release_cuda(torch)
    return int(winner["calibration_pairs"])


def build_sparsity_allocations(context):
    settings = context.settings["allocation"]
    layers = tuple(int(layer) for layer in settings["eligible_layers"])
    original_widths = {
        layer: int(context.blocks_by_layer[layer].module.up_proj.out_features)
        for layer in layers
    }
    winner_rows = [
        row
        for row in context.allocation_artifact["results"]["allocation"]
        if row["policy"] == WINNING_POLICY
    ]
    scores = {int(row["layer"]): float(row["raw_importance"]) for row in winner_rows}
    exact_half = winning_widths(context)
    allocations = {}
    summaries = []
    dense_parameters = parameter_footprint(context.model).parameters
    original_eligible = sum(
        sum(parameter.numel() for parameter in context.blocks_by_layer[layer].module.parameters())
        for layer in layers
    )
    for target in (float(value) for value in settings["target_mlp_removals"]):
        rows, budget = allocate_ranked_swiglu_widths(
            original_widths,
            scores,
            target,
            int(context.settings["model"]["hidden_size"]),
            float(settings["temperature"]),
        )
        widths = {int(row["layer"]): int(row["replacement_width"]) for row in rows}
        if target == 0.5:
            if widths != exact_half:
                raise ValueError("Reimplemented 50% allocation differs from the frozen winner")
            widths = exact_half
            for row in rows:
                row["replacement_width"] = widths[int(row["layer"])]
                row["replacement_width_ratio"] = row["replacement_width"] / row["original_width"]
                row["replacement_parameters"] = row["replacement_width"] * budget.parameter_step
                row["realized_layer_removal"] = 1 - row["replacement_width_ratio"]
        key = f"{target:.1f}"
        allocations[key] = {"target": target, "widths": widths, "rows": list(rows)}
        realized_removed = int(budget.realized_removed_parameters)
        summaries.append(
            {
                "sparsity_key": key,
                "requested_eligible_mlp_removal": target,
                "requested_removed_parameters": int(budget.requested_removed_parameters),
                "realized_removed_parameters": realized_removed,
                "realized_eligible_mlp_removal": realized_removed / original_eligible,
                "realized_whole_model_removal": realized_removed / dense_parameters,
                "retained_total_width": sum(widths.values()),
                "minimum_retention": min(
                    widths[layer] / original_widths[layer] for layer in layers
                ),
                "maximum_retention": max(
                    widths[layer] / original_widths[layer] for layer in layers
                ),
                "whole_neuron_parameter_step": int(budget.parameter_step),
            }
        )
    return allocations, summaries


def run_sparsity_stage(context, operator_config, selected_pairs):
    """Refit the frozen allocation policy at 50%, 40%, 30%, and 20% removal."""

    stage = context.artifact["results"]["sparsity"]
    if stage.get("completed"):
        return
    context.persist("sparsity")
    started = perf_counter()
    allocations, summaries = build_sparsity_allocations(context)
    if not stage["allocation"]:
        stage["allocation"] = [
            {"sparsity_key": key, "target_mlp_removal": item["target"], **row}
            for key, item in allocations.items()
            for row in item["rows"]
        ]
        stage["allocation_summary"] = summaries
        context.persist("sparsity")
    layers = tuple(sorted(winning_widths(context)))
    group_size = int(context.settings["calibration"]["capture_group_size"])
    fitting_keys = {row["fit_key"] for row in stage["operator_fitting"]}
    required = [
        (key, layer)
        for key in allocations
        if key != "0.5"
        for layer in layers
        if f"sparsity:{key}:{layer}" not in fitting_keys
    ]
    if required:
        pair_batches = selected_pairs // (
            context.data["batch_size"] * context.data["sequence_length"]
        )
        training_loader = make_token_loader(
            context.data["calibration_sequences"][: pair_batches * context.data["batch_size"]],
            context.data["batch_size"],
        )
        for offset in range(0, len(layers), group_size):
            group = layers[offset : offset + group_size]
            group_missing = [(key, layer) for key, layer in required if layer in group]
            if not group_missing:
                continue
            paths = [context.blocks_by_layer[layer].path for layer in group]
            training_by_path = collect_modules_io(
                context.model,
                paths,
                training_loader,
                pair_batches,
                context.device,
                storage_device="cpu",
                storage_dtype=context.model_dtype,
            )
            validation_by_path = collect_modules_io(
                context.model,
                paths,
                context.data["operator_validation"],
                int(context.data["partition_batches"]["operator_validation"]),
                context.device,
                storage_device="cpu",
                storage_dtype=context.model_dtype,
            )
            for key, layer in group_missing:
                fit_key = f"sparsity:{key}:{layer}"
                width = allocations[key]["widths"][layer]
                print(f"Sparsity fit removal={key} layer={layer}", flush=True)
                path = context.blocks_by_layer[layer].path
                fit_local_operator(
                    context,
                    operator_config,
                    layer,
                    width,
                    selected_pairs,
                    training_by_path[path],
                    validation_by_path[path],
                    operator_path(context, "sparsity", key, layer),
                    fit_key,
                    "sparsity",
                    {"sparsity_key": key, "requested_mlp_removal": float(key)},
                )
                fitting_keys.add(fit_key)
            training_by_path.clear()
            validation_by_path.clear()
            release_cuda(torch)
            report_memory(f"After sparsity capture group {group}")

    selection_cache = cache_teacher_logits(
        context.model,
        context.data["allocation_selection"],
        int(context.data["partition_batches"]["allocation_selection"]),
        context.device,
        context.settings["recovery"]["validation_cache_dtype"],
    )
    evaluated = {row["sparsity_key"] for row in stage["model_evaluation"]}
    for key, allocation in allocations.items():
        if key in evaluated:
            continue
        state_paths = {
            layer: (
                operator_path(context, "calibration", selected_pairs, layer)
                if key == "0.5"
                else operator_path(context, "sparsity", key, layer)
            )
            for layer in layers
        }
        selection, wiki = evaluate_operator_set(
            context, state_paths, allocation["widths"], selection_cache
        )
        stage["model_evaluation"].append(
            {
                "sparsity_key": key,
                "phase": "pre_recovery",
                "allocation_selection": selection,
                "wikitext_validation": wiki,
            }
        )
        context.persist("sparsity")
    stage["completed"] = True
    context.artifact["results"]["runtime"].append(
        {"stage": "sparsity", "seconds": perf_counter() - started}
    )
    context.persist("sparsity")
    del selection_cache
    release_cuda(torch)


def replacement_state(model, paths):
    return {
        path: {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.get_submodule(path).state_dict().items()
        }
        for path in paths
    }


def load_replacement_state(model, state):
    for path, module_state in state.items():
        model.get_submodule(path).load_state_dict(module_state)


def recovery_memory_record(device):
    record = {}
    status = Path("/proc/self/status")
    if status.exists():
        values = dict(
            line.split(":", 1)
            for line in status.read_text(encoding="utf-8").splitlines()
            if ":" in line
        )
        record["peak_ram_gib"] = int(values["VmHWM"].split()[0]) / 1024**2
    if torch.device(device).type == "cuda":
        record["peak_vram_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
    return record


def checkpoint_metadata(context, key, widths, tokens_seen, updates):
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "run_fingerprint": context.run_fingerprint,
        "sparsity_key": key,
        "model_id": context.settings["model"]["model_id"],
        "revision": context.settings["model"]["revision"],
        "replacement_architecture": "bias-free-variable-width-swiglu",
        "widths": {str(layer): int(width) for layer, width in widths.items()},
        "tokens_seen": int(tokens_seen),
        "optimizer_updates": int(updates),
    }


def save_model_checkpoint(context, path, student, target_paths, metadata):
    atomic_torch_save(
        path,
        {**metadata, "replacement_state": replacement_state(student, target_paths)},
    )


def restore_rng(checkpoint):
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_states") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])


def run_recovery_stage(context, selected_pairs, model_config):
    """Run one resumable 100M-token trajectory for every sparsity model."""

    stage = context.artifact["results"]["recovery"]
    if stage.get("completed"):
        return
    context.persist("recovery")
    started = perf_counter()
    recovery = context.settings["recovery"]
    target_tokens = int(recovery["target_tokens_per_model"])
    sequence_length = int(context.data["sequence_length"])
    microbatch_sequences = int(recovery["microbatch_sequences"])
    microbatch_tokens = sequence_length * microbatch_sequences
    accumulation_steps = int(recovery["gradient_accumulation_steps"])
    effective_batch_tokens = microbatch_tokens * accumulation_steps
    if target_tokens % sequence_length:
        raise ValueError("Recovery target must end on a complete token sequence")
    source_values = context.settings["data"]["recovery_source"]
    local_source_values = context.settings["data"]["local_source"]
    source_identity_fields = ("path", "name", "split", "revision", "data_file")
    if all(
        source_values.get(field) == local_source_values.get(field)
        for field in source_identity_fields
    ):
        raise ValueError(
            "Recovery data must use a different source shard from local fitting and held-out data"
        )
    source = DatasetSpec(**source_values)
    records = load_text_dataset(source)
    token_cache = build_or_open_packed_token_cache(
        records,
        context.tokenizer,
        context.asset_dir / "recovery-data" / "tokens.int32",
        target_tokens,
        sequence_length,
        source_values,
        {
            "model_id": context.settings["model"]["model_id"],
            "revision": context.settings["model"]["tokenizer_revision"],
        },
        source.text_column,
    )
    stage["packed_token_cache"] = {
        "path": relative_to_root(token_cache.path),
        "fingerprint": token_cache.fingerprint,
        "token_count": token_cache.token_count,
        "sequence_length": token_cache.sequence_length,
        "source": source_values,
    }
    context.persist("recovery")
    validation_cache = cache_teacher_logits(
        context.model,
        context.data["recovery_validation"],
        int(context.data["partition_batches"]["recovery_validation"]),
        context.device,
        recovery["validation_cache_dtype"],
    )
    selection_cache = cache_teacher_logits(
        context.model,
        context.data["allocation_selection"],
        int(context.data["partition_batches"]["allocation_selection"]),
        context.device,
        recovery["validation_cache_dtype"],
    )
    allocations, _ = build_sparsity_allocations(context)
    layers = tuple(sorted(winning_widths(context)))
    save_schedule = token_checkpoint_schedule(
        target_tokens,
        int(recovery["checkpoint_interval_tokens"]),
        effective_batch_tokens,
    )
    milestone_at = {}
    for requested in (int(value) for value in recovery["milestone_tokens"]):
        if not 0 < requested <= target_tokens:
            raise ValueError("Recovery milestones must lie within the token target")
        actual = next_optimizer_boundary(
            requested, effective_batch_tokens, target_tokens
        )
        milestone_at.setdefault(actual, []).append(requested)
    combined_schedule = {
        actual: set(requested) for actual, requested in save_schedule
    }
    for actual, requested in milestone_at.items():
        combined_schedule.setdefault(actual, set()).update(requested)
    save_schedule = tuple(
        (actual, tuple(sorted(requested)))
        for actual, requested in sorted(combined_schedule.items())
    )
    for key, allocation in allocations.items():
        trajectory = stage["trajectories"].setdefault(
            key,
            {
                "status": "pending",
                "tokens_seen": 0,
                "optimizer_updates": 0,
                "best_validation_kl": None,
                "best_checkpoint_tokens": 0,
                "best_checkpoint_updates": 0,
                "validation_history": [],
                "milestones": [],
                "first_step": None,
            },
        )
        if trajectory["status"] == "completed":
            continue
        trajectory["status"] = "running"
        context.persist("recovery")
        if torch.device(context.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device)
        student, _ = load_model_and_tokenizer(model_config)
        student_blocks = {block.index: block for block in discover_mlp_blocks(student)}
        target_paths = [student_blocks[layer].path for layer in layers]
        for layer in layers:
            state_path = (
                operator_path(context, "calibration", selected_pairs, layer)
                if key == "0.5"
                else operator_path(context, "sparsity", key, layer)
            )
            module = load_operator(
                state_path,
                int(context.settings["model"]["hidden_size"]),
                allocation["widths"][layer],
                context.device,
            )
            replace_submodule(student, student_blocks[layer].path, module)
        del student_blocks
        release_cuda(torch)
        current_path = context.asset_dir / "recovery" / key / "current.pt"
        best_path = context.asset_dir / "recovery" / key / "best.pt"
        optimizer_state = None
        start_tokens = int(trajectory["tokens_seen"])
        start_updates = int(trajectory["optimizer_updates"])
        elapsed = float(trajectory.get("elapsed_seconds", 0.0))
        if current_path.is_file():
            current = torch.load(current_path, map_location="cpu")
            if current.get("run_fingerprint") != context.run_fingerprint:
                raise ValueError("Recovery checkpoint fingerprint does not match")
            if current.get("sparsity_key") != key:
                raise ValueError("Recovery checkpoint sparsity does not match")
            if current.get("packed_token_fingerprint") != token_cache.fingerprint:
                raise ValueError("Recovery checkpoint packed-token cache does not match")
            checkpoint_trajectory = deepcopy(current["trajectory"])
            checkpoint_trajectory["status"] = "running"
            stage["trajectories"][key] = checkpoint_trajectory
            trajectory = checkpoint_trajectory
            start_tokens = int(current["tokens_seen"])
            start_updates = int(current["optimizer_updates"])
            elapsed = float(current["elapsed_seconds"])
            state_payload = current.pop("replacement_state")
            load_replacement_state(student, state_payload)
            if current.get("best_state_source") == "current":
                best_metadata = checkpoint_metadata(
                    context,
                    key,
                    allocation["widths"],
                    int(trajectory["best_checkpoint_tokens"]),
                    int(trajectory["best_checkpoint_updates"]),
                )
                atomic_torch_save(
                    best_path,
                    {**best_metadata, "replacement_state": state_payload},
                )
            elif not best_path.is_file():
                raise FileNotFoundError(
                    f"Committed recovery checkpoint refers to a missing best state: {best_path}"
                )
            state_payload.clear()
            del state_payload
            optimizer_state = current.pop("optimizer_state")
            restore_rng(current)
            current.clear()
            del current
            context.persist("recovery")
        elif start_tokens:
            raise FileNotFoundError(
                f"Artifact records recovery progress but current checkpoint is missing: {current_path}"
            )
        else:
            initial_kl = evaluate_validation_kl_mixed(
                student,
                validation_cache,
                float(recovery["temperature"]),
                context.device,
            )
            trajectory["best_validation_kl"] = initial_kl
            trajectory["best_checkpoint_tokens"] = 0
            trajectory["best_checkpoint_updates"] = 0
            save_model_checkpoint(
                context,
                best_path,
                student,
                target_paths,
                checkpoint_metadata(context, key, allocation["widths"], 0, 0),
            )
            trajectory["validation_history"].append(
                {
                    "tokens_seen": 0,
                    "optimizer_updates": 0,
                    "recovery_validation_kl": initial_kl,
                    "is_best": True,
                }
            )
            context.persist("recovery")

        def batch_at(offset, count):
            return token_cache.batch(offset, count, microbatch_sequences)

        def on_checkpoint(event, optimizer, first_step):
            current_state = replacement_state(student, target_paths)
            validation_kl = evaluate_validation_kl_mixed(
                student,
                validation_cache,
                float(recovery["temperature"]),
                context.device,
            )
            improved = validation_kl < float(trajectory["best_validation_kl"])
            next_trajectory = deepcopy(trajectory)
            next_trajectory["tokens_seen"] = event.tokens_seen
            next_trajectory["optimizer_updates"] = event.optimizer_updates
            next_trajectory["elapsed_seconds"] = event.elapsed_seconds
            next_trajectory["first_step"] = first_step
            next_trajectory["validation_history"].append(
                {
                    "tokens_seen": event.tokens_seen,
                    "optimizer_updates": event.optimizer_updates,
                    "requested_checkpoint_tokens": list(
                        event.requested_checkpoint_tokens
                    ),
                    "actual_checkpoint_tokens": event.tokens_seen,
                    "recovery_validation_kl": validation_kl,
                    "mean_train_kl_since_resume": event.mean_train_kl,
                    "is_best": improved,
                }
            )
            if improved:
                next_trajectory["best_validation_kl"] = validation_kl
                next_trajectory["best_checkpoint_tokens"] = event.tokens_seen
                next_trajectory["best_checkpoint_updates"] = event.optimizer_updates
            metadata = checkpoint_metadata(
                context,
                key,
                allocation["widths"],
                event.tokens_seen,
                event.optimizer_updates,
            )
            if event.tokens_seen in milestone_at:
                current_metrics = {
                    "recovery_validation_kl": validation_kl,
                    "allocation_selection": evaluate_teacher_cache_mixed(
                        student,
                        selection_cache,
                        float(recovery["temperature"]),
                        context.device,
                    ),
                    "wikitext_validation": evaluate_lm_mixed(
                        student,
                        context.data["model_validation"],
                        context.device,
                        int(context.settings["data"]["model_validation_batches"]),
                    ),
                }
                milestone_path = (
                    context.asset_dir
                    / "recovery"
                    / key
                    / f"milestone-{event.tokens_seen:012d}.pt"
                )
                atomic_torch_save(
                    milestone_path,
                    {**metadata, "replacement_state": current_state},
                )
                if improved:
                    best_metrics = {
                        "checkpoint_tokens": event.tokens_seen,
                        "recovery_validation_kl": validation_kl,
                        "allocation_selection": deepcopy(
                            current_metrics["allocation_selection"]
                        ),
                        "wikitext_validation": deepcopy(
                            current_metrics["wikitext_validation"]
                        ),
                    }
                else:
                    best_checkpoint = torch.load(best_path, map_location="cpu")
                    best_state = best_checkpoint.pop("replacement_state")
                    load_replacement_state(student, best_state)
                    best_metrics = {
                        "checkpoint_tokens": int(best_checkpoint["tokens_seen"]),
                        "recovery_validation_kl": float(
                            next_trajectory["best_validation_kl"]
                        ),
                        "allocation_selection": evaluate_teacher_cache_mixed(
                            student,
                            selection_cache,
                            float(recovery["temperature"]),
                            context.device,
                        ),
                        "wikitext_validation": evaluate_lm_mixed(
                            student,
                            context.data["model_validation"],
                            context.device,
                            int(context.settings["data"]["model_validation_batches"]),
                        ),
                    }
                    load_replacement_state(student, current_state)
                    best_state.clear()
                    best_checkpoint.clear()
                    del best_state, best_checkpoint
                next_trajectory["milestones"].append(
                    {
                        "requested_tokens": list(milestone_at[event.tokens_seen]),
                        "actual_tokens": event.tokens_seen,
                        "optimizer_updates": event.optimizer_updates,
                        "current": current_metrics,
                        "best_under_budget": best_metrics,
                        "current_checkpoint": relative_to_root(milestone_path),
                    }
                )
            current_checkpoint = {
                **metadata,
                "replacement_state": current_state,
                "optimizer_state": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
                "elapsed_seconds": event.elapsed_seconds,
                "packed_token_fingerprint": token_cache.fingerprint,
                "best_state_source": "current" if improved else "best.pt",
                "trajectory": next_trajectory,
            }
            atomic_torch_save(current_path, current_checkpoint)
            if improved:
                atomic_torch_save(
                    best_path,
                    {**metadata, "replacement_state": current_state},
                )
            trajectory.clear()
            trajectory.update(next_trajectory)
            context.persist("recovery")

        result = recover_replacements_by_tokens(
            student=student,
            teacher=context.model,
            target_paths=target_paths,
            batch_at=batch_at,
            target_tokens=target_tokens,
            microbatch_tokens=microbatch_tokens,
            accumulation_steps=accumulation_steps,
            learning_rate=float(recovery["learning_rate"]),
            weight_decay=float(recovery["weight_decay"]),
            temperature=float(recovery["temperature"]),
            device=context.device,
            autocast_dtype=torch.bfloat16,
            start_tokens=start_tokens,
            start_updates=start_updates,
            elapsed_seconds=elapsed,
            optimizer_state=optimizer_state,
            checkpoint_schedule=save_schedule,
            on_checkpoint=on_checkpoint,
        )
        if result.tokens_seen != target_tokens:
            raise RuntimeError("Recovery trajectory ended before its token target")
        best_checkpoint = torch.load(best_path, map_location="cpu")
        best_state = best_checkpoint.pop("replacement_state")
        load_replacement_state(student, best_state)
        selected_checkpoint_tokens = int(best_checkpoint["tokens_seen"])
        best_state.clear()
        best_checkpoint.clear()
        del best_state, best_checkpoint
        trajectory["selected_checkpoint"] = {
            "path": relative_to_root(best_path),
            "tokens_seen": selected_checkpoint_tokens,
            "selection_rule": "lowest fixed recovery-validation KL at or below 100M tokens",
        }
        trajectory["post_recovery_model_evaluation"] = {
            "phase": "post_recovery_best_under_budget",
            "allocation_selection": evaluate_teacher_cache_mixed(
                student,
                selection_cache,
                float(recovery["temperature"]),
                context.device,
            ),
            "wikitext_validation": evaluate_lm_mixed(
                student,
                context.data["model_validation"],
                context.device,
                int(context.settings["data"]["model_validation_batches"]),
            ),
            "recovery_validation_kl": float(trajectory["best_validation_kl"]),
        }
        trajectory["status"] = "completed"
        trajectory["tokens_seen"] = result.tokens_seen
        trajectory["optimizer_updates"] = result.optimizer_updates
        trajectory["elapsed_seconds"] = result.elapsed_seconds
        trajectory["first_step"] = trajectory["first_step"] or result.first_step
        trajectory["memory"] = recovery_memory_record(context.device)
        context.persist("recovery")
        del student, optimizer_state
        release_cuda(torch)
    stage["completed"] = True
    stage["total_recovery_tokens"] = target_tokens * len(allocations)
    stage["effective_batch_tokens"] = effective_batch_tokens
    stage["microbatch_tokens"] = microbatch_tokens
    stage["gradient_accumulation_steps"] = accumulation_steps
    context.artifact["results"]["runtime"].append(
        {"stage": "recovery", "seconds": perf_counter() - started}
    )
    context.persist("recovery")


def run_workflow(context):
    ensure_completed_allocation(context)
    model_config, operator_config = reference_configs(context)
    operator_config = load_live_resources(context)
    context.artifact["results"]["data_partitions"] = {
        "order": list(context.data["partition_batches"]),
        "batches": context.data["partition_batches"],
        "nested_calibration_construction": (
            "calibration + additional_calibration + prefix(appended_calibration)"
        ),
    }
    context.persist("load_model_and_data")
    selected_pairs = run_calibration_stage(context, operator_config)
    run_sparsity_stage(context, operator_config, selected_pairs)
    run_recovery_stage(context, selected_pairs, model_config)
    context.artifact["status"] = "completed"
    context.artifact["completed_at_utc"] = utc_now()
    context.persist(None)


def main():
    args = parse_args()
    settings = load_workflow_config(args.config, WORKFLOW)
    context = prepare_context(settings, args.output, args.resume, args.smoke)
    try:
        run_workflow(context)
    except BaseException as error:
        context.artifact["status"] = "failed"
        context.artifact["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        context.persist("failed")
        raise
    print(f"Wrote {context.output}", flush=True)


if __name__ == "__main__":
    main()
