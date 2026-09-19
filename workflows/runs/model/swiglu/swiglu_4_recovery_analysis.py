"""Run the fourth-generation SwiGLU global-recovery analysis."""

from __future__ import annotations

import argparse
import json
import shutil
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from time import perf_counter
from types import SimpleNamespace

import torch
import torch.nn as nn

from workflows.runs.model._common import (
    default_artifact_path,
    load_artifact,
    load_workflow_config,
    release_cuda,
    resolve_path,
)
from workflows.runs.model.swiglu._shared import (
    atomic_json,
    atomic_torch_save,
    build_local_data,
    deep_merge,
    evaluate_lm_mixed,
    evaluate_teacher_cache_mixed,
    evaluate_validation_kl_mixed,
    load_operator,
    recovery_memory_record,
    relative_to_root,
    sha256_file,
)

from mlp_replacement.compression.adapters import (
    install_lora_adapters,
    lora_parameters,
)
from mlp_replacement.compression.recovery import (
    cache_teacher_logits,
    next_optimizer_boundary,
    recover_trainable_by_tokens,
)
from mlp_replacement.compression.surgery import replace_submodule
from mlp_replacement.config import ModelConfig
from mlp_replacement.data import PackedTokenCache
from mlp_replacement.model import (
    discover_mlp_blocks,
    discover_transformer_layers,
    load_model_and_tokenizer,
)
from mlp_replacement.runlog import environment_record


WORKFLOW = "swiglu-4"
ARTIFACT_SCHEMA = 1
SOURCE_SCHEMA = 1
DEFAULT_CONFIG = Path("workflows/configs/model/swiglu/swiglu-4-recovery-analysis.json")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare global-recovery objectives and trainable scopes at 50% MLP removal"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--source",
        type=Path,
        help="Override the completed swiglu-3 source artifact",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use checked-in reduced token and evaluation budgets",
    )
    return parser.parse_args()


def fingerprint(value):
    import hashlib

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def actual_boundary(requested_tokens, effective_batch_tokens):
    return next_optimizer_boundary(int(requested_tokens), int(effective_batch_tokens))


def resolve_source_asset(recorded_path, source_artifact_path):
    """Resolve a swiglu-3 asset after its artifact directory has been moved."""

    direct = resolve_path(Path(recorded_path))
    if direct.is_file():
        return direct
    parts = PurePosixPath(str(recorded_path).replace("\\", "/")).parts
    asset_index = next(
        (index for index, part in enumerate(parts) if part.endswith(".assets")),
        None,
    )
    if asset_index is None:
        raise FileNotFoundError(f"Recorded source asset has no .assets component: {recorded_path}")
    suffix = Path(*parts[asset_index + 1 :])
    source_artifact_path = Path(source_artifact_path)
    sibling = source_artifact_path.with_suffix("")
    sibling = sibling.with_name(sibling.name + ".assets") / suffix
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(
        "Required swiglu-3 asset is missing. Checked the recorded path and "
        f"the artifact-relative path: {sibling}"
    )


def source_milestone(source, sparsity_key, requested_tokens):
    trajectory = source["results"]["recovery"]["trajectories"][sparsity_key]
    for row in trajectory["milestones"]:
        if int(requested_tokens) in (int(value) for value in row["requested_tokens"]):
            return deepcopy(row)
    raise ValueError(
        f"swiglu-3 has no {requested_tokens:,}-token milestone for sparsity {sparsity_key}"
    )


def source_pre_recovery(source, sparsity_key):
    rows = source["results"]["sparsity"]["model_evaluation"]
    return deepcopy(
        next(
            row
            for row in rows
            if row["sparsity_key"] == sparsity_key and row["phase"] == "pre_recovery"
        )
    )


def source_allocation(source, sparsity_key):
    rows = source["results"]["sparsity"]["allocation"]
    selected = [row for row in rows if row["sparsity_key"] == sparsity_key]
    if not selected:
        raise ValueError(f"swiglu-3 has no allocation rows for sparsity {sparsity_key}")
    return selected


def source_operator_rows(source, calibration_pairs, sparsity_key):
    allocation = {int(row["layer"]): row for row in source_allocation(source, sparsity_key)}
    rows = [
        row
        for row in source["results"]["calibration"]["operator_fitting"]
        if int(row["calibration_pairs"]) == int(calibration_pairs)
    ]
    selected = {}
    for row in rows:
        layer = int(row["layer"])
        if layer in allocation:
            if int(row["replacement_width"]) != int(
                allocation[layer]["replacement_width"]
            ):
                raise ValueError("Selected calibration widths do not match the 50% allocation")
            selected[layer] = deepcopy(row)
    if set(selected) != set(allocation):
        raise ValueError("swiglu-3 does not contain every selected calibration operator state")
    return selected, allocation


def validate_source_contract(settings, source):
    if source.get("workflow") != "swiglu-3" or source.get("status") != "completed":
        raise ValueError("swiglu-4 requires a completed swiglu-3 artifact")
    source_config = source["configuration"]
    expected_model = settings["model"]
    for field in ("model_id", "revision", "tokenizer_revision", "hidden_size", "intermediate_size", "num_layers"):
        if source_config["model"].get(field) != expected_model.get(field):
            raise ValueError(f"swiglu-3 model contract differs at {field}")
    references = settings["references"]
    if int(source["results"]["calibration"]["selected_calibration_pairs"]) != int(
        references["selected_calibration_pairs"]
    ):
        raise ValueError("swiglu-3 selected calibration budget differs from swiglu-4")
    if source_config["references"]["winning_policy"] != references["selected_allocation_policy"]:
        raise ValueError("swiglu-3 allocation policy differs from swiglu-4")
    source_recovery = source_config["recovery"]
    recovery = settings["recovery"]
    comparisons = {
        "microbatch_sequences": recovery["microbatch_sequences"],
        "gradient_accumulation_steps": recovery["gradient_accumulation_steps"],
        "forward_autocast_dtype": recovery["forward_autocast_dtype"],
        "replacement_parameter_dtype": recovery["trainable_parameter_dtype"],
        "optimizer_state_dtype": recovery["optimizer_state_dtype"],
        "validation_cache_dtype": recovery["validation_cache_dtype"],
    }
    for field, expected in comparisons.items():
        if source_recovery.get(field) != expected:
            raise ValueError(f"swiglu-3 recovery contract differs at {field}")


@dataclass
class StudentBundle:
    model: nn.Module
    parameter_groups: list
    train_modules: list
    trainable_names: tuple[str, ...]
    scope: dict


class WorkflowContext:
    def __init__(
        self,
        output,
        scratch,
        settings,
        source_path,
        source,
        source_hash,
        run_fingerprint,
        artifact,
    ):
        self.output = Path(output)
        self.scratch = Path(scratch)
        self.settings = settings
        self.source_path = Path(source_path)
        self.source = source
        self.source_hash = source_hash
        self.run_fingerprint = run_fingerprint
        self.artifact = artifact
        self.current_stage = None
        self.teacher = None
        self.tokenizer = None
        self.device = None
        self.model_config = None
        self.data = None
        self.validation_cache = None
        self.selection_cache = None
        self.token_cache = None
        self.operator_rows = None
        self.allocation = None
        self.replacement_layers = None
        self.protected_layers = None

    @property
    def sidecar(self):
        return self.output.with_suffix(".run.json")

    def persist(self, stage=None):
        self.current_stage = stage
        self.artifact["updated_at_utc"] = utc_now()
        atomic_json(self.output, self.artifact)
        atomic_json(
            self.sidecar,
            {
                "schema_version": 1,
                "workflow": WORKFLOW,
                "status": self.artifact["status"],
                "current_stage": stage,
                "artifact_path": relative_to_root(self.output),
                "temporary_scratch": {
                    "path": relative_to_root(self.scratch),
                    "exists": self.scratch.exists(),
                    "retained_only_for_resume": True,
                },
                "persistent_weight_assets": False,
                "run_fingerprint": self.run_fingerprint,
                "updated_at_utc": self.artifact["updated_at_utc"],
                "error": self.artifact.get("error"),
            },
        )


def prepare_context(settings, output, source_override, resume, smoke):
    effective = deep_merge(settings, settings.get("smoke_overrides", {})) if smoke else deepcopy(settings)
    effective.pop("smoke_overrides", None)
    effective["execution_mode"] = "smoke" if smoke else "scientific"
    if source_override is not None:
        effective["references"]["swiglu_3_artifact"] = str(source_override)
    source_path = resolve_path(Path(effective["references"]["swiglu_3_artifact"]))
    source = load_artifact(source_path, SOURCE_SCHEMA, "completed swiglu-3 workflow")
    validate_source_contract(settings, source)
    source_hash = sha256_file(source_path)
    run_fingerprint = fingerprint(
        {"settings": effective, "source_artifact_sha256": source_hash}
    )
    if output is None:
        if resume:
            raise ValueError("--resume requires the original --output path")
        output = default_artifact_path(WORKFLOW)
    output = resolve_path(output)
    scratch = output.with_suffix("")
    scratch = scratch.with_name(scratch.name + ".scratch")
    if resume:
        if not output.is_file():
            raise FileNotFoundError(f"Resume artifact does not exist: {output}")
        artifact = json.loads(output.read_text(encoding="utf-8"))
        if artifact.get("run_fingerprint") != run_fingerprint:
            raise ValueError("Resume configuration or source artifact does not match")
        if artifact.get("status") == "completed":
            raise ValueError("The requested swiglu-4 artifact is already complete")
        scratch.mkdir(parents=True, exist_ok=True)
        artifact["status"] = "running"
        artifact["error"] = None
    else:
        if output.exists() or output.with_suffix(".run.json").exists() or scratch.exists():
            raise FileExistsError(f"swiglu-4 output or scratch path already exists: {output}")
        scratch.mkdir(parents=True)
        references = effective["references"]
        sparsity_key = references["selected_sparsity_key"]
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
                "source_paths": {"swiglu_3_artifact": relative_to_root(source_path)},
                "source_sha256": {"swiglu_3_artifact": source_hash},
                "source_workflow_fingerprint": source.get("run_fingerprint"),
            },
            "results": {
                "source_state": {
                    "calibration_pairs": int(references["selected_calibration_pairs"]),
                    "sparsity_key": sparsity_key,
                    "allocation_policy": references["selected_allocation_policy"],
                    "pre_recovery": source_pre_recovery(source, sparsity_key),
                    "swiglu_3_10m": source_milestone(source, sparsity_key, 10000000),
                    "allocation_summary": deepcopy(
                        next(
                            row
                            for row in source["results"]["sparsity"]["allocation_summary"]
                            if row["sparsity_key"] == sparsity_key
                        )
                    ),
                },
                "data": {},
                "dense_baseline": None,
                "configuration_testing": {
                    "trajectories": {},
                    "optimizer_selection": None,
                    "objective_selection": None,
                    "winner": None,
                    "optional_weight_decay": deepcopy(
                        effective["configuration_tournament"]["optional_weight_decay_candidate"]
                    ),
                },
                "rmsnorm": {"trajectories": {}, "selection": None, "winner": None},
                "lora": {"trajectories": {}, "comparison": None},
                "final_comparison": [],
                "runtime": [],
            },
            "storage": {
                "persistent_weight_assets": False,
                "persistent_outputs": [
                    relative_to_root(output),
                    relative_to_root(output.with_suffix(".run.json")),
                ],
                "temporary_scratch_policy": effective["recovery"]["temporary_checkpoint_policy"],
            },
            "error": None,
        }
    context = WorkflowContext(
        output,
        scratch,
        effective,
        source_path,
        source,
        source_hash,
        run_fingerprint,
        artifact,
    )
    context.persist("initialization")
    return context


def make_model_config(values):
    return ModelConfig(
        model_id=values["model_id"],
        revision=values["revision"],
        tokenizer_revision=values["tokenizer_revision"],
        device=values["device"],
        dtype=values["dtype"],
        trust_remote_code=bool(values["trust_remote_code"]),
    )


def load_live_resources(context):
    started = perf_counter()
    sparsity_key = context.settings["references"]["selected_sparsity_key"]
    pairs = context.settings["references"]["selected_calibration_pairs"]
    context.operator_rows, context.allocation = source_operator_rows(
        context.source, pairs, sparsity_key
    )
    context.replacement_layers = tuple(sorted(context.operator_rows))
    context.protected_layers = tuple(
        int(value)
        for value in context.source["configuration"]["allocation"]["protected_layers"]
    )
    operator_assets = {
        layer: resolve_source_asset(row["state_path"], context.source_path)
        for layer, row in context.operator_rows.items()
    }
    context.operator_assets = operator_assets
    cache_record = context.source["results"]["recovery"]["packed_token_cache"]
    cache_path = resolve_source_asset(cache_record["path"], context.source_path)
    expected_bytes = int(cache_record["token_count"]) * 4
    if cache_path.stat().st_size != expected_bytes:
        raise ValueError("swiglu-3 packed-token cache size differs from its manifest")
    context.token_cache = PackedTokenCache(
        cache_path,
        int(cache_record["token_count"]),
        int(cache_record["sequence_length"]),
        cache_record["fingerprint"],
    )
    context.model_config = make_model_config(context.settings["model"])
    torch.manual_seed(int(context.settings["seed"]))
    context.teacher, context.tokenizer = load_model_and_tokenizer(context.model_config)
    context.device = next(context.teacher.parameters()).device
    source_settings = deepcopy(context.source["configuration"])
    if context.settings["execution_mode"] == "smoke":
        source_settings["data"]["model_validation_batches"] = 1
        source_settings["data"]["partition_batches"]["recovery_validation"] = 1
        source_settings["data"]["partition_batches"]["allocation_selection"] = 1
    source_settings.pop("smoke_overrides", None)
    data_context = SimpleNamespace(settings=source_settings, tokenizer=context.tokenizer)
    context.data = build_local_data(data_context)
    del context.data["calibration_sequences"]
    recovery = context.settings["recovery"]
    cache_dtype = context.source["configuration"]["recovery"]["validation_cache_dtype"]
    context.validation_cache = cache_teacher_logits(
        context.teacher,
        context.data["recovery_validation"],
        int(context.data["partition_batches"]["recovery_validation"]),
        context.device,
        cache_dtype,
    )
    context.selection_cache = cache_teacher_logits(
        context.teacher,
        context.data["allocation_selection"],
        int(context.data["partition_batches"]["allocation_selection"]),
        context.device,
        cache_dtype,
    )
    sequence_length = int(context.data["sequence_length"])
    microbatch_tokens = sequence_length * int(recovery["microbatch_sequences"])
    effective_batch_tokens = microbatch_tokens * int(
        recovery["gradient_accumulation_steps"]
    )
    full_actual = actual_boundary(recovery["target_tokens"], effective_batch_tokens)
    if full_actual > context.token_cache.token_count:
        raise ValueError("swiglu-3 packed-token cache is shorter than swiglu-4")
    context.artifact["results"]["data"] = {
        "source": deepcopy(cache_record["source"]),
        "packed_token_cache": {
            "source_path": relative_to_root(cache_path),
            "fingerprint": context.token_cache.fingerprint,
            "available_tokens": context.token_cache.token_count,
            "reused_read_only": True,
            "copied_into_output": False,
        },
        "sequence_length": sequence_length,
        "microbatch_tokens": microbatch_tokens,
        "effective_batch_tokens": effective_batch_tokens,
        "evaluation_partition_batches": deepcopy(context.data["partition_batches"]),
        "operator_state_files": {
            str(layer): {
                "source_path": relative_to_root(path),
                "bytes": path.stat().st_size,
                "replacement_width": int(context.operator_rows[layer]["replacement_width"]),
            }
            for layer, path in operator_assets.items()
        },
    }
    context.artifact["results"]["dense_baseline"] = {
        "recovery_validation_kl_t1": evaluate_validation_kl_mixed(
            context.teacher,
            context.validation_cache,
            float(recovery["fixed_selection_temperature"]),
            context.device,
        ),
        "allocation_selection": evaluate_teacher_cache_mixed(
            context.teacher,
            context.selection_cache,
            float(recovery["fixed_selection_temperature"]),
            context.device,
        ),
        "wikitext_validation": evaluate_lm_mixed(
            context.teacher,
            context.data["model_validation"],
            context.device,
            int(source_settings["data"]["model_validation_batches"]),
        ),
    }
    context.artifact["results"]["runtime"].append(
        {"stage": "load_model_data_and_source_state", "seconds": perf_counter() - started}
    )
    context.persist("load_model_data_and_source_state")


def module_parameter_names(model, parameters):
    by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    names = []
    for parameter in parameters:
        try:
            names.append(by_id[id(parameter)])
        except KeyError as exc:
            raise ValueError("Trainable parameter is not registered in the student model") from exc
    if len(names) != len(set(names)):
        raise ValueError("Trainable parameter groups contain duplicate parameters")
    return tuple(names)


def trainable_state(model, names):
    parameters = dict(model.named_parameters())
    return {name: parameters[name].detach().cpu().clone() for name in names}


def load_trainable_state(model, state):
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name]))


def tied_embedding_record(model):
    inputs = model.get_input_embeddings()
    outputs = model.get_output_embeddings()
    tied = outputs is not None and inputs.weight.data_ptr() == outputs.weight.data_ptr()
    return {
        "input_embedding_class": type(inputs).__name__,
        "output_head_class": type(outputs).__name__ if outputs is not None else None,
        "weight_storage_tied": tied,
        "config_tie_word_embeddings": bool(getattr(model.config, "tie_word_embeddings", False)),
    }


def build_student(context, spec):
    torch.manual_seed(int(context.settings["seed"]))
    student, _ = load_model_and_tokenizer(context.model_config)
    embedding_record = tied_embedding_record(student)
    expected_tied = bool(context.settings["model"]["tie_word_embeddings"])
    if embedding_record["weight_storage_tied"] != expected_tied:
        raise ValueError("Loaded embedding/head tying differs from the frozen swiglu-4 contract")
    blocks = {block.index: block for block in discover_mlp_blocks(student)}
    replacement_paths = []
    replacements = []
    for layer in context.replacement_layers:
        width = int(context.operator_rows[layer]["replacement_width"])
        module = load_operator(
            context.operator_assets[layer],
            int(context.settings["model"]["hidden_size"]),
            width,
            context.device,
        )
        replace_submodule(student, blocks[layer].path, module)
        replacement_paths.append(blocks[layer].path)
        replacements.append(module)
    variant = spec["variant"]
    groups = []
    train_modules = []
    scope = {
        "variant": variant,
        "replacement_layers": list(context.replacement_layers),
        "replacement_paths": replacement_paths,
        "protected_layers": list(context.protected_layers),
        "embedding_and_lm_head": embedding_record,
        "excluded_modules": ["input_embeddings", "lm_head"],
    }
    replacement_parameters = [
        parameter for module in replacements for parameter in module.parameters()
    ]
    if variant in ("replacement", "replacement_rmsnorm", "transformer_lora_assisted"):
        groups.append(
            {
                "name": "replacement_mlp",
                "parameters": replacement_parameters,
                "learning_rate": float(spec["learning_rate"]),
                "weight_decay": float(spec["weight_decay"]),
            }
        )
        train_modules.extend(replacements)
    if variant == "replacement_rmsnorm":
        layers = {layer.index: layer for layer in discover_transformer_layers(student)}
        norms = []
        norm_paths = []
        for layer in context.replacement_layers:
            layer_ref = layers[layer]
            path = f"{layer_ref.path}.post_attention_layernorm"
            norm = student.get_submodule(path)
            norm.to(dtype=torch.float32)
            norms.append(norm)
            norm_paths.append(path)
        norm_parameters = [parameter for module in norms for parameter in module.parameters()]
        groups.append(
            {
                "name": "post_attention_rmsnorm",
                "parameters": norm_parameters,
                "learning_rate": float(spec["learning_rate"])
                * float(spec["rmsnorm_learning_rate_multiplier"]),
                "weight_decay": 0.0,
            }
        )
        train_modules.extend(norms)
        scope["rmsnorm_paths"] = norm_paths
        scope["rmsnorm_learning_rate_multiplier"] = float(
            spec["rmsnorm_learning_rate_multiplier"]
        )
    elif variant == "replacement_mlp_lora":
        projection_paths = [
            f"{path}.{projection}"
            for path in replacement_paths
            for projection in ("gate_projection", "up_projection", "down_projection")
        ]
        adapters = install_lora_adapters(
            student,
            projection_paths,
            int(spec["lora_rank"]),
            float(spec["lora_alpha"]),
            float(spec["lora_dropout"]),
        )
        groups.append(
            {
                "name": "replacement_mlp_lora",
                "parameters": lora_parameters(adapters),
                "learning_rate": float(spec["lora_learning_rate"]),
                "weight_decay": float(spec["weight_decay"]),
            }
        )
        train_modules.extend(adapters.values())
        scope["lora_paths"] = projection_paths
    elif variant == "transformer_lora_assisted":
        layers = {layer.index: layer for layer in discover_transformer_layers(student)}
        attention_names = context.settings["lora"]["target_attention_projections"]
        protected_names = context.settings["lora"]["target_protected_mlp_projections"]
        projection_paths = [
            f"{layer.path}.self_attn.{name}"
            for layer in layers.values()
            for name in attention_names
        ]
        projection_paths.extend(
            f"{blocks[layer].path}.{name}"
            for layer in context.protected_layers
            for name in protected_names
        )
        adapters = install_lora_adapters(
            student,
            projection_paths,
            int(spec["lora_rank"]),
            float(spec["lora_alpha"]),
            float(spec["lora_dropout"]),
        )
        groups.append(
            {
                "name": "transformer_lora",
                "parameters": lora_parameters(adapters),
                "learning_rate": float(spec["lora_learning_rate"]),
                "weight_decay": float(spec["weight_decay"]),
            }
        )
        train_modules.extend(adapters.values())
        scope["lora_paths"] = projection_paths
        scope["lora_explanation"] = (
            "Attention projections in every block and original MLP projections in "
            "protected blocks; embeddings and the tied LM head remain frozen."
        )
    elif variant != "replacement":
        raise ValueError(f"Unsupported recovery variant: {variant}")
    all_parameters = [parameter for group in groups for parameter in group["parameters"]]
    if any(parameter.dtype != torch.float32 for parameter in all_parameters):
        raise ValueError("Every swiglu-4 trainable parameter must use an FP32 master weight")
    names = module_parameter_names(student, all_parameters)
    group_rows = []
    offset = 0
    for group in groups:
        count = len(group["parameters"])
        group_names = names[offset : offset + count]
        offset += count
        group_rows.append(
            {
                "name": group["name"],
                "learning_rate": float(group["learning_rate"]),
                "weight_decay": float(group["weight_decay"]),
                "parameter_tensors": count,
                "parameters": sum(parameter.numel() for parameter in group["parameters"]),
                "parameter_names": list(group_names),
            }
        )
    scope["parameter_groups"] = group_rows
    scope["trainable_parameters"] = sum(parameter.numel() for parameter in all_parameters)
    scope["total_parameters_with_adapters"] = sum(
        parameter.numel() for parameter in student.parameters()
    )
    return StudentBundle(student, groups, train_modules, names, scope)


def trajectory_section(context, family):
    if family == "configuration":
        return context.artifact["results"]["configuration_testing"]["trajectories"]
    return context.artifact["results"][family]["trajectories"]


def trajectory_scratch(context, trajectory_id):
    return context.scratch / "trajectories" / trajectory_id


def checkpoint_schedule(context, requested_target):
    recovery = context.settings["recovery"]
    effective = context.artifact["results"]["data"]["effective_batch_tokens"]
    interval = int(recovery["checkpoint_interval_tokens"])
    requested = set(range(interval, int(requested_target), interval))
    requested.add(int(requested_target))
    requested.update(
        int(value)
        for value in recovery["evaluation_tokens"]
        if 0 < int(value) <= int(requested_target)
    )
    grouped = {}
    for value in sorted(requested):
        grouped.setdefault(actual_boundary(value, effective), []).append(value)
    return tuple((actual, tuple(values)) for actual, values in sorted(grouped.items()))


def evaluate_student(context, model):
    temperature = float(context.settings["recovery"]["fixed_selection_temperature"])
    source_settings = context.source["configuration"]
    if context.settings["execution_mode"] == "smoke":
        source_settings = deepcopy(source_settings)
        source_settings["data"]["model_validation_batches"] = 1
    validation_kl = evaluate_validation_kl_mixed(
        model, context.validation_cache, temperature, context.device
    )
    return {
        "recovery_validation_kl": validation_kl,
        "recovery_validation_kl_t1": validation_kl,
        "allocation_selection": evaluate_teacher_cache_mixed(
            model, context.selection_cache, temperature, context.device
        ),
        "wikitext_validation": evaluate_lm_mixed(
            model,
            context.data["model_validation"],
            context.device,
            int(source_settings["data"]["model_validation_batches"]),
        ),
    }


def restore_rng(checkpoint):
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_states") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])


def run_trajectory(context, spec, requested_target):
    section = trajectory_section(context, spec["family"])
    trajectory_id = spec["id"]
    spec_hash = fingerprint(spec)
    trajectory = section.setdefault(
        trajectory_id,
        {
            "id": trajectory_id,
            "label": spec["label"],
            "family": spec["family"],
            "status": "pending",
            "configuration": deepcopy(spec),
            "configuration_fingerprint": spec_hash,
            "tokens_seen": 0,
            "requested_tokens_seen": 0,
            "optimizer_updates": 0,
            "elapsed_seconds": 0.0,
            "best_validation_kl": None,
            "best_checkpoint_tokens": 0,
            "best_checkpoint_updates": 0,
            "validation_history": [],
            "milestones": [],
            "first_step": None,
            "selected_checkpoint": None,
            "memory": None,
        },
    )
    if trajectory["configuration_fingerprint"] != spec_hash:
        raise ValueError(f"Trajectory {trajectory_id} configuration changed during resume")
    effective = int(context.artifact["results"]["data"]["effective_batch_tokens"])
    actual_target = actual_boundary(requested_target, effective)
    if int(trajectory["tokens_seen"]) >= actual_target:
        return trajectory
    print(
        f"[{spec['family']}:{trajectory_id}] recovering to requested "
        f"{int(requested_target):,} tokens (actual {actual_target:,})",
        flush=True,
    )
    context.persist(f"{spec['family']}:{trajectory_id}")
    started = perf_counter()
    scratch = trajectory_scratch(context, trajectory_id)
    scratch.mkdir(parents=True, exist_ok=True)
    current_path = scratch / "current.pt"
    best_path = scratch / "best.pt"
    if torch.device(context.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)
    bundle = build_student(context, spec)
    student = bundle.model
    trajectory["trainable_scope"] = bundle.scope
    trajectory["status"] = "running"
    start_tokens = int(trajectory["tokens_seen"])
    start_updates = int(trajectory["optimizer_updates"])
    elapsed = float(trajectory["elapsed_seconds"])
    optimizer_state = None
    if current_path.is_file():
        current = torch.load(current_path, map_location="cpu")
        if current.get("run_fingerprint") != context.run_fingerprint:
            raise ValueError(f"Trajectory {trajectory_id} checkpoint fingerprint differs")
        if current.get("configuration_fingerprint") != spec_hash:
            raise ValueError(f"Trajectory {trajectory_id} checkpoint configuration differs")
        if current.get("packed_token_fingerprint") != context.token_cache.fingerprint:
            raise ValueError(f"Trajectory {trajectory_id} token stream differs")
        trajectory.clear()
        trajectory.update(deepcopy(current["trajectory"]))
        trajectory["status"] = "running"
        start_tokens = int(current["tokens_seen"])
        start_updates = int(current["optimizer_updates"])
        elapsed = float(current["elapsed_seconds"])
        state_payload = current.pop("trainable_state")
        load_trainable_state(student, state_payload)
        if current.get("best_state_source") == "current":
            atomic_torch_save(
                best_path,
                {
                    "tokens_seen": start_tokens,
                    "optimizer_updates": start_updates,
                    "recovery_validation_kl_t1": trajectory["best_validation_kl"],
                    "trainable_state": state_payload,
                },
            )
        elif not best_path.is_file():
            raise FileNotFoundError(
                f"Trajectory {trajectory_id} checkpoint refers to a missing best state"
            )
        optimizer_state = current.pop("optimizer_state")
        restore_rng(current)
        state_payload.clear()
        current.clear()
    elif start_tokens:
        raise FileNotFoundError(
            f"Trajectory {trajectory_id} records progress but its scratch checkpoint is missing"
        )
    else:
        initial = evaluate_student(context, student)
        trajectory["best_validation_kl"] = initial["recovery_validation_kl_t1"]
        trajectory["validation_history"].append(
            {
                "requested_checkpoint_tokens": [0],
                "actual_checkpoint_tokens": 0,
                "tokens_seen": 0,
                "optimizer_updates": 0,
                "recovery_validation_kl": initial["recovery_validation_kl_t1"],
                "mean_train_kl_since_resume": None,
                "mean_train_ce_since_resume": None,
                "mean_train_loss_since_resume": None,
                "learning_rates": {},
                "elapsed_seconds": 0.0,
                "tokens_per_second": None,
                "is_best": True,
            }
        )
        trajectory["milestones"].append(
            {
                "requested_tokens": [0],
                "actual_tokens": 0,
                "optimizer_updates": 0,
                "current": initial,
                "best_under_budget": {
                    "checkpoint_tokens": 0,
                    "recovery_validation_kl": initial["recovery_validation_kl"],
                    "allocation_selection": deepcopy(initial["allocation_selection"]),
                    "wikitext_validation": deepcopy(initial["wikitext_validation"]),
                },
            }
        )
        atomic_torch_save(
            best_path,
            {
                "tokens_seen": 0,
                "optimizer_updates": 0,
                "metrics": initial,
                "trainable_state": trainable_state(student, bundle.trainable_names),
            },
        )
        context.persist(f"{spec['family']}:{trajectory_id}")
    schedule = checkpoint_schedule(context, requested_target)
    evaluation_requested = {
        int(value)
        for value in context.settings["recovery"]["evaluation_tokens"]
        if int(value) <= int(requested_target)
    }
    evaluation_requested.add(int(requested_target))
    evaluation_actual = {
        actual_boundary(value, effective): value for value in evaluation_requested
    }

    def batch_at(offset, count):
        return context.token_cache.batch(
            offset, count, int(context.settings["recovery"]["microbatch_sequences"])
        )

    def on_checkpoint(event, optimizer, first_step):
        validation_kl = evaluate_validation_kl_mixed(
            student,
            context.validation_cache,
            float(context.settings["recovery"]["fixed_selection_temperature"]),
            context.device,
        )
        improved = validation_kl < float(trajectory["best_validation_kl"])
        next_trajectory = deepcopy(trajectory)
        next_trajectory["tokens_seen"] = int(event.tokens_seen)
        next_trajectory["requested_tokens_seen"] = max(
            int(value) for value in event.requested_checkpoint_tokens
        )
        next_trajectory["optimizer_updates"] = int(event.optimizer_updates)
        next_trajectory["elapsed_seconds"] = float(event.elapsed_seconds)
        next_trajectory["first_step"] = first_step
        group_names = [group["name"] for group in bundle.parameter_groups]
        history = {
            "requested_checkpoint_tokens": list(event.requested_checkpoint_tokens),
            "actual_checkpoint_tokens": int(event.tokens_seen),
            "tokens_seen": int(event.tokens_seen),
            "optimizer_updates": int(event.optimizer_updates),
            "recovery_validation_kl": validation_kl,
            "mean_train_kl_since_resume": event.mean_train_kl,
            "mean_train_ce_since_resume": event.mean_train_ce,
            "mean_train_loss_since_resume": event.mean_train_loss,
            "learning_rates": {
                name: value for name, value in zip(group_names, event.learning_rates, strict=True)
            },
            "elapsed_seconds": float(event.elapsed_seconds),
            "tokens_per_second": (
                float(event.tokens_seen) / float(event.elapsed_seconds)
                if event.elapsed_seconds
                else None
            ),
            "is_best": improved,
        }
        next_trajectory["validation_history"].append(history)
        if improved:
            next_trajectory["best_validation_kl"] = validation_kl
            next_trajectory["best_checkpoint_tokens"] = int(event.tokens_seen)
            next_trajectory["best_checkpoint_updates"] = int(event.optimizer_updates)
        state = trainable_state(student, bundle.trainable_names)
        if int(event.tokens_seen) in evaluation_actual:
            metrics = evaluate_student(context, student)
            if improved:
                best_metrics = metrics
                best_tokens = int(event.tokens_seen)
            else:
                best_checkpoint = torch.load(best_path, map_location="cpu")
                load_trainable_state(student, best_checkpoint["trainable_state"])
                best_metrics = evaluate_student(context, student)
                best_tokens = int(best_checkpoint["tokens_seen"])
                load_trainable_state(student, state)
                best_checkpoint.clear()
            next_trajectory["milestones"].append(
                {
                    "requested_tokens": [int(evaluation_actual[int(event.tokens_seen)])],
                    "actual_tokens": int(event.tokens_seen),
                    "optimizer_updates": int(event.optimizer_updates),
                    "current": metrics,
                    "best_under_budget": {
                        "checkpoint_tokens": best_tokens,
                        "recovery_validation_kl": best_metrics[
                            "recovery_validation_kl"
                        ],
                        "allocation_selection": deepcopy(
                            best_metrics["allocation_selection"]
                        ),
                        "wikitext_validation": deepcopy(
                            best_metrics["wikitext_validation"]
                        ),
                    },
                }
            )
        checkpoint = {
            "schema_version": 1,
            "workflow": WORKFLOW,
            "run_fingerprint": context.run_fingerprint,
            "configuration_fingerprint": spec_hash,
            "trajectory_id": trajectory_id,
            "packed_token_fingerprint": context.token_cache.fingerprint,
            "tokens_seen": int(event.tokens_seen),
            "optimizer_updates": int(event.optimizer_updates),
            "elapsed_seconds": float(event.elapsed_seconds),
            "trainable_state": state,
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "best_state_source": "current" if improved else "best.pt",
            "trajectory": next_trajectory,
        }
        atomic_torch_save(current_path, checkpoint)
        if improved:
            atomic_torch_save(
                best_path,
                {
                    "tokens_seen": int(event.tokens_seen),
                    "optimizer_updates": int(event.optimizer_updates),
                    "recovery_validation_kl_t1": validation_kl,
                    "trainable_state": state,
                },
            )
        trajectory.clear()
        trajectory.update(next_trajectory)
        context.persist(f"{spec['family']}:{trajectory_id}")
        print(
            f"[{spec['family']}:{trajectory_id}] {event.tokens_seen:,} tokens, "
            f"validation KL {validation_kl:.6f}",
            flush=True,
        )

    microbatch_tokens = int(context.artifact["results"]["data"]["microbatch_tokens"])
    full_actual = actual_boundary(context.settings["recovery"]["target_tokens"], effective)
    autocast_dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    autocast_dtype = autocast_dtypes[
        context.settings["recovery"]["forward_autocast_dtype"]
    ]
    result = recover_trainable_by_tokens(
        student=student,
        teacher=context.teacher,
        parameter_groups=bundle.parameter_groups,
        train_modules=bundle.train_modules,
        batch_at=batch_at,
        target_tokens=actual_target,
        schedule_tokens=full_actual,
        microbatch_tokens=microbatch_tokens,
        accumulation_steps=int(context.settings["recovery"]["gradient_accumulation_steps"]),
        temperature=float(spec["temperature"]),
        ce_weight=float(spec["ce_weight"]),
        scheduler=spec["scheduler"],
        warmup_fraction=float(spec["warmup_fraction"]),
        final_lr_ratio=float(spec["final_lr_ratio"]),
        device=context.device,
        autocast_dtype=autocast_dtype,
        start_tokens=start_tokens,
        start_updates=start_updates,
        elapsed_seconds=elapsed,
        optimizer_state=optimizer_state,
        checkpoint_schedule=schedule,
        on_checkpoint=on_checkpoint,
    )
    if result.tokens_seen != actual_target:
        raise RuntimeError(f"Trajectory {trajectory_id} ended before its requested target")
    best = torch.load(best_path, map_location="cpu")
    load_trainable_state(student, best["trainable_state"])
    best_metrics = evaluate_student(context, student)
    trajectory["selected_checkpoint"] = {
        "tokens_seen": int(best["tokens_seen"]),
        "optimizer_updates": int(best["optimizer_updates"]),
        "selection_rule": "lowest fixed T=1 recovery-validation KL at or below this budget",
        "metrics": best_metrics,
        "weights_retained_after_success": False,
    }
    trajectory["post_recovery_model_evaluation"] = {
        "phase": "post_recovery_best_under_budget",
        "allocation_selection": deepcopy(best_metrics["allocation_selection"]),
        "wikitext_validation": deepcopy(best_metrics["wikitext_validation"]),
        "recovery_validation_kl": best_metrics["recovery_validation_kl"],
    }
    trajectory["endpoint"] = {
        "requested_tokens": int(requested_target),
        "actual_tokens": int(actual_target),
        "recovery_validation_kl_t1": next(
            row["recovery_validation_kl"]
            for row in reversed(trajectory["validation_history"])
            if int(row["tokens_seen"]) == actual_target
        ),
        "metrics": next(
            row["current"]
            for row in reversed(trajectory["milestones"])
            if int(row["actual_tokens"]) == actual_target
        ),
    }
    trajectory["endpoint"]["recovery_validation_kl"] = trajectory["endpoint"][
        "recovery_validation_kl_t1"
    ]
    trajectory["status"] = "completed_to_requested_budget"
    trajectory["tokens_seen"] = int(result.tokens_seen)
    trajectory["requested_tokens_seen"] = int(requested_target)
    trajectory["optimizer_updates"] = int(result.optimizer_updates)
    trajectory["elapsed_seconds"] = float(result.elapsed_seconds)
    trajectory["first_step"] = trajectory["first_step"] or result.first_step
    trajectory["memory"] = recovery_memory_record(context.device)
    trajectory["runtime_seconds_this_call"] = perf_counter() - started
    best.clear()
    context.persist(f"{spec['family']}:{trajectory_id}")
    del student, bundle, optimizer_state
    release_cuda(torch)
    return trajectory


def endpoint_kl(context, family, trajectory_id, requested_tokens):
    trajectory = trajectory_section(context, family)[trajectory_id]
    effective = context.artifact["results"]["data"]["effective_batch_tokens"]
    actual = actual_boundary(requested_tokens, effective)
    return next(
        float(row["recovery_validation_kl"])
        for row in trajectory["validation_history"]
        if int(row["tokens_seen"]) == actual
    )


def cleanup_trajectory(context, trajectory_id):
    path = trajectory_scratch(context, trajectory_id).resolve()
    root = context.scratch.resolve()
    if path.parent != (root / "trajectories"):
        raise ValueError("Refusing to remove a scratch path outside this swiglu-4 run")
    if path.is_dir():
        shutil.rmtree(path)


def optimizer_spec(candidate):
    return {
        **deepcopy(candidate),
        "family": "configuration",
        "variant": "replacement",
    }


def objective_spec(candidate, optimizer_winner):
    return {
        "id": candidate["id"],
        "label": candidate["label"],
        "family": "configuration",
        "variant": "replacement",
        "learning_rate": optimizer_winner["learning_rate"],
        "scheduler": optimizer_winner["scheduler"],
        "warmup_fraction": optimizer_winner["warmup_fraction"],
        "final_lr_ratio": optimizer_winner["final_lr_ratio"],
        "temperature": candidate["temperature"],
        "ce_weight": candidate["ce_weight"],
        "weight_decay": candidate["weight_decay"],
        "optimizer_source": optimizer_winner["id"],
    }


def run_configuration_testing(context):
    section = context.artifact["results"]["configuration_testing"]
    settings = context.settings["configuration_tournament"]
    recovery = context.settings["recovery"]
    qualifier = int(recovery["tournament_qualifier_tokens"])
    full = int(recovery["target_tokens"])
    candidates = [optimizer_spec(candidate) for candidate in settings["optimizer_candidates"]]
    for spec in candidates:
        initial_target = full if spec["initial_target"] == "full" else qualifier
        run_trajectory(context, spec, initial_target)
    qualifier_scores = {
        spec["id"]: endpoint_kl(context, "configuration", spec["id"], qualifier)
        for spec in candidates
    }
    optimizer_winner_id = min(qualifier_scores, key=qualifier_scores.get)
    optimizer_winner = next(spec for spec in candidates if spec["id"] == optimizer_winner_id)
    section["optimizer_selection"] = {
        "budget_tokens": qualifier,
        "metric": "fixed_recovery_validation_kl_t1",
        "scores": qualifier_scores,
        "winner": optimizer_winner_id,
    }
    context.persist("configuration:optimizer_selection")
    print(f"[configuration] optimizer winner: {optimizer_winner_id}", flush=True)
    if optimizer_winner["initial_target"] != "full":
        run_trajectory(context, optimizer_winner, full)
    for spec in candidates:
        if spec["id"] != optimizer_winner_id and spec["initial_target"] != "full":
            cleanup_trajectory(context, spec["id"])
    objective_candidates = [
        objective_spec(candidate, optimizer_winner)
        for candidate in settings["objective_candidates"]
    ]
    for spec in objective_candidates:
        run_trajectory(context, spec, qualifier)
    objective_scores = {
        optimizer_winner_id: endpoint_kl(
            context, "configuration", optimizer_winner_id, qualifier
        ),
        **{
            spec["id"]: endpoint_kl(context, "configuration", spec["id"], qualifier)
            for spec in objective_candidates
        },
    }
    promoted_id = min(objective_scores, key=objective_scores.get)
    promoted_spec = (
        optimizer_winner
        if promoted_id == optimizer_winner_id
        else next(spec for spec in objective_candidates if spec["id"] == promoted_id)
    )
    section["objective_selection"] = {
        "budget_tokens": qualifier,
        "metric": "fixed_recovery_validation_kl_t1",
        "scores": objective_scores,
        "promoted": promoted_id,
    }
    context.persist("configuration:objective_selection")
    print(f"[configuration] objective candidate promoted: {promoted_id}", flush=True)
    if promoted_id != optimizer_winner_id:
        run_trajectory(context, promoted_spec, full)
    full_scores = {
        optimizer_winner_id: endpoint_kl(
            context, "configuration", optimizer_winner_id, full
        )
    }
    if promoted_id != optimizer_winner_id:
        full_scores[promoted_id] = endpoint_kl(
            context, "configuration", promoted_id, full
        )
    winner_id = min(full_scores, key=full_scores.get)
    winner_spec = optimizer_winner if winner_id == optimizer_winner_id else promoted_spec
    section["winner"] = {
        "id": winner_id,
        "configuration": deepcopy(winner_spec),
        "metric": "fixed_recovery_validation_kl_t1",
        "target_tokens": full,
        "scores_at_target": full_scores,
    }
    context.persist("configuration:complete")
    print(f"[configuration] target-budget winner: {winner_id}", flush=True)
    keep = {"C0", winner_id}
    for spec in [*candidates, *objective_candidates]:
        if spec["id"] not in keep:
            cleanup_trajectory(context, spec["id"])
    return winner_spec


def run_rmsnorm(context, winning_spec):
    section = context.artifact["results"]["rmsnorm"]
    qualifier = int(context.settings["recovery"]["rmsnorm_qualifier_tokens"])
    full = int(context.settings["recovery"]["target_tokens"])
    specs = []
    for variant in context.settings["rmsnorm"]["variants"]:
        specs.append(
            {
                **deepcopy(winning_spec),
                "id": variant["id"],
                "label": variant["label"],
                "family": "rmsnorm",
                "variant": "replacement_rmsnorm",
                "rmsnorm_learning_rate_multiplier": variant["learning_rate_multiplier"],
                "configuration_source": winning_spec["id"],
            }
        )
    for spec in specs:
        run_trajectory(context, spec, qualifier)
    scores = {
        spec["id"]: endpoint_kl(context, "rmsnorm", spec["id"], qualifier)
        for spec in specs
    }
    winner_id = min(scores, key=scores.get)
    winner_spec = next(spec for spec in specs if spec["id"] == winner_id)
    section["selection"] = {
        "budget_tokens": qualifier,
        "metric": "fixed_recovery_validation_kl_t1",
        "scores": scores,
        "winner": winner_id,
    }
    context.persist("rmsnorm:selection")
    print(f"[rmsnorm] winner: {winner_id}", flush=True)
    run_trajectory(context, winner_spec, full)
    section["winner"] = {
        "id": winner_id,
        "configuration": deepcopy(winner_spec),
        "target_tokens": full,
        "recovery_validation_kl_t1_at_target": endpoint_kl(
            context, "rmsnorm", winner_id, full
        ),
    }
    context.persist("rmsnorm:complete")
    for spec in specs:
        if spec["id"] != winner_id:
            cleanup_trajectory(context, spec["id"])
    return winner_spec


def run_lora(context, winning_spec):
    section = context.artifact["results"]["lora"]
    full = int(context.settings["recovery"]["target_tokens"])
    lora = context.settings["lora"]
    specs = []
    variants = {
        "frozen_replacement_gate_up_down": "replacement_mlp_lora",
        "trainable_replacements_plus_attention_and_protected_mlp_lora": "transformer_lora_assisted",
    }
    for candidate in lora["variants"]:
        specs.append(
            {
                **deepcopy(winning_spec),
                "id": candidate["id"],
                "label": candidate["label"],
                "family": "lora",
                "variant": variants[candidate["scope"]],
                "lora_rank": int(lora["rank"]),
                "lora_alpha": float(lora["alpha"]),
                "lora_dropout": float(lora["dropout"]),
                "lora_learning_rate": float(lora["learning_rate"]),
                "configuration_source": winning_spec["id"],
            }
        )
    for spec in specs:
        run_trajectory(context, spec, full)
    scores = {
        spec["id"]: endpoint_kl(context, "lora", spec["id"], full)
        for spec in specs
    }
    section["comparison"] = {
        "budget_tokens": full,
        "metric": "fixed_recovery_validation_kl_t1",
        "scores": scores,
        "winner": min(scores, key=scores.get),
        "embeddings_and_lm_head_excluded": True,
    }
    context.persist("lora:complete")
    print(f"[lora] target-budget winner: {section['comparison']['winner']}", flush=True)
    return specs


def comparison_row(context, family, trajectory_id, role):
    trajectory = trajectory_section(context, family)[trajectory_id]
    endpoint = trajectory["endpoint"]
    wiki = endpoint["metrics"]["wikitext_validation"]
    dense = context.artifact["results"]["dense_baseline"]["wikitext_validation"]
    return {
        "role": role,
        "family": family,
        "trajectory_id": trajectory_id,
        "requested_tokens": endpoint["requested_tokens"],
        "actual_tokens": endpoint["actual_tokens"],
        "recovery_validation_kl_t1": endpoint["recovery_validation_kl_t1"],
        "wikitext_validation_loss": wiki["loss"],
        "wikitext_validation_perplexity": wiki["perplexity"],
        "perplexity_increase_over_dense": wiki["perplexity"] - dense["perplexity"],
        "trainable_parameters": trajectory["trainable_scope"]["trainable_parameters"],
        "optimizer_updates": trajectory["optimizer_updates"],
        "elapsed_seconds": trajectory["elapsed_seconds"],
        "peak_ram_gib": trajectory["memory"].get("peak_ram_gib"),
        "peak_vram_gib": trajectory["memory"].get("peak_vram_gib"),
    }


def build_final_comparison(context, winning_spec, rms_spec, lora_specs):
    source = context.artifact["results"]["source_state"]["swiglu_3_10m"]
    source_current = source["current"]
    dense = context.artifact["results"]["dense_baseline"]["wikitext_validation"]
    rows = [
        {
            "role": "swiglu3_published_baseline",
            "family": "source",
            "trajectory_id": "swiglu3_0.5_10m",
            "requested_tokens": 10000000,
            "actual_tokens": source["actual_tokens"],
            "recovery_validation_kl_t1": source_current["recovery_validation_kl"],
            "wikitext_validation_loss": source_current["wikitext_validation"]["loss"],
            "wikitext_validation_perplexity": source_current["wikitext_validation"]["perplexity"],
            "perplexity_increase_over_dense": source_current["wikitext_validation"]["perplexity"]
            - dense["perplexity"],
            "trainable_parameters": None,
            "optimizer_updates": source["optimizer_updates"],
            "elapsed_seconds": None,
            "peak_ram_gib": None,
            "peak_vram_gib": None,
        },
        comparison_row(context, "configuration", "C0", "reproduced_sw3_baseline"),
        comparison_row(
            context, "configuration", winning_spec["id"], "configuration_winner"
        ),
        comparison_row(context, "rmsnorm", rms_spec["id"], "rmsnorm_winner"),
    ]
    rows.extend(
        comparison_row(context, "lora", spec["id"], spec["label"])
        for spec in lora_specs
    )
    context.artifact["results"]["final_comparison"] = rows
    context.persist("final_comparison")


def run_workflow(context):
    load_live_resources(context)
    started = perf_counter()
    winner = run_configuration_testing(context)
    context.artifact["results"]["runtime"].append(
        {"stage": "configuration_testing", "seconds": perf_counter() - started}
    )
    started = perf_counter()
    rms_winner = run_rmsnorm(context, winner)
    context.artifact["results"]["runtime"].append(
        {"stage": "rmsnorm", "seconds": perf_counter() - started}
    )
    started = perf_counter()
    lora_specs = run_lora(context, winner)
    context.artifact["results"]["runtime"].append(
        {"stage": "lora", "seconds": perf_counter() - started}
    )
    build_final_comparison(context, winner, rms_winner, lora_specs)
    if context.scratch.is_dir():
        shutil.rmtree(context.scratch)
    context.artifact["storage"]["temporary_scratch_removed"] = True
    context.artifact["status"] = "completed"
    context.artifact["completed_at_utc"] = utc_now()
    context.persist(None)


def main():
    args = parse_args()
    settings = load_workflow_config(args.config, WORKFLOW)
    context = prepare_context(
        settings,
        args.output,
        args.source,
        args.resume,
        args.smoke,
    )
    try:
        run_workflow(context)
    except BaseException as error:
        context.artifact["status"] = "failed"
        context.artifact["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "temporary_scratch_retained_for_resume": context.scratch.exists(),
        }
        context.persist(context.current_stage)
        raise


if __name__ == "__main__":
    main()
