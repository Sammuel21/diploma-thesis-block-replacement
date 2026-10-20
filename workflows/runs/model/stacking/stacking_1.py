"""Run Stacking-1: SwiGLU replacement followed by weight-only quantization.

``run --variant ID`` executes one variant's ordered stage list from the
configuration. Order A quantizes a recovered SwiGLU-7, -8, or -9 model after
recovery (PTQ). Order B recovers SwiGLU-8's fitted token-zero start for 100M
tokens while quantization is simulated (QAT), then converts it. Quantization
of the dense model alone is a baseline and belongs to Quantization-2, not
here. Every variant is evaluated like SwiGLU-8's final evaluation plus the
shared 8K teacher KL.
``list-variants`` prints a group's variant ids in array order, ``inputs``
prints the files one variant reads for the job script to stage, and
``measure-bundle`` measures resident memory in a fresh process. The binding
design is recorded in ``docs/agents/decisions.md`` and
``docs/experiments/model/stacking/stacking-1.md``.
"""

from __future__ import annotations

import argparse
import gc
import math
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from workflows.runs.model.common import PROJECT_ROOT, resolve_path

from mlp_replacement.artifacts import content_digest, file_digest, read_json, write_json_atomic
from mlp_replacement.runlog import environment_record

from ..swiglu.swiglu_7 import (
    code_hashes,
    cpu_tree,
    desired_requests,
    group_metadata,
    load_trainable_state,
    native_task_config,
    output_path,
    parameter_names,
    persist,
    relative_output,
    resolve_storage,
    storage_preflight,
    task_files,
    trainable_state,
    unique_parameters,
    utc_now,
    validate_evaluation_preflight,
    validated_attention_implementation,
)
from ..swiglu.swiglu_8 import (
    comparison,
    enforce,
    equality_check,
    is_digest,
    monitoring_checks,
    pinned_json,
    record_failure,
    release_memory,
    start_key,
    verified_stream,
)
from ..swiglu.swiglu_9 import load_swiglu8_prepared


WORKFLOW = "stacking-1"
SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path("workflows/configs/model/stacking/stacking-1.json")
SWIGLU8_CONFIG = Path("workflows/configs/model/swiglu/swiglu-8.json")
QUANTIZATION_CONFIG = Path("workflows/configs/model/baseline/quantization-1.json")
DEFAULT_SOURCE_ROOT = Path("data/results/workflows/model")
MEASURE_MODULE = "workflows.runs.model.stacking.stacking_1"
GROUPS = ("a100m", "a1b", "b")
# Decided stage lists (decision log, 2026-10-17). Upstream stages ran in SwiGLU-7, -8,
# or -9 and are recorded for provenance only.
ORDERS = {
    "A": {"upstream": ["replace", "fit", "recover"], "stages": ["load_recovered", "pad", "ptq", "export", "evaluate"]},
    "B": {"upstream": ["replace", "fit"], "stages": ["load_start", "pad", "qat_recover", "convert", "export", "evaluate"]},
}
QUANTIZATION = {
    "bits": [8, 4, 3],
    "group_size": 128,
    "padding_multiple": 128,
    "padded_bits": [4, 3],
    "calculated_bits": [3],
    "qat_bits": [4, 3],
}
A100M_SOURCES = [
    *(f"fitted-{target}-100m" for target in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8)),
    *(f"slice-{target}-100m" for target in (0.2, 0.5, 0.8)),
]
# Fitted only: fitted versus slice-only at 1B is SwiGLU-9's question (decision log, 2026-10-17).
A1B_SOURCES = ["fitted-0.2-1b", "fitted-0.5-1b", "fitted-0.8-1b"]
B_STARTS = ["fitted-0.2", "fitted-0.5", "fitted-0.8"]
# 48 runs, stacked models only (decision log, 2026-10-20).
GRID = {
    "a100m": {"order": "A", "sources": A100M_SOURCES, "bits": [8, 4, 3]},
    "a1b": {"order": "A", "sources": A1B_SOURCES, "bits": [8, 4, 3]},
    "b": {"order": "B", "starts": B_STARTS, "bits": [4, 3]},
}


# Configuration and variants


def load_settings(path):
    """Load the configuration and check every block it shares with SwiGLU-8 and Quantization-1."""

    path = resolve_path(Path(path))
    settings = read_json(path)
    if settings.get("schema_version") != SCHEMA_VERSION or settings.get("workflow") != WORKFLOW:
        raise ValueError("Expected a schema-1 Stacking-1 configuration")
    if settings.get("seed") != 21:
        raise ValueError("Stacking-1 fixes seed 21")
    swiglu8 = read_json(resolve_path(SWIGLU8_CONFIG))
    for key in ("model", "data", "evaluation", "strategies", "recovery", "long_context"):
        if settings.get(key) != swiglu8[key]:
            raise ValueError(f"Stacking-1 must reuse SwiGLU-8's {key} contract")
    if settings.get("acceptance") != {"exact_relative": swiglu8["acceptance"]["exact_relative"]}:
        raise ValueError("Stacking-1 uses SwiGLU-8's exact token-zero tolerance")
    quantization = read_json(resolve_path(QUANTIZATION_CONFIG))
    if settings.get("backend") != quantization["backend"]:
        raise ValueError("Stacking-1 runs on Quantization-1's pinned software")
    if settings.get("scope") != {key: quantization["scope"][key]
                                 for key in ("eligible_layers", "protected_layers", "expected_projections")}:
        raise ValueError("Stacking-1 quantizes Quantization-1's 66 MLP projections in layers 1-22")
    if settings.get("quantization") != QUANTIZATION or settings.get("orders") != ORDERS:
        raise ValueError("Stacking-1 fixes the decided quantization formats and stage lists")
    if settings.get("grid") != GRID:
        raise ValueError("Stacking-1 fixes the decided run grid")
    sources = settings.get("sources", {})
    models = sources.get("models", {})
    if set(sources) != {"swiglu8_prepare_sha256", "models"} or not is_digest(sources["swiglu8_prepare_sha256"]):
        raise ValueError("Stacking-1 pins SwiGLU-8's preparation by sha256")
    if set(models) != {*A100M_SOURCES, *A1B_SOURCES}:
        raise ValueError("Stacking-1 pins exactly the models of its order-A grid")
    for name, model in models.items():
        expected_tokens = 100_000_000 if name.endswith("-100m") else 1_000_000_000
        if (
            model.get("workflow") not in ("swiglu-7", "swiglu-8", "swiglu-9")
            or name != f"{model.get('arm')}-{model.get('target')}-{name.rsplit('-', 1)[1]}"
            or int(model.get("recovery_tokens", -1)) != expected_tokens
            or not all(is_digest(model.get(key)) for key in ("result_sha256", "bundle_sha256"))
        ):
            raise ValueError(f"Source model {name} is not pinned consistently")
    return settings, path


def variant_table(settings):
    """Expand the grid into variant ids, in each group's array order."""

    table = {}
    grid = settings["grid"]
    for group in ("a100m", "a1b"):
        for source in grid[group]["sources"]:
            for bits in grid[group]["bits"]:
                table[f"a-{source}-int{bits}"] = {"group": group, "order": "A", "bits": bits, "source": source}
    for start in grid["b"]["starts"]:
        for bits in grid["b"]["bits"]:
            table[f"b-{start}-100m-int{bits}"] = {
                "group": "b", "order": "B", "bits": bits, "start": start, "reference": f"{start}-100m",
            }
    return table


def start_artifact(output_dir, settings, command, resume, identity):
    if resume:
        artifact = read_json(output_dir / "result.json")
        if (
            artifact.get("schema_version") != SCHEMA_VERSION
            or artifact.get("workflow") != WORKFLOW
            or artifact.get("command") != command
            or artifact.get("configuration") != settings
            or artifact.get("identity") != identity
        ):
            raise ValueError("Resume output belongs to a different Stacking-1 run")
        if artifact.get("status") != "completed":
            if artifact.get("environment") != environment_record():
                raise ValueError("Resume environment differs from the recorded Stacking-1 run")
            artifact["status"] = "running"
            artifact["error"] = None
        return artifact
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "experiment_family": "stacking-1",
        "experiment_class": "replacement-plus-weight-quantization",
        "command": command,
        "identity": identity,
        "status": "running",
        "created_at_utc": utc_now(),
        "configuration": deepcopy(settings),
        "environment": environment_record(),
        "results": {},
        "error": None,
    }


def start_target(spec):
    """Target of an order-B start id such as ``fitted-0.5``."""

    return float(spec["start"].split("-", 1)[1])


def source_model(settings, name):
    return settings["sources"]["models"][name]


def source_result(context, name):
    model = source_model(context.settings, name)
    result = pinned_json(context.source_root / model["workflow"], f"{model['run']}/result.json",
                         model["result_sha256"], f"{model['workflow']} {name}")
    if result.get("status") != "completed":
        raise ValueError(f"Source run {name} is not completed")
    return model, result


# Stages


def monitoring_ppl(context, model):
    from mlp_replacement.evaluation.mixed_precision import evaluate_lm_mixed

    return evaluate_lm_mixed(model, context.monitoring["model_validation"], "cuda",
                             context.settings["data"]["model_validation_batches"])


def record_input(context, model):
    """Record the BF16 input's monitoring PPL and check it against its source (K2)."""

    evaluation = context.artifact["results"].setdefault("evaluation", {"likelihood": {}, "tasks": {}})
    evaluation["monitoring_before_quantization"] = monitoring_ppl(context, model)
    reference = context.input_reference
    enforce(context.artifact, context.output_dir, "K2", [
        comparison("K2", evaluation["monitoring_before_quantization"]["perplexity"], reference["perplexity"],
                   context.settings["acceptance"]["exact_relative"], metric="monitoring_perplexity",
                   reference_source=reference["source"]),
    ])


def load_recovered(context):
    from mlp_replacement.evaluation.quantized_bundles import load_inference_bundle

    name = context.spec["source"]
    model_record, result = source_result(context, name)
    bundle = context.source_root / model_record["workflow"] / model_record["run"] / "model"
    digest = file_digest(bundle / "bundle.json")
    enforce(context.artifact, context.output_dir, "K1", [
        equality_check("K1", digest, model_record["bundle_sha256"], metric="source_bundle_manifest_sha256"),
        equality_check("K1", result["results"]["evaluation"]["model"]["manifest_sha256"], digest,
                       metric="source_result_references_bundle"),
    ])
    context.model, context.tokenizer, manifest = load_inference_bundle(bundle, attention_implementation="sdpa")
    validated_attention_implementation(context.model)
    if any(row["down_bias"] for row in manifest["replacements"]):
        raise ValueError("Stacking-1 source replacements must be bias-free")
    context.replacements = [{"path": row["path"], "width": int(row["width"])} for row in manifest["replacements"]]
    context.artifact["results"]["source"] = {
        "name": name, **model_record, "training_tokens": int(result["results"]["recovery"]["tokens_seen"]),
    }
    context.input_reference = {
        "perplexity": result["results"]["evaluation"]["monitoring_after_bf16"]["perplexity"],
        "source": f"{model_record['workflow']} {model_record['run']} monitoring_after_bf16",
    }
    record_input(context, context.model)


def load_teacher_and_validation(context):
    """Load the teacher and its cached monitoring logits once per QAT run."""

    import torch
    from mlp_replacement.compression.teacher_cache import cache_teacher_logits
    from mlp_replacement.config import make_model_config
    from mlp_replacement.model import load_model_and_tokenizer

    if getattr(context, "teacher", None) is None:
        context.teacher, tokenizer = load_model_and_tokenizer(make_model_config(context.settings["model"]))
        context.tokenizer = context.tokenizer or tokenizer
        validated_attention_implementation(context.teacher)
        context.validation_cache = cache_teacher_logits(
            context.teacher, context.monitoring["recovery_validation"],
            len(context.monitoring["recovery_validation"]), torch.device("cuda"), "float16",
        )


def monitoring_row(context, tokens, requested, updates, seconds, mean_train_kl, learning_rates, wikitext):
    """One SwiGLU-8 monitoring row; RNG is restored so evaluation never perturbs training."""

    from mlp_replacement.compression.continuation import capture_rng, restore_rng
    from mlp_replacement.evaluation.mixed_precision import evaluate_validation_kl_mixed

    rng = capture_rng()
    row = {
        "actual_tokens": tokens,
        "requested_tokens": list(requested),
        "optimizer_updates": updates,
        "training_seconds": seconds,
        "mean_train_kl": mean_train_kl,
        "recovery_validation_kl": evaluate_validation_kl_mixed(context.model, context.validation_cache, 1.0, "cuda"),
        "learning_rates": list(learning_rates),
    }
    if wikitext:
        row["wikitext_validation"] = monitoring_ppl(context, context.model)
    restore_rng(rng)
    return row


def load_start(context):
    """Load SwiGLU-8's fitted token-zero start and check it reproduces SwiGLU-8 (K3)."""

    import torch
    from mlp_replacement.compression.continuation import restore_rng
    from mlp_replacement.compression.reconstruction import build_swiglu_student, load_replacement_state
    from mlp_replacement.config import make_model_config

    settings = context.settings
    key = start_key("fitted", start_target(context.spec))
    start_record = context.prepared["results"]["starts"][key]
    if (
        start_record.get("arm") != "fitted"
        or int(start_record.get("tokens_seen", -1)) != 0
        or int(start_record.get("optimizer_updates", -1)) != 0
    ):
        raise ValueError("Order B requires SwiGLU-8's fresh fitted token-zero start")
    start_path = output_path(context.prepared_root, start_record["path"])
    if file_digest(start_path) != start_record["sha256"]:
        raise ValueError("Prepared starting state changed")
    start = torch.load(start_path, map_location="cpu", weights_only=False)
    if (
        int(start.get("tokens_seen", -1)) != 0
        or "optimizer_state" in start
        or start.get("arm") != "fitted"
        or start.get("recipe_fingerprint") != start_record["recipe_fingerprint"]
        or start.get("selection_fingerprint") != start_record["selection"]["fingerprint"]
    ):
        raise ValueError("Prepared start payload differs from its descriptor")
    unused_model, reference = source_result(context, context.spec["reference"])
    load_teacher_and_validation(context)
    context.model, target_paths, unused_modules = build_swiglu_student(
        make_model_config(settings["model"]), settings["model"]["hidden_size"], start_record["allocation"]
    )
    validated_attention_implementation(context.model)
    load_replacement_state(context.model, start["replacement_state"])
    restore_rng(start)
    del start, unused_modules
    context.replacements = [
        {"path": path, "width": int(row["replacement_width"])}
        for path, row in zip(target_paths, (row for row in start_record["allocation"]
                                            if not row.get("retains_dense_module")))
    ]
    context.artifact["results"]["source"] = {
        "name": context.spec["start"], "start_sha256": start_record["sha256"], "selection": start_record["selection"],
        "training_tokens": 0,
    }
    # K3: the start before padding and fake quantization equals SwiGLU-8's token-zero row.
    row = monitoring_row(context, 0, [0], 0, 0.0, None, [settings["recovery"]["learning_rate"]], True)
    context.artifact["results"]["start_monitoring"] = row
    checks = [{**check, "check": "K3"} for check in monitoring_checks(
        row, reference["results"]["recovery"]["validation_history"], settings["acceptance"])]
    enforce(context.artifact, context.output_dir, "K3", checks)


def pad_stage(context):
    from mlp_replacement.compression.quantization import pad_replacements

    bits = context.spec["bits"]
    paths = [row["path"] for row in context.replacements]
    if bits in context.settings["quantization"]["padded_bits"]:
        context.padding = pad_replacements(context.model, paths, context.settings["quantization"]["padding_multiple"])
    else:
        context.padding = [{"path": row["path"], "real_width": row["width"], "width": row["width"]}
                           for row in context.replacements]
    hidden = int(context.settings["model"]["hidden_size"])
    padded = [row for row in context.padding if row["width"] != row["real_width"]]
    summary = {"rows": context.padding, "padded_replacements": len(padded),
               "padding_weights": sum(3 * hidden * (row["width"] - row["real_width"]) for row in padded)}
    if padded:
        # Recorded, not enforced: a GEMM over a longer row may sum in a different order.
        summary["monitoring_after_padding"] = monitoring_ppl(context, context.model)
    context.artifact["results"]["padding"] = summary
    persist(context.output_dir, context.artifact, "padded")


def eligible_paths(context):
    from mlp_replacement.compression.quantization import projection_paths

    paths = projection_paths(context.model, context.settings["scope"]["eligible_layers"])
    if len(paths) != context.settings["scope"]["expected_projections"]:
        raise ValueError("Eligible projection count changed")
    return paths


def bytes_check(context, measured_eligible):
    """K4: for INT8 and INT4, the storage arithmetic must equal the measured storage."""

    bits = context.spec["bits"]
    if bits in context.settings["quantization"]["calculated_bits"]:
        return
    enforce(context.artifact, context.output_dir, "K4", [
        equality_check("K4", context.calculated["eligible_weight_bytes"], measured_eligible,
                       metric="calculated_equals_measured_eligible_bytes", bits=bits),
    ])


def ptq_stage(context):
    from mlp_replacement.compression.quantization import apply_ptq, calculated_weight_bytes, require_backend

    bits = context.spec["bits"]
    require_backend(context.settings["backend"], bits)
    context.paths = eligible_paths(context)
    context.calculated = calculated_weight_bytes(context.model, context.paths, bits)
    started = perf_counter()
    conversion = apply_ptq(context.model, context.paths, bits)
    context.artifact["results"]["conversion"] = {**conversion, "seconds": perf_counter() - started}
    bytes_check(context, conversion["after"]["eligible_weight_storage_bytes"])


def qat_groups(context, trainable):
    return [{
        "name": "replacements",
        "parameters": trainable,
        "parameter_names": parameter_names(context.model, trainable),
        "parameter_count": sum(value.numel() for value in trainable),
        "learning_rate": context.settings["recovery"]["learning_rate"],
        "weight_decay": context.settings["recovery"]["weight_decay"],
    }]


def qat_recover_stage(context):
    """Recover for 100M tokens with fake quantization, on SwiGLU-8's stream and recipe."""

    import torch
    from mlp_replacement.compression.continuation import (
        capture_rng, commit_single_checkpoint, recover_exact_segment, restore_checkpoint, restore_rng,
        segment_origin, segment_schedule,
    )
    from mlp_replacement.compression.quantization import (
        calculated_weight_bytes, mask_padded_gradients, padded_weights_are_zero, prepare_qat, require_backend,
    )
    from mlp_replacement.data import PackedTokenCache
    from workflows.runs.model.swiglu.shared import recovery_memory_record

    settings, artifact, output_dir = context.settings, context.artifact, context.output_dir
    bits = context.spec["bits"]
    recovery = settings["recovery"]
    require_backend(settings["backend"], bits)
    load_teacher_and_validation(context)
    context.paths = eligible_paths(context)
    context.calculated = calculated_weight_bytes(context.model, context.paths, bits)
    replaced = {row["path"] for row in context.replacements}
    trainable_paths = [path for path in context.paths if path.rsplit(".", 1)[0] in replaced]
    # S7-0 scope: every parameter of the replacement modules, and nothing else.
    modules = [context.model.get_submodule(path) for path in sorted(replaced)]
    scope = {*parameter_names(context.model, unique_parameters(modules))}
    if scope != {f"{path}.weight" for path in trainable_paths}:
        raise ValueError("Order B must train exactly SwiGLU-8's S7-0 replacement scope")
    trainable = prepare_qat(context.model, context.paths, bits, trainable_paths)
    hooks = mask_padded_gradients(context.model, getattr(context, "padding", []))
    groups = qat_groups(context, trainable)
    artifact["results"]["trainable_scope"] = group_metadata(groups, context.model)
    artifact["results"]["storage_preflight"] = storage_preflight(
        output_dir, groups, context.model, recovery["disk_reserve_fraction"])
    stream = context.prepared["results"]["stream"]
    if int(stream["token_count"]) != int(recovery["target_tokens"]):
        raise ValueError("SwiGLU-8's prepared stream does not cover the 100M-token recipe")
    stream_path = verified_stream(context.prepared_root, stream)
    artifact["recovery_stream"] = {"sha256": stream["sha256"], "token_count": int(stream["token_count"]),
                                   "source": "swiglu-8 prepare-001"}
    cache = PackedTokenCache(stream_path, int(stream["token_count"]), recovery["sequence_length"], stream["sha256"])
    checkpoint_dir = output_dir / "checkpoint"
    results = artifact["results"].setdefault("recovery", {
        "validation_history": [], "tokens_seen": 0, "optimizer_updates": 0,
        "training_seconds": 0.0, "evaluation_seconds": 0.0, "checkpoint_seconds": 0.0,
    })

    def payload(tokens, updates, optimizer_state, rng):
        return {
            "schema_version": SCHEMA_VERSION, "workflow": WORKFLOW, "run_fingerprint": context.run_fingerprint,
            "stream_sha256": stream["sha256"], "variant": context.variant, "tokens_seen": tokens,
            "optimizer_updates": updates, "trainable_state": trainable_state(context.model, groups),
            "optimizer_state": optimizer_state, "recovery": deepcopy(artifact["results"]["recovery"]),
            "fake_quantization_enabled": True, **rng,
        }

    recovered = None
    if context.resume:
        try:
            recovered, descriptor = restore_checkpoint(checkpoint_dir, context.run_fingerprint)
        except FileNotFoundError:
            if int(results["tokens_seen"]) > 0:
                raise
        if recovered is not None:
            if recovered.get("fake_quantization_enabled") is not True or recovered["stream_sha256"] != stream["sha256"]:
                raise ValueError("Checkpoint does not continue this fake-quantized run")
            load_trainable_state(context.model, recovered["trainable_state"], groups)
            artifact["results"]["recovery"] = results = deepcopy(recovered["recovery"])
            artifact["checkpoint"] = descriptor
            restore_rng(recovered)
    if recovered is None:
        optimizer_state = None
        results.update({"tokens_seen": 0, "optimizer_updates": 0})
        if not results["validation_history"]:
            results["validation_history"].append(monitoring_row(
                context, 0, [0], 0, 0.0, None, [group["learning_rate"] for group in groups], True))
        artifact["checkpoint"] = commit_single_checkpoint(checkpoint_dir, payload(0, 0, None, capture_rng()))
        persist(output_dir, artifact, "recovery_initialized")
    else:
        optimizer_state = recovered["optimizer_state"]
    del recovered
    cursor, updates = int(results["tokens_seen"]), int(results["optimizer_updates"])
    final_tokens = int(recovery["target_tokens"])
    while cursor < final_tokens:
        origin = segment_origin(cursor, recovery["segment_endpoints"])
        end = next(value for value in recovery["segment_endpoints"] if value > cursor)
        schedule = segment_schedule(origin, end, desired_requests(settings), recovery["effective_batch_tokens"])

        def on_checkpoint(event, optimizer, first_step):
            started = perf_counter()
            rng = capture_rng()
            wikitext = any(point in recovery["monitoring_ppl_tokens"] for point in event.requested_checkpoint_tokens)
            row = monitoring_row(context, event.tokens_seen, event.requested_checkpoint_tokens,
                                 event.optimizer_updates, event.elapsed_seconds, event.mean_train_kl,
                                 event.learning_rates, wikitext)
            if not math.isfinite(row["recovery_validation_kl"]) or not math.isfinite(event.mean_train_kl):
                raise FloatingPointError("Non-finite recovery loss; invalid state was not committed")
            current = artifact["results"]["recovery"]
            current["validation_history"].append(row)
            current.update({
                "tokens_seen": event.tokens_seen, "optimizer_updates": event.optimizer_updates,
                "training_seconds": event.elapsed_seconds, "first_step": current.get("first_step") or first_step,
                "memory": recovery_memory_record(torch.device("cuda")),
            })
            current["evaluation_seconds"] += perf_counter() - started
            checkpoint_started = perf_counter()
            artifact["checkpoint"] = commit_single_checkpoint(checkpoint_dir, payload(
                event.tokens_seen, event.optimizer_updates, cpu_tree(optimizer.state_dict()), rng))
            current["checkpoint_seconds"] += perf_counter() - checkpoint_started
            persist(output_dir, artifact, "recovery",
                    {"variant": context.variant, "tokens_seen": event.tokens_seen, "target_tokens": final_tokens})
            print(f"variant={context.variant} tokens={event.tokens_seen:,} KL={row['recovery_validation_kl']:.6f}",
                  flush=True)

        result = recover_exact_segment(
            origin=origin, end=end, cursor=cursor, schedule=schedule,
            student=context.model, teacher=context.teacher,
            parameter_groups=[{key: group[key] for key in ("name", "parameters", "learning_rate", "weight_decay")}
                              for group in groups],
            train_modules=[context.model.get_submodule(path) for path in trainable_paths],
            batch_at=lambda offset, count: cache.batch(offset, count, recovery["microbatch_sequences"],
                                                       allow_partial_sequence=True),
            microbatch_tokens=recovery["sequence_length"] * recovery["microbatch_sequences"],
            accumulation_steps=recovery["gradient_accumulation_steps"],
            temperature=recovery["temperature"], ce_weight=recovery["ce_weight"],
            scheduler=recovery["scheduler"], warmup_fraction=recovery["warmup_fraction"],
            final_lr_ratio=recovery["final_lr_ratio"], device=torch.device("cuda"),
            autocast_dtype=torch.bfloat16, start_updates=updates,
            elapsed_seconds=artifact["results"]["recovery"]["training_seconds"],
            optimizer_state=optimizer_state, on_checkpoint=on_checkpoint,
            optimizer_backend=recovery["optimizer_backend"],
        )
        cursor, updates = result.tokens_seen, result.optimizer_updates
        if cursor < final_tokens:
            state, unused_descriptor = restore_checkpoint(checkpoint_dir, context.run_fingerprint)
            optimizer_state = state["optimizer_state"]
            restore_rng(state)
            del state
            gc.collect()
    for handle in hooks:
        handle.remove()
    if any(row["width"] != row["real_width"] for row in getattr(context, "padding", [])):
        enforce(artifact, output_dir, "K5-training", [equality_check(
            "K5", padded_weights_are_zero(context.model, context.padding), True,
            metric="padded_master_weights_zero_after_training")])
    artifact["results"]["recovery"]["selection"] = "configured-final-endpoint"
    del context.teacher, context.validation_cache
    context.teacher = None
    groups.clear()
    trainable.clear()
    release_memory()
    persist(output_dir, artifact, "recovered")


def convert_stage(context):
    from mlp_replacement.compression.quantization import convert_qat, quantization_footprint

    started = perf_counter()
    conversion = convert_qat(context.model, context.paths, context.spec["bits"])
    context.artifact["results"]["conversion"] = {**conversion, "seconds": perf_counter() - started}
    bytes_check(context, quantization_footprint(context.model, context.paths)["eligible_weight_storage_bytes"])


def export_stage(context):
    from mlp_replacement.evaluation.quantized_bundles import export_quantized_bundle

    padding = getattr(context, "padding", [])
    replacements = [{"path": row["path"], "width": row["width"], "real_width": row["real_width"],
                     "down_bias": False} for row in padding]
    provenance = {
        "workflow": WORKFLOW, "run_fingerprint": context.run_fingerprint, "model": context.settings["model"],
        "scope": context.settings["scope"], "variant": context.variant, "order": context.spec["order"],
        "stages": context.settings["orders"][context.spec["order"]],
        "source": context.artifact["results"]["source"],
        "training_tokens": int(context.artifact["results"]["source"]["training_tokens"])
        + int(context.artifact["results"].get("recovery", {}).get("tokens_seen", 0)),
    }
    started = perf_counter()
    manifest = export_quantized_bundle(context.model, context.tokenizer, context.output_dir / "model",
                                       context.paths, context.spec["bits"], context.settings["backend"],
                                       provenance, replacements=replacements, calculated_bytes=context.calculated)
    context.artifact["results"]["model"] = {
        "path": "model", "manifest_sha256": file_digest(context.output_dir / "model" / "bundle.json"),
        "export_seconds": perf_counter() - started, "tensor_file_bytes": manifest["tensor_file_bytes"],
    }
    context.model = None
    release_memory()
    persist(context.output_dir, context.artifact, "bundle_exported")


def padded_output_check(context, model, padding):
    """K5: appended neurons of the deployed model output exactly zero gate and up values."""

    import torch
    from mlp_replacement.evaluation.mixed_precision import autocast_context

    padded = [row for row in padding if row["width"] != row["real_width"]]
    if not padded:
        return
    observed = []
    handles = []
    for row in padded:
        module = model.get_submodule(row["path"])
        for name in ("gate_projection", "up_projection"):
            def hook(unused_module, unused_inputs, output, row=row, name=name):
                observed.append((row["path"], name, bool(output[..., row["real_width"]:].eq(0).all().item())))
            handles.append(getattr(module, name).register_forward_hook(hook))
    batch = next(iter(context.monitoring["model_validation"]))
    try:
        with torch.no_grad(), autocast_context("cuda"):
            model(input_ids=batch["input_ids"].to("cuda"), attention_mask=batch["attention_mask"].to("cuda"),
                  use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    enforce(context.artifact, context.output_dir, "K5", [
        equality_check("K5", zero, True, metric="padded_outputs_exactly_zero", path=path, projection=name)
        for path, name, zero in observed
    ])


def footprint_record(context, manifest):
    """Measured and calculated bytes, and savings against the dense BF16 model."""

    settings = context.settings
    bits = manifest["bits"]
    hidden, intermediate = int(settings["model"]["hidden_size"]), int(settings["model"]["intermediate_size"])
    dense = context.prepared["results"]["dense_evaluation"]["footprint"]
    dense_bytes = int(dense["parameter_bytes"])
    dense_eligible_bytes = 2 * 3 * hidden * intermediate * len(settings["scope"]["eligible_layers"])
    measured, calculated = manifest["footprint"], manifest["calculated_bytes"]
    padding = context.artifact["results"].get("padding", {"padding_weights": 0})
    basis = "calculated" if bits in settings["quantization"]["calculated_bits"] else "measured"
    frontier = calculated if basis == "calculated" else {
        "weight_bytes": measured["weight_storage_bytes"], "eligible_weight_bytes": measured["eligible_weight_storage_bytes"]}
    bits_per_weight = {8: None, 4: 4.25, 3: 3.25}[bits]
    return {
        "measured": measured,
        "calculated": calculated,
        "measured_storage_note": "INT3 is stored one value per int8; its measured bytes are not its size"
        if bits == 3 else None,
        "logical_parameters": measured["logical_parameters"],
        "real_logical_parameters": measured["logical_parameters"] - padding["padding_weights"],
        "padding_weights": padding["padding_weights"],
        "padding_bytes": padding["padding_weights"] * bits_per_weight / 8 if bits_per_weight else 0.0,
        "dense_weight_bytes": dense_bytes,
        "dense_eligible_weight_bytes": dense_eligible_bytes,
        "frontier_basis": basis,
        "frontier_weight_bytes": frontier["weight_bytes"],
        "whole_model_byte_saving": 1.0 - frontier["weight_bytes"] / dense_bytes,
        "eligible_mlp_byte_saving": 1.0 - frontier["eligible_weight_bytes"] / dense_eligible_bytes,
        "whole_model_parameter_removal": 1.0 - (measured["logical_parameters"] - padding["padding_weights"])
        / int(dense["parameters"]),
    }


def evaluate_stage(context):
    """SwiGLU-8's final evaluation (``swiglu_7.evaluate_final_model``) plus the 8K KL, on the packed bundle."""

    import numpy as np
    from importlib.metadata import version
    from mlp_replacement.config import make_model_config
    from mlp_replacement.evaluation.final_quality import (
        evaluate_pinned_task, evaluate_rolling_likelihood, paired_accuracy_difference,
    )
    from mlp_replacement.evaluation.quantized_bundles import load_quantized_bundle, validate_quantized_bundle
    from mlp_replacement.evaluation.teacher_kl import evaluate_teacher_kl
    from mlp_replacement.model import load_model_and_tokenizer

    settings, artifact, output_dir, prepared, prepared_root = (
        context.settings, context.artifact, context.output_dir, context.prepared, context.prepared_root)
    evaluation = artifact["results"].setdefault("evaluation", {"likelihood": {}, "tasks": {}})
    bundle_path = output_dir / "model"
    manifest = validate_quantized_bundle(bundle_path)
    if manifest["provenance"]["run_fingerprint"] != context.run_fingerprint:
        raise ValueError("Existing bundle belongs to another run")
    evaluation["model"] = {"path": relative_output(output_dir, bundle_path),
                           "manifest_sha256": file_digest(bundle_path / "bundle.json")}
    evaluation["footprint"] = {
        **footprint_record(context, manifest), "tensor_file_bytes": manifest["tensor_file_bytes"],
        "bundle_bytes": sum(path.stat().st_size for path in bundle_path.rglob("*") if path.is_file()),
    }
    if "resident_memory" not in evaluation:
        measurement_path = context.work_dir / "resident-memory.json"
        subprocess.run([sys.executable, "-m", MEASURE_MODULE, "measure-bundle", "--bundle", str(bundle_path),
                        "--output", str(measurement_path)], cwd=PROJECT_ROOT, check=True)
        evaluation["resident_memory"] = read_json(measurement_path)
        if evaluation["resident_memory"]["bundle_manifest_sha256"] != evaluation["model"]["manifest_sha256"]:
            raise ValueError("Resident-memory measurement belongs to another bundle")
        persist(output_dir, artifact, "resident_memory")
    model, tokenizer, manifest = load_quantized_bundle(bundle_path, attention_implementation="sdpa")
    try:
        padded_output_check(context, model, manifest.get("replacements", []))
        evaluation["attention_implementation"] = validated_attention_implementation(model)
        if int(model.config.max_position_embeddings) < max(settings["evaluation"]["contexts"]):
            raise ValueError("Final model does not support the configured evaluation context")
        if "monitoring_after_quantization" not in evaluation:
            evaluation["monitoring_after_quantization"] = monitoring_ppl(context, model)
            after = evaluation["monitoring_after_quantization"]["perplexity"]
            if "recovery" in artifact["results"]:
                # QAT: like SwiGLU-8's BF16 conversion delta, against the last fake-quantized row.
                fake = artifact["results"]["recovery"]["validation_history"][-1]["wikitext_validation"]
                evaluation["conversion_ppl_delta"] = after - fake["perplexity"]
            else:
                evaluation["quantization_ppl_delta"] = (
                    after - evaluation["monitoring_before_quantization"]["perplexity"])
            persist(output_dir, artifact, "monitoring_evaluation")
        protocol = prepared["results"]["evaluation_protocol"]
        if (version("lm_eval") != settings["evaluation"]["harness_version"]
                or task_files() != protocol["harness_source_hashes"]):
            raise ValueError("Evaluation harness version or source differs from the frozen protocol")
        for split, record in protocol["corpora"].items():
            path = output_path(prepared_root, record["path"])
            if file_digest(path) != record["sha256"]:
                raise ValueError(f"Prepared WikiText {split} tokens changed")
            token_ids = np.memmap(path, mode="r", dtype=np.int32)
            if len(token_ids) != int(record["token_count"]):
                raise ValueError(f"Prepared WikiText {split} extent changed")
            for context_length, stride in zip(settings["evaluation"]["contexts"], settings["evaluation"]["strides"],
                                              strict=True):
                key = f"{split}-{context_length}"
                if key not in evaluation["likelihood"]:
                    evaluation["likelihood"][key] = evaluate_rolling_likelihood(
                        model, token_ids, context_length, stride, "cuda")
                    persist(output_dir, artifact, f"likelihood:{key}")
        dense = prepared["results"]["dense_evaluation"]
        evaluation["likelihood_comparison_to_dense"] = {
            key: {
                "student_loss": metrics["loss"], "dense_loss": dense["likelihood"][key]["loss"],
                "student_minus_dense_loss": metrics["loss"] - dense["likelihood"][key]["loss"],
                "student_perplexity": metrics["perplexity"], "dense_perplexity": dense["likelihood"][key]["perplexity"],
                "student_minus_dense_perplexity": metrics["perplexity"] - dense["likelihood"][key]["perplexity"],
            }
            for key, metrics in evaluation["likelihood"].items()
        }
        task_comparisons = {}
        for task in settings["evaluation"]["tasks"]:
            record = evaluation["tasks"].get(task)
            if record is None:
                config = native_task_config(task)
                config["dataset_kwargs"] = {**(config.get("dataset_kwargs") or {}),
                                            "revision": protocol["tasks"][task]["revision"]}
                config["num_fewshot"] = 0
                task_result = evaluate_pinned_task(model, tokenizer, task, settings["evaluation"], config)
                task_path = output_dir / "evaluation" / "benchmarks" / f"{task}.json"
                write_json_atomic(task_path, task_result)
                record = {"path": relative_output(output_dir, task_path), "sha256": file_digest(task_path)}
                evaluation["tasks"][task] = record
                persist(output_dir, artifact, f"benchmark:{task}")
            student_path = output_path(output_dir, record["path"])
            if file_digest(student_path) != record["sha256"]:
                raise ValueError(f"Student benchmark result changed: {task}")
            dense_record = dense["tasks"][task]
            dense_path = output_path(prepared_root, dense_record["path"])
            if file_digest(dense_path) != dense_record["sha256"]:
                raise ValueError(f"Dense benchmark result changed: {task}")
            student_task, dense_task = read_json(student_path), read_json(dense_path)
            metric = settings["evaluation"]["primary_metrics"][task]
            task_comparisons[task] = {
                "dense": dense_task["results"][task][f"{metric},none"],
                "student": student_task["results"][task][f"{metric},none"],
                **paired_accuracy_difference(dense_task["samples"][task], student_task["samples"][task], metric,
                                             settings["evaluation"]["bootstrap_resamples"], settings["seed"]),
            }
        evaluation["comparison_to_dense"] = {
            "tasks": task_comparisons,
            "macro_accuracy": sum(row["student"] for row in task_comparisons.values()) / len(task_comparisons),
            "macro_delta": sum(row["student_minus_dense"] for row in task_comparisons.values()) / len(task_comparisons),
        }
        persist(output_dir, artifact, "benchmarks_compared")
        if "teacher_kl_8192" not in evaluation:
            kl_record = prepared["results"]["long_context_data"]["kl_stream"]
            kl_path = verified_stream(prepared_root, kl_record)
            teacher, unused_tokenizer = load_model_and_tokenizer(make_model_config(settings["model"]))
            started = perf_counter()
            try:
                result = evaluate_teacher_kl(model, teacher, kl_path, kl_record,
                                             int(settings["long_context"]["sequence_length"]))
            finally:
                del teacher, unused_tokenizer
                release_memory()
            evaluation["teacher_kl_8192"] = {**result, "seconds": perf_counter() - started}
            persist(output_dir, artifact, "final_teacher_kl")
        evaluation["status"] = "completed"
    finally:
        del model, tokenizer
        release_memory()


STAGES = {
    "load_recovered": load_recovered,
    "load_start": load_start,
    "pad": pad_stage,
    "ptq": ptq_stage,
    "qat_recover": qat_recover_stage,
    "convert": convert_stage,
    "export": export_stage,
    "evaluate": evaluate_stage,
}


# Commands


def existing_bundle(artifact, output_dir, run_fingerprint):
    """Return True when an earlier attempt already exported this run's bundle."""

    from mlp_replacement.evaluation.quantized_bundles import validate_quantized_bundle

    bundle, staging = output_dir / "model", output_dir / "model.building"
    if not bundle.exists():
        if staging.exists():
            if staging.resolve().parent != output_dir.resolve():
                raise ValueError("Incomplete bundle staging path escaped the output directory")
            shutil.rmtree(staging)
        return False
    manifest = validate_quantized_bundle(bundle)
    if manifest["provenance"]["run_fingerprint"] != run_fingerprint:
        raise ValueError("Existing bundle belongs to another run")
    artifact["results"].setdefault("model", {"path": "model",
                                             "manifest_sha256": file_digest(bundle / "bundle.json")})
    return True


def run(args, settings, config_path):
    import torch

    table = variant_table(settings)
    if args.variant not in table:
        raise ValueError(f"Unknown Stacking-1 variant: {args.variant}")
    spec = table[args.variant]
    work_dir, output_dir = resolve_storage(args.work_dir, args.output_dir, args.resume)
    identity = {"variant": args.variant, **spec}
    artifact = start_artifact(output_dir, settings, "run", args.resume, identity)
    try:
        prepared_path, prepared, prepared_root = load_swiglu8_prepared(args.prepared, settings)
        validate_evaluation_preflight(prepared, prepared_root, settings)
        source_root = resolve_path(Path(args.source_root))
        if spec["order"] == "A":
            pins = source_model(settings, spec["source"])
        else:
            pins = {"start_sha256": prepared["results"]["starts"][start_key("fitted", start_target(spec))]["sha256"],
                    "reference": source_model(settings, spec["reference"])}
        contract = {
            "configuration": settings,
            "configuration_sha256": file_digest(config_path),
            "prepared_sha256": file_digest(prepared_path),
            "identity": identity,
            "source": pins,
            "code_hashes": code_hashes(),
        }
        run_fingerprint = content_digest(contract)
        if args.resume and artifact.get("run_fingerprint") != run_fingerprint:
            raise ValueError("Resume inputs, configuration, or maintained code changed")
        if artifact.get("status") == "completed":
            from mlp_replacement.evaluation.quantized_bundles import validate_quantized_bundle

            validate_quantized_bundle(output_dir / "model")
            print(f"Already completed: {output_dir}")
            return
        artifact.update(run_fingerprint=run_fingerprint, code_hashes=contract["code_hashes"],
                        configuration_sha256=contract["configuration_sha256"],
                        prepared={"sha256": contract["prepared_sha256"]}, status="running", error=None)
        artifact["results"]["stages"] = settings["orders"][spec["order"]]
        persist(output_dir, artifact, "loading_inputs")
        monitoring_record = prepared["results"]["monitoring_evaluation"]
        monitoring_path = output_path(prepared_root, monitoring_record["path"])
        if file_digest(monitoring_path) != monitoring_record["sha256"]:
            raise ValueError("Prepared monitoring batches changed")
        context = SimpleNamespace(
            settings=settings, variant=args.variant, spec=spec, artifact=artifact, output_dir=output_dir,
            work_dir=work_dir, prepared=prepared, prepared_root=prepared_root, source_root=source_root,
            run_fingerprint=run_fingerprint, resume=args.resume, model=None, tokenizer=None, teacher=None,
            monitoring=torch.load(monitoring_path, map_location="cpu", weights_only=False),
        )
        stages = list(settings["orders"][spec["order"]]["stages"])
        if existing_bundle(artifact, output_dir, run_fingerprint):
            stages = ["evaluate"]
        timings = artifact["results"].setdefault("stage_seconds", {})
        for stage in stages:
            persist(output_dir, artifact, f"stage:{stage}")
            started = perf_counter()
            STAGES[stage](context)
            timings[stage] = timings.get(stage, 0.0) + perf_counter() - started
        artifact["status"] = "completed"
        persist(output_dir, artifact, "validating_final_artifacts")
        completed = read_json(output_dir / "result.json")
        if completed.get("status") != "completed" or completed.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError("Completed result did not survive its JSON round trip")
        if completed["results"]["evaluation"]["model"]["manifest_sha256"] != file_digest(
            output_dir / "model" / "bundle.json"
        ):
            raise RuntimeError("Completed result does not reference the final bundle manifest")
        checkpoint_dir = output_dir / "checkpoint"
        for path in list(checkpoint_dir.glob("checkpoint-*")) if checkpoint_dir.exists() else []:
            path.unlink()
        if checkpoint_dir.exists():
            checkpoint_dir.rmdir()
        artifact.pop("checkpoint", None)
        persist(output_dir, artifact, "completed")
    except BaseException as error:
        record_failure(output_dir, work_dir, artifact, error)
        raise


def list_variants(args, settings):
    for variant, spec in variant_table(settings).items():
        if spec["group"] == args.group:
            print(variant)


def variant_inputs(settings, spec, prepared):
    """Name the files one variant reads, relative to the source root or SwiGLU-8's preparation."""

    results = prepared["results"]
    lines = [f"prepared {path}" for path in (
        *(record["path"] for record in results["evaluation_protocol"]["corpora"].values()),
        *(record["path"] for record in results["dense_evaluation"]["tasks"].values()),
        results["monitoring_evaluation"]["path"],
        results["long_context_data"]["kl_stream"]["path"],
    )]
    if spec["order"] == "A":
        model = source_model(settings, spec["source"])
        lines.append(f"source {model['workflow']}/{model['run']}")
    else:
        model = source_model(settings, spec["reference"])
        lines.append(f"reference {model['workflow']}/{model['run']}/result.json")
        lines.append(f"prepared {results['starts'][start_key('fitted', start_target(spec))]['path']}")
        lines.append(f"prepared {results['stream']['path']}")
    return lines


def print_inputs(args, settings):
    """Print the staging list for a job script; reads only SwiGLU-8's preparation record."""

    spec = variant_table(settings)[args.variant]
    for line in variant_inputs(settings, spec, read_json(resolve_path(args.prepared))):
        print(line)


def measure_bundle(args):
    from mlp_replacement.evaluation.bundles import measure_resident_bundle
    from mlp_replacement.evaluation.quantized_bundles import load_quantized_bundle

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    measurement = measure_resident_bundle(Path(args.bundle), loader=load_quantized_bundle)
    measurement["bundle_manifest_sha256"] = file_digest(Path(args.bundle) / "bundle.json")
    write_json_atomic(output, measurement)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_parser.add_argument("--variant", required=True)
    run_parser.add_argument("--prepared", type=Path, required=True,
                            help="SwiGLU-8 prepare-001/result.json")
    run_parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT,
                            help="Directory holding the swiglu-7, swiglu-8, and swiglu-9 run outputs")
    run_parser.add_argument("--work-dir", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--resume", action="store_true")
    listing = subparsers.add_parser("list-variants")
    listing.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    listing.add_argument("--group", choices=GROUPS, required=True)
    inputs = subparsers.add_parser("inputs")
    inputs.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    inputs.add_argument("--variant", required=True)
    inputs.add_argument("--prepared", type=Path, required=True, help="SwiGLU-8 prepare-001/result.json")
    measure = subparsers.add_parser("measure-bundle")
    measure.add_argument("--bundle", type=Path, required=True)
    measure.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "measure-bundle":
        measure_bundle(args)
        return
    settings, config_path = load_settings(args.config)
    if args.command == "list-variants":
        list_variants(args, settings)
    elif args.command == "inputs":
        print_inputs(args, settings)
    else:
        run(args, settings, config_path)


if __name__ == "__main__":
    main()
