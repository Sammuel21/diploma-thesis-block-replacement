"""Run Recovery-1: the recovery schedule screen, constant against 1-sqrt decays.

``train --task-index i`` runs one family: dense INT3 QAT, B 20% INT4 (order B), or
SwiGLU alone at 20% in BF16. It rebuilds the start of the family's recorded constant
run and stops unless token 0 reproduces that run. It then trains a constant trunk to
100M tokens, keeping the trainable and optimizer state at each decay start, and
afterwards restores each state and decays it to 100M with the 1-sqrt schedule. A
decay therefore differs from the trunk only after its start. Every row is scored on
the 8K selection stream, which decides, and on the family's old history metric,
which checks the trunk against its recorded run. ``summarize`` applies the winner
rule on the login node. The binding design is in ``docs/agents/decisions.md``
(2026-11-07) and ``docs/experiments/model/recovery/recovery-1.md``.
"""

from __future__ import annotations

import argparse
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from workflows.runs.model.common import resolve_path

from mlp_replacement.artifacts import file_digest, read_json
from mlp_replacement.compression.continuation import capture_rng, restore_rng, segment_schedule
from mlp_replacement.runlog import environment_record

from ..baseline import quantization as dense_runner
from ..stacking import stacking_1 as stacking_runner
from ..swiglu.shared import recovery_memory_record
from ..swiglu.swiglu_7 import (
    code_hashes,
    configure_strategy,
    cpu_tree,
    group_metadata,
    load_trainable_state,
    output_path,
    parameter_names,
    persist,
    resolve_storage,
    trainable_state,
    unique_parameters,
    utc_now,
)
from ..swiglu.swiglu_8 import (
    comparison,
    enforce,
    equality_check,
    is_digest,
    pinned_json,
    record_checks,
    record_failure,
    release_memory,
    verified_stream,
)
from ..swiglu.swiglu_9 import finished_outputs, load_swiglu8_prepared, write_once


WORKFLOW = "recovery-1"
SCHEMA_VERSION = 1
EXPERIMENT_CLASS = "recovery-schedule-screen"
DEFAULT_CONFIG = Path("workflows/configs/model/recovery/recovery-1.json")
DEFAULT_SOURCE_ROOT = Path("data/results/workflows/model")
# Array order; the job script repeats it.
FAMILY_KINDS = ("dense-qat", "order-b", "swiglu-bf16")
# Every recipe key the dense QAT runner and SwiGLU-8 must share for one comparison.
RECIPE_KEYS = (
    "learning_rate", "optimizer", "optimizer_backend", "weight_decay", "temperature", "ce_weight",
    "scheduler", "warmup_fraction", "final_lr_ratio", "sequence_length", "microbatch_sequences",
    "gradient_accumulation_steps", "effective_batch_tokens", "forward_autocast_dtype",
    "trainable_parameter_dtype", "optimizer_state_dtype",
)
STREAM_KEYS = ("recovery_sha256", "selection_sha256", "final_kl_sha256", "monitoring_sha256")


# Configuration


def load_settings(path):
    """Load the screen and the two configurations whose recipe it reuses unchanged."""

    path = resolve_path(Path(path))
    settings = read_json(path)
    if settings.get("schema_version") != SCHEMA_VERSION or settings.get("workflow") != WORKFLOW:
        raise ValueError("Expected a schema-1 Recovery-1 configuration")
    stacking_settings, unused_path = stacking_runner.load_settings(settings["configs"]["stacking"])
    budget = int(settings["budget_tokens"])
    dense_settings = dense_runner.load_settings(settings["configs"]["dense"], qat_tokens=budget)
    recipe = stacking_settings["recovery"]
    if any(dense_settings["qat"][key] != recipe[key] for key in RECIPE_KEYS):
        raise ValueError("Recovery-1 needs one recipe: dense QAT and SwiGLU-8 recovery must agree")
    schedule = settings["schedule"]
    percents = schedule["decay_percent"]
    if (
        not settings.get("seed") == stacking_settings["seed"] == dense_settings["seed"] == 21
        or recipe["scheduler"] != "constant"
        or int(recipe["target_tokens"]) != budget
        or recipe["segment_endpoints"] != [budget]
        or float(schedule["learning_rate"]) != float(recipe["learning_rate"])
        or schedule["scheduler"] != "wsd_one_minus_sqrt"
        or float(schedule["warmup_fraction"]) != 0.0
        or not 0.0 <= float(schedule["final_lr_ratio"]) < 1.0
        or percents != sorted(set(percents))
        or not all(type(value) is int and 0 < value < 100 for value in percents)
    ):
        raise ValueError("Recovery-1 decays the constant 3e-5 recipe of the 100M runs, without warmup or a new peak")
    families = settings["families"]
    if [family["kind"] for family in families] != list(FAMILY_KINDS) or len({f["name"] for f in families}) != 3:
        raise ValueError("Recovery-1 runs dense INT3 QAT, B 20% INT4, and SwiGLU 20% BF16, in that order")
    pins = [
        *(settings["streams"][key] for key in STREAM_KEYS),
        *settings["sources"].values(),
        *(family[key]["result_sha256"] for family in families for key in ("recorded_100m", "constant_1b")),
        *(family["selection_token_zero"]["result_sha256"] for family in families if family["selection_token_zero"]),
    ]
    if set(settings["streams"]) != set(STREAM_KEYS) or not all(is_digest(value) for value in pins):
        raise ValueError("Recovery-1 pins every stream and source by sha256")
    if settings["sources"]["swiglu8_prepare_sha256"] != stacking_settings["sources"]["swiglu8_prepare_sha256"]:
        raise ValueError("Recovery-1 uses the SwiGLU-8 preparation that Stacking-1 pins")
    if dense_settings["recovery_stream_sha256"][str(budget)] != settings["streams"]["recovery_sha256"]:
        raise ValueError("The dense QAT stream pin must be SwiGLU-8's 100M stream")
    for family in families[1:]:
        source = stacking_settings["sources"]["models"][f"{family['start']}-100m"]
        if family["kind"] == "swiglu-bf16" and family["recorded_100m"]["result_sha256"] != source["result_sha256"]:
            raise ValueError("SwiGLU's recorded run must be the SwiGLU-8 run that order B's start check uses")
    requests = settings["rows"]["trunk_requests"]
    starts = decay_starts(settings, recipe)
    if (
        requests != sorted(set(requests))
        or requests[-1] != budget
        or not {row["requested"] for row in starts.values()} <= set(requests)
        or not set(settings["rule"]["reference_requests"]) <= set(requests)
        or not 0.0 < float(settings["rule"]["bar"]) < 1.0
    ):
        raise ValueError("The trunk rows must cover every decay start and every reference row")
    return settings, stacking_settings, dense_settings, path


def decay_starts(settings, recipe):
    """Each decay's start: its request (1 - fraction of the budget), rounded up to an update boundary.

    At that token the trunk holds the state a stable-decay run would have, because
    both used the constant rate through the update that ends there.
    """

    budget = int(settings["budget_tokens"])
    step = int(recipe["effective_batch_tokens"])
    starts = {}
    for percent in settings["schedule"]["decay_percent"]:
        requested = budget - budget * int(percent) // 100
        starts[int(percent)] = {
            "requested": requested,
            "tokens": -(-requested // step) * step,
            "midpoint": requested + (budget - requested) // 2,
        }
    return starts


def output_name(family):
    return f"{family['name']}-run-001"


def start_artifact(settings, family, config_path):
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "experiment_family": WORKFLOW,
        "experiment_class": EXPERIMENT_CLASS,
        "command": "train",
        "identity": {"family": family["name"], "kind": family["kind"], "bits": family["bits"]},
        "status": "running",
        "created_at_utc": utc_now(),
        "configuration": deepcopy(settings),
        "configuration_sha256": file_digest(config_path),
        "reused_configuration_sha256": {
            name: file_digest(resolve_path(Path(value))) for name, value in settings["configs"].items()
        },
        "environment": environment_record(),
        "code_hashes": code_hashes(),
        "results": {},
        "error": None,
    }


# Inputs and scores


def checked_stream(path, record, label):
    """An explicitly given stream file, checked against its record before use."""

    path = resolve_path(Path(path))
    return [
        equality_check("inputs", path.stat().st_size, 4 * int(record["token_count"]), metric=f"{label}_bytes"),
        equality_check("inputs", file_digest(path), record["sha256"], metric=f"{label}_file_sha256"),
    ], path


def open_inputs(args, settings, stacking_settings, dense_settings, family, artifact, output_dir, work_dir):
    """Verify every stream and pinned record, then open the recovery stream."""

    import torch
    from mlp_replacement.data import PackedTokenCache

    budget = int(settings["budget_tokens"])
    streams = settings["streams"]
    source_root = resolve_path(Path(args.source_root))
    prepared_path, prepared, prepared_root = load_swiglu8_prepared(args.prepared, stacking_settings)
    block = pinned_json(source_root / "swiglu-8", "block-check-001/result.json",
                        settings["sources"]["swiglu8_block_check_sha256"], "SwiGLU-8 block check")
    stream_record = prepared["results"]["stream"]
    final_record = prepared["results"]["long_context_data"]["kl_stream"]
    monitoring_record = prepared["results"]["monitoring_evaluation"]
    selection_record = block["results"]["long_context_data"]["selection_stream"]
    selection_checks, selection_path = checked_stream(args.selection_stream, selection_record, "selection_stream")
    enforce(artifact, output_dir, "inputs", [
        equality_check("inputs", stream_record["sha256"], streams["recovery_sha256"], metric="recovery_stream_sha256"),
        equality_check("inputs", int(stream_record["token_count"]), budget, metric="recovery_stream_tokens"),
        equality_check("inputs", selection_record["sha256"], streams["selection_sha256"], metric="selection_stream_sha256"),
        equality_check("inputs", final_record["sha256"], streams["final_kl_sha256"], metric="final_kl_stream_sha256"),
        equality_check("inputs", monitoring_record["sha256"], streams["monitoring_sha256"], metric="monitoring_sha256"),
        *selection_checks,
    ])
    # verified_stream rereads each file and stops on a size or sha256 mismatch.
    stream_path = verified_stream(prepared_root, stream_record)
    final_path = verified_stream(prepared_root, final_record)
    monitoring = None
    if family["kind"] != "dense-qat":
        path = output_path(prepared_root, monitoring_record["path"])
        if file_digest(path) != monitoring_record["sha256"]:
            raise ValueError("SwiGLU-8's monitoring batches changed")
        monitoring = torch.load(path, map_location="cpu", weights_only=False)
    recorded = {
        key: pinned_json(source_root / family[key]["workflow"], f"{family[key]['run']}/result.json",
                         family[key]["result_sha256"], f"{family[key]['workflow']} {family[key]['run']}")
        for key in ("recorded_100m", "constant_1b")
    }
    if any(value.get("status") != "completed" for value in recorded.values()):
        raise ValueError("Recovery-1 compares only with completed runs")
    recipe = stacking_settings["recovery"]
    cache = PackedTokenCache(stream_path, budget, recipe["sequence_length"], stream_record["sha256"])
    length = int(stacking_settings["long_context"]["sequence_length"])
    artifact["results"]["inputs"] = {
        "swiglu8_prepare_sha256": file_digest(prepared_path),
        "recovery_stream": {key: stream_record[key] for key in ("sha256", "token_count")},
        "selection_stream": {key: selection_record[key] for key in ("sha256", "token_count", "protocol")},
        "final_kl_stream": {key: final_record[key] for key in ("sha256", "token_count", "protocol")},
    }
    return SimpleNamespace(
        settings=settings, stacking_settings=stacking_settings, dense_settings=dense_settings, family=family,
        artifact=artifact, output_dir=output_dir, work_dir=work_dir, recipe=recipe, budget=budget,
        prepared=prepared, prepared_root=prepared_root, source_root=source_root, monitoring=monitoring,
        recorded=recorded, length=length, device=torch.device("cuda"),
        selection=SimpleNamespace(path=selection_path, record=selection_record),
        final=SimpleNamespace(path=final_path, record=final_record),
        batch_at=lambda offset, count: cache.batch(offset, count, recipe["microbatch_sequences"],
                                                   allow_partial_sequence=True),
        model=None, teacher=None, groups=None, train_modules=None, hooks=[], context=None,
    )


def stream_kl(model, teacher, stream, length):
    from mlp_replacement.evaluation.teacher_kl import evaluate_teacher_kl

    return float(evaluate_teacher_kl(model, teacher, stream.path, stream.record, length,
                                     protocol=stream.record["protocol"], device="cuda")["kl"])


def score(task, tokens, requested, updates, seconds, mean_train_kl, learning_rates):
    """One row: the selection-stream 8K KL, which decides, and the family's old history metric.

    The old metric is the final 8K stream for dense QAT (Quantization-2's rows) and
    SwiGLU-8's monitoring KL for the SwiGLU models (with WikiText perplexity where
    the recorded rows have it). RNG is restored, so scoring never perturbs training.
    """

    rng = capture_rng()
    if task.context is None:
        row = {
            "actual_tokens": int(tokens), "requested_tokens": [int(value) for value in requested],
            "optimizer_updates": int(updates), "training_seconds": float(seconds),
            "mean_train_kl": mean_train_kl, "learning_rates": list(learning_rates),
            "legacy_kl": stream_kl(task.model, task.teacher, task.final, task.length),
        }
    else:
        wikitext = int(tokens) == 0 or any(point in task.recipe["monitoring_ppl_tokens"] for point in requested)
        row = stacking_runner.monitoring_row(task.context, tokens, requested, updates, seconds, mean_train_kl,
                                             learning_rates, wikitext)
        row["legacy_kl"] = row["recovery_validation_kl"]
    row["selection_kl"] = stream_kl(task.model, task.teacher, task.selection, task.length)
    restore_rng(rng)
    values = [row["selection_kl"], row["legacy_kl"], *([] if mean_train_kl is None else [mean_train_kl])]
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("Non-finite KL; the run stopped")
    return row


# Family starts


def open_dense(task):
    """Dense INT3 QAT, prepared as Quantization-2's ``qat-int3-100m-002`` was (quantization.py)."""

    from mlp_replacement.compression.quantization import (
        fake_quantization_check, prepare_qat, projection_paths, require_backend,
    )

    settings = task.dense_settings
    bits = int(task.family["bits"])
    dense_runner.seed_run(settings["seed"])
    require_backend(settings["backend"], bits)
    student, unused_tokenizer = dense_runner.load_teacher(settings)
    paths = projection_paths(student, settings["scope"]["eligible_layers"])
    if len(paths) != settings["scope"]["expected_projections"]:
        raise ValueError("Eligible projection count changed")
    task.teacher, unused_tokenizer = dense_runner.load_teacher(settings)
    parameters = list(prepare_qat(student, paths, bits))
    check_fake_quantization(task, fake_quantization_check(student, paths, bits), bits)
    task.model = student
    task.groups = [{
        "name": "eligible_mlp", "parameters": parameters, "parameter_names": parameter_names(student, parameters),
        "parameter_count": sum(value.numel() for value in parameters),
        "learning_rate": task.recipe["learning_rate"], "weight_decay": task.recipe["weight_decay"],
    }]
    task.train_modules = [student.get_submodule(path) for path in paths]


def check_fake_quantization(task, check, bits):
    """K6: training sees exactly the deployed weights (decision log, 2026-10-25 and 2026-11-01)."""

    task.artifact["results"][f"int{bits}_fake_quantization_check"] = check
    enforce(task.artifact, task.output_dir, "K6-start", [equality_check(
        "K6", len(check["mismatched"]), 0, metric=f"int{bits}_fake_differs_from_deployed_projections",
        projections=check["projections"])])


def stacking_context(task, bits):
    """The Stacking-1 context that its start, padding, and monitoring helpers read."""

    family = task.family
    return SimpleNamespace(
        settings=task.stacking_settings, variant=f"{WORKFLOW}:{family['name']}",
        spec={"group": "b", "order": "B", "bits": bits, "start": family["start"],
              "reference": f"{family['start']}-100m"},
        artifact=task.artifact, output_dir=task.output_dir, work_dir=task.work_dir, prepared=task.prepared,
        prepared_root=task.prepared_root, source_root=task.source_root, run_fingerprint=None, resume=False,
        model=None, tokenizer=None, teacher=None, monitoring=task.monitoring,
    )


def open_order_b(task):
    """B 20% INT4: SwiGLU-8's fitted start, padded and fake-quantized as Stacking-1's order B.

    The preparation repeats ``stacking_1.qat_recover_stage`` up to its training loop:
    every projection of layers 1-22 is fake-quantized and only the replacement
    modules train (the S7-0 scope).
    """

    from mlp_replacement.compression.quantization import (
        fake_quantization_check, mask_padded_gradients, prepare_qat, require_backend,
    )

    bits = int(task.family["bits"])
    context = task.context = stacking_context(task, bits)
    require_backend(task.stacking_settings["backend"], bits)
    stacking_runner.load_start(context)
    stacking_runner.pad_stage(context)
    context.paths = stacking_runner.eligible_paths(context)
    replaced = {row["path"] for row in context.replacements}
    trainable_paths = [path for path in context.paths if path.rsplit(".", 1)[0] in replaced]
    modules = [context.model.get_submodule(path) for path in sorted(replaced)]
    scope = set(parameter_names(context.model, unique_parameters(modules)))
    if scope != {f"{path}.weight" for path in trainable_paths}:
        raise ValueError("Order B must train exactly SwiGLU-8's S7-0 replacement scope")
    trainable = prepare_qat(context.model, context.paths, bits, trainable_paths)
    check_fake_quantization(task, fake_quantization_check(context.model, context.paths, bits), bits)
    task.hooks = mask_padded_gradients(context.model, getattr(context, "padding", []))
    task.model, task.teacher = context.model, context.teacher
    task.groups = stacking_runner.qat_groups(context, trainable)
    task.train_modules = [context.model.get_submodule(path) for path in trainable_paths]


def open_swiglu(task):
    """SwiGLU alone at 20% in BF16: SwiGLU-8's fitted start with its S7-0 scope (swiglu_8.py, train)."""

    context = task.context = stacking_context(task, None)
    stacking_runner.load_start(context)
    target_paths = [row["path"] for row in context.replacements]
    groups, train_modules, adapters = configure_strategy(context.model, target_paths, "S7-0",
                                                         task.stacking_settings)
    if adapters:
        raise ValueError("S7-0 trains the replacements only")
    task.model, task.teacher, task.groups, task.train_modules = context.model, context.teacher, groups, train_modules


OPENERS = {"dense-qat": open_dense, "order-b": open_order_b, "swiglu-bf16": open_swiglu}


# Fidelity checks


def request_key(row):
    return tuple(int(value) for value in row["requested_tokens"])


def recorded_curve(family, recorded):
    """The recorded constant run's old-metric rows, keyed by their requested tokens."""

    recovery = recorded["results"]["recovery"]
    if family["kind"] == "dense-qat":
        return {request_key(row): {"kl": float(row["kl"]["kl"]), "perplexity": None} for row in recovery["history"]}
    return {
        request_key(row): {
            "kl": float(row["recovery_validation_kl"]),
            "perplexity": row.get("wikitext_validation", {}).get("perplexity"),
        }
        for row in recovery["validation_history"]
    }


def curve_checks(rows, curve, tolerance, check):
    """Compare each row with the recorded row at the same request; rows without one are skipped."""

    checks = []
    for row in rows:
        reference = curve.get(request_key(row))
        if reference is None:
            continue
        labels = {"tokens": int(row["actual_tokens"])}
        checks.append(comparison(check, row["legacy_kl"], reference["kl"], tolerance, metric="legacy_kl", **labels))
        if "wikitext_validation" in row and reference["perplexity"] is not None:
            checks.append(comparison(check, row["wikitext_validation"]["perplexity"], reference["perplexity"],
                                     tolerance, metric="wikitext_validation_perplexity", **labels))
    return checks


def check_start(task):
    """Stop unless token 0 reproduces the recorded run and the selection stream scores the teacher at 0."""

    acceptance = task.settings["acceptance"]
    self_kl = stream_kl(task.teacher, task.teacher, task.selection, task.length)
    enforce(task.artifact, task.output_dir, "teacher-self-kl", [{
        "check": "teacher-self-kl", "metric": "selection_stream_teacher_self_kl", "value": self_kl,
        "tolerance": float(acceptance["teacher_self_kl_absolute"]),
        "passed": abs(self_kl) <= float(acceptance["teacher_self_kl_absolute"]),
    }])
    row = score(task, 0, [0], 0, 0.0, None, [group["learning_rate"] for group in task.groups])
    trunk = task.artifact["results"]["trunk"] = {"history": [row], "tokens_seen": 0, "optimizer_updates": 0}
    trunk["trainable_scope"] = group_metadata(task.groups, task.model)
    curve = recorded_curve(task.family, task.recorded["recorded_100m"])
    checks = curve_checks([row], curve, float(acceptance["exact_relative"]), "token-0")
    if not checks:
        raise ValueError("The recorded run has no token-0 row")
    enforce(task.artifact, task.output_dir, "token-0", checks)
    reference = task.family["selection_token_zero"]
    if reference is not None:
        # Recorded, not enforced: Heterogeneous-1 scored a refit of this model, not the start file.
        assembled = pinned_json(task.source_root / reference["workflow"], f"{reference['run']}/result.json",
                                reference["result_sha256"], f"{reference['workflow']} {reference['run']}")
        point = next(value for value in assembled["results"]["points"] if value["name"] == reference["point"])
        record_checks(task.artifact, task.output_dir, "selection-token-0", [comparison(
            "selection-token-0", row["selection_kl"], point["kl_8192"], float(acceptance["exact_relative"]),
            metric="selection_kl")])


# Training


def recover(task, start_tokens, start_updates, optimizer_state, schedule, on_checkpoint, decay_start_tokens=None):
    """One call of the shared loop from origin 0, so the schedule sees absolute token counts."""

    import torch
    from mlp_replacement.compression.recovery import recover_trainable_by_tokens

    # recover_exact_segment is not used: it is constant-only and counts tokens from its segment.

    recipe = task.recipe
    decaying = decay_start_tokens is not None
    return recover_trainable_by_tokens(
        student=task.model, teacher=task.teacher,
        parameter_groups=[{key: group[key] for key in ("name", "parameters", "learning_rate", "weight_decay")}
                          for group in task.groups],
        train_modules=task.train_modules, batch_at=task.batch_at,
        target_tokens=task.budget, schedule_tokens=task.budget,
        microbatch_tokens=recipe["sequence_length"] * recipe["microbatch_sequences"],
        accumulation_steps=recipe["gradient_accumulation_steps"],
        temperature=recipe["temperature"], ce_weight=recipe["ce_weight"],
        scheduler=task.settings["schedule"]["scheduler"] if decaying else recipe["scheduler"],
        warmup_fraction=0.0,
        final_lr_ratio=task.settings["schedule"]["final_lr_ratio"] if decaying else recipe["final_lr_ratio"],
        device=task.device, autocast_dtype=torch.bfloat16,
        start_tokens=start_tokens, start_updates=start_updates, elapsed_seconds=0.0,
        optimizer_state=optimizer_state,
        checkpoint_schedule=tuple((actual, requested) for actual, requested in schedule if actual > start_tokens),
        on_checkpoint=on_checkpoint, optimizer_backend=recipe["optimizer_backend"],
        decay_start_tokens=decay_start_tokens,
    )


def train_trunk(task, starts):
    """The constant run to 100M; it keeps the state at every decay start for the branches."""

    step = int(task.recipe["effective_batch_tokens"])
    trunk = task.artifact["results"]["trunk"]
    snapshots = {}
    percent_at = {row["tokens"]: percent for percent, row in starts.items()}

    def on_checkpoint(event, optimizer, first_step):
        row = score(task, event.tokens_seen, event.requested_checkpoint_tokens, event.optimizer_updates,
                    event.elapsed_seconds, event.mean_train_kl, event.learning_rates)
        trunk["history"].append(row)
        trunk.update(tokens_seen=event.tokens_seen, optimizer_updates=event.optimizer_updates,
                     training_seconds=event.elapsed_seconds, first_step=trunk.get("first_step") or first_step,
                     memory=recovery_memory_record(task.device))
        if event.tokens_seen in percent_at:
            # Copied to CPU: the loop owns the live tensors and continues from them.
            snapshots[percent_at[event.tokens_seen]] = {
                "tokens": event.tokens_seen, "updates": event.optimizer_updates,
                "trainable_state": trainable_state(task.model, task.groups),
                "optimizer_state": cpu_tree(optimizer.state_dict()), **capture_rng(),
            }
        persist(task.output_dir, task.artifact, "trunk",
                {"family": task.family["name"], "tokens_seen": event.tokens_seen, "target_tokens": task.budget})
        print(f"family={task.family['name']} trunk tokens={event.tokens_seen:,} "
              f"selection={row['selection_kl']:.6f} legacy={row['legacy_kl']:.6f}", flush=True)

    schedule = segment_schedule(0, task.budget, task.settings["rows"]["trunk_requests"], step)
    result = recover(task, 0, 0, None, schedule, on_checkpoint)
    if result.tokens_seen != task.budget or set(snapshots) != set(starts):
        raise ValueError("The trunk did not reach 100M with every decay start kept")
    if any(snapshots[percent]["tokens"] != row["tokens"] for percent, row in starts.items()):
        raise ValueError("A decay start is not on the update boundary its schedule uses")
    curve = recorded_curve(task.family, task.recorded["recorded_100m"])
    record_checks(task.artifact, task.output_dir, "trunk-curve", curve_checks(
        trunk["history"][1:], curve, float(task.settings["acceptance"]["review_relative"]), "curve"))
    return snapshots


def train_branch(task, percent, start, snapshot):
    """Restore the trunk at the decay start and decay to 100M; nothing else differs from the trunk."""

    step = int(task.recipe["effective_batch_tokens"])
    branches = task.artifact["results"].setdefault("branches", {})
    branch = branches[str(percent)] = {
        "decay_percent": percent, "decay_start_requested": start["requested"],
        "decay_start_tokens": start["tokens"], "start_updates": snapshot["updates"], "history": [],
    }
    load_trainable_state(task.model, snapshot.pop("trainable_state"), task.groups)
    restore_rng(snapshot)

    def on_checkpoint(event, optimizer, first_step):
        row = score(task, event.tokens_seen, event.requested_checkpoint_tokens, event.optimizer_updates,
                    event.elapsed_seconds, event.mean_train_kl, event.learning_rates)
        branch["history"].append(row)
        branch.update(tokens_seen=event.tokens_seen, optimizer_updates=event.optimizer_updates,
                      training_seconds=event.elapsed_seconds)
        persist(task.output_dir, task.artifact, f"branch-{percent}",
                {"family": task.family["name"], "tokens_seen": event.tokens_seen, "target_tokens": task.budget})
        print(f"family={task.family['name']} decay={percent}% tokens={event.tokens_seen:,} "
              f"selection={row['selection_kl']:.6f} legacy={row['legacy_kl']:.6f} lr={event.learning_rates[0]:.3e}",
              flush=True)

    schedule = segment_schedule(0, task.budget, [start["midpoint"], task.budget], step)
    # The loop loads and then clears this state; each snapshot serves one branch only.
    result = recover(task, start["tokens"], snapshot["updates"], snapshot.pop("optimizer_state"), schedule,
                     on_checkpoint, decay_start_tokens=start["tokens"])
    if result.tokens_seen != task.budget:
        raise ValueError(f"The {percent}% decay did not reach 100M")


def train(args, settings, stacking_settings, dense_settings, config_path):
    """Run one family: inputs, start checks, the constant trunk, then each decay from its snapshot."""

    families = settings["families"]
    if not 0 <= args.task_index < len(families):
        raise ValueError(f"Train task index must be 0-{len(families) - 1}")
    family = families[args.task_index]
    work_dir, output_dir = resolve_storage(args.work_dir, args.output_dir, False)
    artifact = start_artifact(settings, family, config_path)
    persist(output_dir, artifact, "loading_inputs")
    try:
        task = open_inputs(args, settings, stacking_settings, dense_settings, family, artifact, output_dir, work_dir)
        persist(output_dir, artifact, "loading_models")
        OPENERS[family["kind"]](task)
        check_start(task)
        starts = decay_starts(settings, task.recipe)
        artifact["results"]["decay_starts"] = {str(percent): row for percent, row in starts.items()}
        snapshots = train_trunk(task, starts)
        for percent in sorted(starts):
            train_branch(task, percent, starts[percent], snapshots.pop(percent))
            release_memory()
        for handle in task.hooks:
            handle.remove()
        artifact["status"] = "completed"
        persist(output_dir, artifact, "completed")
    except BaseException as error:
        record_failure(output_dir, work_dir, artifact, error)
        raise


# The winner rule


def last_request(row):
    return int(row["requested_tokens"][-1])


def family_decision(trunk_rows, branches, rule, metric="selection_kl"):
    """Apply the winner rule to one family (decision log, 2026-11-07).

    The reference is the trunk's lowest value at the reference rows, so one constant
    spike cannot hand a decay the win. A decay qualifies if its 100M value is more
    than the bar below the reference. The shortest decay within the bar of the best
    qualifier wins, because it stays closest to constant; with none, constant stays.
    """

    bar = float(rule["bar"])
    wanted = {int(value) for value in rule["reference_requests"]}
    rows = [row for row in trunk_rows if last_request(row) in wanted]
    if {last_request(row) for row in rows} != wanted:
        raise ValueError("The trunk lacks a reference row")
    reference = min(float(row[metric]) for row in rows)
    candidates = []
    for percent, branch in sorted(branches.items(), key=lambda item: int(item[0])):
        final = branch["history"][-1]
        value = float(final[metric])
        candidates.append({
            "decay_percent": int(percent), "value": value, "gain": 1.0 - value / reference,
            "qualifies": value < reference * (1.0 - bar),
        })
    qualifiers = [row for row in candidates if row["qualifies"]]
    winner = None
    if qualifiers:
        best = min(row["value"] for row in qualifiers)
        winner = min((row for row in qualifiers if row["value"] <= best * (1.0 + bar)),
                     key=lambda row: row["decay_percent"])
    return {
        "metric": metric,
        "reference": reference,
        "reference_tokens": int(min(rows, key=lambda row: float(row[metric]))["actual_tokens"]),
        "trunk_100m": float(trunk_rows[-1][metric]),
        "candidates": candidates,
        "decay_percent": None if winner is None else winner["decay_percent"],
    }


def constant_curve(family, recorded):
    """The 1B constant run's old-metric curve as (tokens, value), in token order."""

    recovery = recorded["results"]["recovery"]
    if family["kind"] == "dense-qat":
        rows = [(int(row["tokens_seen"]), float(row["kl"]["kl"])) for row in recovery["history"]]
    else:
        rows = [(int(row["actual_tokens"]), float(row["recovery_validation_kl"]))
                for row in recovery["validation_history"]]
    return sorted(rows)


def token_equivalent(curve, value):
    """First token count at which the constant curve reaches ``value``, linear between rows; None if never."""

    for index, (tokens, level) in enumerate(curve):
        if level <= value:
            if index == 0:
                return float(tokens)
            earlier_tokens, earlier_level = curve[index - 1]
            return earlier_tokens + (earlier_level - value) / (earlier_level - level) * (tokens - earlier_tokens)
    return None


def family_flags(artifact, selection, legacy):
    """Reasons to hold a family's decision until the researcher has reviewed them."""

    acceptance = artifact["results"]["acceptance"]
    flags = []
    for group in ("trunk-curve", "selection-token-0"):
        summary = acceptance.get(group)
        if summary is not None and not summary["passed"]:
            failed = [row for row in summary["checks"] if not row["passed"]]
            flags.append({"flag": group, "failed": failed})
    for chosen, other in zip(selection["candidates"], legacy["candidates"]):
        if chosen["qualifies"] != other["qualifies"]:
            flags.append({"flag": "legacy-disagrees", "decay_percent": chosen["decay_percent"],
                          "selection_qualifies": chosen["qualifies"], "legacy_qualifies": other["qualifies"]})
    return flags


def summarize(args, settings):
    """Apply the winner rule to the three finished families; safe to re-run."""

    root = resolve_path(args.results_root)
    source_root = resolve_path(Path(args.source_root))
    families = settings["families"]
    artifacts = [read_json(path) for path in finished_outputs(root, [output_name(family) for family in families])]
    if any(artifact.get("workflow") != WORKFLOW or artifact.get("status") != "completed" for artifact in artifacts):
        raise ValueError("Every Recovery-1 family must have completed")
    if any(artifact["code_hashes"] != artifacts[0]["code_hashes"] for artifact in artifacts):
        raise ValueError("All three families must have run the same code")
    accepted = set(args.accept_flags or [])
    unknown = accepted - {family["name"] for family in families}
    if unknown:
        raise ValueError(f"Unknown families: {', '.join(sorted(unknown))}")
    rule = settings["rule"]
    results = {}
    for family, artifact in zip(families, artifacts):
        trunk = artifact["results"]["trunk"]["history"]
        branches = artifact["results"]["branches"]
        selection = family_decision(trunk, branches, rule, "selection_kl")
        legacy = family_decision(trunk, branches, rule, "legacy_kl")
        reference = family["constant_1b"]
        recorded = pinned_json(source_root / reference["workflow"], f"{reference['run']}/result.json",
                               reference["result_sha256"], f"{reference['workflow']} {reference['run']}")
        curve = constant_curve(family, recorded)
        flags = family_flags(artifact, selection, legacy)
        curve_check = artifact["results"]["acceptance"]["trunk-curve"]["checks"]
        held = bool(flags) and family["name"] not in accepted
        # The selection stream has no 1B curve, so the token equivalent uses the old metric.
        equivalents = {"trunk": token_equivalent(curve, trunk[-1]["legacy_kl"])}
        for percent, branch in branches.items():
            equivalents[percent] = token_equivalent(curve, branch["history"][-1]["legacy_kl"])
        results[family["name"]] = {
            "decision": "held: review the flags" if held else (
                "constant" if selection["decay_percent"] is None else f"decay {selection['decay_percent']}%"),
            "decay_percent": None if held else selection["decay_percent"],
            "selection": selection,
            "legacy": legacy,
            "flags": flags,
            "flags_accepted": bool(flags) and family["name"] in accepted,
            # Two runs of one configuration: the trunk and the recorded constant run.
            "noise_sample": max((row["relative_difference"] for row in curve_check), default=None),
            "token_equivalent": equivalents,
            "constant_1b": {"workflow": reference["workflow"], "run": reference["run"]},
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "experiment_family": WORKFLOW,
        "experiment_class": EXPERIMENT_CLASS,
        "command": "summarize",
        "identity": {"kind": "summary"},
        "status": "completed",
        "created_at_utc": max(artifact["updated_at_utc"] for artifact in artifacts),
        "configuration": deepcopy(settings),
        "results": {"families": results, "accepted_flags": sorted(accepted)},
        "error": None,
    }
    print_summary(results)
    held = [name for name, entry in results.items() if entry["decision"].startswith("held")]
    if held:
        # Nothing is written until every flag is reviewed, so summary-001 is the decided record.
        print(f"Not written: {', '.join(held)} held for review; rerun with --accept-flags after the review")
        return
    write_once(root / "summary-001", summary, "summary")


def print_summary(results):
    for name, entry in results.items():
        selection = entry["selection"]
        print(f"{name}: {entry['decision']} (reference {selection['reference']:.6f} at "
              f"{selection['reference_tokens']:,} tokens; noise sample {entry['noise_sample']})")
        for row in selection["candidates"]:
            mark = "qualifies" if row["qualifies"] else "no"
            equivalent = entry["token_equivalent"][str(row["decay_percent"])]
            label = "> 1B" if equivalent is None else f"{equivalent / 1e6:.0f}M"
            print(f"  decay {row['decay_percent']}%: {row['value']:.6f}, gain {row['gain']:+.2%}, {mark}; "
                  f"constant-token equivalent {label}")
        for flag in entry["flags"]:
            state = "accepted after review" if entry["flags_accepted"] else "REVIEW before use"
            print(f"  FLAG {flag['flag']} ({state}): {flag}")


# Command line


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--task-index", type=int, required=True)
    train_parser.add_argument("--work-dir", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--prepared", type=Path, required=True,
                              help="SwiGLU-8's prepare-001/result.json")
    train_parser.add_argument("--selection-stream", type=Path, required=True,
                              help="A copy of SwiGLU-8's 8K selection stream; checked against its record")
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--results-root", type=Path, required=True)
    summary_parser.add_argument("--accept-flags", nargs="*", default=[],
                                help="Families whose flags the researcher has reviewed")
    for subparser in (train_parser, summary_parser):
        subparser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        subparser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    settings, stacking_settings, dense_settings, config_path = load_settings(args.config)
    if args.command == "summarize":
        summarize(args, settings)
    else:
        train(args, settings, stacking_settings, dense_settings, config_path)


if __name__ == "__main__":
    main()
