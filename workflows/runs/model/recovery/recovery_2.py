"""Run Recovery-2: decay length at 1B, constant against 50% and 20% 1-sqrt decays.

``train --task-index i`` runs one family: dense INT3 QAT, B 20% INT4 (order B), or
SwiGLU alone at 20% in BF16. It trains on the 1B stream of the family's recorded
constant 1B run and copies that run's update layout: one segment from token 0 for
dense (Quantization-2), the two segments 0-100M and 100M-1B for B and SwiGLU
(SwiGLU-7's recipe, used by Stacking-1). It stops unless token 0 reproduces the
recorded run and Recovery-1. It then trains a constant trunk to 1B, keeping the
trainable and optimizer state at each decay start, and afterwards restores each
state and decays it to 1B with the 1-sqrt schedule. Every row is scored on the 8K
selection stream, which decides, and on the family's old history metric, which
checks the trunk against its recorded run. ``summarize`` applies the winner rule on
the login node. The binding design is in ``docs/agents/decisions.md`` (2026-11-08)
and ``docs/experiments/model/recovery/recovery-2.md``.
"""

from __future__ import annotations

import argparse
import statistics
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from workflows.runs.model.common import resolve_path

from mlp_replacement.artifacts import file_digest, read_json
from mlp_replacement.compression.continuation import capture_rng, restore_rng, segment_origin, segment_schedule
from mlp_replacement.runlog import environment_record

from ..baseline import quantization as dense_runner
from ..stacking import stacking_1 as stacking_runner
from ..swiglu.shared import recovery_memory_record
from ..swiglu.swiglu_7 import (
    code_hashes,
    cpu_tree,
    group_metadata,
    load_trainable_state,
    output_path,
    persist,
    resolve_storage,
    trainable_state,
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
from .recovery_1 import (
    FAMILY_KINDS,
    OPENERS,
    RECIPE_KEYS,
    STREAM_KEYS,
    checked_stream,
    curve_checks,
    family_flags,
    last_request,
    output_name,
    recorded_curve,
    score,
    stream_kl,
)


WORKFLOW = "recovery-2"
SCHEMA_VERSION = 1
EXPERIMENT_CLASS = "recovery-schedule-screen"
DEFAULT_CONFIG = Path("workflows/configs/model/recovery/recovery-2.json")
DEFAULT_SOURCE_ROOT = Path("data/results/workflows/model")
DECAY_PERCENT = [20, 50]
# The recorded 1B runs' rows and WikiText points; dense QAT and SwiGLU-7 must share them.
ROW_KEYS = ("validation_interval_tokens", "monitoring_ppl_tokens")
SOURCE_KEYS = ("swiglu8_prepare_sha256", "swiglu8_block_check_sha256", "swiglu7_prepare_sha256")
# The median decides (decision log, 2026-11-08); the lowest is Recovery-1's rule, reported only.
REDUCERS = {"median": statistics.median, "lowest": min}


# Configuration


def load_settings(path):
    """Load the screen and the two configurations whose 1B recipe it reuses unchanged."""

    path = resolve_path(Path(path))
    settings = read_json(path)
    if settings.get("schema_version") != SCHEMA_VERSION or settings.get("workflow") != WORKFLOW:
        raise ValueError("Expected a schema-1 Recovery-2 configuration")
    stacking_settings, unused_path = stacking_runner.load_settings(settings["configs"]["stacking"])
    budget = int(settings["budget_tokens"])
    dense_settings = dense_runner.load_settings(settings["configs"]["dense"], qat_tokens=budget)
    recipe = stacking_settings["recovery_1b"]
    if (
        any(dense_settings["qat"][key] != recipe[key] for key in (*RECIPE_KEYS, *ROW_KEYS))
        # Recovery-1's order-B and SwiGLU openers read the rate from Stacking-1's 100M block.
        or any(stacking_settings["recovery"][key] != recipe[key] for key in RECIPE_KEYS)
    ):
        raise ValueError("Recovery-2 needs one recipe: dense QAT and SwiGLU-7's 1B recovery must agree")
    schedule = settings["schedule"]
    endpoints = [int(value) for value in recipe["segment_endpoints"]]
    if (
        not settings.get("seed") == stacking_settings["seed"] == dense_settings["seed"] == 21
        or recipe["scheduler"] != "constant"
        or int(recipe["target_tokens"]) != budget
        or endpoints[-1] != budget
        or float(schedule["learning_rate"]) != float(recipe["learning_rate"])
        or schedule["scheduler"] != "wsd_one_minus_sqrt"
        or float(schedule["warmup_fraction"]) != 0.0
        or float(schedule["final_lr_ratio"]) != 0.0
        or schedule["decay_percent"] != DECAY_PERCENT
    ):
        raise ValueError("Recovery-2 decays the constant 3e-5 recipe of the 1B runs over 20% and 50%, "
                         "without warmup or a new peak")
    families = settings["families"]
    if [family["kind"] for family in families] != list(FAMILY_KINDS) or len({f["name"] for f in families}) != 3:
        raise ValueError("Recovery-2 runs dense INT3 QAT, B 20% INT4, and SwiGLU 20% BF16, in that order")
    for family in families:
        # Dense QAT trained in one call from token 0; SwiGLU-7 and Stacking-1 in the recipe's segments.
        expected = [budget] if family["kind"] == "dense-qat" else endpoints
        if family["segment_endpoints"] != expected:
            raise ValueError(f"{family['name']} must keep its recorded run's update layout {expected}")
        if family["recovery_1"]["workflow"] != "recovery-1" or family["recovery_1"]["run"] != output_name(family):
            raise ValueError(f"{family['name']} compares its token 0 with its own Recovery-1 run")
    pins = [
        *(settings["streams"][key] for key in STREAM_KEYS),
        *settings["sources"].values(),
        *(family[key]["result_sha256"] for family in families for key in ("recorded_1b", "recovery_1")),
    ]
    if (
        set(settings["streams"]) != set(STREAM_KEYS)
        or set(settings["sources"]) != set(SOURCE_KEYS)
        or not all(is_digest(value) for value in pins)
    ):
        raise ValueError("Recovery-2 pins every stream and source by sha256")
    for key in ("swiglu8_prepare_sha256", "swiglu7_prepare_sha256"):
        if settings["sources"][key] != stacking_settings["sources"][key]:
            raise ValueError(f"Recovery-2 uses the preparation that Stacking-1 pins ({key})")
    if dense_settings["recovery_stream_sha256"][str(budget)] != settings["streams"]["recovery_sha256"]:
        raise ValueError("The recovery stream pin must be the 1B stream of Quantization-2's 1B QAT")
    swiglu = families[2]
    source = stacking_settings["sources"]["models"][f"{swiglu['start']}-1b"]
    if [swiglu["recorded_1b"][key] for key in ("workflow", "run", "result_sha256")] != [
            source[key] for key in ("workflow", "run", "result_sha256")]:
        raise ValueError("SwiGLU's recorded 1B run must be the SwiGLU-7 run Stacking-1 pins")
    rule = settings["rule"]
    requests = trunk_requests(settings, recipe)
    if (
        int(settings["rows"]["interval_tokens"]) != int(recipe["validation_interval_tokens"])
        or rule["reference"] != "median"
        or not set(rule["reference_requests"]) <= set(requests)
        or not 0.0 < float(rule["bar"]) < 1.0
        or any(row["requested"] not in requests
               for family in families for row in decay_starts(settings, recipe, family).values())
    ):
        raise ValueError("The trunk rows must cover every decay start and every reference row, "
                         "and the median decides")
    return settings, stacking_settings, dense_settings, path


def trunk_requests(settings, recipe):
    """The recorded 1B runs' rows: every 25M, the WikiText points, and each segment end."""

    budget = int(settings["budget_tokens"])
    interval = int(settings["rows"]["interval_tokens"])
    return sorted({
        *range(interval, budget + 1, interval),
        *(int(value) for value in recipe["monitoring_ppl_tokens"]),
        *(int(value) for value in recipe["segment_endpoints"]),
    })


def decay_starts(settings, recipe, family):
    """Each decay's start: its request, rounded up to an update boundary of the family's layout.

    Updates are counted from the start of the segment that holds the request, as in the
    recorded run, so at that token the trunk holds the state a stable-decay run would have.
    """

    budget = int(settings["budget_tokens"])
    step = int(recipe["effective_batch_tokens"])
    starts = {}
    for percent in settings["schedule"]["decay_percent"]:
        requested = budget - budget * int(percent) // 100
        origin = segment_origin(requested, family["segment_endpoints"])
        starts[int(percent)] = {
            "requested": requested,
            "origin": origin,
            "tokens": origin + -(-(requested - origin) // step) * step,
        }
    return starts


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


# Inputs


def open_inputs(args, settings, stacking_settings, dense_settings, family, artifact, output_dir, work_dir):
    """Verify every stream and pinned record, then open SwiGLU-7's 1B stream."""

    import torch
    from mlp_replacement.data import PackedTokenCache

    budget = int(settings["budget_tokens"])
    streams, sources = settings["streams"], settings["sources"]
    recipe = stacking_settings["recovery_1b"]
    source_root = resolve_path(Path(args.source_root))
    prepared_path, prepared, prepared_root = load_swiglu8_prepared(args.prepared, stacking_settings)
    block = pinned_json(source_root / "swiglu-8", "block-check-001/result.json",
                        sources["swiglu8_block_check_sha256"], "SwiGLU-8 block check")
    swiglu7_root = source_root / "swiglu-7" / "prepare-001"
    swiglu7 = pinned_json(swiglu7_root, "result.json", sources["swiglu7_prepare_sha256"], "SwiGLU-7 preparation")
    stream_record = swiglu7["results"]["stream"]
    final_record = prepared["results"]["long_context_data"]["kl_stream"]
    monitoring_record = prepared["results"]["monitoring_evaluation"]
    selection_record = block["results"]["long_context_data"]["selection_stream"]
    selection_checks, selection_path = checked_stream(args.selection_stream, selection_record, "selection_stream")
    enforce(artifact, output_dir, "inputs", [
        equality_check("inputs", stream_record["sha256"], streams["recovery_sha256"], metric="recovery_stream_sha256"),
        equality_check("inputs", int(stream_record["token_count"]), budget, metric="recovery_stream_tokens"),
        equality_check("inputs", int(stream_record["sequence_length"]), int(recipe["sequence_length"]),
                       metric="recovery_stream_sequence_length"),
        equality_check("inputs", selection_record["sha256"], streams["selection_sha256"], metric="selection_stream_sha256"),
        equality_check("inputs", final_record["sha256"], streams["final_kl_sha256"], metric="final_kl_stream_sha256"),
        equality_check("inputs", monitoring_record["sha256"], streams["monitoring_sha256"], metric="monitoring_sha256"),
        *selection_checks,
    ])
    # verified_stream rereads each file and stops on a size or sha256 mismatch.
    stream_path = verified_stream(swiglu7_root, stream_record)
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
        for key in ("recorded_1b", "recovery_1")
    }
    if any(value.get("status") != "completed" for value in recorded.values()):
        raise ValueError("Recovery-2 compares only with completed runs")
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


def check_start(task):
    """Stop unless token 0 reproduces the recorded 1B run and Recovery-1, and the teacher scores 0."""

    acceptance = task.settings["acceptance"]
    exact = float(acceptance["exact_relative"])
    self_kl = stream_kl(task.teacher, task.teacher, task.selection, task.length)
    enforce(task.artifact, task.output_dir, "teacher-self-kl", [{
        "check": "teacher-self-kl", "metric": "selection_stream_teacher_self_kl", "value": self_kl,
        "tolerance": float(acceptance["teacher_self_kl_absolute"]),
        "passed": abs(self_kl) <= float(acceptance["teacher_self_kl_absolute"]),
    }])
    row = score(task, 0, [0], 0, 0.0, None, [group["learning_rate"] for group in task.groups])
    trunk = task.artifact["results"]["trunk"] = {"history": [row], "tokens_seen": 0, "optimizer_updates": 0}
    trunk["trainable_scope"] = group_metadata(task.groups, task.model)
    checks = curve_checks([row], recorded_curve(task.family, task.recorded["recorded_1b"]), exact, "token-0")
    if not checks:
        raise ValueError("The recorded run has no token-0 row")
    enforce(task.artifact, task.output_dir, "token-0", checks)
    # Recovery-1 scored the same start on the same selection stream with the same code.
    earlier = task.recorded["recovery_1"]["results"]["trunk"]["history"][0]
    if int(earlier["actual_tokens"]) != 0:
        raise ValueError("Recovery-1's first trunk row is not its token-0 row")
    enforce(task.artifact, task.output_dir, "selection-token-0", [comparison(
        "selection-token-0", row["selection_kl"], earlier["selection_kl"], exact, metric="selection_kl")])


# Training


def recover(task, origin, end, cursor, updates, optimizer_state, schedule, on_checkpoint,
            decay_start_tokens=None, elapsed_seconds=0.0):
    """One call of the shared loop over the segment from ``origin`` to ``end`` of the recorded layout.

    Inside the loop, tokens are counted from ``origin``, as in ``recover_exact_segment``,
    which accepts only the constant rate. The 1-sqrt rate depends only on t - D and
    T - D, so shifting t, D, and T by the origin leaves every update's rate unchanged.
    With origin 0 this is Recovery-1's call.
    """

    import torch
    from mlp_replacement.compression.recovery import recover_trainable_by_tokens

    recipe = task.recipe
    decaying = decay_start_tokens is not None

    def forward_event(event, optimizer, first_step):
        on_checkpoint(replace(event, tokens_seen=origin + event.tokens_seen), optimizer, first_step)

    result = recover_trainable_by_tokens(
        student=task.model, teacher=task.teacher,
        parameter_groups=[{key: group[key] for key in ("name", "parameters", "learning_rate", "weight_decay")}
                          for group in task.groups],
        train_modules=task.train_modules,
        batch_at=lambda offset, count: task.batch_at(origin + offset, count),
        target_tokens=end - origin, schedule_tokens=end - origin,
        microbatch_tokens=recipe["sequence_length"] * recipe["microbatch_sequences"],
        accumulation_steps=recipe["gradient_accumulation_steps"],
        temperature=recipe["temperature"], ce_weight=recipe["ce_weight"],
        scheduler=task.settings["schedule"]["scheduler"] if decaying else recipe["scheduler"],
        warmup_fraction=0.0,
        final_lr_ratio=task.settings["schedule"]["final_lr_ratio"] if decaying else recipe["final_lr_ratio"],
        device=task.device, autocast_dtype=torch.bfloat16,
        start_tokens=cursor - origin, start_updates=updates, elapsed_seconds=elapsed_seconds,
        optimizer_state=optimizer_state,
        checkpoint_schedule=tuple((actual - origin, requested) for actual, requested in schedule if actual > cursor),
        on_checkpoint=forward_event, optimizer_backend=recipe["optimizer_backend"],
        decay_start_tokens=decay_start_tokens - origin if decaying else None,
    )
    return replace(result, tokens_seen=origin + result.tokens_seen)


def train_trunk(task, starts):
    """The constant run to 1B in its recorded layout; it keeps the state at every decay start."""

    step = int(task.recipe["effective_batch_tokens"])
    trunk = task.artifact["results"]["trunk"]
    endpoints = [int(value) for value in task.family["segment_endpoints"]]
    requests = trunk_requests(task.settings, task.recipe)
    snapshots, carried = {}, {}
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
        if event.tokens_seen in endpoints[:-1]:
            # The recorded run started its next segment from this optimizer state (SwiGLU-7, Stacking-1).
            carried["optimizer_state"] = cpu_tree(optimizer.state_dict())
        persist(task.output_dir, task.artifact, "trunk",
                {"family": task.family["name"], "tokens_seen": event.tokens_seen, "target_tokens": task.budget})
        print(f"family={task.family['name']} trunk tokens={event.tokens_seen:,} "
              f"selection={row['selection_kl']:.6f} legacy={row['legacy_kl']:.6f}", flush=True)

    origin, updates, optimizer_state, seconds, result = 0, 0, None, 0.0, None
    for end in endpoints:
        schedule = segment_schedule(origin, end, requests, step)
        result = recover(task, origin, end, origin, updates, optimizer_state, schedule, on_checkpoint,
                         elapsed_seconds=seconds)
        if result.tokens_seen != end:
            raise ValueError(f"The trunk segment ending at {end:,} did not finish")
        optimizer_state = carried.pop("optimizer_state", None)
        if end != endpoints[-1] and optimizer_state is None:
            raise ValueError(f"No optimizer state was kept at the segment end {end:,}")
        origin, updates, seconds = end, result.optimizer_updates, result.elapsed_seconds
    if result.tokens_seen != task.budget or set(snapshots) != set(starts):
        raise ValueError("The trunk did not reach 1B with every decay start kept")
    if any(snapshots[percent]["tokens"] != row["tokens"] for percent, row in starts.items()):
        raise ValueError("A decay start is not on the update boundary its schedule uses")
    curve = recorded_curve(task.family, task.recorded["recorded_1b"])
    record_checks(task.artifact, task.output_dir, "trunk-curve", curve_checks(
        trunk["history"][1:], curve, float(task.settings["acceptance"]["review_relative"]), "curve"))
    return snapshots


def train_branch(task, percent, start, snapshot):
    """Restore the trunk at the decay start and decay to 1B; nothing else differs from the trunk."""

    step = int(task.recipe["effective_batch_tokens"])
    branches = task.artifact["results"].setdefault("branches", {})
    branch = branches[str(percent)] = {
        "decay_percent": percent, "decay_start_requested": start["requested"],
        "decay_start_tokens": start["tokens"], "segment_origin": start["origin"],
        "start_updates": snapshot["updates"], "history": [],
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

    requests = [value for value in trunk_requests(task.settings, task.recipe) if value > start["requested"]]
    schedule = segment_schedule(start["origin"], task.budget, requests, step)
    # The loop loads and then clears this state; each snapshot serves one branch only.
    result = recover(task, start["origin"], task.budget, start["tokens"], snapshot["updates"],
                     snapshot.pop("optimizer_state"), schedule, on_checkpoint, decay_start_tokens=start["tokens"])
    if result.tokens_seen != task.budget:
        raise ValueError(f"The {percent}% decay did not reach 1B")


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
        starts = decay_starts(settings, task.recipe, family)
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


def family_decision(trunk_rows, branches, rule, metric="selection_kl", reference="median"):
    """Apply the winner rule to one family (decision log, 2026-11-07 and 2026-11-08).

    The reference is the trunk's median (Recovery-2) or lowest (Recovery-1) value at
    the four reference rows. A decay qualifies if its 1B value is more than the bar
    below the reference. The shortest decay within the bar of the best qualifier wins,
    because it stays closest to constant; with none, constant stays.
    """

    bar = float(rule["bar"])
    wanted = [int(value) for value in rule["reference_requests"]]
    rows = {last_request(row): float(row[metric]) for row in trunk_rows if last_request(row) in wanted}
    if set(rows) != set(wanted):
        raise ValueError("The trunk lacks a reference row")
    value = float(REDUCERS[reference]([rows[tokens] for tokens in wanted]))
    candidates = []
    for percent, branch in sorted(branches.items(), key=lambda item: int(item[0])):
        final = float(branch["history"][-1][metric])
        candidates.append({
            "decay_percent": int(percent), "value": final, "gain": 1.0 - final / value,
            "qualifies": final < value * (1.0 - bar),
        })
    qualifiers = [row for row in candidates if row["qualifies"]]
    winner = None
    if qualifiers:
        best = min(row["value"] for row in qualifiers)
        winner = min((row for row in qualifiers if row["value"] <= best * (1.0 + bar)),
                     key=lambda row: row["decay_percent"])
    return {
        "metric": metric,
        "reference_kind": reference,
        "reference": value,
        "reference_rows": {str(tokens): rows[tokens] for tokens in wanted},
        "trunk_1b": float(trunk_rows[-1][metric]),
        "candidates": candidates,
        "decay_percent": None if winner is None else winner["decay_percent"],
    }


def summarize(args, settings):
    """Apply the winner rule to the three finished families; safe to re-run."""

    root = resolve_path(args.results_root)
    families = settings["families"]
    artifacts = [read_json(path) for path in finished_outputs(root, [output_name(family) for family in families])]
    if any(artifact.get("workflow") != WORKFLOW or artifact.get("status") != "completed" for artifact in artifacts):
        raise ValueError("Every Recovery-2 family must have completed")
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
        selection = family_decision(trunk, branches, rule, "selection_kl", rule["reference"])
        legacy = family_decision(trunk, branches, rule, "legacy_kl", rule["reference"])
        flags = family_flags(artifact, selection, legacy)
        curve_check = artifact["results"]["acceptance"]["trunk-curve"]["checks"]
        held = bool(flags) and family["name"] not in accepted
        results[family["name"]] = {
            "decision": "held: review the flags" if held else (
                "constant" if selection["decay_percent"] is None else f"decay {selection['decay_percent']}%"),
            "decay_percent": None if held else selection["decay_percent"],
            "selection": selection,
            # Recovery-1's lowest-of-four reference, reported and never deciding (decision log, 2026-11-08).
            "lowest_of_four": family_decision(trunk, branches, rule, "selection_kl", "lowest"),
            "legacy": legacy,
            "flags": flags,
            "flags_accepted": bool(flags) and family["name"] in accepted,
            # Two runs of one configuration: the trunk and the recorded constant 1B run.
            "noise_sample": max((row["relative_difference"] for row in curve_check), default=None),
            "recorded_1b": {key: family["recorded_1b"][key] for key in ("workflow", "run")},
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
        rows = ", ".join(f"{int(tokens) // 1_000_000}M {value:.6f}" for tokens, value in selection["reference_rows"].items())
        print(f"{name}: {entry['decision']} (median reference {selection['reference']:.6f} of {rows}; "
              f"noise sample {entry['noise_sample']})")
        for row in selection["candidates"]:
            mark = "qualifies" if row["qualifies"] else "no"
            print(f"  decay {row['decay_percent']}%: {row['value']:.6f}, gain {row['gain']:+.2%}, {mark}")
        lowest = entry["lowest_of_four"]
        label = "constant" if lowest["decay_percent"] is None else f"decay {lowest['decay_percent']}%"
        print(f"  lowest-of-four reference {lowest['reference']:.6f} would give: {label} (reported only)")
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
    train_parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--results-root", type=Path, required=True)
    summary_parser.add_argument("--accept-flags", nargs="*", default=[],
                                help="Families whose flags the researcher has reviewed")
    for subparser in (train_parser, summary_parser):
        subparser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
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
