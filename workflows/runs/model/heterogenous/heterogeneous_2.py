"""Run Heterogeneous-2 Step 1: forced models trained to 100M and scored by KL.

Heterogeneous-1 picked no new operator, so this step trains the models it forced:
eight layers must use a non-SwiGLU block and the knapsack solves the rest. ``train``
runs one configured model. It rebuilds the model that Heterogeneous-1's ``assemble``
recorded, refits its blocks, stops unless token 0 reproduces the record, trains with
SwiGLU-8's recipe at the run's learning rate, and scores the 8K KL. ``summarize``
applies the Step 2 rule and the learning-rate reading on the login node. The binding
design is in ``docs/agents/decisions.md`` and
``docs/experiments/model/heterogenous/heterogeneous-2.md``.
"""

from __future__ import annotations

import argparse
import math
from copy import deepcopy
from pathlib import Path

import torch

from workflows.runs.model.common import resolve_path
from workflows.runs.model.heterogenous.heterogeneous_1 import (
    DEFAULT_SWIGLU8_ROOT,
    assemble_output_name,
    completed_table_rows,
    fit_needed,
    indexed,
    load_settings as load_heterogeneous1_settings,
    menu_prices,
    open_task,
    replaced_items,
    score_point,
)
from workflows.runs.model.swiglu.shared import recovery_memory_record
from workflows.runs.model.swiglu.swiglu_7 import (
    code_hashes,
    configure_strategy,
    desired_requests,
    group_metadata,
    output_path,
    persist,
    resolve_storage,
    utc_now,
)
from workflows.runs.model.swiglu.swiglu_8 import (
    REPRODUCTION_REVIEW_RELATIVE,
    AcceptanceFailure,
    comparison,
    enforce,
    equality_check,
    is_digest,
    pinned_json,
    record_checks,
    record_failure,
    verified_stream,
)
from workflows.runs.model.swiglu.swiglu_9 import finished_outputs, write_once

from mlp_replacement.artifacts import file_digest, read_json
from mlp_replacement.compression.continuation import recover_exact_segment, segment_schedule
from mlp_replacement.compression.surgery import temporary_fp32_replacements
from mlp_replacement.compression.teacher_cache import cache_teacher_logits
from mlp_replacement.data import PackedTokenCache
from mlp_replacement.evaluation.mixed_precision import evaluate_validation_kl_mixed
from mlp_replacement.evaluation.teacher_kl import evaluate_teacher_kl
from mlp_replacement.runlog import environment_record


WORKFLOW = "heterogeneous-2"
SCHEMA_VERSION = 1
EXPERIMENT_CLASS = "forced-models-after-recovery"
DEFAULT_CONFIG = Path("workflows/configs/model/heterogenous/heterogeneous-2.json")
DEFAULT_HETEROGENEOUS1_ROOT = Path("data/results/workflows/model/heterogeneous-1")
STRATEGY = "S7-0"
SWIGLU_MODEL = "swiglu-8"
FORCED_MODEL = "forced-8"
TOKEN_ZERO_METRICS = ("monitoring_kl", "kl_128", "kl_8192")


# Configuration and artifacts


def is_control(run, reference):
    """The control is SwiGLU-8's own model at SwiGLU-8's own rate."""

    return run["model"] == SWIGLU_MODEL and run["learning_rate"] == reference["recovery"]["learning_rate"]


def load_settings(path):
    """Load the run table and Heterogeneous-1's configuration, which carries SwiGLU-8's recipe."""

    path = resolve_path(Path(path))
    settings = read_json(path)
    if settings.get("schema_version") != SCHEMA_VERSION or settings.get("workflow") != WORKFLOW:
        raise ValueError("Expected a schema-1 Heterogeneous-2 configuration")
    reference, unused_path = load_heterogeneous1_settings(settings["heterogeneous1_config"])
    runs = settings["runs"]
    pins = settings["sources"]["heterogeneous1_assemble_sha256"]
    if (
        len({run["name"] for run in runs}) != len(runs)
        or any(run["model"] not in (SWIGLU_MODEL, FORCED_MODEL) for run in runs)
        or {str(run["target"]) for run in runs} != set(pins)
        or not all(is_digest(value) for value in pins.values())
    ):
        raise ValueError("Every Heterogeneous-2 run needs a unique name, a known model, and a pinned assembly")
    if [run for run in runs if is_control(run, reference)] != runs[:1]:
        raise ValueError("The first Heterogeneous-2 run must be the only control")
    return settings, reference, path


def run_output_name(name):
    return f"{name}-run-001"


def start_artifact(settings, run, config_path):
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "experiment_family": WORKFLOW,
        "experiment_class": EXPERIMENT_CLASS,
        "command": "train",
        "identity": dict(run),
        "status": "running",
        "created_at_utc": utc_now(),
        "configuration": deepcopy(settings),
        "configuration_sha256": file_digest(config_path),
        "heterogeneous1_configuration_sha256": file_digest(resolve_path(Path(settings["heterogeneous1_config"]))),
        "environment": environment_record(),
        "code_hashes": code_hashes(),
        "results": {},
        "error": None,
    }


# Fidelity checks


def monitoring_curve(history):
    """Map each requested token count to its monitoring KL."""

    return {
        int(token): float(row["recovery_validation_kl"])
        for row in history
        for token in row["requested_tokens"]
    }


def token_zero_checks(start, point, recorded_history, run, tolerance):
    """Compare a rebuilt model with its assembly record and, for SwiGLU-8's model, with SwiGLU-8's run."""

    checks = [
        comparison("token-0", start[metric], point[metric], tolerance, metric=metric)
        for metric in TOKEN_ZERO_METRICS
    ]
    if run["model"] == SWIGLU_MODEL:
        checks.append(comparison("token-0", start["monitoring_kl"], recorded_history[0]["recovery_validation_kl"],
                                 tolerance, metric="swiglu8_monitoring_kl"))
    return checks


def curve_checks(history, recorded_history, kl, recorded_kl, limit):
    """Compare the control run with SwiGLU-8's recorded run at every point after token 0."""

    recorded = monitoring_curve(recorded_history)
    checks = [
        comparison("curve", row["recovery_validation_kl"], recorded[int(row["requested_tokens"][-1])], limit,
                   metric="monitoring_kl", tokens=row["actual_tokens"])
        for row in history[1:]
    ]
    return [*checks, comparison("curve", kl, recorded_kl, limit, metric="kl_8192")]


# Training


def recover_point(task, run, start, modules, rows, schedule, recorded_history):
    """Train one checked model with SwiGLU-8's loop; only the learning rate can differ.

    The rate goes into a copy, because Heterogeneous-1's table check compares the
    original settings with the configuration its tables recorded.
    """

    settings = task.settings
    recovery = settings["recovery"]
    context = task.context
    artifact = task.artifact
    temperature = float(recovery["temperature"])
    rate = float(run["learning_rate"])
    rate_settings = {**settings, "recovery": {**recovery, "learning_rate": rate}}
    replacements = {item[0]: modules[item] for item in replaced_items(rows)}
    paths = [context.blocks[layer].path for layer in sorted(replacements)]
    recorded = monitoring_curve(recorded_history)
    with temporary_fp32_replacements(context.model, context.blocks, replacements):
        groups, train_modules, unused_adapters = configure_strategy(context.model, paths, STRATEGY, rate_settings)
        scope = group_metadata(groups, context.model)
        artifact["results"]["trainable_scope"] = scope
        enforce(artifact, task.output_dir, "trainable-parameters", [equality_check(
            "trainable-parameters", scope["trainable_parameters"],
            sum(int(row["replacement_parameters"]) for row in rows if not row["retains_dense_module"]),
        )])
        results = artifact["results"]["recovery"] = {
            "validation_history": [{
                "actual_tokens": 0,
                "requested_tokens": [0],
                "optimizer_updates": 0,
                "training_seconds": 0.0,
                "mean_train_kl": None,
                "recovery_validation_kl": start["monitoring_kl"],
                "learning_rates": [group["learning_rate"] for group in groups],
            }],
            "tokens_seen": 0,
            "optimizer_updates": 0,
            "training_seconds": 0.0,
        }

        def on_checkpoint(event, unused_optimizer, first_step):
            validation_kl = evaluate_validation_kl_mixed(context.model, task.validation_cache, temperature,
                                                         context.device)
            if not math.isfinite(validation_kl) or not math.isfinite(event.mean_train_kl):
                raise FloatingPointError("Non-finite recovery loss; the run stopped")
            results["validation_history"].append({
                "actual_tokens": event.tokens_seen,
                "requested_tokens": list(event.requested_checkpoint_tokens),
                "optimizer_updates": event.optimizer_updates,
                "training_seconds": event.elapsed_seconds,
                "mean_train_kl": event.mean_train_kl,
                "recovery_validation_kl": validation_kl,
                "learning_rates": list(event.learning_rates),
            })
            results.update({
                "tokens_seen": event.tokens_seen,
                "optimizer_updates": event.optimizer_updates,
                "training_seconds": event.elapsed_seconds,
                "first_step": results.get("first_step") or first_step,
                "memory": recovery_memory_record(context.device),
            })
            persist(task.output_dir, artifact, "recovery",
                    {"name": run["name"], "tokens_seen": event.tokens_seen,
                     "target_tokens": int(recovery["target_tokens"])})
            swiglu8 = recorded[int(event.requested_checkpoint_tokens[-1])]
            print(f"run={run['name']} tokens={event.tokens_seen:,} KL={validation_kl:.6f} "
                  f"swiglu8={swiglu8:.6f} gap={gap(validation_kl, swiglu8):+.2%}", flush=True)

        result = recover_exact_segment(
            origin=0,
            end=int(recovery["target_tokens"]),
            cursor=0,
            schedule=schedule,
            student=context.model,
            teacher=task.teacher,
            parameter_groups=[
                {key: group[key] for key in ("name", "parameters", "learning_rate", "weight_decay")}
                for group in groups
            ],
            train_modules=train_modules,
            batch_at=lambda offset, count: task.cache.batch(
                offset, count, recovery["microbatch_sequences"], allow_partial_sequence=True
            ),
            microbatch_tokens=recovery["sequence_length"] * recovery["microbatch_sequences"],
            accumulation_steps=recovery["gradient_accumulation_steps"],
            temperature=recovery["temperature"],
            ce_weight=recovery["ce_weight"],
            scheduler=recovery["scheduler"],
            warmup_fraction=recovery["warmup_fraction"],
            final_lr_ratio=recovery["final_lr_ratio"],
            device=context.device,
            autocast_dtype=torch.bfloat16,
            start_updates=0,
            elapsed_seconds=0.0,
            optimizer_state=None,
            on_checkpoint=on_checkpoint,
            optimizer_backend=recovery["optimizer_backend"],
        )
        results.update({"tokens_seen": result.tokens_seen, "optimizer_updates": result.optimizer_updates,
                        "training_seconds": result.elapsed_seconds})
        persist(task.output_dir, artifact, "teacher_kl_8192")
        artifact["results"]["teacher_kl_8192"] = evaluate_teacher_kl(
            context.model, task.teacher, task.kl_path, task.kl_record, task.length, device=context.device
        )


def train_model(task, run, point, modules, recorded_history):
    """Stop on any start mismatch, then train; nothing trains until token 0 is reproduced."""

    recovery = task.settings["recovery"]
    start = score_point(task, modules, point, task.validation_cache)
    task.artifact["results"]["start"] = start
    enforce(task.artifact, task.output_dir, "token-0", token_zero_checks(
        start, point, recorded_history, run, task.settings["acceptance"]["exact_relative"]
    ))
    schedule = segment_schedule(0, int(recovery["target_tokens"]), desired_requests(task.settings),
                                recovery["effective_batch_tokens"])
    enforce(task.artifact, task.output_dir, "schedule", [equality_check(
        "schedule",
        [[actual, list(requested)] for actual, requested in schedule],
        [[row["actual_tokens"], row["requested_tokens"]] for row in recorded_history[1:]],
    )])
    recover_point(task, run, start, modules, point["rows"], schedule, recorded_history)


def attach_inputs(task, reference, swiglu8_root):
    """Add SwiGLU-8's recovery stream, monitoring batches, and 8K KL stream to the task."""

    prepared = task.prepared["results"]
    root = resolve_path(Path(swiglu8_root)) / "prepare-001"
    stream = prepared["stream"]
    task.cache = PackedTokenCache(verified_stream(root, stream), int(stream["token_count"]),
                                  reference["recovery"]["sequence_length"], stream["sha256"])
    task.kl_record = prepared["long_context_data"]["kl_stream"]
    task.kl_path = verified_stream(root, task.kl_record)
    record = prepared["monitoring_evaluation"]
    path = output_path(root, record["path"])
    if file_digest(path) != record["sha256"]:
        raise ValueError("SwiGLU-8's monitoring batches changed")
    monitoring = torch.load(path, map_location="cpu", weights_only=False)
    task.validation_cache = cache_teacher_logits(
        task.teacher, monitoring["recovery_validation"], len(monitoring["recovery_validation"]),
        task.context.device, "float16",
    )


def train(args, settings, reference, config_path):
    """Run one configured model: rebuild it, check token 0, train, and score the 8K KL.

    A control outside the 1% limit keeps training, because one transient must not
    cost the curve. Its result stays complete, and only the job fails, so the runs
    that wait for it do not start until the researcher has reviewed the curve.
    """

    run = indexed(settings["runs"], args.task_index, "Train")
    target = str(run["target"])
    work_dir, output_dir = resolve_storage(args.work_dir, args.output_dir, False)
    artifact = start_artifact(settings, run, config_path)
    curve = None
    try:
        assembled = pinned_json(args.heterogeneous1_root, f"{assemble_output_name(target)}/result.json",
                                settings["sources"]["heterogeneous1_assemble_sha256"][target],
                                f"Heterogeneous-1's assembly at {target}")
        point = next(value for value in assembled["results"]["points"] if value["name"] == run["model"])
        swiglu8 = pinned_json(args.swiglu8_root, f"fitted-target-{target}-run-001/result.json",
                              reference["sources"]["swiglu8_fitted_runs_sha256"][target],
                              f"SwiGLU-8's run at {target}")
        recorded_history = swiglu8["results"]["recovery"]["validation_history"]
        persist(output_dir, artifact, "loading_models")
        task = open_task(reference, work_dir, output_dir, artifact, "train", args.swiglu8_root)
        table_rows, unused_inputs = completed_table_rows(resolve_path(args.heterogeneous1_root), reference)
        prices = menu_prices(task.prepared["results"]["width_curves"]["rows"], table_rows)
        modules = {}
        checks = []
        fit_needed(task, modules, point["rows"], prices, checks)
        enforce(artifact, output_dir, "refit-prices", checks)
        attach_inputs(task, reference, args.swiglu8_root)
        train_model(task, run, point, modules, recorded_history)
        if is_control(run, reference):
            curve = record_checks(artifact, output_dir, "curve", curve_checks(
                artifact["results"]["recovery"]["validation_history"], recorded_history,
                artifact["results"]["teacher_kl_8192"]["kl"],
                swiglu8["results"]["evaluation"]["teacher_kl_8192"]["kl"], REPRODUCTION_REVIEW_RELATIVE,
            ))
        artifact["status"] = "completed"
        persist(output_dir, artifact, "completed")
    except BaseException as error:
        record_failure(output_dir, work_dir, artifact, error)
        raise
    if curve is not None and not curve["passed"]:
        raise AcceptanceFailure("The control left the 1% limit; review it before the other runs start")


# Reading the results


def gap(value, reference):
    """Relative gap; positive means ``value`` is worse than its reference."""

    return float(value) / float(reference) - 1.0


def band(value, bar):
    if value < -bar:
        return "better"
    return "level" if value <= bar else "worse"


def run_record(history, kl):
    return {"curve": monitoring_curve(history), "kl_8192": float(kl)}


def step_two_reading(forced, swiglu, reading):
    """Apply the Step 2 rule to one target, and cross-check its band with the 8K KL."""

    bar = float(reading["bar"])
    decision = int(reading["decision_tokens"])
    earlier = int(reading["shrink_from_tokens"])
    final_gap = gap(forced["curve"][decision], swiglu["curve"][decision])
    earlier_gap = gap(forced["curve"][earlier], swiglu["curve"][earlier])
    shrink = earlier_gap - final_gap
    position = band(final_gap, bar)
    long_gap = gap(forced["kl_8192"], swiglu["kl_8192"])
    long_position = band(long_gap, bar)
    return {
        "gap_at_decision": final_gap,
        "gap_at_shrink_start": earlier_gap,
        "shrink": shrink,
        "band": position,
        "go_to_1b": position != "worse" or shrink > bar,
        "gap_8k": long_gap,
        "band_8k": long_position,
        "flag_8k_disagrees": long_position != position,
    }


def rate_reading(forced_recipe, forced_high, swiglu_recipe, swiglu_high, reading):
    """Compare each family at its better rate, so a rate that hurts one family cannot flatter the other."""

    bar = float(reading["bar"])
    decision = int(reading["decision_tokens"])
    forced_best = min((forced_recipe, forced_high), key=lambda record: record["curve"][decision])
    swiglu_best = min((swiglu_recipe, swiglu_high), key=lambda record: record["curve"][decision])
    recipe_gap = gap(forced_recipe["curve"][decision], swiglu_recipe["curve"][decision])
    better_gap = gap(forced_best["curve"][decision], swiglu_best["curve"][decision])
    shift = recipe_gap - better_gap
    swiglu_gain = -gap(swiglu_high["curve"][decision], swiglu_recipe["curve"][decision])
    return {
        "gap_at_recipe_rate": recipe_gap,
        "gap_at_better_rates": better_gap,
        "shift_toward_new_blocks": shift,
        "favours_new_blocks": shift > bar,
        "forced_prefers_high_rate": forced_best is forced_high,
        "swiglu_prefers_high_rate": swiglu_best is swiglu_high,
        "swiglu_gain_at_high_rate": swiglu_gain,
        "flag_swiglu_recipe_limit": swiglu_gain > bar,
        "step_two_at_better_rates": step_two_reading(forced_best, swiglu_best, reading),
    }


def target_entry(forced, swiglu, forced_high, swiglu_high, reading):
    """Read one target. A check that favours new blocks replaces the decision with its better-rate reading."""

    decision = step_two_reading(forced, swiglu, reading)
    entry = {
        "gap_by_tokens": {
            str(tokens): gap(forced["curve"][tokens], swiglu["curve"][tokens]) for tokens in sorted(swiglu["curve"])
        },
        "at_recipe_rate": decision,
        "rate_check": None,
    }
    if forced_high is not None:
        entry["rate_check"] = rate_reading(forced, forced_high, swiglu, swiglu_high, reading)
        if entry["rate_check"]["favours_new_blocks"]:
            decision = entry["rate_check"]["step_two_at_better_rates"]
    entry["decision"] = decision
    return entry


def summarize(args, settings, reference):
    """Apply the reading rules to the finished runs; safe to re-run."""

    root = resolve_path(args.results_root)
    runs = settings["runs"]
    reading = settings["reading"]
    paths = finished_outputs(root, [run_output_name(run["name"]) for run in runs])
    artifacts = [read_json(path) for path in paths]
    if any(artifact.get("workflow") != WORKFLOW or artifact.get("status") != "completed" for artifact in artifacts):
        raise ValueError("Every Heterogeneous-2 run must have completed")
    if any(artifact["code_hashes"] != artifacts[0]["code_hashes"] for artifact in artifacts):
        raise ValueError("The control can only vouch for runs that used its code")
    records = {
        (run["model"], str(run["target"]), float(run["learning_rate"])): run_record(
            artifact["results"]["recovery"]["validation_history"], artifact["results"]["teacher_kl_8192"]["kl"]
        )
        for run, artifact in zip(runs, artifacts)
    }
    recipe_rate = float(reference["recovery"]["learning_rate"])
    high_rate = float(reading["high_learning_rate"])
    targets = {}
    for target in sorted({str(run["target"]) for run in runs if run["model"] == FORCED_MODEL}):
        swiglu8 = pinned_json(args.swiglu8_root, f"fitted-target-{target}-run-001/result.json",
                              reference["sources"]["swiglu8_fitted_runs_sha256"][target],
                              f"SwiGLU-8's run at {target}")
        targets[target] = target_entry(
            records[(FORCED_MODEL, target, recipe_rate)],
            run_record(swiglu8["results"]["recovery"]["validation_history"],
                       swiglu8["results"]["evaluation"]["teacher_kl_8192"]["kl"]),
            records.get((FORCED_MODEL, target, high_rate)),
            records.get((SWIGLU_MODEL, target, high_rate)),
            reading,
        )
    curve = artifacts[0]["results"]["acceptance"]["curve"]
    results = {
        "control": {
            "passed": curve["passed"],
            "largest_relative_difference": max(row["relative_difference"] for row in curve["checks"]),
            "checks": curve["checks"],
        },
        "targets": targets,
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
        "results": results,
        "error": None,
    }
    write_once(root / "summary-001", summary, "summary")
    print_summary(results)


def print_summary(results):
    control = results["control"]
    label = "within the limit" if control["passed"] else "FLAGGED: review it before using any result"
    print(f"control curve: {label}, largest difference {control['largest_relative_difference']:.3%}")
    for target, entry in results["targets"].items():
        decision = entry["decision"]
        outcome = "go to 1B" if decision["go_to_1b"] else "stop"
        print(f"target {target}: gap {decision['gap_at_decision']:+.2%} ({decision['band']}), "
              f"shrink {decision['shrink']:+.2%}, 8K gap {decision['gap_8k']:+.2%} -> {outcome}")
        if decision["flag_8k_disagrees"]:
            print(f"  FLAG: the 8K KL puts this target in another band ({decision['band_8k']}); you decide")
        check = entry["rate_check"]
        if check is None:
            continue
        print(f"  rate check: gap {check['gap_at_recipe_rate']:+.2%} at the recipe rate, "
              f"{check['gap_at_better_rates']:+.2%} at better rates, shift {check['shift_toward_new_blocks']:+.2%}")
        if check["favours_new_blocks"]:
            print("  FAVOURS NEW BLOCKS: this decision uses the better rates; the 1e-4 runs at the "
                  "other targets wait for your go")
        if check["flag_swiglu_recipe_limit"]:
            print(f"  FLAG: SwiGLU itself gains {check['swiglu_gain_at_high_rate']:.2%} at the high rate; "
                  "report it as a limit of its recipe")


# Command line


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--task-index", type=int, required=True)
    train_parser.add_argument("--work-dir", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--heterogeneous1-root", type=Path, default=DEFAULT_HETEROGENEOUS1_ROOT)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--results-root", type=Path, required=True)
    for subparser in (train_parser, summary_parser):
        subparser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        subparser.add_argument("--swiglu8-root", type=Path, default=DEFAULT_SWIGLU8_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    settings, reference, config_path = load_settings(args.config)
    if args.command == "summarize":
        summarize(args, settings, reference)
    else:
        train(args, settings, reference, config_path)


if __name__ == "__main__":
    main()
