"""Run SwiGLU-8: local-fitting attribution and the 20-80% control spectrum.

``block-check`` compares local-fitting settings on three layers with three seeds
and selects on a dedicated 8K selection stream (Part A).
``prepare`` builds the shared neuron selection, width curves, allocations, and
both arms' token-zero starts (Part B). ``train`` recovers one arm at one target
to 100M tokens and runs the final evaluation. The binding design is recorded in
``docs/agents/decisions.md`` and ``docs/experiments/model/swiglu/swiglu-8.md``.
"""

from __future__ import annotations

import argparse
import gc
import math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from workflows.runs.model.common import relative_to_root, resolve_path

from mlp_replacement.artifacts import (
    atomic_torch_save,
    content_digest,
    file_digest,
    read_json,
    sha256_file,
)
from mlp_replacement.compression.allocation import (
    allocate_discrete_swiglu_widths,
    build_swiglu_width_curve,
)
from mlp_replacement.compression.continuation import (
    capture_rng,
    commit_single_checkpoint,
    recover_exact_segment,
    restore_checkpoint,
    restore_rng,
    save_tensor_atomic,
    segment_origin,
    segment_schedule,
)
from mlp_replacement.compression.reconstruction import (
    build_swiglu_student,
    load_replacement_state,
)
from mlp_replacement.compression.surgery import temporary_fp32_replacements
from mlp_replacement.compression.teacher_cache import cache_teacher_logits
from mlp_replacement.config import DatasetSpec
from mlp_replacement.data import PackedTokenCache, load_text_dataset, make_token_loader
from mlp_replacement.data_streams import build_finite_stream
from mlp_replacement.evaluation import evaluate_operator
from mlp_replacement.evaluation.bundles import load_bundle, validate_bundle
from mlp_replacement.evaluation.mixed_precision import evaluate_teacher_cache_mixed
from mlp_replacement.evaluation.teacher_kl import (
    TEACHER_KL_PROTOCOL,
    build_teacher_kl_stream,
    evaluate_teacher_kl,
)
from mlp_replacement.model import discover_mlp_blocks, load_model_and_tokenizer
from mlp_replacement.operators import (
    GatedMLPReplacement,
    initialize_gated_mlp_from_teacher,
    refit_down_projection_least_squares,
    swiglu_neuron_importance_scores,
)
from mlp_replacement.runlog import environment_record

from .shared import (
    build_local_data,
    evaluate_lm_mixed,
    evaluate_validation_kl_mixed,
    make_model_config,
    recovery_memory_record,
    resolve_source_asset,
)
from .swiglu5.fitting import (
    capture_dense_pairs,
    fit_key,
    fit_operator,
    load_fit_operator,
    operator_state_path,
)
from .swiglu_7 import (
    code_hashes,
    configure_strategy,
    cpu_tree,
    desired_requests,
    effective_preparation_settings,
    evaluate_final_model,
    group_metadata,
    load_trainable_state,
    measure_bundle,
    output_path,
    persist,
    prepare_dense_evaluation,
    prepare_evaluation_protocol,
    prepare_monitoring_evaluation,
    prepare_recovery_stream,
    relative_output,
    resolve_storage,
    storage_preflight,
    trainable_state,
    utc_now,
    validate_evaluation_preflight,
    validated_attention_implementation,
)


WORKFLOW = "swiglu-8"
SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path("workflows/configs/model/swiglu/swiglu-8.json")
SWIGLU7_CONFIG = Path("workflows/configs/model/swiglu/swiglu-7.json")
QUANTIZATION_CONFIG = Path("workflows/configs/model/baseline/quantization-1.json")
DEFAULT_SWIGLU7_ROOT = Path("data/results/workflows/model/swiglu-7")
MEASURE_MODULE = "workflows.runs.model.swiglu.swiglu_8"
TARGETS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8)
SWIGLU7_TARGETS = (0.2, 0.3, 0.4, 0.5)
ARMS = ("fitted", "slice")
CONTEXTS = (128, 8192)
PART_A_SEEDS = (21, 22, 23)
# SwiGLU-7's recipe at seed 21 is the anchor that A1-A3 compare with SwiGLU-7.
CURRENT_SETTING = "c1e-3-128"
CURRENT_SEED = 21
SELECTION_KL_PROTOCOL = "teacher-kl-t1-all-valid-positions-c4-shard0-selection-24x8192-v1"


def part_a_settings():
    """Build the decided Part A grid (decision log, 2026-10-11).

    Every combination of training context, schedule, and learning rate is
    tested, plus batch 256 with the current recipe and the exact least-squares
    down projection at both contexts. Cosine decays over a fixed 32 epochs; a
    patience equal to that budget never stops a fit early, and the best
    validation epoch is kept as for every other setting.
    """

    grid = {}
    for context, suffix in ((128, "128"), (8192, "8k")):
        for scheduler, prefix, epochs, patience in (("constant", "c", 64, 3), ("cosine", "cos", 32, 32)):
            for rate, label in ((0.001, "1e-3"), (0.0003, "3e-4"), (0.0001, "1e-4")):
                grid[f"{prefix}{label}-{suffix}"] = {
                    "method": "gradient",
                    "calibration_context": context,
                    "learning_rate": rate,
                    "scheduler": scheduler,
                    "batch_size": 2048,
                    "max_epochs": epochs,
                    "early_stopping_patience": patience,
                }
    grid[f"{CURRENT_SETTING}-b256"] = {**grid[CURRENT_SETTING], "batch_size": 256}
    for context, suffix in ((128, "128"), (8192, "8k")):
        grid[f"ols-{suffix}"] = {"method": "least_squares", "calibration_context": context}
    return grid


PART_A_SETTINGS = part_a_settings()
ADOPTION_RULE = {
    "deciding_context": 8192,
    "minimum_mean_reduction": 0.05,
    "maximum_combination_increase": 0.05,
    "seed_noise": "sum of the two settings' sample seed standard deviations over the current mean",
}
ACCEPTANCE = {
    "exact_relative": 0.0001,
    "fit_relative": 0.001,
    "teacher_self_kl_absolute": 1e-06,
}


class AcceptanceFailure(RuntimeError):
    """Stop a stage whose output does not reproduce its recorded reference."""


# Configuration and source artifacts


def is_digest(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_settings(path):
    path = resolve_path(Path(path))
    settings = read_json(path)
    if settings.get("schema_version") != SCHEMA_VERSION or settings.get("workflow") != WORKFLOW:
        raise ValueError("Expected a schema-1 SwiGLU-8 configuration")
    if (
        settings.get("seed") != 21
        or settings.get("targets") != list(TARGETS)
        or settings.get("arms") != list(ARMS)
    ):
        raise ValueError("SwiGLU-8 fixes seed 21, the eight targets, and both arms")
    reference = read_json(resolve_path(SWIGLU7_CONFIG))
    for key in ("model", "data", "calibration", "local_fitting", "evaluation"):
        if settings.get(key) != reference[key]:
            raise ValueError(f"SwiGLU-8 must reuse SwiGLU-7's {key} contract")
    if settings.get("strategies") != {"S7-0": reference["strategies"]["S7-0"]}:
        raise ValueError("SwiGLU-8 recovers with SwiGLU-7's replacement-only scope")
    recovery = {
        **reference["recovery"],
        "target_tokens": 100_000_000,
        "segment_endpoints": [100_000_000],
        "monitoring_ppl_tokens": [10_000_000, 25_000_000, 50_000_000, 100_000_000],
    }
    if settings.get("recovery") != recovery:
        raise ValueError("SwiGLU-8 fixes the S7-0 recovery recipe at 100M tokens")
    allocation = {
        **reference["allocation"],
        "width_ratios": [0.0625, 0.125, *reference["allocation"]["width_ratios"]],
    }
    if settings.get("allocation") != allocation:
        raise ValueError("SwiGLU-8 extends SwiGLU-7's allocation with widths 0.0625 and 0.125")
    if settings.get("preparation") != {
        "initialization": "legacy_subset",
        "allocation_method": "discrete_width_curve",
        "start_tokens": 0,
    }:
        raise ValueError("SwiGLU-8 fixes teacher-subset starts and discrete allocation")
    quantization = read_json(resolve_path(QUANTIZATION_CONFIG))["data"]
    long_context = settings.get("long_context", {})
    if long_context != {
        "source": quantization["kl_source"],
        "sequence_length": quantization["sequence_length"],
        "kl_sequences": quantization["kl_sequences"],
        "selection_sequences": quantization["kl_sequences"],
        "calibration_sequences": 48,
        "validation_sequences": 1,
    }:
        raise ValueError(
            "SwiGLU-8 uses Quantization-1's 8K KL source, an equal-size 8K selection stream, "
            "and 48+1 8K fitting sequences"
        )
    if (
        long_context["calibration_sequences"] * long_context["sequence_length"]
        != settings["calibration"]["selected_pairs"]
    ):
        raise ValueError("8K calibration must match the 128-token calibration pair count")
    if settings.get("block_check") != {
        "layers": [3, 11, 19],
        "width_ratios": [0.25, 0.5],
        "seeds": list(PART_A_SEEDS),
        "settings": PART_A_SETTINGS,
        "adoption": ADOPTION_RULE,
    }:
        raise ValueError("SwiGLU-8 fixes the decided Part A grid, seeds, and adoption rule")
    current = PART_A_SETTINGS[CURRENT_SETTING]
    if any(current[key] != settings["local_fitting"][key]
           for key in ("learning_rate", "scheduler", "batch_size", "max_epochs", "early_stopping_patience")):
        raise ValueError("The current Part A setting must be SwiGLU-7's local-fitting recipe")
    if settings.get("acceptance") != ACCEPTANCE:
        raise ValueError("SwiGLU-8 fixes the decided acceptance tolerances")
    part_b = settings.get("part_b", {})
    recipe = part_b.get("fitting_recipe")
    digest = part_b.get("block_check_sha256")
    if (
        set(part_b) != {"fitting_recipe", "block_check_sha256"}
        or (recipe is None) != (digest is None)
        or (recipe is not None and recipe not in PART_A_SETTINGS)
        or (digest is not None and not is_digest(digest))
    ):
        raise ValueError("part_b names one Part A setting and its block-check sha256, or neither")
    sources = settings.get("sources", {})
    runs = sources.get("swiglu7_runs_sha256", {})
    prefix = sources.get("swiglu7_recovery_prefix_sha256")
    if (
        set(sources)
        != {"swiglu7_prepare_sha256", "swiglu7_runs_sha256", "swiglu7_recovery_prefix_sha256"}
        or not is_digest(sources["swiglu7_prepare_sha256"])
        or set(runs) != {str(target) for target in SWIGLU7_TARGETS}
        or not all(is_digest(value) for value in runs.values())
        or (prefix is not None and not is_digest(prefix))
    ):
        raise ValueError("SwiGLU-8 pins its SwiGLU-7 source artifacts by sha256")
    return settings, path


def pinned_json(root, relative, expected_sha256, label):
    path = resolve_path(Path(root)) / relative
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if file_digest(path) != expected_sha256:
        raise ValueError(f"{label} differs from its pinned sha256: {path}")
    return read_json(path)


def swiglu7_prepared(settings, root):
    return pinned_json(
        root,
        "prepare-001/result.json",
        settings["sources"]["swiglu7_prepare_sha256"],
        "SwiGLU-7 preparation",
    )


def swiglu7_run(settings, root, target):
    return pinned_json(
        root,
        f"S7-0-target-{target:.1f}-run-001/result.json",
        settings["sources"]["swiglu7_runs_sha256"][str(target)],
        f"SwiGLU-7 S7-0 run at {target}",
    )


def swiglu7_fit_rows(prepared):
    return {
        (int(row["layer"]), int(row["replacement_width"])): row
        for row in prepared["results"]["local_fitting"]
        if row["initialization"] == "legacy_subset" and row["capture_context"] == "dense"
    }


def swiglu7_curve_rows(prepared):
    return {
        (int(row["layer"]), int(row["replacement_width"])): row
        for row in prepared["results"]["width_curves"]["rows"]
    }


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
            raise ValueError("Resume output belongs to a different SwiGLU-8 run")
        if artifact.get("status") != "completed":
            if artifact.get("environment") != environment_record():
                raise ValueError("Resume environment differs from the recorded SwiGLU-8 run")
            artifact["status"] = "running"
            artifact["error"] = None
        return artifact
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "experiment_family": "swiglu-8",
        "experiment_class": "local-fitting-attribution",
        "command": command,
        "identity": identity,
        "status": "running",
        "created_at_utc": utc_now(),
        "configuration": deepcopy(settings),
        "environment": environment_record(),
        "results": {},
        "error": None,
    }


def record_failure(output_dir, work_dir, artifact, error):
    artifact["status"] = "failed"
    artifact["error"] = f"{type(error).__name__}: {error}".replace(str(work_dir), "<work-dir>")
    persist(output_dir, artifact, "failed")


def release_memory():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Decisions and acceptance checks


def relative_difference(value, reference):
    return abs(float(value) - float(reference)) / abs(float(reference))


def comparison(check, value, reference, tolerance, **labels):
    difference = relative_difference(value, reference)
    return {
        "check": check,
        **labels,
        "value": float(value),
        "reference": float(reference),
        "relative_difference": difference,
        "tolerance": float(tolerance),
        "passed": difference <= tolerance,
    }


def equality_check(check, value, reference, **labels):
    return {"check": check, **labels, "value": value, "reference": reference,
            "passed": value == reference}


def enforce(artifact, output_dir, group, checks):
    """Record one acceptance group and stop the stage if any check fails."""

    summary = {"checks": checks, "passed": all(row["passed"] for row in checks)}
    artifact["results"].setdefault("acceptance", {})[group] = summary
    persist(output_dir, artifact, f"acceptance:{group}")
    if not summary["passed"]:
        failed = [row for row in checks if not row["passed"]]
        raise AcceptanceFailure(
            f"{group} failed {len(failed)} of {len(checks)} checks; first: {failed[0]}"
        )
    return summary


def slice_checks(rows, reference_fits, tolerance):
    return [
        comparison(
            "A1",
            row["validation_128"]["nmse"],
            reference_fits[(row["layer"], row["replacement_width"])]["initial_local_nmse"],
            tolerance,
            layer=row["layer"],
            width=row["replacement_width"],
            metric="slice_nmse_128",
        )
        for row in rows
    ]


def current_recipe_checks(rows, reference_fits, reference_curves, tolerance):
    checks = []
    for row in rows:
        if (row["setting"], row["seed"]) != (CURRENT_SETTING, CURRENT_SEED):
            continue
        key = (row["layer"], row["replacement_width"])
        labels = {"layer": row["layer"], "width": row["replacement_width"]}
        for metric in ("local_nmse", "local_mse"):
            checks.append(
                comparison("A2", row[metric], reference_fits[key][metric], tolerance,
                           metric=metric, **labels)
            )
        checks.append(
            comparison("A3", row["kl_128"], reference_curves[key]["raw_teacher_kl"], tolerance,
                       metric="single_block_kl_128", **labels)
        )
    return checks


def monitoring_checks(row, reference_history, tolerances):
    """Compare one monitoring row with the S7-0 row at the same request."""

    requested = [int(value) for value in row["requested_tokens"]]
    reference = next(
        (
            entry
            for entry in reference_history
            if [int(value) for value in entry["requested_tokens"]] == requested
        ),
        None,
    )
    if reference is None:
        raise ValueError(f"SwiGLU-7 has no monitoring row for {requested}")
    check, tolerance = (
        ("B1", tolerances["exact_relative"])
        if requested == [0]
        else ("B2", tolerances["fit_relative"])
    )
    labels = {"tokens": int(row["actual_tokens"])}
    checks = [
        comparison(check, row["recovery_validation_kl"], reference["recovery_validation_kl"],
                   tolerance, metric="recovery_validation_kl", **labels)
    ]
    if ("wikitext_validation" in row) != ("wikitext_validation" in reference):
        raise ValueError(f"Monitoring rows at {requested} differ in WikiText evaluation")
    if "wikitext_validation" in row:
        checks.append(
            comparison(check, row["wikitext_validation"]["perplexity"],
                       reference["wikitext_validation"]["perplexity"], tolerance,
                       metric="wikitext_validation_perplexity", **labels)
        )
    return checks


def seed_summary(values):
    """Mean and sample standard deviation over seeds; one value has no spread."""

    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    return mean, math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def expected_seeds(setting, settings):
    if settings["block_check"]["settings"][setting]["method"] == "least_squares":
        return [None]
    return [int(seed) for seed in settings["block_check"]["seeds"]]


def adoption_decision(rows, settings):
    """Apply the decided multi-seed Part A adoption rule to completed fit rows.

    Each setting is summarized by its seed mean and sample standard deviation
    per layer-width combination and context. A setting qualifies when its mean
    8K reduction against the current setting exceeds the minimum and the mean
    seed noise, (sd_setting + sd_current) / mean_current, and no combination
    is worse than allowed at either context. The largest mean 8K reduction
    wins; otherwise the current setting stays.
    """

    rule = settings["block_check"]["adoption"]
    deciding = int(rule["deciding_context"])
    combinations = sorted({(row["layer"], row["replacement_width"]) for row in rows})
    values = {}
    for row in rows:
        for context in CONTEXTS:
            values.setdefault((row["setting"], row["layer"], row["replacement_width"], context), {})[
                row["seed"]
            ] = float(row[f"kl_{context}"])
    summaries = {}
    for setting in settings["block_check"]["settings"]:
        for layer, width in combinations:
            for context in CONTEXTS:
                by_seed = values.get((setting, layer, width, context), {})
                if sorted(by_seed, key=str) != sorted(expected_seeds(setting, settings), key=str):
                    raise ValueError(f"Block check lacks seeds for {setting} at layer {layer}, width {width}")
                summaries[(setting, layer, width, context)] = seed_summary(
                    [by_seed[seed] for seed in expected_seeds(setting, settings)]
                )

    def summary_rows(setting):
        return [
            {"layer": layer, "width": width, "context": context, "mean": mean, "seed_sd": deviation}
            for (name, layer, width, context), (mean, deviation) in sorted(summaries.items())
            if name == setting
        ]

    table = {}
    for setting in settings["block_check"]["settings"]:
        if setting == CURRENT_SETTING:
            continue
        reductions = {}
        noise = {}
        for layer, width in combinations:
            for context in CONTEXTS:
                current_mean, current_sd = summaries[(CURRENT_SETTING, layer, width, context)]
                mean, deviation = summaries[(setting, layer, width, context)]
                reductions[(layer, width, context)] = 1.0 - mean / current_mean
                noise[(layer, width, context)] = (deviation + current_sd) / current_mean
        mean_reduction = {
            context: sum(reductions[(layer, width, context)] for layer, width in combinations)
            / len(combinations)
            for context in CONTEXTS
        }
        mean_noise = sum(noise[(layer, width, deciding)] for layer, width in combinations) / len(
            combinations
        )
        worst = min(reductions.values())
        reasons = []
        if not mean_reduction[deciding] > rule["minimum_mean_reduction"]:
            reasons.append(
                f"mean {deciding}-token reduction is not above {rule['minimum_mean_reduction']:.0%}"
            )
        if not mean_reduction[deciding] > mean_noise:
            reasons.append("mean reduction is not above the seed noise")
        if worst < -rule["maximum_combination_increase"]:
            reasons.append(
                f"a combination is more than {rule['maximum_combination_increase']:.0%} worse"
            )
        table[setting] = {
            "mean_reduction": {str(context): value for context, value in mean_reduction.items()},
            "mean_seed_noise": mean_noise,
            "worst_reduction": worst,
            "reductions": [
                {"layer": layer, "width": width, "context": context, "reduction": value,
                 "seed_noise": noise[(layer, width, context)]}
                for (layer, width, context), value in sorted(reductions.items())
            ],
            "seed_summaries": summary_rows(setting),
            "qualifies": not reasons,
            "reasons": reasons,
        }
    qualifying = [setting for setting, row in table.items() if row["qualifies"]]
    selected = (
        max(qualifying, key=lambda setting: table[setting]["mean_reduction"][str(deciding)])
        if qualifying
        else CURRENT_SETTING
    )
    return {
        "rule": deepcopy(rule),
        "reference_setting": CURRENT_SETTING,
        "reference_seed_summaries": summary_rows(CURRENT_SETTING),
        "settings": table,
        "qualifying": qualifying,
        "selected_recipe": selected,
    }


def stage_two_indicator(rows, setting):
    """Report whether the adopted setting still overfits at width 0.5.

    A layer counts when most of its seeds reach their best epoch by epoch 3
    while their training MSE is still falling.
    """

    candidates = [row for row in rows if row["setting"] == setting and row["width_ratio"] == 0.5]
    if not candidates or any(not row.get("history") for row in candidates):
        return {"setting": setting, "applicable": False, "overfitting_remains": False}
    votes = {}
    for row in candidates:
        history = [entry for entry in row["history"] if entry["phase"] == "full"]
        best = max(1, int(row["best_epoch"]))
        overfits = (
            int(row["best_epoch"]) <= 3
            and len(history) > best
            and history[-1]["train_mse"] < history[best - 1]["train_mse"]
        )
        votes.setdefault(row["layer"], []).append(overfits)
    layers = sorted(layer for layer, flags in votes.items() if 2 * sum(flags) > len(flags))
    return {
        "setting": setting,
        "applicable": True,
        "definition": (
            "best epoch <= 3 while training MSE still falls, at width 0.5, in most seeds of a layer"
        ),
        "overfitting_layers": layers,
        "overfitting_remains": len(layers) >= 2,
    }


def selected_indices(ranking, width):
    return ranking[: int(width)].sort().values


def selection_record(allocation, rankings):
    """Fingerprint the widths and teacher neurons one target uses in both arms."""

    widths = {str(row["layer"]): int(row["replacement_width"]) for row in allocation}
    digests = {
        str(row["layer"]): content_digest(
            selected_indices(rankings[int(row["layer"])], row["replacement_width"]).tolist()
        )
        for row in allocation
        if not row["retains_dense_module"]
    }
    return {
        "widths": widths,
        "index_digests": digests,
        "fingerprint": content_digest({"widths": widths, "index_digests": digests}),
    }


def start_key(arm, target):
    return f"{arm}:{float(target)}"


def width_from_ratio(settings, ratio):
    original = int(settings["model"]["intermediate_size"])
    return min(original, max(1, round(original * float(ratio))))


def allocation_rows(allocation):
    return [
        {
            "layer": int(row["layer"]),
            "original_width": int(row["original_width"]),
            "replacement_width": int(row["replacement_width"]),
            "replacement_parameters": int(row["replacement_parameters"]),
            "retains_dense_module": bool(row["retains_dense_module"]),
            "has_output_bias": bool(row["has_output_bias"]),
            "is_boundary_width": bool(row["is_boundary_width"]),
        }
        for row in allocation.rows
    ]


def width_curves(rows, settings):
    original = int(settings["model"]["intermediate_size"])
    hidden = int(settings["model"]["hidden_size"])
    expected = {width_from_ratio(settings, ratio) for ratio in settings["allocation"]["width_ratios"]}
    curves = {}
    for layer in settings["allocation"]["eligible_layers"]:
        points = {
            int(row["replacement_width"]): {"teacher_kl": float(row["raw_teacher_kl"]),
                                            "has_output_bias": False}
            for row in rows
            if int(row["layer"]) == int(layer)
        }
        if set(points) != expected:
            raise ValueError(f"Width curve for layer {layer} lacks the configured widths")
        curves[int(layer)] = build_swiglu_width_curve(int(layer), original, points, hidden)
    return curves


# Data


def partition_record_indices(settings, record_indices):
    batch = int(settings["data"]["capture_batch_size"])
    partitions = {}
    offset = 0
    for name, batches in settings["data"]["partition_batches"].items():
        size = int(batches) * batch
        partitions[name] = record_indices[offset : offset + size]
        offset += size
    if offset != len(record_indices):
        raise ValueError("Recorded window sources do not cover the local partitions")
    return partitions


def replay_kl_records(records, tokenizer, kl_tokens, text_column="text"):
    """Return the last source record packed into the KL stream, verifying its bytes."""

    import numpy as np

    position = 0
    total = len(kl_tokens)
    for index in range(len(records)):
        text = str(records[index].get(text_column) or "")
        if not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False).input_ids
        if not ids:
            continue
        document = np.asarray([*ids, tokenizer.eos_token_id], dtype=np.int32)[: total - position]
        if not np.array_equal(np.asarray(kl_tokens[position : position + len(document)]), document):
            raise ValueError("Indexed shard order does not reproduce the KL stream")
        position += len(document)
        if position == total:
            return index
    raise ValueError("Indexed shard ended before reproducing the KL stream")


def unused_records(records, start, excluded, consumed):
    """Yield records in order from ``start``, skipping excluded indices."""

    for index in range(start, len(records)):
        if index in excluded:
            continue
        consumed.append(index)
        yield records[index]


def build_fitting_streams(
    records, tokenizer, start, excluded, paths, sequence_length, sequence_counts, text_column="text"
):
    """Pack document-disjoint 8K streams after ``start`` in ``sequence_counts`` order."""

    built = {}
    cursor = start
    for name, count in sequence_counts.items():
        consumed = []
        record = build_finite_stream(
            unused_records(records, cursor, excluded, consumed),
            tokenizer,
            paths[name],
            int(count) * int(sequence_length),
            text_column=text_column,
        )
        record.update(
            first_record=consumed[0],
            last_record=consumed[-1],
            records_read=len(consumed),
            excluded_records_skipped=sum(
                1 for index in excluded if consumed[0] <= index <= consumed[-1]
            ),
        )
        built[name] = record
        # The last packed document may be truncated; the next stream starts after it.
        cursor = consumed[-1] + 1
    return built


def verified_stream(root, record):
    path = output_path(root, record["path"])
    if path.stat().st_size != 4 * int(record["token_count"]) or file_digest(path) != record["sha256"]:
        raise ValueError(f"Prepared token stream changed: {path}")
    return path


def prepare_long_context_data(context, settings, output_dir, artifact, include_fitting,
                              include_selection=False):
    """Freeze the shared 8K KL stream and, when needed, the later 8K streams.

    The 8K calibration, validation, and Part A selection streams follow the
    last KL-stream document in that order and skip every record that supplied
    a 128-token window. None of them shares a document with the 8K final
    evaluation, with each other, or with any 128-token partition, so Part A
    never selects on data that the final evaluation uses.
    """

    import numpy as np

    long_context = settings["long_context"]
    length = int(long_context["sequence_length"])
    records = artifact["results"].setdefault("long_context_data", {})
    prepared = output_dir / "prepared"
    if "kl_stream" not in records:
        path = prepared / "kl-tokens.int32"
        path.unlink(missing_ok=True)  # an unrecorded file has no provenance
        record = build_teacher_kl_stream(
            long_context["source"], context.tokenizer, path, long_context["kl_sequences"], length
        )
        record.update(path=relative_output(output_dir, path), protocol=TEACHER_KL_PROTOCOL)
        records["kl_stream"] = record
        persist(output_dir, artifact, "long_context:kl_stream")
    kl_path = verified_stream(output_dir, records["kl_stream"])
    counts = {}
    if include_fitting or include_selection:
        counts.update(calibration=long_context["calibration_sequences"],
                      validation=long_context["validation_sequences"])
    if include_selection:
        counts["selection"] = long_context["selection_sequences"]
    needs_streams = any(f"{name}_stream" not in records for name in counts)
    if "audit" in records and not needs_streams:
        for name in counts:
            verified_stream(output_dir, records[f"{name}_stream"])
        return records
    source = settings["data"]["local_source"]
    if any(source[key] != long_context["source"][key] for key in ("path", "revision", "data_file")):
        raise ValueError("8K fitting data must come from the KL evaluation shard")
    shard = load_text_dataset(DatasetSpec(**source))
    last_kl_record = replay_kl_records(
        shard, context.tokenizer, np.fromfile(kl_path, dtype=np.int32), source["text_column"]
    )
    partitions = partition_record_indices(settings, context.record_indices)
    records["audit"] = {
        "kl_record_range": [0, last_kl_record],
        "local_window_records": len(context.record_indices),
        "local_window_records_digest": content_digest(context.record_indices),
        "local_windows_from_kl_records": {
            name: sum(1 for index in indices if index <= last_kl_record)
            for name, indices in partitions.items()
        },
    }
    if needs_streams:
        paths = {name: prepared / f"long-context-{name}.int32" for name in counts}
        for path in paths.values():
            path.unlink(missing_ok=True)
        built = build_fitting_streams(
            shard,
            context.tokenizer,
            last_kl_record + 1,
            set(context.record_indices),
            paths,
            length,
            counts,
            source["text_column"],
        )
        for name, record in built.items():
            record["path"] = relative_output(output_dir, paths[name])
            if name == "selection":
                record["protocol"] = SELECTION_KL_PROTOCOL
            records[f"{name}_stream"] = record
        previous_last = last_kl_record
        disjoint = True
        for name in counts:
            disjoint = disjoint and built[name]["first_record"] > previous_last
            previous_last = built[name]["last_record"]
        records["audit"]["document_disjoint"] = disjoint
        if not records["audit"]["document_disjoint"]:
            raise RuntimeError("8K streams overlap each other or the KL evaluation documents")
    persist(output_dir, artifact, "long_context:audit")
    for name in counts:
        verified_stream(output_dir, records[f"{name}_stream"])
    return records


def long_context_view(context, output_dir, records, settings):
    """Expose the 8K fitting streams through the capture interface."""

    import numpy as np
    import torch

    length = int(settings["long_context"]["sequence_length"])

    def sequences(record):
        values = np.fromfile(verified_stream(output_dir, record), dtype=np.int32).astype(np.int64)
        return [
            torch.from_numpy(values[start : start + length].copy())
            for start in range(0, len(values), length)
        ]

    validation = sequences(records["validation_stream"])
    view = SimpleNamespace(**vars(context))
    view.data = {
        "calibration_sequences": sequences(records["calibration_stream"]),
        "operator_validation": make_token_loader(validation, 1),
        "partition_batches": {"operator_validation": len(validation)},
        "batch_size": 1,
        "sequence_length": length,
    }
    return view


# Models and blocks


def fitting_context(settings, work_dir, output_dir, artifact, stage):
    """Load the teacher and rebuild SwiGLU-7's 128-token local partitions."""

    import torch

    context = SimpleNamespace()
    context.settings = effective_preparation_settings(settings)
    context.output = work_dir / "fitting-context.json"
    context.asset_dir = work_dir / "fit-assets"
    context.artifact = {"results": {"local_fitting": []}}
    context.persist = lambda label=None: persist(
        output_dir,
        artifact,
        f"{stage}:{label or 'working'}",
        {"local_fits_completed": len(context.artifact["results"]["local_fitting"])},
    )
    model_config = make_model_config(settings["model"])
    context.model, context.tokenizer = load_model_and_tokenizer(model_config)
    context.attention_implementation = validated_attention_implementation(context.model)
    context.device = next(context.model.parameters()).device
    context.model_dtype = next(context.model.parameters()).dtype
    context.blocks = {row.index: row for row in discover_mlp_blocks(context.model)}
    if set(context.blocks) != set(range(settings["model"]["num_layers"])):
        raise ValueError("Loaded model topology differs from the SwiGLU-8 contract")
    context.record_indices = []
    context.data = build_local_data(context, record_indices=context.record_indices)
    torch.manual_seed(settings["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings["seed"])
    return context, model_config


def recipe_context(context, setting, seed):
    """View the fitting context with one Part A setting's optimizer and seed."""

    view = SimpleNamespace(**vars(context))
    view.settings = deepcopy(context.settings)
    if setting["method"] == "gradient":
        for key in ("learning_rate", "scheduler", "batch_size", "max_epochs", "early_stopping_patience"):
            view.settings["local_fitting"][key] = setting[key]
        view.settings["seed"] = int(seed)
    return view


def fit_context_kind(setting_id, seed):
    # The current setting at seed 21 keeps SwiGLU-7's fit keys so its records compare directly.
    if (setting_id, seed) == (CURRENT_SETTING, CURRENT_SEED):
        return "dense"
    return setting_id if seed is None else f"{setting_id}-s{seed}"


def neuron_ranking(context, layer, inputs):
    import torch

    scores = swiglu_neuron_importance_scores(
        context.blocks[layer].module,
        inputs,
        int(context.settings["local_fitting"]["batch_size"]),
    )
    return torch.argsort(scores, descending=True)


def slice_module(context, layer, width, indices):
    import torch

    module = GatedMLPReplacement(int(context.settings["model"]["hidden_size"]), int(width)).to(
        context.device, dtype=torch.float32
    )
    initialize_gated_mlp_from_teacher(module, context.blocks[layer].module, indices)
    return module.eval()


def single_block_kl_short(context, layer, module, selection_cache):
    with temporary_fp32_replacements(context.model, context.blocks, {layer: module}):
        return evaluate_teacher_cache_mixed(
            context.model,
            selection_cache,
            float(context.settings["recovery"]["temperature"]),
            context.device,
        )


def evaluate_block(context, layer, module, pairs, selection_cache, teacher, stream_path, stream_record,
                   length):
    """Report local error on both validation sets and single-block KL at both contexts.

    The 8K score uses the Part A selection stream, never the final-evaluation
    KL stream.
    """

    path = context.blocks[layer].path
    batch = int(context.settings["local_fitting"]["batch_size"])
    record = {}
    for context_length in CONTEXTS:
        metrics = evaluate_operator(module, pairs[context_length]["validation"][path],
                                    context.device, batch)
        record[f"validation_{context_length}"] = {
            "mse": metrics.mse,
            "nmse": metrics.relative_mse,
            "cosine": metrics.cosine_similarity,
        }
    short = single_block_kl_short(context, layer, module, selection_cache)
    with temporary_fp32_replacements(context.model, context.blocks, {layer: module}):
        long = evaluate_teacher_kl(context.model, teacher, stream_path, stream_record, length,
                                   protocol=stream_record["protocol"], device=context.device)
    record.update(
        single_block_kl_128=short,
        single_block_kl_8192=long,
        kl_128=short["teacher_kl"],
        kl_8192=long["kl"],
    )
    return record


def least_squares_operator(context, layer, width, indices, training_pairs, validation_pairs,
                           context_kind):
    """Solve the exact down projection of a teacher slice and record it like a gradient fit."""

    identifier = fit_key("legacy_subset", layer, width, context_kind)
    existing = {row["fit_key"]: row for row in context.artifact["results"]["local_fitting"]}
    if identifier in existing:
        row = existing[identifier]
        state_path = resolve_source_asset(row["state_path"], context.output)
        if not state_path.is_file() or sha256_file(state_path) != row["state_sha256"]:
            raise ValueError(f"Persisted local-fit state changed: {state_path}")
        return row
    batch = int(context.settings["local_fitting"]["batch_size"])
    started = perf_counter()
    module = slice_module(context, layer, width, indices)
    initial = evaluate_operator(module, validation_pairs, context.device, batch)
    solve = refit_down_projection_least_squares(module, training_pairs, context.device, batch)
    final = evaluate_operator(module, validation_pairs, context.device, batch)
    state = {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}
    state_path = operator_state_path(context, "legacy_subset", layer, width, context_kind)
    atomic_torch_save(state_path, state)
    row = {
        "fit_key": identifier,
        "initialization": "legacy_subset",
        "capture_context": context_kind,
        "layer": int(layer),
        "replacement_width": int(width),
        "has_output_bias": False,
        "calibration_pairs": int(training_pairs.num_tokens),
        "best_epoch": 0,
        "epochs_completed": 0,
        "updates": 0,
        "initial_local_mse": float(initial.mse),
        "initial_local_nmse": float(initial.relative_mse),
        "initial_local_cosine": float(initial.cosine_similarity),
        "output_reconstruction": None,
        "local_mse": float(final.mse),
        "local_nmse": float(final.relative_mse),
        "local_cosine": float(final.cosine_similarity),
        "fit_seconds": perf_counter() - started,
        "state_path": relative_to_root(state_path),
        "state_sha256": sha256_file(state_path),
        "parameter_count": sum(value.numel() for value in state.values()),
        "least_squares": solve,
        "history": [],
    }
    context.artifact["results"]["local_fitting"].append(row)
    context.persist("local_fitting")
    module.to("cpu")
    del module, state
    release_memory()
    return row


def fit_setting(context, setting_id, setting, seed, layer, width, indices, training_pairs,
                validation_pairs):
    """Fit one teacher-slice replacement with a named Part A setting and seed."""

    kind = fit_context_kind(setting_id, seed)
    if setting["method"] == "least_squares":
        return least_squares_operator(context, layer, width, indices, training_pairs,
                                      validation_pairs, kind)
    return fit_operator(
        recipe_context(context, setting, seed),
        layer,
        width,
        "legacy_subset",
        training_pairs,
        validation_pairs,
        indices,
        context_kind=kind,
    )


def fit_block(context, setting_id, setting, seed, layer, width, indices, pairs):
    """Fit one Part A block with its setting's own calibration context."""

    path = context.blocks[layer].path
    fitting_pairs = pairs[int(setting["calibration_context"])]
    row = fit_setting(context, setting_id, setting, seed, layer, width, indices,
                      fitting_pairs["training"][path], fitting_pairs["validation"][path])
    module = load_fit_operator(context, row)
    record = {
        key: deepcopy(row[key])
        for key in (
            "fit_key", "calibration_pairs", "best_epoch", "epochs_completed", "updates",
            "fit_seconds", "initial_local_mse", "initial_local_nmse", "local_mse", "local_nmse",
            "local_cosine", "state_sha256", "history", "least_squares",
        )
        if key in row
    }
    return module, record


def teacher_self_kl(context, teacher, kl_path, kl_record, settings):
    result = evaluate_teacher_kl(
        context.model, teacher, kl_path, kl_record,
        int(settings["long_context"]["sequence_length"]), device=context.device,
    )
    return {
        "check": "P5",
        "metric": "teacher_against_itself_8k_kl",
        "value": result["kl"],
        "tolerance": settings["acceptance"]["teacher_self_kl_absolute"],
        "passed": abs(result["kl"]) <= settings["acceptance"]["teacher_self_kl_absolute"],
    }


# Part A: block check


def block_check(args, settings, config_path):
    work_dir, output_dir = resolve_storage(args.work_dir, args.output_dir, args.resume)
    artifact = start_artifact(output_dir, settings, "block-check", args.resume, {"kind": "block-check"})
    try:
        reference = swiglu7_prepared(settings, args.swiglu7_root)
        contract = {
            "configuration": settings,
            "configuration_sha256": file_digest(config_path),
            "code_hashes": code_hashes(),
            "swiglu7_prepare_sha256": settings["sources"]["swiglu7_prepare_sha256"],
        }
        run_fingerprint = content_digest(contract)
        if args.resume and artifact.get("run_fingerprint") != run_fingerprint:
            raise ValueError("Block-check configuration, sources, or maintained code changed")
        if artifact.get("status") == "completed":
            return
        artifact.update(
            run_fingerprint=run_fingerprint,
            code_hashes=contract["code_hashes"],
            configuration_sha256=contract["configuration_sha256"],
            status="running",
            error=None,
        )
        results = artifact["results"]
        results.setdefault("slices", [])
        results.setdefault("fits", [])
        persist(output_dir, artifact, "loading_models")
        context, model_config = fitting_context(settings, work_dir, output_dir, artifact, "block_check")
        long_data = prepare_long_context_data(context, settings, output_dir, artifact, True,
                                              include_selection=True)
        kl_record = long_data["kl_stream"]
        kl_path = verified_stream(output_dir, kl_record)
        selection_record = long_data["selection_stream"]
        selection_path = verified_stream(output_dir, selection_record)
        length = int(settings["long_context"]["sequence_length"])
        teacher, unused_tokenizer = load_model_and_tokenizer(model_config)
        validated_attention_implementation(teacher)
        if "P5" not in results.get("acceptance", {}):
            enforce(artifact, output_dir, "P5",
                    [teacher_self_kl(context, teacher, kl_path, kl_record, settings)])
        selection_cache = cache_teacher_logits(
            context.model,
            context.data["allocation_selection"],
            context.data["partition_batches"]["allocation_selection"],
            context.device,
            "float16",
        )
        layers = [int(value) for value in settings["block_check"]["layers"]]
        widths = [width_from_ratio(settings, ratio) for ratio in settings["block_check"]["width_ratios"]]
        original = int(settings["model"]["intermediate_size"])
        training, validation = capture_dense_pairs(context, layers)
        long_training, long_validation = capture_dense_pairs(
            long_context_view(context, output_dir, long_data, settings), layers
        )
        pairs = {
            128: {"training": training, "validation": validation},
            8192: {"training": long_training, "validation": long_validation},
        }
        rankings = {
            layer: neuron_ranking(context, layer, training[context.blocks[layer].path].inputs)
            for layer in layers
        }
        results["ranking_digests"] = {
            str(layer): content_digest(ranking.tolist()) for layer, ranking in rankings.items()
        }
        reference_fits = swiglu7_fit_rows(reference)
        reference_curves = swiglu7_curve_rows(reference)
        tolerances = settings["acceptance"]
        evaluation_inputs = (selection_cache, teacher, selection_path, selection_record, length)

        def completed(rows, setting, seed, layer, width):
            return any(
                (row["setting"], row["seed"], row["layer"], row["replacement_width"])
                == (setting, seed, layer, width)
                for row in rows
            )

        # Unfitted slices come first: A1 checks selection and data before any fitting.
        for layer in layers:
            for width in widths:
                if completed(results["slices"], "slice", None, layer, width):
                    continue
                module = slice_module(context, layer, width, selected_indices(rankings[layer], width))
                results["slices"].append(
                    {"setting": "slice", "seed": None, "layer": layer, "replacement_width": width,
                     "width_ratio": width / original,
                     **evaluate_block(context, layer, module, pairs, *evaluation_inputs)}
                )
                del module
                release_memory()
                persist(output_dir, artifact, f"block_check:slice:{layer}:{width}")
        enforce(artifact, output_dir, "A1",
                slice_checks(results["slices"], reference_fits, tolerances["exact_relative"]))
        grid = settings["block_check"]["settings"]
        # The SwiGLU-7 anchor runs first so A2-A3 stop the job before the rest of the grid.
        runs = [(CURRENT_SETTING, CURRENT_SEED)] + [
            (setting, seed)
            for setting in grid
            for seed in expected_seeds(setting, settings)
            if (setting, seed) != (CURRENT_SETTING, CURRENT_SEED)
        ]
        for setting_id, seed in runs:
            for layer in layers:
                for width in widths:
                    if completed(results["fits"], setting_id, seed, layer, width):
                        continue
                    indices = selected_indices(rankings[layer], width)
                    module, fit = fit_block(context, setting_id, grid[setting_id], seed, layer, width,
                                            indices, pairs)
                    row = {
                        "setting": setting_id,
                        "seed": seed,
                        "layer": layer,
                        "replacement_width": width,
                        "width_ratio": width / original,
                        **grid[setting_id],
                        **fit,
                        **evaluate_block(context, layer, module, pairs, *evaluation_inputs),
                    }
                    results["fits"].append(row)
                    module.to("cpu")
                    del module
                    release_memory()
                    persist(output_dir, artifact, f"block_check:{setting_id}:s{seed}:{layer}:{width}")
                    print(
                        f"setting={setting_id} seed={seed} layer={layer} width={width} "
                        f"kl128={row['kl_128']:.6f} kl8192={row['kl_8192']:.6f}",
                        flush=True,
                    )
            if (setting_id, seed) == (CURRENT_SETTING, CURRENT_SEED):
                enforce(artifact, output_dir, "A2-A3",
                        current_recipe_checks(results["fits"], reference_fits, reference_curves,
                                              tolerances["fit_relative"]))
        results["adoption"] = adoption_decision(results["fits"], settings)
        results["stage_two_indicator"] = stage_two_indicator(
            results["fits"], results["adoption"]["selected_recipe"]
        )
        artifact["status"] = "completed"
        persist(output_dir, artifact, "completed")
    except BaseException as error:
        record_failure(output_dir, work_dir, artifact, error)
        raise


def load_block_check(path, settings):
    path = resolve_path(Path(path))
    if file_digest(path) != settings["part_b"]["block_check_sha256"]:
        raise ValueError("Block-check result differs from part_b.block_check_sha256")
    block = read_json(path)
    acceptance = block["results"].get("acceptance", {})
    if (
        block.get("workflow") != WORKFLOW
        or block.get("command") != "block-check"
        or block.get("status") != "completed"
        or not {"P5", "A1", "A2-A3"} <= set(acceptance)
        or not all(group["passed"] for group in acceptance.values())
    ):
        raise ValueError("Preparation requires a completed, accepted SwiGLU-8 block check")
    if block["results"]["adoption"]["selected_recipe"] != settings["part_b"]["fitting_recipe"]:
        raise ValueError("part_b.fitting_recipe differs from the block check's adoption outcome")
    return block


# Part B: preparation


def preparation_storage_preflight(work_dir, output_dir, settings):
    import shutil

    hidden = int(settings["model"]["hidden_size"])
    width = int(settings["model"]["intermediate_size"])
    layers = len(settings["allocation"]["eligible_layers"])
    original = layers * 3 * hidden * width
    start_bytes = len(ARMS) * sum(int(original * (1.0 - target)) * 4 for target in TARGETS)
    fit_bytes = layers * sum(
        3 * hidden * width_from_ratio(settings, ratio) * 4
        for ratio in settings["allocation"]["width_ratios"]
        if ratio < 1.0
    )
    output_required = int(
        (int(settings["recovery"]["target_tokens"]) * 4 + start_bytes + 2_000_000_000) * 1.15
    )
    work_required = int((fit_bytes + 2_000_000_000) * 1.15)
    output_free = shutil.disk_usage(output_dir).free
    work_free = shutil.disk_usage(work_dir).free
    shared = work_dir.stat().st_dev == output_dir.stat().st_dev
    if (shared and min(output_free, work_free) < output_required + work_required) or (
        not shared and (output_free < output_required or work_free < work_required)
    ):
        raise RuntimeError("SwiGLU-8 preparation lacks output or temporary storage capacity")
    return {
        "estimated_output_bytes": output_required,
        "estimated_work_bytes": work_required,
        "observed_output_free_bytes": output_free,
        "observed_work_free_bytes": work_free,
        "shared_filesystem": shared,
    }


def recovery_prefix_checks(stream, settings, swiglu7_root):
    """Compare the 100M stream with SwiGLU-7's first 100M recovery tokens."""

    checks = []
    path = resolve_path(Path(swiglu7_root)) / "prepare-001/prepared/recovery-tokens.int32"
    if path.is_file():
        checks.append(
            equality_check("P4", stream["sha256"], file_digest(path, 4 * int(stream["token_count"])),
                           metric="swiglu7_stream_prefix_sha256")
        )
    expected = settings["sources"]["swiglu7_recovery_prefix_sha256"]
    if expected is not None:
        checks.append(equality_check("P4", stream["sha256"], expected,
                                     metric="configured_prefix_sha256"))
    if not checks:
        raise ValueError(
            "P4 needs SwiGLU-7's recovery stream under --swiglu7-root or "
            "sources.swiglu7_recovery_prefix_sha256"
        )
    return checks


def dense_checks(artifact, output_dir, reference, settings, swiglu7_root):
    """Compare the dense teacher evaluation with SwiGLU-7's recorded reference."""

    dense = artifact["results"]["dense_evaluation"]
    recorded = reference["results"]["dense_evaluation"]
    tolerance = settings["acceptance"]["exact_relative"]
    checks = [
        comparison("P3", dense["likelihood"][key]["perplexity"],
                   recorded["likelihood"][key]["perplexity"], tolerance, metric=f"perplexity:{key}")
        for key in sorted(recorded["likelihood"])
    ]
    checks.append(
        comparison("P3", dense["monitoring_validation"]["perplexity"],
                   recorded["monitoring_validation"]["perplexity"], tolerance,
                   metric="monitoring_perplexity")
    )
    root = resolve_path(Path(swiglu7_root)) / "prepare-001"
    for task in settings["evaluation"]["tasks"]:
        metric = settings["evaluation"]["primary_metrics"][task]
        value = read_json(output_path(output_dir, dense["tasks"][task]["path"]))["results"][task][
            f"{metric},none"
        ]
        reference_path = root / recorded["tasks"][task]["path"]
        if reference_path.is_file() and file_digest(reference_path) == recorded["tasks"][task]["sha256"]:
            expected = read_json(reference_path)["results"][task][f"{metric},none"]
            checks.append(equality_check("P3", value, expected, metric=f"accuracy:{task}"))
        else:
            checks.append({"check": "P3", "metric": f"accuracy:{task}", "value": value,
                           "reference": None, "status": "reference_unavailable", "passed": True})
    return checks


def prepare_operators(context, settings, output_dir, artifact, reference, recipe_id, long_data):
    """Rank neurons, extend or refit width curves, allocate, and fit start widths."""

    results = artifact["results"]
    recipe = settings["block_check"]["settings"][recipe_id]
    seed = None if recipe["method"] == "least_squares" else int(settings["seed"])
    hidden = int(settings["model"]["hidden_size"])
    original = int(settings["model"]["intermediate_size"])
    batch = int(settings["local_fitting"]["batch_size"])
    layers = [int(value) for value in settings["allocation"]["eligible_layers"]]
    group_size = int(settings["local_fitting"]["capture_group_size"])
    tolerances = settings["acceptance"]
    reference_fits = swiglu7_fit_rows(reference)
    reference_curves = swiglu7_curve_rows(reference)
    long_view = (
        long_context_view(context, output_dir, long_data, settings)
        if int(recipe["calibration_context"]) == 8192
        else None
    )
    configured = sorted(
        {width_from_ratio(settings, ratio) for ratio in settings["allocation"]["width_ratios"]}
        - {original}
    )
    swiglu7_widths = {width for unused_layer, width in reference_curves} - {original}
    curve_widths = (
        [width for width in configured if width not in swiglu7_widths]
        if recipe_id == CURRENT_SETTING
        else configured
    )
    selection_cache = cache_teacher_logits(
        context.model,
        context.data["allocation_selection"],
        context.data["partition_batches"]["allocation_selection"],
        context.device,
        "float16",
    )
    rankings = {}
    curve_rows = []
    fit_rows = {}
    audit = []

    def capture(group, short_needed):
        short = capture_dense_pairs(context, group) if short_needed or long_view is None else None
        fitting = capture_dense_pairs(long_view, group) if long_view is not None else short
        return short, fitting

    def fit(layer, width, fitting):
        path = context.blocks[layer].path
        row = fit_setting(context, recipe_id, recipe, seed, layer, width,
                          selected_indices(rankings[layer], width), fitting[0][path], fitting[1][path])
        fit_rows[(layer, width)] = row
        return row

    # Pass 1: rankings, the selection audit, and the width-curve points.
    for offset in range(0, len(layers), group_size):
        group = layers[offset : offset + group_size]
        short, fitting = capture(group, True)
        for layer in group:
            path = context.blocks[layer].path
            rankings[layer] = neuron_ranking(context, layer, short[0][path].inputs)
            for width in sorted(width for row_layer, width in reference_fits if row_layer == layer):
                module = slice_module(context, layer, width, selected_indices(rankings[layer], width))
                metrics = evaluate_operator(module, short[1][path], context.device, batch)
                audit.append(
                    comparison("P1", metrics.relative_mse,
                               reference_fits[(layer, width)]["initial_local_nmse"],
                               tolerances["exact_relative"], layer=layer, width=width,
                               metric="slice_nmse_128")
                )
                del module
            enforce(artifact, output_dir, "P1", audit)
            for width in curve_widths:
                row = fit(layer, width, fitting)
                module = load_fit_operator(context, row)
                kl = single_block_kl_short(context, layer, module, selection_cache)
                module.to("cpu")
                del module
                curve_rows.append({"layer": layer, "replacement_width": width,
                                   "raw_teacher_kl": kl["teacher_kl"], "source": WORKFLOW,
                                   "fit_key": row["fit_key"]})
                print(f"curve layer={layer} width={width} kl128={kl['teacher_kl']:.6f}", flush=True)
        del short, fitting
        release_memory()
    if recipe_id == CURRENT_SETTING:
        curve_rows += [
            {"layer": layer, "replacement_width": width, "raw_teacher_kl": float(row["raw_teacher_kl"]),
             "source": "swiglu-7", "fit_key": row.get("fit_key")}
            for (layer, width), row in sorted(reference_curves.items())
        ]
    else:
        curve_rows += [
            {"layer": layer, "replacement_width": original, "raw_teacher_kl": 0.0,
             "source": "dense", "fit_key": None}
            for layer in layers
        ]
    curve_rows.sort(key=lambda row: (row["layer"], row["replacement_width"]))
    curves = width_curves(curve_rows, settings)
    monotone = {
        (layer, point.replacement_width): point.monotone_teacher_kl
        for layer, points in curves.items()
        for point in points
    }
    for row in curve_rows:
        row["monotone_teacher_kl"] = monotone[(row["layer"], row["replacement_width"])]
    results["width_curves"] = {"recipe_id": recipe_id, "rows": curve_rows}
    persist(output_dir, artifact, "width_curves_completed")

    allocations = {}
    solvers = {}
    resolve_report = []
    for target in TARGETS:
        solved = allocate_discrete_swiglu_widths(curves, target, hidden)
        solver = {key: value for key, value in asdict(solved).items() if key != "rows"}
        if target in SWIGLU7_TARGETS:
            start = reference["results"]["starts"][str(target)]
            allocations[target] = deepcopy(start["allocation"])
            solvers[target] = {**deepcopy(start["allocation_solver"]), "source": "swiglu-7"}
            kept = {int(row["layer"]): int(row["replacement_width"]) for row in start["allocation"]}
            resolved = {int(row["layer"]): int(row["replacement_width"]) for row in solved.rows}
            resolve_report.append({
                "target": target,
                "changed_layers": [
                    {"layer": layer, "swiglu7_width": kept[layer], "resolved_width": resolved[layer]}
                    for layer in sorted(kept)
                    if kept[layer] != resolved[layer]
                ],
                "swiglu7_predicted_teacher_kl": start["allocation_solver"]["predicted_teacher_kl"],
                "resolved_predicted_teacher_kl": solved.predicted_teacher_kl,
                "resolved_solver": solver,
            })
        else:
            allocations[target] = allocation_rows(solved)
            solvers[target] = {**solver, "source": WORKFLOW}
    results["allocation"] = {
        "targets": {
            str(target): {"allocation": allocations[target], "solver": solvers[target]}
            for target in TARGETS
        },
        "swiglu7_width_resolve_report": resolve_report,
    }
    persist(output_dir, artifact, "allocation_completed")

    # Pass 2: fits for every width a fitted start needs and pass 1 did not produce.
    needed = sorted(
        {
            (int(row["layer"]), int(row["replacement_width"]))
            for allocation in allocations.values()
            for row in allocation
            if not row["retains_dense_module"]
        }
        - set(fit_rows)
    )
    needed_layers = sorted({layer for layer, unused_width in needed})
    for offset in range(0, len(needed_layers), group_size):
        group = needed_layers[offset : offset + group_size]
        unused_short, fitting = capture(group, False)
        for layer, width in needed:
            if layer in group:
                fit(layer, width, fitting)
        del unused_short, fitting
        release_memory()
    results["local_fitting"] = [
        {key: deepcopy(value) for key, value in row.items() if key != "state_path"}
        for unused_key, row in sorted(fit_rows.items())
    ]
    persist(output_dir, artifact, "local_fitting_completed")
    if recipe_id == CURRENT_SETTING:
        checks = []
        keys = sorted(
            {
                (int(row["layer"]), int(row["replacement_width"]))
                for target in SWIGLU7_TARGETS
                for row in allocations[target]
                if not row["retains_dense_module"]
            }
        )
        for key in keys:
            for metric in ("local_nmse", "local_mse"):
                checks.append(
                    comparison("P2", fit_rows[key][metric], reference_fits[key][metric],
                               tolerances["fit_relative"], layer=key[0], width=key[1], metric=metric)
                )
        enforce(artifact, output_dir, "P2", checks)

    path = output_dir / "prepared" / "selection" / "rankings.pt"
    save_tensor_atomic(path, {str(layer): ranking.clone() for layer, ranking in rankings.items()})
    results["selection"] = {
        "path": relative_output(output_dir, path),
        "sha256": file_digest(path),
        "ranking_digests": {
            str(layer): content_digest(ranking.tolist()) for layer, ranking in rankings.items()
        },
    }
    persist(output_dir, artifact, "selection_saved")
    return rankings, allocations, solvers, fit_rows


def build_start(context, settings, output_dir, artifact, target, arm, allocation, solver, rankings,
                fit_rows, recipe_id):
    """Save one arm's token-zero replacement state from the shared selection."""

    import torch

    hidden = int(settings["model"]["hidden_size"])
    original = int(settings["model"]["intermediate_size"])
    selection = selection_record(allocation, rankings)
    state = {}
    for row in allocation:
        if row["retains_dense_module"]:
            continue
        layer = int(row["layer"])
        width = int(row["replacement_width"])
        path = context.blocks[layer].path
        if arm == "fitted":
            fit = fit_rows[(layer, width)]
            source = resolve_source_asset(fit["state_path"], context.output)
            if file_digest(source) != fit["state_sha256"]:
                raise ValueError(f"Fitted operator state changed: {source}")
            state[path] = torch.load(source, map_location="cpu", weights_only=False)
        else:
            module = slice_module(context, layer, width, selected_indices(rankings[layer], width))
            state[path] = {name: value.detach().cpu().clone()
                           for name, value in module.state_dict().items()}
            del module
    recipe = recipe_id if arm == "fitted" else None
    recipe_fingerprint = content_digest({
        "target": target, "arm": arm, "recipe_id": recipe, "selection": selection,
        "allocation": allocation, "allocation_solver": solver,
    })
    payload = {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "stage": "prepared-start",
        "target": target,
        "arm": arm,
        "recipe_id": recipe,
        "recipe_fingerprint": recipe_fingerprint,
        "selection_fingerprint": selection["fingerprint"],
        "tokens_seen": 0,
        "optimizer_updates": 0,
        "training_seconds": 0.0,
        "replacement_state": state,
        **capture_rng(),
    }
    destination = output_dir / "prepared" / "starts" / f"{arm}-target-{target}.pt"
    save_tensor_atomic(destination, payload)
    retained = sum(int(row["replacement_parameters"]) for row in allocation)
    total = len(allocation) * 3 * hidden * original
    record = {
        "target": target,
        "arm": arm,
        "recipe_id": recipe,
        "recipe_fingerprint": recipe_fingerprint,
        "construction": f"swiglu8_{arm}",
        "selection": selection,
        "tokens_seen": 0,
        "optimizer_updates": 0,
        "training_seconds": 0.0,
        "allocation": deepcopy(allocation),
        "allocation_solver": deepcopy(solver),
        "original_eligible_parameters": total,
        "retained_eligible_parameters": retained,
        "realized_eligible_mlp_removal": 1.0 - retained / total,
        "path": relative_output(output_dir, destination),
        "sha256": file_digest(destination),
    }
    del state, payload
    release_memory()
    artifact["results"].setdefault("starts", {})[start_key(arm, target)] = record
    persist(output_dir, artifact, f"prepared_start:{arm}:{target}")
    return record


def validate_completed_preparation(artifact, output_dir, settings):
    results = artifact["results"]
    required = {
        "stream", "monitoring_evaluation", "evaluation_protocol", "dense_evaluation",
        "long_context_data", "width_curves", "allocation", "local_fitting", "selection",
        "starts", "acceptance",
    }
    if not required <= set(results):
        raise ValueError("Completed SwiGLU-8 preparation is missing required results")
    stream = results["stream"]
    stream_path = output_path(output_dir, stream["path"])
    if (
        int(stream["token_count"]) != int(settings["recovery"]["target_tokens"])
        or stream_path.stat().st_size != 4 * int(stream["token_count"])
        or file_digest(stream_path) != stream["sha256"]
    ):
        raise ValueError("Completed preparation recovery stream changed")
    monitor = results["monitoring_evaluation"]
    if file_digest(output_path(output_dir, monitor["path"])) != monitor["sha256"]:
        raise ValueError("Completed preparation monitoring batches changed")
    for record in results["evaluation_protocol"]["corpora"].values():
        verified_stream(output_dir, record)
    if results["dense_evaluation"].get("status") != "completed":
        raise ValueError("Completed preparation lacks the dense evaluation reference")
    verified_stream(output_dir, results["long_context_data"]["kl_stream"])
    selection = results["selection"]
    if file_digest(output_path(output_dir, selection["path"])) != selection["sha256"]:
        raise ValueError("Completed preparation neuron rankings changed")
    if set(results["starts"]) != {start_key(arm, target) for arm in ARMS for target in TARGETS}:
        raise ValueError("Completed preparation is missing a starting state")
    for record in results["starts"].values():
        if file_digest(output_path(output_dir, record["path"])) != record["sha256"]:
            raise ValueError("Completed preparation starting state changed")
    for target in TARGETS:
        fitted, sliced = (results["starts"][start_key(arm, target)] for arm in ARMS)
        if fitted["selection"]["fingerprint"] != sliced["selection"]["fingerprint"]:
            raise ValueError(f"Arms at target {target} do not share their neuron selection")
    groups = {"P1", "P3", "P4", "P5"}
    if settings["part_b"]["fitting_recipe"] == CURRENT_SETTING:
        groups.add("P2")
    acceptance = results["acceptance"]
    if not groups <= set(acceptance) or not all(group["passed"] for group in acceptance.values()):
        raise ValueError("Completed preparation lacks passed acceptance checks")


def prepare(args, settings, config_path):
    import torch
    from transformers import AutoTokenizer

    work_dir, output_dir = resolve_storage(args.work_dir, args.output_dir, args.resume)
    artifact = start_artifact(output_dir, settings, "prepare", args.resume, {"kind": "shared"})
    try:
        recipe_id = settings["part_b"]["fitting_recipe"]
        if recipe_id is None:
            raise ValueError("Set part_b from the approved block check before preparation")
        block = load_block_check(args.block_check, settings)
        reference = swiglu7_prepared(settings, args.swiglu7_root)
        contract = {
            "configuration": settings,
            "configuration_sha256": file_digest(config_path),
            "code_hashes": code_hashes(),
            "swiglu7_prepare_sha256": settings["sources"]["swiglu7_prepare_sha256"],
            "block_check_sha256": settings["part_b"]["block_check_sha256"],
        }
        run_fingerprint = content_digest(contract)
        if args.resume and artifact.get("run_fingerprint") != run_fingerprint:
            raise ValueError("Preparation configuration, sources, or maintained code changed")
        was_completed = artifact.get("status") == "completed"
        artifact.update(
            run_fingerprint=run_fingerprint,
            code_hashes=contract["code_hashes"],
            configuration_sha256=contract["configuration_sha256"],
            status="running",
            error=None,
        )
        if was_completed:
            validate_completed_preparation(artifact, output_dir, settings)
            artifact["status"] = "completed"
            persist(output_dir, artifact, "completed")
            return
        results = artifact["results"]
        if "storage_preflight" not in results:
            results["storage_preflight"] = preparation_storage_preflight(work_dir, output_dir, settings)
            persist(output_dir, artifact, "storage_preflight")
        tokenizer = AutoTokenizer.from_pretrained(
            settings["model"]["model_id"],
            revision=settings["model"]["tokenizer_revision"],
            trust_remote_code=settings["model"]["trust_remote_code"],
        )
        prepare_evaluation_protocol(settings, output_dir, artifact, tokenizer)
        stream = prepare_recovery_stream(settings, output_dir, artifact, tokenizer)
        if "P4" not in results.get("acceptance", {}):
            enforce(artifact, output_dir, "P4",
                    recovery_prefix_checks(stream, settings, args.swiglu7_root))
        persist(output_dir, artifact, "loading_models")
        context, model_config = fitting_context(settings, work_dir, output_dir, artifact, "prepare")
        prepare_monitoring_evaluation(settings, output_dir, artifact, tokenizer, local_data=context.data)
        recipe = settings["block_check"]["settings"][recipe_id]
        long_data = prepare_long_context_data(
            context, settings, output_dir, artifact, int(recipe["calibration_context"]) == 8192
        )
        kl_record = long_data["kl_stream"]
        kl_path = verified_stream(output_dir, kl_record)
        if results.get("dense_evaluation", {}).get("status") != "completed":
            prepare_dense_evaluation(context, settings, output_dir, artifact)
        if "P3" not in results.get("acceptance", {}):
            enforce(artifact, output_dir, "P3",
                    dense_checks(artifact, output_dir, reference, settings, args.swiglu7_root))
        if "P5" not in results.get("acceptance", {}):
            teacher, unused_tokenizer = load_model_and_tokenizer(model_config)
            validated_attention_implementation(teacher)
            checks = [
                teacher_self_kl(context, teacher, kl_path, kl_record, settings),
                *(
                    equality_check("P5", long_data[name]["sha256"],
                                   block["results"]["long_context_data"][name]["sha256"],
                                   metric=f"{name}_matches_block_check")
                    for name in ("kl_stream", "calibration_stream", "validation_stream")
                    if name in long_data
                ),
            ]
            del teacher, unused_tokenizer
            release_memory()
            enforce(artifact, output_dir, "P5", checks)
        expected_starts = {start_key(arm, target) for arm in ARMS for target in TARGETS}
        if set(results.get("starts", {})) != expected_starts:
            results["starts"] = {}
            rankings, allocations, solvers, fit_rows = prepare_operators(
                context, settings, output_dir, artifact, reference, recipe_id, long_data
            )
            for target in TARGETS:
                for arm in ARMS:
                    build_start(context, settings, output_dir, artifact, target, arm,
                                allocations[target], solvers[target], rankings, fit_rows, recipe_id)
        del context
        release_memory()
        validate_completed_preparation(artifact, output_dir, settings)
        artifact["status"] = "completed"
        persist(output_dir, artifact, "completed")
    except BaseException as error:
        record_failure(output_dir, work_dir, artifact, error)
        raise


# Part B: recovery and final evaluation


def load_prepared(path, settings):
    path = resolve_path(Path(path))
    prepared = read_json(path)
    if (
        prepared.get("workflow") != WORKFLOW
        or prepared.get("command") != "prepare"
        or prepared.get("status") != "completed"
        or prepared.get("configuration") != settings
        or not all(group["passed"] for group in prepared["results"]["acceptance"].values())
    ):
        raise ValueError("Training requires a completed, accepted, matching SwiGLU-8 preparation")
    return path, prepared, path.parent


def check_monitoring(artifact, output_dir, reference_history, settings):
    checks = [
        check
        for row in artifact["results"]["recovery"]["validation_history"]
        for check in monitoring_checks(row, reference_history, settings["acceptance"])
    ]
    enforce(artifact, output_dir, "B1-B2", checks)


def final_teacher_kl(artifact, output_dir, prepared_root, record, settings):
    """Score the reloaded BF16 bundle with the shared 8K teacher-KL protocol."""

    evaluation = artifact["results"]["evaluation"]
    if "teacher_kl_8192" in evaluation:
        return
    path = verified_stream(prepared_root, record)
    model, tokenizer, unused_manifest = load_bundle(output_dir / "model")
    teacher, unused_tokenizer = load_model_and_tokenizer(make_model_config(settings["model"]))
    started = perf_counter()
    try:
        result = evaluate_teacher_kl(model, teacher, path, record,
                                     int(settings["long_context"]["sequence_length"]))
    finally:
        del model, tokenizer, teacher, unused_tokenizer
        release_memory()
    evaluation["teacher_kl_8192"] = {**result, "seconds": perf_counter() - started}
    persist(output_dir, artifact, "final_teacher_kl")


def train(args, settings, config_path):
    import torch

    work_dir, output_dir = resolve_storage(args.work_dir, args.output_dir, args.resume)
    target = float(args.target)
    arm = args.arm
    if target not in TARGETS or arm not in ARMS:
        raise ValueError("Training target or arm is outside the fixed SwiGLU-8 grid")
    identity = {"arm": arm, "target": target}
    artifact = start_artifact(output_dir, settings, "train", args.resume, identity)
    try:
        prepared_path, prepared, prepared_root = load_prepared(args.prepared, settings)
        validate_evaluation_preflight(prepared, prepared_root, settings)
        starts = prepared["results"]["starts"]
        start_record = starts[start_key(arm, target)]
        partner = starts[start_key(next(value for value in ARMS if value != arm), target)]
        if (
            start_record["selection"]["fingerprint"] != partner["selection"]["fingerprint"]
            or start_record["selection"]["widths"] != partner["selection"]["widths"]
        ):
            raise ValueError("Both arms at one target must share their neurons and widths")
        if (
            start_record.get("arm") != arm
            or int(start_record.get("tokens_seen", -1)) != 0
            or int(start_record.get("optimizer_updates", -1)) != 0
        ):
            raise ValueError("SwiGLU-8 requires a fresh token-zero start for this arm")
        start_path = output_path(prepared_root, start_record["path"])
        if file_digest(start_path) != start_record["sha256"]:
            raise ValueError("Prepared starting state changed")
        stream = prepared["results"]["stream"]
        stream_path = verified_stream(prepared_root, stream)
        kl_record = prepared["results"]["long_context_data"]["kl_stream"]
        verified_stream(prepared_root, kl_record)
        reproduction = (
            arm == "fitted"
            and target in SWIGLU7_TARGETS
            and settings["part_b"]["fitting_recipe"] == CURRENT_SETTING
        )
        reference_history = (
            swiglu7_run(settings, args.swiglu7_root, target)["results"]["recovery"]["validation_history"]
            if reproduction
            else None
        )
        contract = {
            "configuration": settings,
            "configuration_sha256": file_digest(config_path),
            "prepared_sha256": file_digest(prepared_path),
            "start_sha256": start_record["sha256"],
            "stream_sha256": stream["sha256"],
            "identity": identity,
            "code_hashes": code_hashes(),
            "swiglu7_run_sha256": (
                settings["sources"]["swiglu7_runs_sha256"][str(target)] if reproduction else None
            ),
        }
        run_fingerprint = content_digest(contract)
        if args.resume and artifact.get("run_fingerprint") != run_fingerprint:
            raise ValueError("Resume inputs, configuration, or maintained code changed")
        was_completed = artifact.get("status") == "completed"
        artifact.update(
            run_fingerprint=run_fingerprint,
            code_hashes=contract["code_hashes"],
            configuration_sha256=contract["configuration_sha256"],
            prepared={"sha256": contract["prepared_sha256"]},
            start={"sha256": contract["start_sha256"], "tokens_seen": 0},
            reproduction_check=reproduction,
            status="running",
            error=None,
        )
        checkpoint_dir = output_dir / "checkpoint"
        if was_completed:
            validate_bundle(output_dir / "model")
            if artifact["results"]["evaluation"]["model"]["manifest_sha256"] != file_digest(
                output_dir / "model" / "bundle.json"
            ):
                raise ValueError("Completed result and bundle manifest differ")
            for path in list(checkpoint_dir.glob("checkpoint-*")) if checkpoint_dir.exists() else []:
                path.unlink()
            if checkpoint_dir.exists():
                checkpoint_dir.rmdir()
            artifact.pop("checkpoint", None)
            artifact["status"] = "completed"
            artifact["results"]["checkpoint_cleanup"] = "completed_after_result_and_bundle_validation"
            persist(output_dir, artifact, "completed")
            return
        artifact["results"].setdefault(
            "recovery",
            {
                "validation_history": [],
                "tokens_seen": 0,
                "optimizer_updates": 0,
                "training_seconds": 0.0,
                "evaluation_seconds": 0.0,
                "checkpoint_seconds": 0.0,
            },
        )
        persist(output_dir, artifact, "loading_models")
        start = torch.load(start_path, map_location="cpu", weights_only=False)
        if (
            int(start.get("tokens_seen", -1)) != 0
            or int(start.get("optimizer_updates", -1)) != 0
            or "optimizer_state" in start
            or start.get("arm") != arm
            or start.get("recipe_fingerprint") != start_record["recipe_fingerprint"]
            or start.get("selection_fingerprint") != start_record["selection"]["fingerprint"]
        ):
            raise ValueError("Prepared start payload differs from its descriptor")
        model_config = make_model_config(settings["model"])
        teacher, tokenizer = load_model_and_tokenizer(model_config)
        student, target_paths, unused_train_modules = build_swiglu_student(
            model_config, settings["model"]["hidden_size"], start_record["allocation"]
        )
        artifact["results"]["attention_implementation"] = {
            "teacher": validated_attention_implementation(teacher),
            "student": validated_attention_implementation(student),
        }
        load_replacement_state(student, start["replacement_state"])
        restore_rng(start)
        groups, train_modules, adapters = configure_strategy(student, target_paths, "S7-0", settings)
        artifact["results"]["trainable_scope"] = group_metadata(groups, student)
        artifact["results"]["storage_preflight"] = storage_preflight(
            output_dir, groups, student, settings["recovery"]["disk_reserve_fraction"]
        )
        device = next(student.parameters()).device
        monitoring_record = prepared["results"]["monitoring_evaluation"]
        monitoring_path = output_path(prepared_root, monitoring_record["path"])
        if file_digest(monitoring_path) != monitoring_record["sha256"]:
            raise ValueError("Prepared validation batches changed")
        monitoring = torch.load(monitoring_path, map_location="cpu", weights_only=False)
        validation_cache = cache_teacher_logits(
            teacher,
            monitoring["recovery_validation"],
            len(monitoring["recovery_validation"]),
            device,
            "float16",
        )
        cache = PackedTokenCache(
            stream_path, int(stream["token_count"]), settings["recovery"]["sequence_length"],
            stream["sha256"],
        )
        recovery = settings["recovery"]
        recovered = None
        if args.resume:
            try:
                recovered, descriptor = restore_checkpoint(checkpoint_dir, run_fingerprint)
            except FileNotFoundError:
                if int(artifact["results"]["recovery"]["tokens_seen"]) > 0:
                    raise
            if recovered is not None:
                load_trainable_state(student, recovered["trainable_state"], groups)
                artifact["results"]["recovery"] = deepcopy(recovered["recovery"])
                artifact["checkpoint"] = descriptor
                restore_rng(recovered)
        if recovered is None:
            optimizer_state = None
            artifact["results"]["recovery"].update({"tokens_seen": 0, "optimizer_updates": 0})
            if not artifact["results"]["recovery"]["validation_history"]:
                rng = capture_rng()
                artifact["results"]["recovery"]["validation_history"].append(
                    {
                        "actual_tokens": 0,
                        "requested_tokens": [0],
                        "optimizer_updates": 0,
                        "training_seconds": 0.0,
                        "mean_train_kl": None,
                        "recovery_validation_kl": evaluate_validation_kl_mixed(
                            student, validation_cache, 1.0, device
                        ),
                        "learning_rates": [group["learning_rate"] for group in groups],
                        "wikitext_validation": evaluate_lm_mixed(
                            student,
                            monitoring["model_validation"],
                            device,
                            settings["data"]["model_validation_batches"],
                        ),
                    }
                )
                restore_rng(rng)
            initial_payload = {
                "schema_version": SCHEMA_VERSION,
                "workflow": WORKFLOW,
                "run_fingerprint": run_fingerprint,
                "stream_sha256": stream["sha256"],
                "arm": arm,
                "target": target,
                "tokens_seen": 0,
                "optimizer_updates": 0,
                "trainable_state": trainable_state(student, groups),
                "optimizer_state": None,
                "recovery": deepcopy(artifact["results"]["recovery"]),
                **capture_rng(),
            }
            artifact["checkpoint"] = commit_single_checkpoint(checkpoint_dir, initial_payload)
            persist(output_dir, artifact, "recovery_initialized")
            del initial_payload
        else:
            optimizer_state = recovered["optimizer_state"]
        if reproduction:
            check_monitoring(artifact, output_dir, reference_history, settings)
        del start, recovered, unused_train_modules
        cursor = int(artifact["results"]["recovery"]["tokens_seen"])
        updates = int(artifact["results"]["recovery"]["optimizer_updates"])
        final_tokens = int(recovery["target_tokens"])
        while cursor < final_tokens:
            origin = segment_origin(cursor, recovery["segment_endpoints"])
            end = next(value for value in recovery["segment_endpoints"] if value > cursor)
            schedule = segment_schedule(
                origin, end, desired_requests(settings), recovery["effective_batch_tokens"]
            )

            def on_checkpoint(event, optimizer, first_step):
                started = perf_counter()
                rng = capture_rng()
                validation_kl = evaluate_validation_kl_mixed(student, validation_cache, 1.0, device)
                if not math.isfinite(validation_kl) or not math.isfinite(event.mean_train_kl):
                    raise FloatingPointError("Non-finite recovery loss; invalid state was not committed")
                row = {
                    "actual_tokens": event.tokens_seen,
                    "requested_tokens": list(event.requested_checkpoint_tokens),
                    "optimizer_updates": event.optimizer_updates,
                    "training_seconds": event.elapsed_seconds,
                    "mean_train_kl": event.mean_train_kl,
                    "recovery_validation_kl": validation_kl,
                    "learning_rates": list(event.learning_rates),
                }
                if any(
                    point in recovery["monitoring_ppl_tokens"]
                    for point in event.requested_checkpoint_tokens
                ):
                    row["wikitext_validation"] = evaluate_lm_mixed(
                        student,
                        monitoring["model_validation"],
                        device,
                        settings["data"]["model_validation_batches"],
                    )
                restore_rng(rng)
                results = artifact["results"]["recovery"]
                results["validation_history"].append(row)
                results.update(
                    {
                        "tokens_seen": event.tokens_seen,
                        "optimizer_updates": event.optimizer_updates,
                        "training_seconds": event.elapsed_seconds,
                        "first_step": results.get("first_step") or first_step,
                        "memory": recovery_memory_record(device),
                    }
                )
                results["evaluation_seconds"] += perf_counter() - started
                checkpoint_started = perf_counter()
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "workflow": WORKFLOW,
                    "run_fingerprint": run_fingerprint,
                    "stream_sha256": stream["sha256"],
                    "arm": arm,
                    "target": target,
                    "tokens_seen": event.tokens_seen,
                    "optimizer_updates": event.optimizer_updates,
                    "trainable_state": trainable_state(student, groups),
                    "optimizer_state": cpu_tree(optimizer.state_dict()),
                    "recovery": deepcopy(results),
                    **rng,
                }
                artifact["checkpoint"] = commit_single_checkpoint(checkpoint_dir, payload)
                results["checkpoint_seconds"] += perf_counter() - checkpoint_started
                persist(
                    output_dir,
                    artifact,
                    "recovery",
                    {"arm": arm, "target": target, "tokens_seen": event.tokens_seen,
                     "target_tokens": final_tokens},
                )
                print(
                    f"arm={arm} target={target} tokens={event.tokens_seen:,} KL={validation_kl:.6f}",
                    flush=True,
                )
                if reproduction:
                    check_monitoring(artifact, output_dir, reference_history, settings)

            result = recover_exact_segment(
                origin=origin,
                end=end,
                cursor=cursor,
                schedule=schedule,
                student=student,
                teacher=teacher,
                parameter_groups=[
                    {
                        "name": group["name"],
                        "parameters": group["parameters"],
                        "learning_rate": group["learning_rate"],
                        "weight_decay": group["weight_decay"],
                    }
                    for group in groups
                ],
                train_modules=train_modules,
                batch_at=lambda offset, count: cache.batch(
                    offset,
                    count,
                    recovery["microbatch_sequences"],
                    allow_partial_sequence=True,
                ),
                microbatch_tokens=recovery["sequence_length"] * recovery["microbatch_sequences"],
                accumulation_steps=recovery["gradient_accumulation_steps"],
                temperature=recovery["temperature"],
                ce_weight=recovery["ce_weight"],
                scheduler=recovery["scheduler"],
                warmup_fraction=recovery["warmup_fraction"],
                final_lr_ratio=recovery["final_lr_ratio"],
                device=device,
                autocast_dtype=torch.bfloat16,
                start_updates=updates,
                elapsed_seconds=artifact["results"]["recovery"]["training_seconds"],
                optimizer_state=optimizer_state,
                on_checkpoint=on_checkpoint,
                optimizer_backend=recovery["optimizer_backend"],
            )
            cursor, updates = result.tokens_seen, result.optimizer_updates
            if cursor < final_tokens:
                state, unused_descriptor = restore_checkpoint(checkpoint_dir, run_fingerprint)
                optimizer_state = state["optimizer_state"]
                restore_rng(state)
                del state
                gc.collect()
        del validation_cache, monitoring, teacher
        release_memory()
        for group in groups:
            group["parameters"].clear()
        train_modules.clear()
        persist(output_dir, artifact, "final_evaluation")
        final_evaluation_started = perf_counter()
        evaluate_final_model(
            artifact,
            output_dir,
            work_dir,
            prepared,
            prepared_root,
            settings,
            student,
            tokenizer,
            start_record["allocation"],
            adapters,
            run_fingerprint,
            measure_module=MEASURE_MODULE,
        )
        del student, tokenizer
        final_teacher_kl(artifact, output_dir, prepared_root, kl_record, settings)
        artifact["results"]["final_evaluation_wall_seconds"] = (
            float(artifact["results"].get("final_evaluation_wall_seconds", 0.0))
            + perf_counter()
            - final_evaluation_started
        )
        if reproduction and not artifact["results"]["acceptance"]["B1-B2"]["passed"]:
            raise AcceptanceFailure("B1-B2 did not pass")
        artifact["status"] = "completed"
        artifact["results"]["checkpoint_cleanup"] = "pending_validation"
        persist(output_dir, artifact, "validating_final_artifacts")
        completed = read_json(output_dir / "result.json")
        if completed.get("status") != "completed" or completed.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError("Completed result did not survive its JSON round trip")
        if completed["results"]["evaluation"]["model"]["manifest_sha256"] != file_digest(
            output_dir / "model" / "bundle.json"
        ):
            raise RuntimeError("Completed result does not reference the final bundle manifest")
        validate_bundle(output_dir / "model")
        for path in list(checkpoint_dir.glob("checkpoint-*")) if checkpoint_dir.exists() else []:
            path.unlink()
        if checkpoint_dir.exists():
            checkpoint_dir.rmdir()
        artifact.pop("checkpoint", None)
        artifact["results"]["checkpoint_cleanup"] = "completed_after_result_and_bundle_validation"
        persist(output_dir, artifact, "completed")
    except BaseException as error:
        record_failure(output_dir, work_dir, artifact, error)
        raise


# Command line


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("block-check", "prepare", "train"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        subparser.add_argument("--work-dir", type=Path, required=True)
        subparser.add_argument("--output-dir", type=Path, required=True)
        subparser.add_argument("--swiglu7-root", type=Path, default=DEFAULT_SWIGLU7_ROOT)
        subparser.add_argument("--resume", action="store_true")
    subparsers.choices["prepare"].add_argument("--block-check", type=Path, required=True)
    train_parser = subparsers.choices["train"]
    train_parser.add_argument("--prepared", type=Path, required=True)
    train_parser.add_argument("--arm", choices=ARMS, required=True)
    train_parser.add_argument("--target", type=float, choices=TARGETS, required=True)
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
    if args.command == "block-check":
        block_check(args, settings, config_path)
    elif args.command == "prepare":
        prepare(args, settings, config_path)
    else:
        train(args, settings, config_path)


if __name__ == "__main__":
    main()
