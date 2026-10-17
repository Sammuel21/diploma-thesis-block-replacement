"""Run SwiGLU-9: local fitting after 1B recovery tokens and calibration scaling.

``block-check`` runs Part A for one test layer. It first reproduces SwiGLU-8's
MSE fit at 393,216 pairs, then fits MSE and two L2-SP strengths at four nested
calibration sizes. ``merge-block-check`` joins the three layer tasks and applies
SwiGLU-8's adoption rule against its three-seed reference. ``train`` recovers one
arm at one target to 1B tokens from SwiGLU-8's token-zero start on SwiGLU-7's 1B
stream (Part B). ``reference-kl`` scores SwiGLU-7's fitted 1B bundles with the
shared 8K KL, and ``summarize-part-b`` applies the decided per-target verdict.
The binding design is recorded in ``docs/agents/decisions.md`` and
``docs/experiments/model/swiglu/swiglu-9.md``.
"""

from __future__ import annotations

import argparse
import gc
import math
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from workflows.runs.model.common import resolve_path

from mlp_replacement.artifacts import content_digest, file_digest, read_json
from mlp_replacement.capture import ActivationPairs, collect_modules_io
from mlp_replacement.compression.continuation import (
    capture_rng,
    commit_single_checkpoint,
    recover_exact_segment,
    restore_checkpoint,
    restore_rng,
    segment_origin,
    segment_schedule,
)
from mlp_replacement.compression.reconstruction import (
    build_swiglu_student,
    load_replacement_state,
)
from mlp_replacement.compression.teacher_cache import cache_teacher_logits
from mlp_replacement.config import DatasetSpec
from mlp_replacement.data import PackedTokenCache, load_text_dataset, sample_partitioned_windows
from mlp_replacement.evaluation import evaluate_operator
from mlp_replacement.evaluation.bundles import load_bundle, validate_bundle
from mlp_replacement.evaluation.teacher_kl import evaluate_teacher_kl
from mlp_replacement.model import load_model_and_tokenizer
from mlp_replacement.runlog import environment_record

from .shared import (
    evaluate_lm_mixed,
    evaluate_validation_kl_mixed,
    make_model_config,
    recovery_memory_record,
)
from .swiglu5.fitting import capture_dense_pairs, fit_operator, load_fit_operator
from .swiglu_7 import (
    code_hashes,
    configure_strategy,
    cpu_tree,
    desired_requests,
    evaluate_final_model,
    group_metadata,
    load_trainable_state,
    measure_bundle,
    output_path,
    persist,
    resolve_storage,
    storage_preflight,
    trainable_state,
    utc_now,
    validate_evaluation_preflight,
    validated_attention_implementation,
)
from .swiglu_8 import (
    CONTEXTS,
    REPRODUCTION_REVIEW_RELATIVE,
    AcceptanceFailure,
    check_monitoring,
    comparison,
    enforce,
    equality_check,
    evaluate_block,
    final_teacher_kl,
    fit_context_kind,
    fitting_context,
    is_digest,
    long_context_view,
    monitoring_checks,
    neuron_ranking,
    pinned_json,
    prepare_long_context_data,
    recipe_context,
    record_checks,
    record_failure,
    relative_difference,
    release_memory,
    seed_summary,
    selected_indices,
    slice_module,
    stage_two_indicator,
    start_key,
    teacher_self_kl,
    verified_stream,
    width_from_ratio,
)
from .swiglu_8 import load_settings as load_swiglu8_settings


WORKFLOW = "swiglu-9"
SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path("workflows/configs/model/swiglu/swiglu-9.json")
SWIGLU7_CONFIG = Path("workflows/configs/model/swiglu/swiglu-7.json")
SWIGLU8_CONFIG = Path("workflows/configs/model/swiglu/swiglu-8.json")
DEFAULT_SWIGLU7_ROOT = Path("data/results/workflows/model/swiglu-7")
DEFAULT_SWIGLU8_ROOT = Path("data/results/workflows/model/swiglu-8")
MEASURE_MODULE = "workflows.runs.model.swiglu.swiglu_9"
LAYERS = (3, 11, 19)
CALIBRATION_SIZES = (393_216, 786_432, 1_572_864, 3_145_728)
# None is plain MSE; a number is the L2-SP strength ratio rho (implementation plan, 2026-10-17).
LOSSES = {"mse": None, "l2sp-weak": 0.1, "l2sp-strong": 1.0}
REFERENCE_SETTING = "c1e-3-128"
REFERENCE_SEEDS = (21, 22, 23)
TARGETS = (0.2, 0.5, 0.8)
REUSED_FITTED_TARGETS = (0.2, 0.5)
# Array order of the Part B training job; the job script repeats it.
RUNS = (("fitted", 0.5), ("fitted", 0.8), ("slice", 0.2), ("slice", 0.5), ("slice", 0.8))
TWIN = ("fitted", 0.5)


def size_label(pairs):
    return f"{round(int(pairs) / 1000)}k"


def part_a_settings():
    """Build the decided Part A grid of loss × calibration size (decision log, 2026-10-17).

    The first entry, MSE at 393,216 pairs, is SwiGLU-8's reused cell. It runs
    once more per layer as the reproduction fit that also calibrates L2-SP.
    """

    return {
        f"{loss}-{size_label(pairs)}": {"loss": loss, "calibration_pairs": pairs, "l2_sp_ratio": ratio}
        for pairs in CALIBRATION_SIZES
        for loss, ratio in LOSSES.items()
    }


PART_A_SETTINGS = part_a_settings()
REPRODUCTION_SETTING = next(iter(PART_A_SETTINGS))
NEW_SETTINGS = tuple(setting for setting in PART_A_SETTINGS if setting != REPRODUCTION_SETTING)
EXTENSION = {
    "seed": 21,
    "windows": (max(CALIBRATION_SIZES) - min(CALIBRATION_SIZES)) // 128,
    "excluded_records": (
        "every record that supplied a 128-token window and every record up to the last "
        "8K selection-stream document"
    ),
}
L2_SP_CALIBRATION = {
    "reference_setting": REPRODUCTION_SETTING,
    "alpha": (
        "2 * l2_sp_ratio * (training MSE of the slice - training MSE of the reference fit) / "
        "squared distance of the reference fit from the slice, on the 393,216-pair prefix, "
        "per layer and width"
    ),
    "fixed_across_sizes": True,
}
ADOPTION_RULE = {
    "deciding_context": 8192,
    "minimum_mean_reduction": 0.05,
    "maximum_combination_increase": 0.05,
    "reference": "swiglu-8 block check, c1e-3-128 at seeds 21, 22, and 23",
    "seed_noise": "twice the reference's sample seed standard deviation over the reference mean",
}
PART_B = {
    "targets": list(TARGETS),
    "fitting_recipe": REFERENCE_SETTING,
    "runs": [
        {"arm": arm, "target": target, "role": "noise_twin" if (arm, target) == TWIN else "new"}
        for arm, target in RUNS
    ],
    "reused_fitted_targets": list(REUSED_FITTED_TARGETS),
    "verdict": {
        "gap_bar": 0.05,
        "noise_window_tokens": [500_000_000, 1_000_000_000],
        "noise_includes_final_teacher_kl": True,
    },
}


# Configuration and source artifacts


def load_settings(path):
    """Load the SwiGLU-9 configuration and check every block it shares with SwiGLU-7 and -8."""

    path = resolve_path(Path(path))
    settings = read_json(path)
    if settings.get("schema_version") != SCHEMA_VERSION or settings.get("workflow") != WORKFLOW:
        raise ValueError("Expected a schema-1 SwiGLU-9 configuration")
    if settings.get("seed") != 21:
        raise ValueError("SwiGLU-9 fixes seed 21")
    swiglu7 = read_json(resolve_path(SWIGLU7_CONFIG))
    swiglu8 = read_json(resolve_path(SWIGLU8_CONFIG))
    for key in ("model", "data", "calibration", "local_fitting", "evaluation", "recovery"):
        if settings.get(key) != swiglu7[key]:
            raise ValueError(f"SwiGLU-9 must reuse SwiGLU-7's {key} contract")
    if settings.get("strategies") != {"S7-0": swiglu7["strategies"]["S7-0"]}:
        raise ValueError("SwiGLU-9 recovers with SwiGLU-7's replacement-only scope")
    for key in ("long_context", "allocation"):
        if settings.get(key) != swiglu8[key]:
            raise ValueError(f"SwiGLU-9 must reuse SwiGLU-8's {key} contract")
    if settings.get("acceptance") != {
        **swiglu8["acceptance"], "reproduction_review_relative": REPRODUCTION_REVIEW_RELATIVE,
    }:
        raise ValueError("SwiGLU-9 fixes SwiGLU-8's acceptance tolerances and its 1% review limit")
    block_check = swiglu8["block_check"]
    if block_check["layers"] != list(LAYERS) or block_check["width_ratios"] != [0.25, 0.5]:
        raise ValueError("SwiGLU-9 Part A uses SwiGLU-8 Part A's six layer-width combinations")
    if CALIBRATION_SIZES[0] != settings["calibration"]["selected_pairs"]:
        raise ValueError("The smallest calibration size must be SwiGLU-8's 393,216 pairs")
    if settings.get("part_a") != {
        "layers": list(LAYERS),
        "width_ratios": [0.25, 0.5],
        "seed": 21,
        "recipe_id": REFERENCE_SETTING,
        "recipe": block_check["settings"][REFERENCE_SETTING],
        "calibration_sizes": list(CALIBRATION_SIZES),
        "extension": EXTENSION,
        "losses": LOSSES,
        "l2_sp_calibration": L2_SP_CALIBRATION,
        "settings": PART_A_SETTINGS,
        "adoption": ADOPTION_RULE,
    }:
        raise ValueError("SwiGLU-9 fixes the decided Part A grid, recipe, and adoption rule")
    if settings.get("part_b") != PART_B or swiglu8["part_b"]["fitting_recipe"] != REFERENCE_SETTING:
        raise ValueError("SwiGLU-9 fixes the decided Part B runs and verdict")
    sources = settings.get("sources", {})
    expected = {
        "swiglu7_prepare_sha256", "swiglu7_runs_sha256", "swiglu8_prepare_sha256",
        "swiglu8_block_check_sha256", "swiglu8_runs_sha256",
    }
    digests = [
        sources.get("swiglu7_prepare_sha256"),
        sources.get("swiglu8_prepare_sha256"),
        sources.get("swiglu8_block_check_sha256"),
        *sources.get("swiglu7_runs_sha256", {}).values(),
        *sources.get("swiglu8_runs_sha256", {}).values(),
    ]
    if (
        set(sources) != expected
        or set(sources["swiglu7_runs_sha256"]) != {str(target) for target in REUSED_FITTED_TARGETS}
        or set(sources["swiglu8_runs_sha256"]) != {start_key(arm, target) for arm, target in RUNS}
        or not all(is_digest(value) for value in digests)
    ):
        raise ValueError("SwiGLU-9 pins its SwiGLU-7 and SwiGLU-8 source artifacts by sha256")
    if (
        sources["swiglu7_prepare_sha256"] != swiglu8["sources"]["swiglu7_prepare_sha256"]
        or sources["swiglu8_block_check_sha256"] != swiglu8["part_b"]["block_check_sha256"]
        or any(
            sources["swiglu7_runs_sha256"][key] != swiglu8["sources"]["swiglu7_runs_sha256"][key]
            for key in sources["swiglu7_runs_sha256"]
        )
    ):
        raise ValueError("SwiGLU-9 pins the same SwiGLU-7 and block-check artifacts as SwiGLU-8")
    return settings, path


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
            raise ValueError("Resume output belongs to a different SwiGLU-9 run")
        if artifact.get("status") != "completed":
            if artifact.get("environment") != environment_record():
                raise ValueError("Resume environment differs from the recorded SwiGLU-9 run")
            artifact["status"] = "running"
            artifact["error"] = None
        return artifact
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "experiment_family": "swiglu-9",
        "experiment_class": "local-fitting-at-1b-and-calibration-scaling",
        "command": command,
        "identity": identity,
        "status": "running",
        "created_at_utc": utc_now(),
        "configuration": deepcopy(settings),
        "environment": environment_record(),
        "results": {},
        "error": None,
    }


def swiglu8_block_check(settings, root):
    block = pinned_json(root, "block-check-001/result.json",
                        settings["sources"]["swiglu8_block_check_sha256"], "SwiGLU-8 block check")
    if block.get("status") != "completed" or block["results"]["adoption"]["selected_recipe"] != REFERENCE_SETTING:
        raise ValueError("SwiGLU-9 requires SwiGLU-8's completed block check, which kept c1e-3-128")
    return block


def swiglu8_run(settings, root, arm, target):
    return pinned_json(root, f"{arm}-target-{target}-run-001/result.json",
                       settings["sources"]["swiglu8_runs_sha256"][start_key(arm, target)],
                       f"SwiGLU-8 {arm} run at {target}")


def swiglu7_run(settings, root, target):
    return pinned_json(root, f"S7-0-target-{target:.1f}-run-001/result.json",
                       settings["sources"]["swiglu7_runs_sha256"][str(target)],
                       f"SwiGLU-7 S7-0 run at {target}")


def load_swiglu8_prepared(path, settings):
    """Load SwiGLU-8's completed preparation, the source of every Part B start and protocol."""

    path = resolve_path(Path(path))
    if file_digest(path) != settings["sources"]["swiglu8_prepare_sha256"]:
        raise ValueError("SwiGLU-8 preparation differs from its pinned sha256")
    prepared = read_json(path)
    swiglu8_settings, unused_path = load_swiglu8_settings(SWIGLU8_CONFIG)
    if (
        prepared.get("workflow") != "swiglu-8"
        or prepared.get("command") != "prepare"
        or prepared.get("status") != "completed"
        or prepared.get("configuration") != swiglu8_settings
        or not all(group["passed"] for group in prepared["results"]["acceptance"].values())
    ):
        raise ValueError("SwiGLU-9 requires SwiGLU-8's completed, accepted preparation")
    return path, prepared, path.parent


# Part A: calibration data, fitting, and L2-SP strengths


def block_check_layer(task_index):
    if not 0 <= int(task_index) < len(LAYERS):
        raise ValueError(f"Block-check task index must be 0-{len(LAYERS) - 1}")
    return LAYERS[int(task_index)]


def block_check_output_name(task_index):
    return f"block-check-001-task-{int(task_index):02d}"


def calibration_extension(context, settings, long_data):
    """Sample the windows that grow calibration beyond SwiGLU-8's 393,216 pairs.

    They come from the local C4 shard with the same window sampler, one window
    per document. Every record that supplied a 128-token window and every record
    up to the last 8K selection-stream document is excluded, so no calibration
    set shares a document with an 8K stream or another 128-token partition.
    """

    extension = settings["part_a"]["extension"]
    source = DatasetSpec(**settings["data"]["local_source"])
    records = load_text_dataset(source)
    last_long_context = int(long_data["selection_stream"]["last_record"])
    base = {int(index) for index in context.record_indices}
    excluded = base | set(range(last_long_context + 1))
    indices = []
    windows = sample_partitioned_windows(
        records,
        context.tokenizer,
        {"extension": int(extension["windows"])},
        int(settings["data"]["sequence_length"]),
        int(extension["seed"]),
        source.text_column,
        record_indices=indices,
        excluded_records=excluded,
    )["extension"]
    audit = {
        "windows": len(windows),
        "seed": int(extension["seed"]),
        "record_indices_digest": content_digest(indices),
        "record_range": [min(indices), max(indices)],
        "excluded_long_context_records": [0, last_long_context],
        "excluded_base_window_records": len(base),
        "document_disjoint": not (set(indices) & excluded) and len(set(indices)) == len(indices),
    }
    return windows, audit


def data_identity_checks(long_data, audit, reference, settings):
    """D1: the base windows, 8K streams, and extension match SwiGLU-8 and stay disjoint."""

    recorded = reference["results"]["long_context_data"]
    checks = [
        equality_check("D1", long_data["audit"]["local_window_records_digest"],
                       recorded["audit"]["local_window_records_digest"], metric="base_window_records_digest"),
        *(
            equality_check("D1", long_data[name]["sha256"], recorded[name]["sha256"], metric=f"{name}_sha256")
            for name in ("kl_stream", "calibration_stream", "validation_stream", "selection_stream")
        ),
        equality_check("D1", audit["windows"], int(settings["part_a"]["extension"]["windows"]),
                       metric="extension_windows"),
        equality_check("D1", audit["document_disjoint"], True, metric="extension_document_disjoint"),
    ]
    return checks


def capture_layer_pairs(context, layer, sequences, pairs):
    """Capture one layer's 128-token pairs once, at the largest calibration size.

    Pairs flatten batch by batch in sequence order, so the first N pairs are
    exactly the pairs a capture of the first N / 128 sequences would give.
    """

    view = SimpleNamespace(**vars(context))
    view.settings = deepcopy(context.settings)
    view.settings["references"]["selected_calibration_pairs"] = int(pairs)
    view.data = {**context.data, "calibration_sequences": sequences}
    path = context.blocks[layer].path
    training, validation = capture_dense_pairs(view, [layer])
    if training[path].num_tokens != int(pairs):
        raise RuntimeError("Calibration capture does not reach the largest pair count")
    return training[path], validation[path]


def capture_long_validation(context, output_dir, long_data, settings, layer):
    """Capture the 8K validation pairs that report 8K NMSE; 8K training pairs are not needed."""

    view = long_context_view(context, output_dir, long_data, settings)
    path = context.blocks[layer].path
    return collect_modules_io(
        context.model,
        [path],
        view.data["operator_validation"],
        int(view.data["partition_batches"]["operator_validation"]),
        context.device,
        storage_device="cpu",
        storage_dtype=context.model_dtype,
    )[path]


def calibration_prefix(pairs, count):
    if int(count) > pairs.num_tokens:
        raise ValueError("Requested calibration size exceeds the captured pairs")
    return ActivationPairs(inputs=pairs.inputs[: int(count)], targets=pairs.targets[: int(count)])


def squared_distance(module, reference):
    """Return ||w - w0||^2 over every parameter, accumulated in FP64."""

    reference_state = reference.state_dict()
    return float(sum(
        (value.detach().double() - reference_state[name].detach().double()).square().sum()
        for name, value in module.state_dict().items()
    ))


def l2_sp_alpha(ratio, mse_gain, distance):
    """Project-proposed L2-SP strength: alpha = 2 * ratio * mse_gain / distance.

    Along the reference fit's direction, a quadratic training MSE with this
    gain places the L2-SP minimum at 1 / (1 + ratio) of the fit's displacement.
    """

    if not mse_gain > 0 or not distance > 0:
        raise ValueError("L2-SP calibration needs a fit that lowers the training MSE and moves the slice")
    return 2.0 * float(ratio) * float(mse_gain) / float(distance)


def l2_sp_calibration(context, start, fitted, training_pairs, distance):
    batch = int(context.settings["local_fitting"]["batch_size"])
    slice_mse = float(evaluate_operator(start, training_pairs, context.device, batch).mse)
    fitted_mse = float(evaluate_operator(fitted, training_pairs, context.device, batch).mse)
    gain = slice_mse - fitted_mse
    return {
        "calibration_pairs": training_pairs.num_tokens,
        "slice_training_mse": slice_mse,
        "fitted_training_mse": fitted_mse,
        "training_mse_gain": gain,
        "squared_distance": distance,
        "alpha": {
            loss: l2_sp_alpha(ratio, gain, distance) for loss, ratio in LOSSES.items() if ratio is not None
        },
    }


def slice_checks(rows, reference, tolerance):
    """A1: unfitted-slice NMSE at 128 tokens matches SwiGLU-8's slices."""

    recorded = {(row["layer"], row["replacement_width"]): row for row in reference["results"]["slices"]}
    return [
        comparison("A1", row["validation_128"]["nmse"],
                   recorded[(row["layer"], row["replacement_width"])]["validation_128"]["nmse"], tolerance,
                   layer=row["layer"], width=row["replacement_width"], metric="slice_nmse_128")
        for row in rows
    ]


def reproduction_checks(rows, reference, tolerance):
    """A2-A3: the reproduction fit matches SwiGLU-8's c1e-3-128 fit at seed 21."""

    recorded = {
        (row["layer"], row["replacement_width"]): row
        for row in reference["results"]["fits"]
        if (row["setting"], row["seed"]) == (REFERENCE_SETTING, 21)
    }
    checks = []
    for row in rows:
        if row["setting"] != REPRODUCTION_SETTING:
            continue
        expected = recorded[(row["layer"], row["replacement_width"])]
        labels = {"layer": row["layer"], "width": row["replacement_width"]}
        for check, metric in (("A2", "local_nmse"), ("A2", "local_mse"), ("A3", "kl_128"), ("A3", "kl_8192")):
            checks.append(comparison(check, row[metric], expected[metric], tolerance, metric=metric, **labels))
    return checks


def block_check(args, settings, config_path):
    """Run Part A for one layer: the reproduction fit, then the 11 new settings at both widths.

    The layer's pairs are captured once at 3,145,728 pairs; every size is a
    prefix of that capture, and the neuron ranking comes from the 393,216-pair
    prefix, so only the loss and the number of pairs change between settings.
    """

    layer = block_check_layer(args.task_index)
    identity = {"kind": "block-check-task", "task_index": int(args.task_index), "layer": layer}
    work_dir, output_dir = resolve_storage(args.work_dir, args.output_dir, args.resume)
    artifact = start_artifact(output_dir, settings, "block-check", args.resume, identity)
    try:
        reference = swiglu8_block_check(settings, args.swiglu8_root)
        contract = {
            "configuration": settings,
            "configuration_sha256": file_digest(config_path),
            "code_hashes": code_hashes(),
            "swiglu8_block_check_sha256": settings["sources"]["swiglu8_block_check_sha256"],
            "task": identity,
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
        calibration = results.setdefault("l2_sp_calibration", {})
        persist(output_dir, artifact, "loading_models")
        context, model_config = fitting_context(settings, work_dir, output_dir, artifact, "block_check")
        long_data = prepare_long_context_data(context, settings, output_dir, artifact, True,
                                              include_selection=True)
        extension, audit = calibration_extension(context, settings, long_data)
        if results.setdefault("calibration_extension", audit) != audit:
            raise ValueError("Resumed extension windows differ from the recorded ones")
        tolerances = settings["acceptance"]
        enforce(artifact, output_dir, "D1", data_identity_checks(long_data, audit, reference, settings))
        kl_record = long_data["kl_stream"]
        kl_path = verified_stream(output_dir, kl_record)
        selection_record = long_data["selection_stream"]
        selection_path = verified_stream(output_dir, selection_record)
        length = int(settings["long_context"]["sequence_length"])
        teacher, unused_tokenizer = load_model_and_tokenizer(model_config)
        validated_attention_implementation(teacher)
        if "P5" not in results.get("acceptance", {}):
            enforce(artifact, output_dir, "P5", [teacher_self_kl(context, teacher, kl_path, kl_record, settings)])
        selection_cache = cache_teacher_logits(
            context.model,
            context.data["allocation_selection"],
            context.data["partition_batches"]["allocation_selection"],
            context.device,
            "float16",
        )
        path = context.blocks[layer].path
        training, validation = capture_layer_pairs(
            context, layer, [*context.data["calibration_sequences"], *extension], max(CALIBRATION_SIZES)
        )
        del extension
        long_validation = capture_long_validation(context, output_dir, long_data, settings, layer)
        release_memory()
        pairs = {128: {"validation": {path: validation}}, 8192: {"validation": {path: long_validation}}}
        base = calibration_prefix(training, CALIBRATION_SIZES[0])
        ranking = neuron_ranking(context, layer, base.inputs)
        results["ranking_digest"] = content_digest(ranking.tolist())
        enforce(artifact, output_dir, "D2", [
            equality_check("D2", results["ranking_digest"], reference["results"]["ranking_digests"][str(layer)],
                           metric="ranking_digest", layer=layer)
        ])
        widths = [width_from_ratio(settings, ratio) for ratio in settings["part_a"]["width_ratios"]]
        original = int(settings["model"]["intermediate_size"])
        evaluation_inputs = (selection_cache, teacher, selection_path, selection_record, length)

        for width in widths:
            if any(row["replacement_width"] == width for row in results["slices"]):
                continue
            module = slice_module(context, layer, width, selected_indices(ranking, width))
            results["slices"].append(
                {"setting": "slice", "seed": None, "layer": layer, "replacement_width": width,
                 "width_ratio": width / original,
                 **evaluate_block(context, layer, module, pairs, *evaluation_inputs)}
            )
            del module
            release_memory()
            persist(output_dir, artifact, f"block_check:slice:{layer}:{width}")
        enforce(artifact, output_dir, "A1", slice_checks(results["slices"], reference, tolerances["exact_relative"]))

        grid = settings["part_a"]["settings"]
        recipe = settings["part_a"]["recipe"]
        seed = int(settings["part_a"]["seed"])

        def completed(setting_id, width):
            return any((row["setting"], row["replacement_width"]) == (setting_id, width) for row in results["fits"])

        def run_fit(setting_id, width):
            setting = grid[setting_id]
            reproduction = setting_id == REPRODUCTION_SETTING
            alpha = 0.0 if setting["l2_sp_ratio"] is None else calibration[str(width)]["alpha"][setting["loss"]]
            indices = selected_indices(ranking, width)
            # The reproduction fit keeps SwiGLU-8's anchor fit key; the others are named by setting.
            kind = fit_context_kind(REFERENCE_SETTING if reproduction else setting_id, seed)
            fit = fit_operator(
                recipe_context(context, recipe, seed), layer, width, "legacy_subset",
                calibration_prefix(training, setting["calibration_pairs"]), validation, indices,
                context_kind=kind, l2_sp_alpha=alpha,
            )
            if int(fit["calibration_pairs"]) != int(setting["calibration_pairs"]):
                raise RuntimeError("Fit used a different calibration size than its setting")
            module = load_fit_operator(context, fit)
            start = slice_module(context, layer, width, indices)
            distance = squared_distance(module, start)
            if reproduction:
                calibration[str(width)] = l2_sp_calibration(context, start, module, base, distance)
            del start
            row = {
                "setting": setting_id,
                "seed": seed,
                "layer": layer,
                "replacement_width": width,
                "width_ratio": width / original,
                **recipe,
                **setting,
                "l2_sp_alpha": alpha,
                **{
                    key: deepcopy(fit[key])
                    for key in (
                        "fit_key", "best_epoch", "epochs_completed", "updates", "fit_seconds",
                        "initial_local_mse", "initial_local_nmse", "local_mse", "local_nmse",
                        "local_cosine", "state_sha256", "history",
                    )
                },
                "starting_point_squared_distance": distance,
                "displacement_ratio": distance / calibration[str(width)]["squared_distance"],
                **evaluate_block(context, layer, module, pairs, *evaluation_inputs),
            }
            results["fits"].append(row)
            module.to("cpu")
            del module
            release_memory()
            persist(output_dir, artifact, f"block_check:{setting_id}:{layer}:{width}")
            print(
                f"setting={setting_id} layer={layer} width={width} alpha={alpha:.6g} "
                f"kl128={row['kl_128']:.6f} kl8192={row['kl_8192']:.6f}",
                flush=True,
            )

        # The reproduction fits come first: A2-A3 stop the task before any new setting runs.
        for width in widths:
            if not completed(REPRODUCTION_SETTING, width):
                run_fit(REPRODUCTION_SETTING, width)
        enforce(artifact, output_dir, "A2-A3",
                reproduction_checks(results["fits"], reference, tolerances["fit_relative"]))
        for setting_id in NEW_SETTINGS:
            for width in widths:
                if not completed(setting_id, width):
                    run_fit(setting_id, width)
        artifact["status"] = "completed"
        persist(output_dir, artifact, "completed")
    except BaseException as error:
        record_failure(output_dir, work_dir, artifact, error)
        raise


# Part A: merge and adoption


def adoption_decision(rows, reference, settings):
    """Apply SwiGLU-8's adoption rule to the single-seed SwiGLU-9 settings.

    The reference is SwiGLU-8's c1e-3-128 at seeds 21-23, summarized per
    combination and context by its seed mean and sample standard deviation. The
    new settings run one seed, so their spread is taken to equal the reference's,
    and the seed noise is 2 * sd_reference / mean_reference. A setting qualifies
    when its mean 8K reduction exceeds both the minimum and that noise and no
    combination is worse than allowed at either context; the largest mean 8K
    reduction wins, otherwise the current recipe stays.
    """

    rule = settings["part_a"]["adoption"]
    deciding = int(rule["deciding_context"])
    reference_rows = [row for row in reference["results"]["fits"] if row["setting"] == REFERENCE_SETTING]
    combinations = sorted({(int(row["layer"]), int(row["replacement_width"])) for row in reference_rows})
    summaries = {}
    for layer, width in combinations:
        for context in CONTEXTS:
            by_seed = {
                row["seed"]: float(row[f"kl_{context}"])
                for row in reference_rows
                if (int(row["layer"]), int(row["replacement_width"])) == (layer, width)
            }
            if sorted(by_seed) != list(REFERENCE_SEEDS):
                raise ValueError(f"SwiGLU-8 reference lacks seeds at layer {layer}, width {width}")
            summaries[(layer, width, context)] = seed_summary([by_seed[seed] for seed in REFERENCE_SEEDS])
    values = {}
    for row in rows:
        key = (row["setting"], int(row["layer"]), int(row["replacement_width"]))
        if key in values:
            raise ValueError(f"Duplicate Part A fit: {key}")
        values[key] = {context: float(row[f"kl_{context}"]) for context in CONTEXTS}
    table = {}
    for setting in NEW_SETTINGS:
        missing = [combination for combination in combinations if (setting, *combination) not in values]
        if missing:
            raise ValueError(f"Part A lacks {setting} at {missing}")
        reductions = {}
        noise = {}
        for layer, width in combinations:
            for context in CONTEXTS:
                mean, deviation = summaries[(layer, width, context)]
                reductions[(layer, width, context)] = 1.0 - values[(setting, layer, width)][context] / mean
                noise[(layer, width, context)] = 2.0 * deviation / mean
        mean_reduction = {
            context: sum(reductions[(layer, width, context)] for layer, width in combinations) / len(combinations)
            for context in CONTEXTS
        }
        mean_noise = sum(noise[(layer, width, deciding)] for layer, width in combinations) / len(combinations)
        worst = min(reductions.values())
        reasons = []
        if not mean_reduction[deciding] > rule["minimum_mean_reduction"]:
            reasons.append(f"mean {deciding}-token reduction is not above {rule['minimum_mean_reduction']:.0%}")
        if not mean_reduction[deciding] > mean_noise:
            reasons.append("mean reduction is not above the seed noise")
        if worst < -rule["maximum_combination_increase"]:
            reasons.append(f"a combination is more than {rule['maximum_combination_increase']:.0%} worse")
        table[setting] = {
            **settings["part_a"]["settings"][setting],
            "mean_reduction": {str(context): value for context, value in mean_reduction.items()},
            "mean_seed_noise": mean_noise,
            "worst_reduction": worst,
            "reductions": [
                {"layer": layer, "width": width, "context": context, "reduction": value,
                 "seed_noise": noise[(layer, width, context)]}
                for (layer, width, context), value in sorted(reductions.items())
            ],
            "qualifies": not reasons,
            "reasons": reasons,
        }
    qualifying = [setting for setting, row in table.items() if row["qualifies"]]
    selected = (
        max(qualifying, key=lambda setting: table[setting]["mean_reduction"][str(deciding)])
        if qualifying
        else REPRODUCTION_SETTING
    )
    return {
        "rule": deepcopy(rule),
        "reference": {"workflow": "swiglu-8", "setting": REFERENCE_SETTING, "seeds": list(REFERENCE_SEEDS)},
        "reference_seed_summaries": [
            {"layer": layer, "width": width, "context": context, "mean": mean, "seed_sd": deviation}
            for (layer, width, context), (mean, deviation) in sorted(summaries.items())
        ],
        "settings": table,
        "qualifying": qualifying,
        "selected_setting": selected,
    }


def merge_block_check(tasks, settings, task_records, reference):
    """Join the three per-layer tasks into the single Part A result.

    Every task must be completed, must have used the same configuration, code,
    software, 8K streams, and extension windows, and must hold all 24 fits of its
    layer with passed D1, D2, P5, A1, and A2-A3 checks.
    """

    expected = [
        {"kind": "block-check-task", "task_index": index, "layer": layer} for index, layer in enumerate(LAYERS)
    ]
    if [task.get("identity") for task in tasks] != expected:
        raise ValueError("The merge needs exactly one task per Part A layer, in layer order")
    widths = [width_from_ratio(settings, ratio) for ratio in settings["part_a"]["width_ratios"]]
    grid = settings["part_a"]["settings"]
    first = tasks[0]
    environment_keys = ("python", "packages", "cuda_runtime", "gpu")
    for task in tasks:
        layer = task["identity"]["layer"]
        if task.get("workflow") != WORKFLOW or task.get("command") != "block-check" or task.get("status") != "completed":
            raise ValueError(f"Block-check task for layer {layer} is not completed")
        if task.get("configuration") != settings:
            raise ValueError(f"Block-check task for layer {layer} used a different configuration")
        for key in ("configuration_sha256", "code_hashes"):
            if task.get(key) != first.get(key):
                raise ValueError(f"Block-check task for layer {layer} differs from the first task in {key}")
        if any(task["environment"].get(key) != first["environment"].get(key) for key in environment_keys):
            raise ValueError(f"Block-check task for layer {layer} ran in a different software or GPU environment")
        for key in ("long_context_data", "calibration_extension"):
            if task["results"].get(key) != first["results"].get(key):
                raise ValueError(f"Block-check task for layer {layer} saw different {key}")
        fits = task["results"].get("fits", [])
        if (
            sorted((row["setting"], row["replacement_width"]) for row in fits)
            != sorted((setting, width) for setting in grid for width in widths)
            or any(row["layer"] != layer for row in fits)
        ):
            raise ValueError(f"Block-check task for layer {layer} lacks its fits")
        acceptance = task["results"].get("acceptance", {})
        if (
            not {"D1", "D2", "P5", "A1", "A2-A3"} <= set(acceptance)
            or not all(group["passed"] for group in acceptance.values())
            or len(task["results"].get("slices", [])) != len(widths)
            or set(task["results"].get("l2_sp_calibration", {})) != {str(width) for width in widths}
        ):
            raise ValueError(f"Block-check task for layer {layer} lacks passed checks, slices, or calibration")
    fits = [row for task in tasks for row in task["results"]["fits"]]
    adoption = adoption_decision(fits, reference, settings)
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "experiment_family": "swiglu-9",
        "experiment_class": "local-fitting-at-1b-and-calibration-scaling",
        "command": "block-check",
        "identity": {"kind": "block-check", "tasks": len(tasks)},
        "status": "completed",
        "created_at_utc": min(task["created_at_utc"] for task in tasks),
        "configuration": deepcopy(settings),
        "environment": deepcopy(first["environment"]),
        "run_fingerprint": content_digest([task["run_fingerprint"] for task in tasks]),
        "code_hashes": deepcopy(first["code_hashes"]),
        "configuration_sha256": first["configuration_sha256"],
        "results": {
            "long_context_data": deepcopy(first["results"]["long_context_data"]),
            "calibration_extension": deepcopy(first["results"]["calibration_extension"]),
            "ranking_digests": {str(task["identity"]["layer"]): task["results"]["ranking_digest"] for task in tasks},
            "acceptance": {str(task["identity"]["layer"]): deepcopy(task["results"]["acceptance"]) for task in tasks},
            "l2_sp_calibration": {
                str(task["identity"]["layer"]): deepcopy(task["results"]["l2_sp_calibration"]) for task in tasks
            },
            "slices": [deepcopy(row) for task in tasks for row in task["results"]["slices"]],
            "fits": deepcopy(fits),
            "adoption": adoption,
            "stage_two_indicator": {setting: stage_two_indicator(fits, setting) for setting in grid},
            "tasks": [
                {
                    **record,
                    "task_index": task["identity"]["task_index"],
                    "layer": task["identity"]["layer"],
                    "run_fingerprint": task["run_fingerprint"],
                    "created_at_utc": task["created_at_utc"],
                    "updated_at_utc": task["updated_at_utc"],
                    "gpu": task["environment"].get("gpu"),
                }
                for task, record in zip(tasks, task_records)
            ],
        },
        "error": None,
    }


def write_once(output_dir, artifact, label):
    """Write a login-node summary once; an identical re-run changes nothing."""

    existing = output_dir / "result.json"
    if existing.is_file():
        if read_json(existing).get("results") != artifact["results"]:
            raise ValueError(f"A different {label} already exists: {existing}")
        print(f"Already written and identical: {existing}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        persist(output_dir, artifact, label)
        print(f"Wrote {existing}")
    return existing


def finished_outputs(root, names):
    locked = [name for name in names if (root / f"{name}.perun-lock").exists()]
    if locked:
        raise ValueError(f"These outputs did not finish their copy to PROJECT: {', '.join(locked)}")
    missing = [name for name in names if not (root / name / "result.json").is_file()]
    if missing:
        raise FileNotFoundError(f"These outputs have no result yet: {', '.join(missing)}")
    return [root / name / "result.json" for name in names]


def merge_block_check_command(args, settings):
    """Merge the three completed block-check tasks under one results root; safe to re-run."""

    root = resolve_path(args.results_root)
    output_dir = resolve_path(args.output_dir) if args.output_dir else root / "block-check-001"
    names = [block_check_output_name(index) for index in range(len(LAYERS))]
    paths = finished_outputs(root, names)
    merged = merge_block_check(
        [read_json(path) for path in paths],
        settings,
        [{"output": name, "result_sha256": file_digest(path)} for name, path in zip(names, paths)],
        swiglu8_block_check(settings, args.swiglu8_root),
    )
    existing = write_once(output_dir, merged, "merged")
    adoption = merged["results"]["adoption"]
    print(f"selected setting: {adoption['selected_setting']} | qualifying: {adoption['qualifying']}")
    print(f"block-check sha256: {file_digest(existing)}")


# Part B: recovery to 1B tokens


def run_role(arm, target):
    for run in PART_B["runs"]:
        if (run["arm"], float(run["target"])) == (arm, float(target)):
            return run["role"]
    raise ValueError(f"SwiGLU-9 has no {arm} run at target {target}")


def train_output_name(arm, target):
    return f"{arm}-target-{target}-1b-run-001"


def recovery_stream(settings, swiglu7_root, stream_root, prepared):
    """Verify SwiGLU-7's 1B stream and that SwiGLU-8's 100M stream is its prefix (P4)."""

    reference = pinned_json(swiglu7_root, "prepare-001/result.json",
                            settings["sources"]["swiglu7_prepare_sha256"], "SwiGLU-7 preparation")
    record = deepcopy(reference["results"]["stream"])
    recovery = settings["recovery"]
    if (
        int(record["token_count"]) != int(recovery["target_tokens"])
        or int(record["sequence_length"]) != int(recovery["sequence_length"])
    ):
        raise ValueError("SwiGLU-7's recovery stream does not cover the 1B-token recipe")
    path = verified_stream(resolve_path(Path(stream_root)), record)
    prefix = prepared["results"]["stream"]
    check = equality_check("P4", file_digest(path, 4 * int(prefix["token_count"])), prefix["sha256"],
                           metric="swiglu8_stream_is_prefix")
    return record, path, check


def environment_difference(environment, reference):
    """List the recorded software and hardware fields that differ from a reference run."""

    difference = {}
    for key in sorted(set(environment) | set(reference)):
        if key == "packages":
            packages = {
                name: {"value": environment.get(key, {}).get(name), "reference": reference.get(key, {}).get(name)}
                for name in sorted(set(environment.get(key, {})) | set(reference.get(key, {})))
                if environment.get(key, {}).get(name) != reference.get(key, {}).get(name)
            }
            if packages:
                difference[key] = packages
        elif environment.get(key) != reference.get(key):
            difference[key] = {"value": environment.get(key), "reference": reference.get(key)}
    return difference


def check_against_swiglu8(artifact, output_dir, reference_history, settings):
    """Stop on a token-0 mismatch with SwiGLU-8's 100M run (C1); record 10-100M for review (C2)."""

    covered = {tuple(int(value) for value in row["requested_tokens"]) for row in reference_history}
    checks = [
        {**check, "check": "C" + check["check"][1:]}
        for row in artifact["results"]["recovery"]["validation_history"]
        if tuple(int(value) for value in row["requested_tokens"]) in covered
        for check in monitoring_checks(row, reference_history, settings["acceptance"])
    ]
    enforce(artifact, output_dir, "C1", [row for row in checks if row["check"] == "C1"])
    later = [row for row in checks if row["check"] == "C2"]
    if later:
        record_checks(artifact, output_dir, "C2", later)


def train(args, settings, config_path):
    import torch

    work_dir, output_dir = resolve_storage(args.work_dir, args.output_dir, args.resume)
    arm = args.arm
    target = float(args.target)
    role = run_role(arm, target)
    twin = role == "noise_twin"
    identity = {"arm": arm, "target": target, "role": role}
    artifact = start_artifact(output_dir, settings, "train", args.resume, identity)
    try:
        prepared_path, prepared, prepared_root = load_swiglu8_prepared(args.prepared, settings)
        validate_evaluation_preflight(prepared, prepared_root, settings)
        starts = prepared["results"]["starts"]
        start_record = starts[start_key(arm, target)]
        partner = starts[start_key("slice" if arm == "fitted" else "fitted", target)]
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
            raise ValueError("SwiGLU-9 requires SwiGLU-8's fresh token-zero start for this arm")
        start_path = output_path(prepared_root, start_record["path"])
        if file_digest(start_path) != start_record["sha256"]:
            raise ValueError("Prepared starting state changed")
        stream_root = args.recovery_stream_root or resolve_path(Path(args.swiglu7_root)) / "prepare-001"
        stream, stream_path, prefix_check = recovery_stream(settings, args.swiglu7_root, stream_root, prepared)
        kl_record = prepared["results"]["long_context_data"]["kl_stream"]
        verified_stream(prepared_root, kl_record)
        swiglu8_history = swiglu8_run(settings, args.swiglu8_root, arm, target)["results"]["recovery"][
            "validation_history"
        ]
        swiglu7_reference = swiglu7_run(settings, args.swiglu7_root, target) if twin else None
        contract = {
            "configuration": settings,
            "configuration_sha256": file_digest(config_path),
            "prepared_sha256": file_digest(prepared_path),
            "start_sha256": start_record["sha256"],
            "stream_sha256": stream["sha256"],
            "identity": identity,
            "code_hashes": code_hashes(),
            "swiglu8_run_sha256": settings["sources"]["swiglu8_runs_sha256"][start_key(arm, target)],
            "swiglu7_run_sha256": settings["sources"]["swiglu7_runs_sha256"][str(target)] if twin else None,
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
            recovery_stream={"sha256": stream["sha256"], "token_count": int(stream["token_count"]),
                             "source": "swiglu-7 prepare-001"},
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
        if "P4" not in artifact["results"].get("acceptance", {}):
            enforce(artifact, output_dir, "P4", [prefix_check])
        if twin:
            artifact["results"]["environment_difference_from_swiglu7"] = environment_difference(
                artifact["environment"], swiglu7_reference["environment"]
            )
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
            stream_path, int(stream["token_count"]), settings["recovery"]["sequence_length"], stream["sha256"],
        )
        recovery = settings["recovery"]

        def check_reproduction():
            check_against_swiglu8(artifact, output_dir, swiglu8_history, settings)
            if twin:
                check_monitoring(artifact, output_dir,
                                 swiglu7_reference["results"]["recovery"]["validation_history"], settings)

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
        check_reproduction()
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
                if any(point in recovery["monitoring_ppl_tokens"] for point in event.requested_checkpoint_tokens):
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
                check_reproduction()

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
        acceptance = artifact["results"]["acceptance"]
        if not acceptance["C1"]["passed"] or (twin and not acceptance["B1"]["passed"]):
            raise AcceptanceFailure("C1 or B1 did not pass")
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


def reference_kl(args, settings, config_path):
    """Score SwiGLU-7's fitted 1B bundles at 20% and 50% with the shared 8K teacher KL."""

    identity = {"kind": "reference-kl", "targets": list(REUSED_FITTED_TARGETS)}
    work_dir, output_dir = resolve_storage(args.work_dir, args.output_dir, args.resume)
    artifact = start_artifact(output_dir, settings, "reference-kl", args.resume, identity)
    try:
        prepared_path, prepared, prepared_root = load_swiglu8_prepared(args.prepared, settings)
        kl_record = prepared["results"]["long_context_data"]["kl_stream"]
        kl_path = verified_stream(prepared_root, kl_record)
        contract = {
            "configuration": settings,
            "configuration_sha256": file_digest(config_path),
            "code_hashes": code_hashes(),
            "prepared_sha256": file_digest(prepared_path),
            "swiglu7_runs_sha256": settings["sources"]["swiglu7_runs_sha256"],
            "identity": identity,
        }
        run_fingerprint = content_digest(contract)
        if args.resume and artifact.get("run_fingerprint") != run_fingerprint:
            raise ValueError("Reference-KL configuration, sources, or maintained code changed")
        if artifact.get("status") == "completed":
            return
        artifact.update(
            run_fingerprint=run_fingerprint,
            code_hashes=contract["code_hashes"],
            configuration_sha256=contract["configuration_sha256"],
            prepared={"sha256": contract["prepared_sha256"]},
            status="running",
            error=None,
        )
        results = artifact["results"]
        targets = results.setdefault("targets", {})
        persist(output_dir, artifact, "loading_models")
        model_config = make_model_config(settings["model"])
        teacher, unused_tokenizer = load_model_and_tokenizer(model_config)
        validated_attention_implementation(teacher)
        length = int(settings["long_context"]["sequence_length"])
        if "P5" not in results.get("acceptance", {}):
            copy, unused_copy_tokenizer = load_model_and_tokenizer(model_config)
            context = SimpleNamespace(model=copy, device=next(copy.parameters()).device)
            enforce(artifact, output_dir, "P5", [teacher_self_kl(context, teacher, kl_path, kl_record, settings)])
            del copy, unused_copy_tokenizer, context
            release_memory()
        for target in REUSED_FITTED_TARGETS:
            key = str(target)
            if key in targets:
                continue
            run = swiglu7_run(settings, args.swiglu7_root, target)
            name = f"S7-0-target-{target:.1f}-run-001"
            bundle = resolve_path(Path(args.swiglu7_root)) / name / "model"
            validate_bundle(bundle)
            manifest = file_digest(bundle / "bundle.json")
            if manifest != run["results"]["evaluation"]["model"]["manifest_sha256"]:
                raise ValueError(f"SwiGLU-7 bundle differs from its run record: {bundle}")
            model, tokenizer, unused_manifest = load_bundle(bundle)
            started = perf_counter()
            try:
                result = evaluate_teacher_kl(model, teacher, kl_path, kl_record, length)
            finally:
                del model, tokenizer
                release_memory()
            targets[key] = {
                "swiglu7_run": name,
                "swiglu7_run_sha256": settings["sources"]["swiglu7_runs_sha256"][key],
                "manifest_sha256": manifest,
                "teacher_kl_8192": {**result, "seconds": perf_counter() - started},
            }
            persist(output_dir, artifact, f"reference_kl:{key}")
            print(f"SwiGLU-7 S7-0 target={target} 8K KL={result['kl']:.6f}", flush=True)
        artifact["status"] = "completed"
        persist(output_dir, artifact, "completed")
    except BaseException as error:
        record_failure(output_dir, work_dir, artifact, error)
        raise


# Part B: verdict and descriptive summary


def relative_gap(slice_value, fitted_value):
    """g = (slice-only - fitted) / fitted (decision log, 2026-10-17)."""

    return (float(slice_value) - float(fitted_value)) / float(fitted_value)


def part_b_verdict(gap, noise, bar):
    """Classify one target with the decided 5% rule (decision log, 2026-10-17).

    With noise below the bar, g below it closes the question and g at or above
    it means local fitting still matters. With noise at or above the bar, a gap
    within the noise is "cannot_tell" and a larger positive gap still matters. A
    slice-only advantage beyond the noise is not covered by the logged rule.
    """

    if noise < bar:
        return "closed" if gap < bar else "still_matters"
    if abs(gap) <= noise:
        return "cannot_tell"
    if gap > noise:
        return "still_matters"
    return "outside_logged_rule"


def history_by_request(history):
    return {tuple(int(value) for value in row["requested_tokens"]): row for row in history}


def twin_noise(twin_history, reference_history, twin_kl, reference_kl_value, window, interval):
    """Largest relative twin difference over the 25M checkpoints in the window and the final 8K KL.

    SwiGLU-7's run is the reference (denominator), as in the B2 review.
    """

    twin_rows = history_by_request(twin_history)
    reference_rows = history_by_request(reference_history)
    points = []
    for tokens in range(int(window[0]), int(window[1]) + 1, int(interval)):
        key = (tokens,)
        if key not in twin_rows or key not in reference_rows:
            raise ValueError(f"Both fitted-50% runs need a monitoring row at {tokens:,} tokens")
        points.append({
            "metric": "recovery_validation_kl",
            "tokens": tokens,
            "value": float(twin_rows[key]["recovery_validation_kl"]),
            "reference": float(reference_rows[key]["recovery_validation_kl"]),
        })
    points.append({"metric": "teacher_kl_8192", "tokens": None, "value": float(twin_kl),
                   "reference": float(reference_kl_value)})
    for point in points:
        point["relative_difference"] = relative_difference(point["value"], point["reference"])
    largest = max(points, key=lambda point: point["relative_difference"])
    return {"value": largest["relative_difference"], "largest_at": largest, "points": points}


def gap_curve(fitted_history, slice_history):
    """Monitoring gap (slice - fitted) / fitted at every checkpoint both runs share."""

    fitted_rows = history_by_request(fitted_history)
    slice_rows = history_by_request(slice_history)
    if set(fitted_rows) != set(slice_rows):
        raise ValueError("Fitted and slice-only runs have different monitoring checkpoints")
    curve = []
    for key in sorted(fitted_rows, key=lambda request: fitted_rows[request]["actual_tokens"]):
        fitted, sliced = fitted_rows[key], slice_rows[key]
        if int(fitted["actual_tokens"]) != int(sliced["actual_tokens"]):
            raise ValueError(f"Arms reached checkpoint {list(key)} at different token counts")
        curve.append({
            "tokens": int(fitted["actual_tokens"]),
            "requested_tokens": list(key),
            "fitted": float(fitted["recovery_validation_kl"]),
            "slice": float(sliced["recovery_validation_kl"]),
            "gap": relative_gap(sliced["recovery_validation_kl"], fitted["recovery_validation_kl"]),
        })
    return curve


def closing_point(curve, bar):
    """First checkpoint after which the monitoring gap stays below the bar until 1B, or None."""

    point = None
    for row in reversed(curve):
        if row["gap"] >= bar:
            break
        point = row["tokens"]
    return point


def token_equivalent(curve):
    """Recovery tokens the slice-only model needs to reach the fitted KL at each checkpoint.

    The slice-only curve's first crossing of the fitted value is interpolated
    linearly between neighbouring checkpoints; None means it is not reached by 1B.
    """

    rows = []
    for fitted in curve:
        tokens = None
        for index, sliced in enumerate(curve):
            if sliced["slice"] <= fitted["fitted"]:
                if index == 0:
                    tokens = float(sliced["tokens"])
                else:
                    before = curve[index - 1]
                    fraction = (before["slice"] - fitted["fitted"]) / (before["slice"] - sliced["slice"])
                    tokens = before["tokens"] + fraction * (sliced["tokens"] - before["tokens"])
                break
        rows.append({
            "fitted_tokens": fitted["tokens"],
            "fitted_kl": fitted["fitted"],
            "slice_tokens": tokens,
            "ratio": tokens / fitted["tokens"] if tokens is not None and fitted["tokens"] > 0 else None,
        })
    return rows


def evaluation_differences(evaluation, reference):
    """Final perplexity and task-accuracy differences between two runs of the same model."""

    perplexity = {
        key: {
            "value": evaluation["likelihood"][key]["perplexity"],
            "reference": reference["likelihood"][key]["perplexity"],
            "relative_difference": relative_difference(
                evaluation["likelihood"][key]["perplexity"], reference["likelihood"][key]["perplexity"]
            ),
        }
        for key in sorted(reference["likelihood"])
        if key in evaluation["likelihood"]
    }
    tasks = {
        task: {
            "value": evaluation["comparison_to_dense"]["tasks"][task]["student"],
            "reference": row["student"],
            "difference": evaluation["comparison_to_dense"]["tasks"][task]["student"] - row["student"],
        }
        for task, row in sorted(reference["comparison_to_dense"]["tasks"].items())
    }
    return {
        "perplexity": perplexity,
        "task_accuracy": tasks,
        "macro_accuracy_difference": (
            evaluation["comparison_to_dense"]["macro_accuracy"] - reference["comparison_to_dense"]["macro_accuracy"]
        ),
    }


def summarize_part_b(runs, reference, swiglu7_runs, settings):
    """Apply the decided Part B verdict and the descriptive analyses to completed runs.

    ``runs`` maps (arm, target) to SwiGLU-9 train artifacts, ``reference`` is
    the reference-KL artifact, and ``swiglu7_runs`` maps the reused targets to
    SwiGLU-7's S7-0 1B artifacts.
    """

    rule = settings["part_b"]["verdict"]
    bar = float(rule["gap_bar"])
    if set(runs) != set(RUNS):
        raise ValueError("The Part B summary needs all five SwiGLU-9 runs")
    hashes = {reference.get("configuration_sha256")}
    for (arm, target), run in runs.items():
        if (
            run.get("workflow") != WORKFLOW
            or run.get("command") != "train"
            or run.get("status") != "completed"
            or run.get("configuration") != settings
            or run.get("identity") != {"arm": arm, "target": target, "role": run_role(arm, target)}
        ):
            raise ValueError(f"SwiGLU-9 {arm} run at {target} is not a completed run of this configuration")
        acceptance = run["results"]["acceptance"]
        if not acceptance["C1"]["passed"] or ((arm, target) == TWIN and not acceptance["B1"]["passed"]):
            raise ValueError(f"SwiGLU-9 {arm} run at {target} lacks a passed token-0 check")
        hashes.add(run.get("configuration_sha256"))
    if (
        reference.get("workflow") != WORKFLOW
        or reference.get("command") != "reference-kl"
        or reference.get("status") != "completed"
        or reference.get("configuration") != settings
        or len(hashes) != 1
        or len({content_digest(item.get("code_hashes")) for item in [reference, *runs.values()]}) != 1
    ):
        raise ValueError("Part B summary inputs differ in configuration or code")

    def final_kl(artifact):
        return float(artifact["results"]["evaluation"]["teacher_kl_8192"]["kl"])

    def history(artifact):
        return artifact["results"]["recovery"]["validation_history"]

    twin = runs[TWIN]
    reference_kls = {
        float(key): float(value["teacher_kl_8192"]["kl"]) for key, value in reference["results"]["targets"].items()
    }
    noise = twin_noise(
        history(twin), history(swiglu7_runs[TWIN[1]]), final_kl(twin), reference_kls[TWIN[1]],
        rule["noise_window_tokens"], settings["recovery"]["validation_interval_tokens"],
    )
    targets = {}
    for target in TARGETS:
        reused = target in REUSED_FITTED_TARGETS
        fitted_kl = reference_kls[target] if reused else final_kl(runs[("fitted", target)])
        fitted_history = history(swiglu7_runs[target]) if reused else history(runs[("fitted", target)])
        slice_run = runs[("slice", target)]
        gap = relative_gap(final_kl(slice_run), fitted_kl)
        curve = gap_curve(fitted_history, history(slice_run))
        targets[str(target)] = {
            "fitted_source": f"swiglu-7 S7-0-target-{target:.1f}-run-001" if reused else train_output_name("fitted", target),
            "fitted_teacher_kl_8192": fitted_kl,
            "slice_teacher_kl_8192": final_kl(slice_run),
            "gap": gap,
            "verdict": part_b_verdict(gap, noise["value"], bar),
            "gap_curve": curve,
            "closing_point_tokens": closing_point(curve, bar),
            "token_equivalent": token_equivalent(curve),
        }
    review = [
        {"run": train_output_name(arm, target), **check}
        for (arm, target), run in sorted(runs.items())
        for group in ("B2", "C2")
        for check in run["results"]["acceptance"].get(group, {}).get("checks", [])
        if not check["passed"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "experiment_family": "swiglu-9",
        "experiment_class": "local-fitting-at-1b-and-calibration-scaling",
        "command": "summarize-part-b",
        "identity": {"kind": "part-b-summary"},
        "status": "completed",
        "created_at_utc": max(run["updated_at_utc"] for run in [reference, *runs.values()]),
        "configuration": deepcopy(settings),
        "results": {
            "rule": deepcopy(rule),
            "noise": noise,
            "noise_valid": noise["value"] < bar,
            "targets": targets,
            "twin": {
                "gap_with_twin_at_0.5": relative_gap(final_kl(runs[("slice", 0.5)]), final_kl(twin)),
                "final_evaluation_differences": evaluation_differences(
                    twin["results"]["evaluation"], swiglu7_runs[TWIN[1]]["results"]["evaluation"]
                ),
                "environment_difference_from_swiglu7": deepcopy(
                    twin["results"].get("environment_difference_from_swiglu7")
                ),
            },
            "flagged_for_review": review,
            "inputs": {
                train_output_name(arm, target): run["run_fingerprint"] for (arm, target), run in sorted(runs.items())
            },
        },
        "error": None,
    }


def summarize_part_b_command(args, settings):
    """Write the Part B verdict from completed runs under one results root; safe to re-run."""

    root = resolve_path(args.results_root)
    output_dir = resolve_path(args.output_dir) if args.output_dir else root / "part-b-summary-001"
    names = [train_output_name(arm, target) for arm, target in RUNS]
    paths = finished_outputs(root, [*names, "reference-kl-001"])
    runs = {run: read_json(path) for run, path in zip(RUNS, paths)}
    swiglu7_runs = {target: swiglu7_run(settings, args.swiglu7_root, target) for target in REUSED_FITTED_TARGETS}
    summary = summarize_part_b(runs, read_json(paths[-1]), swiglu7_runs, settings)
    write_once(output_dir, summary, "part_b_summary")
    results = summary["results"]
    print(f"twin noise: {results['noise']['value']:.4%} (valid: {results['noise_valid']})")
    for target, row in results["targets"].items():
        print(f"target={target} g={row['gap']:.4%} verdict={row['verdict']} "
              f"closing point={row['closing_point_tokens']}")


# Command line


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("block-check", "train", "reference-kl"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        subparser.add_argument("--work-dir", type=Path, required=True)
        subparser.add_argument("--output-dir", type=Path, required=True)
        subparser.add_argument("--swiglu7-root", type=Path, default=DEFAULT_SWIGLU7_ROOT)
        subparser.add_argument("--swiglu8-root", type=Path, default=DEFAULT_SWIGLU8_ROOT)
        subparser.add_argument("--resume", action="store_true")
    subparsers.choices["block-check"].add_argument("--task-index", type=int, required=True)
    train_parser = subparsers.choices["train"]
    train_parser.add_argument("--prepared", type=Path, required=True)
    train_parser.add_argument("--arm", choices=("fitted", "slice"), required=True)
    train_parser.add_argument("--target", type=float, choices=TARGETS, required=True)
    train_parser.add_argument("--recovery-stream-root", type=Path,
                              help="Directory holding SwiGLU-7's prepared/recovery-tokens.int32")
    subparsers.choices["reference-kl"].add_argument("--prepared", type=Path, required=True)
    merge = subparsers.add_parser("merge-block-check")
    merge.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    merge.add_argument("--results-root", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path)
    merge.add_argument("--swiglu8-root", type=Path, default=DEFAULT_SWIGLU8_ROOT)
    summary = subparsers.add_parser("summarize-part-b")
    summary.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    summary.add_argument("--results-root", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path)
    summary.add_argument("--swiglu7-root", type=Path, default=DEFAULT_SWIGLU7_ROOT)
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
    elif args.command == "merge-block-check":
        merge_block_check_command(args, settings)
    elif args.command == "train":
        train(args, settings, config_path)
    elif args.command == "reference-kl":
        reference_kl(args, settings, config_path)
    else:
        summarize_part_b_command(args, settings)


if __name__ == "__main__":
    main()
