"""Run Heterogeneous-1: operator recipes, cost tables, and hybrid allocation.

``select`` runs one operator's recipe grid at one layer and size, and
``merge-select`` applies SwiGLU-8's adoption rule per operator. ``baseline-table``
prices zero, mean, and SwiGLU width 256 at all 22 layers. Once the researcher pins
the adopted recipes in the configuration, ``cost-table`` prices the other items at
one layer per task and ``recover`` trains each test operator alone for 10M tokens.
``solve`` runs the knapsack over the full menu, ``assemble`` builds the chosen
models at token 0, and ``summarize`` reports ranking swaps and real against
predicted damage. The binding design is in ``docs/agents/decisions.md`` and
``docs/experiments/model/heterogenous/heterogeneous-1.md``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch

from workflows.runs.model.common import resolve_path
from workflows.runs.model.swiglu.swiglu5.fitting import (
    capture_dense_pairs,
    fit_operator,
    history_rows,
    load_fit_operator,
    operator_config,
)
from workflows.runs.model.swiglu.swiglu_7 import (
    code_hashes,
    configure_strategy,
    output_path,
    persist,
    resolve_storage,
    utc_now,
    validated_attention_implementation,
)
from workflows.runs.model.swiglu.swiglu_8 import (
    ADOPTION_RULE,
    PART_A_SETTINGS,
    SWIGLU7_TARGETS,
    TARGETS,
    AcceptanceFailure,
    adoption_decision,
    comparison,
    enforce,
    equality_check,
    evaluate_block,
    fit_context_kind,
    fitting_context,
    is_digest,
    long_context_view,
    neuron_ranking,
    pinned_json,
    prepare_long_context_data,
    recipe_context,
    record_failure,
    release_memory,
    selected_indices,
    single_block_kl_short,
    slice_module,
    verified_stream,
)
from workflows.runs.model.swiglu.swiglu_9 import finished_outputs, write_once

from mlp_replacement.artifacts import content_digest, file_digest, read_json
from mlp_replacement.capture import ActivationPairs
from mlp_replacement.compression.allocation import (
    allocate_discrete_swiglu_widths,
    build_operator_curve,
    build_swiglu_width_curve,
)
from mlp_replacement.compression.continuation import recover_exact_segment, segment_schedule
from mlp_replacement.compression.surgery import temporary_fp32_replacements
from mlp_replacement.compression.teacher_cache import cache_teacher_logits
from mlp_replacement.data import PackedTokenCache
from mlp_replacement.evaluation import evaluate_operator
from mlp_replacement.evaluation.mixed_precision import (
    evaluate_teacher_cache_mixed,
    evaluate_validation_kl_mixed,
)
from mlp_replacement.evaluation.teacher_kl import evaluate_teacher_kl
from mlp_replacement.model import load_model_and_tokenizer
from mlp_replacement.operators import (
    BottleneckMLPReplacement,
    DenseHybridReplacement,
    LinearReplacement,
    MeanReplacement,
    ZeroReplacement,
    fit_operator_fp32_detailed,
    fit_ridge_linear,
)
from mlp_replacement.runlog import environment_record


WORKFLOW = "heterogeneous-1"
SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path("workflows/configs/model/heterogenous/heterogeneous-1.json")
SWIGLU8_CONFIG = Path("workflows/configs/model/swiglu/swiglu-8.json")
DEFAULT_SWIGLU8_ROOT = Path("data/results/workflows/model/swiglu-8")
SHARED_BLOCKS = (
    "seed", "targets", "strategies", "model", "data", "long_context", "calibration",
    "local_fitting", "allocation", "recovery", "acceptance",
)
REFERENCE_SETTING = "c1e-3-128"
REFERENCE_SEED = 21
HYBRIDS = ("linear_swiglu", "linear_plain_mlp")
FITTED_OPERATORS = ("linear", "plain_mlp", *HYBRIDS)
FIXED_OPERATORS = ("zero", "mean")
LINEAR_FIRST = f"linear-first-{REFERENCE_SETTING}"
GRADIENT_SETTINGS = {
    name: setting for name, setting in PART_A_SETTINGS.items() if setting["method"] == "gradient"
}
# Closed forms reuse Part A's one-run method, so the adoption rule treats them alike.
RIDGE_SETTINGS = {
    "ridge-128": {"method": "least_squares", "calibration_context": 128},
    "ridge-8k": {"method": "least_squares", "calibration_context": 8192},
}
FIT_KEYS = (
    "fit_key", "calibration_pairs", "parameter_count", "best_epoch", "epochs_completed", "updates",
    "fit_seconds", "initial_local_mse", "initial_local_nmse", "local_mse", "local_nmse", "local_cosine",
    "history",
)
STREAMS = ("kl_stream", "calibration_stream", "validation_stream", "selection_stream")
ENVIRONMENT_KEYS = ("python", "packages", "cuda_runtime", "gpu")


# Configuration, artifacts, and task maps


def load_settings(path):
    """Load the configuration and check every block it shares with SwiGLU-8."""

    path = resolve_path(Path(path))
    settings = read_json(path)
    if settings.get("schema_version") != SCHEMA_VERSION or settings.get("workflow") != WORKFLOW:
        raise ValueError("Expected a schema-1 Heterogeneous-1 configuration")
    reference = read_json(resolve_path(SWIGLU8_CONFIG))
    for key in SHARED_BLOCKS:
        if settings.get(key) != reference[key]:
            raise ValueError(f"Heterogeneous-1 must reuse SwiGLU-8's {key} contract")
    selection = settings["recipe_selection"]
    if (
        selection["reference_setting"] != REFERENCE_SETTING
        or selection["seeds"][0] != REFERENCE_SEED
        or set(selection["sizes"]) != {"swiglu", *FITTED_OPERATORS}
        or set(settings["cost_tables"]["widths"]) != set(FITTED_OPERATORS)
    ):
        raise ValueError("Heterogeneous-1 fixes its reference setting, seeds, and operators")
    sources = settings["sources"]
    digests = [
        sources["swiglu8_prepare_sha256"],
        sources["swiglu8_block_check_sha256"],
        *sources["swiglu8_fitted_runs_sha256"].values(),
    ]
    if (
        set(sources["swiglu8_fitted_runs_sha256"]) != {str(target) for target in TARGETS}
        or not all(is_digest(value) for value in digests)
    ):
        raise ValueError("Heterogeneous-1 pins its SwiGLU-8 inputs by sha256")
    return settings, path


def start_artifact(settings, command, identity, config_path):
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "experiment_family": WORKFLOW,
        "experiment_class": "operator-recipes-and-hybrid-allocation",
        "command": command,
        "identity": identity,
        "status": "running",
        "created_at_utc": utc_now(),
        "configuration": deepcopy(settings),
        "configuration_sha256": file_digest(config_path),
        "environment": environment_record(),
        "code_hashes": code_hashes(),
        "results": {},
        "error": None,
    }


def select_tasks(settings):
    """List the recipe tasks in array order: operator, then layer, then size."""

    selection = settings["recipe_selection"]
    return [
        {"operator": operator, "layer": int(layer), "width": int(width)}
        for operator, widths in selection["sizes"].items()
        for layer in selection["layers"]
        for width in widths
    ]


def indexed(values, index, label):
    if not 0 <= int(index) < len(values):
        raise ValueError(f"{label} task index must be 0-{len(values) - 1}")
    return values[int(index)]


def select_output_name(index):
    return f"select-001-task-{int(index):02d}"


def cost_output_name(layer):
    return f"cost-table-001-layer-{int(layer):02d}"


def recovery_output_name(layer):
    return f"recovery-001-layer-{int(layer):02d}"


def assemble_output_name(target):
    return f"assemble-001-target-{target}"


def recipe_grid(operator):
    """List one operator's settings: Part A's 13 gradient cells plus its own additions."""

    grid = dict(GRADIENT_SETTINGS)
    if operator == "linear":
        grid.update(RIDGE_SETTINGS)
    if operator in HYBRIDS:
        grid[LINEAR_FIRST] = {**grid[REFERENCE_SETTING], "frozen_linear": True}
    return grid


def grid_runs(grid, seeds):
    """Order a grid by seed, so the reference at seed 21 runs first; closed forms run once."""

    runs = [(name, int(seed)) for seed in seeds for name, value in grid.items() if value["method"] == "gradient"]
    return runs + [(name, None) for name, value in grid.items() if value["method"] != "gradient"]


def item_units(operator, width, hidden):
    """Return one menu item's price in hidden-size units (standard parameter bookkeeping)."""

    units = {
        "zero": 0,
        "mean": 1,
        "linear": hidden,
        "swiglu": 3 * width,
        "plain_mlp": 2 * width,
        "linear_swiglu": hidden + 3 * width,
        "linear_plain_mlp": hidden + 2 * width,
    }
    return units[operator]


def state_parameters(module):
    return sum(value.numel() for value in module.state_dict().values())


def check_parameters(module, operator, width, hidden):
    """Check 4: an item holds exactly its menu price, a whole number of budget units."""

    expected = hidden * item_units(operator, width, hidden)
    actual = state_parameters(module)
    if actual != expected or actual % hidden:
        raise AcceptanceFailure(f"check-4: {operator} width {width} has {actual} values, expected {expected}")
    return actual


# Operators and fitting


def nonlinear_start(context, operator, layer, width, ranking, seed):
    """Build a nonlinear start: top teacher neurons, or a seeded random plain MLP."""

    if operator in ("swiglu", "linear_swiglu"):
        return slice_module(context, layer, width, selected_indices(ranking, width))
    hidden = int(context.settings["model"]["hidden_size"])
    torch.manual_seed(int(seed))
    module = BottleneckMLPReplacement(hidden, width / hidden, activation="silu")
    if module.up_projection.out_features != int(width):
        raise RuntimeError("Plain-MLP width was rounded unexpectedly")
    return module.to(device=context.device, dtype=torch.float32)


def residual_pairs(module, pairs, device, batch_size):
    """Return pairs whose targets are what ``module`` leaves of the teacher output."""

    targets = pairs.targets.to(dtype=torch.float32, copy=True)
    with torch.no_grad():
        for start in range(0, pairs.num_tokens, batch_size):
            inputs = pairs.inputs[start : start + batch_size].to(device=device, dtype=torch.float32)
            targets[start : start + batch_size] -= module(inputs).cpu()
    return ActivationPairs(inputs=pairs.inputs, targets=targets)


def hybrid_start(context, operator, layer, width, ranking, seed, training, ridge):
    """Start a hybrid: its nonlinear start, plus the ridge fit of what that start leaves."""

    hidden = int(context.settings["model"]["hidden_size"])
    nonlinear = nonlinear_start(context, operator, layer, width, ranking, seed)
    residual = residual_pairs(nonlinear, training, context.device,
                              int(context.settings["local_fitting"]["batch_size"]))
    linear = fit_ridge_linear(residual, ridge=ridge, device=context.device, dtype=torch.float64)
    module = DenseHybridReplacement(hidden, nonlinear).to(device=context.device, dtype=torch.float32)
    with torch.no_grad():
        module.linear.weight.copy_(linear.projection.weight)
    return module


def start_module(context, operator, layer, width, ranking, seed, training, ridge):
    if operator in HYBRIDS:
        return hybrid_start(context, operator, layer, width, ranking, seed, training, ridge)
    if operator == "linear":
        torch.manual_seed(int(seed))
        module = LinearReplacement(int(context.settings["model"]["hidden_size"]))
        return module.to(device=context.device, dtype=torch.float32)
    return nonlinear_start(context, operator, layer, width, ranking, seed)


def fixed_module(context, operator, training):
    """Build a parameter-free item; mean is the FP64 mean of the calibration targets."""

    if operator == "zero":
        return ZeroReplacement()
    mean = torch.sum(training.targets, dim=0, dtype=torch.float64) / training.num_tokens
    return MeanReplacement(mean.float()).to(context.device)


def fit_new_operator(context, operator, layer, width, setting_id, setting, seed, ranking, training,
                     validation, ridge):
    """Fit one non-SwiGLU item with one setting and record it like a SwiGLU fit."""

    device = context.device
    batch = int(context.settings["local_fitting"]["batch_size"])
    started = perf_counter()
    if setting["method"] == "least_squares":
        module = fit_ridge_linear(training, ridge=ridge, device=device, dtype=torch.float64).to(device)
        initial = fit = None
    else:
        module = start_module(context, operator, layer, width, ranking, seed, training, ridge)
        initial = evaluate_operator(module, validation, device, batch)
        if setting.get("frozen_linear"):
            module.linear.weight.requires_grad_(False)
        config = operator_config(recipe_context(context, setting, seed))
        fit = fit_operator_fp32_detailed(module, training, validation, config, device)
        module = fit.module
    final = evaluate_operator(module, validation, device, batch)
    key = f"{operator}:{setting_id}:s{seed}:layer-{int(layer):02d}:width-{int(width):05d}"
    return module, {
        "fit_key": key,
        "calibration_pairs": int(training.num_tokens),
        "parameter_count": state_parameters(module),
        "best_epoch": int(fit.best_epoch) if fit else 0,
        "epochs_completed": len(fit.history) if fit else 0,
        "updates": int(fit.updates) if fit else 0,
        "fit_seconds": perf_counter() - started,
        "initial_local_mse": float(initial.mse) if initial else None,
        "initial_local_nmse": float(initial.relative_mse) if initial else None,
        "local_mse": float(final.mse),
        "local_nmse": float(final.relative_mse),
        "local_cosine": float(final.cosine_similarity),
        "history": history_rows(key, "full", fit.history) if fit else [],
    }


def fit_item(task, operator, layer, width, setting_id, setting, seed, ranking, pairs):
    """Fit one item on its setting's own calibration context; SwiGLU keeps its existing path."""

    context = task.context
    path = context.blocks[layer].path
    fitting = pairs[int(setting["calibration_context"])]
    training, validation = fitting["training"][path], fitting["validation"][path]
    if operator == "swiglu":
        row = fit_operator(
            recipe_context(context, setting, seed), layer, width, "legacy_subset", training, validation,
            selected_indices(ranking, width), context_kind=fit_context_kind(setting_id, seed),
        )
        module, record = load_fit_operator(context, row), {key: deepcopy(row[key]) for key in FIT_KEYS}
    else:
        module, record = fit_new_operator(
            context, operator, layer, width, setting_id, setting, seed, ranking, training, validation,
            float(task.settings["recipe_selection"]["ridge"]),
        )
    check_parameters(module, operator, width, int(task.settings["model"]["hidden_size"]))
    return module, record


# Shared task setup


def open_task(settings, work_dir, output_dir, artifact, stage, swiglu8_root):
    """Load models and data, then run the identity and self-KL checks of every GPU task."""

    sources = settings["sources"]
    block = pinned_json(swiglu8_root, "block-check-001/result.json", sources["swiglu8_block_check_sha256"],
                        "SwiGLU-8 block check")
    prepared = pinned_json(swiglu8_root, "prepare-001/result.json", sources["swiglu8_prepare_sha256"],
                           "SwiGLU-8 preparation")
    context, model_config = fitting_context(settings, work_dir, output_dir, artifact, stage)
    long_data = prepare_long_context_data(context, settings, output_dir, artifact, True, include_selection=True)
    recorded = block["results"]["long_context_data"]
    enforce(artifact, output_dir, "data-identity", [
        equality_check("data-identity", long_data["audit"]["local_window_records_digest"],
                       recorded["audit"]["local_window_records_digest"], metric="window_records_digest"),
        *(
            equality_check("data-identity", long_data[name]["sha256"], recorded[name]["sha256"],
                           metric=f"{name}_sha256")
            for name in STREAMS
        ),
    ])
    teacher, unused_tokenizer = load_model_and_tokenizer(model_config)
    validated_attention_implementation(teacher)
    record = long_data["selection_stream"]
    task = SimpleNamespace(
        settings=settings,
        artifact=artifact,
        output_dir=output_dir,
        context=context,
        teacher=teacher,
        block=block,
        prepared=prepared,
        long_data=long_data,
        selection_record=record,
        selection_path=verified_stream(output_dir, record),
        length=int(settings["long_context"]["sequence_length"]),
        selection_cache=cache_teacher_logits(
            context.model,
            context.data["allocation_selection"],
            context.data["partition_batches"]["allocation_selection"],
            context.device,
            "float16",
        ),
    )
    limit = float(settings["acceptance"]["teacher_self_kl_absolute"])
    enforce(artifact, output_dir, "check-6", [
        {"check": "check-6", "metric": metric, "value": value, "tolerance": limit, "passed": abs(value) <= limit}
        for metric, value in inserted_scores(task).items()
    ])
    return task


def inserted_scores(task):
    """Score the model as it stands. Callers insert any replacement first.

    This never calls ``temporary_fp32_replacements``, whose exit restores the
    original MLP and would silently undo a replacement that is already inserted.
    """

    context = task.context
    short = evaluate_teacher_cache_mixed(
        context.model, task.selection_cache, float(task.settings["recovery"]["temperature"]), context.device
    )
    long = evaluate_teacher_kl(context.model, task.teacher, task.selection_path, task.selection_record,
                               task.length, protocol=task.selection_record["protocol"], device=context.device)
    return {"kl_128": short["teacher_kl"], "kl_8192": long["kl"]}


def capture_pairs(task, layers, long_needed):
    """Capture fitting pairs at 128 tokens and, when a setting needs them, at 8K."""

    context = task.context
    training, validation = capture_dense_pairs(context, layers)
    pairs = {128: {"training": training, "validation": validation}}
    if long_needed:
        view = long_context_view(context, task.output_dir, task.long_data, task.settings)
        long_training, long_validation = capture_dense_pairs(view, layers)
        pairs[8192] = {"training": long_training, "validation": long_validation}
    return pairs


def checked_ranking(task, layer, pairs):
    """Rank teacher neurons on the 393,216 pairs and require SwiGLU-8's ranking."""

    path = task.context.blocks[layer].path
    ranking = neuron_ranking(task.context, layer, pairs[128]["training"][path].inputs)
    enforce(task.artifact, task.output_dir, f"ranking-{layer}", [
        equality_check("data-identity", content_digest(ranking.tolist()),
                       task.prepared["results"]["selection"]["ranking_digests"][str(layer)],
                       metric="ranking_digest", layer=layer)
    ])
    return ranking


def scored_fit(task, operator, layer, width, setting_id, setting, seed, ranking, pairs):
    """Fit one item and score it as Part A does, at both contexts."""

    module, fit = fit_item(task, operator, layer, width, setting_id, setting, seed, ranking, pairs)
    row = {
        "operator": operator,
        "setting": setting_id,
        "seed": seed,
        "layer": layer,
        "replacement_width": width,
        **setting,
        **fit,
        **evaluate_block(task.context, layer, module, pairs, task.selection_cache, task.teacher,
                         task.selection_path, task.selection_record, task.length),
    }
    module.to("cpu")
    del module
    release_memory()
    persist(task.output_dir, task.artifact, f"{operator}:{setting_id}:s{seed}:{layer}:{width}")
    print(f"operator={operator} setting={setting_id} seed={seed} layer={layer} width={width} "
          f"kl128={row['kl_128']:.6f} kl8192={row['kl_8192']:.6f}", flush=True)
    return row


# Stage R: recipe selection


def anchor_checks(row, block, tolerance):
    """Check 2: SwiGLU at width 2048 reproduces SwiGLU-8's block check at this layer."""

    reference = next(
        fit for fit in block["results"]["fits"]
        if (fit["setting"], fit["seed"], fit["layer"], fit["replacement_width"])
        == (REFERENCE_SETTING, REFERENCE_SEED, row["layer"], row["replacement_width"])
    )
    return [
        comparison("check-2", row[metric], reference[metric], tolerance, metric=metric, layer=row["layer"])
        for metric in ("local_nmse", "local_mse", "kl_128", "kl_8192")
    ]


def curve_checks(row, prepared, tolerance):
    """Check 3: SwiGLU at width 1024 reproduces its SwiGLU-8 width-curve point and local fit."""

    key = (row["layer"], row["replacement_width"])
    curve = next(
        value for value in prepared["results"]["width_curves"]["rows"]
        if (value["layer"], value["replacement_width"]) == key
    )
    fit = next(
        value for value in prepared["results"]["local_fitting"]
        if (value["layer"], value["replacement_width"]) == key
    )
    labels = {"layer": row["layer"], "width": row["replacement_width"]}
    return [
        comparison("check-3", row["kl_128"], curve["raw_teacher_kl"], tolerance, metric="kl_128", **labels),
        comparison("check-3", row["local_nmse"], fit["local_nmse"], tolerance, metric="local_nmse", **labels),
        comparison("check-3", row["local_mse"], fit["local_mse"], tolerance, metric="local_mse", **labels),
    ]


def zero_linear_check(task, operator, layer, width, ranking, tolerance):
    """Check 5: a hybrid whose linear part is zero reproduces its nonlinear part alone."""

    context = task.context
    nonlinear = nonlinear_start(context, operator, layer, width, ranking, REFERENCE_SEED)
    hybrid = DenseHybridReplacement(int(task.settings["model"]["hidden_size"]), nonlinear)
    hybrid = hybrid.to(device=context.device, dtype=torch.float32)
    torch.nn.init.zeros_(hybrid.linear.weight)
    alone = single_block_kl_short(context, layer, nonlinear, task.selection_cache)["teacher_kl"]
    joined = single_block_kl_short(context, layer, hybrid, task.selection_cache)["teacher_kl"]
    del hybrid, nonlinear
    release_memory()
    return comparison("check-5", joined, alone, tolerance, operator=operator, layer=layer, width=width)


def select(args, settings, config_path):
    """Run one operator's recipe grid at one layer and size, as one array task."""

    item = indexed(select_tasks(settings), args.task_index, "Select")
    operator, layer, width = item["operator"], item["layer"], item["width"]
    identity = {"kind": "select-task", "task_index": int(args.task_index), **item}
    work_dir, output_dir = resolve_storage(args.work_dir, args.output_dir, False)
    artifact = start_artifact(settings, "select", identity, config_path)
    try:
        persist(output_dir, artifact, "loading_models")
        task = open_task(settings, work_dir, output_dir, artifact, "select", args.swiglu8_root)
        tolerances = settings["acceptance"]
        selection = settings["recipe_selection"]
        reference = GRADIENT_SETTINGS[REFERENCE_SETTING]
        pairs = capture_pairs(task, [layer], True)
        ranking = checked_ranking(task, layer, pairs)
        results = artifact["results"]
        results["fits"] = []
        # A short SwiGLU refit stops the task before it spends hours on a broken setup.
        anchor = scored_fit(task, "swiglu", layer, int(selection["anchor_width"]), REFERENCE_SETTING, reference,
                            REFERENCE_SEED, ranking, pairs)
        results["anchor"] = anchor
        enforce(artifact, output_dir, "check-2", anchor_checks(anchor, task.block, tolerances["fit_relative"]))
        if operator in HYBRIDS:
            enforce(artifact, output_dir, "check-5", [
                zero_linear_check(task, operator, layer, width, ranking, tolerances["exact_relative"])
            ])
        grid = recipe_grid(operator)
        for setting_id, seed in grid_runs(grid, selection["seeds"]):
            row = scored_fit(task, operator, layer, width, setting_id, grid[setting_id], seed, ranking, pairs)
            results["fits"].append(row)
            if operator == "swiglu" and (setting_id, seed) == (REFERENCE_SETTING, REFERENCE_SEED):
                enforce(artifact, output_dir, "check-3", curve_checks(row, task.prepared, tolerances["fit_relative"]))
        if operator == "swiglu":
            results["fits"].append(scored_fit(
                task, operator, layer, int(selection["swiglu_reference_width"]), REFERENCE_SETTING, reference,
                REFERENCE_SEED, ranking, pairs,
            ))
        artifact["status"] = "completed"
        persist(output_dir, artifact, "completed")
    except BaseException as error:
        record_failure(output_dir, work_dir, artifact, error)
        raise


def overfitting_indicator(rows, setting):
    """Report whether an adopted setting still overfits at its operator's largest tested size.

    SwiGLU-8's vote, moved from width ratio 0.5 to the largest size tested here: a
    layer counts when most of its seeds reach their best epoch by epoch 3 while
    training MSE keeps falling, and the indicator fires in at least two layers.
    """

    candidates = [row for row in rows if row["setting"] == setting]
    if not candidates or any(not row["history"] for row in candidates):
        return {"setting": setting, "applicable": False, "overfitting_remains": False}
    width = max(row["replacement_width"] for row in candidates)
    votes = {}
    for row in candidates:
        if row["replacement_width"] != width:
            continue
        history = [entry for entry in row["history"] if entry["phase"] == "full"]
        best = max(1, int(row["best_epoch"]))
        votes.setdefault(row["layer"], []).append(
            int(row["best_epoch"]) <= 3
            and len(history) > best
            and history[-1]["train_mse"] < history[best - 1]["train_mse"]
        )
    layers = sorted(layer for layer, flags in votes.items() if 2 * sum(flags) > len(flags))
    return {
        "setting": setting,
        "applicable": True,
        "width": width,
        "definition": "best epoch <= 3 while training MSE still falls, in most seeds of a layer",
        "overfitting_layers": layers,
        "overfitting_remains": len(layers) >= 2,
    }


def epoch_cap_share(rows, grid):
    """Report the share of each constant-rate setting's fits that early stopping never ended.

    Cosine settings always run their whole budget, so a share means nothing for them.
    """

    shares = {}
    for name, setting in grid.items():
        fits = [row for row in rows if row["setting"] == name]
        if setting.get("scheduler") == "constant" and fits:
            shares[name] = sum(row["epochs_completed"] >= setting["max_epochs"] for row in fits) / len(fits)
    return shares


def without_recipes(settings):
    return {key: value for key, value in settings.items() if key != "recipes"}


def merge_select(tasks, settings, task_records):
    """Join the recipe tasks and apply SwiGLU-8's multi-seed adoption rule per operator."""

    expected = [
        {"kind": "select-task", "task_index": index, **item} for index, item in enumerate(select_tasks(settings))
    ]
    if [task.get("identity") for task in tasks] != expected:
        raise ValueError("The merge needs exactly one task per recipe task, in array order")
    first = tasks[0]
    selection = settings["recipe_selection"]
    for task in tasks:
        label = task["identity"]["task_index"]
        if task.get("workflow") != WORKFLOW or task.get("command") != "select" or task.get("status") != "completed":
            raise ValueError(f"Select task {label} is not completed")
        if without_recipes(task["configuration"]) != without_recipes(settings):
            raise ValueError(f"Select task {label} used a different configuration")
        if task.get("code_hashes") != first.get("code_hashes"):
            raise ValueError(f"Select task {label} ran different code")
        if any(task["environment"].get(key) != first["environment"].get(key) for key in ENVIRONMENT_KEYS):
            raise ValueError(f"Select task {label} ran in a different software or GPU environment")
        if task["results"]["long_context_data"] != first["results"]["long_context_data"]:
            raise ValueError(f"Select task {label} saw different data")
        operator = task["identity"]["operator"]
        runs = len(grid_runs(recipe_grid(operator), selection["seeds"])) + (operator == "swiglu")
        if len(task["results"]["fits"]) != runs:
            raise ValueError(f"Select task {label} lacks its fits")
        acceptance = task["results"]["acceptance"]
        required = {"data-identity", "check-6", "check-2"}
        required |= {"check-3"} if operator == "swiglu" else set()
        required |= {"check-5"} if operator in HYBRIDS else set()
        if not required <= set(acceptance) or not all(group["passed"] for group in acceptance.values()):
            raise ValueError(f"Select task {label} lacks passed checks")
    rows = [row for task in tasks for row in task["results"]["fits"]]
    adoption, indicators, epoch_caps = {}, {}, {}
    for operator, widths in selection["sizes"].items():
        grid = recipe_grid(operator)
        operator_rows = [
            row for row in rows if row["operator"] == operator and row["replacement_width"] in widths
        ]
        view = {"block_check": {"settings": grid, "seeds": selection["seeds"], "adoption": ADOPTION_RULE}}
        adoption[operator] = adoption_decision(operator_rows, view)
        indicators[operator] = overfitting_indicator(operator_rows, adoption[operator]["selected_recipe"])
        epoch_caps[operator] = epoch_cap_share(operator_rows, grid)
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "experiment_family": WORKFLOW,
        "experiment_class": "operator-recipes-and-hybrid-allocation",
        "command": "merge-select",
        "identity": {"kind": "select", "tasks": len(tasks)},
        "status": "completed",
        "created_at_utc": min(task["created_at_utc"] for task in tasks),
        "configuration": deepcopy(settings),
        "environment": deepcopy(first["environment"]),
        "code_hashes": deepcopy(first["code_hashes"]),
        "results": {
            "long_context_data": deepcopy(first["results"]["long_context_data"]),
            "anchors": [deepcopy(task["results"]["anchor"]) for task in tasks],
            "fits": deepcopy(rows),
            "adoption": adoption,
            "overfitting_indicator": indicators,
            "epoch_cap_share": epoch_caps,
            "tasks": [
                {**record, "created_at_utc": task["created_at_utc"], "updated_at_utc": task["updated_at_utc"],
                 "gpu": task["environment"].get("gpu"), **task["identity"]}
                for task, record in zip(tasks, task_records)
            ],
        },
        "error": None,
    }


def merge_select_command(args, settings):
    """Merge the completed recipe tasks under one results root; safe to re-run."""

    root = resolve_path(args.results_root)
    names = [select_output_name(index) for index in range(len(select_tasks(settings)))]
    paths = finished_outputs(root, names)
    merged = merge_select(
        [read_json(path) for path in paths], settings,
        [{"output": name, "result_sha256": file_digest(path)} for name, path in zip(names, paths)],
    )
    existing = write_once(root / "select-001", merged, "merged")
    for operator, row in merged["results"]["adoption"].items():
        indicator = merged["results"]["overfitting_indicator"][operator]
        print(f"{operator}: selected {row['selected_recipe']} | qualifying {row['qualifying']} "
              f"| overfitting indicator {indicator.get('overfitting_layers', 'not applicable')}")
    fired = [name for name, row in merged["results"]["overfitting_indicator"].items() if row["overfitting_remains"]]
    if fired:
        print(f"STOP: the overfitting indicator fired for {fired}; plan the follow-up before going on.")
    print(f"sha256 for recipes.select_sha256: {file_digest(existing)}")


def load_selection(root, settings):
    """Load the reviewed selection; every later stage uses exactly the pinned recipes."""

    recipes = settings["recipes"]
    if not is_digest(recipes["select_sha256"]) or not all(recipes["settings"].values()):
        raise ValueError("Pin the reviewed recipes in the configuration before this stage")
    selection = pinned_json(root, "select-001/result.json", recipes["select_sha256"], "Recipe selection")
    adoption = selection["results"]["adoption"]
    if any(adoption[operator]["selected_recipe"] != value for operator, value in recipes["settings"].items()):
        raise ValueError("recipes.settings differs from the adoption outcome")
    return selection


def operator_recipe(settings, operator):
    """Return one operator's recipe; SwiGLU menu items keep c1e-3-128 by decision."""

    setting_id = REFERENCE_SETTING if operator == "swiglu" else settings["recipes"]["settings"][operator]
    seed = None if setting_id in RIDGE_SETTINGS else int(settings["seed"])
    return setting_id, recipe_grid(operator)[setting_id], seed


def needs_long_pairs(settings, operators):
    return any(int(operator_recipe(settings, operator)[1]["calibration_context"]) == 8192 for operator in operators)


# Stages Z, S, C: cost tables


def table_row(task, operator, layer, width, module, fit):
    """Price one item: its single-block KL on the allocation-selection partition."""

    hidden = int(task.settings["model"]["hidden_size"])
    parameters = check_parameters(module, operator, width, hidden)
    kl = single_block_kl_short(task.context, layer, module, task.selection_cache)["teacher_kl"]
    print(f"table operator={operator} layer={layer} width={width} kl128={kl:.6f}", flush=True)
    return {"operator": operator, "layer": layer, "replacement_width": width,
            "replacement_parameters": parameters, "raw_teacher_kl": kl, "fit": fit}


def baseline_table(args, settings, config_path):
    """Price zero, mean, and SwiGLU width 256 at every layer; none depends on recipe selection."""

    work_dir, output_dir = resolve_storage(args.work_dir, args.output_dir, False)
    artifact = start_artifact(settings, "baseline-table", {"kind": "baseline-table"}, config_path)
    try:
        persist(output_dir, artifact, "loading_models")
        task = open_task(settings, work_dir, output_dir, artifact, "baseline_table", args.swiglu8_root)
        rows = artifact["results"].setdefault("table", [])
        layers = [int(value) for value in settings["allocation"]["eligible_layers"]]
        group_size = int(settings["local_fitting"]["capture_group_size"])
        new_width = int(settings["cost_tables"]["swiglu_new_width"])
        for offset in range(0, len(layers), group_size):
            group = layers[offset : offset + group_size]
            pairs = capture_pairs(task, group, False)
            for layer in group:
                ranking = checked_ranking(task, layer, pairs)
                training = pairs[128]["training"][task.context.blocks[layer].path]
                for operator in FIXED_OPERATORS:
                    module = fixed_module(task.context, operator, training)
                    rows.append(table_row(task, operator, layer, 0, module, None))
                module, fit = fit_item(task, "swiglu", layer, new_width, REFERENCE_SETTING,
                                       GRADIENT_SETTINGS[REFERENCE_SETTING], REFERENCE_SEED, ranking, pairs)
                rows.append(table_row(task, "swiglu", layer, new_width, module, fit))
                del module
                release_memory()
                persist(output_dir, artifact, f"baseline_table:{layer}")
            del pairs
            release_memory()
        artifact["status"] = "completed"
        persist(output_dir, artifact, "completed")
    except BaseException as error:
        record_failure(output_dir, work_dir, artifact, error)
        raise


def cost_table(args, settings, config_path):
    """Fit and price every non-SwiGLU menu item at one layer with the adopted recipes."""

    layers = [int(value) for value in settings["allocation"]["eligible_layers"]]
    layer = indexed(layers, args.task_index, "Cost-table")
    work_dir, output_dir = resolve_storage(args.work_dir, args.output_dir, False)
    artifact = start_artifact(settings, "cost-table", {"kind": "cost-table", "layer": layer}, config_path)
    try:
        load_selection(args.results_root, settings)
        persist(output_dir, artifact, "loading_models")
        task = open_task(settings, work_dir, output_dir, artifact, "cost_table", args.swiglu8_root)
        pairs = capture_pairs(task, [layer], needs_long_pairs(settings, FITTED_OPERATORS))
        ranking = checked_ranking(task, layer, pairs)
        rows = artifact["results"].setdefault("table", [])
        for operator, widths in settings["cost_tables"]["widths"].items():
            setting_id, setting, seed = operator_recipe(settings, operator)
            for width in widths:
                module, fit = fit_item(task, operator, layer, int(width), setting_id, setting, seed, ranking, pairs)
                rows.append(table_row(task, operator, layer, int(width), module, fit))
                module.to("cpu")
                del module
                release_memory()
                persist(output_dir, artifact, f"cost_table:{operator}:{width}")
        artifact["status"] = "completed"
        persist(output_dir, artifact, "completed")
    except BaseException as error:
        record_failure(output_dir, work_dir, artifact, error)
        raise


# Stage D: single-block recovery


def select_row(selection, operator, setting_id, seed, layer, width):
    key = (operator, setting_id, seed, layer, width)
    return next(
        row for row in [*selection["results"]["fits"], *selection["results"]["anchors"]]
        if (row["operator"], row["setting"], row["seed"], row["layer"], row["replacement_width"]) == key
    )


def recover_item(task, cache, layer, module):
    """Train one inserted operator alone with S7-0's recipe and score it at each checkpoint."""

    settings = task.settings
    recovery = settings["recovery"]
    plan = settings["single_block_recovery"]
    context = task.context
    tokens = int(plan["tokens"])
    history = []
    with temporary_fp32_replacements(context.model, context.blocks, {layer: module}):
        groups, train_modules, unused_adapters = configure_strategy(
            context.model, [context.blocks[layer].path], "S7-0", settings
        )
        history.append({"tokens": 0, "mean_train_kl": None, **inserted_scores(task)})

        def on_checkpoint(event, unused_optimizer, unused_first_step):
            history.append({"tokens": event.tokens_seen, "mean_train_kl": event.mean_train_kl,
                            **inserted_scores(task)})

        result = recover_exact_segment(
            origin=0,
            end=tokens,
            cursor=0,
            schedule=segment_schedule(0, tokens, plan["checkpoints"], recovery["effective_batch_tokens"]),
            student=context.model,
            teacher=task.teacher,
            parameter_groups=[
                {key: group[key] for key in ("name", "parameters", "learning_rate", "weight_decay")}
                for group in groups
            ],
            train_modules=train_modules,
            batch_at=lambda offset, count: cache.batch(
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
    return {"history": history, "tokens_seen": result.tokens_seen, "optimizer_updates": result.optimizer_updates,
            "training_seconds": result.elapsed_seconds}


def recover(args, settings, config_path):
    """Run one layer's single-block recoveries with the adopted recipes, at seed 21."""

    layers = settings["recipe_selection"]["layers"]
    layer = int(indexed(layers, args.task_index, "Recover"))
    work_dir, output_dir = resolve_storage(args.work_dir, args.output_dir, False)
    artifact = start_artifact(settings, "recover", {"kind": "recover", "layer": layer}, config_path)
    try:
        selection = load_selection(args.results_root, settings)
        persist(output_dir, artifact, "loading_models")
        task = open_task(settings, work_dir, output_dir, artifact, "recover", args.swiglu8_root)
        record = task.prepared["results"]["stream"]
        stream = verified_stream(resolve_path(Path(args.swiglu8_root)) / "prepare-001", record)
        cache = PackedTokenCache(stream, int(record["token_count"]), settings["recovery"]["sequence_length"],
                                 record["sha256"])
        pairs = capture_pairs(task, [layer], needs_long_pairs(settings, FITTED_OPERATORS))
        ranking = checked_ranking(task, layer, pairs)
        rows = artifact["results"].setdefault("runs", [])
        checks = []
        tolerance = settings["acceptance"]["fit_relative"]
        for size, members in settings["single_block_recovery"]["size_groups"].items():
            for operator, width in members:
                setting_id, setting, seed = operator_recipe(settings, operator)
                module, fit = fit_item(task, operator, layer, int(width), setting_id, setting, seed, ranking, pairs)
                run = recover_item(task, cache, layer, module)
                recorded = select_row(selection, operator, setting_id, seed, layer, int(width))
                labels = {"operator": operator, "layer": layer, "width": int(width)}
                checks += [
                    comparison("check-8", run["history"][0][metric], recorded[metric], tolerance,
                               metric=metric, **labels)
                    for metric in ("kl_128", "kl_8192")
                ]
                rows.append({"size": size, "operator": operator, "replacement_width": int(width), "layer": layer,
                             "setting": setting_id, "seed": seed, "fit": fit, **run})
                enforce(artifact, output_dir, "check-8", checks)
                module.to("cpu")
                del module
                release_memory()
                print(f"recovery operator={operator} width={width} layer={layer} "
                      f"kl8192 {run['history'][0]['kl_8192']:.6f} -> {run['history'][-1]['kl_8192']:.6f}", flush=True)
        artifact["status"] = "completed"
        persist(output_dir, artifact, "completed")
    except BaseException as error:
        record_failure(output_dir, work_dir, artifact, error)
        raise


# Stage K: knapsack solves


def build_menu(curve_rows, table_rows, settings, operators):
    """Build each layer's menu. SwiGLU is one family and every other operator is its own."""

    original = int(settings["model"]["intermediate_size"])
    hidden = int(settings["model"]["hidden_size"])
    swiglu_rows = [*curve_rows, *(row for row in table_rows if row["operator"] == "swiglu")]
    menu = {}
    for layer in (int(value) for value in settings["allocation"]["eligible_layers"]):
        evaluations = {
            int(row["replacement_width"]): {"teacher_kl": float(row["raw_teacher_kl"]), "has_output_bias": False}
            for row in swiglu_rows
            if int(row["layer"]) == layer
        }
        points = list(build_swiglu_width_curve(layer, original, evaluations, hidden))
        for operator in operators:
            points.extend(build_operator_curve(layer, operator, original, [
                {"width": row["replacement_width"], "parameters": row["replacement_parameters"],
                 "teacher_kl": row["raw_teacher_kl"]}
                for row in table_rows
                if row["operator"] == operator and int(row["layer"]) == layer
            ]))
        menu[layer] = tuple(points)
    return menu


def solve_record(menu, target, hidden):
    solved = allocate_discrete_swiglu_widths(menu, target, hidden)
    return {
        "rows": list(solved.rows),
        "predicted_teacher_kl": solved.predicted_teacher_kl,
        "realized_mlp_removal": solved.realized_mlp_removal,
        "realized_retained_parameters": solved.realized_retained_parameters,
        "requested_retained_parameters": solved.requested_retained_parameters,
        "boundary_layer": solved.boundary_layer,
        "non_swiglu_layers": [int(row["layer"]) for row in solved.rows if "operator" in row],
    }


def widths_of(rows):
    return [int(row["replacement_width"]) for row in sorted(rows, key=lambda row: int(row["layer"]))]


def allocator_checks(curve_rows, prepared, settings):
    """Check 1: the SwiGLU-only menu reproduces SwiGLU-8's recorded solves at all eight targets.

    At 20-50% SwiGLU-8 kept SwiGLU-7's widths, so the reference is its recorded
    re-solve: the widths it used, with the layers its report lists as changed.
    """

    hidden = int(settings["model"]["hidden_size"])
    menu = build_menu(curve_rows, [], settings, ())
    allocation = prepared["results"]["allocation"]
    report = {float(row["target"]): row for row in allocation["swiglu7_width_resolve_report"]}
    checks = []
    for target in TARGETS:
        solved = solve_record(menu, target, hidden)
        recorded = allocation["targets"][str(target)]
        expected = {int(row["layer"]): int(row["replacement_width"]) for row in recorded["allocation"]}
        predicted = recorded["solver"]["predicted_teacher_kl"]
        if target in SWIGLU7_TARGETS:
            expected.update({int(row["layer"]): int(row["resolved_width"]) for row in report[target]["changed_layers"]})
            predicted = report[target]["resolved_predicted_teacher_kl"]
        checks += [
            equality_check("check-1", widths_of(solved["rows"]), [expected[layer] for layer in sorted(expected)],
                           metric="widths", target=target),
            comparison("check-1", solved["predicted_teacher_kl"], predicted, 1e-12,
                       metric="predicted_teacher_kl", target=target),
        ]
    return checks


def menu_price_checks(menu, hidden):
    """Check 4 again on the assembled menu: every price matches the size table's formula."""

    checks = []
    for layer, points in menu.items():
        for point in points:
            units = 3 * point.original_width if point.retains_dense_module else item_units(
                point.operator, point.replacement_width, hidden
            )
            checks.append({
                "check": "check-4", "layer": layer, "operator": point.operator, "width": point.replacement_width,
                "value": point.replacement_parameters, "reference": units * hidden,
                "passed": point.replacement_parameters == units * hidden
                and point.replacement_parameters % hidden == 0,
            })
    return checks


def restricted(menu, layers):
    """Limit the listed layers to non-SwiGLU items; dense counts as SwiGLU."""

    return {
        layer: tuple(point for point in points if point.operator != "swiglu") if layer in layers else points
        for layer, points in menu.items()
    }


def forced_points(menu, target, hidden, counts):
    """Apply the degradation curve's forcing rule at one target.

    Each layer's extra damage is the predicted total with that layer limited to
    non-SwiGLU items, minus the unrestricted total. Layers rank by that value,
    ties by layer, so layers already non-SwiGLU come first. Each point limits the
    first N layers and lets the knapsack solve the rest.
    """

    base = solve_record(menu, target, hidden)["predicted_teacher_kl"]
    extra = {
        layer: solve_record(restricted(menu, {layer}), target, hidden)["predicted_teacher_kl"] - base
        for layer in sorted(menu)
    }
    order = sorted(extra, key=lambda layer: (extra[layer], layer))
    return {
        "extra_damage": {str(layer): value for layer, value in extra.items()},
        "order": order,
        "points": [
            {"forced_count": int(count), "forced_layers": sorted(order[: int(count)]),
             **solve_record(restricted(menu, set(order[: int(count)])), target, hidden)}
            for count in counts
        ],
    }


def completed_table_rows(root, settings):
    """Read the baseline and cost tables; each must have completed."""

    layers = [int(value) for value in settings["allocation"]["eligible_layers"]]
    names = ["baseline-table-001", *(cost_output_name(layer) for layer in layers)]
    paths = finished_outputs(root, names)
    rows = []
    inputs = {}
    for name, path in zip(names, paths):
        artifact = read_json(path)
        if artifact.get("workflow") != WORKFLOW or artifact.get("status") != "completed":
            raise ValueError(f"Table {name} is not completed")
        rows += artifact["results"]["table"]
        inputs[name] = file_digest(path)
    return rows, inputs


def solve(args, settings, config_path):
    """Run check 1, the SwiGLU-only re-solve with width 256, the full menu, and the forcing rule."""

    work_dir, output_dir = resolve_storage(args.work_dir, args.output_dir, False)
    artifact = start_artifact(settings, "solve", {"kind": "solve"}, config_path)
    try:
        root = resolve_path(args.results_root)
        load_selection(root, settings)
        prepared = pinned_json(args.swiglu8_root, "prepare-001/result.json",
                               settings["sources"]["swiglu8_prepare_sha256"], "SwiGLU-8 preparation")
        curve_rows = prepared["results"]["width_curves"]["rows"]
        enforce(artifact, output_dir, "check-1", allocator_checks(curve_rows, prepared, settings))
        table_rows, inputs = completed_table_rows(root, settings)
        hidden = int(settings["model"]["hidden_size"])
        recorded = build_menu(curve_rows, [], settings, ())
        swiglu_only = build_menu(curve_rows, [row for row in table_rows if row["operator"] == "swiglu"], settings, ())
        full = build_menu(curve_rows, table_rows, settings, (*FIXED_OPERATORS, *FITTED_OPERATORS))
        enforce(artifact, output_dir, "check-4", menu_price_checks(full, hidden))
        plan = settings["allocation_solve"]
        results = artifact["results"]
        results["inputs"] = inputs
        results["targets"] = {}
        for target in [*TARGETS, *plan["optional_targets"]]:
            control = solve_record(swiglu_only, target, hidden)
            without_256 = solve_record(recorded, target, hidden)
            entry = {
                "swiglu_recorded_menu": without_256,
                "swiglu_with_256": {**control, "changed": widths_of(control["rows"]) != widths_of(without_256["rows"])},
                "full_menu": solve_record(full, target, hidden),
            }
            if target in plan["degradation_targets"]:
                entry["forcing"] = forced_points(full, target, hidden, plan["forced_layers"])
            results["targets"][str(target)] = entry
            persist(output_dir, artifact, f"solve:{target}")
            print(f"target={target} swiglu={control['predicted_teacher_kl']:.6f} "
                  f"full={entry['full_menu']['predicted_teacher_kl']:.6f} "
                  f"non_swiglu_layers={entry['full_menu']['non_swiglu_layers']}", flush=True)
        artifact["status"] = "completed"
        persist(output_dir, artifact, "completed")
    except BaseException as error:
        record_failure(output_dir, work_dir, artifact, error)
        raise


# Stage T: assembled models at token 0


def row_item(row):
    return int(row["layer"]), row.get("operator", "swiglu"), int(row["replacement_width"])


def replaced_items(rows):
    return {row_item(row) for row in rows if not row["retains_dense_module"]}


def assemble_points(entry, recorded):
    """List one target's models. SwiGLU-8's own model comes first, as the reproduction check."""

    points = [{"name": "swiglu-8", "rows": recorded["allocation"],
               "predicted_teacher_kl": recorded["solver"]["predicted_teacher_kl"]}]
    if entry["swiglu_with_256"]["changed"]:
        points.append({"name": "swiglu-with-256", **entry["swiglu_with_256"]})
    hybrid = entry["full_menu"]
    if hybrid["non_swiglu_layers"]:
        points.append({"name": "hybrid", **hybrid})
    for point in entry.get("forcing", {}).get("points", []):
        if replaced_items(point["rows"]) != replaced_items(hybrid["rows"]):
            points.append({"name": f"forced-{point['forced_count']}", **point})
    return points


def menu_prices(curve_rows, table_rows):
    """Map every measured item to its recorded KL; boundary widths have no measured price."""

    prices = {
        (int(row["layer"]), "swiglu", int(row["replacement_width"])): float(row["raw_teacher_kl"])
        for row in curve_rows
    }
    prices.update({row_item(row): float(row["raw_teacher_kl"]) for row in table_rows})
    return prices


def fit_needed(task, modules, rows, prices, checks):
    """Fit every item the rows need and no earlier call built, one capture group at a time."""

    settings = task.settings
    context = task.context
    needed = sorted(replaced_items(rows) - set(modules))
    layers = sorted({item[0] for item in needed})
    group_size = int(settings["local_fitting"]["capture_group_size"])
    long_needed = needs_long_pairs(settings, {item[1] for item in needed} - set(FIXED_OPERATORS))
    for offset in range(0, len(layers), group_size):
        group = layers[offset : offset + group_size]
        pairs = capture_pairs(task, group, long_needed)
        for layer in group:
            ranking = checked_ranking(task, layer, pairs)
            for item in (value for value in needed if value[0] == layer):
                unused_layer, operator, width = item
                if operator in FIXED_OPERATORS:
                    modules[item] = fixed_module(context, operator, pairs[128]["training"][context.blocks[layer].path])
                else:
                    setting_id, setting, seed = operator_recipe(settings, operator)
                    modules[item] = fit_item(task, operator, layer, width, setting_id, setting, seed, ranking, pairs)[0]
                if item in prices:
                    kl = single_block_kl_short(context, layer, modules[item], task.selection_cache)["teacher_kl"]
                    checks.append(comparison("check-8", kl, prices[item], settings["acceptance"]["fit_relative"],
                                             layer=layer, operator=operator, width=width))
        del pairs
        release_memory()


def score_point(task, modules, point, validation_cache):
    """Measure one assembled model at token 0 against its predicted sum."""

    context = task.context
    replacements = {item[0]: modules[item] for item in replaced_items(point["rows"])}
    with temporary_fp32_replacements(context.model, context.blocks, replacements):
        monitoring = evaluate_validation_kl_mixed(
            context.model, validation_cache, float(task.settings["recovery"]["temperature"]), context.device
        )
        scores = inserted_scores(task)
    predicted = float(point["predicted_teacher_kl"])
    return {
        "name": point["name"],
        "rows": point["rows"],
        "forced_layers": point.get("forced_layers"),
        "predicted_teacher_kl": predicted,
        "monitoring_kl": monitoring,
        "joint_over_predicted": monitoring / predicted,
        **scores,
    }


def assemble(args, settings, config_path):
    """Build one target's models at token 0 and compare real with predicted damage.

    Every item is refitted rather than loaded, so no fitted state is stored; each
    refit is also a reproduction check. SwiGLU-8's own model runs first and stops
    the task before anything new is spent if it misses its recorded token-0 KL.
    """

    target = indexed(list(TARGETS), args.task_index, "Assemble")
    work_dir, output_dir = resolve_storage(args.work_dir, args.output_dir, False)
    artifact = start_artifact(settings, "assemble", {"kind": "assemble", "target": target}, config_path)
    try:
        root = resolve_path(args.results_root)
        load_selection(root, settings)
        solved = read_json(root / "solve-001" / "result.json")
        if solved.get("status") != "completed":
            raise ValueError("The knapsack solve is not completed")
        persist(output_dir, artifact, "loading_models")
        task = open_task(settings, work_dir, output_dir, artifact, "assemble", args.swiglu8_root)
        prepared = task.prepared
        fitted_run = pinned_json(args.swiglu8_root, f"fitted-target-{target}-run-001/result.json",
                                 settings["sources"]["swiglu8_fitted_runs_sha256"][str(target)],
                                 f"SwiGLU-8 fitted run at {target}")
        monitoring_record = prepared["results"]["monitoring_evaluation"]
        monitoring_path = output_path(resolve_path(Path(args.swiglu8_root)) / "prepare-001", monitoring_record["path"])
        if file_digest(monitoring_path) != monitoring_record["sha256"]:
            raise ValueError("SwiGLU-8's monitoring batches changed")
        monitoring = torch.load(monitoring_path, map_location="cpu", weights_only=False)
        validation_cache = cache_teacher_logits(
            task.teacher, monitoring["recovery_validation"], len(monitoring["recovery_validation"]),
            task.context.device, "float16",
        )
        table_rows, unused_inputs = completed_table_rows(root, settings)
        prices = menu_prices(prepared["results"]["width_curves"]["rows"], table_rows)
        points = assemble_points(solved["results"]["targets"][str(target)],
                                 prepared["results"]["allocation"]["targets"][str(target)])
        modules = {}
        checks = []
        tolerances = settings["acceptance"]
        scored = artifact["results"].setdefault("points", [])
        for point in points:
            fit_needed(task, modules, point["rows"], prices, checks)
            enforce(artifact, output_dir, "check-8", checks)
            scored.append(score_point(task, modules, point, validation_cache))
            persist(output_dir, artifact, f"assemble:{point['name']}")
            print(f"target={target} point={point['name']} predicted={scored[-1]['predicted_teacher_kl']:.6f} "
                  f"monitoring={scored[-1]['monitoring_kl']:.6f}", flush=True)
            if len(scored) == 1:
                enforce(artifact, output_dir, "check-7", [comparison(
                    "check-7", scored[0]["monitoring_kl"],
                    fitted_run["results"]["recovery"]["validation_history"][0]["recovery_validation_kl"],
                    tolerances["exact_relative"], target=target,
                )])
        artifact["status"] = "completed"
        persist(output_dir, artifact, "completed")
    except BaseException as error:
        record_failure(output_dir, work_dir, artifact, error)
        raise


# Report


def ranking_swap(before, after, margin):
    """Apply the swap rule to one layer's equal-size operators.

    ``before`` and ``after`` map operators to their KL at token 0 and at the end of
    recovery. A swap counts when the leader changes and the new leader's final KL
    is more than ``margin`` below the old leader's.
    """

    old = min(before, key=before.get)
    new = min(after, key=after.get)
    lead = 1.0 - after[new] / after[old]
    return {"leader_before": old, "leader_after": new, "lead": lead, "swap": old != new and lead > margin}


def swap_table(runs, margin, metric):
    """Rank each layer's equal-size operators before and after recovery on one metric."""

    groups = {}
    for run in runs:
        label = f"{run['operator']}-{run['replacement_width']}"
        before, after = run["history"][0][metric], run["history"][-1][metric]
        group = groups.setdefault((run["layer"], run["size"]), ({}, {}))
        group[0][label], group[1][label] = before, after
    return [
        {"layer": layer, "size": size, "before": before, "after": after, **ranking_swap(before, after, margin)}
        for (layer, size), (before, after) in sorted(groups.items())
    ]


def summarize(args, settings):
    """Report ranking swaps, picked operators, and real against predicted damage; safe to re-run."""

    root = resolve_path(args.results_root)
    layers = [int(value) for value in settings["recipe_selection"]["layers"]]
    names = [recovery_output_name(layer) for layer in layers]
    names += [assemble_output_name(target) for target in TARGETS]
    paths = finished_outputs(root, [*names, "solve-001"])
    artifacts = [read_json(path) for path in paths]
    if any(artifact.get("status") != "completed" for artifact in artifacts):
        raise ValueError("Every recovery, assembly, and solve task must have completed")
    *inputs, solved = artifacts
    runs = [run for artifact in inputs[: len(layers)] for run in artifact["results"]["runs"]]
    margin = float(settings["single_block_recovery"]["swap_margin"])
    swaps = swap_table(runs, margin, "kl_8192")
    picked = {
        target: sorted({row["operator"] for row in entry["full_menu"]["rows"] if "operator" in row})
        for target, entry in solved["results"]["targets"].items()
    }
    any_swap = any(row["swap"] for row in swaps)
    results = {
        "swaps_8192": swaps,
        "swaps_128_reported": swap_table(runs, margin, "kl_128"),
        "picked_operators": picked,
        "hybrid_targets": [str(target) for target in TARGETS if picked[str(target)]],
        "swiglu_changed_by_256": [
            target for target, entry in solved["results"]["targets"].items() if entry["swiglu_with_256"]["changed"]
        ],
        "assembled": {
            str(artifact["identity"]["target"]): [
                {key: point[key] for key in ("name", "predicted_teacher_kl", "monitoring_kl", "joint_over_predicted",
                                             "kl_128", "kl_8192")}
                for point in artifact["results"]["points"]
            ]
            for artifact in inputs[len(layers):]
        },
        "forced_runs_required": any_swap or not any(picked[str(target)] for target in TARGETS),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "experiment_family": WORKFLOW,
        "experiment_class": "operator-recipes-and-hybrid-allocation",
        "command": "summarize",
        "identity": {"kind": "summary"},
        "status": "completed",
        "created_at_utc": max(artifact["updated_at_utc"] for artifact in artifacts),
        "configuration": deepcopy(settings),
        "results": results,
        "error": None,
    }
    write_once(root / "summary-001", summary, "summary")
    print(f"hybrid targets: {results['hybrid_targets']}")
    print(f"ranking swaps: {[(row['layer'], row['size']) for row in swaps if row['swap']]}")
    print(f"forced runs required in Heterogeneous-2: {results['forced_runs_required']}")


# Command line


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("select", "baseline-table", "cost-table", "recover", "solve", "assemble"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        subparser.add_argument("--work-dir", type=Path, required=True)
        subparser.add_argument("--output-dir", type=Path, required=True)
        subparser.add_argument("--swiglu8-root", type=Path, default=DEFAULT_SWIGLU8_ROOT)
    for name in ("select", "cost-table", "recover", "assemble"):
        subparsers.choices[name].add_argument("--task-index", type=int, required=True)
    for name in ("cost-table", "recover", "solve", "assemble"):
        subparsers.choices[name].add_argument("--results-root", type=Path, required=True)
    for name in ("merge-select", "summarize"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        subparser.add_argument("--results-root", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    settings, config_path = load_settings(args.config)
    commands = {
        "select": select,
        "baseline-table": baseline_table,
        "cost-table": cost_table,
        "recover": recover,
        "solve": solve,
        "assemble": assemble,
    }
    if args.command == "merge-select":
        merge_select_command(args, settings)
    elif args.command == "summarize":
        summarize(args, settings)
    else:
        commands[args.command](args, settings, config_path)


if __name__ == "__main__":
    main()
