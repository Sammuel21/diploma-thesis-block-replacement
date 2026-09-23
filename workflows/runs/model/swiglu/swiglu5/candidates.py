"""SwiGLU-5 candidates; extracted with the established protocol unchanged."""

from __future__ import annotations

import gc
from copy import deepcopy
from dataclasses import asdict

import torch

from workflows.runs.model.common import relative_to_root, release_cuda

from mlp_replacement.artifacts import sha256_file
from mlp_replacement.capture import ActivationPairs, collect_modules_io
from mlp_replacement.compression.reconstruction import build_swiglu_student, load_operator
from mlp_replacement.compression.surgery import replace_submodule, temporary_fp32_replacements
from mlp_replacement.data import make_token_loader
from mlp_replacement.evaluation.mixed_precision import (
    autocast_context,
    evaluate_lm_mixed,
    evaluate_teacher_cache_mixed,
    evaluate_validation_kl_mixed,
)
from mlp_replacement.model import discover_mlp_blocks, load_model_and_tokenizer
from mlp_replacement.operators import swiglu_neuron_importance_scores

from ..shared import resolve_source_asset, source_operator_rows
from .context import CANDIDATE_DEFINITIONS
from .fitting import (
    capture_dense_pairs,
    discrete_allocation,
    ensure_fit_for_width,
    fit_key,
    fit_operator,
    selected_neurons,
)


def candidate_parameter_summary(context, allocation_rows):
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


def legacy_candidate(context, target):
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
        "candidate_name": CANDIDATE_DEFINITIONS["S5-C0"]["name"],
        "target": float(target),
        "initialization": "exact_swiglu_3",
        "allocation_method": "exact_swiglu_3_ranked_widths",
        "allocation": rows,
        **candidate_parameter_summary(context, rows),
    }


def candidate_modules(context, candidate, device):
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


def evaluate_candidate(context, candidate, selection_cache, validation_cache):
    modules = candidate_modules(context, candidate, context.device)
    try:
        with temporary_fp32_replacements(context.model, context.blocks, modules):
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


def build_c1_c3_candidates(context, selection_cache, validation_cache):
    results = context.artifact["results"]["candidates"]
    layers = tuple(int(value) for value in context.settings["compatibility"]["eligible_layers"])
    group_size = int(context.settings["local_fitting"]["capture_group_size"])
    targets = tuple(float(value) for value in context.settings["compatibility"]["target_mlp_removals"])
    allocations = {
        (initialization, target): discrete_allocation(context, initialization, target)
        for initialization in ("legacy_subset", "output_aware")
        for target in targets
    }
    descriptors = {}
    for target in targets:
        key = str(target)
        descriptors[(key, "S5-C0")] = legacy_candidate(context, target)
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
                "candidate_name": CANDIDATE_DEFINITIONS[candidate_id]["name"],
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
        training_by_path, validation_by_path = capture_dense_pairs(context, group)
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
                fit_row = ensure_fit_for_width(
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
            descriptor.update(candidate_parameter_summary(context, descriptor["allocation"]))
        target_results = results.setdefault(key, {})
        if candidate_id not in target_results:
            descriptor["pre_recovery"] = evaluate_candidate(
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


def dense_targets(module, inputs, batch_size, device):
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


def build_composition_candidates(context, selection_cache, validation_cache):
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
        parent_modules = candidate_modules(context, parent, context.device)
        composition_rows = []
        try:
            with temporary_fp32_replacements(context.model, context.blocks, parent_modules):
                for offset in range(0, len(layers), group_size):
                    group = layers[offset : offset + group_size]
                    paths = [context.blocks[layer].path for layer in group]
                    with autocast_context(context.device):
                        captured_training = collect_modules_io(
                            context.model,
                            paths,
                            training_loader,
                            pair_batches,
                            context.device,
                            storage_device="cpu",
                            storage_dtype=context.model_dtype,
                        )
                    with autocast_context(context.device):
                        captured_validation = collect_modules_io(
                            context.model,
                            paths,
                            context.data["operator_validation"],
                            int(
                                context.data["partition_batches"][
                                    "operator_validation"
                                ]
                            ),
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
                            dense_targets(
                                dense,
                                train_inputs,
                                int(context.settings["local_fitting"]["batch_size"]),
                                context.device,
                            ),
                        )
                        validation_pairs = ActivationPairs(
                            valid_inputs,
                            dense_targets(
                                dense,
                                valid_inputs,
                                int(context.settings["local_fitting"]["batch_size"]),
                                context.device,
                            ),
                        )
                        selected = selected_neurons(context, layer, training_pairs, width)
                        fit_row = fit_operator(
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
            "candidate_name": CANDIDATE_DEFINITIONS["S5-C4"]["name"],
            "target": float(key),
            "initialization": "composition_aware",
            "allocation_method": "best_pre_recovery_parent_widths",
            "parent_candidate_id": parent["candidate_id"],
            "capture_context_frozen": True,
            "recapture_rounds": 1,
            "allocation": sorted(composition_rows, key=lambda row: int(row["layer"])),
        }
        candidate.update(candidate_parameter_summary(context, candidate["allocation"]))
        candidate["pre_recovery"] = evaluate_candidate(
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


def load_candidate_student(context, candidate, model_config):
    student, tokenizer = load_model_and_tokenizer(model_config)
    blocks = {block.index: block for block in discover_mlp_blocks(student)}
    modules = candidate_modules(context, candidate, next(student.parameters()).device)
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


def blank_candidate_student(context, model_config, candidate):
    """Adapt the historical candidate record to reusable reconstruction."""

    return build_swiglu_student(
        model_config, context.settings["model"]["hidden_size"], candidate["allocation"]
    )
