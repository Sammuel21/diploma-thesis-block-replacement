"""SwiGLU-5 search; extracted with the established protocol unchanged."""

from __future__ import annotations

import gc
import shutil
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import torch

from workflows.runs.model.common import relative_to_root, release_cuda, resolve_path

from mlp_replacement.artifacts import sha256_file
from mlp_replacement.compression.recovery import (
    next_optimizer_boundary,
)
from mlp_replacement.compression.teacher_cache import (
    build_teacher_final_hidden_cache,
    cache_teacher_logits,
    validate_teacher_final_hidden_cache,
)
from mlp_replacement.evaluation.mixed_precision import (
    evaluate_lm_mixed,
    evaluate_teacher_cache_mixed,
    evaluate_validation_kl_mixed,
)
from mlp_replacement.model import resolve_dtype

from .candidates import build_c1_c3_candidates, build_composition_candidates
from .context import (
    CANDIDATE_DEFINITIONS,
    import_swiglu3_evidence,
    load_search_resources,
    packed_source_cache,
    record_stage_runtime,
    storage_preflight,
    utc_now,
)
from .fitting import build_width_curves, fit_key, operator_state_path
from .recovery import (
    calibrate_recovery,
    clone_teacher_head,
    find_full_evaluation,
    recover_candidate,
)


def prune_search_candidate_checkpoints(context, candidate):
    """Prune non-selected recovery state while preserving its manifest."""

    recovery = candidate["recovery"]
    changed = False
    asset_root = context.asset_dir.resolve()
    for row in recovery.get("durable_checkpoints", []):
        if not bool(row.get("retained", True)):
            continue
        path = resolve_path(Path(row["path"])).resolve()
        try:
            path.relative_to(asset_root)
        except ValueError as error:
            raise ValueError(
                f"Refusing to prune checkpoint outside the run assets: {path}"
            ) from error
        if path.is_file():
            path.unlink()
        row["retained"] = False
        row["pruned_after_selection"] = True
        changed = True
    current_path = (
        context.asset_dir
        / "recovery"
        / str(float(candidate["target"]))
        / candidate["candidate_id"]
        / "current.pt"
    )
    if current_path.is_file():
        current_path.resolve().relative_to(asset_root)
        current_path.unlink()
        changed = True
    if changed:
        context.persist("checkpoint_pruning")


def retain_search_winner_endpoint(context, candidate):
    recovery = candidate["recovery"]
    requested = int(context.settings["recovery"]["finalist_tokens"])
    actual = int(recovery["tokens_seen"])
    expected = next_optimizer_boundary(
        requested,
        int(context.settings["recovery"]["effective_batch_tokens"]),
    )
    if actual != expected:
        raise ValueError("Search winner is not at the exact 5M endpoint")
    recovery_dir = (
        context.asset_dir
        / "recovery"
        / str(float(candidate["target"]))
        / candidate["candidate_id"]
    )
    current_path = recovery_dir / "current.pt"
    endpoint_path = recovery_dir / f"endpoint-{actual:012d}.pt"
    if not current_path.is_file():
        raise FileNotFoundError(f"Search winner checkpoint is missing: {current_path}")
    if endpoint_path.exists():
        raise FileExistsError(f"Search winner endpoint already exists: {endpoint_path}")
    current_path.replace(endpoint_path)
    record = {
        "requested_tokens": [requested],
        "actual_tokens": actual,
        "path": relative_to_root(endpoint_path),
        "sha256": sha256_file(endpoint_path),
        "retained": True,
    }
    recovery["durable_checkpoints"] = [record]
    return record


def pending_candidate_fit_keys(context):
    needed = set()
    for candidates in context.artifact["results"]["candidates"].values():
        for candidate in candidates.values():
            if int(candidate["recovery"].get("tokens_seen", 0)) > 0:
                continue
            needed.update(
                row["fit_key"]
                for row in candidate["allocation"]
                if row.get("fit_key") is not None
            )
    return needed


def prune_search_local_fit_states(context, retain_fit_keys=(), stage="search_complete"):
    """Delete fitted tensors once no untrained candidate can consume them."""

    retain_fit_keys = set(retain_fit_keys)
    asset_root = context.asset_dir.resolve()
    removed = 0
    removed_bytes = 0
    for row in context.artifact["results"]["local_fitting"]:
        if not bool(row.get("state_retained", True)):
            continue
        if row["fit_key"] in retain_fit_keys:
            continue
        path = operator_state_path(
            context,
            row["initialization"],
            row["layer"],
            row["replacement_width"],
            row["capture_context"],
        ).resolve()
        try:
            path.relative_to(asset_root)
        except ValueError as error:
            raise ValueError(
                f"Refusing to prune local fit outside the run assets: {path}"
            ) from error
        if path.is_file():
            removed_bytes += path.stat().st_size
            path.unlink()
            removed += 1
        row["state_retained"] = False
        row["pruned_at_stage"] = stage
    for rows in context.artifact["results"]["width_curves"].values():
        for row in rows:
            if (
                row.get("state_path") is not None
                and row.get("fit_key") not in retain_fit_keys
            ):
                row["state_retained"] = False
    return {
        "stage": stage,
        "state_files_removed": removed,
        "bytes_removed": removed_bytes,
        "retained_fit_states": len(retain_fit_keys),
        "retained_evidence": "fit histories, metrics, hashes, and candidate recipes",
    }


def prune_teacher_hidden_cache(context, hidden_cache):
    """Remove the search-only teacher cache after its last consumer."""

    hidden_cache.release()
    gc.collect()
    record = context.artifact["results"].get("teacher_hidden_cache", {})
    root_value = record.get("root")
    if not root_value:
        return {"removed": False, "bytes_removed": 0}
    root = resolve_path(Path(root_value)).resolve()
    asset_root = context.asset_dir.resolve()
    try:
        root.relative_to(asset_root)
    except ValueError as error:
        raise ValueError(f"Refusing to prune cache outside run assets: {root}") from error
    removed_bytes = 0
    if root.is_dir():
        removed_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        shutil.rmtree(root)
    record["retained"] = False
    record["pruned_after_search"] = True
    record["bytes_removed"] = removed_bytes
    return {"removed": True, "bytes_removed": removed_bytes}


def runtime_guard(context, observed_tokens_per_second, allow_over_budget=False):
    completed_seconds = sum(
        float(row["seconds"]) for row in context.artifact["results"]["runtime"]
    )
    candidate_tokens = 38_000_000
    raw_projection = completed_seconds + candidate_tokens / float(observed_tokens_per_second)
    projected = raw_projection * (
        1.0 + float(context.settings["runtime_guard"]["reserve_fraction"])
    )
    within_limit = projected <= int(
        context.settings["runtime_guard"]["maximum_projected_seconds"]
    )
    record = {
        "completed_preparation_seconds": completed_seconds,
        "candidate_token_positions": candidate_tokens,
        "observed_tokens_per_second": float(observed_tokens_per_second),
        "raw_projected_seconds": raw_projection,
        "reserve_fraction": float(context.settings["runtime_guard"]["reserve_fraction"]),
        "projected_seconds": projected,
        "maximum_projected_seconds": int(context.settings["runtime_guard"]["maximum_projected_seconds"]),
        "within_configured_limit": within_limit,
        "over_budget_allowed": bool(allow_over_budget),
        "override_used": bool(allow_over_budget and not within_limit),
        "passed": bool(within_limit or allow_over_budget),
    }
    context.artifact["results"]["runtime_guard"] = record
    context.persist("runtime_guard")
    return record


def rank_qualifier_challengers(context, target_key, candidate_ids):
    candidates = context.artifact["results"]["candidates"][target_key]
    qualifier_requested = int(context.settings["recovery"]["qualifier_tokens"])
    challengers = []
    for candidate_id in candidate_ids:
        trajectory = candidates[candidate_id]["recovery"]
        endpoint = next(
            row
            for row in trajectory["validation_history"]
            if qualifier_requested in [int(value) for value in row["requested_tokens"]]
        )
        challengers.append((float(endpoint["recovery_validation_kl"]), candidate_id))
    return [
        candidate_id
        for unused_validation_kl, candidate_id in sorted(challengers)
    ]


def select_finalists(context, target_key):
    count = int(context.settings["selection"]["challenger_finalists"])
    selected = rank_qualifier_challengers(
        context,
        target_key,
        ("S5-C1", "S5-C2", "S5-C3", "S5-C4"),
    )[:count]
    return ["S5-C0", *selected]


def select_winner(context, target_key, finalists):
    candidates = context.artifact["results"]["candidates"][target_key]
    requested = int(context.settings["recovery"]["finalist_tokens"])
    control_eval = find_full_evaluation(candidates["S5-C0"]["recovery"], requested)
    control_ppl = float(control_eval["wikitext_validation"]["perplexity"])
    eligible = []
    decisions = []
    for candidate_id in finalists:
        evaluation = find_full_evaluation(candidates[candidate_id]["recovery"], requested)
        ppl = float(evaluation["wikitext_validation"]["perplexity"])
        passes = candidate_id == "S5-C0" or ppl <= control_ppl
        row = {
            "candidate_id": candidate_id,
            "candidate_name": CANDIDATE_DEFINITIONS[candidate_id]["name"],
            "recovery_validation_kl": float(evaluation["recovery_validation_kl"]),
            "wikitext_validation_perplexity": ppl,
            "control_perplexity": control_ppl,
            "passes_ppl_guardrail": passes,
        }
        decisions.append(row)
        if passes:
            eligible.append(row)
    pool = eligible
    minimum_kl = min(row["recovery_validation_kl"] for row in pool)
    tolerance = float(context.settings["selection"]["kl_tie_tolerance"])
    tied = [row for row in pool if row["recovery_validation_kl"] <= minimum_kl + tolerance]
    winner = min(
        tied,
        key=lambda row: (
            row["wikitext_validation_perplexity"], row["candidate_id"]
        ),
    )
    winner_candidate = candidates[winner["candidate_id"]]
    source_history = context.artifact["results"]["published_swiglu_3"][target_key]["validation_history"]
    source_nearest = min(
        source_history,
        key=lambda row: abs(
            int(row["tokens_seen"])
            - int(
                next_optimizer_boundary(
                    requested,
                    int(context.settings["recovery"]["effective_batch_tokens"]),
                )
            )
        ),
    )
    dense = context.artifact["results"]["dense_baseline"]
    control = next(row for row in decisions if row["candidate_id"] == "S5-C0")
    winner_eval = find_full_evaluation(winner_candidate["recovery"], requested)
    return {
        "finalists": finalists,
        "guardrail_decisions": decisions,
        "winner_candidate_id": winner["candidate_id"],
        "winner_candidate_name": CANDIDATE_DEFINITIONS[winner["candidate_id"]]["name"],
        "selection_rule": (
            "challenger PPL no worse than S5-C0; lowest fixed T=1 KL; "
            "KL differences <=1e-6 tie-break by PPL then candidate ID; "
            "fall back to S5-C0 when no challenger passes"
        ),
        "gaps": {
            "to_dense": {
                "recovery_validation_kl": float(winner_eval["recovery_validation_kl"]),
                "wikitext_perplexity": float(winner_eval["wikitext_validation"]["perplexity"])
                - float(dense["wikitext_validation"]["perplexity"]),
            },
            "to_s5_c0": {
                "recovery_validation_kl": float(winner_eval["recovery_validation_kl"])
                - float(control["recovery_validation_kl"]),
                "wikitext_perplexity": float(winner_eval["wikitext_validation"]["perplexity"])
                - float(control["wikitext_validation_perplexity"]),
            },
            "to_swiglu_3_nearest_5m_validation": {
                "source_tokens": int(source_nearest["tokens_seen"]),
                "recovery_validation_kl": float(winner_eval["recovery_validation_kl"])
                - float(source_nearest["recovery_validation_kl"]),
                "ppl_comparison_available": False,
            },
        },
    }


def width_curves_complete(context):
    layers = tuple(
        int(value) for value in context.settings["compatibility"]["eligible_layers"]
    )
    original_width = int(context.settings["model"]["intermediate_size"])
    widths = {
        min(original_width, max(1, round(original_width * float(ratio))))
        for ratio in context.settings["allocation"]["width_ratios"]
    }
    expected = {(layer, width) for layer in layers for width in widths}
    for initialization in ("legacy_subset", "output_aware"):
        rows = context.artifact["results"]["width_curves"].get(
            initialization, []
        )
        observed = {
            (int(row["layer"]), int(row["replacement_width"])) for row in rows
        }
        if observed != expected or any(
            "monotone_teacher_kl" not in row for row in rows
        ):
            return False
    return True


def initial_candidates_complete(context):
    expected_layers = {
        int(value) for value in context.settings["compatibility"]["eligible_layers"]
    }
    required_ids = {"S5-C0", "S5-C1", "S5-C2", "S5-C3"}
    candidates_by_target = context.artifact["results"]["candidates"]
    for target in context.settings["compatibility"]["target_mlp_removals"]:
        candidates = candidates_by_target.get(str(float(target)), {})
        if not required_ids <= set(candidates):
            return False
        for candidate_id in required_ids:
            candidate = candidates[candidate_id]
            layers = {int(row["layer"]) for row in candidate.get("allocation", [])}
            if layers != expected_layers or not candidate.get("pre_recovery"):
                return False
    return True


def run_search(context, allow_over_budget=False):
    """Execute the complete compute-bounded SwiGLU-5 tournament."""

    started = perf_counter()
    import_swiglu3_evidence(context)
    storage_preflight(context)
    model_config = load_search_resources(context)
    context.artifact["results"]["data"] = {
        "partition_order": list(context.data["partition_batches"]),
        "partition_batches": deepcopy(context.data["partition_batches"]),
        "selected_calibration_pairs": int(
            context.settings["references"]["selected_calibration_pairs"]
        ),
        "sequence_length": int(context.data["sequence_length"]),
    }
    record_stage_runtime(
        context,
        "resume_load_model_and_data" if context.resumed else "load_model_and_data",
        started,
    )

    started = perf_counter()
    cache_dtype = context.source["configuration"]["recovery"].get(
        "validation_cache_dtype", "float16"
    )
    validation_cache = cache_teacher_logits(
        context.model,
        context.data["recovery_validation"],
        int(context.data["partition_batches"]["recovery_validation"]),
        context.device,
        cache_dtype,
    )
    selection_cache = cache_teacher_logits(
        context.model,
        context.data["allocation_selection"],
        int(context.data["partition_batches"]["allocation_selection"]),
        context.device,
        cache_dtype,
    )
    if context.artifact["results"].get("dense_baseline") is None:
        context.artifact["results"]["dense_baseline"] = {
            "recovery_validation_kl": evaluate_validation_kl_mixed(
                context.model,
                validation_cache,
                float(context.settings["recovery"]["temperature"]),
                context.device,
            ),
            "allocation_selection": evaluate_teacher_cache_mixed(
                context.model,
                selection_cache,
                float(context.settings["recovery"]["temperature"]),
                context.device,
            ),
            "wikitext_validation": evaluate_lm_mixed(
                context.model,
                context.data["model_validation"],
                context.device,
                int(context.settings["data"]["model_validation_batches"]),
            ),
        }
    record_stage_runtime(
        context,
        (
            "resume_fixed_evaluation_caches"
            if context.resumed
            else "dense_and_fixed_evaluation_caches"
        ),
        started,
    )

    if not width_curves_complete(context):
        started = perf_counter()
        build_width_curves(context, selection_cache)
        record_stage_runtime(context, "width_curves", started)
    else:
        context.persist("width_curves_reused")

    started = perf_counter()
    if not initial_candidates_complete(context):
        build_c1_c3_candidates(context, selection_cache, validation_cache)
    build_composition_candidates(context, selection_cache, validation_cache)
    record_stage_runtime(
        context,
        "resume_candidate_assembly" if context.resumed else "candidate_assembly",
        started,
    )
    context.artifact["results"].setdefault("local_fit_state_pruning", []).append(
        prune_search_local_fit_states(
            context,
            pending_candidate_fit_keys(context),
            stage="candidate_assembly",
        )
    )
    context.persist("candidate_state_pruning")

    started = perf_counter()
    token_cache = packed_source_cache(context)
    hidden_settings = context.settings["teacher_hidden_cache"]
    hidden_cache_dtype = resolve_dtype(
        hidden_settings["dtype"], torch.device(context.device)
    )
    hidden_cache = build_teacher_final_hidden_cache(
        context.model,
        token_cache,
        context.asset_dir / "teacher-final-hidden",
        int(hidden_settings["target_tokens"]),
        int(hidden_settings["capture_batch_sequences"]),
        context.device,
        {
            "model_id": context.settings["model"]["model_id"],
            "revision": context.settings["model"]["revision"],
        },
        cache_dtype=hidden_cache_dtype,
        shard_tokens=int(hidden_settings["shard_tokens"]),
        minimum_free_gib=float(hidden_settings["minimum_free_gib"]),
    )
    equivalence = validate_teacher_final_hidden_cache(
        context.model,
        hidden_cache,
        token_cache,
        context.device,
        hidden_settings["validation_sample_offsets"],
        capture_batch_sequences=int(hidden_settings["capture_batch_sequences"]),
        temperature=float(context.settings["recovery"]["temperature"]),
        maximum_mean_kl=float(hidden_settings["maximum_mean_kl"]),
    )
    context.artifact["results"]["teacher_hidden_cache"] = {
        "root": relative_to_root(hidden_cache.root),
        "manifest": deepcopy(hidden_cache.manifest),
        "equivalence_validation": equivalence,
    }
    teacher_head = clone_teacher_head(context.model, context.device)
    record_stage_runtime(context, "teacher_final_hidden_cache", started)

    del context.model, context.blocks
    context.model = None
    context.blocks = None
    gc.collect()
    release_cuda(torch)

    started = perf_counter()
    kernel = calibrate_recovery(context, model_config, hidden_cache, teacher_head)
    record_stage_runtime(context, "kernel_calibration", started)
    guard = runtime_guard(
        context,
        kernel["observed_tokens_per_second"],
        allow_over_budget=allow_over_budget,
    )
    if not guard["passed"]:
        context.artifact["results"]["local_fit_state_pruning"].append(
            prune_search_local_fit_states(
                context,
                stage="budget_guard_rejected",
            )
        )
        context.artifact["results"]["teacher_hidden_cache_cleanup"] = (
            prune_teacher_hidden_cache(context, hidden_cache)
        )
        context.artifact["status"] = "budget_guard_rejected"
        context.artifact["completed_at_utc"] = utc_now()
        context.persist(None)
        return

    started = perf_counter()
    qualifier = int(context.settings["recovery"]["qualifier_tokens"])
    finalist_target = int(context.settings["recovery"]["finalist_tokens"])
    qualifier_selection = context.artifact["results"]["selection"].setdefault(
        "qualifier", {}
    )
    for target_key in ("0.2", "0.5"):
        candidates = context.artifact["results"]["candidates"][target_key]
        recover_candidate(
            context,
            candidates["S5-C0"],
            model_config,
            token_cache,
            hidden_cache,
            teacher_head,
            validation_cache,
            selection_cache,
            qualifier,
            kernel,
        )
        context.artifact["results"]["local_fit_state_pruning"].append(
            prune_search_local_fit_states(
                context,
                pending_candidate_fit_keys(context),
                stage=f"{target_key}-S5-C0-qualified",
            )
        )
        completed_challengers = []
        for candidate_id in ("S5-C1", "S5-C2", "S5-C3", "S5-C4"):
            recover_candidate(
                context,
                candidates[candidate_id],
                model_config,
                token_cache,
                hidden_cache,
                teacher_head,
                validation_cache,
                selection_cache,
                qualifier,
                kernel,
            )
            completed_challengers.append(candidate_id)
            retained = set(
                rank_qualifier_challengers(
                    context,
                    target_key,
                    completed_challengers,
                )[: int(context.settings["selection"]["challenger_finalists"])]
            )
            for completed_id in completed_challengers:
                if completed_id not in retained:
                    prune_search_candidate_checkpoints(
                        context,
                        candidates[completed_id],
                    )
            context.artifact["results"]["local_fit_state_pruning"].append(
                prune_search_local_fit_states(
                    context,
                    pending_candidate_fit_keys(context),
                    stage=f"{target_key}-{candidate_id}-qualified",
                )
            )
        finalists = select_finalists(context, target_key)
        qualifier_selection[target_key] = {
            "requested_tokens": qualifier,
            "finalists": finalists,
        }
        context.persist("qualifier_selection")
        for candidate_id in finalists:
            recover_candidate(
                context,
                candidates[candidate_id],
                model_config,
                token_cache,
                hidden_cache,
                teacher_head,
                validation_cache,
                selection_cache,
                finalist_target,
                kernel,
            )
            context.artifact["results"]["local_fit_state_pruning"].append(
                prune_search_local_fit_states(
                    context,
                    pending_candidate_fit_keys(context),
                    stage=f"{target_key}-{candidate_id}-finalist",
                )
            )
        selection = select_winner(
            context, target_key, finalists
        )
        winner = selection["winner_candidate_id"]
        selection["winner_endpoint"] = retain_search_winner_endpoint(
            context, candidates[winner]
        )
        context.artifact["results"]["selection"][target_key] = selection
        for candidate_id, candidate in candidates.items():
            if candidate_id != winner:
                prune_search_candidate_checkpoints(context, candidate)
        context.persist("winner_selection")
    record_stage_runtime(context, "candidate_recovery_and_selection", started)
    context.artifact["results"]["recovery_work"] = {
        "qualifier_candidate_token_positions": 20_000_000,
        "continuation_candidate_token_positions": 18_000_000,
        "total_candidate_token_positions": 38_000_000,
        "effective_batch_tokens": int(context.settings["recovery"]["effective_batch_tokens"]),
        "checkpoint_storage_policy": (
            "retain only exact qualifier states still eligible for continuation; "
            "process targets sequentially; retain one 5M winner endpoint per target"
        ),
    }
    context.artifact["results"]["local_fit_state_pruning"].append(
        prune_search_local_fit_states(
            context,
            stage="search_complete",
        )
    )
    context.artifact["results"]["teacher_hidden_cache_cleanup"] = (
        prune_teacher_hidden_cache(context, hidden_cache)
    )
    context.artifact["status"] = "completed"
    context.artifact["completed_at_utc"] = utc_now()
    context.persist(None)
