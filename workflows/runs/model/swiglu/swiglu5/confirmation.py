"""SwiGLU-5 confirmation; extracted with the established protocol unchanged."""

from __future__ import annotations

import gc
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import torch

from workflows.runs.model.common import relative_to_root, release_cuda, resolve_path

from mlp_replacement.artifacts import atomic_torch_save, fingerprint, sha256_file
from mlp_replacement.compression.reconstruction import load_replacement_state, replacement_state
from mlp_replacement.compression.recovery import (
    next_optimizer_boundary,
    recover_trainable_by_tokens,
)
from mlp_replacement.compression.teacher_cache import cache_teacher_logits
from mlp_replacement.config import make_model_config
from mlp_replacement.evaluation.mixed_precision import (
    evaluate_lm_mixed,
    evaluate_teacher_cache_mixed,
    evaluate_validation_kl_mixed,
)
from mlp_replacement.model import discover_mlp_blocks, load_model_and_tokenizer

from ..shared import build_local_data, recovery_memory_record, source_milestone
from .candidates import blank_candidate_student
from .context import CONFIRMATION_WORKFLOW, packed_source_cache, utc_now
from .recovery import find_full_evaluation, requested_actual_map, restore_rng


def confirmation_candidate(context):
    key = str(float(context.settings["target"]))
    candidate_id = context.settings["selected_candidate_id"]
    return context.search["results"]["candidates"][key][candidate_id]


def load_selected_search_checkpoint(context):
    record = context.artifact["provenance"]
    path = resolve_path(Path(record["selected_endpoint"]["resolved_path"]))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("run_fingerprint") != context.search["run_fingerprint"]:
        raise ValueError("Selected endpoint belongs to a different search run")
    candidate = confirmation_candidate(context)
    expected = fingerprint(
        {key: value for key, value in candidate.items() if key != "recovery"}
    )
    if checkpoint.get("candidate_fingerprint") != expected:
        raise ValueError("Selected endpoint candidate recipe differs")
    return checkpoint


def online_profile_trial(context, model_config, candidate, teacher, geometry):
    checkpoint = load_selected_search_checkpoint(context)
    student, unused_target_paths, train_modules = blank_candidate_student(
        context, model_config, candidate
    )
    load_replacement_state(student, checkpoint["replacement_state"])
    restore_rng(checkpoint)
    token_cache = packed_source_cache(context)
    microbatch_sequences = int(geometry["microbatch_sequences"])
    microbatch_tokens = microbatch_sequences * token_cache.sequence_length
    effective = int(context.settings["recovery"]["effective_batch_tokens"])
    if microbatch_tokens * int(geometry["gradient_accumulation_steps"]) != effective:
        raise ValueError("Confirmation profile geometry changes effective batch size")
    start_tokens = int(checkpoint["tokens_seen"])
    target_tokens = start_tokens + effective
    events = []
    torch.cuda.reset_peak_memory_stats(context.device)
    try:
        result = recover_trainable_by_tokens(
            student=student,
            teacher=teacher,
            parameter_groups=[
                {
                    "name": "replacements",
                    "parameters": [
                        parameter for module in train_modules for parameter in module.parameters()
                    ],
                    "learning_rate": float(context.settings["recovery"]["learning_rate"]),
                    "weight_decay": float(context.settings["recovery"]["weight_decay"]),
                }
            ],
            train_modules=train_modules,
            batch_at=lambda offset, count: token_cache.batch(
                offset, count, microbatch_sequences
            ),
            target_tokens=target_tokens,
            schedule_tokens=int(context.settings["recovery"]["target_tokens"]),
            microbatch_tokens=microbatch_tokens,
            accumulation_steps=int(geometry["gradient_accumulation_steps"]),
            temperature=float(context.settings["recovery"]["temperature"]),
            ce_weight=float(context.settings["recovery"]["ce_weight"]),
            scheduler=str(context.settings["recovery"]["scheduler"]),
            warmup_fraction=float(context.settings["recovery"]["warmup_fraction"]),
            final_lr_ratio=float(context.settings["recovery"]["final_lr_ratio"]),
            device=context.device,
            autocast_dtype=torch.bfloat16,
            start_tokens=start_tokens,
            start_updates=int(checkpoint["optimizer_updates"]),
            elapsed_seconds=0.0,
            optimizer_state=checkpoint["optimizer_state"],
            checkpoint_schedule=((target_tokens, (target_tokens,)),),
            on_checkpoint=lambda event, unused_optimizer, unused_first_step: events.append(event),
            optimizer_backend=str(context.settings["recovery"]["optimizer_backend"]),
        )
        peak = torch.cuda.max_memory_allocated(context.device) / 1024**3
        event = events[-1]
        return {
            **geometry,
            "effective_batch_tokens": effective,
            "tokens_per_second": effective / result.elapsed_seconds,
            "training_seconds": result.elapsed_seconds,
            "mean_train_kl": event.mean_train_kl,
            "peak_vram_gib": peak,
            "fits_memory": peak <= float(context.settings["recovery"]["maximum_peak_vram_gib"]),
        }
    finally:
        del checkpoint, student, train_modules
        gc.collect()
        release_cuda(torch)


def calibrate_online_confirmation(context, model_config, candidate, teacher):
    existing = context.artifact["results"]["kernel_calibration"]
    if existing.get("selected_geometry"):
        return existing
    if (
        context.settings["recovery"]["optimizer"] != "AdamW"
        or context.settings["recovery"]["optimizer_backend"] != "fused"
    ):
        raise ValueError("SwiGLU-5 confirmation requires fused CUDA AdamW")
    profiles = []
    for geometry in context.settings["recovery"]["microbatch_candidates"]:
        try:
            profiles.append(
                online_profile_trial(
                    context, model_config, candidate, teacher, geometry
                )
            )
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            torch.cuda.empty_cache()
            profiles.append(
                {**geometry, "fits_memory": False, "error": f"{type(error).__name__}: {error}"}
            )
    viable = [row for row in profiles if row.get("fits_memory")]
    if not viable:
        raise RuntimeError("No online-teacher geometry fits the VRAM envelope")
    selected = max(viable, key=lambda row: int(row["microbatch_sequences"]))
    record = {
        "profiles": profiles,
        "disposable_state_restored_between_trials": True,
        "teacher_mode": "online_dense",
        "execution_mode": "eager",
        "selected_geometry": {
            "microbatch_sequences": int(selected["microbatch_sequences"]),
            "gradient_accumulation_steps": int(selected["gradient_accumulation_steps"]),
            "effective_batch_tokens": int(selected["effective_batch_tokens"]),
        },
    }
    context.artifact["results"]["kernel_calibration"] = record
    context.persist("kernel_calibration")
    return record


def source_comparison_row(context, milestone, current, trajectory):
    source_current = milestone["current"]
    requested = int(milestone["requested_tokens"][0])
    source_elapsed = None
    source_throughput = None
    if requested == 100_000_000:
        source_trajectory = context.source["results"]["recovery"]["trajectories"][
            str(float(context.settings["target"]))
        ]
        source_elapsed = float(source_trajectory["elapsed_seconds"])
        source_throughput = int(milestone["actual_tokens"]) / source_elapsed
    current_training_seconds = float(
        current.get("cumulative_training_seconds", trajectory["training_seconds"])
    )
    current_elapsed = float(
        current.get(
            "cumulative_elapsed_seconds",
            current_training_seconds
            + float(trajectory.get("evaluation_seconds", 0.0))
            + float(trajectory.get("checkpoint_seconds", 0.0)),
        )
    )
    current_throughput = int(current["actual_tokens"]) / current_elapsed
    source_memory = (
        context.source["results"]["recovery"]["trajectories"][
            str(float(context.settings["target"]))
        ].get("memory", {})
        if requested == 100_000_000
        else {}
    )
    current_memory = current.get("memory", trajectory.get("memory", {}))
    return {
        "requested_tokens": requested,
        "swiglu_3_actual_tokens": int(milestone["actual_tokens"]),
        "swiglu_5_actual_tokens": int(current["actual_tokens"]),
        "same_actual_boundary": int(milestone["actual_tokens"]) == int(current["actual_tokens"]),
        "swiglu_3": {
            "recovery_validation_kl": float(source_current["recovery_validation_kl"]),
            "wikitext_validation_loss": float(source_current["wikitext_validation"]["loss"]),
            "wikitext_validation_perplexity": float(source_current["wikitext_validation"]["perplexity"]),
            "optimizer_updates": int(milestone["optimizer_updates"]),
            "elapsed_seconds": source_elapsed,
            "tokens_per_second": source_throughput,
            "memory": deepcopy(source_memory),
        },
        "swiglu_5": {
            "recovery_validation_kl": float(current["recovery_validation_kl"]),
            "wikitext_validation_loss": float(current["wikitext_validation"]["loss"]),
            "wikitext_validation_perplexity": float(current["wikitext_validation"]["perplexity"]),
            "optimizer_updates": int(current["optimizer_updates"]),
            "training_seconds": current_training_seconds,
            "elapsed_seconds": current_elapsed,
            "tokens_per_second": current_throughput,
            "memory": deepcopy(current_memory),
        },
        "difference_swiglu_5_minus_swiglu_3": {
            "recovery_validation_kl": float(current["recovery_validation_kl"])
            - float(source_current["recovery_validation_kl"]),
            "wikitext_validation_loss": float(current["wikitext_validation"]["loss"])
            - float(source_current["wikitext_validation"]["loss"]),
            "wikitext_validation_perplexity": float(current["wikitext_validation"]["perplexity"])
            - float(source_current["wikitext_validation"]["perplexity"]),
            "optimizer_updates": int(current["optimizer_updates"])
            - int(milestone["optimizer_updates"]),
            "elapsed_seconds": (
                current_elapsed - source_elapsed if source_elapsed is not None else None
            ),
            "tokens_per_second": (
                current_throughput - source_throughput
                if source_throughput is not None
                else None
            ),
            "peak_ram_gib": (
                float(current_memory["peak_ram_gib"]) - float(source_memory["peak_ram_gib"])
                if "peak_ram_gib" in current_memory and "peak_ram_gib" in source_memory
                else None
            ),
            "peak_vram_gib": (
                float(current_memory["peak_vram_gib"]) - float(source_memory["peak_vram_gib"])
                if "peak_vram_gib" in current_memory and "peak_vram_gib" in source_memory
                else None
            ),
        },
        "runtime_comparison_note": (
            None
            if source_elapsed is not None
            else (
                "SwiGLU-3 did not persist observed 10M elapsed-time or "
                "milestone-memory records"
            )
        ),
        "intentional_method_differences": [
            "selected SwiGLU-5 initialization/allocation recipe versus exact SwiGLU-3 recipe",
            "constant 3e-5 versus constant 1e-5 learning rate",
            "fused CUDA AdamW versus recorded SwiGLU-3 AdamW implementation",
            "profiled microbatch geometry versus SwiGLU-3 2-sequence x 8 accumulation geometry",
        ],
    }


def run_confirmation(context):
    """Continue one selected 5M search endpoint to the 100M boundary."""

    stage_started = perf_counter()
    model_config = make_model_config(context.settings["model"])
    teacher, tokenizer = load_model_and_tokenizer(model_config)
    context.model = teacher
    context.tokenizer = tokenizer
    context.device = next(teacher.parameters()).device
    context.model_dtype = next(teacher.parameters()).dtype
    teacher_blocks = {block.index: block for block in discover_mlp_blocks(teacher)}
    if set(teacher_blocks) != set(range(int(context.settings["model"]["num_layers"]))):
        raise ValueError("Confirmation teacher topology differs from the pinned model")
    if teacher.get_input_embeddings().weight.data_ptr() != teacher.get_output_embeddings().weight.data_ptr():
        raise ValueError("Confirmation teacher must retain tied embeddings")
    del teacher_blocks
    context.data = build_local_data(context)
    token_cache = packed_source_cache(context)
    endpoint = load_selected_search_checkpoint(context)
    if endpoint.get("packed_token_fingerprint") != token_cache.fingerprint:
        raise ValueError("Search endpoint and SwiGLU-3 token stream differ")
    start_requested = int(context.settings["recovery"]["start_tokens"])
    effective = int(context.settings["recovery"]["effective_batch_tokens"])
    expected_start = next_optimizer_boundary(
        start_requested, effective, token_cache.token_count
    )
    if int(endpoint["tokens_seen"]) != expected_start:
        raise ValueError("Selected search endpoint is not the exact 5M boundary")
    del endpoint
    cache_dtype = context.source["configuration"]["recovery"].get(
        "validation_cache_dtype", "float16"
    )
    validation_cache = cache_teacher_logits(
        teacher,
        context.data["recovery_validation"],
        int(context.data["partition_batches"]["recovery_validation"]),
        context.device,
        cache_dtype,
    )
    selection_cache = cache_teacher_logits(
        teacher,
        context.data["allocation_selection"],
        int(context.data["partition_batches"]["allocation_selection"]),
        context.device,
        cache_dtype,
    )
    context.artifact["results"]["runtime"].append(
        {"stage": "load_and_validate", "seconds": perf_counter() - stage_started}
    )
    context.persist("load_and_validate")

    candidate = confirmation_candidate(context)
    stage_started = perf_counter()
    kernel = calibrate_online_confirmation(
        context, model_config, candidate, teacher
    )
    context.artifact["results"]["runtime"].append(
        {"stage": "kernel_calibration", "seconds": perf_counter() - stage_started}
    )
    context.persist("kernel_calibration")

    student, target_paths, train_modules = blank_candidate_student(
        context, model_config, candidate
    )
    endpoint = load_selected_search_checkpoint(context)
    trajectory = context.artifact["results"]["trajectory"]
    best_path = context.asset_dir / "recovery" / "best.pt"
    final_path = context.asset_dir / "recovery" / "final.pt"
    load_replacement_state(student, endpoint["replacement_state"])
    optimizer_state = endpoint["optimizer_state"]
    restore_rng(endpoint)
    start_tokens = int(endpoint["tokens_seen"])
    start_updates = int(endpoint["optimizer_updates"])
    training_seconds = float(endpoint["training_seconds"])
    search_full = find_full_evaluation(candidate["recovery"], start_requested)
    trajectory.update(
        {
            "status": "running",
            "tokens_seen": start_tokens,
            "optimizer_updates": start_updates,
            "training_seconds": training_seconds,
            "evaluation_seconds": float(
                candidate["recovery"].get("evaluation_seconds", 0.0)
            ),
            "checkpoint_seconds": float(
                candidate["recovery"].get("checkpoint_seconds", 0.0)
            ),
            "validation_history": [
                {
                    "requested_tokens": [start_requested],
                    "actual_tokens": start_tokens,
                    "optimizer_updates": start_updates,
                    "recovery_validation_kl": search_full["recovery_validation_kl"],
                    "source": "swiglu-5-search-endpoint",
                }
            ],
            "full_evaluations": [deepcopy(search_full)],
            "best_validation_kl": float(search_full["recovery_validation_kl"]),
            "best_checkpoint_tokens": start_tokens,
            "best_checkpoint_updates": start_updates,
        }
    )
    context.persist("recovery")

    target_tokens = int(context.settings["recovery"]["target_tokens"])
    if target_tokens > token_cache.token_count:
        raise ValueError("Confirmation target exceeds the finite SwiGLU-3 token stream")
    interval = int(context.settings["recovery"]["validation_interval_tokens"])
    validation_requested = list(range(interval, target_tokens + 1, interval))
    validation_requested = [value for value in validation_requested if value > start_requested]
    full_requested = [
        int(value)
        for value in context.settings["recovery"]["full_evaluation_tokens"]
        if value > start_requested
    ]
    requested_union = sorted(set(validation_requested + full_requested))
    schedule_map = requested_actual_map(
        requested_union,
        effective,
        target_tokens,
        target_tokens,
    )
    validation_map = requested_actual_map(
        validation_requested,
        effective,
        target_tokens,
        target_tokens,
    )
    full_map = requested_actual_map(
        full_requested,
        effective,
        target_tokens,
        target_tokens,
    )
    geometry = kernel["selected_geometry"]
    microbatch_sequences = int(geometry["microbatch_sequences"])
    microbatch_tokens = microbatch_sequences * token_cache.sequence_length
    trajectory["status"] = "running"
    context.persist("recovery")
    if torch.device(context.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)

    def on_checkpoint(event, unused_optimizer, first_step):
        evaluation_started = perf_counter()
        validation_kl = evaluate_validation_kl_mixed(
            student,
            validation_cache,
            float(context.settings["recovery"]["temperature"]),
            context.device,
        )
        trajectory["validation_history"].append(
            {
                "requested_tokens": list(validation_map[event.tokens_seen]),
                "actual_tokens": event.tokens_seen,
                "optimizer_updates": event.optimizer_updates,
                "recovery_validation_kl": validation_kl,
                "mean_train_kl_since_resume": event.mean_train_kl,
            }
        )
        if event.tokens_seen in full_map:
            trajectory["full_evaluations"].append(
                {
                    "requested_tokens": list(full_map[event.tokens_seen]),
                    "actual_tokens": event.tokens_seen,
                    "optimizer_updates": event.optimizer_updates,
                    "recovery_validation_kl": validation_kl,
                    "allocation_selection": evaluate_teacher_cache_mixed(
                        student,
                        selection_cache,
                        float(context.settings["recovery"]["temperature"]),
                        context.device,
                    ),
                    "wikitext_validation": evaluate_lm_mixed(
                        student,
                        context.data["model_validation"],
                        context.device,
                        int(context.settings["data"]["model_validation_batches"]),
                    ),
                    "memory": recovery_memory_record(context.device),
                }
            )
        trajectory["evaluation_seconds"] = float(
            trajectory.get("evaluation_seconds", 0.0)
        ) + perf_counter() - evaluation_started
        improved = validation_kl < float(trajectory["best_validation_kl"])
        if improved:
            trajectory["best_validation_kl"] = validation_kl
            trajectory["best_checkpoint_tokens"] = event.tokens_seen
            trajectory["best_checkpoint_updates"] = event.optimizer_updates
        trajectory["tokens_seen"] = event.tokens_seen
        trajectory["optimizer_updates"] = event.optimizer_updates
        trajectory["training_seconds"] = event.elapsed_seconds
        trajectory["first_step"] = trajectory.get("first_step") or first_step
        if improved:
            checkpoint_started = perf_counter()
            atomic_torch_save(
                best_path,
                {
                    "schema_version": 1,
                    "workflow": CONFIRMATION_WORKFLOW,
                    "run_fingerprint": context.run_fingerprint,
                    "tokens_seen": event.tokens_seen,
                    "optimizer_updates": event.optimizer_updates,
                    "replacement_state": replacement_state(student, target_paths),
                },
            )
            trajectory["checkpoint_seconds"] = float(
                trajectory.get("checkpoint_seconds", 0.0)
            ) + perf_counter() - checkpoint_started
        if event.tokens_seen in full_map:
            full_row = trajectory["full_evaluations"][-1]
            full_row["cumulative_training_seconds"] = event.elapsed_seconds
            full_row["cumulative_evaluation_seconds"] = float(
                trajectory["evaluation_seconds"]
            )
            full_row["cumulative_checkpoint_seconds"] = float(
                trajectory["checkpoint_seconds"]
            )
            full_row["cumulative_elapsed_seconds"] = (
                full_row["cumulative_training_seconds"]
                + full_row["cumulative_evaluation_seconds"]
                + full_row["cumulative_checkpoint_seconds"]
            )
        context.persist("recovery")

    stage_started = perf_counter()
    try:
        result = recover_trainable_by_tokens(
            student=student,
            teacher=teacher,
            parameter_groups=[
                {
                    "name": "replacements",
                    "parameters": [
                        parameter for module in train_modules for parameter in module.parameters()
                    ],
                    "learning_rate": float(context.settings["recovery"]["learning_rate"]),
                    "weight_decay": float(context.settings["recovery"]["weight_decay"]),
                }
            ],
            train_modules=train_modules,
            batch_at=lambda offset, count: token_cache.batch(
                offset, count, microbatch_sequences
            ),
            target_tokens=target_tokens,
            schedule_tokens=target_tokens,
            microbatch_tokens=microbatch_tokens,
            accumulation_steps=int(geometry["gradient_accumulation_steps"]),
            temperature=float(context.settings["recovery"]["temperature"]),
            ce_weight=float(context.settings["recovery"]["ce_weight"]),
            scheduler=str(context.settings["recovery"]["scheduler"]),
            warmup_fraction=float(context.settings["recovery"]["warmup_fraction"]),
            final_lr_ratio=float(context.settings["recovery"]["final_lr_ratio"]),
            device=context.device,
            autocast_dtype=torch.bfloat16,
            start_tokens=start_tokens,
            start_updates=start_updates,
            elapsed_seconds=training_seconds,
            optimizer_state=optimizer_state,
            checkpoint_schedule=tuple(schedule_map.items()),
            on_checkpoint=on_checkpoint,
            optimizer_backend=str(context.settings["recovery"]["optimizer_backend"]),
        )
        if result.tokens_seen != target_tokens:
            raise RuntimeError("Confirmation ended before the 100M boundary")
        trajectory["tokens_seen"] = result.tokens_seen
        trajectory["optimizer_updates"] = result.optimizer_updates
        trajectory["training_seconds"] = result.elapsed_seconds
        trajectory["memory"] = recovery_memory_record(context.device)
        trajectory["status"] = "completed"
        checkpoint_started = perf_counter()
        best_is_final = int(trajectory["best_checkpoint_tokens"]) == result.tokens_seen
        if best_is_final:
            if final_path.exists():
                raise FileExistsError(f"Final checkpoint already exists: {final_path}")
            best_path.replace(final_path)
            retained_best_path = final_path
            retained_best_contents = "replacement_weights_only"
        else:
            atomic_torch_save(
                final_path,
                {
                    "schema_version": 1,
                    "workflow": CONFIRMATION_WORKFLOW,
                    "run_fingerprint": context.run_fingerprint,
                    "tokens_seen": result.tokens_seen,
                    "optimizer_updates": result.optimizer_updates,
                    "replacement_state": replacement_state(student, target_paths),
                },
            )
            if int(trajectory["best_checkpoint_tokens"]) == start_tokens:
                retained_best_path = resolve_path(
                    Path(
                        context.artifact["provenance"]["selected_endpoint"][
                            "resolved_path"
                        ]
                    )
                )
                retained_best_contents = "search_endpoint_with_optimizer_state"
            else:
                retained_best_path = best_path
                retained_best_contents = "replacement_weights_only"
        trajectory["checkpoint_seconds"] = float(
            trajectory.get("checkpoint_seconds", 0.0)
        ) + perf_counter() - checkpoint_started
        final_evaluation = find_full_evaluation(trajectory, target_tokens)
        final_evaluation["cumulative_checkpoint_seconds"] = float(
            trajectory["checkpoint_seconds"]
        )
        final_evaluation["cumulative_elapsed_seconds"] = (
            float(final_evaluation["cumulative_training_seconds"])
            + float(final_evaluation["cumulative_evaluation_seconds"])
            + float(final_evaluation["cumulative_checkpoint_seconds"])
        )
        trajectory["final_checkpoint"] = {
            "path": relative_to_root(final_path),
            "sha256": sha256_file(final_path),
            "tokens_seen": result.tokens_seen,
            "contents": "replacement_weights_only",
        }
        trajectory["best_checkpoint"] = {
            "path": relative_to_root(retained_best_path),
            "sha256": sha256_file(retained_best_path),
            "tokens_seen": int(trajectory["best_checkpoint_tokens"]),
            "selection_rule": "lowest fixed validation KL at or below 100M",
            "contents": retained_best_contents,
        }
        context.artifact["results"]["runtime"].append(
            {"stage": "confirmation_recovery", "seconds": perf_counter() - stage_started}
        )
        comparisons = []
        target_key = str(float(context.settings["target"]))
        for requested in context.settings["comparison"]["swiglu_3_equal_token_milestones"]:
            source_m = source_milestone(context.source, target_key, int(requested))
            current_m = find_full_evaluation(trajectory, int(requested))
            comparisons.append(
                source_comparison_row(context, source_m, current_m, trajectory)
            )
        context.artifact["results"]["paired_swiglu_3_comparison"] = comparisons
        context.artifact["status"] = "completed"
        context.artifact["completed_at_utc"] = utc_now()
        context.persist(None)
    finally:
        del endpoint, student, teacher, train_modules
        gc.collect()
        release_cuda(torch)
