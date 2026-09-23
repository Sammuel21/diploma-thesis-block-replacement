"""SwiGLU-5 recovery; extracted with the established protocol unchanged."""

from __future__ import annotations

import gc
from copy import deepcopy
from time import perf_counter

import torch

from workflows.runs.model.common import release_cuda

from mlp_replacement.artifacts import atomic_torch_save, fingerprint
from mlp_replacement.compression.reconstruction import load_replacement_state, replacement_state
from mlp_replacement.compression.recovery import (
    next_optimizer_boundary,
    recover_trainable_by_tokens,
)
from mlp_replacement.evaluation.mixed_precision import (
    evaluate_lm_mixed,
    evaluate_teacher_cache_mixed,
    evaluate_validation_kl_mixed,
)

from ..shared import recovery_memory_record
from .candidates import blank_candidate_student, load_candidate_student
from .context import SEARCH_WORKFLOW, packed_source_cache


def checkpoint_rng():
    return {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_rng(checkpoint):
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_states") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])


def clone_teacher_head(model, device):
    head = deepcopy(model.get_output_embeddings()).to(device)
    for parameter in head.parameters():
        parameter.requires_grad = False
    head.eval()
    return head


def profile_trial(
    context,
    model_config,
    candidate,
    cache,
    teacher_head,
    geometry,
    compile_model=False,
):
    if torch.device(context.device).type != "cuda":
        raise RuntimeError("SwiGLU-5 kernel calibration requires CUDA")
    torch.manual_seed(int(context.settings["seed"]))
    torch.cuda.manual_seed_all(int(context.settings["seed"]))
    torch.cuda.reset_peak_memory_stats(context.device)
    student, unused_target_paths, train_modules = load_candidate_student(
        context, candidate, model_config
    )
    execution_model = student
    compile_error = None
    if compile_model:
        try:
            execution_model = torch.compile(student, mode="reduce-overhead")
        except BaseException as error:
            compile_error = f"{type(error).__name__}: {error}"
    events = []
    microbatch_sequences = int(geometry["microbatch_sequences"])
    microbatch_tokens = microbatch_sequences * int(
        context.settings["compatibility"]["sequence_length"]
    )
    effective = int(context.settings["recovery"]["effective_batch_tokens"])
    profile_updates = int(context.settings["recovery"]["profile_optimizer_updates"])
    if profile_updates < 1:
        raise ValueError("Recovery profiling requires at least one optimizer update")
    profile_tokens = effective * profile_updates
    if microbatch_tokens * int(geometry["gradient_accumulation_steps"]) != effective:
        raise ValueError("Profile geometry does not preserve the effective batch")
    started = perf_counter()
    try:
        result = recover_trainable_by_tokens(
            student=execution_model,
            teacher=None,
            parameter_groups=[
                {
                    "name": "replacements",
                    "parameters": [
                        parameter
                        for module in train_modules
                        for parameter in module.parameters()
                    ],
                    "learning_rate": float(context.settings["recovery"]["learning_rate"]),
                    "weight_decay": float(context.settings["recovery"]["weight_decay"]),
                }
            ],
            train_modules=train_modules,
            batch_at=lambda offset, count: packed_source_cache(context).batch(
                offset, count, microbatch_sequences
            ),
            target_tokens=profile_tokens,
            schedule_tokens=profile_tokens,
            microbatch_tokens=microbatch_tokens,
            accumulation_steps=int(geometry["gradient_accumulation_steps"]),
            temperature=float(context.settings["recovery"]["temperature"]),
            ce_weight=float(context.settings["recovery"]["ce_weight"]),
            scheduler="constant",
            warmup_fraction=0.0,
            final_lr_ratio=1.0,
            device=context.device,
            autocast_dtype=torch.bfloat16,
            checkpoint_schedule=((profile_tokens, (profile_tokens,)),),
            on_checkpoint=lambda event, unused_optimizer, unused_first_step: events.append(event),
            teacher_hidden_at=cache.batch,
            teacher_head=teacher_head,
            optimizer_backend=str(context.settings["recovery"]["optimizer_backend"]),
        )
        wall_seconds = perf_counter() - started
        peak = torch.cuda.max_memory_allocated(context.device) / 1024**3
        event = events[-1]
        return {
            **geometry,
            "execution": "compiled" if compile_model else "eager",
            "compile_error": compile_error,
            "tokens": result.tokens_seen,
            "training_seconds": result.elapsed_seconds,
            "wall_seconds": wall_seconds,
            "tokens_per_second": result.tokens_seen / result.elapsed_seconds,
            "mean_train_loss": event.mean_train_loss,
            "mean_train_kl": event.mean_train_kl,
            "peak_vram_gib": peak,
            "fits_memory": peak
            <= float(context.settings["recovery"]["maximum_peak_vram_gib"]),
        }
    finally:
        del execution_model, student, train_modules
        gc.collect()
        release_cuda(torch)


def calibrate_recovery(context, model_config, cache, teacher_head):
    existing = context.artifact["results"]["kernel_calibration"]
    if existing.get("selected_geometry"):
        return existing
    if (
        context.settings["recovery"]["optimizer"] != "AdamW"
        or context.settings["recovery"]["optimizer_backend"] != "fused"
    ):
        raise ValueError("SwiGLU-5 search requires fused CUDA AdamW")
    pre_profile_rng = checkpoint_rng()
    control = context.artifact["results"]["candidates"]["0.2"]["S5-C0"]
    profiles = []
    for geometry in context.settings["recovery"]["microbatch_candidates"]:
        try:
            profiles.append(
                profile_trial(
                    context,
                    model_config,
                    control,
                    cache,
                    teacher_head,
                    geometry,
                )
            )
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            torch.cuda.empty_cache()
            profiles.append(
                {
                    **geometry,
                    "execution": "eager",
                    "fits_memory": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    viable = [row for row in profiles if row.get("fits_memory")]
    if not viable:
        raise RuntimeError("No profiled recovery geometry fits the VRAM envelope")
    selected = max(viable, key=lambda row: int(row["microbatch_sequences"]))
    compiled = None
    compile_selected = False
    try:
        compiled = profile_trial(
            context,
            model_config,
            control,
            cache,
            teacher_head,
            {
                "microbatch_sequences": selected["microbatch_sequences"],
                "gradient_accumulation_steps": selected["gradient_accumulation_steps"],
            },
            compile_model=True,
        )
        speedup = compiled["tokens_per_second"] / selected["tokens_per_second"] - 1.0
        loss_difference = abs(
            float(compiled["mean_train_loss"]) - float(selected["mean_train_loss"])
        )
        compile_selected = (
            compiled.get("compile_error") is None
            and compiled["fits_memory"]
            and speedup >= float(context.settings["recovery"]["compile_minimum_speedup"])
            and loss_difference <= float(context.settings["recovery"]["compile_maximum_loss_difference"])
        )
        compiled["speedup_over_eager"] = speedup
        compiled["loss_difference_from_eager"] = loss_difference
    except BaseException as error:
        compiled = {"error": f"{type(error).__name__}: {error}"}
    record = {
        "disposable_state_restored_between_trials": True,
        "rng_restored_before_experiment": True,
        "profile_optimizer_updates": int(
            context.settings["recovery"]["profile_optimizer_updates"]
        ),
        "profiles": profiles,
        "selected_geometry": {
            "microbatch_sequences": int(selected["microbatch_sequences"]),
            "gradient_accumulation_steps": int(selected["gradient_accumulation_steps"]),
            "effective_batch_tokens": int(context.settings["recovery"]["effective_batch_tokens"]),
        },
        "compile_profile": compiled,
        "execution_mode": "compiled_reduce_overhead" if compile_selected else "eager",
        "observed_tokens_per_second": (
            float(compiled["tokens_per_second"])
            if compile_selected
            else float(selected["tokens_per_second"])
        ),
    }
    context.artifact["results"]["kernel_calibration"] = record
    context.persist("kernel_calibration")
    restore_rng(pre_profile_rng)
    return record


def requested_actual_map(
    requested_values,
    effective_batch,
    maximum_requested_tokens,
    actual_token_limit,
):
    """Map in-phase requested milestones to their optimizer boundaries."""

    if int(actual_token_limit) < int(maximum_requested_tokens):
        raise ValueError(
            "Actual token limit cannot precede the active requested-token phase"
        )
    mapped = {}
    for requested in requested_values:
        requested = int(requested)
        if requested > int(maximum_requested_tokens):
            raise ValueError(
                f"Requested milestone {requested:,} exceeds the active "
                f"{int(maximum_requested_tokens):,}-token phase"
            )
        actual = next_optimizer_boundary(
            requested,
            effective_batch,
            int(actual_token_limit),
        )
        mapped.setdefault(actual, []).append(requested)
    return {actual: tuple(values) for actual, values in sorted(mapped.items())}


def find_full_evaluation(trajectory, requested_tokens):
    """Return the first valid evaluation at or after a requested milestone."""

    matches = [
        row
        for row in trajectory["full_evaluations"]
        if int(requested_tokens)
        in [int(value) for value in row["requested_tokens"]]
    ]
    if not matches:
        raise ValueError(
            f"Candidate has no full evaluation at {requested_tokens:,} tokens"
        )
    valid_matches = [
        row
        for row in matches
        if int(row["actual_tokens"]) >= int(requested_tokens)
    ]
    if not valid_matches:
        raise ValueError(
            f"Candidate has only premature evaluations labeled for "
            f"{requested_tokens:,} tokens"
        )
    return min(
        valid_matches,
        key=lambda row: (
            int(row["actual_tokens"]),
            int(row["optimizer_updates"]),
        ),
    )


def recover_candidate(
    context,
    candidate,
    model_config,
    token_cache,
    hidden_cache,
    teacher_head,
    validation_cache,
    selection_cache,
    requested_target,
    kernel,
):
    recovery = candidate["recovery"]
    effective = int(context.settings["recovery"]["effective_batch_tokens"])
    actual_target = next_optimizer_boundary(
        int(requested_target), effective, int(hidden_cache.token_count)
    )
    if int(recovery.get("tokens_seen", 0)) >= actual_target:
        return
    if torch.device(context.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)
    target_key = str(float(candidate["target"]))
    candidate_id = candidate["candidate_id"]
    recovery_dir = context.asset_dir / "recovery" / target_key / candidate_id
    current_path = recovery_dir / "current.pt"
    if current_path.is_file():
        student, target_paths, train_modules = blank_candidate_student(
            context, model_config, candidate
        )
    else:
        student, target_paths, train_modules = load_candidate_student(
            context, candidate, model_config
        )
    execution_model = student
    if kernel["execution_mode"] == "compiled_reduce_overhead":
        execution_model = torch.compile(student, mode="reduce-overhead")
    optimizer_state = None
    start_tokens = int(recovery.get("tokens_seen", 0))
    start_updates = int(recovery.get("optimizer_updates", 0))
    elapsed = float(recovery.get("training_seconds", 0.0))
    if current_path.is_file():
        checkpoint = torch.load(current_path, map_location="cpu", weights_only=False)
        if checkpoint.get("run_fingerprint") != context.run_fingerprint:
            raise ValueError("Candidate recovery checkpoint fingerprint differs")
        if checkpoint.get("candidate_fingerprint") != fingerprint(
            {key: value for key, value in candidate.items() if key != "recovery"}
        ):
            raise ValueError("Candidate recovery checkpoint recipe differs")
        if checkpoint.get("packed_token_fingerprint") != token_cache.fingerprint:
            raise ValueError("Candidate recovery token stream differs")
        load_replacement_state(student, checkpoint["replacement_state"])
        optimizer_state = checkpoint["optimizer_state"]
        recovery.clear()
        recovery.update(checkpoint["recovery"])
        start_tokens = int(checkpoint["tokens_seen"])
        start_updates = int(checkpoint["optimizer_updates"])
        elapsed = float(checkpoint["training_seconds"])
        restore_rng(checkpoint)
    elif start_tokens:
        raise FileNotFoundError(
            f"Candidate continuation checkpoint is missing: {current_path}"
        )
    else:
        recovery["validation_history"].append(
            {
                "requested_tokens": [0],
                "actual_tokens": 0,
                "optimizer_updates": 0,
                "recovery_validation_kl": candidate["pre_recovery"]["recovery_validation_kl"],
            }
        )
        recovery["full_evaluations"].append(
            {
                "requested_tokens": [0],
                "actual_tokens": 0,
                "optimizer_updates": 0,
                "recovery_validation_kl": candidate["pre_recovery"]["recovery_validation_kl"],
                "allocation_selection": deepcopy(candidate["pre_recovery"]["allocation_selection"]),
                "wikitext_validation": deepcopy(candidate["pre_recovery"]["wikitext_validation"]),
            }
        )
    recovery["status"] = "running"
    context.persist("recovery")

    validation_requested = list(
        range(
            int(context.settings["recovery"]["validation_interval_tokens"]),
            int(requested_target) + 1,
            int(context.settings["recovery"]["validation_interval_tokens"]),
        )
    )
    if int(requested_target) not in validation_requested:
        validation_requested.append(int(requested_target))
    full_requested = [
        int(value)
        for value in context.settings["recovery"]["full_evaluation_tokens"]
        if int(value) <= int(requested_target)
        and start_tokens
        < next_optimizer_boundary(int(value), effective, actual_target)
        <= actual_target
    ]
    requested_union = sorted(
        {
            value
            for value in validation_requested + full_requested
            if value <= int(requested_target)
            and start_tokens
            < next_optimizer_boundary(value, effective, actual_target)
            <= actual_target
        }
    )
    schedule_map = requested_actual_map(
        requested_union,
        effective,
        int(requested_target),
        actual_target,
    )
    validation_map = requested_actual_map(
        validation_requested,
        effective,
        int(requested_target),
        actual_target,
    )
    full_map = requested_actual_map(
        full_requested,
        effective,
        int(requested_target),
        actual_target,
    )
    geometry = kernel["selected_geometry"]
    microbatch_sequences = int(geometry["microbatch_sequences"])
    microbatch_tokens = microbatch_sequences * int(token_cache.sequence_length)

    def persist_checkpoint(event, optimizer, first_step):
        evaluation_seconds = 0.0
        checkpoint_seconds = 0.0
        validation_kl = None
        if event.tokens_seen in validation_map:
            evaluation_started = perf_counter()
            validation_kl = evaluate_validation_kl_mixed(
                student,
                validation_cache,
                float(context.settings["recovery"]["temperature"]),
                context.device,
            )
            evaluation_seconds += perf_counter() - evaluation_started
            recovery["validation_history"].append(
                {
                    "requested_tokens": list(validation_map[event.tokens_seen]),
                    "actual_tokens": event.tokens_seen,
                    "optimizer_updates": event.optimizer_updates,
                    "recovery_validation_kl": validation_kl,
                    "mean_train_kl_since_resume": event.mean_train_kl,
                }
            )
        if event.tokens_seen in full_map:
            evaluation_started = perf_counter()
            if validation_kl is None:
                validation_kl = evaluate_validation_kl_mixed(
                    student,
                    validation_cache,
                    float(context.settings["recovery"]["temperature"]),
                    context.device,
                )
            selection = evaluate_teacher_cache_mixed(
                student,
                selection_cache,
                float(context.settings["recovery"]["temperature"]),
                context.device,
            )
            wiki = evaluate_lm_mixed(
                student,
                context.data["model_validation"],
                context.device,
                int(context.settings["data"]["model_validation_batches"]),
            )
            evaluation_seconds += perf_counter() - evaluation_started
            recovery["full_evaluations"].append(
                {
                    "requested_tokens": list(full_map[event.tokens_seen]),
                    "actual_tokens": event.tokens_seen,
                    "optimizer_updates": event.optimizer_updates,
                    "recovery_validation_kl": validation_kl,
                    "allocation_selection": selection,
                    "wikitext_validation": wiki,
                    "memory": recovery_memory_record(context.device),
                }
            )
        recovery["tokens_seen"] = event.tokens_seen
        recovery["optimizer_updates"] = event.optimizer_updates
        recovery["training_seconds"] = event.elapsed_seconds
        recovery["evaluation_seconds"] = float(recovery.get("evaluation_seconds", 0.0)) + evaluation_seconds
        recovery["first_step"] = recovery.get("first_step") or first_step
        if event.tokens_seen == actual_target:
            checkpoint_started = perf_counter()
            checkpoint = {
                "schema_version": 1,
                "workflow": SEARCH_WORKFLOW,
                "run_fingerprint": context.run_fingerprint,
                "candidate_fingerprint": fingerprint(
                    {key: value for key, value in candidate.items() if key != "recovery"}
                ),
                "target": candidate["target"],
                "candidate_id": candidate_id,
                "tokens_seen": event.tokens_seen,
                "optimizer_updates": event.optimizer_updates,
                "training_seconds": event.elapsed_seconds,
                "packed_token_fingerprint": token_cache.fingerprint,
                "replacement_state": replacement_state(student, target_paths),
                "optimizer_state": optimizer.state_dict(),
                "recovery": deepcopy(recovery),
                **checkpoint_rng(),
            }
            atomic_torch_save(current_path, checkpoint)
            checkpoint_seconds += perf_counter() - checkpoint_started
        recovery["checkpoint_seconds"] = float(recovery.get("checkpoint_seconds", 0.0)) + checkpoint_seconds
        context.persist("recovery")

    try:
        result = recover_trainable_by_tokens(
            student=execution_model,
            teacher=None,
            parameter_groups=[
                {
                    "name": "replacements",
                    "parameters": [
                        parameter
                        for module in train_modules
                        for parameter in module.parameters()
                    ],
                    "learning_rate": float(context.settings["recovery"]["learning_rate"]),
                    "weight_decay": float(context.settings["recovery"]["weight_decay"]),
                }
            ],
            train_modules=train_modules,
            batch_at=lambda offset, count: token_cache.batch(
                offset, count, microbatch_sequences
            ),
            target_tokens=actual_target,
            schedule_tokens=next_optimizer_boundary(
                int(context.settings["recovery"]["finalist_tokens"]),
                effective,
                hidden_cache.token_count,
            ),
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
            elapsed_seconds=elapsed,
            optimizer_state=optimizer_state,
            checkpoint_schedule=tuple(schedule_map.items()),
            on_checkpoint=persist_checkpoint,
            teacher_hidden_at=hidden_cache.batch,
            teacher_head=teacher_head,
            optimizer_backend=str(context.settings["recovery"]["optimizer_backend"]),
        )
        if result.tokens_seen != actual_target:
            raise RuntimeError("Candidate recovery ended before its token target")
        recovery["tokens_seen"] = result.tokens_seen
        recovery["optimizer_updates"] = result.optimizer_updates
        recovery["training_seconds"] = result.elapsed_seconds
        recovery["first_step"] = recovery.get("first_step") or result.first_step
        recovery["status"] = (
            "completed_5m"
            if int(requested_target) == int(context.settings["recovery"]["finalist_tokens"])
            else "completed_2m"
        )
        recovery["memory"] = recovery_memory_record(context.device)
        context.persist("recovery")
    finally:
        del execution_model, student, train_modules
        gc.collect()
        release_cuda(torch)
