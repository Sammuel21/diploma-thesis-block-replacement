"""Recover one SwiGLU-6 model, with an explicit process boundary at 100M."""

import argparse
import gc
import math
import shutil
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch

from workflows.runs.model.common import resolve_path
from mlp_replacement.artifacts import contained_path, content_digest, file_digest, read_json
from mlp_replacement.compression.continuation import (
    capture_rng, commit_checkpoint, recover_exact_segment, restore_checkpoint,
    restore_rng, save_tensor_atomic, segment_origin, segment_schedule,
)
from mlp_replacement.compression.recovery import cache_teacher_logits
from mlp_replacement.data import PackedTokenCache
from mlp_replacement.model import load_model_and_tokenizer
from mlp_replacement.runlog import environment_record
from .shared import evaluate_lm_mixed, evaluate_validation_kl_mixed, make_model_config, recovery_memory_record
from .swiglu_5 import blank_candidate_student, load_replacement_state, replacement_state
from .swiglu_6 import DEFAULT_CONFIG, asset_directory, code_hashes, load_prepared, load_settings, new_artifact, persist, sources


def desired_requests(settings):
    recovery = settings["recovery"]
    final_tokens = recovery["segment_endpoints"][-1]
    return sorted(set(
        list(range(5_000_000, 100_000_001, recovery["legacy_validation_interval_tokens"]))
        + list(range(100_000_000, final_tokens + 1, recovery["checkpoint_interval_tokens"]))
        + recovery["legacy_ppl_tokens"] + recovery["segment_endpoints"]
    ))


def verify_replay(current, historical, settings):
    original = next(row for row in historical["results"]["trajectory"]["full_evaluations"]
                    if row["actual_tokens"] == 100_000_000)
    deltas = {"kl": current["recovery_validation_kl"] - original["recovery_validation_kl"],
              "ppl": current["wikitext_validation"]["perplexity"] - original["wikitext_validation"]["perplexity"]}
    return {**deltas, "passed": abs(deltas["kl"]) <= settings["replay_kl_tolerance"]
            and abs(deltas["ppl"]) <= settings["replay_ppl_tolerance"]}


def run_recovery(args, output, artifact, settings):
    target = str(args.target)
    if args.target not in settings["targets"]:
        raise ValueError("Target must be 0.2 or 0.5")
    prepared_path, prepared = load_prepared(args.prepared, settings)
    search, source, provenance = sources(settings)
    for key in ("search", "swiglu_3", "prefix"):
        if provenance[key]["sha256"] != prepared["provenance"][key]["sha256"]:
            raise ValueError(f"Prepared source changed: {key}")
    prepared_assets = asset_directory(prepared_path)
    stream = prepared["results"]["stream"]
    token_path = contained_path(prepared_assets, stream["path"])
    if file_digest(token_path) != stream["sha256"] or token_path.stat().st_size != 4 * stream["token_count"]:
        raise ValueError("Prepared stream content changed")
    legacy_record = prepared["legacy_evaluation"]
    legacy_path = contained_path(prepared_assets, legacy_record["path"])
    if file_digest(legacy_path) != legacy_record["sha256"]:
        raise ValueError("Prepared validation batches changed")
    contract = {"configuration": settings, "target": args.target,
                "search_sha256": provenance["search"]["sha256"],
                "endpoint_sha256": provenance["targets"][target]["endpoint"]["sha256"],
                "stream_sha256": stream["sha256"], "validation_sha256": legacy_record["sha256"],
                "code_hashes": code_hashes()}
    run_fingerprint = content_digest(contract)
    if args.resume and artifact.get("run_fingerprint") != run_fingerprint:
        raise ValueError("Resume inputs differ from the recorded scientific contract")
    artifact.update({"run_fingerprint": run_fingerprint, "target": args.target,
                     "code_hashes": contract["code_hashes"],
                     "provenance": provenance, "prepared_sha256": file_digest(prepared_path),
                     "prepared_path": str(prepared_path)})
    assets = asset_directory(output)
    assets.mkdir(parents=True, exist_ok=True)
    candidate = search["results"]["candidates"][target][settings["candidate_id"]]
    trainable_count = sum(3 * settings["model"]["hidden_size"] * row["replacement_width"]
                          + (settings["model"]["hidden_size"] if row.get("has_output_bias") else 0)
                          for row in candidate["allocation"] if not row.get("retains_dense_module"))
    # Three full generations cover previous/current/atomic temporary; milestone
    # weight files and complete BF16 bundles are also reserved.
    milestone_count = len(settings["recovery"]["segment_endpoints"])
    bundle_count = 1 + len(settings["targets"]) * milestone_count
    required_bytes = int((trainable_count * (12 * 3 + 4 * milestone_count)
                          + bundle_count * 1_800_000_000 * 2)
                         * (1 + settings["recovery"]["disk_reserve_fraction"]))
    artifact["storage_preflight"] = {"additional_free_bytes_required": required_bytes,
                                      "free_bytes": shutil.disk_usage(assets).free}
    if shutil.disk_usage(assets).free < required_bytes:
        raise RuntimeError(f"Recovery/export reserve requires {required_bytes:,} free bytes")
    state = None
    if args.resume:
        try:
            state, record = restore_checkpoint(assets / "checkpoints", run_fingerprint)
        except FileNotFoundError:
            if artifact["results"].get("tokens_seen", settings["recovery"]["start_tokens"]) > settings["recovery"]["start_tokens"]:
                raise
    if state is not None:
        artifact["results"] = deepcopy(state["results"])
        artifact["resumed_from"] = record
        if state["stream_sha256"] != stream["sha256"]:
            raise ValueError("Checkpoint stream differs")
    else:
        state = torch.load(provenance["targets"][target]["endpoint"]["path"], map_location="cpu", weights_only=False)
        artifact["results"] = {"validation_history": [], "milestones": {}, "training_seconds": 0.0,
                               "inherited_search_training_seconds": state["training_seconds"],
                               "evaluation_seconds": 0.0, "checkpoint_seconds": 0.0,
                               "tokens_seen": state["tokens_seen"], "optimizer_updates": state["optimizer_updates"]}
    if artifact["results"].get("replay_check", {}).get("passed") is False:
        raise ValueError("100M replay failed; inspect the discrepancy before a new experiment")
    final_tokens = settings["recovery"]["segment_endpoints"][-1]
    if int(state["tokens_seen"]) == final_tokens:
        artifact["status"] = "completed"
        persist(output, artifact, "completed")
        return
    observed_environment = environment_record()
    if args.resume and artifact.get("environment") is not None and artifact["environment"] != observed_environment:
        raise ValueError("Recovery environment changed since the committed run")
    historical = read_json(provenance["targets"][target]["confirmation"]["path"])
    for key in ("torch", "transformers", "datasets"):
        if observed_environment["packages"][key] != historical["environment"]["packages"][key]:
            raise ValueError(f"Replay requires the historical {key} version")
    artifact["environment"] = observed_environment
    persist(output, artifact, "loading_models")
    model_config = make_model_config(settings["model"])
    teacher, tokenizer = load_model_and_tokenizer(model_config)
    context = SimpleNamespace(settings=settings)
    student, target_paths, train_modules = blank_candidate_student(context, model_config, candidate)
    load_replacement_state(student, state["replacement_state"])
    device = next(student.parameters()).device
    legacy = torch.load(legacy_path, map_location="cpu", weights_only=False)
    validation_cache = cache_teacher_logits(teacher, legacy["recovery_validation"],
                                           len(legacy["recovery_validation"]), device,
                                           source["configuration"]["recovery"].get("validation_cache_dtype", "float16"))
    recovery = settings["recovery"]
    cache = PackedTokenCache(token_path, stream["token_count"], 128, stream["sha256"])
    cursor, updates = int(state["tokens_seen"]), int(state["optimizer_updates"])
    optimizer_state = state["optimizer_state"]
    restore_rng(state)
    del state
    if not artifact["results"]["validation_history"]:
        rng = capture_rng()
        start_row = {"actual_tokens": cursor, "requested_tokens": [5_000_000],
                     "training_seconds": 0.0,
                     "recovery_validation_kl": evaluate_validation_kl_mixed(student, validation_cache, 1.0, device),
                     "wikitext_validation": evaluate_lm_mixed(student, legacy["model_validation"], device, 24)}
        artifact["results"]["validation_history"].append(start_row)
        restore_rng(rng)
        artifact["checkpoint"] = commit_checkpoint(assets / "checkpoints", {
            "schema_version": 1, "workflow": "swiglu-6", "run_fingerprint": run_fingerprint,
            "stream_sha256": stream["sha256"], "replacement_state": replacement_state(student, target_paths),
            "optimizer_state": optimizer_state, "tokens_seen": cursor,
            "optimizer_updates": updates, "results": deepcopy(artifact["results"]), **rng,
        })
        persist(output, artifact, "recovery")
    while cursor < final_tokens:
        origin = segment_origin(cursor, recovery["segment_endpoints"])
        end = next(value for value in recovery["segment_endpoints"] if value > cursor)
        schedule = segment_schedule(origin, end, desired_requests(settings))

        def on_checkpoint(event, optimizer, first_step):
            rng = capture_rng()
            started = perf_counter()
            row = {"actual_tokens": event.tokens_seen, "requested_tokens": list(event.requested_checkpoint_tokens),
                   "training_seconds": event.elapsed_seconds,
                   "optimizer_updates": event.optimizer_updates, "mean_train_kl": event.mean_train_kl,
                   "recovery_validation_kl": evaluate_validation_kl_mixed(student, validation_cache, 1.0, device)}
            if not math.isfinite(row["recovery_validation_kl"]) or not math.isfinite(event.mean_train_kl):
                raise FloatingPointError("Non-finite recovery loss; no invalid state will be committed")
            if any(point in recovery["legacy_ppl_tokens"] for point in event.requested_checkpoint_tokens):
                row["wikitext_validation"] = evaluate_lm_mixed(student, legacy["model_validation"], device, 24)
            results = artifact["results"]
            results["validation_history"].append(row)
            results.update({"tokens_seen": event.tokens_seen, "optimizer_updates": event.optimizer_updates,
                            "training_seconds": event.elapsed_seconds, "memory": recovery_memory_record(device)})
            results["evaluation_seconds"] += perf_counter() - started
            if event.tokens_seen == 100_000_000:
                results["replay_check"] = verify_replay(row, historical, recovery)
            restore_rng(rng)
            is_milestone = event.tokens_seen in recovery["segment_endpoints"]
            is_checkpoint = is_milestone or any(point % recovery["checkpoint_interval_tokens"] == 0
                                               for point in event.requested_checkpoint_tokens)
            if is_checkpoint:
                started = perf_counter()
                weights = replacement_state(student, target_paths)
                if is_milestone:
                    weight_path = assets / "milestones" / f"weights-{event.tokens_seen:012d}.pt"
                    save_tensor_atomic(weight_path, {"replacement_state": weights, "tokens_seen": event.tokens_seen,
                                                   "run_fingerprint": run_fingerprint})
                    results["milestones"][str(event.tokens_seen)] = {
                        "path": str(weight_path.relative_to(assets)), "sha256": file_digest(weight_path),
                        "metrics": deepcopy(row)}
                payload = {"schema_version": 1, "workflow": "swiglu-6", "run_fingerprint": run_fingerprint,
                           "stream_sha256": stream["sha256"], "replacement_state": weights,
                           "optimizer_state": optimizer.state_dict(), "tokens_seen": event.tokens_seen,
                           "optimizer_updates": event.optimizer_updates, "results": deepcopy(results), **rng}
                artifact["checkpoint"] = commit_checkpoint(assets / "checkpoints", payload)
                results["checkpoint_seconds"] += perf_counter() - started
                del payload, weights
            persist(output, artifact, "recovery")
            print(f"target={target} tokens={event.tokens_seen:,} KL={row['recovery_validation_kl']:.6f}", flush=True)
            if results.get("replay_check", {}).get("passed") is False:
                raise ValueError("100M replay exceeds tolerance; inspect before continuing")

        result = recover_exact_segment(
            origin=origin, end=end, cursor=cursor, schedule=schedule,
            student=student, teacher=teacher,
            parameter_groups=[{"name": "replacements", "parameters": [parameter for module in train_modules
                              for parameter in module.parameters()], "learning_rate": recovery["learning_rate"],
                               "weight_decay": 0.0}], train_modules=train_modules,
            batch_at=lambda offset, count: cache.batch(offset, count, recovery["microbatch_sequences"]),
            microbatch_tokens=128 * recovery["microbatch_sequences"], accumulation_steps=2,
            temperature=1.0, ce_weight=0.0, scheduler="constant", warmup_fraction=0.0,
            final_lr_ratio=1.0, device=device, autocast_dtype=torch.bfloat16,
            start_updates=updates, elapsed_seconds=artifact["results"]["training_seconds"],
            optimizer_state=optimizer_state, on_checkpoint=on_checkpoint, optimizer_backend="fused",
        )
        cursor, updates = result.tokens_seen, result.optimizer_updates
        if cursor == 100_000_000 and recovery["pause_after_replay"]:
            artifact["status"] = "paused_after_replay"
            persist(output, artifact, "resume_required")
            print("100M replay passed and was checkpointed. Resume this output to continue to 1B.", flush=True)
            return
        if cursor < final_tokens:
            state, unused_record = restore_checkpoint(assets / "checkpoints", run_fingerprint)
            optimizer_state = state["optimizer_state"]
            restore_rng(state)
            del state
            gc.collect()
    artifact["status"] = "completed"
    persist(output, artifact, "completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--target", type=float, choices=(0.2, 0.5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    settings = load_settings(args.config)
    output, artifact = new_artifact(args.output, settings, "recovery", args.resume)
    try:
        artifact["status"], artifact["error"] = "running", None
        run_recovery(args, output, artifact, settings)
    except BaseException as error:
        artifact["status"] = "failed"
        artifact["error"] = f"{type(error).__name__}: {error}"
        persist(output, artifact, "failed")
        raise


if __name__ == "__main__":
    main()
