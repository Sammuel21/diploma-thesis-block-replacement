"""Quantization-1: scoped PTQ/QAT, deployment evaluation, and reporting."""

import argparse
import csv
import gc
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

from mlp_replacement.artifacts import (
    contained_path, content_digest, file_digest, read_json, write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = ROOT / "workflows/configs/model/baseline/quantization-1.json"
VARIANTS = ("dense-bf16", "ptq-int8", "ptq-int4", "qat-int4")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_settings(path, qat_tokens=None):
    settings = read_json(resolve_path(path))
    if settings.get("workflow") != "quantization-1" or settings.get("schema_version") != 1:
        raise ValueError("Expected a Quantization-1 configuration")
    if qat_tokens is not None:
        settings["qat"]["target_tokens"] = qat_tokens
    if settings["scope"]["eligible_layers"] != list(range(1, 23)):
        raise ValueError("Quantization-1 quantizes MLPs 1–22 only")
    if settings["scope"]["protected_layers"] != [0, 23]:
        raise ValueError("Quantization-1 protects boundary MLPs")
    if settings["scope"]["projections"] != ["gate_proj", "up_proj", "down_proj"]:
        raise ValueError("Quantization-1 uses exactly the three SwiGLU projections")
    if settings["scope"]["expected_projections"] != 66:
        raise ValueError("Quantization-1 expects 66 eligible projections")
    if settings["model"]["revision"] != settings["model"]["tokenizer_revision"]:
        raise ValueError("Model and tokenizer must share their frozen revision")
    if settings["model"]["dtype"] != "bfloat16" or settings["model"]["device"] != "cuda":
        raise ValueError("Quantization-1 deploys BF16 activations on CUDA")
    if settings["data"]["sequence_length"] != 8192 or settings["data"]["kl_sequences"] != 24:
        raise ValueError("Quantization-1 uses the fixed 24-sequence 8K KL diagnostic")
    if settings["data"]["recovery_source"]["first_shard"] != 1:
        raise ValueError("Recovery must exclude KL diagnostic shard 0")
    for key, expected in {"sequence_length": 8192, "microbatch_sequences": 1,
                          "gradient_accumulation_steps": 1, "scheduler": "constant",
                          "ce_weight": 0.0, "temperature": 1.0,
                          "trainable_parameter_dtype": "float32", "optimizer_state_dtype": "float32",
                          "forward_autocast_dtype": "bfloat16", "optimizer": "AdamW",
                          "optimizer_backend": "fused", "weight_decay": 0.0,
                          "warmup_fraction": 0.0, "final_lr_ratio": 1.0,
                          "effective_batch_tokens": 8192, "checkpoint_interval_tokens": 25_000_000,
                          "validation_interval_tokens": 25_000_000}.items():
        if settings["qat"][key] != expected:
            raise ValueError(f"Unsupported Quantization-1 recovery setting: {key}")
    tokens = settings["qat"]["target_tokens"]
    if tokens is not None and (type(tokens) is not int or tokens < 1):
        raise ValueError("QAT requires an explicitly specified positive integer token budget")
    if any(value not in VARIANTS for value in settings["variants"]):
        raise ValueError("Unknown default quantization variant")
    if "qat-int4" in settings["variants"]:
        raise ValueError("QAT is optional and cannot be in the default run list")
    if settings["runtime"]["attention_implementation"] != "sdpa":
        raise ValueError("Quantization-1 uses the common eager SDPA execution path")
    return settings


def scientific_settings(settings):
    return {key: value for key, value in settings.items() if key not in ("outputs", "references")}


def preparation_settings(settings):
    return {key: settings[key] for key in ("model", "data", "evaluation", "seed")}


def source_hashes():
    paths = [*sorted((ROOT / "src/mlp_replacement").rglob("*.py")), Path(__file__)]
    return {str(path.relative_to(ROOT)): file_digest(path) for path in paths}


def start_artifact(directory, settings, command, identity, resume):
    """Reuse the crash-aware log, restoring it only for the same scientific run."""

    from mlp_replacement.runlog import ExperimentLog, environment_record

    directory = Path(directory)
    contract = {"configuration": scientific_settings(settings), "command": command,
                "identity": identity, "source_hashes": source_hashes()}
    fingerprint = content_digest(contract)
    environment = environment_record()
    for package in ("torchao", "mslk", "triton", "accelerate", "safetensors"):
        try:
            observed = version(package)
        except PackageNotFoundError:
            observed = None
        environment["packages"][package] = observed
    if resume:
        artifact = read_json(directory / "result.json")
        if artifact["run_fingerprint"] != fingerprint:
            raise ValueError("Resume configuration, input, or source fingerprint changed")
        log = ExperimentLog.__new__(ExperimentLog)
        log.path = directory / "run.json"
        log.data = read_json(log.path)
        log.data.setdefault("attempts", []).append({"started_at": utc_now(),
                                                   "environment": environment})
        log.data.update(status="running", error=None, finished_at=None)
        artifact.update(status="running", error=None, finished_at=None)
    else:
        if directory.exists():
            raise FileExistsError(f"Output already exists; explicit compatible resume required: {directory}")
        directory.mkdir(parents=True)
        log = ExperimentLog(directory / "run.json", settings)
        log.data["environment"] = environment
        artifact = {"schema_version": 1, "workflow": "quantization-1", "command": command,
                    "identity": identity, "configuration": settings,
                    "run_fingerprint": fingerprint, "source_hashes": contract["source_hashes"],
                    "status": "running", "created_at": utc_now(), "results": {}}
        write_json_atomic(directory / "effective-config.json", settings)
    persist(directory, artifact, log, "initialization")
    return artifact, log


def persist(directory, artifact, log, stage):
    artifact["updated_at"] = utc_now()
    artifact["current_stage"] = stage
    write_json_atomic(Path(directory) / "result.json", artifact)
    log.begin(stage)


def finish(directory, artifact, log):
    artifact.update(status="completed", finished_at=utc_now(), current_stage=None)
    write_json_atomic(Path(directory) / "result.json", artifact)
    log.complete({"path": "result.json", "run_fingerprint": artifact["run_fingerprint"]})


def release_cuda():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def seed_run(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_teacher(settings):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = settings["model"]
    for package in ("transformers", "datasets"):
        if version(package) != settings["backend"][package]:
            raise ValueError(f"Install pinned {package} for Quantization-1")
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_id"],
                                            revision=model_config["tokenizer_revision"],
                                            trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_id"], revision=model_config["revision"],
        torch_dtype=torch.bfloat16, trust_remote_code=False,
        attn_implementation=settings["runtime"]["attention_implementation"],
    ).eval().to("cuda")
    for key in ("hidden_size", "intermediate_size", "tie_word_embeddings"):
        if getattr(model.config, key) != model_config[key]:
            raise ValueError(f"Pinned model topology changed: {key}")
    if model.config.num_hidden_layers != model_config["num_layers"]:
        raise ValueError("Pinned model layer count changed")
    model.requires_grad_(False)
    return model, tokenizer


def harness_files():
    import lm_eval

    root = Path(lm_eval.__file__).parent
    return {str(path.relative_to(root)): file_digest(path) for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix in (".py", ".yaml")}


def task_config(task, evaluation):
    from lm_eval.tasks import TaskManager
    from lm_eval.tasks._yaml_loader import load_yaml

    entry = TaskManager().task_index[task]
    path = entry.yaml_path if hasattr(entry, "yaml_path") else entry["yaml_path"]
    config = load_yaml(path, resolve_func=True, recursive=True)
    pinned = evaluation["task_datasets"][task]
    split = config.get("test_split") or config.get("validation_split")
    if config["dataset_path"] != pinned["dataset"] or split != pinned["split"]:
        raise ValueError(f"Pinned task definition changed: {task}")
    config["dataset_kwargs"] = {**(config.get("dataset_kwargs") or {}), "revision": pinned["revision"]}
    config["num_fewshot"] = 0
    return config


def verified_tokens(root, record):
    import numpy as np

    path = contained_path(root, record["path"])
    if path.stat().st_size != record["token_count"] * 4 or file_digest(path) != record["sha256"]:
        raise ValueError(f"Prepared token stream changed: {path}")
    return np.memmap(path, mode="r", dtype=np.int32)


def load_prepared(path, settings):
    path = resolve_path(path)
    artifact = read_json(path)
    if artifact["workflow"] != "quantization-1" or artifact["command"] != "prepare" or artifact["status"] != "completed":
        raise ValueError("Expected completed Quantization-1 preparation")
    prepared = artifact["results"]["prepared"]
    if prepared["configuration_fingerprint"] != content_digest(preparation_settings(settings)):
        raise ValueError("Prepared model, tokenizer, data, or quality configuration changed")
    body = {key: value for key, value in prepared.items() if key != "protocol_fingerprint"}
    if prepared["protocol_fingerprint"] != content_digest(body):
        raise ValueError("Prepared protocol fingerprint changed")
    if version("lm_eval") != settings["evaluation"]["harness_version"] or harness_files() != prepared["harness_source_hashes"]:
        raise ValueError("Evaluation harness differs from the frozen preparation")
    for record in [*prepared["corpora"].values(), prepared["kl_stream"]]:
        verified_tokens(path.parent, record)
    return path.parent, prepared


def prepare(directory, artifact, log, settings):
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from mlp_replacement.data_streams import build_finite_stream

    persist(directory, artifact, log, "prepare")
    for package in ("transformers", "datasets"):
        if version(package) != settings["backend"][package]:
            raise ValueError(f"Install pinned {package} before preparation")
    if version("lm_eval") != settings["evaluation"]["harness_version"]:
        raise ValueError("Install the pinned evaluation harness before preparation")
    tokenizer = AutoTokenizer.from_pretrained(settings["model"]["model_id"],
                                            revision=settings["model"]["tokenizer_revision"])
    evaluation = settings["evaluation"]
    for task in evaluation["tasks"]:
        task_config(task, evaluation)
    prepared = artifact["results"].setdefault("prepared", {
        "configuration_fingerprint": content_digest(preparation_settings(settings)),
        "harness_source_hashes": harness_files(), "corpora": {},
        "kl_protocol": "teacher-kl-t1-all-valid-positions-c4-shard0-24x8192-v1",
    })
    if prepared["harness_source_hashes"] != harness_files():
        raise ValueError("Harness source changed during preparation resume")
    for split in ("validation", "test"):
        if split not in prepared["corpora"]:
            rows = load_dataset(evaluation["wikitext_dataset"], evaluation["wikitext_name"],
                                revision=evaluation["wikitext_revision"], split=split)
            text = "\n\n".join(str(row.get("text") or "") for row in rows)
            values = np.asarray(tokenizer(text, add_special_tokens=False,
                                          return_attention_mask=False).input_ids, dtype=np.int32)
            path = directory / "prepared" / f"wikitext-{split}.int32"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            with temporary.open("wb") as output:
                values.tofile(output)
                output.flush()
                os.fsync(output.fileno())
            if path.exists():
                if file_digest(path) != file_digest(temporary):
                    raise ValueError("Interrupted preparation corpus differs")
                temporary.unlink()
            else:
                temporary.replace(path)
            prepared["corpora"][split] = {"path": str(path.relative_to(directory)),
                                          "token_count": len(values), "sha256": file_digest(path)}
            persist(directory, artifact, log, f"prepared:{split}")
    if "kl_stream" not in prepared:
        source = settings["data"]["kl_source"]
        path = directory / "prepared" / "kl.int32"
        record_path = path.with_suffix(".json")
        # Persist the stream descriptor separately before advancing the result stage.
        if record_path.exists():
            record = read_json(record_path)
        else:
            if path.exists():
                raise ValueError("Interrupted KL stream lacks its descriptor; inspect it before retry")
            records = load_dataset(source["path"], data_files=source["data_file"],
                                   revision=source["revision"], split=source["split"], streaming=True)
            record = build_finite_stream(records, tokenizer, path,
                                         settings["data"]["kl_sequences"] * settings["data"]["sequence_length"],
                                         text_column=source["text_column"])
            record["path"] = str(path.relative_to(directory))
            write_json_atomic(record_path, record)
        prepared["kl_stream"] = record
    for record in [*prepared["corpora"].values(), prepared["kl_stream"]]:
        verified_tokens(directory, record)
    prepared.pop("protocol_fingerprint", None)
    prepared["protocol_fingerprint"] = content_digest(prepared)
    persist(directory, artifact, log, "prepared")


def evaluate_online_kl(student, teacher, prepared_root, prepared, settings):
    import torch
    from mlp_replacement.compression.recovery import online_distillation_loss
    from mlp_replacement.data import PackedTokenCache

    record = prepared["kl_stream"]
    cache = PackedTokenCache(contained_path(prepared_root, record["path"]), record["token_count"],
                             settings["data"]["sequence_length"], record["sha256"])
    total = 0.0
    count = 0
    student.eval()
    teacher.eval()
    with torch.inference_mode():
        for offset in range(0, cache.token_count, cache.sequence_length):
            loss, tokens = online_distillation_loss(student, teacher,
                                                    cache.batch(offset, cache.sequence_length, 1),
                                                    1.0, "cuda", torch.bfloat16)
            total += float(loss.item()) * tokens
            count += tokens
    if count != record["token_count"] or not math.isfinite(total):
        raise ValueError("KL coverage/finite-loss contract failed")
    return {"kl": total / count, "valid_positions": count,
            "protocol": prepared["kl_protocol"], "stream_sha256": record["sha256"]}


def cpu_tree(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def recover_qat(student, teacher, tokenizer, paths, directory, work_dir, artifact, log,
                prepared_root, prepared, settings):
    """Adapt the established exact-token loop, not a second optimizer loop."""

    import torch
    from datasets import load_dataset
    from mlp_replacement.compression.continuation import (
        capture_rng, commit_single_checkpoint, recover_exact_segment, restore_checkpoint,
        restore_rng, segment_schedule,
    )
    from mlp_replacement.compression.quantization import prepare_int4_qat
    from mlp_replacement.data import PackedTokenCache
    from mlp_replacement.data_streams import build_finite_stream
    from mlp_replacement.evaluation.final_quality import evaluate_rolling_likelihood

    recovery = settings["qat"]
    budget = recovery["target_tokens"]
    if type(budget) is not int or budget < 1:
        raise ValueError("Selecting QAT requires an explicit positive token budget")
    persist(directory, artifact, log, "qat_preparation")
    # Two checkpoint generations briefly coexist during atomic replacement.
    # FP32 masters + two FP32 Adam moments use 12 bytes per eligible weight.
    eligible_count = sum(student.get_submodule(path).weight.numel() for path in paths)
    output_required = math.ceil((24 * eligible_count + 2 * sum(p.numel() for p in student.parameters()))
                                * (1 + recovery["disk_reserve_fraction"]))
    work_required = budget * 4
    shared_filesystem = directory.stat().st_dev == work_dir.stat().st_dev
    if (shutil.disk_usage(directory).free < output_required + (work_required if shared_filesystem else 0)
            or shutil.disk_usage(work_dir).free < work_required):
        raise OSError("Insufficient storage for the declared QAT stream and atomic checkpoints")
    source = settings["data"]["recovery_source"]
    stream_path = work_dir / "recovery.int32"
    descriptor = work_dir / "recovery.json"
    stream_contract = content_digest({"source": source, "budget": budget,
                                      "model": settings["model"], "sequence_length": 8192})
    if descriptor.exists():
        stream = read_json(descriptor)
        if stream["fingerprint"] != stream_contract:
            raise ValueError("Recovery stream configuration changed")
        verified_tokens(work_dir, stream)
    else:
        if stream_path.exists():
            raise ValueError("Recovery stream lacks its verified descriptor; inspect before retry")
        files = [f"en/c4-train.{index:05d}-of-01024.json.gz"
                 for index in range(source["first_shard"], source["shard_count"])]
        records = load_dataset(source["path"], revision=source["revision"],
                               data_files=files, split="train", streaming=True)
        stream = build_finite_stream(records, tokenizer, stream_path, budget, text_column=source["text_column"])
        stream["fingerprint"] = stream_contract
        write_json_atomic(descriptor, stream)
    prior = artifact["results"].get("recovery_stream")
    durable = {key: value for key, value in stream.items() if key != "path"}
    if prior is not None and prior != durable:
        raise ValueError("Regenerated recovery stream changed on resume")
    artifact["results"]["recovery_stream"] = durable
    parameters = prepare_int4_qat(student, paths)
    state = None
    checkpoints = directory / "checkpoints"
    if artifact["results"].get("recovery", {}).get("tokens_seen", 0) > 0 and not list(checkpoints.glob("checkpoint-*.json")):
        raise ValueError("Recorded QAT progress has no resumable checkpoint")
    if list(checkpoints.glob("checkpoint-*.json")):
        state, descriptor = restore_checkpoint(checkpoints, artifact["run_fingerprint"])
        if state["stream_sha256"] != stream["sha256"]:
            raise ValueError("Checkpoint recovery stream changed")
        if state.get("fake_quantization_enabled") is not True:
            raise ValueError("Checkpoint disabled the required fake quantization")
        for path in paths:
            student.get_submodule(path).load_state_dict(state["eligible_state"][path], strict=True)
        restore_rng(state)
    cursor = int(state["tokens_seen"]) if state else 0
    result = artifact["results"].setdefault("recovery", {"history": [], "tokens_seen": 0})
    if state:
        result.update(tokens_seen=cursor, optimizer_updates=state["optimizer_updates"],
                      elapsed_seconds=state["elapsed_seconds"], first_step=state["first_step"],
                      history=state["history"])
        result.setdefault("monitoring_seconds", state.get("monitoring_seconds", 0))
        result.setdefault("checkpoint_seconds", state.get("checkpoint_seconds", 0))
    cache = PackedTokenCache(stream_path, budget, 8192, stream["sha256"])
    validation = verified_tokens(prepared_root, prepared["corpora"]["validation"])

    def monitoring(tokens, requests):
        started = perf_counter()
        row = {"tokens_seen": tokens, "requested_tokens": list(requests),
               "kl": evaluate_online_kl(student, teacher, prepared_root, prepared, settings)}
        if tokens == 0 or budget in requests or set(requests) & set(recovery["monitoring_ppl_tokens"]):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                row["validation_likelihood_8192"] = evaluate_rolling_likelihood(
                    student, validation, 8192, 4096, "cuda")
        result["monitoring_seconds"] = result.get("monitoring_seconds", 0) + perf_counter() - started
        return row

    if cursor == 0 and not result["history"]:
        result["history"].append(monitoring(0, (0,)))
        persist(directory, artifact, log, "qat_initial_validation")

    def checkpoint(event, optimizer, first_step):
        row = monitoring(event.tokens_seen, event.requested_checkpoint_tokens)
        row["training"] = {"mean_kl": event.mean_train_kl, "mean_loss": event.mean_train_loss}
        result["history"].append(row)
        first_step = result.get("first_step") or first_step
        result.update(tokens_seen=event.tokens_seen, optimizer_updates=event.optimizer_updates,
                      elapsed_seconds=event.elapsed_seconds, first_step=first_step)
        payload = {"run_fingerprint": artifact["run_fingerprint"],
                   "stream_sha256": stream["sha256"],
                   "tokens_seen": event.tokens_seen, "optimizer_updates": event.optimizer_updates,
                   "elapsed_seconds": event.elapsed_seconds, "first_step": first_step,
                   "monitoring_seconds": result["monitoring_seconds"],
                   "checkpoint_seconds": result.get("checkpoint_seconds", 0),
                   "history": deepcopy(result["history"]),
                   "eligible_state": {path: cpu_tree(student.get_submodule(path).state_dict()) for path in paths},
                   "optimizer_state": cpu_tree(optimizer.state_dict()),
                   "fake_quantization_enabled": True, **capture_rng()}
        started = perf_counter()
        result["checkpoint"] = commit_single_checkpoint(checkpoints, payload)
        result["checkpoint_seconds"] = result.get("checkpoint_seconds", 0) + perf_counter() - started
        persist(directory, artifact, log, "qat_checkpoint")

    interval = recovery["validation_interval_tokens"]
    requests = [*range(interval, budget + 1, interval), *recovery["monitoring_ppl_tokens"], budget]
    schedule = segment_schedule(0, budget, requests, 8192)
    if cursor < budget:
        recovered = recover_exact_segment(
            origin=0, end=budget, cursor=cursor, schedule=schedule,
            batch_at=lambda offset, count: cache.batch(offset, count, 1, allow_partial_sequence=True),
            on_checkpoint=checkpoint, student=student, teacher=teacher,
            parameter_groups=[{"name": "eligible_mlp", "parameters": parameters,
                               "learning_rate": recovery["learning_rate"], "weight_decay": 0.0}],
            train_modules=[student.get_submodule(path) for path in paths], microbatch_tokens=8192,
            accumulation_steps=1, temperature=1.0, ce_weight=0.0, scheduler="constant",
            warmup_fraction=0.0, final_lr_ratio=1.0, device="cuda", autocast_dtype=torch.bfloat16,
            start_updates=state["optimizer_updates"] if state else 0,
            elapsed_seconds=state["elapsed_seconds"] if state else 0.0,
            optimizer_state=state["optimizer_state"] if state else None,
            optimizer_backend="fused",
        )
        result.update(tokens_seen=recovered.tokens_seen, optimizer_updates=recovered.optimizer_updates,
                      elapsed_seconds=recovered.elapsed_seconds)
    if result["tokens_seen"] != budget:
        raise ValueError("QAT did not reach its configured final endpoint")
    result["selection"] = "configured-final-endpoint"
    result["trainable_parameters"] = sum(value.numel() for value in parameters)
    result["unique_training_tokens"] = budget
    persist(directory, artifact, log, "qat_completed")


def build_variant_bundle(directory, work_dir, artifact, log, settings, prepared_root, prepared):
    import torch
    from mlp_replacement.compression.quantization import (
        apply_ptq, convert_int4_qat, frozen_state_hashes, projection_paths,
        quantization_footprint, recipe_record, require_backend, verify_frozen_state,
    )
    from mlp_replacement.evaluation.bundles import export_bundle
    from mlp_replacement.evaluation.quantized_bundles import export_quantized_bundle

    variant = artifact["identity"]["variant"]
    bits = 4 if variant.endswith("int4") else 8 if variant.endswith("int8") else None
    require_backend(settings["backend"], bits)
    persist(directory, artifact, log, "model_loading")
    model, tokenizer = load_teacher(settings)
    paths = projection_paths(model, settings["scope"]["eligible_layers"])
    if len(paths) != settings["scope"]["expected_projections"]:
        raise ValueError("Eligible projection count changed")
    before = quantization_footprint(model, paths)
    teacher = None
    started = perf_counter()
    try:
        if variant == "qat-int4":
            frozen = frozen_state_hashes(model, paths)
            teacher, unused_tokenizer = load_teacher(settings)
            recover_qat(model, teacher, tokenizer, paths, directory, work_dir, artifact, log,
                        prepared_root, prepared, settings)
            verify_frozen_state(model, paths, frozen)
            artifact["results"]["recovery_wall_seconds"] = perf_counter() - started
            started = perf_counter()
            artifact["results"]["conversion"] = convert_int4_qat(model, paths)
            verify_frozen_state(model, paths, frozen)
        elif bits:
            artifact["results"]["conversion"] = apply_ptq(model, paths, bits)
        artifact["results"]["conversion_seconds"] = perf_counter() - started
        artifact["results"]["quantization_recipe"] = recipe_record(bits) if bits else None
        provenance = {"workflow": "quantization-1", "run_fingerprint": artifact["run_fingerprint"],
                      "model": settings["model"], "scope": settings["scope"], "variant": variant,
                      "training_tokens": settings["qat"]["target_tokens"] if variant == "qat-int4" else 0}
        persist(directory, artifact, log, "bundle_export")
        started = perf_counter()
        if bits:
            manifest = export_quantized_bundle(model, tokenizer, directory / "model", paths, bits,
                                                settings["backend"], provenance)
        else:
            manifest = export_bundle(model, tokenizer, directory / "model", [], provenance)
        artifact["results"]["export_seconds"] = perf_counter() - started
        artifact["results"]["dense_footprint"] = before
        artifact["results"]["model"] = {"path": "model", "manifest_sha256": file_digest(directory / "model/bundle.json")}
        artifact["results"]["calibration_tokens"] = 0
        artifact["results"]["training_tokens"] = provenance["training_tokens"]
        persist(directory, artifact, log, "bundle_exported")
    finally:
        del model, teacher, tokenizer
        release_cuda()


def scoped_footprint(model, settings):
    from mlp_replacement.model import discover_mlp_blocks
    from mlp_replacement.compression.quantization import scoped_storage_footprint

    names = {f"{ref.path}.{name}" for ref in discover_mlp_blocks(model)
             if ref.index in settings["scope"]["eligible_layers"]
             for name, parameter in ref.module.named_parameters()}
    return scoped_storage_footprint(model, names)


def measure_runtime(directory, bundle, prepared_root, prepared, settings, artifact, log):
    if "runtime" in artifact["results"]:
        return
    persist(directory, artifact, log, "runtime_measurement")
    worker_output = directory / "runtime.json"
    command = [sys.executable, "-m", "workflows.runs.model.baseline.quantization", "measure-runtime",
               "--config", str(directory / "effective-config.json"), "--bundle", str(bundle),
               "--prepared", str(prepared_root / "result.json"), "--output-dir", str(directory)]
    process = subprocess.Popen(command, cwd=ROOT)
    try:
        if process.wait() != 0:
            error_path = directory / "runtime-error.json"
            detail = read_json(error_path)["message"] if error_path.exists() else "see worker stderr"
            raise RuntimeError(f"Fresh-process runtime measurement failed: {detail}")
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    runtime = read_json(worker_output)
    if runtime["bundle_manifest_sha256"] != file_digest(bundle / "bundle.json"):
        raise ValueError("Runtime measurement belongs to another model bundle")
    artifact["results"]["runtime"] = runtime
    persist(directory, artifact, log, "runtime_measured")


def evaluate_bundle(directory, bundle, prepared_root, prepared, settings, artifact, log):
    from mlp_replacement.evaluation.final_quality import evaluate_pinned_task, evaluate_rolling_likelihood
    from mlp_replacement.evaluation.quantized_bundles import load_inference_bundle

    measure_runtime(directory, bundle, prepared_root, prepared, settings, artifact, log)
    persist(directory, artifact, log, "quality_evaluation")
    started = perf_counter()
    model, tokenizer, manifest = load_inference_bundle(
        bundle, attention_implementation=settings["runtime"]["attention_implementation"])
    evaluation = artifact["results"].setdefault("evaluation", {
        "protocol_fingerprint": prepared["protocol_fingerprint"], "likelihood": {}, "tasks": {},
    })
    try:
        footprint = scoped_footprint(model, settings)
        if manifest.get("footprint") is not None and footprint != manifest["footprint"]:
            raise ValueError("Deployed model footprint differs from its bundle")
        footprint["tensor_file_bytes"] = manifest["tensor_file_bytes"]
        footprint["bundle_bytes"] = sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file())
        artifact["results"]["footprint"] = footprint
        artifact["results"].setdefault("quantization_recipe", manifest.get("quantization_recipe"))
        if model.config.max_position_embeddings < max(settings["evaluation"]["contexts"]):
            raise ValueError("Model does not support the frozen evaluation context")
        if "kl" not in evaluation:
            teacher, unused_tokenizer = load_teacher(settings)
            try:
                evaluation["kl"] = evaluate_online_kl(model, teacher, prepared_root, prepared, settings)
            finally:
                del teacher, unused_tokenizer
                release_cuda()
            persist(directory, artifact, log, "quality:kl")
        for split, record in prepared["corpora"].items():
            values = verified_tokens(prepared_root, record)
            for context, stride in zip(settings["evaluation"]["contexts"], settings["evaluation"]["strides"], strict=True):
                key = f"{split}-{context}"
                if key not in evaluation["likelihood"]:
                    evaluation["likelihood"][key] = evaluate_rolling_likelihood(model, values, context, stride, "cuda")
                    persist(directory, artifact, log, f"quality:{key}")
        for task in settings["evaluation"]["tasks"]:
            if task not in evaluation["tasks"]:
                task_result = evaluate_pinned_task(model, tokenizer, task, settings["evaluation"], task_config(task, settings["evaluation"]))
                path = directory / "evaluation" / f"{task}.json"
                write_json_atomic(path, task_result)
                evaluation["tasks"][task] = {"path": str(path.relative_to(directory)), "sha256": file_digest(path)}
                persist(directory, artifact, log, f"quality:{task}")
            record = evaluation["tasks"][task]
            if file_digest(contained_path(directory, record["path"])) != record["sha256"]:
                raise ValueError(f"Benchmark records changed: {task}")
        evaluation["status"] = "completed"
        persist(directory, artifact, log, "quality_completed")
    finally:
        artifact["results"]["evaluation_seconds"] = artifact["results"].get("evaluation_seconds", 0) + perf_counter() - started
        del model, tokenizer
        release_cuda()


def validate_reference(bundle, result_path, settings):
    """Require provenance absent from older BF16 bundle manifests."""

    manifest = read_json(bundle / "bundle.json")
    provenance = manifest["provenance"]
    if provenance.get("workflow") == "quantization-1":
        configuration = {"model": provenance["model"], "allocation": provenance["scope"]}
        tokens = provenance["training_tokens"]
        label = provenance["variant"]
        recovery_cost = {}
    else:
        source = read_json(result_path)
        configuration = source["configuration"]
        tokens = source["results"]["recovery"]["tokens_seen"]
        label = source["identity"]
        recovery_cost = {key: source["results"]["recovery"].get(key)
                         for key in ("training_seconds", "elapsed_seconds", "evaluation_seconds", "checkpoint_seconds")}
        if source["status"] != "completed" or provenance["run_fingerprint"] != source["run_fingerprint"]:
            raise ValueError("Reference result does not describe this completed bundle")
        if any(int(row["path"].split(".layers.")[1].split(".")[0]) in settings["scope"]["protected_layers"]
               for row in manifest.get("replacements", [])):
            raise ValueError("Reference changes a protected MLP topology")
    for key in ("model_id", "revision", "tokenizer_revision", "hidden_size", "num_layers"):
        if configuration["model"][key] != settings["model"][key]:
            raise ValueError(f"Reference model/tokenizer differs: {key}")
    for key in ("eligible_layers", "protected_layers"):
        if configuration["allocation"][key] != settings["scope"][key]:
            raise ValueError(f"Reference compression scope differs: {key}")
    target = source["identity"].get("target") if provenance.get("workflow") != "quantization-1" else None
    return {"label": label, "training_tokens": tokens, "target": target, "recovery_cost": recovery_cost}


def report(directory, artifact, log, settings, result_paths):
    from mlp_replacement.evaluation.final_quality import paired_accuracy_difference

    rows = []
    loaded = []
    for path in result_paths:
        path = resolve_path(path)
        run = read_json(path)
        if run["workflow"] != "quantization-1" or run["status"] != "completed" or "evaluation" not in run["results"]:
            raise ValueError(f"Report requires a completed Quantization-1 evaluation: {path}")
        loaded.append((path.parent, run))
    dense_runs = [(root, run) for root, run in loaded if run["identity"].get("variant") == "dense-bf16"]
    if len(dense_runs) != 1:
        raise ValueError("Reporting requires exactly one dense BF16 control")
    dense_root, dense = dense_runs[0]
    protocol = dense["results"]["evaluation"]["protocol_fingerprint"]
    dense_footprint = dense["results"]["footprint"]
    dense_runtime = dense["results"]["runtime"]["execution_fingerprint"]
    for root, run in loaded:
        result = run["results"]
        evaluation = result["evaluation"]
        if evaluation["protocol_fingerprint"] != protocol:
            raise ValueError("Report input quality protocols differ; reevaluate the reference")
        expected_likelihood = {f"{split}-{context}" for split in ("validation", "test")
                               for context in settings["evaluation"]["contexts"]}
        if evaluation.get("status") != "completed" or set(evaluation["likelihood"]) != expected_likelihood:
            raise ValueError("Report input lacks the complete final likelihood protocol")
        tasks = {}
        for task in settings["evaluation"]["tasks"]:
            records = []
            for base, source in ((dense_root, dense), (root, run)):
                record = source["results"]["evaluation"]["tasks"][task]
                path = contained_path(base, record["path"])
                if file_digest(path) != record["sha256"]:
                    raise ValueError(f"Report benchmark changed: {task}")
                records.append(read_json(path))
            metric = settings["evaluation"]["primary_metrics"][task]
            tasks[task] = {
                "accuracy": records[1]["results"][task][f"{metric},none"],
                **paired_accuracy_difference(records[0]["samples"][task], records[1]["samples"][task],
                                             metric, settings["evaluation"]["bootstrap_resamples"], settings["seed"]),
            }
        footprint = result["footprint"]
        name = run["identity"].get("variant") or str(run["identity"].get("label", "reference"))
        row = {"variant": name, "status": "completed", "result_sha256": file_digest(root / "result.json"),
               "quality_protocol": protocol, **footprint,
               "whole_model_parameter_removal": 1 - footprint["logical_parameters"] / dense_footprint["logical_parameters"],
               "eligible_parameter_removal": 1 - footprint["eligible_parameters"] / dense_footprint["eligible_parameters"],
               "whole_model_weight_byte_savings": 1 - footprint["weight_storage_bytes"] / dense_footprint["weight_storage_bytes"],
               "eligible_weight_byte_savings": 1 - footprint["eligible_weight_storage_bytes"] / dense_footprint["eligible_weight_storage_bytes"],
               "training_tokens": result["training_tokens"], "calibration_tokens": result.get("calibration_tokens"),
               "test_ppl_8192": evaluation["likelihood"]["test-8192"]["perplexity"],
               "likelihood": evaluation["likelihood"], "kl": evaluation["kl"],
               "kl_8192": evaluation["kl"]["kl"], "tasks": tasks,
               "macro_accuracy": sum(item["accuracy"] for item in tasks.values()) / len(tasks),
               "runtime_comparable_to_dense": result["runtime"]["execution_fingerprint"] == dense_runtime,
               "runtime": result["runtime"],
               "quantization_recipe": result.get("quantization_recipe"),
               "resources": result.get("resources"),
               "conversion_seconds": result.get("conversion_seconds"),
               "export_seconds": result.get("export_seconds"), "evaluation_seconds": result.get("evaluation_seconds"),
               "recovery_seconds": (result.get("recovery", {}).get("elapsed_seconds")
                                    if result["training_tokens"] else 0.0),
               "recovery_monitoring_seconds": result.get("recovery", {}).get("monitoring_seconds"),
               "recovery_checkpoint_seconds": result.get("recovery", {}).get("checkpoint_seconds"),
               "allocation_gpu_hours": None,
               "workflow_gpu_hours_proxy": read_json(root / "run.json").get("workflow_gpu_hours_proxy"),
               "same_training_budget_as_dense": result["training_tokens"] == 0}
        rows.append(row)
    supplied_targets = {run["identity"].get("reference", {}).get("target") for root, run in loaded}
    for reference in settings["references"]:
        if reference["target"] not in supplied_targets:
            rows.append({"variant": reference["label"], "status": "pending-compatible-bundle-evaluation", "target": reference["target"]})
    artifact["results"]["comparison"] = {"rows": rows, "notes": [
        "INT8 nominally matches 50% eligible MLP weight-byte removal before metadata.",
        "INT4 is one nominal 75% point; 70% and 80% are neighbouring budgets, not exact matches.",
        "Training budgets are explicit. PTQ/structural comparisons need not have equal recovery cost.",
        "Speed comparisons require matching execution fingerprints; all raw measurements remain available.",
        "Allocation accounting is Unknown unless scheduler evidence is supplied; workflow timing is a proxy.",
    ]}
    write_json_atomic(directory / "comparison.json", artifact["results"]["comparison"])
    columns = ["variant", "status", "logical_parameters", "eligible_parameters",
               "whole_model_parameter_removal", "eligible_parameter_removal",
               "weight_storage_bytes", "eligible_weight_storage_bytes", "eligible_weight_byte_savings",
               "whole_model_weight_byte_savings", "tensor_file_bytes", "bundle_bytes", "kl_8192",
               "macro_accuracy", "training_tokens", "runtime_comparable_to_dense",
               "conversion_seconds", "export_seconds", "evaluation_seconds", "recovery_seconds",
               "workflow_gpu_hours_proxy", "resident_gpu_bytes", "resident_rss_bytes"]
    for split in ("validation", "test"):
        columns += [f"{split}_ppl_{context}" for context in settings["evaluation"]["contexts"]]
    for task in settings["evaluation"]["tasks"]:
        columns += [f"{task}_accuracy", f"{task}_delta", f"{task}_ci_low", f"{task}_ci_high"]
    speed_metrics = ("time_to_first_token_ms", "mean_subsequent_token_latency_ms",
                     "generation_latency_ms", "output_tokens_per_second", "peak_gpu_allocated_bytes")
    for workload in settings["runtime"]["workloads"]:
        prefix = f"b{workload['batch_size']}_p{workload['prompt_tokens']}"
        columns += [f"{prefix}_{metric}" for metric in speed_metrics]
    csv_rows = []
    for row in rows:
        flattened = dict(row)
        for key, measurement in row.get("likelihood", {}).items():
            split, context = key.split("-")
            flattened[f"{split}_ppl_{context}"] = measurement["perplexity"]
        for task, measurement in row.get("tasks", {}).items():
            flattened.update({f"{task}_accuracy": measurement["accuracy"],
                              f"{task}_delta": measurement["student_minus_dense"],
                              f"{task}_ci_low": measurement["ci95"][0],
                              f"{task}_ci_high": measurement["ci95"][1]})
        resident = row.get("runtime", {}).get("resident_memory", {})
        flattened.update(resident_gpu_bytes=resident.get("gpu_allocated_delta_bytes"),
                         resident_rss_bytes=resident.get("host_rss_delta_bytes"))
        for workload in row.get("runtime", {}).get("generation", {}).get("workloads", []):
            prefix = f"b{workload['batch_size']}_p{workload['prompt_tokens']}"
            flattened.update({f"{prefix}_{metric}": workload["median"][metric] for metric in speed_metrics})
            flattened[f"{prefix}_peak_gpu_allocated_bytes"] = workload["peak_gpu_allocated_bytes"]
        csv_rows.append(flattened)
    temporary = directory / "comparison.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(directory / "comparison.csv")
    lines = ["# Quantization-1 comparison", "", "| Variant | Status | Weight bytes | 8K test PPL | Task macro | Training tokens |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "—")) for key in
                                         ("variant", "status", "weight_storage_bytes", "test_ppl_8192", "macro_accuracy", "training_tokens")) + " |")
    lines += ["", "## Full-corpus perplexity", "",
              "| Variant | Split | Context 128 | Context 2048 | Context 8192 |",
              "| --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        if row["status"] == "completed":
            for split in ("validation", "test"):
                values = [f"{row['likelihood'][f'{split}-{context}']['perplexity']:.4f}" for context in (128, 2048, 8192)]
                lines.append(f"| {row['variant']} | {split} | " + " | ".join(values) + " |")
    lines += ["", "## Paired downstream accuracy", "",
              "Accuracy and deltas use fractions; confidence intervals are paired example bootstraps.", "",
              "| Variant | Task | Accuracy | Student minus dense | 95% interval |",
              "| --- | --- | ---: | ---: | --- |"]
    for row in rows:
        for task, measurement in row.get("tasks", {}).items():
            low, high = measurement["ci95"]
            lines.append(f"| {row['variant']} | {task} | {measurement['accuracy']:.4f} | "
                         f"{measurement['student_minus_dense']:.4f} | [{low:.4f}, {high:.4f}] |")
    lines += ["", "## Storage and resident memory", "",
              "| Variant | Eligible weight bytes | Whole-model byte savings | Weight-file bytes | Bundle bytes | Resident GPU bytes | Resident RSS bytes |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in csv_rows:
        if row["status"] == "completed":
            lines.append("| " + " | ".join(str(row.get(key, "—")) for key in
                         ("variant", "eligible_weight_storage_bytes", "whole_model_weight_byte_savings",
                          "tensor_file_bytes", "bundle_bytes", "resident_gpu_bytes", "resident_rss_bytes")) + " |")
    lines += ["", "## Recovery and evaluation cost", "",
              "| Variant | Training tokens | Recovery seconds | Conversion seconds | Export seconds | Evaluation seconds | Workflow GPU-hours proxy |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        if row["status"] == "completed":
            lines.append("| " + " | ".join(str(row.get(key, "—")) for key in
                         ("variant", "training_tokens", "recovery_seconds", "conversion_seconds",
                          "export_seconds", "evaluation_seconds", "workflow_gpu_hours_proxy")) + " |")
    lines += ["", "## Generation measurements (medians)", "",
              "P95 and individual timings are retained in JSON. Peak VRAM is the maximum across measured requests.", "",
              "| Variant | Batch | Prompt | TTFT ms | Mean token ms | Total ms | Tokens/s | Peak VRAM bytes | Same execution |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        for workload in row.get("runtime", {}).get("generation", {}).get("workloads", []):
            metrics = workload["median"]
            lines.append(f"| {row['variant']} | {workload['batch_size']} | {workload['prompt_tokens']} | "
                         f"{metrics['time_to_first_token_ms']:.3f} | {metrics['mean_subsequent_token_latency_ms']:.3f} | "
                         f"{metrics['generation_latency_ms']:.3f} | {metrics['output_tokens_per_second']:.3f} | "
                         f"{workload['peak_gpu_allocated_bytes']} | "
                         f"{row['runtime_comparable_to_dense']} |")
    lines += ["", *artifact["results"]["comparison"]["notes"], ""]
    temporary = directory / "comparison.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(directory / "comparison.md")
    persist(directory, artifact, log, "report_completed")


def runtime_worker(args, settings):
    import torch
    from mlp_replacement.evaluation.generation import measure_bundle_runtime
    from mlp_replacement.runlog import environment_record, installed_version
    from mlp_replacement.compression.quantization import require_backend

    require_backend(settings["backend"], None)
    root, prepared = load_prepared(args.prepared, settings)
    values = verified_tokens(root, prepared["corpora"]["validation"])
    result = measure_bundle_runtime(resolve_path(args.bundle), values, settings["runtime"])
    environment = environment_record()
    environment["packages"].update({package: installed_version(package)
                                    for package in ("torchao", "mslk", "triton", "accelerate", "safetensors")})
    properties = torch.cuda.get_device_properties(0)
    execution = {"environment": environment, "runtime": settings["runtime"],
                 "device_uuid": str(getattr(properties, "uuid", "unavailable")),
                 "compute_capability": [properties.major, properties.minor],
                 "prompt_sha256": prepared["corpora"]["validation"]["sha256"],
                 "measurement_source_hashes": {name: file_digest(ROOT / "src/mlp_replacement/evaluation" / name)
                                                for name in ("generation.py", "quantized_bundles.py", "bundles.py")}}
    result.update(execution=execution, execution_fingerprint=content_digest(execution),
                  bundle_manifest_sha256=file_digest(resolve_path(args.bundle) / "bundle.json"))
    write_json_atomic(resolve_path(args.output_dir) / "runtime.json", result)


def resource_record():
    import psutil
    import torch

    record = {"host_rss_bytes": psutil.Process().memory_info().rss,
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
              "scheduler_allocation_gpu_hours": None, "mean_gpu_utilization": None}
    if sys.platform != "win32":
        import resource
        record["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return record


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "evaluate", "report", "measure-runtime"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-dir")
    parser.add_argument("--prepared")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--qat-tokens", type=int)
    parser.add_argument("--bundle")
    parser.add_argument("--reference-result")
    parser.add_argument("--label", default="architectural-reference")
    parser.add_argument("--results", nargs="+")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = load_settings(args.config, args.qat_tokens)
    directory = resolve_path(args.output_dir)
    if args.command == "measure-runtime":
        try:
            runtime_worker(args, settings)
        except BaseException as error:
            write_json_atomic(directory / "runtime-error.json", {
                "type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(), "timestamp": utc_now(),
            })
            raise
        return
    if args.command in ("run", "evaluate") and not args.prepared:
        raise ValueError("Run/evaluate requires --prepared RESULT_JSON")
    if args.command == "run" and args.variant is None:
        raise ValueError("Run requires one explicit --variant")
    if args.variant == "qat-int4" and settings["qat"]["target_tokens"] is None:
        raise ValueError("QAT is optional and requires --qat-tokens or a configured target_tokens")
    if args.command == "evaluate" and not args.bundle:
        raise ValueError("Evaluate requires a deployment --bundle")
    if args.command == "report" and not args.results:
        raise ValueError("Report requires completed --results, including dense BF16")
    identity = {}
    if args.command in ("run", "evaluate"):
        prepared_root, prepared = load_prepared(args.prepared, settings)
        identity["protocol_fingerprint"] = prepared["protocol_fingerprint"]
        if directory == prepared_root or directory.is_relative_to(prepared_root):
            raise ValueError("Run output cannot be inside immutable prepared inputs")
    if args.command == "run":
        identity.update(variant=args.variant, qat_tokens=settings["qat"]["target_tokens"] if args.variant == "qat-int4" else 0)
    if args.command == "evaluate":
        bundle = resolve_path(args.bundle)
        if directory == bundle or directory.is_relative_to(bundle):
            raise ValueError("Evaluation output cannot be inside an immutable model bundle")
        reference = validate_reference(bundle, resolve_path(args.reference_result) if args.reference_result else bundle.parent / "result.json", settings)
        identity.update(label=args.label, bundle_sha256=file_digest(bundle / "bundle.json"), reference=reference)
    if args.command == "report":
        identity["results"] = [file_digest(resolve_path(path)) for path in args.results]
    artifact, log = start_artifact(directory, settings, args.command, identity, args.resume)
    started = perf_counter()

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}; preserving committed stages/checkpoint")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        seed_run(settings["seed"])
        work_dir = resolve_path(args.work_dir) if args.work_dir else directory.parent / f"{directory.name}-work"
        if directory.is_relative_to(work_dir) or work_dir.is_relative_to(directory):
            raise ValueError("Disposable work and durable output directories must be separate")
        if args.command in ("run", "evaluate") and (work_dir == prepared_root or work_dir.is_relative_to(prepared_root)):
            raise ValueError("Disposable work cannot be inside immutable prepared inputs")
        if args.command == "evaluate" and (work_dir == bundle or work_dir.is_relative_to(bundle)):
            raise ValueError("Disposable work cannot be inside an immutable model bundle")
        work_dir.mkdir(parents=True, exist_ok=True)
        if args.command == "prepare":
            prepare(directory, artifact, log, settings)
        elif args.command == "run":
            bundle = directory / "model"
            if not bundle.exists():
                staging = directory / "model.building"
                if staging.exists() and args.resume:
                    if staging.is_symlink() or staging.resolve().parent != directory.resolve():
                        raise ValueError("Incomplete bundle staging escaped the owned output")
                    shutil.rmtree(staging)
                build_variant_bundle(directory, work_dir, artifact, log, settings, prepared_root, prepared)
            else:
                manifest = read_json(bundle / "bundle.json")
                if manifest["provenance"]["run_fingerprint"] != artifact["run_fingerprint"]:
                    raise ValueError("Existing bundle belongs to another run")
                artifact["results"].setdefault("training_tokens", manifest["provenance"]["training_tokens"])
            evaluate_bundle(directory, bundle, prepared_root, prepared, settings, artifact, log)
        elif args.command == "evaluate":
            artifact["results"]["training_tokens"] = reference["training_tokens"]
            costs = reference["recovery_cost"]
            artifact["results"]["recovery"] = {
                "elapsed_seconds": costs.get("training_seconds") or costs.get("elapsed_seconds"),
                "monitoring_seconds": costs.get("evaluation_seconds"),
                "checkpoint_seconds": costs.get("checkpoint_seconds"),
                "provenance": "supplied-original-result",
            }
            artifact["results"]["source_bundle"] = {"manifest_sha256": identity["bundle_sha256"]}
            evaluate_bundle(directory, bundle, prepared_root, prepared, settings, artifact, log)
        else:
            report(directory, artifact, log, settings, args.results)
        elapsed = perf_counter() - started
        artifact["results"]["resources"] = resource_record()
        log.data["workflow_seconds"] = log.data.get("workflow_seconds", 0) + elapsed
        log.data["workflow_gpu_hours_proxy"] = log.data["workflow_seconds"] / 3600 if args.command != "report" else None
        log.data["scheduler_allocation_gpu_hours"] = None
        finish(directory, artifact, log)
        if args.command == "run" and (directory / "checkpoints").exists():
            for path in (directory / "checkpoints").glob("checkpoint-*"):
                path.unlink()
        print(directory / "result.json", flush=True)
    except BaseException as error:
        artifact.update(status="failed", error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        write_json_atomic(directory / "result.json", artifact)
        log.data["workflow_seconds"] = log.data.get("workflow_seconds", 0) + perf_counter() - started
        log.data["workflow_gpu_hours_proxy"] = log.data["workflow_seconds"] / 3600 if args.command != "report" else None
        log.fail(error)
        raise


if __name__ == "__main__":
    main()
