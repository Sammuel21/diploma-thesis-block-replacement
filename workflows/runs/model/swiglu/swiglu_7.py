"""Run the SwiGLU-7 retraining-scope analysis and production evaluation."""

from __future__ import annotations

import argparse
import gc
import math
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from workflows.runs.model.common import PROJECT_ROOT, resolve_path

from mlp_replacement.artifacts import (
    contained_path,
    content_digest,
    file_digest,
    read_json,
    write_json_atomic,
)
from mlp_replacement.compression.adapters import (
    install_lora_adapters,
    lora_parameters,
    merge_lora_adapters,
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
    replacement_state,
)
from mlp_replacement.compression.recovery import cache_teacher_logits
from mlp_replacement.data import PackedTokenCache
from mlp_replacement.data_streams import build_finite_stream
from mlp_replacement.evaluation.bundles import (
    export_bundle,
    load_bundle,
    measure_resident_bundle,
    validate_bundle,
)
from mlp_replacement.evaluation.final_quality import (
    evaluate_pinned_task,
    evaluate_rolling_likelihood,
    paired_accuracy_difference,
)
from mlp_replacement.model import (
    discover_mlp_blocks,
    discover_transformer_layers,
    load_model_and_tokenizer,
)
from mlp_replacement.runlog import environment_record

from .shared import (
    build_local_data,
    evaluate_lm_mixed,
    evaluate_validation_kl_mixed,
    make_model_config,
    recovery_memory_record,
)


WORKFLOW = "swiglu-7"
SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path("workflows/configs/model/swiglu/swiglu-7.json")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def code_hashes():
    paths = [
        *PROJECT_ROOT.joinpath("src/mlp_replacement").rglob("*.py"),
        *PROJECT_ROOT.joinpath("workflows/runs").rglob("*.py"),
    ]
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): file_digest(path)
        for path in sorted(paths)
    }


def load_settings(path):
    path = resolve_path(Path(path))
    settings = read_json(path)
    if settings.get("schema_version") != SCHEMA_VERSION or settings.get("workflow") != WORKFLOW:
        raise ValueError("Expected a schema-1 SwiGLU-7 configuration")
    if settings.get("seed") != 21 or settings.get("targets") != [0.2, 0.3, 0.4, 0.5]:
        raise ValueError("SwiGLU-7 fixes seed 21 and the 20%-50% target grid")
    expected_scopes = {
        "S7-0": "replacement_only",
        "S7-1": "full_transformer_body",
        "S7-2": "full_mlp_attention_lora",
    }
    if {
        key: value.get("trainable_scope")
        for key, value in settings.get("strategies", {}).items()
    } != expected_scopes:
        raise ValueError("SwiGLU-7 requires the three fixed retraining scopes")
    lora = settings["strategies"]["S7-2"].get("lora", {})
    if (
        lora.get("rank") != 16
        or lora.get("alpha") != 32
        or lora.get("dropout") != 0.0
        or lora.get("attention_projections") != ["q_proj", "k_proj", "v_proj", "o_proj"]
    ):
        raise ValueError("S7-2 fixes rank-16, alpha-32 attention LoRA with zero dropout")
    recovery = settings["recovery"]
    fixed = {
        "target_tokens": 1_000_000_000,
        "segment_endpoints": [100_000_000, 1_000_000_000],
        "checkpoint_interval_tokens": 25_000_000,
        "validation_interval_tokens": 25_000_000,
        "monitoring_ppl_tokens": [
            10_000_000,
            25_000_000,
            50_000_000,
            100_000_000,
            250_000_000,
            500_000_000,
            1_000_000_000,
        ],
        "optimizer": "AdamW",
        "optimizer_backend": "fused",
        "learning_rate": 3e-5,
        "weight_decay": 0.0,
        "temperature": 1.0,
        "ce_weight": 0.0,
        "scheduler": "constant",
        "warmup_fraction": 0.0,
        "final_lr_ratio": 1.0,
        "sequence_length": 8192,
        "microbatch_sequences": 1,
        "gradient_accumulation_steps": 1,
        "effective_batch_tokens": 8192,
        "forward_autocast_dtype": "bfloat16",
        "trainable_parameter_dtype": "float32",
        "optimizer_state_dtype": "float32",
        "disk_reserve_fraction": 0.15,
    }
    if any(recovery.get(key) != value for key, value in fixed.items()):
        raise ValueError("SwiGLU-7 fixes the established recovery recipe")
    if "sources" in settings:
        raise ValueError("SwiGLU-7 must not depend on prior experiment artifacts")
    if settings["preparation"] != {
        "recipe_id": "S7-A0",
        "initialization": "legacy_subset",
        "allocation_method": "discrete_width_curve",
        "start_tokens": 0,
    }:
        raise ValueError("SwiGLU-7 fixes one fresh discrete-allocation recipe")
    data = settings["data"]
    if (
        data.get("sequence_length") != 128
        or data.get("capture_batch_size") != 2
        or data.get("local_source", {}).get("revision") is None
        or data.get("recovery_source", {}).get("revision") is None
        or data.get("model_validation_source", {}).get("revision") is None
        or data.get("recovery_source", {}).get("first_shard") != 1
        or data.get("recovery_source", {}).get("shard_count") != 1024
    ):
        raise ValueError("SwiGLU-7 requires pinned, independently prepared data")
    if settings.get("calibration") != {
        "selected_pairs": 393_216,
        "pair_counts": [393_216],
    }:
        raise ValueError("SwiGLU-7 fixes the local calibration budget")
    allocation = settings["allocation"]
    if (
        allocation.get("eligible_layers") != list(range(1, 23))
        or allocation.get("protected_layers") != [0, 23]
        or allocation.get("width_ratios")
        != [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
        or allocation.get("selection_metric") != "singleton_fixed_teacher_kl_t1"
        or allocation.get("allow_dense_layers") is not True
        or allocation.get("boundary_width_policy")
        != "best_interpolated_marginal_gain"
    ):
        raise ValueError("SwiGLU-7 fixes discrete singleton-KL width allocation")
    evaluation = settings["evaluation"]
    task_datasets = evaluation.get("task_datasets", {})
    if (
        evaluation.get("contexts") != [128, 2048, 8192]
        or evaluation.get("strides") != [64, 1024, 4096]
        or evaluation.get("benchmark_context_length") != 2048
        or evaluation.get("tasks")
        != ["piqa", "arc_easy", "arc_challenge", "winogrande", "hellaswag"]
        or evaluation.get("seed") != settings["seed"]
        or evaluation.get("wikitext_dataset")
        != data["model_validation_source"]["path"]
        or evaluation.get("wikitext_name")
        != data["model_validation_source"]["name"]
        or evaluation.get("wikitext_revision")
        != data["model_validation_source"]["revision"]
        or set(task_datasets) != set(evaluation.get("tasks", []))
        or any(
            not row.get("dataset")
            or not row.get("revision")
            or row.get("split") not in {"validation", "test"}
            for row in task_datasets.values()
        )
    ):
        raise ValueError("SwiGLU-7 fixes native-context likelihood and frozen task evaluation")
    return settings, path


def resolve_storage(work_dir, output_dir, resume):
    work = resolve_path(Path(work_dir)).resolve()
    output = resolve_path(Path(output_dir)).resolve()
    if work == output or work in output.parents or output in work.parents:
        raise ValueError("--work-dir and --output-dir must be separate, non-nested directories")
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"Work directory must be fresh and empty: {work}")
    work.mkdir(parents=True, exist_ok=True)
    if resume:
        if not (output / "result.json").is_file():
            raise FileNotFoundError("--resume requires an existing output-dir/result.json")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to use non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return work, output


def output_path(output_dir, relative):
    return contained_path(output_dir, relative)


def relative_output(output_dir, path):
    return Path(path).resolve().relative_to(Path(output_dir).resolve()).as_posix()


def persist(output_dir, artifact, stage, progress=None):
    artifact["updated_at_utc"] = utc_now()
    write_json_atomic(output_dir / "result.json", artifact)
    write_json_atomic(
        output_dir / "run.json",
        {
            "schema_version": SCHEMA_VERSION,
            "workflow": WORKFLOW,
            "command": artifact["command"],
            "status": artifact["status"],
            "stage": stage,
            "configuration": artifact["configuration"],
            "environment": artifact.get("environment"),
            "progress": progress,
            "run_fingerprint": artifact.get("run_fingerprint"),
            "updated_at_utc": artifact["updated_at_utc"],
            "error": artifact.get("error"),
        },
    )


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
            raise ValueError("Resume output belongs to a different SwiGLU-7 run")
        observed_environment = environment_record()
        if artifact.get("status") != "completed" and artifact.get("environment") != observed_environment:
            raise ValueError("Resume environment differs from the recorded SwiGLU-7 run")
        if artifact.get("status") != "completed":
            artifact["status"] = "running"
            artifact["error"] = None
        return artifact
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "experiment_family": "swiglu-7",
        "experiment_class": "retraining-scope-analysis",
        "command": command,
        "identity": identity,
        "status": "running",
        "created_at_utc": utc_now(),
        "configuration": deepcopy(settings),
        "environment": environment_record(),
        "results": {},
        "error": None,
    }


def effective_preparation_settings(settings):
    """Adapt the explicit S7 configuration to the maintained fitting helpers."""

    effective = deepcopy(settings)
    effective["references"] = {
        "selected_calibration_pairs": settings["calibration"]["selected_pairs"]
    }
    effective["compatibility"] = {
        "eligible_layers": deepcopy(settings["allocation"]["eligible_layers"]),
        "protected_layers": deepcopy(settings["allocation"]["protected_layers"]),
    }
    return effective


def validated_attention_implementation(model):
    implementation = getattr(model.config, "_attn_implementation", None)
    if implementation not in {"sdpa", "flash_attention_2", "flash_attention_3"}:
        raise RuntimeError(
            "SwiGLU-7 native-8K execution requires SDPA or FlashAttention; "
            f"the loaded model selected {implementation!r}"
        )
    return implementation


def preparation_storage_preflight(work_dir, output_dir, settings):
    hidden = int(settings["model"]["hidden_size"])
    width = int(settings["model"]["intermediate_size"])
    layers = len(settings["allocation"]["eligible_layers"])
    original_parameters = layers * 3 * hidden * width
    start_bytes = sum(
        int(original_parameters * (1.0 - float(target))) * 4
        for target in settings["targets"]
    )
    curve_widths = {
        min(width, max(1, round(width * float(ratio))))
        for ratio in settings["allocation"]["width_ratios"]
        if float(ratio) < 1.0
    }
    curve_bytes = layers * sum(3 * hidden * value * 4 for value in curve_widths)
    output_required = int(
        (
            int(settings["recovery"]["target_tokens"]) * 4
            + start_bytes
            + 2_000_000_000
        )
        * 1.15
    )
    work_required = int((curve_bytes + 2_000_000_000) * 1.15)
    output_free = shutil.disk_usage(output_dir).free
    work_free = shutil.disk_usage(work_dir).free
    same_device = work_dir.stat().st_dev == output_dir.stat().st_dev
    if same_device:
        if min(output_free, work_free) < output_required + work_required:
            raise RuntimeError(
                "SwiGLU-7 preparation requires at least "
                f"{output_required + work_required:,} free bytes on shared storage"
            )
    elif output_free < output_required or work_free < work_required:
        raise RuntimeError(
            "SwiGLU-7 preparation lacks output or temporary storage capacity"
        )
    return {
        "estimated_output_bytes": output_required,
        "estimated_work_bytes": work_required,
        "observed_output_free_bytes": output_free,
        "observed_work_free_bytes": work_free,
        "shared_filesystem": same_device,
    }


def local_preparation_data(settings, tokenizer):
    context = SimpleNamespace(
        settings=effective_preparation_settings(settings),
        tokenizer=tokenizer,
    )
    return build_local_data(context)


def prepare_monitoring_evaluation(settings, output_dir, artifact, tokenizer):
    import torch

    existing = artifact["results"].get("monitoring_evaluation")
    if existing is not None:
        path = output_path(output_dir, existing["path"])
        if file_digest(path) != existing["sha256"]:
            raise ValueError("Prepared local evaluation batches changed")
        return None
    local_data = local_preparation_data(settings, tokenizer)
    path = output_dir / "prepared" / "local-evaluation-batches.pt"
    save_tensor_atomic(
        path,
        {
            key: list(local_data[key])
            for key in ("recovery_validation", "model_validation")
        },
    )
    artifact["results"]["monitoring_evaluation"] = {
        "path": relative_output(output_dir, path),
        "sha256": file_digest(path),
        "sequence_length": int(settings["data"]["sequence_length"]),
        "c4_revision": settings["data"]["local_source"]["revision"],
        "wikitext_revision": settings["data"]["model_validation_source"]["revision"],
    }
    persist(output_dir, artifact, "local_evaluation_prepared")
    return local_data


def prepare_recovery_stream(settings, output_dir, artifact, tokenizer):
    path = output_dir / "prepared" / "recovery-tokens.int32"
    existing = artifact["results"].get("stream")
    if existing is not None:
        path = output_path(output_dir, existing["path"])
        if (
            file_digest(path) != existing["sha256"]
            or path.stat().st_size != 4 * int(existing["token_count"])
        ):
            raise ValueError("Prepared recovery stream changed")
        return existing

    # The file rename and result.json commit are separate atomic operations.
    # A terminated process can therefore leave an unrecorded complete-looking
    # stream; it has no durable provenance and must be regenerated on resume.
    path.unlink(missing_ok=True)

    from datasets import load_dataset
    from huggingface_hub import HfApi

    source = settings["data"]["recovery_source"]
    info = HfApi().dataset_info(
        source["path"],
        revision=source["revision"],
        files_metadata=True,
    )
    if info.sha != source["revision"]:
        raise ValueError("Resolved C4 revision differs from the pinned revision")
    inventory = {item.rfilename: item for item in info.siblings}
    used_shards = []

    def records():
        for index in range(int(source["first_shard"]), int(source["shard_count"])):
            name = f"en/c4-train.{index:05d}-of-{int(source['shard_count']):05d}.json.gz"
            if name not in inventory:
                raise FileNotFoundError(f"Pinned C4 shard is missing: {name}")
            entry = inventory[name]
            lfs = entry.lfs
            used_shards.append(
                {
                    "path": name,
                    "git_blob": entry.blob_id,
                    "sha256": (
                        lfs.get("sha256")
                        if isinstance(lfs, dict)
                        else getattr(lfs, "sha256", None)
                    ),
                }
            )
            yield from load_dataset(
                source["path"],
                revision=source["revision"],
                data_files={"train": name},
                split="train",
                streaming=True,
            )

    def progress(tokens, documents):
        persist(
            output_dir,
            artifact,
            "recovery_stream",
            {"tokens": tokens, "documents": documents},
        )

    record = build_finite_stream(
        records(),
        tokenizer,
        path,
        settings["recovery"]["target_tokens"],
        text_column=source["text_column"],
        on_progress=progress,
    )
    record.update(
        {
            "path": relative_output(output_dir, path),
            "sequence_length": settings["recovery"]["sequence_length"],
            "dataset": source["path"],
            "dataset_revision": source["revision"],
            "source_shards": used_shards,
        }
    )
    artifact["results"]["stream"] = record
    persist(output_dir, artifact, "recovery_stream_prepared")
    return record


def write_int32_corpus(path, values):
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype=np.int32)
    if path.exists():
        if path.stat().st_size != array.nbytes:
            raise ValueError(f"Prepared corpus extent differs: {path}")
        observed = np.memmap(path, mode="r", dtype=np.int32)
        if not np.array_equal(observed, array):
            raise ValueError(f"Prepared corpus contents differ: {path}")
    else:
        temporary = path.with_name(path.name + ".tmp")
        temporary.unlink(missing_ok=True)
        array.tofile(temporary)
        temporary.replace(path)
    return {
        "path": path,
        "sha256": file_digest(path),
        "token_count": int(array.size),
    }


def prepare_evaluation_protocol(settings, output_dir, artifact, tokenizer):
    existing = artifact["results"].get("evaluation_protocol")
    if existing is not None:
        for record in existing["corpora"].values():
            path = output_path(output_dir, record["path"])
            if (
                file_digest(path) != record["sha256"]
                or path.stat().st_size != 4 * int(record["token_count"])
            ):
                raise ValueError("Prepared evaluation corpus changed")
        return existing

    if version("lm_eval") != settings["evaluation"]["harness_version"]:
        raise ValueError("Install the pinned evaluation harness before preparation")

    import numpy as np
    from datasets import load_dataset
    from huggingface_hub import HfApi

    evaluation = settings["evaluation"]
    tasks = {}
    for task in evaluation["tasks"]:
        config = native_task_config(task)
        pinned = evaluation["task_datasets"][task]
        split = config.get("test_split") or config["validation_split"]
        if config["dataset_path"] != pinned["dataset"] or split != pinned["split"]:
            raise ValueError(f"Pinned benchmark definition changed: {task}")
        resolved = HfApi().dataset_info(
            pinned["dataset"],
            revision=pinned["revision"],
        ).sha
        if resolved != pinned["revision"]:
            raise ValueError(f"Resolved benchmark revision changed: {task}")
        tasks[task] = deepcopy(pinned)
    corpora = {}
    for split in ("validation", "test"):
        rows = load_dataset(
            evaluation["wikitext_dataset"],
            evaluation["wikitext_name"],
            revision=evaluation["wikitext_revision"],
            split=split,
        )
        text = "\n\n".join(str(row.get("text") or "") for row in rows)
        ids = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        ).input_ids
        path = output_dir / "prepared" / "evaluation" / f"wikitext-{split}.int32"
        record = write_int32_corpus(path, np.asarray(ids, dtype=np.int32))
        corpora[split] = {
            **record,
            "path": relative_output(output_dir, record["path"]),
        }
    protocol = {
        "harness_version": evaluation["harness_version"],
        "harness_source_hashes": task_files(),
        "tasks": tasks,
        "wikitext_revision": evaluation["wikitext_revision"],
        "corpora": corpora,
    }
    protocol["protocol_fingerprint"] = content_digest(protocol)
    artifact["results"]["evaluation_protocol"] = protocol
    persist(output_dir, artifact, "evaluation_protocol_prepared")
    return protocol


def preparation_context(settings, work_dir, output_dir, artifact, local_data=None):
    import torch

    effective = effective_preparation_settings(settings)
    context = SimpleNamespace()
    context.settings = effective
    context.output = work_dir / "refit-context.json"
    context.asset_dir = work_dir / "refit-assets"
    context.artifact = {
        "results": {
            "width_curves": {
                "legacy_subset": [],
            },
            "local_fitting": [],
        }
    }
    context.persist = lambda stage=None: persist(
        output_dir,
        artifact,
        f"preparing_fitted_starts:{stage or 'working'}",
        {"local_fits_completed": len(context.artifact["results"]["local_fitting"])},
    )
    model_config = make_model_config(settings["model"])
    context.model, context.tokenizer = load_model_and_tokenizer(model_config)
    context.attention_implementation = validated_attention_implementation(context.model)
    context.device = next(context.model.parameters()).device
    context.model_dtype = next(context.model.parameters()).dtype
    context.blocks = {row.index: row for row in discover_mlp_blocks(context.model)}
    expected = set(int(value) for value in settings["allocation"]["eligible_layers"])
    if (
        set(context.blocks) != set(range(settings["model"]["num_layers"]))
        or not expected <= set(context.blocks)
    ):
        raise ValueError("Loaded model topology differs from the S7 fitting contract")
    context.data = local_data if local_data is not None else build_local_data(context)
    torch.manual_seed(settings["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings["seed"])
    return context, model_config


def clean_width_curve_rows(rows):
    cleaned = deepcopy(rows)
    for row in cleaned:
        row.pop("state_path", None)
        row.pop("state_sha256", None)
    return cleaned


def clean_local_fitting_rows(rows):
    cleaned = deepcopy(rows)
    for row in cleaned:
        row.pop("state_path", None)
        row.pop("state_sha256", None)
    return cleaned


def prepare_dense_evaluation(context, settings, output_dir, artifact):
    import numpy as np
    import torch

    protocol = artifact["results"]["evaluation_protocol"]
    dense = artifact["results"].setdefault(
        "dense_evaluation",
        {"likelihood": {}, "tasks": {}, "status": "running"},
    )
    parameters = list(context.model.parameters())
    buffers = list(context.model.buffers())
    dense["footprint"] = {
        "parameters": sum(value.numel() for value in parameters),
        "parameter_bytes": sum(value.numel() * value.element_size() for value in parameters),
        "buffer_bytes": sum(value.numel() * value.element_size() for value in buffers),
    }
    dense["attention_implementation"] = context.attention_implementation
    monitor_record = artifact["results"]["monitoring_evaluation"]
    monitor_path = output_path(output_dir, monitor_record["path"])
    if file_digest(monitor_path) != monitor_record["sha256"]:
        raise ValueError("Prepared monitoring evaluation batches changed")
    monitoring = torch.load(monitor_path, map_location="cpu", weights_only=False)
    if "monitoring_validation" not in dense:
        dense["monitoring_validation"] = evaluate_lm_mixed(
            context.model,
            monitoring["model_validation"],
            "cuda",
            settings["data"]["model_validation_batches"],
        )
        persist(output_dir, artifact, "dense_monitoring_evaluation")

    corpora = {}
    for split, record in protocol["corpora"].items():
        path = output_path(output_dir, record["path"])
        if file_digest(path) != record["sha256"]:
            raise ValueError(f"Prepared WikiText {split} corpus changed")
        values = np.memmap(path, mode="r", dtype=np.int32)
        if len(values) != int(record["token_count"]):
            raise ValueError(f"Prepared WikiText {split} extent changed")
        corpora[split] = values
    for split, values in corpora.items():
        for context_length, stride in zip(
            settings["evaluation"]["contexts"],
            settings["evaluation"]["strides"],
            strict=True,
        ):
            key = f"{split}-{context_length}"
            if key not in dense["likelihood"]:
                dense["likelihood"][key] = evaluate_rolling_likelihood(
                    context.model,
                    values,
                    context_length,
                    stride,
                    "cuda",
                )
                persist(output_dir, artifact, f"dense_likelihood:{key}")

    for task in settings["evaluation"]["tasks"]:
        if task in dense["tasks"]:
            record = dense["tasks"][task]
            if file_digest(output_path(output_dir, record["path"])) != record["sha256"]:
                raise ValueError(f"Prepared dense benchmark changed: {task}")
            continue
        config = native_task_config(task)
        config["dataset_kwargs"] = {
            **(config.get("dataset_kwargs") or {}),
            "revision": protocol["tasks"][task]["revision"],
        }
        config["num_fewshot"] = 0
        result = evaluate_pinned_task(
            context.model,
            context.tokenizer,
            task,
            settings["evaluation"],
            config,
        )
        path = output_dir / "prepared" / "evaluation" / "dense" / f"{task}.json"
        write_json_atomic(path, result)
        dense["tasks"][task] = {
            "path": relative_output(output_dir, path),
            "sha256": file_digest(path),
        }
        persist(output_dir, artifact, f"dense_benchmark:{task}")
    dense["status"] = "completed"
    persist(output_dir, artifact, "dense_evaluation_completed")
    return dense


def build_start_candidates(context, settings, targets):
    import torch

    from mlp_replacement.operators import swiglu_neuron_importance_scores
    from .swiglu5.fitting import capture_dense_pairs, discrete_allocation, ensure_fit_for_width

    initialization = settings["preparation"]["initialization"]
    allocations = {target: discrete_allocation(context, initialization, target) for target in targets}
    candidates = {
        target: {
            "recipe_id": settings["preparation"]["recipe_id"],
            "target": target,
            "initialization": initialization,
            "allocation_method": "discrete_width_curve",
            "allocation_solver": {
                key: value for key, value in asdict(allocations[target]).items() if key != "rows"
            },
            "widths": {int(row["layer"]): int(row["replacement_width"]) for row in allocations[target].rows},
            "allocation": [],
        }
        for target in targets
    }
    layers = tuple(int(value) for value in context.settings["compatibility"]["eligible_layers"])
    group_size = int(context.settings["local_fitting"]["capture_group_size"])
    hidden = int(settings["model"]["hidden_size"])
    original_width = int(settings["model"]["intermediate_size"])
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
        for candidate in candidates.values():
            for layer in group:
                width = candidate["widths"][layer]
                path = context.blocks[layer].path
                fit = ensure_fit_for_width(
                    context,
                    initialization,
                    layer,
                    width,
                    training_by_path[path],
                    validation_by_path[path],
                    neuron_ranking=rankings[layer],
                )
                if fit is None:
                    row = {
                        "layer": layer,
                        "original_width": original_width,
                        "replacement_width": width,
                        "replacement_parameters": 3 * hidden * original_width,
                        "retains_dense_module": True,
                        "has_output_bias": False,
                    }
                else:
                    row = {
                        "layer": layer,
                        "original_width": original_width,
                        "replacement_width": width,
                        "replacement_parameters": int(fit["parameter_count"]),
                        "retains_dense_module": False,
                        "has_output_bias": bool(fit["has_output_bias"]),
                        "state_path": fit["state_path"],
                        "state_sha256": fit["state_sha256"],
                    }
                row["is_boundary_width"] = candidate["allocation_solver"].get("boundary_layer") == layer
                candidate["allocation"].append(row)
        training_by_path.clear()
        validation_by_path.clear()
        rankings.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for candidate in candidates.values():
        candidate.pop("widths")
        candidate["allocation"].sort(key=lambda row: int(row["layer"]))
        original = len(candidate["allocation"]) * 3 * hidden * original_width
        retained = sum(int(row["replacement_parameters"]) for row in candidate["allocation"])
        candidate.update(
            {
                "original_eligible_parameters": original,
                "retained_eligible_parameters": retained,
                "removed_eligible_parameters": original - retained,
                "realized_eligible_mlp_removal": 1.0 - retained / original,
            }
        )
    return candidates


def cpu_tree(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return deepcopy(value)


def fit_state_map(candidate, student):
    import torch

    blocks = {row.index: row for row in discover_mlp_blocks(student)}
    state = {}
    for row in candidate["allocation"]:
        if row.get("retains_dense_module"):
            continue
        path = resolve_path(Path(row["state_path"]))
        if file_digest(path) != row["state_sha256"]:
            raise ValueError(f"Refitted operator state changed: {path}")
        state[blocks[int(row["layer"])].path] = torch.load(path, map_location="cpu", weights_only=False)
    return state


def create_fitted_start(
    model_config,
    candidate,
    settings,
    output_dir,
    artifact,
):
    import torch

    target = float(candidate["target"])
    student, target_paths, train_modules = build_swiglu_student(
        model_config, settings["model"]["hidden_size"], candidate["allocation"]
    )
    load_replacement_state(student, fit_state_map(candidate, student))
    rng = capture_rng()
    clean_allocation = deepcopy(candidate["allocation"])
    for row in clean_allocation:
        row.pop("state_path", None)
        row.pop("state_sha256", None)
        row.pop("fit_key", None)
    recipe = {
        key: deepcopy(value)
        for key, value in candidate.items()
        if key != "allocation"
    }
    recipe["allocation"] = clean_allocation
    recipe_fingerprint = content_digest(recipe)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "stage": "prepared-start",
        "target": target,
        "recipe_id": settings["preparation"]["recipe_id"],
        "recipe_fingerprint": recipe_fingerprint,
        "tokens_seen": 0,
        "optimizer_updates": 0,
        "training_seconds": 0.0,
        "replacement_state": replacement_state(student, target_paths),
        **rng,
    }
    destination = output_dir / "prepared" / "starts" / f"target-{target:.1f}.pt"
    save_tensor_atomic(destination, payload)
    record = {
        "target": target,
        "recipe_id": settings["preparation"]["recipe_id"],
        "recipe_fingerprint": recipe_fingerprint,
        "construction": "fresh_s7_discrete_width_curve_fit",
        "tokens_seen": 0,
        "optimizer_updates": 0,
        "training_seconds": 0.0,
        "allocation": clean_allocation,
        "allocation_solver": deepcopy(candidate["allocation_solver"]),
        "path": relative_output(output_dir, destination),
        "sha256": file_digest(destination),
    }
    del student, train_modules, payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    artifact["results"].setdefault("starts", {})[str(target)] = record
    persist(output_dir, artifact, f"prepared_start:{target}", {"target": target})
    return record


def validate_completed_preparation(artifact, output_dir, settings):
    results = artifact["results"]
    required = {
        "stream",
        "monitoring_evaluation",
        "evaluation_protocol",
        "dense_evaluation",
        "width_curves",
        "local_fitting",
        "starts",
    }
    if not required <= set(results):
        raise ValueError("Completed preparation is missing required S7-owned results")
    stream = results["stream"]
    stream_path = output_path(output_dir, stream["path"])
    if (
        int(stream["token_count"]) != settings["recovery"]["target_tokens"]
        or stream_path.stat().st_size != 4 * int(stream["token_count"])
        or file_digest(stream_path) != stream["sha256"]
    ):
        raise ValueError("Completed preparation recovery stream changed")
    monitor = results["monitoring_evaluation"]
    if file_digest(output_path(output_dir, monitor["path"])) != monitor["sha256"]:
        raise ValueError("Completed preparation monitoring batches changed")
    protocol = results["evaluation_protocol"]
    for record in protocol["corpora"].values():
        path = output_path(output_dir, record["path"])
        if (
            file_digest(path) != record["sha256"]
            or path.stat().st_size != 4 * int(record["token_count"])
        ):
            raise ValueError("Completed preparation evaluation corpus changed")
    dense = results["dense_evaluation"]
    expected_likelihood = {
        f"{split}-{context}"
        for split in ("validation", "test")
        for context in settings["evaluation"]["contexts"]
    }
    if dense.get("status") != "completed" or set(dense["likelihood"]) != expected_likelihood:
        raise ValueError("Completed preparation lacks the dense likelihood reference")
    if set(dense["tasks"]) != set(settings["evaluation"]["tasks"]):
        raise ValueError("Completed preparation lacks dense benchmark records")
    for record in dense["tasks"].values():
        if file_digest(output_path(output_dir, record["path"])) != record["sha256"]:
            raise ValueError("Completed preparation dense benchmark changed")
    curves = results["width_curves"]
    curve_rows = curves.get("rows", [])
    expected_curve_points = {
        (layer, min(settings["model"]["intermediate_size"], max(1, round(
            settings["model"]["intermediate_size"] * ratio
        ))))
        for layer in settings["allocation"]["eligible_layers"]
        for ratio in settings["allocation"]["width_ratios"]
    }
    observed_curve_points = {
        (int(row["layer"]), int(row["replacement_width"])) for row in curve_rows
    }
    if (
        curves.get("initialization") != settings["preparation"]["initialization"]
        or observed_curve_points != expected_curve_points
        or len(curve_rows) != len(expected_curve_points)
        or any("state_path" in row or "state_sha256" in row for row in curve_rows)
    ):
        raise ValueError("Completed preparation width curves are incomplete")
    fitting_rows = results["local_fitting"]
    fit_keys = [row.get("fit_key") for row in fitting_rows]
    if (
        len(fitting_rows) < len(settings["allocation"]["eligible_layers"]) * 6
        or None in fit_keys
        or len(fit_keys) != len(set(fit_keys))
        or any("state_path" in row or "state_sha256" in row for row in fitting_rows)
    ):
        raise ValueError("Completed preparation local-fit evidence is incomplete")
    expected_starts = {str(value) for value in settings["targets"]}
    if set(results["starts"]) != expected_starts:
        raise ValueError("Completed preparation is missing a starting state")
    for record in results["starts"].values():
        if (
            record.get("construction") != "fresh_s7_discrete_width_curve_fit"
            or record.get("recipe_id") != settings["preparation"]["recipe_id"]
            or len(record.get("allocation", []))
            != len(settings["allocation"]["eligible_layers"])
            or file_digest(output_path(output_dir, record["path"])) != record["sha256"]
        ):
            raise ValueError("Completed preparation starting state changed")


def prepare(args, settings, config_path):
    work_dir, output_dir = resolve_storage(args.work_dir, args.output_dir, args.resume)
    artifact = start_artifact(output_dir, settings, "prepare", args.resume, {"kind": "shared"})
    try:
        contract = {
            "configuration": settings,
            "configuration_sha256": file_digest(config_path),
            "code_hashes": code_hashes(),
        }
        run_fingerprint = content_digest(contract)
        if args.resume and artifact.get("run_fingerprint") != run_fingerprint:
            raise ValueError("Preparation configuration or maintained code changed")
        was_completed = artifact.get("status") == "completed"
        artifact.update(
            {
                "run_fingerprint": run_fingerprint,
                "code_hashes": contract["code_hashes"],
                "provenance": {
                    "base_model": {
                        "model_id": settings["model"]["model_id"],
                        "revision": settings["model"]["revision"],
                        "tokenizer_revision": settings["model"]["tokenizer_revision"],
                    },
                    "datasets": {
                        "c4": settings["data"]["recovery_source"]["revision"],
                        "wikitext": settings["evaluation"]["wikitext_revision"],
                    },
                    "prior_experiment_artifacts": [],
                },
                "configuration_sha256": contract["configuration_sha256"],
                "status": "running",
                "error": None,
            }
        )
        if was_completed:
            validate_completed_preparation(artifact, output_dir, settings)
            return

        if "storage_preflight" not in artifact["results"]:
            artifact["results"]["storage_preflight"] = preparation_storage_preflight(
                work_dir,
                output_dir,
                settings,
            )
            persist(output_dir, artifact, "storage_preflight")

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            settings["model"]["model_id"],
            revision=settings["model"]["tokenizer_revision"],
            trust_remote_code=settings["model"]["trust_remote_code"],
        )
        prepare_evaluation_protocol(settings, output_dir, artifact, tokenizer)
        local_data = prepare_monitoring_evaluation(
            settings,
            output_dir,
            artifact,
            tokenizer,
        )
        prepare_recovery_stream(settings, output_dir, artifact, tokenizer)
        artifact["results"].setdefault("starts", {})
        missing = [
            float(target)
            for target in settings["targets"]
            if str(float(target)) not in artifact["results"]["starts"]
        ]
        dense_complete = (
            artifact["results"].get("dense_evaluation", {}).get("status")
            == "completed"
        )
        context = None
        if missing or not dense_complete:
            context, model_config = preparation_context(
                settings,
                work_dir,
                output_dir,
                artifact,
                local_data=local_data,
            )
        if missing:
            from .swiglu5.fitting import build_width_curves

            selection_cache = cache_teacher_logits(
                context.model,
                context.data["allocation_selection"],
                context.data["partition_batches"]["allocation_selection"],
                context.device,
                "float16",
            )
            build_width_curves(
                context,
                selection_cache,
                initializations=(settings["preparation"]["initialization"],),
            )
            artifact["results"]["width_curves"] = {
                "initialization": settings["preparation"]["initialization"],
                "rows": clean_width_curve_rows(
                    context.artifact["results"]["width_curves"][
                        settings["preparation"]["initialization"]
                    ]
                ),
            }
            persist(output_dir, artifact, "width_curves_completed")
            candidates = build_start_candidates(context, settings, missing)
            fitting_by_key = {
                row["fit_key"]: row
                for row in artifact["results"].get("local_fitting", [])
            }
            fitting_by_key.update(
                {
                    row["fit_key"]: row
                    for row in clean_local_fitting_rows(
                        context.artifact["results"]["local_fitting"]
                    )
                }
            )
            artifact["results"]["local_fitting"] = [
                fitting_by_key[key] for key in sorted(fitting_by_key)
            ]
            persist(output_dir, artifact, "local_fitting_completed")
            for target in missing:
                create_fitted_start(
                    model_config,
                    candidates[target],
                    settings,
                    output_dir,
                    artifact,
                )
            del selection_cache, candidates
        if not dense_complete:
            prepare_dense_evaluation(context, settings, output_dir, artifact)
        if context is not None:
            del context
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        validate_completed_preparation(artifact, output_dir, settings)
        artifact["status"] = "completed"
        persist(output_dir, artifact, "completed")
    except BaseException as error:
        artifact["status"] = "failed"
        artifact["error"] = f"{type(error).__name__}: {error}".replace(
            str(work_dir), "<work-dir>"
        )
        persist(output_dir, artifact, "failed")
        raise


def load_prepared(path, settings):
    path = resolve_path(Path(path))
    prepared = read_json(path)
    if (
        prepared.get("workflow") != WORKFLOW
        or prepared.get("command") != "prepare"
        or prepared.get("status") != "completed"
        or prepared.get("configuration") != settings
    ):
        raise ValueError("Training requires a completed, matching SwiGLU-7 preparation")
    return path, prepared, path.parent


def parameter_names(model, parameters):
    names = {id(value): name for name, value in model.named_parameters()}
    selected = []
    for parameter in parameters:
        if id(parameter) not in names:
            raise ValueError("Trainable parameter is not owned by the student model")
        selected.append(names[id(parameter)])
    if len(selected) != len(set(selected)):
        raise ValueError("Trainable parameter selection contains duplicates")
    return selected


def unique_parameters(modules, excluded=()):
    excluded_ids = {id(value) for value in excluded}
    seen = set(excluded_ids)
    result = []
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    return result


def cast_trainable_fp32(parameters):
    import torch

    for parameter in parameters:
        if not parameter.is_floating_point():
            raise TypeError("SwiGLU-7 trainable parameters must be floating point")
        parameter.data = parameter.data.to(dtype=torch.float32)


def rms_norm_modules(model):
    result = [module for module in model.modules() if module.__class__.__name__.lower().endswith("rmsnorm")]
    if not result:
        raise ValueError("No RMSNorm modules were discovered")
    return result


def configure_strategy(model, target_paths, strategy_id, settings):
    replacement_modules = [model.get_submodule(path) for path in target_paths]
    replacements = unique_parameters(replacement_modules)
    strategy = settings["strategies"][strategy_id]
    adapters = {}
    if strategy_id == "S7-0":
        groups = [("replacements", replacements)]
        train_modules = replacement_modules
    elif strategy_id == "S7-1":
        vocabulary_ids = {
            id(parameter)
            for module in (model.get_input_embeddings(), model.get_output_embeddings())
            for parameter in module.parameters()
        }
        body = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in vocabulary_ids
            and id(parameter) not in {id(value) for value in replacements}
        ]
        if not body:
            raise ValueError("Full-transformer strategy discovered no additional body parameters")
        groups = [("replacements", replacements), ("transformer_body", body)]
        train_modules = [model]
    elif strategy_id == "S7-2":
        layer_refs = discover_transformer_layers(model)
        suffixes = strategy["lora"]["attention_projections"]
        paths = [f"{layer.path}.self_attn.{suffix}" for layer in layer_refs for suffix in suffixes]
        adapters = install_lora_adapters(
            model,
            paths,
            strategy["lora"]["rank"],
            strategy["lora"]["alpha"],
            strategy["lora"]["dropout"],
        )
        mlps = [row.module for row in discover_mlp_blocks(model)]
        norms = rms_norm_modules(model)
        if len(mlps) != int(settings["model"]["num_layers"]):
            raise ValueError("S7-2 did not discover every transformer MLP")
        if len(norms) != 2 * int(settings["model"]["num_layers"]) + 1:
            raise ValueError("S7-2 requires both per-layer RMSNorms and the final RMSNorm")
        full_mlp_norm = unique_parameters([*mlps, *norms], replacements)
        low_rank = lora_parameters(adapters)
        groups = [
            ("replacements", replacements),
            ("remaining_mlps_and_rmsnorms", full_mlp_norm),
            ("attention_lora", low_rank),
        ]
        train_modules = [*mlps, *norms, *adapters.values()]
    else:
        raise ValueError(f"Unknown SwiGLU-7 strategy: {strategy_id}")
    flat = [parameter for unused_name, values in groups for parameter in values]
    if len({id(value) for value in flat}) != len(flat):
        raise ValueError("Strategy parameter groups overlap")
    cast_trainable_fp32(flat)
    records = []
    recovery = settings["recovery"]
    for name, parameters in groups:
        records.append(
            {
                "name": name,
                "parameters": parameters,
                "parameter_names": parameter_names(model, parameters),
                "parameter_count": sum(value.numel() for value in parameters),
                "learning_rate": recovery["learning_rate"],
                "weight_decay": recovery["weight_decay"],
            }
        )
    embedding_ids = {
        id(parameter)
        for module in (model.get_input_embeddings(), model.get_output_embeddings())
        for parameter in module.parameters()
    }
    if any(id(value) in embedding_ids for value in flat):
        raise ValueError("SwiGLU-7 must keep embeddings and the language-model head frozen")
    return records, train_modules, adapters


def trainable_state(model, groups):
    named = dict(model.named_parameters())
    return {
        name: named[name].detach().cpu().clone()
        for group in groups
        for name in group["parameter_names"]
    }


def load_trainable_state(model, state, groups):
    named = dict(model.named_parameters())
    expected = {name for group in groups for name in group["parameter_names"]}
    if set(state) != expected:
        raise ValueError("Checkpoint trainable tensors do not match the strategy")
    for name, value in state.items():
        parameter = named[name]
        if parameter.shape != value.shape:
            raise ValueError(f"Checkpoint tensor shape changed: {name}")
        parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def group_metadata(groups, model):
    total = sum(value.numel() for value in model.parameters())
    records = [
        {
            "name": group["name"],
            "parameter_names": list(group["parameter_names"]),
            "parameter_tensors": len(group["parameters"]),
            "parameters": int(group["parameter_count"]),
            "learning_rate": float(group["learning_rate"]),
            "weight_decay": float(group["weight_decay"]),
        }
        for group in groups
    ]
    trainable = sum(row["parameters"] for row in records)
    return {
        "groups": records,
        "trainable_parameters": trainable,
        "model_parameters": total,
        "trainable_fraction": trainable / total,
    }


def desired_requests(settings):
    recovery = settings["recovery"]
    return sorted(
        set(
            list(
                range(
                    recovery["validation_interval_tokens"],
                    recovery["target_tokens"] + 1,
                    recovery["validation_interval_tokens"],
                )
            )
            + recovery["monitoring_ppl_tokens"]
            + recovery["segment_endpoints"]
        )
    )


def task_files():
    import lm_eval

    root = Path(lm_eval.__file__).parent
    return {
        str(path.relative_to(root)): file_digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in (".py", ".yaml")
    }


def native_task_config(task):
    from lm_eval.tasks import TaskManager
    from lm_eval.tasks._yaml_loader import load_yaml

    entry = TaskManager().task_index[task]
    path = entry.yaml_path if hasattr(entry, "yaml_path") else entry["yaml_path"]
    return load_yaml(path, resolve_func=True, recursive=True)


def validate_evaluation_preflight(prepared, prepared_root, settings):
    import psutil  # noqa: F401
    import safetensors  # noqa: F401

    protocol = prepared["results"]["evaluation_protocol"]
    if version("lm_eval") != settings["evaluation"]["harness_version"]:
        raise ValueError("Install the pinned evaluation harness before recovery starts")
    if task_files() != protocol["harness_source_hashes"]:
        raise ValueError("Evaluation harness source differs from the frozen protocol")
    for split, record in protocol["corpora"].items():
        path = output_path(prepared_root, record["path"])
        if file_digest(path) != record["sha256"] or path.stat().st_size != 4 * int(record["token_count"]):
            raise ValueError(f"Prepared WikiText {split} corpus changed")
    dense = prepared["results"]["dense_evaluation"]
    for split in ("validation", "test"):
        record = dense["likelihood"].get(f"{split}-8192")
        if record is None or int(record.get("context_length", -1)) != 8192:
            raise ValueError("Prepared dense evaluation lacks native-context likelihood")
    for task in settings["evaluation"]["tasks"]:
        record = dense["tasks"][task]
        path = output_path(prepared_root, record["path"])
        if file_digest(path) != record["sha256"]:
            raise ValueError(f"Prepared dense benchmark changed: {task}")
        config = native_task_config(task)
        expected = protocol["tasks"][task]
        split = config.get("test_split") or config.get("validation_split")
        if config.get("dataset_path") != expected["dataset"] or split != expected["split"]:
            raise ValueError(f"Frozen benchmark definition changed: {task}")


def storage_preflight(output_dir, groups, model, reserve_fraction):
    trainable = sum(int(group["parameter_count"]) for group in groups)
    model_parameters = sum(value.numel() for value in model.parameters())
    atomic_checkpoint_bytes = 24 * trainable
    checkpoint_plus_bundle_bytes = 12 * trainable + 2 * model_parameters
    required = int(
        (max(atomic_checkpoint_bytes, checkpoint_plus_bundle_bytes) + 2_000_000_000)
        * (1.0 + float(reserve_fraction))
    )
    free = shutil.disk_usage(output_dir).free
    if free < required:
        raise RuntimeError(
            f"SwiGLU-7 requires at least {required:,} free output bytes; found {free:,}"
        )
    return {
        "trainable_parameters": trainable,
        "model_parameters": model_parameters,
        "estimated_required_free_bytes": required,
        "observed_free_bytes": free,
        "reserve_fraction": float(reserve_fraction),
    }


def evaluate_final_model(
    artifact,
    output_dir,
    work_dir,
    prepared,
    prepared_root,
    settings,
    student,
    tokenizer,
    allocation,
    adapters,
    run_fingerprint,
):
    import numpy as np
    import torch

    evaluation = artifact["results"].setdefault("evaluation", {"likelihood": {}, "tasks": {}})
    monitoring_record = prepared["results"]["monitoring_evaluation"]
    monitoring_path = output_path(prepared_root, monitoring_record["path"])
    if file_digest(monitoring_path) != monitoring_record["sha256"]:
        raise ValueError("Prepared monitoring evaluation batches changed")
    monitoring = torch.load(
        monitoring_path,
        map_location="cpu",
        weights_only=False,
    )
    if "monitoring_before_bf16" not in evaluation:
        evaluation["monitoring_before_bf16"] = evaluate_lm_mixed(
            student,
            monitoring["model_validation"],
            "cuda",
            settings["data"]["model_validation_batches"],
        )
    merged = []
    if adapters:
        merged = merge_lora_adapters(student, adapters)
        adapters.clear()
    replacements = []
    for row in allocation:
        if row.get("retains_dense_module"):
            continue
        path = next(
            ref.path for ref in discover_mlp_blocks(student) if ref.index == int(row["layer"])
        )
        replacements.append(
            {
                "path": path,
                "width": int(row["replacement_width"]),
                "down_bias": bool(row.get("has_output_bias", False)),
            }
        )
    bundle_path = output_dir / "model"
    bundle_staging = output_dir / "model.building"
    if not bundle_path.exists() and bundle_staging.exists():
        if bundle_staging.resolve().parent != output_dir.resolve():
            raise ValueError("Incomplete bundle staging path escaped the output directory")
        shutil.rmtree(bundle_staging)
    if not bundle_path.exists():
        manifest = export_bundle(
            student,
            tokenizer,
            bundle_path,
            replacements,
            {
                "workflow": WORKFLOW,
                "run_fingerprint": run_fingerprint,
                "strategy": artifact["identity"]["strategy"],
                "target": artifact["identity"]["target"],
                "lora_merged": merged,
            },
        )
        evaluation["footprint"] = {
            key: manifest[key]
            for key in ("parameters", "parameter_bytes", "buffer_bytes", "tensor_file_bytes", "bundle_bytes")
        }
        persist(output_dir, artifact, "bundle_exported")
    manifest = validate_bundle(bundle_path)
    if manifest["provenance"]["run_fingerprint"] != run_fingerprint:
        raise ValueError("Existing final bundle belongs to another run")
    evaluation["model"] = {
        "path": relative_output(output_dir, bundle_path),
        "manifest_sha256": file_digest(bundle_path / "bundle.json"),
    }
    evaluation.setdefault(
        "footprint",
        {
            **{
                key: manifest[key]
                for key in ("parameters", "parameter_bytes", "buffer_bytes", "tensor_file_bytes")
            },
            "bundle_bytes": sum(path.stat().st_size for path in bundle_path.rglob("*") if path.is_file()),
        },
    )
    student.to("cpu")
    del student, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if "resident_memory" not in evaluation:
        measurement_path = work_dir / "resident-memory.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "workflows.runs.model.swiglu.swiglu_7",
                "measure-bundle",
                "--bundle",
                str(bundle_path),
                "--output",
                str(measurement_path),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        evaluation["resident_memory"] = read_json(measurement_path)
        if evaluation["resident_memory"]["bundle_manifest_sha256"] != file_digest(
            bundle_path / "bundle.json"
        ):
            raise ValueError("Resident-memory measurement belongs to another bundle")
        persist(output_dir, artifact, "resident_memory")
    model, tokenizer, unused_manifest = load_bundle(bundle_path)
    evaluation["attention_implementation"] = validated_attention_implementation(model)
    if int(model.config.max_position_embeddings) < max(settings["evaluation"]["contexts"]):
        raise ValueError("Final model does not support the configured evaluation context")
    if "monitoring_after_bf16" not in evaluation:
        evaluation["monitoring_after_bf16"] = evaluate_lm_mixed(
            model,
            monitoring["model_validation"],
            "cuda",
            settings["data"]["model_validation_batches"],
        )
        evaluation["bf16_conversion_ppl_delta"] = (
            evaluation["monitoring_after_bf16"]["perplexity"]
            - evaluation["monitoring_before_bf16"]["perplexity"]
        )
        persist(output_dir, artifact, "monitoring_evaluation")
    protocol = prepared["results"]["evaluation_protocol"]
    if (
        version("lm_eval") != settings["evaluation"]["harness_version"]
        or task_files() != protocol["harness_source_hashes"]
    ):
        raise ValueError("Evaluation harness version or source differs from the frozen protocol")
    corpora = {}
    for split, record in protocol["corpora"].items():
        path = output_path(prepared_root, record["path"])
        if file_digest(path) != record["sha256"]:
            raise ValueError(f"Prepared WikiText {split} tokens changed")
        values = np.memmap(path, mode="r", dtype=np.int32)
        if len(values) != int(record["token_count"]):
            raise ValueError(f"Prepared WikiText {split} extent changed")
        corpora[split] = values
    for split, token_ids in corpora.items():
        for context, stride in zip(
            settings["evaluation"]["contexts"],
            settings["evaluation"]["strides"],
            strict=True,
        ):
            key = f"{split}-{context}"
            if key not in evaluation["likelihood"]:
                evaluation["likelihood"][key] = evaluate_rolling_likelihood(
                    model, token_ids, context, stride, "cuda"
                )
                persist(output_dir, artifact, f"likelihood:{key}")
    dense = prepared["results"]["dense_evaluation"]
    if "likelihood_comparison_to_dense" not in evaluation:
        evaluation["likelihood_comparison_to_dense"] = {}
        for key, student_metrics in evaluation["likelihood"].items():
            dense_metrics = dense["likelihood"][key]
            evaluation["likelihood_comparison_to_dense"][key] = {
                "student_loss": student_metrics["loss"],
                "dense_loss": dense_metrics["loss"],
                "student_minus_dense_loss": (
                    student_metrics["loss"] - dense_metrics["loss"]
                ),
                "student_perplexity": student_metrics["perplexity"],
                "dense_perplexity": dense_metrics["perplexity"],
                "student_minus_dense_perplexity": (
                    student_metrics["perplexity"] - dense_metrics["perplexity"]
                ),
            }
        persist(output_dir, artifact, "likelihood_comparison_to_dense")
    task_comparisons = {}
    for task in settings["evaluation"]["tasks"]:
        record = evaluation["tasks"].get(task)
        if record is None:
            config = native_task_config(task)
            config["dataset_kwargs"] = {
                **(config.get("dataset_kwargs") or {}),
                "revision": protocol["tasks"][task]["revision"],
            }
            config["num_fewshot"] = 0
            task_result = evaluate_pinned_task(model, tokenizer, task, settings["evaluation"], config)
            task_path = output_dir / "evaluation" / "benchmarks" / f"{task}.json"
            write_json_atomic(task_path, task_result)
            record = {"path": relative_output(output_dir, task_path), "sha256": file_digest(task_path)}
            evaluation["tasks"][task] = record
            persist(output_dir, artifact, f"benchmark:{task}")
        student_task_path = output_path(output_dir, record["path"])
        if file_digest(student_task_path) != record["sha256"]:
            raise ValueError(f"Student benchmark result changed: {task}")
        dense_record = dense["tasks"][task]
        dense_path = output_path(prepared_root, dense_record["path"])
        if file_digest(dense_path) != dense_record["sha256"]:
            raise ValueError(f"Dense benchmark result changed: {task}")
        student_task = read_json(student_task_path)
        dense_task = read_json(dense_path)
        metric = settings["evaluation"]["primary_metrics"][task]
        difference = paired_accuracy_difference(
            dense_task["samples"][task],
            student_task["samples"][task],
            metric,
            settings["evaluation"]["bootstrap_resamples"],
            settings["seed"],
        )
        task_comparisons[task] = {
            "dense": dense_task["results"][task][f"{metric},none"],
            "student": student_task["results"][task][f"{metric},none"],
            **difference,
        }
    evaluation["comparison_to_dense"] = {
        "tasks": task_comparisons,
        "macro_accuracy": sum(row["student"] for row in task_comparisons.values()) / len(task_comparisons),
        "macro_delta": sum(
            row["student_minus_dense"] for row in task_comparisons.values()
        )
        / len(task_comparisons),
    }
    evaluation["footprint"]["whole_model_parameter_removal"] = (
        1.0 - evaluation["footprint"]["parameters"] / dense["footprint"]["parameters"]
    )
    evaluation["status"] = "completed"
    del model, tokenizer, monitoring
    gc.collect()
    torch.cuda.empty_cache()


def train(args, settings, config_path):
    import torch

    work_dir, output_dir = resolve_storage(args.work_dir, args.output_dir, args.resume)
    target = float(args.target)
    strategy_id = args.strategy
    if target not in settings["targets"] or strategy_id not in settings["strategies"]:
        raise ValueError("Training target or strategy is outside the fixed SwiGLU-7 grid")
    identity = {"strategy": strategy_id, "target": target}
    artifact = start_artifact(output_dir, settings, "train", args.resume, identity)
    try:
        prepared_path, prepared, prepared_root = load_prepared(args.prepared, settings)
        validate_evaluation_preflight(prepared, prepared_root, settings)
        start_record = prepared["results"]["starts"][str(target)]
        if (
            start_record.get("construction") != "fresh_s7_discrete_width_curve_fit"
            or int(start_record.get("tokens_seen", -1)) != 0
            or int(start_record.get("optimizer_updates", -1)) != 0
        ):
            raise ValueError("SwiGLU-7 requires a fresh fitted token-zero start")
        start_path = output_path(prepared_root, start_record["path"])
        if file_digest(start_path) != start_record["sha256"]:
            raise ValueError("Prepared fitted starting state changed")
        stream = prepared["results"]["stream"]
        stream_path = output_path(prepared_root, stream["path"])
        if (
            file_digest(stream_path) != stream["sha256"]
            or stream_path.stat().st_size != 4 * int(stream["token_count"])
        ):
            raise ValueError("Prepared recovery stream changed")
        contract = {
            "configuration": settings,
            "configuration_sha256": file_digest(config_path),
            "prepared_sha256": file_digest(prepared_path),
            "start_sha256": start_record["sha256"],
            "stream_sha256": stream["sha256"],
            "identity": identity,
            "code_hashes": code_hashes(),
        }
        run_fingerprint = content_digest(contract)
        if args.resume and artifact.get("run_fingerprint") != run_fingerprint:
            raise ValueError("Resume inputs, configuration, or maintained code changed")
        was_completed = artifact.get("status") == "completed"
        artifact.update(
            {
                "run_fingerprint": run_fingerprint,
                "code_hashes": contract["code_hashes"],
                "configuration_sha256": contract["configuration_sha256"],
                "prepared": {"sha256": contract["prepared_sha256"]},
                "start": {"sha256": contract["start_sha256"], "tokens_seen": start_record["tokens_seen"]},
                "status": "running",
                "error": None,
            }
        )
        if was_completed:
            validate_bundle(output_dir / "model")
            recorded_manifest = artifact["results"]["evaluation"]["model"][
                "manifest_sha256"
            ]
            if recorded_manifest != file_digest(output_dir / "model" / "bundle.json"):
                raise ValueError("Completed result and bundle manifest differ")
            checkpoint_dir = output_dir / "checkpoint"
            for path in (list(checkpoint_dir.glob("checkpoint-*")) if checkpoint_dir.exists() else []):
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
                "tokens_seen": int(start_record["tokens_seen"]),
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
            or "stream_sha256" in start
            or start.get("recipe_id") != start_record.get("recipe_id")
            or start.get("recipe_fingerprint")
            != start_record.get("recipe_fingerprint")
        ):
            raise ValueError("Prepared start contains inherited recovery state")
        if int(start["tokens_seen"]) != int(start_record["tokens_seen"]):
            raise ValueError("Prepared start descriptor and payload disagree")
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
        # Give all three scopes at this target the same token-zero RNG state.
        # Resume restores the later in-run state below.
        restore_rng(start)
        groups, train_modules, adapters = configure_strategy(
            student, target_paths, strategy_id, settings
        )
        artifact["results"]["trainable_scope"] = group_metadata(groups, student)
        artifact["results"]["storage_preflight"] = storage_preflight(
            output_dir,
            groups,
            student,
            settings["recovery"]["disk_reserve_fraction"],
        )
        if strategy_id == "S7-2":
            artifact["results"]["trainable_scope"]["lora"] = deepcopy(settings["strategies"]["S7-2"]["lora"])
        device = next(student.parameters()).device
        monitoring_record = prepared["results"]["monitoring_evaluation"]
        monitoring_path = output_path(prepared_root, monitoring_record["path"])
        if file_digest(monitoring_path) != monitoring_record["sha256"]:
            raise ValueError("Prepared validation batches changed")
        monitoring = torch.load(
            monitoring_path,
            map_location="cpu",
            weights_only=False,
        )
        validation_cache = cache_teacher_logits(
            teacher,
            monitoring["recovery_validation"],
            len(monitoring["recovery_validation"]),
            device,
            "float16",
        )
        cache = PackedTokenCache(
            stream_path,
            int(stream["token_count"]),
            settings["recovery"]["sequence_length"],
            stream["sha256"],
        )
        recovery = settings["recovery"]
        checkpoint_dir = output_dir / "checkpoint"
        recovered = None
        if args.resume:
            try:
                recovered, descriptor = restore_checkpoint(checkpoint_dir, run_fingerprint)
            except FileNotFoundError:
                if int(artifact["results"]["recovery"]["tokens_seen"]) > int(start["tokens_seen"]):
                    raise
            if recovered is not None:
                load_trainable_state(student, recovered["trainable_state"], groups)
                artifact["results"]["recovery"] = deepcopy(recovered["recovery"])
                artifact["checkpoint"] = descriptor
                restore_rng(recovered)
        if recovered is None:
            optimizer_state = None
            artifact["results"]["recovery"].update(
                {
                    "tokens_seen": 0,
                    "optimizer_updates": 0,
                }
            )
            if not artifact["results"]["recovery"]["validation_history"]:
                rng = capture_rng()
                artifact["results"]["recovery"]["validation_history"].append(
                    {
                        "actual_tokens": int(start["tokens_seen"]),
                        "requested_tokens": [int(start["tokens_seen"])],
                        "optimizer_updates": int(start["optimizer_updates"]),
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
                "strategy": strategy_id,
                "target": target,
                "tokens_seen": int(start["tokens_seen"]),
                "optimizer_updates": int(start["optimizer_updates"]),
                "trainable_state": trainable_state(student, groups),
                "optimizer_state": cpu_tree(optimizer_state),
                "recovery": deepcopy(artifact["results"]["recovery"]),
                **capture_rng(),
            }
            artifact["checkpoint"] = commit_single_checkpoint(checkpoint_dir, initial_payload)
            persist(output_dir, artifact, "recovery_initialized")
            del initial_payload
        else:
            optimizer_state = recovered["optimizer_state"]
        del start, recovered, unused_train_modules
        cursor = int(artifact["results"]["recovery"]["tokens_seen"])
        updates = int(artifact["results"]["recovery"]["optimizer_updates"])
        final_tokens = int(recovery["target_tokens"])
        while cursor < final_tokens:
            origin = segment_origin(cursor, recovery["segment_endpoints"])
            end = next(value for value in recovery["segment_endpoints"] if value > cursor)
            schedule = segment_schedule(
                origin,
                end,
                desired_requests(settings),
                recovery["effective_batch_tokens"],
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
                    "strategy": strategy_id,
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
                    {
                        "strategy": strategy_id,
                        "target": target,
                        "tokens_seen": event.tokens_seen,
                        "target_tokens": final_tokens,
                    },
                )
                print(
                    f"strategy={strategy_id} target={target} "
                    f"tokens={event.tokens_seen:,} KL={validation_kl:.6f}",
                    flush=True,
                )

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
        gc.collect()
        torch.cuda.empty_cache()
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
        )
        artifact["results"]["final_evaluation_wall_seconds"] = (
            float(artifact["results"].get("final_evaluation_wall_seconds", 0.0))
            + perf_counter()
            - final_evaluation_started
        )
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
        records = list(checkpoint_dir.glob("checkpoint-*")) if checkpoint_dir.exists() else []
        for path in records:
            path.unlink()
        if checkpoint_dir.exists():
            checkpoint_dir.rmdir()
        artifact.pop("checkpoint", None)
        artifact["results"]["checkpoint_cleanup"] = "completed_after_result_and_bundle_validation"
        persist(output_dir, artifact, "completed")
    except BaseException as error:
        artifact["status"] = "failed"
        artifact["error"] = f"{type(error).__name__}: {error}".replace(str(work_dir), "<work-dir>")
        persist(output_dir, artifact, "failed")
        raise


def measure_bundle(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    measurement = measure_resident_bundle(Path(args.bundle))
    measurement["bundle_manifest_sha256"] = file_digest(Path(args.bundle) / "bundle.json")
    write_json_atomic(output, measurement)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        subparser.add_argument("--work-dir", type=Path, required=True)
        subparser.add_argument("--output-dir", type=Path, required=True)
        subparser.add_argument("--resume", action="store_true")
    train_parser = subparsers.choices["train"]
    train_parser.add_argument("--prepared", type=Path, required=True)
    train_parser.add_argument("--strategy", choices=("S7-0", "S7-1", "S7-2"), required=True)
    train_parser.add_argument("--target", type=float, choices=(0.2, 0.3, 0.4, 0.5), required=True)
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
    if args.command == "prepare":
        prepare(args, settings, config_path)
    else:
        train(args, settings, config_path)


if __name__ == "__main__":
    main()
