"""Freeze and execute the five-model SwiGLU-6 final evaluation."""

import argparse
import gc
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from workflows.runs.model.common import PROJECT_ROOT, resolve_path
from mlp_replacement.artifacts import contained_path, content_digest, file_digest, read_json, write_json_atomic
from mlp_replacement.evaluation.bundles import export_bundle, load_bundle, measure_resident_bundle, validate_bundle
from mlp_replacement.evaluation.final_quality import (
    evaluate_pinned_task, evaluate_rolling_likelihood, paired_accuracy_difference,
)
from .swiglu_6 import DEFAULT_CONFIG, asset_directory, code_hashes, load_prepared, load_settings, new_artifact, persist


def task_files():
    import lm_eval

    root = Path(lm_eval.__file__).parent
    return root, {str(path.relative_to(root)): file_digest(path)
                  for path in sorted(root.rglob("*")) if path.is_file() and path.suffix in (".py", ".yaml")}


def native_task_config(task):
    from lm_eval.tasks import TaskManager
    from lm_eval.tasks._yaml_loader import load_yaml

    manager = TaskManager()
    entry = manager.task_index[task]
    path = entry.yaml_path if hasattr(entry, "yaml_path") else entry["yaml_path"]
    return load_yaml(path, resolve_func=True, recursive=True)


def freeze_protocol(args, settings):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    import numpy as np

    output, artifact = new_artifact(args.output, settings, "evaluation-protocol")
    try:
        prepared_path, prepared = load_prepared(args.prepared, settings)
        evaluation = settings["evaluation"]
        if version("lm_eval") != evaluation["harness_version"]:
            raise ValueError("Install the pinned harness version in the evaluation environment")
        root, hashes = task_files()
        tasks = {}
        for task in evaluation["tasks"]:
            config = native_task_config(task)
            dataset = config["dataset_path"]
            revision = HfApi().dataset_info(dataset).sha
            tasks[task] = {"dataset": dataset, "revision": revision,
                           "split": config.get("test_split") or config["validation_split"]}
        assets = asset_directory(output)
        assets.mkdir(parents=True)
        tokenizer = AutoTokenizer.from_pretrained(settings["model"]["model_id"], revision=settings["model"]["tokenizer_revision"])
        wiki_revision = prepared["legacy_evaluation"]["wikitext_revision"]
        corpora = {}
        for split in ("validation", "test"):
            records = load_dataset(evaluation["wikitext_dataset"], evaluation["wikitext_name"],
                                   revision=wiki_revision, split=split)
            text = "\n\n".join(str(record.get("text") or "") for record in records)
            ids = np.asarray(tokenizer(text, add_special_tokens=False, return_attention_mask=False).input_ids, dtype=np.int32)
            path = assets / f"wikitext-{split}.bin"
            ids.tofile(path)
            corpora[split] = {"path": path.name, "sha256": file_digest(path), "token_count": len(ids)}
        artifact.update({"prepared_sha256": file_digest(prepared_path),
                         "harness_source_hashes": hashes, "tasks": tasks,
                         "wikitext_revision": wiki_revision, "corpora": corpora})
        artifact["protocol_fingerprint"] = content_digest({key: artifact[key] for key in
            ("configuration", "prepared_sha256", "harness_source_hashes", "tasks", "wikitext_revision", "corpora")})
        artifact["status"] = "completed"
        persist(output, artifact, "completed")
    except BaseException as error:
        artifact["status"], artifact["error"] = "failed", f"{type(error).__name__}: {error}"
        persist(output, artifact, "failed")
        raise


def frozen_cohort(args, settings):
    cohort = [{"id": "dense", "target": 0.0, "tokens": 0, "replacements": []}]
    targets = set()
    provenance = None
    for input_path in args.recovery:
        path = resolve_path(input_path)
        artifact = read_json(path)
        if artifact.get("workflow") != "swiglu-6" or artifact.get("stage") != "recovery" or artifact.get("status") != "completed":
            raise ValueError("Final evaluation requires completed 1B recovery artifacts")
        if artifact["configuration"] != settings:
            raise ValueError("Recovery configuration differs from final evaluation")
        if artifact["results"]["tokens_seen"] != settings["recovery"]["segment_endpoints"][-1]:
            raise ValueError("Recovery did not reach the fixed final endpoint")
        target = artifact["target"]
        if target in targets:
            raise ValueError("Duplicate recovery target")
        targets.add(target)
        if provenance is not None and artifact["prepared_sha256"] != provenance:
            raise ValueError("Recovery targets used different prepared streams")
        provenance = artifact["prepared_sha256"]
        search_record = artifact["provenance"]["search"]
        search_path = resolve_path(Path(settings["sources"]["search"]))
        if file_digest(search_path) != search_record["sha256"]:
            raise ValueError("Source allocation changed")
        search = read_json(search_path)
        candidate = search["results"]["candidates"][str(target)][settings["candidate_id"]]
        for tokens in settings["recovery"]["segment_endpoints"]:
            record = artifact["results"]["milestones"][str(tokens)]
            weights = contained_path(asset_directory(path), record["path"])
            if file_digest(weights) != record["sha256"]:
                raise ValueError("Fixed endpoint weights changed")
            cohort.append({"id": f"target-{target}-{tokens}", "target": target, "tokens": tokens,
                           "weights_path": str(weights), "weights_sha256": record["sha256"],
                           "recovery_sha256": file_digest(path), "run_fingerprint": artifact["run_fingerprint"],
                           "allocation": candidate["allocation"], "legacy_metrics": record["metrics"]})
    if targets != set(settings["targets"]):
        raise ValueError("Supply both completed recovery targets")
    return cohort, provenance


def export_cohort_model(row, settings, bundle_path, legacy):
    import torch
    from mlp_replacement.model import load_model_and_tokenizer
    from .shared import evaluate_lm_mixed, make_model_config
    from .swiglu_5 import blank_candidate_student, load_replacement_state

    model_config = make_model_config(settings["model"])
    if row["id"] == "dense":
        model, tokenizer = load_model_and_tokenizer(model_config)
        replacements = []
    else:
        from transformers import AutoTokenizer

        model, paths, modules = blank_candidate_student(SimpleNamespace(settings=settings), model_config,
                                                        {"allocation": row["allocation"]})
        state = torch.load(row["weights_path"], map_location="cpu", weights_only=False)
        if state["run_fingerprint"] != row["run_fingerprint"] or state["tokens_seen"] != row["tokens"]:
            raise ValueError("Milestone payload differs from frozen cohort")
        load_replacement_state(model, state["replacement_state"])
        del state
        tokenizer = AutoTokenizer.from_pretrained(settings["model"]["model_id"], revision=settings["model"]["tokenizer_revision"])
        replacements = [{"path": path, "width": module.bottleneck_size,
                         "down_bias": module.down_projection.bias is not None}
                        for path, module in zip(paths, modules)]
        del modules
    before = evaluate_lm_mixed(model, legacy["model_validation"], "cuda", 24)
    manifest = export_bundle(model, tokenizer, bundle_path, replacements,
                             {"workflow": "swiglu-6", "model": settings["model"], "cohort": row,
                              "legacy_before_bf16": before})
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return manifest, before


def summarize(artifact, assets, settings):
    tasks = settings["evaluation"]["tasks"]
    models = artifact["results"]["models"]
    dense = models["dense"]
    comparisons = {}
    for model_id, model in models.items():
        per_task = {}
        for task in tasks:
            dense_task = read_json(contained_path(assets, dense["tasks"][task]["path"]))
            student_task = read_json(contained_path(assets, model["tasks"][task]["path"]))
            metric = settings["evaluation"]["primary_metrics"][task]
            difference = paired_accuracy_difference(dense_task["samples"][task], student_task["samples"][task],
                                                     metric, settings["evaluation"]["bootstrap_resamples"], settings["seed"])
            dense_value = dense_task["results"][task][f"{metric},none"]
            student_value = student_task["results"][task][f"{metric},none"]
            per_task[task] = {"dense": dense_value, "student": student_value, **difference}
        comparisons[model_id] = {"tasks": per_task,
                                 "macro_accuracy": sum(value["student"] for value in per_task.values()) / len(tasks),
                                 "macro_delta": sum(value["student_minus_dense"] for value in per_task.values()) / len(tasks)}
        parameters = model["footprint"]["parameters"]
        model["footprint"]["whole_model_parameter_removal"] = 1 - parameters / dense["footprint"]["parameters"]
    artifact["results"]["comparisons"] = comparisons


def evaluate(args, settings):
    import numpy as np
    import torch
    from mlp_replacement.runlog import environment_record
    from .shared import evaluate_lm_mixed

    output, artifact = new_artifact(args.output, settings, "evaluation", args.resume)
    try:
        prepared_path, prepared = load_prepared(args.prepared, settings)
        protocol_path = resolve_path(args.protocol)
        protocol = read_json(protocol_path)
        if protocol.get("workflow") != "swiglu-6" or protocol.get("stage") != "evaluation-protocol" or protocol.get("status") != "completed":
            raise ValueError("A completed frozen evaluation protocol is required")
        if protocol["configuration"] != settings or protocol["prepared_sha256"] != file_digest(prepared_path):
            raise ValueError("Evaluation protocol inputs differ")
        root, hashes = task_files()
        if hashes != protocol["harness_source_hashes"] or version("lm_eval") != settings["evaluation"]["harness_version"]:
            raise ValueError("Harness source or task definitions changed since protocol freeze")
        cohort, prepared_digest = frozen_cohort(args, settings)
        if prepared_digest != file_digest(prepared_path):
            raise ValueError("Recovery and evaluation preparation artifacts differ")
        observed_environment = environment_record()
        contract = {"cohort": cohort, "protocol_sha256": file_digest(protocol_path),
                    "code_hashes": code_hashes(), "environment": observed_environment}
        fingerprint = content_digest(contract)
        if args.resume and artifact.get("evaluation_fingerprint") != fingerprint:
            raise ValueError("Frozen evaluation cohort/protocol changed")
        artifact.update({"cohort": cohort, "evaluation_fingerprint": fingerprint,
                         "protocol_sha256": file_digest(protocol_path), "environment": observed_environment,
                         "code_hashes": contract["code_hashes"],
                         "status": "running", "error": None})
        artifact["results"].setdefault("models", {})
        assets = asset_directory(output)
        assets.mkdir(parents=True, exist_ok=True)
        legacy_record = prepared["legacy_evaluation"]
        legacy_path = contained_path(asset_directory(prepared_path), legacy_record["path"])
        if file_digest(legacy_path) != legacy_record["sha256"]:
            raise ValueError("Historical validation batches changed")
        legacy = torch.load(legacy_path, map_location="cpu", weights_only=False)
        corpora = {}
        for split, record in protocol["corpora"].items():
            path = contained_path(asset_directory(protocol_path), record["path"])
            if file_digest(path) != record["sha256"]:
                raise ValueError("Frozen WikiText tokens changed")
            corpora[split] = np.memmap(path, mode="r", dtype=np.int32)
            if len(corpora[split]) != record["token_count"]:
                raise ValueError("Frozen WikiText extent changed")
        persist(output, artifact, "cohort_frozen")
        for row in cohort:
            model_id = row["id"]
            result = artifact["results"]["models"].setdefault(model_id, {"likelihood": {}, "tasks": {}})
            bundle_path = assets / "bundles" / model_id
            if not bundle_path.exists():
                manifest, before = export_cohort_model(row, settings, bundle_path, legacy)
                result["legacy_before_bf16"] = before
                result["footprint"] = {key: manifest[key] for key in
                                       ("parameters", "parameter_bytes", "buffer_bytes", "tensor_file_bytes", "bundle_bytes")}
                persist(output, artifact, f"exported:{model_id}")
            manifest = validate_bundle(bundle_path)
            if manifest["provenance"]["cohort"] != row:
                raise ValueError("An existing bundle belongs to a different frozen model")
            result.setdefault("legacy_before_bf16", manifest["provenance"]["legacy_before_bf16"])
            if "footprint" not in result:
                result["footprint"] = {key: manifest[key] for key in
                                       ("parameters", "parameter_bytes", "buffer_bytes", "tensor_file_bytes")}
                result["footprint"]["bundle_bytes"] = sum(path.stat().st_size for path in bundle_path.rglob("*") if path.is_file())
            if "resident_memory" not in result:
                memory_path = assets / f"{model_id}-resident.json"
                if not memory_path.exists():
                    subprocess.run([sys.executable, "-m", "workflows.runs.model.swiglu.swiglu_6_evaluate",
                                    "--measure-bundle", str(bundle_path), "--output", str(memory_path)],
                                   cwd=PROJECT_ROOT, check=True)
                result["resident_memory"] = read_json(memory_path)
                if result["resident_memory"]["bundle_manifest_sha256"] != file_digest(bundle_path / "bundle.json"):
                    raise ValueError("Resident-memory result belongs to another bundle")
                persist(output, artifact, f"memory:{model_id}")
            model, tokenizer, unused_manifest = load_bundle(bundle_path)
            if int(model.config.max_position_embeddings) < max(settings["evaluation"]["contexts"]):
                raise ValueError("Model does not support the declared evaluation contexts")
            if "legacy_after_bf16" not in result:
                result["legacy_after_bf16"] = evaluate_lm_mixed(model, legacy["model_validation"], "cuda", 24)
                before = result.get("legacy_before_bf16", row.get("legacy_metrics", {}).get("wikitext_validation"))
                if before:
                    result["bf16_conversion_ppl_delta"] = result["legacy_after_bf16"]["perplexity"] - before["perplexity"]
            for split, ids in corpora.items():
                for context, stride in zip(settings["evaluation"]["contexts"], settings["evaluation"]["strides"]):
                    key = f"{split}-{context}"
                    if key not in result["likelihood"]:
                        result["likelihood"][key] = evaluate_rolling_likelihood(model, ids, context, stride, "cuda")
                        persist(output, artifact, f"likelihood:{model_id}:{key}")
            for task in settings["evaluation"]["tasks"]:
                if task in result["tasks"]:
                    record = result["tasks"][task]
                    if file_digest(contained_path(assets, record["path"])) != record["sha256"]:
                        raise ValueError("Completed benchmark artifact changed")
                    continue
                config = native_task_config(task)
                config["dataset_kwargs"] = {**(config.get("dataset_kwargs") or {}),
                                            "revision": protocol["tasks"][task]["revision"]}
                config["num_fewshot"] = 0
                task_result = evaluate_pinned_task(model, tokenizer, task, settings["evaluation"], config)
                task_path = assets / "benchmarks" / f"{model_id}-{task}.json"
                write_json_atomic(task_path, task_result)
                result["tasks"][task] = {"path": str(task_path.relative_to(assets)), "sha256": file_digest(task_path)}
                del task_result
                persist(output, artifact, f"benchmark:{model_id}:{task}")
            result["status"] = "completed"
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            persist(output, artifact, f"completed:{model_id}")
        summarize(artifact, assets, settings)
        artifact["status"] = "completed"
        persist(output, artifact, "completed")
    except BaseException as error:
        artifact["status"], artifact["error"] = "failed", f"{type(error).__name__}: {error}"
        persist(output, artifact, "failed")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--recovery", type=Path, action="append", default=[])
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--measure-bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.measure_bundle:
        if args.output.exists():
            raise FileExistsError(args.output)
        measurement = measure_resident_bundle(args.measure_bundle)
        measurement["bundle_manifest_sha256"] = file_digest(args.measure_bundle / "bundle.json")
        write_json_atomic(args.output, measurement)
        return
    if not args.prepared:
        parser.error("--prepared is required")
    settings = load_settings(args.config)
    if args.freeze_only:
        if args.resume:
            parser.error("Protocol freezing writes a new immutable output")
        freeze_protocol(args, settings)
    else:
        if not args.protocol or len(args.recovery) != 2:
            parser.error("Final evaluation requires --protocol and two --recovery artifacts")
        evaluate(args, settings)


if __name__ == "__main__":
    main()
