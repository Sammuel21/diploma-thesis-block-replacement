"""SwiGLU-5 context; extracted with the established protocol unchanged."""

from __future__ import annotations

import json
import math
import shutil
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch

from workflows.runs.model.common import load_artifact, relative_to_root, resolve_path

from mlp_replacement.artifacts import atomic_json, fingerprint, sha256_file
from mlp_replacement.config import deep_merge, make_model_config
from mlp_replacement.data import PackedTokenCache
from mlp_replacement.model import discover_mlp_blocks, load_model_and_tokenizer
from mlp_replacement.runlog import environment_record

from ..shared import (
    build_local_data,
    resolve_source_asset,
    source_allocation,
    source_milestone,
    source_operator_rows,
    source_pre_recovery,
    validate_swiglu3_contract,
)


SEARCH_WORKFLOW = "swiglu-5-search"
CONFIRMATION_WORKFLOW = "swiglu-5-confirmation"
SCHEMA_VERSION = 1
CANDIDATE_DEFINITIONS = {
    "S5-C0": {
        "name": "Legacy allocation control",
        "short_label": "Legacy control",
        "initialization": "Exact SwiGLU-3 fitted operators",
        "allocation": "SwiGLU-3 ranked widths",
        "question": (
            "How much does the stronger recovery recipe improve the original "
            "SwiGLU-3 construction?"
        ),
    },
    "S5-C1": {
        "name": "Output-reconstructed initialization",
        "short_label": "Output reconstruction",
        "initialization": "Output-aware reconstruction",
        "allocation": "SwiGLU-3 ranked widths",
        "question": "Does a better local initialization help without changing widths?",
    },
    "S5-C2": {
        "name": "Discrete layer allocation",
        "short_label": "Discrete allocation",
        "initialization": "Legacy teacher-neuron subset",
        "allocation": "Discrete per-layer width optimization",
        "question": "Does concentrating compression in tolerant layers improve recovery?",
    },
    "S5-C3": {
        "name": "Reconstruction plus discrete allocation",
        "short_label": "Combined strategy",
        "initialization": "Output-aware reconstruction",
        "allocation": "Discrete per-layer width optimization",
        "question": "Are reconstruction and discrete allocation complementary?",
    },
    "S5-C4": {
        "name": "Composition-aware refinement",
        "short_label": "Composition-aware",
        "initialization": "Refit on compressed-model activation context",
        "allocation": "Widths of the best pre-recovery parent",
        "question": "Does fitting in the assembled student's context reduce composition error?",
    },
}
RECOVERY_PROTOCOL_DEFINITION = {
    "name": "Replacement-only teacher distillation",
    "objective": "Teacher-to-student KL at temperature 1",
    "learning_rate": "Constant 3e-5",
    "trainable_scope": "Compressed SwiGLU replacements only",
    "effective_tokens_per_update": 2048,
}
METRIC_DEFINITIONS = {
    "recovery_validation_kl": (
        "Fixed teacher-to-student KL at temperature 1; lower is better and dense is zero."
    ),
    "wikitext_validation_perplexity": (
        "WikiText-2 validation perplexity on the fixed workflow split; lower is better."
    ),
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def default_output(workflow, target=None):
    leaf = "search" if workflow == SEARCH_WORKFLOW else "confirmation"
    suffix = f"-target-{target}" if target is not None else ""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path("data/results/workflows/model/swiglu-5") / leaf / (
        f"{workflow}{suffix}-{stamp}.json"
    )


def asset_directory(output):
    output = Path(output)
    return output.with_suffix("").with_name(output.stem + ".assets")


@dataclass
class SwiGLU5Context:
    workflow: str
    output: Path
    asset_dir: Path
    settings: dict
    run_fingerprint: str
    artifact: dict
    source_path: Path | None = None
    source: dict | None = None
    search_path: Path | None = None
    search: dict | None = None
    model: object | None = None
    tokenizer: object | None = None
    device: object | None = None
    model_dtype: object | None = None
    blocks: dict | None = None
    data: dict | None = None
    resumed: bool = False

    @property
    def sidecar(self):
        return self.output.with_suffix(".run.json")

    def persist(self, stage=None):
        self.artifact["updated_at_utc"] = utc_now()
        atomic_json(self.output, self.artifact)
        atomic_json(
            self.sidecar,
            {
                "schema_version": 1,
                "workflow": self.workflow,
                "status": self.artifact["status"],
                "current_stage": stage,
                "artifact_path": relative_to_root(self.output),
                "asset_directory": relative_to_root(self.asset_dir),
                "run_fingerprint": self.run_fingerprint,
                "updated_at_utc": self.artifact["updated_at_utc"],
                "error": self.artifact.get("error"),
            },
        )


def strict_source_assets(source, source_path, calibration_pairs, targets):
    assets = {}
    packed = source["results"]["recovery"]["packed_token_cache"]
    token_path = resolve_source_asset(packed["path"], source_path)
    assets["packed_tokens"] = {
        "path": relative_to_root(token_path),
        "sha256": sha256_file(token_path),
        "fingerprint": packed["fingerprint"],
    }
    manifest_path = token_path.with_suffix(token_path.suffix + ".json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"SwiGLU-3 token manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("fingerprint") != packed["fingerprint"]:
        raise ValueError("SwiGLU-3 packed-token manifest fingerprint differs")
    assets["packed_token_manifest"] = {
        "path": relative_to_root(manifest_path),
        "sha256": sha256_file(manifest_path),
    }
    assets["operators"] = {}
    for target in targets:
        key = str(float(target))
        rows, unused_allocation = source_operator_rows(source, calibration_pairs, key)
        assets["operators"][key] = {}
        for layer, row in sorted(rows.items()):
            path = resolve_source_asset(row["state_path"], source_path)
            assets["operators"][key][str(layer)] = {
                "path": relative_to_root(path),
                "sha256": sha256_file(path),
            }
    return assets, token_path


def validate_search_resume_artifact(context, expected_provenance):
    artifact = context.artifact
    if artifact.get("workflow") != SEARCH_WORKFLOW:
        raise ValueError("Resume artifact is not a SwiGLU-5 search")
    if artifact.get("status") != "failed":
        raise ValueError("SwiGLU-5 search resume requires a failed artifact")
    if artifact.get("run_fingerprint") != context.run_fingerprint:
        raise ValueError("Resume configuration or SwiGLU-3 source changed")
    if artifact.get("configuration") != context.settings:
        raise ValueError("Resume artifact configuration differs")

    provenance = artifact.get("provenance", {})
    if provenance.get("configuration", {}).get("sha256") != expected_provenance[
        "configuration"
    ]["sha256"]:
        raise ValueError("Resume configuration file changed")
    source_record = provenance.get("swiglu_3", {})
    expected_source = expected_provenance["swiglu_3"]
    if source_record.get("sha256") != expected_source["sha256"]:
        raise ValueError("Resume SwiGLU-3 source artifact changed")
    if source_record.get("assets") != expected_source["assets"]:
        raise ValueError("Resume SwiGLU-3 source assets changed")
    if not context.asset_dir.is_dir():
        raise FileNotFoundError(
            f"SwiGLU-5 resume asset directory is missing: {context.asset_dir}"
        )

    results = artifact.get("results", {})
    for candidates in results.get("candidates", {}).values():
        for candidate in candidates.values():
            recovery = candidate.get("recovery", {})
            if int(recovery.get("tokens_seen", 0)) != 0:
                raise ValueError(
                    "This resume path only supports failures before recovery training"
                )

    validated_files = 0
    validated_bytes = 0
    asset_root = context.asset_dir.resolve()
    for row in results.get("local_fitting", []):
        if not bool(row.get("state_retained", True)):
            continue
        path = resolve_source_asset(row["state_path"], context.output).resolve()
        try:
            path.relative_to(asset_root)
        except ValueError as error:
            raise ValueError(
                f"Resume operator state lies outside the run assets: {path}"
            ) from error
        if sha256_file(path) != row["state_sha256"]:
            raise ValueError(f"Resume operator state changed: {path}")
        validated_files += 1
        validated_bytes += path.stat().st_size

    prior_updated = artifact.get("updated_at_utc")
    prior_elapsed = None
    if artifact.get("created_at_utc") and prior_updated:
        prior_elapsed = (
            datetime.fromisoformat(prior_updated)
            - datetime.fromisoformat(artifact["created_at_utc"])
        ).total_seconds()
    results.setdefault("continuations", []).append(
        {
            "resumed_at_utc": utc_now(),
            "prior_status": artifact["status"],
            "prior_error": deepcopy(artifact.get("error")),
            "prior_elapsed_wall_seconds": prior_elapsed,
            "validated_local_fit_states": validated_files,
            "validated_local_fit_bytes": validated_bytes,
        }
    )
    return {
        "validated_local_fit_states": validated_files,
        "validated_local_fit_bytes": validated_bytes,
    }


def prepare_search_context(
    settings,
    config_path,
    source_path,
    output,
    resume=False,
):
    if resume and output is None:
        raise ValueError("SwiGLU-5 search --resume requires the original --output")
    source_path = resolve_path(Path(source_path or settings["references"]["swiglu_3_artifact"]))
    source = load_artifact(source_path, 1, "completed SwiGLU-3 source")
    effective = deep_merge(source["configuration"], settings)
    effective["workflow"] = SEARCH_WORKFLOW
    contract = validate_swiglu3_contract(effective, source)
    targets = effective["compatibility"]["target_mlp_removals"]
    source_assets, unused_token_path = strict_source_assets(
        source,
        source_path,
        int(effective["references"]["selected_calibration_pairs"]),
        targets,
    )
    provenance = {
        "configuration": {
            "path": relative_to_root(resolve_path(Path(config_path))),
            "sha256": sha256_file(resolve_path(Path(config_path))),
        },
        "swiglu_3": {
            "path": relative_to_root(source_path),
            "sha256": sha256_file(source_path),
            "contract": contract,
            "assets": source_assets,
        },
    }
    run_fingerprint = fingerprint(
        {"configuration": effective, "source_sha256": provenance["swiglu_3"]["sha256"]}
    )
    output = resolve_path(Path(output or default_output(SEARCH_WORKFLOW)))
    asset_dir = asset_directory(output)
    if resume:
        artifact = load_artifact(output, SCHEMA_VERSION, "failed SwiGLU-5 search")
        context = SwiGLU5Context(
            SEARCH_WORKFLOW,
            output,
            asset_dir,
            effective,
            run_fingerprint,
            artifact,
            source_path=source_path,
            source=source,
            resumed=True,
        )
        validation = validate_search_resume_artifact(context, provenance)
        context.artifact["status"] = "running"
        context.artifact["error"] = None
        context.artifact["definitions"] = {
            "candidates": deepcopy(CANDIDATE_DEFINITIONS),
            "recovery_protocol": deepcopy(RECOVERY_PROTOCOL_DEFINITION),
            "metrics": deepcopy(METRIC_DEFINITIONS),
        }
        for candidates in context.artifact["results"].get("candidates", {}).values():
            for candidate_id, candidate in candidates.items():
                candidate["candidate_name"] = CANDIDATE_DEFINITIONS[candidate_id][
                    "name"
                ]
        context.artifact["results"]["resume_validation"] = validation
        context.persist("resume_validation")
        return context
    if output.exists() or asset_dir.exists():
        raise FileExistsError(f"SwiGLU-5 output already exists: {output}")
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "workflow": SEARCH_WORKFLOW,
        "experiment_family": "swiglu-5",
        "experiment_class": "homogeneous-swiglu-global-recovery",
        "status": "running",
        "created_at_utc": utc_now(),
        "run_fingerprint": run_fingerprint,
        "environment": environment_record(),
        "configuration": effective,
        "provenance": provenance,
        "definitions": {
            "candidates": deepcopy(CANDIDATE_DEFINITIONS),
            "recovery_protocol": deepcopy(RECOVERY_PROTOCOL_DEFINITION),
            "metrics": deepcopy(METRIC_DEFINITIONS),
        },
        "results": {
            "published_swiglu_3": {},
            "dense_baseline": None,
            "data": {},
            "teacher_hidden_cache": {},
            "kernel_calibration": {},
            "width_curves": {"legacy_subset": [], "output_aware": []},
            "local_fitting": [],
            "candidates": {},
            "selection": {},
            "runtime": [],
        },
        "error": None,
    }
    context = SwiGLU5Context(
        SEARCH_WORKFLOW,
        output,
        asset_dir,
        effective,
        run_fingerprint,
        artifact,
        source_path=source_path,
        source=source,
    )
    context.persist("prepare")
    return context


def load_search_resources(context):
    model_config = make_model_config(context.settings["model"])
    context.model, context.tokenizer = load_model_and_tokenizer(model_config)
    context.device = next(context.model.parameters()).device
    context.model_dtype = next(context.model.parameters()).dtype
    context.blocks = {block.index: block for block in discover_mlp_blocks(context.model)}
    expected = set(int(value) for value in context.settings["compatibility"]["eligible_layers"])
    if set(context.blocks) != set(range(int(context.settings["model"]["num_layers"]))) or not expected <= set(context.blocks):
        raise ValueError("Loaded model topology differs from the SwiGLU-3 contract")
    if any(
        int(context.blocks[layer].module.up_proj.out_features)
        != int(context.settings["model"]["intermediate_size"])
        for layer in context.blocks
    ):
        raise ValueError("Loaded model MLP widths differ from the pinned contract")
    input_embeddings = context.model.get_input_embeddings()
    output_embeddings = context.model.get_output_embeddings()
    if (
        bool(context.settings["model"].get("tie_word_embeddings"))
        and input_embeddings.weight.data_ptr() != output_embeddings.weight.data_ptr()
    ):
        raise ValueError("Loaded model does not preserve the required tied vocabulary table")
    context.data = build_local_data(context)
    torch.manual_seed(int(context.settings["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(context.settings["seed"]))
    return model_config


def packed_source_cache(context):
    record = context.source["results"]["recovery"]["packed_token_cache"]
    path = resolve_source_asset(record["path"], context.source_path)
    expected_bytes = int(record["token_count"]) * 4
    if path.stat().st_size != expected_bytes:
        raise ValueError("SwiGLU-3 packed-token byte extent differs")
    return PackedTokenCache(
        path=path,
        token_count=int(record["token_count"]),
        sequence_length=int(record["sequence_length"]),
        fingerprint=str(record["fingerprint"]),
    )


def import_swiglu3_evidence(context):
    imported = {}
    for target in context.settings["compatibility"]["target_mlp_removals"]:
        key = str(float(target))
        trajectory = context.source["results"]["recovery"]["trajectories"][key]
        imported[key] = {
            "configuration": {
                "learning_rate": context.source["configuration"]["recovery"]["learning_rate"],
                "optimizer": context.source["configuration"]["recovery"]["optimizer"],
                "scope": "replacement_only",
            },
            "allocation": source_allocation(context.source, key),
            "pre_recovery": source_pre_recovery(context.source, key),
            "validation_history": deepcopy(trajectory["validation_history"]),
            "milestones": [
                source_milestone(context.source, key, 10_000_000),
                source_milestone(context.source, key, 100_000_000),
            ],
            "selected_checkpoint": deepcopy(trajectory.get("selected_checkpoint")),
            "post_recovery": deepcopy(trajectory.get("post_recovery_model_evaluation")),
            "elapsed_seconds": trajectory.get("elapsed_seconds"),
            "memory": deepcopy(trajectory.get("memory")),
        }
    context.artifact["results"]["published_swiglu_3"] = imported
    context.persist("import_swiglu_3")


def storage_preflight(context):
    """Conservatively bound the low-storage search before local fitting."""

    context.asset_dir.mkdir(parents=True, exist_ok=True)
    hidden = int(context.settings["model"]["hidden_size"])
    dense_width = int(context.settings["model"]["intermediate_size"])
    layers = len(context.settings["compatibility"]["eligible_layers"])
    widths = {
        min(dense_width, max(1, round(dense_width * float(ratio))))
        for ratio in context.settings["allocation"]["width_ratios"]
        if float(ratio) < 1.0
    }
    curve_state_parameters = layers * sum(
        (3 * hidden * width) + (3 * hidden * width + hidden)
        for width in widths
    )
    targets = [
        float(value)
        for value in context.settings["compatibility"]["target_mlp_removals"]
    ]
    c1_state_parameters = 0
    per_target_recovery_parameters = {}
    original_eligible = layers * 3 * hidden * dense_width
    for target in targets:
        allocation = source_allocation(context.source, str(float(target)))
        source_output_aware = sum(
            3 * hidden * int(row["replacement_width"]) + hidden
            for row in allocation
        )
        c1_state_parameters += source_output_aware
        requested_retained = original_eligible - round(
            original_eligible * target
        )
        per_target_recovery_parameters[str(target)] = max(
            requested_retained,
            source_output_aware,
        )
    composition_state_parameters = sum(per_target_recovery_parameters.values())
    boundary_state_parameters = (
        2 * len(targets) * (3 * hidden * dense_width + hidden)
    )
    local_state_bytes = 4 * (
        curve_state_parameters
        + c1_state_parameters
        + composition_state_parameters
        + boundary_state_parameters
    )
    widest_target_parameters = max(per_target_recovery_parameters.values())
    transient_checkpoint_bytes = 12 * 4 * widest_target_parameters
    selected_initial_state_bytes = 4 * 4 * sum(
        per_target_recovery_parameters.values()
    )
    hidden_cache_bytes = (
        int(context.settings["teacher_hidden_cache"]["target_tokens"])
        * hidden
        * 2
    )
    recovery_peak_bytes = (
        hidden_cache_bytes
        + transient_checkpoint_bytes
        + selected_initial_state_bytes
    )
    estimated_peak_bytes = max(local_state_bytes, recovery_peak_bytes)
    reserve_fraction = float(
        context.settings["storage_preflight"]["reserve_fraction"]
    )
    required_bytes = math.ceil(estimated_peak_bytes * (1.0 + reserve_fraction))
    free_bytes = shutil.disk_usage(context.asset_dir.parent).free
    record = {
        "policy": "minimal_exact_stage_resume",
        "hidden_cache_bytes": hidden_cache_bytes,
        "local_fit_state_bytes_upper_bound": local_state_bytes,
        "selected_initial_state_bytes_upper_bound": selected_initial_state_bytes,
        "transient_checkpoint_bytes_upper_bound": transient_checkpoint_bytes,
        "recovery_peak_bytes_upper_bound": recovery_peak_bytes,
        "estimated_peak_bytes": estimated_peak_bytes,
        "reserve_fraction": reserve_fraction,
        "required_free_bytes": required_bytes,
        "observed_free_bytes": free_bytes,
        "passed": free_bytes >= required_bytes,
    }
    context.artifact["results"]["storage_preflight"] = record
    context.persist("storage_preflight")
    if not record["passed"]:
        raise OSError(
            "SwiGLU-5 storage preflight requires "
            f"{required_bytes / 1024**3:.1f} GiB free but found "
            f"{free_bytes / 1024**3:.1f} GiB"
        )
    return record


def record_stage_runtime(context, stage, started):
    context.artifact["results"]["runtime"].append(
        {"stage": stage, "seconds": perf_counter() - started}
    )
    context.persist(stage)


def prepare_confirmation_context(
    settings,
    config_path,
    search_path,
    target,
    output,
):
    target = float(target)
    if target not in [float(value) for value in settings["compatibility"]["allowed_targets"]]:
        raise ValueError("Confirmation --target must be exactly 0.2 or 0.5")
    target_key = str(target)
    search_path = resolve_path(Path(search_path))
    search = load_artifact(search_path, 1, "completed SwiGLU-5 search")
    if search.get("workflow") != SEARCH_WORKFLOW or search.get("status") != "completed":
        raise ValueError("Confirmation requires a completed SwiGLU-5 search artifact")
    selection = search["results"]["selection"].get(target_key)
    if not selection or not selection.get("winner_candidate_id"):
        raise ValueError(f"Search artifact has no selected winner for target {target_key}")
    source_record = search["provenance"]["swiglu_3"]
    source_path = resolve_path(Path(source_record["path"]))
    if not source_path.is_file():
        source_path = resolve_source_asset(source_record["path"], search_path)
    if sha256_file(source_path) != source_record["sha256"]:
        raise ValueError("Original SwiGLU-3 source artifact changed")
    source = load_artifact(source_path, 1, "original completed SwiGLU-3 source")
    validate_swiglu3_contract(search["configuration"], source)
    source_assets = source_record["assets"]
    for asset_name in ("packed_tokens", "packed_token_manifest"):
        asset_record = source_assets[asset_name]
        asset_path = resolve_source_asset(asset_record["path"], source_path)
        if sha256_file(asset_path) != asset_record["sha256"]:
            raise ValueError(
                f"Original SwiGLU-3 {asset_name.replace('_', ' ')} changed"
            )
    endpoint_record = selection["winner_endpoint"]
    endpoint_path = resolve_source_asset(endpoint_record["path"], search_path)
    if sha256_file(endpoint_path) != endpoint_record["sha256"]:
        raise ValueError("Selected SwiGLU-5 endpoint checkpoint changed")
    provenance = {
        "configuration": {
            "path": relative_to_root(resolve_path(Path(config_path))),
            "sha256": sha256_file(resolve_path(Path(config_path))),
        },
        "search": {
            "path": relative_to_root(search_path),
            "sha256": sha256_file(search_path),
            "run_fingerprint": search["run_fingerprint"],
        },
        "swiglu_3": deepcopy(source_record),
        "selected_endpoint": {
            **deepcopy(endpoint_record),
            "resolved_path": relative_to_root(endpoint_path),
        },
    }
    effective = deep_merge(search["configuration"], settings)
    effective["workflow"] = CONFIRMATION_WORKFLOW
    effective["target"] = target
    effective["selected_candidate_id"] = selection["winner_candidate_id"]
    run_fingerprint = fingerprint(
        {
            "configuration": effective,
            "search_sha256": provenance["search"]["sha256"],
            "endpoint_sha256": endpoint_record["sha256"],
        }
    )
    output = resolve_path(Path(output or default_output(CONFIRMATION_WORKFLOW, target_key)))
    asset_dir = asset_directory(output)
    if output.exists() or asset_dir.exists():
        raise FileExistsError(f"Confirmation output already exists: {output}")
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "workflow": CONFIRMATION_WORKFLOW,
        "experiment_family": "swiglu-5",
        "experiment_class": "homogeneous-swiglu-global-recovery",
        "status": "running",
        "created_at_utc": utc_now(),
        "run_fingerprint": run_fingerprint,
        "environment": environment_record(),
        "configuration": effective,
        "provenance": provenance,
        "definitions": {
            "candidates": deepcopy(CANDIDATE_DEFINITIONS),
            "recovery_protocol": deepcopy(RECOVERY_PROTOCOL_DEFINITION),
            "metrics": deepcopy(METRIC_DEFINITIONS),
        },
        "results": {
            "target": target,
            "selected_candidate_id": selection["winner_candidate_id"],
            "selected_candidate_name": CANDIDATE_DEFINITIONS[
                selection["winner_candidate_id"]
            ]["name"],
            "kernel_calibration": {},
            "trajectory": {
                "status": "pending",
                "tokens_seen": 0,
                "optimizer_updates": 0,
                "validation_history": [],
                "full_evaluations": [],
                "checkpoint_policy": "final_and_best_weights_only_no_resume",
            },
            "paired_swiglu_3_comparison": [],
            "runtime": [],
        },
        "error": None,
    }
    context = SwiGLU5Context(
        CONFIRMATION_WORKFLOW,
        output,
        asset_dir,
        effective,
        run_fingerprint,
        artifact,
        source_path=source_path,
        source=source,
        search_path=search_path,
        search=search,
    )
    context.persist("prepare")
    return context
