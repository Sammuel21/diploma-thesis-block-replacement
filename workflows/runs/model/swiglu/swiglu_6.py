"""SwiGLU-6 provenance and artifact boundaries; earlier families are read-only."""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from workflows.runs.model.common import PROJECT_ROOT, resolve_path
from mlp_replacement.artifacts import content_digest, file_digest, read_json, write_json_atomic
from .shared import resolve_source_asset


DEFAULT_CONFIG = Path("workflows/configs/model/swiglu/swiglu-6.json")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def code_hashes():
    """Record maintained Python sources used by the scientific process."""

    paths = [*PROJECT_ROOT.joinpath("src/mlp_replacement").rglob("*.py"),
             *PROJECT_ROOT.joinpath("workflows/runs").rglob("*.py")]
    return {path.relative_to(PROJECT_ROOT).as_posix(): file_digest(path) for path in sorted(paths)}


def load_settings(path):
    settings = read_json(resolve_path(Path(path)))
    if settings.get("workflow") != "swiglu-6" or settings.get("schema_version") != 1:
        raise ValueError("Expected a schema-1 SwiGLU-6 configuration")
    recovery = settings["recovery"]
    fixed = {"start_tokens": 5001216, "learning_rate": 3e-5, "weight_decay": 0.0,
             "temperature": 1.0, "ce_weight": 0.0, "scheduler": "constant",
             "optimizer_backend": "fused", "microbatch_sequences": 8,
             "gradient_accumulation_steps": 2, "effective_batch_tokens": 2048,
             "segment_endpoints": [100000000, 1000000000, 2000000000]}
    if any(recovery.get(key) != value for key, value in fixed.items()):
        raise ValueError("This SwiGLU-6 protocol fixes the original recipe and 100M/1B/2B boundaries")
    if settings["targets"] != [0.2, 0.5] or settings["candidate_id"] != "S5-C2":
        raise ValueError("SwiGLU-6 currently confirms the two existing S5-C2 models")
    if settings["data"]["sequence_length"] != 128 or settings["data"]["target_tokens"] != 2_000_000_000:
        raise ValueError("SwiGLU-6 requires a 2B-token stream of length-128 sequences")
    if settings["seed"] != 21 or settings["data"]["first_shard"] != 1 or settings["data"]["historical_prefix_tokens"] != 100_000_000:
        raise ValueError("Preserve the original seed, excluded calibration shard, and 100M prefix")
    evaluation = settings["evaluation"]
    if (evaluation["contexts"] != [128, 2048] or evaluation["strides"] != [64, 1024]
            or evaluation["tasks"] != ["piqa", "arc_easy", "arc_challenge", "winogrande", "hellaswag"]
            or evaluation["seed"] != settings["seed"]):
        raise ValueError("Preserve the declared full-corpus and downstream protocol")
    return settings


def new_artifact(output, settings, stage, resume=False):
    output = resolve_path(Path(output))
    if output.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite {output}")
        artifact = read_json(output)
        if artifact.get("workflow") != "swiglu-6" or artifact["configuration"] != settings or artifact["stage"] != stage:
            raise ValueError("Resume configuration/workflow differs")
        return output, artifact
    if resume:
        raise FileNotFoundError("Resume requires an existing explicit output")
    if output.with_suffix(".run.json").exists():
        raise FileExistsError("Run sidecar already exists; choose a new output")
    assets = asset_directory(output)
    if assets.exists():
        raise FileExistsError(f"Assets already exist: {assets}")
    artifact = {"schema_version": 1, "workflow": "swiglu-6", "stage": stage,
                "status": "running", "created_at_utc": utc_now(),
                "configuration": deepcopy(settings), "results": {}, "error": None}
    persist(output, artifact, "initializing")
    return output, artifact


def asset_directory(output):
    output = Path(output)
    return output.with_name(output.stem + ".assets")


def persist(output, artifact, active_stage):
    artifact["updated_at_utc"] = utc_now()
    write_json_atomic(output, artifact)
    write_json_atomic(Path(output).with_suffix(".run.json"), {
        "workflow": "swiglu-6", "status": artifact["status"],
        "stage": active_stage, "updated_at_utc": artifact["updated_at_utc"],
        "error": artifact.get("error"),
    })


def sources(settings):
    """Verify historical metadata and endpoint files without rewriting them."""

    import torch

    paths = settings["sources"]
    search_path = resolve_path(Path(paths["search"]))
    source_path = resolve_path(Path(paths["swiglu_3"]))
    search, source = read_json(search_path), read_json(source_path)
    if search.get("status") != "completed" or search.get("workflow") != "swiglu-5-search":
        raise ValueError("A completed SwiGLU-5 search is required")
    if file_digest(source_path) != search["provenance"]["swiglu_3"]["sha256"]:
        raise ValueError("The original SwiGLU-3 artifact changed")
    for key in ("model_id", "revision", "tokenizer_revision", "hidden_size", "intermediate_size", "num_layers"):
        if settings["model"][key] != search["configuration"]["model"][key]:
            raise ValueError(f"Source model contract differs: {key}")
    records = {"search": {"path": str(search_path), "sha256": file_digest(search_path)},
               "swiglu_3": {"path": str(source_path), "sha256": file_digest(source_path)}, "targets": {}}
    for target in settings["targets"]:
        key = str(target)
        selection = search["results"]["selection"][key]
        if selection["winner_candidate_id"] != settings["candidate_id"]:
            raise ValueError("Historical selected candidate is not S5-C2")
        candidate = search["results"]["candidates"][key][settings["candidate_id"]]
        endpoint = selection["winner_endpoint"]
        endpoint_path = resolve_source_asset(endpoint["path"], search_path)
        if file_digest(endpoint_path) != endpoint["sha256"]:
            raise ValueError("Historical search checkpoint changed")
        state = torch.load(endpoint_path, map_location="cpu", weights_only=False)
        recipe = {name: value for name, value in candidate.items() if name != "recovery"}
        if state["candidate_fingerprint"] != content_digest(recipe) or state["run_fingerprint"] != search["run_fingerprint"]:
            raise ValueError("Search checkpoint provenance differs")
        if state["tokens_seen"] != settings["recovery"]["start_tokens"] or "optimizer_state" not in state:
            raise ValueError("Exact 5M checkpoint with optimizer state is required")
        confirmation_path = resolve_path(Path(paths["confirmations"][key]))
        confirmation = read_json(confirmation_path)
        if confirmation.get("status") != "completed" or confirmation.get("workflow") != "swiglu-5-confirmation":
            raise ValueError("Completed historical confirmations are required")
        if confirmation["results"]["target"] != target or confirmation["results"]["selected_candidate_id"] != settings["candidate_id"]:
            raise ValueError("Historical confirmation belongs to a different model")
        records["targets"][key] = {
            "endpoint": {"path": str(endpoint_path), "sha256": endpoint["sha256"]},
            "confirmation": {"path": str(confirmation_path), "sha256": file_digest(confirmation_path)},
            "candidate_fingerprint": state["candidate_fingerprint"],
            "packed_token_fingerprint": state["packed_token_fingerprint"],
        }
        del state
    packed = search["provenance"]["swiglu_3"]["assets"]["packed_tokens"]
    prefix = resolve_source_asset(packed["path"], source_path)
    if file_digest(prefix) != packed["sha256"]:
        raise ValueError("Historical recovery stream changed")
    records["prefix"] = {"path": str(prefix), "sha256": packed["sha256"]}
    return search, source, records


def load_prepared(path, settings):
    path = resolve_path(Path(path))
    prepared = read_json(path)
    if prepared.get("workflow") != "swiglu-6" or prepared.get("stage") != "prepare" or prepared.get("status") != "completed":
        raise ValueError("A completed SwiGLU-6 preparation artifact is required")
    if prepared["configuration"] != settings:
        raise ValueError("Preparation and execution configurations differ")
    return path, prepared


def legacy_context(search, tokenizer):
    return SimpleNamespace(settings=deepcopy(search["configuration"]), tokenizer=tokenizer)
