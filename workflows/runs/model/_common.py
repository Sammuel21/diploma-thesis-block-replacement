"""Execution plumbing shared by model-wide workflow entry points."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repository from either a checkout or staged Perun copy."""

    resolved = (start or Path.cwd()).resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "src" / "mlp_replacement").is_dir():
            return candidate
    raise RuntimeError("Could not locate the repository root")


PROJECT_ROOT = find_project_root()
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from mlp_replacement.runlog import ExperimentLog, json_value  # noqa: E402


def utc_timestamp() -> str:
    """Return a collision-resistant timestamp suitable for a filename."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def default_artifact_path(workflow: str, label: str | None = None) -> Path:
    """Choose a unique, repository-relative science-artifact path."""

    suffix = f"-{label}" if label else ""
    filename = f"{workflow}{suffix}-{utc_timestamp()}.json"
    return Path("data/results/workflows/model") / workflow / filename


def resolve_path(path: Path) -> Path:
    """Resolve repository-relative CLI paths without depending on CWD later."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def require_new_path(path: Path) -> Path:
    """Refuse to overwrite an existing experiment artifact."""

    resolved = resolve_path(path)
    if resolved.exists():
        raise FileExistsError(f"Artifact already exists: {resolved}")
    return resolved


def sidecar_path(artifact_path: Path) -> Path:
    """Derive the crash-aware run-log path for a science artifact."""

    return artifact_path.with_suffix(".run.json")


def start_run_log(artifact_path: Path, workflow: str, configuration: dict) -> ExperimentLog:
    """Create a crash-aware sidecar before expensive resources are loaded."""

    return ExperimentLog(
        sidecar_path(artifact_path),
        {
            "workflow": workflow,
            "artifact_path": str(artifact_path),
            **configuration,
        },
    )


def load_artifact(path: Path, schema_version: int, label: str) -> dict:
    """Load a prerequisite artifact and verify its schema contract."""

    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Required {label} is missing: {resolved}. Runtime artifacts under "
            "data/ are git-ignored; transfer the artifact into the submitted "
            "tree, pass its staged path explicitly, or run its prerequisite stage."
        )
    artifact = json.loads(resolved.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != schema_version:
        raise ValueError(
            f"{label} uses schema {artifact.get('schema_version')!r}; "
            f"expected {schema_version}"
        )
    return artifact


def load_workflow_config(path: Path, workflow: str) -> dict:
    """Load one explicit workflow configuration and verify its owner."""

    resolved = resolve_path(path)
    configuration = json.loads(resolved.read_text(encoding="utf-8"))
    if configuration.get("workflow") != workflow:
        raise ValueError(
            f"Configuration {resolved} belongs to "
            f"{configuration.get('workflow')!r}, not {workflow!r}"
        )
    if configuration.get("schema_version") != 1:
        raise ValueError(
            f"Configuration {resolved} uses unsupported schema "
            f"{configuration.get('schema_version')!r}"
        )
    return configuration


def write_artifact(path: Path, artifact: dict) -> None:
    """Atomically persist a notebook-compatible JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            json_value(artifact),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def json_records(frame) -> list[dict]:
    """Match the notebooks' finite, high-precision DataFrame serialization."""

    return json.loads(frame.to_json(orient="records", double_precision=15))


def report_memory(stage: str) -> None:
    """Print current and peak process RAM when Linux procfs is available."""

    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return
    status = dict(
        line.split(":", 1)
        for line in status_path.read_text(encoding="utf-8").splitlines()
        if ":" in line
    )
    rss = int(status["VmRSS"].split()[0]) / 1024**2
    peak = int(status["VmHWM"].split()[0]) / 1024**2
    print(f"{stage}: RAM {rss:.2f} GiB, peak {peak:.2f} GiB", flush=True)


def synchronize_cuda(torch_module, active_device) -> None:
    """Synchronize only when a CUDA device is active."""

    if active_device is not None and torch_module.device(active_device).type == "cuda":
        torch_module.cuda.synchronize(active_device)


def release_cuda(torch_module) -> None:
    """Release cached CUDA allocator blocks between memory-heavy stages."""

    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
