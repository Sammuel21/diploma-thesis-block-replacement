import json
import math
import platform
import sys
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import torch


def utc_time():
    """Return a stable UTC timestamp for run records."""

    return datetime.now(timezone.utc).isoformat()


def json_value(value):
    """Convert metric dataclasses and common scalar containers to JSON values."""

    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def installed_version(package):
    """Return an installed package version without making it a hard dependency."""

    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def environment_record():
    """Capture the small software and hardware record needed to interpret a run."""

    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "torch": torch.__version__,
            "transformers": installed_version("transformers"),
            "datasets": installed_version("datasets"),
            "lm_eval": installed_version("lm_eval"),
        },
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
    }


class ExperimentLog:
    """Maintain one crash-aware JSON record for a single experiment process."""

    def __init__(self, path, config):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f"Run log already exists: {self.path}")
        started = utc_time()
        self.data = {
            "schema_version": 1,
            "run_id": self.path.stem,
            "status": "running",
            "current_stage": "initialization",
            "started_at": started,
            "updated_at": started,
            "finished_at": None,
            "config": json_value(config),
            "environment": environment_record(),
            "stages": {},
            "result": None,
            "error": None,
        }
        self.write()

    def begin(self, stage):
        """Mark the stage currently executing before expensive work begins."""

        self.data["current_stage"] = stage
        self.data["updated_at"] = utc_time()
        self.write()

    def record(self, stage, value):
        """Store one completed stage value and persist the complete run record."""

        self.data["stages"][stage] = json_value(value)
        self.data["current_stage"] = None
        self.data["updated_at"] = utc_time()
        self.write()

    def complete(self, result):
        """Store the final workflow result and mark the run complete."""

        finished = utc_time()
        self.data["status"] = "completed"
        self.data["current_stage"] = None
        self.data["updated_at"] = finished
        self.data["finished_at"] = finished
        self.data["result"] = json_value(result)
        self.write()

    def fail(self, error):
        """Record an exception and preserve all progress written before it."""

        finished = utc_time()
        self.data["status"] = "failed"
        self.data["updated_at"] = finished
        self.data["finished_at"] = finished
        self.data["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        self.write()

    def write(self):
        """Atomically replace the JSON file so interruptions do not truncate it."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(self.path)
