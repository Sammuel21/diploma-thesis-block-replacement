"""Speed-1: generation speed of saved deployment bundles, measured in one job on one GPU."""

import argparse
import re
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

from mlp_replacement.artifacts import (
    contained_path, content_digest, file_digest, read_json, write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = ROOT / "workflows/configs/model/baseline/speed-1.json"
# Quantization-1 owns the fresh-process runtime protocol; Speed-1 only schedules it.
MEASUREMENT_MODULE = "workflows.runs.model.baseline.quantization"
PASS_ORDERS = ("forward", "reverse")
SUMMARY_METRICS = ("time_to_first_token_ms", "output_tokens_per_second",
                   "decode_tokens_per_second", "peak_gpu_allocated_bytes")
SOURCE_FILES = (
    "workflows/runs/model/baseline/speed.py",
    "workflows/runs/model/baseline/quantization.py",
    "workflows/configs/model/baseline/quantization-1.json",
    "src/mlp_replacement/evaluation/generation.py",
    "src/mlp_replacement/evaluation/quantized_bundles.py",
    "src/mlp_replacement/evaluation/bundles.py",
)
LABEL_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def relative_input(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Speed-1 inputs must be relative to the source root: {value}")
    return value


def load_settings(path):
    settings = read_json(resolve_path(path))
    if settings.get("workflow") != "speed-1" or settings.get("schema_version") != 1:
        raise ValueError("Expected a Speed-1 configuration")
    passes = settings["passes"]
    if not passes or len(set(passes)) != len(passes) or any(value not in PASS_ORDERS for value in passes):
        raise ValueError("Speed-1 passes must be distinct values from forward/reverse")
    measurement = settings["measurement"]
    relative_input(measurement["prepared"])
    if not SHA256_PATTERN.match(measurement["prepared_sha256"]):
        raise ValueError("Speed-1 pins the Quantization-1 preparation by sha256")
    labels = [model["label"] for model in settings["models"]]
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("Speed-1 needs at least two models with distinct labels")
    for model in settings["models"]:
        if not LABEL_PATTERN.match(model["label"]):
            raise ValueError(f"Unsafe model label: {model['label']}")
        relative_input(model["bundle"])
        if not SHA256_PATTERN.match(model["bundle_manifest_sha256"]):
            raise ValueError(f"Model {model['label']} must pin its bundle manifest sha256")
    if settings["reference_label"] not in labels:
        raise ValueError("The reference label must name a measured model")
    return settings


def measurement_plan(settings):
    """Every model once per pass; the reverse pass exposes drift during the job."""

    labels = [model["label"] for model in settings["models"]]
    plan = []
    for pass_name in settings["passes"]:
        order = labels if pass_name == "forward" else labels[::-1]
        plan.extend((pass_name, label) for label in order)
    return plan


def verify_inputs(settings, source_root):
    """Check pinned bundles and prompts before any GPU work starts."""

    prepared = contained_path(source_root, settings["measurement"]["prepared"])
    if file_digest(prepared) != settings["measurement"]["prepared_sha256"]:
        raise ValueError(f"Quantization-1 preparation changed: {prepared}")
    bundles = {}
    for model in settings["models"]:
        bundle = contained_path(source_root, model["bundle"])
        if file_digest(bundle / "bundle.json") != model["bundle_manifest_sha256"]:
            raise ValueError(f"Bundle manifest changed for {model['label']}: {bundle}")
        bundles[model["label"]] = bundle
    return prepared, bundles


def source_hashes():
    return {path: file_digest(ROOT / path) for path in SOURCE_FILES}


def measure(directory, pass_name, label, bundle, prepared, protocol_config, expected_sha256):
    """Run one bundle in a fresh process; reuse a verified earlier measurement on resume."""

    output = contained_path(directory, f"measurements/{pass_name}-{label}")
    runtime_path = output / "runtime.json"
    if runtime_path.exists():
        runtime = read_json(runtime_path)
        if runtime["bundle_manifest_sha256"] != expected_sha256:
            raise ValueError(f"Stored measurement belongs to another bundle: {runtime_path}")
        return runtime
    if output.exists():
        # A measurement interrupted before runtime.json was committed is discarded and repeated.
        shutil.rmtree(output)
    command = [sys.executable, "-m", MEASUREMENT_MODULE, "measure-runtime",
               "--config", str(protocol_config), "--bundle", str(bundle),
               "--prepared", str(prepared), "--output-dir", str(output)]
    process = subprocess.Popen(command, cwd=ROOT)
    try:
        if process.wait() != 0:
            error_path = output / "runtime-error.json"
            detail = read_json(error_path)["message"] if error_path.exists() else "see worker stderr"
            raise RuntimeError(f"Runtime measurement failed for {pass_name}/{label}: {detail}")
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    runtime = read_json(runtime_path)
    if runtime["bundle_manifest_sha256"] != expected_sha256:
        raise ValueError(f"Runtime measurement belongs to another bundle: {pass_name}/{label}")
    return runtime


def workload_key(workload):
    return f"b{workload['batch_size']}-p{workload['prompt_tokens']}-g{workload['generated_tokens']}"


def summarize(settings, measurements):
    """Average the per-pass medians and report how far the passes disagree."""

    fingerprints = sorted({runtime["execution_fingerprint"] for runtime in measurements.values()})
    devices = sorted({runtime["execution"]["device_uuid"] for runtime in measurements.values()})
    reference = settings["reference_label"]
    rows = {}
    for model in settings["models"]:
        label = model["label"]
        runs = [measurements[(pass_name, label)] for pass_name in settings["passes"]]
        workloads = {}
        for index, workload in enumerate(runs[0]["generation"]["workloads"]):
            key = workload_key(workload)
            if any(workload_key(run["generation"]["workloads"][index]) != key for run in runs):
                raise ValueError(f"Passes measured different workloads for {label}")
            metrics = {}
            for metric in SUMMARY_METRICS:
                values = [run["generation"]["workloads"][index]["median"][metric] for run in runs]
                value = mean(values)
                metrics[metric] = {"value": value, "pass_values": values,
                                   "pass_spread": (max(values) - min(values)) / value if value else 0.0}
            workloads[key] = metrics
        resident = [run["resident_memory"]["gpu_allocated_delta_bytes"] for run in runs]
        rows[label] = {"workloads": workloads, "resident_gpu_allocated_bytes": mean(resident)}
    for label, row in rows.items():
        for key, metrics in row["workloads"].items():
            for metric, record in metrics.items():
                base = rows[reference]["workloads"][key][metric]["value"]
                record["relative_to_reference"] = record["value"] / base if base else None
    spreads = [record["pass_spread"] for row in rows.values() for metrics in row["workloads"].values()
               for metric, record in metrics.items() if metric != "peak_gpu_allocated_bytes"]
    return {"reference_label": reference, "passes": settings["passes"],
            "single_execution_fingerprint": len(fingerprints) == 1,
            "execution_fingerprints": fingerprints, "device_uuids": devices,
            "max_timing_pass_spread": max(spreads) if spreads else None, "models": rows}


def write_report(directory, summary):
    labels = list(summary["models"])
    keys = list(summary["models"][labels[0]]["workloads"])
    header = ["Model"]
    for key in keys:
        header += [f"TTFT {key} (ms)", f"Tokens/s {key}"]
    header += ["Peak VRAM, largest workload (GiB)", "Largest timing pass spread"]
    lines = ["# Speed-1 comparison", "",
             f"Same execution fingerprint for every measurement: {summary['single_execution_fingerprint']}. "
             f"Values are means of the {len(summary['passes'])} pass medians.", "",
             "| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for label in labels:
        workloads = summary["models"][label]["workloads"]
        cells = [label]
        for key in keys:
            cells += [f"{workloads[key]['time_to_first_token_ms']['value']:.1f}",
                      f"{workloads[key]['output_tokens_per_second']['value']:.1f}"]
        peak = max(workloads[key]["peak_gpu_allocated_bytes"]["value"] for key in keys)
        spread = max(record["pass_spread"] for metrics in workloads.values()
                     for metric, record in metrics.items() if metric != "peak_gpu_allocated_bytes")
        cells += [f"{peak / 2**30:.2f}", f"{spread:.1%}"]
        lines.append("| " + " | ".join(cells) + " |")
    temporary = Path(directory) / "comparison.md.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(Path(directory) / "comparison.md")


def start_artifact(directory, settings, identity, resume):
    from mlp_replacement.runlog import ExperimentLog, environment_record

    contract = {"configuration": settings, "identity": identity, "source_hashes": source_hashes()}
    fingerprint = content_digest(contract)
    if resume:
        artifact = read_json(directory / "result.json")
        if artifact["run_fingerprint"] != fingerprint:
            raise ValueError("Resume configuration, input, or source fingerprint changed")
        log = ExperimentLog.__new__(ExperimentLog)
        log.path = directory / "run.json"
        log.data = read_json(log.path)
        log.data.setdefault("attempts", []).append({"started_at": utc_now(),
                                                   "environment": environment_record()})
        log.data.update(status="running", error=None, finished_at=None)
        artifact.update(status="running", error=None, finished_at=None)
    else:
        if directory.exists():
            raise FileExistsError(f"Output already exists; explicit compatible resume required: {directory}")
        directory.mkdir(parents=True)
        log = ExperimentLog(directory / "run.json", settings)
        artifact = {"schema_version": 1, "workflow": "speed-1", "identity": identity,
                    "configuration": settings, "run_fingerprint": fingerprint,
                    "source_hashes": contract["source_hashes"], "status": "running",
                    "created_at": utc_now(), "results": {"measurements": {}}}
    persist(directory, artifact, log, "initialization")
    return artifact, log


def persist(directory, artifact, log, stage):
    artifact["updated_at"] = utc_now()
    artifact["current_stage"] = stage
    write_json_atomic(Path(directory) / "result.json", artifact)
    log.begin(stage)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--source-root", required=True,
                        help="Directory holding the quantization-1/ and swiglu-8/ result folders")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = load_settings(args.config)
    directory = resolve_path(args.output_dir)
    source_root = resolve_path(args.source_root)
    if directory.is_relative_to(source_root / "quantization-1") or directory.is_relative_to(source_root / "swiglu-8"):
        raise ValueError("Speed-1 output cannot be inside its read-only source results")
    prepared, bundles = verify_inputs(settings, source_root)
    protocol_config = resolve_path(settings["measurement"]["protocol_config"])
    identity = {"prepared_sha256": settings["measurement"]["prepared_sha256"],
                "bundles": {model["label"]: model["bundle_manifest_sha256"] for model in settings["models"]}}
    artifact, log = start_artifact(directory, settings, identity, args.resume)
    started = perf_counter()

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}; completed measurements are kept")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        measurements = {}
        for pass_name, label in measurement_plan(settings):
            key = f"{pass_name}-{label}"
            persist(directory, artifact, log, f"measure:{key}")
            runtime = measure(directory, pass_name, label, bundles[label], prepared, protocol_config,
                              identity["bundles"][label])
            measurements[(pass_name, label)] = runtime
            artifact["results"]["measurements"][key] = {
                "path": f"measurements/{key}/runtime.json",
                "execution_fingerprint": runtime["execution_fingerprint"],
                "device_uuid": runtime["execution"]["device_uuid"],
            }
            log.record(f"measure:{key}", artifact["results"]["measurements"][key])
        persist(directory, artifact, log, "summary")
        summary = summarize(settings, measurements)
        artifact["results"]["summary"] = summary
        write_report(directory, summary)
        log.data["workflow_seconds"] = log.data.get("workflow_seconds", 0) + perf_counter() - started
        artifact.update(status="completed", finished_at=utc_now(), current_stage=None)
        write_json_atomic(directory / "result.json", artifact)
        log.complete({"path": "result.json", "run_fingerprint": artifact["run_fingerprint"],
                      "single_execution_fingerprint": summary["single_execution_fingerprint"]})
    except BaseException as error:
        log.data["workflow_seconds"] = log.data.get("workflow_seconds", 0) + perf_counter() - started
        artifact.update(status="failed", finished_at=utc_now(),
                        error={"type": type(error).__name__, "message": str(error)})
        write_json_atomic(directory / "result.json", artifact)
        log.fail(error)
        raise


if __name__ == "__main__":
    main()
