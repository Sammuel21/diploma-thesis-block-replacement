"""Execution helpers shared by the maintained homogeneous-SwiGLU runners."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path, PurePosixPath

import torch

from workflows.runs.model.common import relative_to_root, resolve_path

from mlp_replacement.artifacts import atomic_json, atomic_torch_save, fingerprint, sha256_file
from mlp_replacement.compression.reconstruction import load_operator
from mlp_replacement.config import DatasetSpec, deep_merge, make_model_config
from mlp_replacement.data import (
    contiguous_token_windows,
    load_text_dataset,
    make_token_loader,
    sample_partitioned_windows,
)
from mlp_replacement.evaluation.mixed_precision import (
    autocast_context, evaluate_lm_mixed, evaluate_teacher_cache_mixed,
    evaluate_validation_kl_mixed,
)


def resolve_source_asset(recorded_path, source_artifact_path):
    """Resolve a recorded asset directly or beside a relocated artifact."""

    direct = resolve_path(Path(recorded_path))
    if direct.is_file():
        return direct
    parts = PurePosixPath(str(recorded_path).replace("\\", "/")).parts
    asset_index = next(
        (index for index, part in enumerate(parts) if part.endswith(".assets")),
        None,
    )
    if asset_index is None:
        raise FileNotFoundError(
            f"Recorded source asset has no .assets component: {recorded_path}"
        )
    suffix = Path(*parts[asset_index + 1 :])
    source_artifact_path = Path(source_artifact_path)
    sibling = source_artifact_path.with_suffix("")
    sibling = sibling.with_name(sibling.name + ".assets") / suffix
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(
        "Required source asset is missing at both its recorded and "
        f"artifact-relative locations: {sibling}"
    )


def source_milestone(source, sparsity_key, requested_tokens):
    """Return one published SwiGLU-3 recovery milestone."""

    trajectory = source["results"]["recovery"]["trajectories"][str(sparsity_key)]
    for row in trajectory["milestones"]:
        requested = row["requested_tokens"]
        requested = requested if isinstance(requested, list) else [requested]
        if int(requested_tokens) in (int(value) for value in requested):
            return deepcopy(row)
    raise ValueError(
        f"SwiGLU-3 has no {int(requested_tokens):,}-token "
        f"milestone for sparsity {sparsity_key}"
    )


def source_pre_recovery(source, sparsity_key):
    """Return the published pre-recovery model evaluation."""

    return deepcopy(
        next(
            row
            for row in source["results"]["sparsity"]["model_evaluation"]
            if row["sparsity_key"] == str(sparsity_key)
            and row["phase"] == "pre_recovery"
        )
    )


def source_allocation(source, sparsity_key):
    """Return all published allocation rows for one sparsity target."""

    selected = [
        deepcopy(row)
        for row in source["results"]["sparsity"]["allocation"]
        if row["sparsity_key"] == str(sparsity_key)
    ]
    if not selected:
        raise ValueError(f"SwiGLU-3 has no allocation for {sparsity_key}")
    return selected


def source_operator_rows(source, calibration_pairs, sparsity_key):
    """Return exact operator-state rows for a published SwiGLU-3 target."""

    sparsity_key = str(sparsity_key)
    allocation = {
        int(row["layer"]): row for row in source_allocation(source, sparsity_key)
    }
    if sparsity_key == "0.5":
        candidates = [
            row
            for row in source["results"]["calibration"]["operator_fitting"]
            if int(row["calibration_pairs"]) == int(calibration_pairs)
        ]
    else:
        candidates = [
            row
            for row in source["results"]["sparsity"]["operator_fitting"]
            if row["sparsity_key"] == sparsity_key
            and int(row["calibration_pairs"]) == int(calibration_pairs)
        ]
    selected = {}
    for row in candidates:
        layer = int(row["layer"])
        if layer not in allocation:
            continue
        if int(row["replacement_width"]) != int(
            allocation[layer]["replacement_width"]
        ):
            continue
        selected[layer] = deepcopy(row)
    if set(selected) != set(allocation):
        missing = sorted(set(allocation) - set(selected))
        raise ValueError(
            f"SwiGLU-3 is missing exact operator states for layers {missing}"
        )
    return selected, allocation


def validate_swiglu3_contract(settings, source):
    """Validate every SwiGLU-3 field required by SwiGLU-5."""

    if (
        source.get("schema_version")
        != int(settings["references"]["required_schema_version"])
        or source.get("workflow") != "swiglu-3"
        or source.get("status") != "completed"
    ):
        raise ValueError("SwiGLU-5 requires a completed schema-1 SwiGLU-3 artifact")
    source_config = source["configuration"]
    for field in (
        "model_id",
        "revision",
        "tokenizer_revision",
        "dtype",
        "trust_remote_code",
        "hidden_size",
        "intermediate_size",
        "num_layers",
    ):
        if source_config["model"].get(field) != settings["model"].get(field):
            raise ValueError(f"SwiGLU-3 model contract differs at {field}")
    compatibility = settings["compatibility"]
    allocation = source_config["allocation"]
    if [int(value) for value in allocation["eligible_layers"]] != [
        int(value) for value in compatibility["eligible_layers"]
    ]:
        raise ValueError("SwiGLU-3 eligible layers differ")
    if [int(value) for value in allocation["protected_layers"]] != [
        int(value) for value in compatibility["protected_layers"]
    ]:
        raise ValueError("SwiGLU-3 protected layers differ")
    if int(source["results"]["calibration"]["selected_calibration_pairs"]) != int(
        settings["references"]["selected_calibration_pairs"]
    ):
        raise ValueError("SwiGLU-3 selected calibration budget differs")
    if source_config["references"]["winning_policy"] != settings["references"][
        "selected_allocation_policy"
    ]:
        raise ValueError("SwiGLU-3 allocation policy differs")
    cache = source["results"]["recovery"]["packed_token_cache"]
    if int(cache["sequence_length"]) != int(compatibility["sequence_length"]):
        raise ValueError("SwiGLU-3 sequence length differs")
    if int(cache["token_count"]) != int(compatibility["packed_token_count"]):
        raise ValueError("SwiGLU-3 packed-token extent differs")
    if cache["fingerprint"] != compatibility["packed_token_fingerprint"]:
        raise ValueError("SwiGLU-3 packed-token fingerprint differs")
    if int(source["results"]["recovery"]["effective_batch_tokens"]) != int(
        compatibility["effective_batch_tokens"]
    ):
        raise ValueError("SwiGLU-3 effective batch differs")
    source_data = source_config["data"]
    for field in (
        "local_source",
        "recovery_source",
        "model_validation_source",
    ):
        if source_data[field] != compatibility[field]:
            raise ValueError(f"SwiGLU-3 data contract differs at {field}")
    if cache["source"] != compatibility["recovery_source"]:
        raise ValueError("SwiGLU-3 packed stream source differs")
    partition_batches = source_data["partition_batches"]
    if int(partition_batches["recovery_validation"]) != int(
        compatibility["recovery_validation_batches"]
    ):
        raise ValueError("SwiGLU-3 recovery-validation split differs")
    if int(partition_batches["allocation_selection"]) != int(
        compatibility["allocation_selection_batches"]
    ):
        raise ValueError("SwiGLU-3 allocation-selection split differs")
    if int(source_data["model_validation_batches"]) != int(
        compatibility["model_validation_batches"]
    ):
        raise ValueError("SwiGLU-3 WikiText evaluation extent differs")
    source_recovery = source_config["recovery"]
    if float(source_recovery["temperature"]) != float(
        compatibility["evaluation_temperature"]
    ):
        raise ValueError("SwiGLU-3 evaluation temperature differs")
    for field in (
        "forward_autocast_dtype",
        "replacement_parameter_dtype",
        "optimizer_state_dtype",
    ):
        expected_field = (
            "trainable_parameter_dtype"
            if field == "replacement_parameter_dtype"
            else field
        )
        if source_recovery[field] != compatibility[expected_field]:
            raise ValueError(f"SwiGLU-3 precision contract differs at {field}")
    for target in compatibility["target_mlp_removals"]:
        key = str(float(target))
        state_rows, allocation_rows = source_operator_rows(
            source,
            settings["references"]["selected_calibration_pairs"],
            key,
        )
        expected_layers = {
            int(value) for value in compatibility["eligible_layers"]
        }
        if set(state_rows) != expected_layers or set(allocation_rows) != expected_layers:
            raise ValueError(
                f"SwiGLU-3 target {key} does not cover the exact eligible layers"
            )
        for layer, row in allocation_rows.items():
            if (
                float(row["target_mlp_removal"]) != float(target)
                or int(row["original_width"])
                != int(settings["model"]["intermediate_size"])
                or int(row["replacement_parameters"])
                != 3
                * int(settings["model"]["hidden_size"])
                * int(row["replacement_width"])
            ):
                raise ValueError(
                    f"SwiGLU-3 target {key} allocation differs at layer {layer}"
                )
        source_milestone(source, key, 10000000)
        source_milestone(source, key, 100000000)
    return {
        "workflow": source["workflow"],
        "schema_version": source["schema_version"],
        "status": source["status"],
        "run_fingerprint": source.get("run_fingerprint"),
        "packed_token_fingerprint": cache["fingerprint"],
        "targets": [str(float(value)) for value in compatibility["target_mlp_removals"]],
        "passed": True,
    }


def build_local_data(context):
    """Recreate the frozen SwiGLU-3 local-data partition contract."""

    values = context.settings["data"]
    sequence_length = int(values["sequence_length"])
    batch_size = int(values["capture_batch_size"])
    partitions = {
        name: int(count) for name, count in values["partition_batches"].items()
    }
    expected_order = [
        "calibration",
        "operator_validation",
        "old_recovery",
        "recovery_validation",
        "additional_calibration",
        "allocation_selection",
        "appended_calibration",
    ]
    if list(partitions) != expected_order:
        raise ValueError("Local-data partitions must preserve the frozen order")
    maximum_pairs = max(
        int(value) for value in context.settings["calibration"]["pair_counts"]
    )
    original_pairs = (
        partitions["calibration"] + partitions["additional_calibration"]
    ) * batch_size * sequence_length
    appended_pairs = (
        partitions["appended_calibration"] * batch_size * sequence_length
    )
    if original_pairs + appended_pairs != maximum_pairs:
        raise ValueError("Appended calibration data does not reach the pair budget")

    source = DatasetSpec(**values["local_source"])
    records = load_text_dataset(source)
    windows = sample_partitioned_windows(
        records,
        context.tokenizer,
        {name: count * batch_size for name, count in partitions.items()},
        sequence_length,
        int(context.settings["seed"]),
        source.text_column,
    )
    calibration_sequences = (
        windows["calibration"]
        + windows["additional_calibration"]
        + windows["appended_calibration"]
    )
    if len(calibration_sequences) * sequence_length != maximum_pairs:
        raise RuntimeError("Nested calibration sequence count is inconsistent")

    validation_source = DatasetSpec(**values["model_validation_source"])
    validation_records = load_text_dataset(validation_source)
    validation_batches = int(values["model_validation_batches"])
    validation_sequences = contiguous_token_windows(
        validation_records,
        context.tokenizer,
        validation_batches * batch_size,
        sequence_length,
        validation_source.text_column,
    )
    return {
        "calibration_sequences": calibration_sequences,
        "operator_validation": make_token_loader(
            windows["operator_validation"], batch_size
        ),
        "allocation_selection": make_token_loader(
            windows["allocation_selection"], batch_size
        ),
        "recovery_validation": make_token_loader(
            windows["recovery_validation"], batch_size
        ),
        "model_validation": make_token_loader(validation_sequences, batch_size),
        "partition_batches": partitions,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
    }


def recovery_memory_record(device):
    """Record process and CUDA recovery peaks."""

    record = {}
    status = Path("/proc/self/status")
    if status.exists():
        values = dict(
            line.split(":", 1)
            for line in status.read_text(encoding="utf-8").splitlines()
            if ":" in line
        )
        record["peak_ram_gib"] = int(values["VmHWM"].split()[0]) / 1024**2
    if torch.device(device).type == "cuda":
        record["peak_vram_gib"] = (
            torch.cuda.max_memory_allocated(device) / 1024**3
        )
    return record
