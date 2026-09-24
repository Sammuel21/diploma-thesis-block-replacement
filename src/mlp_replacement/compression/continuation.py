"""Segment-aware continuation and durable checkpoints for long recovery."""

import os
import random
from json import JSONDecodeError
from dataclasses import replace
from pathlib import Path

from mlp_replacement.artifacts import contained_path, file_digest, read_json, write_json_atomic


def segment_origin(cursor, endpoints):
    """Return the last completed exact endpoint, including a partial update."""

    return max([0, *(int(point) for point in endpoints if int(point) <= cursor)])


def segment_schedule(origin, end, requested, effective_tokens=2048):
    """Round requests relative to a segment, preserving its exact final update."""

    grouped = {}
    for request in sorted(set(int(value) for value in requested)):
        if not origin < request <= end:
            continue
        updates = (request - origin + effective_tokens - 1) // effective_tokens
        actual = min(end, origin + updates * effective_tokens)
        grouped.setdefault(actual, []).append(request)
    grouped.setdefault(end, [])
    if end not in grouped[end]:
        grouped[end].append(end)
    return tuple((actual, tuple(values)) for actual, values in sorted(grouped.items()))


def capture_rng():
    import numpy as np
    import torch

    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    import numpy as np
    import torch

    if "python_rng_state" in state:
        random.setstate(state["python_rng_state"])
    if "numpy_rng_state" in state:
        np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_states") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_states"])


def save_tensor_atomic(path, value):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def commit_checkpoint(directory, payload):
    """Commit payload then descriptor; retain two verified generations."""

    directory = Path(directory)
    cursor = int(payload["tokens_seen"])
    path = directory / f"checkpoint-{cursor:012d}.pt"
    save_tensor_atomic(path, payload)
    descriptor = {
        "path": path.name,
        "sha256": file_digest(path),
        "tokens_seen": cursor,
        "run_fingerprint": payload["run_fingerprint"],
    }
    write_json_atomic(path.with_suffix(".json"), descriptor)
    generations = valid_checkpoint_records(directory, payload["run_fingerprint"])
    for old in generations[2:]:
        old_path = contained_path(directory, old["path"])
        old_path.unlink()
        old_path.with_suffix(".json").unlink()
    return descriptor


def commit_single_checkpoint(directory, payload):
    """Commit one verified generation, then remove every older generation.

    The tensor payload is flushed and renamed before its descriptor is
    committed.  Older verified generations remain available until that new
    descriptor exists and its digest has been checked.  This is the bounded
    checkpoint policy for directory-contract workflows; ``commit_checkpoint``
    deliberately keeps its historical two-generation behavior.
    """

    directory = Path(directory)
    cursor = int(payload["tokens_seen"])
    path = directory / f"checkpoint-{cursor:012d}.pt"
    save_tensor_atomic(path, payload)
    descriptor = {
        "path": path.name,
        "sha256": file_digest(path),
        "tokens_seen": cursor,
        "run_fingerprint": payload["run_fingerprint"],
    }
    write_json_atomic(path.with_suffix(".json"), descriptor)
    generations = valid_checkpoint_records(directory, payload["run_fingerprint"])
    if not generations or generations[0] != descriptor:
        raise RuntimeError("New checkpoint generation did not verify after commit")
    for old in generations[1:]:
        old_path = contained_path(directory, old["path"])
        old_path.unlink()
        old_path.with_suffix(".json").unlink()
    retained = {path.resolve(), path.with_suffix(".json").resolve()}
    for orphan in directory.glob("checkpoint-*"):
        if orphan.resolve() not in retained:
            orphan.unlink()
    return descriptor


def valid_checkpoint_records(directory, run_fingerprint):
    """Ignore interrupted or corrupt generations; never accept foreign state."""

    records = []
    for descriptor in Path(directory).glob("checkpoint-*.json"):
        try:
            record = read_json(descriptor)
            if record["run_fingerprint"] != run_fingerprint:
                raise ValueError("Checkpoint directory contains a different experiment")
            path = contained_path(directory, record["path"])
            if path.is_file() and file_digest(path) == record["sha256"]:
                records.append(record)
        except (OSError, KeyError, TypeError, JSONDecodeError):
            continue
    return sorted(records, key=lambda row: int(row["tokens_seen"]), reverse=True)


def restore_checkpoint(directory, run_fingerprint):
    import torch

    records = valid_checkpoint_records(directory, run_fingerprint)
    if not records:
        raise FileNotFoundError("No complete, verified recovery checkpoint is available")
    record = records[0]
    state = torch.load(contained_path(directory, record["path"]), map_location="cpu", weights_only=False)
    if state["run_fingerprint"] != run_fingerprint or int(state["tokens_seen"]) != int(record["tokens_seen"]):
        raise ValueError("Checkpoint descriptor and payload disagree")
    return state, record


def recover_exact_segment(*, origin, end, cursor, schedule, batch_at, on_checkpoint, **kwargs):
    """Reuse the established loop with relative cursors after partial updates.

    Existing recovery APIs and their callers remain unchanged. The caller owns
    optimizer/RNG restoration between segments; schedules and callbacks expose
    absolute token counts while the original loop receives aligned local counts.
    """

    from mlp_replacement.compression.recovery import recover_trainable_by_tokens

    if kwargs.get("scheduler") != "constant":
        raise ValueError("Segment-relative continuation currently requires constant learning rate")
    if not origin <= cursor < end:
        raise ValueError("Invalid continuation segment")

    def forward_event(event, optimizer, first_step):
        absolute = replace(event, tokens_seen=origin + event.tokens_seen)
        on_checkpoint(absolute, optimizer, first_step)

    result = recover_trainable_by_tokens(
        batch_at=lambda offset, count: batch_at(origin + offset, count),
        target_tokens=end - origin,
        schedule_tokens=end - origin,
        start_tokens=cursor - origin,
        checkpoint_schedule=tuple(
            (actual - origin, requested)
            for actual, requested in schedule
            if actual > cursor
        ),
        on_checkpoint=forward_event,
        **kwargs,
    )
    return replace(result, tokens_seen=origin + result.tokens_seen)
