"""Teacher logits and immutable final-hidden caches for recovery."""

import hashlib
import json
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from ..artifacts import sha256_file
from ..model import autocast_context, resolve_dtype


@dataclass(frozen=True)
class TeacherBatch:
    """Store one recovery batch and its cached dense-model predictions."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    logits: torch.Tensor


@dataclass(frozen=True)
class TeacherCache:
    """Keep dense-model predictions available after its MLPs are replaced."""

    batches: tuple[TeacherBatch, ...]

    def __len__(self):
        return len(self.batches)


@dataclass(frozen=True)
class TeacherHiddenShard:
    """Describe one immutable final-hidden-state cache shard."""

    token_start: int
    token_end: int
    path: Path
    sha256: str


@dataclass(frozen=True)
class TeacherFinalHiddenCache:
    """Read dense final hidden states aligned with a packed token stream."""

    root: Path
    token_count: int
    sequence_length: int
    hidden_size: int
    dtype: str
    token_fingerprint: str
    head_fingerprint: str
    shards: tuple[TeacherHiddenShard, ...]
    manifest: dict

    def batch(self, token_offset, token_count):
        """Load one contiguous token interval from immutable cache shards."""

        token_offset = int(token_offset)
        token_count = int(token_count)
        if token_offset < 0 or token_count < 1:
            raise ValueError("Teacher-hidden cache reads require positive ranges")
        end = token_offset + token_count
        if end > self.token_count:
            raise ValueError("Teacher-hidden cache read exceeds its extent")
        chunks = []
        cursor = token_offset
        for shard in self.shards:
            if shard.token_end <= cursor or shard.token_start >= end:
                continue
            values = load_hidden_shard(shard.path, shard.sha256)
            start = max(cursor, shard.token_start) - shard.token_start
            stop = min(end, shard.token_end) - shard.token_start
            chunks.append(values[start:stop])
            cursor = min(end, shard.token_end)
            if cursor == end:
                break
        if cursor != end:
            raise RuntimeError("Teacher-hidden cache has a gap in the requested range")
        return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]

    def release(self):
        """Release process-local memory maps before deleting cache shards."""

        load_hidden_shard.cache_clear()


@lru_cache(maxsize=2)
def load_hidden_shard(path, sha256):
    """Memory-map and retain the current sequential cache shards."""

    path = Path(path)
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_sha256(tensor):
    values = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(values.numpy().tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_teacher_hidden_cache(root, expected_fingerprint=None):
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Teacher-hidden manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected_fingerprint is not None and (
        manifest.get("fingerprint") != expected_fingerprint
    ):
        raise ValueError("Teacher-hidden cache fingerprint differs")
    shards = []
    cursor = 0
    for row in manifest["shards"]:
        token_start = int(row["token_start"])
        token_end = int(row["token_end"])
        if token_start != cursor or token_end <= token_start:
            raise ValueError("Teacher-hidden manifest shard coverage is invalid")
        path = root / row["file"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"Teacher-hidden shard is missing or changed: {path}")
        shards.append(
            TeacherHiddenShard(
                token_start=token_start,
                token_end=token_end,
                path=path,
                sha256=row["sha256"],
            )
        )
        cursor = token_end
    if cursor != int(manifest["token_count"]):
        raise ValueError("Teacher-hidden manifest does not cover its token extent")
    expected_files = {manifest_path}
    expected_files.update(shard.path for shard in shards)
    unexpected = [path for path in root.iterdir() if path not in expected_files]
    if unexpected:
        raise FileExistsError(
            "Completed teacher-hidden cache contains unexpected files: "
            + ", ".join(str(path) for path in unexpected)
        )
    return TeacherFinalHiddenCache(
        root=root,
        token_count=int(manifest["token_count"]),
        sequence_length=int(manifest["sequence_length"]),
        hidden_size=int(manifest["hidden_size"]),
        dtype=str(manifest["dtype"]),
        token_fingerprint=str(manifest["token_fingerprint"]),
        head_fingerprint=str(manifest["head_fingerprint"]),
        shards=tuple(shards),
        manifest=manifest,
    )


def build_teacher_final_hidden_cache(
    model,
    packed_tokens,
    root,
    token_count,
    batch_sequences,
    device,
    model_identity,
    cache_dtype=torch.bfloat16,
    shard_tokens=65536,
    minimum_free_gib=25.0,
):
    """Build or validate a sharded dense final-hidden-state cache."""

    root = Path(root)
    token_count = int(token_count)
    batch_sequences = int(batch_sequences)
    shard_tokens = int(shard_tokens)
    sequence_length = int(packed_tokens.sequence_length)
    capture_batch_tokens = batch_sequences * sequence_length
    if (
        token_count < 1
        or token_count > packed_tokens.token_count
        or token_count % sequence_length
        or shard_tokens % sequence_length
        or batch_sequences < 1
        or shard_tokens % capture_batch_tokens
    ):
        raise ValueError("Teacher-hidden cache geometry is inconsistent")
    output_head = model.get_output_embeddings()
    if output_head is None or not hasattr(output_head, "weight"):
        raise TypeError("Dense teacher must expose a frozen output embedding")
    head_fingerprint = tensor_sha256(output_head.weight)
    payload = {
        "schema_version": 1,
        "format": "final-normalized-hidden-bf16-shards-v1",
        "model": model_identity,
        "token_fingerprint": packed_tokens.fingerprint,
        "token_count": token_count,
        "sequence_length": sequence_length,
        "hidden_size": int(output_head.weight.shape[1]),
        "dtype": str(cache_dtype).removeprefix("torch."),
        "head_fingerprint": head_fingerprint,
        "shard_tokens": shard_tokens,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    fingerprint = hashlib.sha256(encoded).hexdigest()
    manifest_path = root / "manifest.json"
    partial_manifest_path = root / "partial-manifest.json"
    if manifest_path.exists():
        return load_teacher_hidden_cache(root, fingerprint)
    partial_manifest = None
    if partial_manifest_path.is_file():
        partial_manifest = json.loads(
            partial_manifest_path.read_text(encoding="utf-8")
        )
        if partial_manifest.get("fingerprint") != fingerprint:
            raise ValueError("Partial teacher-hidden cache fingerprint differs")
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        for temporary in root.glob("hidden-*.pt.tmp"):
            if temporary.is_file():
                temporary.unlink()
    expected_shards = {
        f"hidden-{start:012d}-{min(token_count, start + shard_tokens):012d}.pt": (
            start,
            min(token_count, start + shard_tokens),
        )
        for start in range(0, token_count, shard_tokens)
    }
    existing = tuple(root.iterdir()) if root.exists() else ()
    unexpected = [
        path
        for path in existing
        if not (
            path == partial_manifest_path
            or (path.is_file() and path.name in expected_shards)
        )
    ]
    if unexpected:
        raise FileExistsError(
            "Teacher-hidden cache contains uncommitted or unexpected files: "
            + ", ".join(str(path) for path in unexpected)
        )
    partial_rows = {}
    if partial_manifest is not None:
        for row in partial_manifest.get("shards", []):
            file_name = str(row["file"])
            if file_name in partial_rows or file_name not in expected_shards:
                raise ValueError("Partial teacher-hidden shard manifest is invalid")
            expected_start, expected_end = expected_shards[file_name]
            path = root / file_name
            if (
                int(row["token_start"]) != expected_start
                or int(row["token_end"]) != expected_end
                or not path.is_file()
                or sha256_file(path) != row["sha256"]
            ):
                raise ValueError(
                    f"Partial teacher-hidden shard is missing or changed: {path}"
                )
            partial_rows[file_name] = row
    free_gib = shutil.disk_usage(parent).free / 1024**3
    existing_shards = [
        path for path in existing if path.is_file() and path.name in expected_shards
    ]
    if not existing_shards and free_gib < float(minimum_free_gib):
        raise OSError(
            f"Teacher-hidden cache requires {minimum_free_gib:.1f} GiB free; "
            f"{free_gib:.1f} GiB is available"
        )
    root.mkdir(parents=True, exist_ok=True)
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise TypeError("Dense teacher does not expose its base Transformer as .model")
    was_training = model.training
    model.eval()
    shard_rows = []
    try:
        with torch.no_grad():
            for shard_start in range(0, token_count, shard_tokens):
                shard_end = min(token_count, shard_start + shard_tokens)
                path = root / f"hidden-{shard_start:012d}-{shard_end:012d}.pt"
                if path.is_file():
                    try:
                        values = torch.load(
                            path,
                            map_location="cpu",
                            weights_only=True,
                            mmap=True,
                        )
                    except TypeError:
                        values = torch.load(path, map_location="cpu")
                    expected_shape = (
                        shard_end - shard_start,
                        int(output_head.weight.shape[1]),
                    )
                    if tuple(values.shape) != expected_shape or values.dtype != cache_dtype:
                        raise ValueError(
                            f"Incomplete cache shard has invalid content: {path}"
                        )
                    shard_sha256 = sha256_file(path)
                    partial_row = partial_rows.get(path.name)
                    if (
                        partial_row is not None
                        and shard_sha256 != partial_row["sha256"]
                    ):
                        raise ValueError(
                            f"Incomplete cache shard changed after commit: {path}"
                        )
                    shard_rows.append(
                        {
                            "token_start": shard_start,
                            "token_end": shard_end,
                            "file": path.name,
                            "sha256": shard_sha256,
                        }
                    )
                    atomic_json(
                        partial_manifest_path,
                        {**payload, "fingerprint": fingerprint, "shards": shard_rows},
                    )
                    del values
                    continue
                chunks = []
                cursor = shard_start
                while cursor < shard_end:
                    count = min(
                        capture_batch_tokens,
                        shard_end - cursor,
                    )
                    batch = packed_tokens.batch(cursor, count, batch_sequences)
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    with autocast_context(device, cache_dtype):
                        hidden = backbone(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                            return_dict=True,
                        ).last_hidden_state
                    chunks.append(
                        hidden.detach().reshape(-1, hidden.shape[-1]).to(
                            device="cpu", dtype=cache_dtype
                        )
                    )
                    cursor += count
                values = torch.cat(chunks, dim=0)
                temporary = path.with_suffix(path.suffix + ".tmp")
                try:
                    torch.save(values, temporary)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                shard_rows.append(
                    {
                        "token_start": shard_start,
                        "token_end": shard_end,
                        "file": path.name,
                        "sha256": sha256_file(path),
                    }
                )
                atomic_json(
                    partial_manifest_path,
                    {**payload, "fingerprint": fingerprint, "shards": shard_rows},
                )
                del values, chunks
    finally:
        model.train(was_training)
    manifest = {
        **payload,
        "fingerprint": fingerprint,
        "shards": shard_rows,
    }
    atomic_json(manifest_path, manifest)
    partial_manifest_path.unlink(missing_ok=True)
    return load_teacher_hidden_cache(root, fingerprint)


def validate_teacher_final_hidden_cache(
    model,
    cache,
    packed_tokens,
    device,
    sample_offsets,
    capture_batch_sequences,
    sample_sequences=2,
    temperature=1.0,
    maximum_mean_kl=1e-5,
):
    """Compare cached and online dense distributions at fixed offsets.

    Replay the batch geometry used to capture the hidden states.  BF16 model
    output can vary with CUDA kernel geometry, so changing the batch size here
    would measure that numerical difference instead of cache fidelity.
    """

    head = model.get_output_embeddings()
    if head is None or getattr(model, "model", None) is None:
        raise TypeError("Teacher cache validation requires model and output head")
    if cache.token_fingerprint != packed_tokens.fingerprint:
        raise ValueError("Teacher cache and packed-token fingerprints differ")
    if int(cache.sequence_length) != int(packed_tokens.sequence_length):
        raise ValueError("Teacher cache and packed-token sequence lengths differ")
    if cache.head_fingerprint != tensor_sha256(head.weight):
        raise ValueError("Teacher cache and output-head fingerprints differ")
    capture_batch_sequences = int(capture_batch_sequences)
    sample_sequences = int(sample_sequences)
    if capture_batch_sequences < 1 or sample_sequences < 1:
        raise ValueError("Teacher cache validation batch sizes must be positive")
    sample_offsets = tuple(int(value) for value in sample_offsets)
    if not sample_offsets:
        raise ValueError("Teacher cache validation requires at least one sample")
    sequence_length = int(cache.sequence_length)
    capture_batch_tokens = capture_batch_sequences * sequence_length
    sample_tokens = sample_sequences * sequence_length
    if int(cache.manifest["shard_tokens"]) % capture_batch_tokens:
        raise ValueError(
            "Teacher cache shards do not preserve the capture-batch boundaries"
        )
    cache_dtype = resolve_dtype(cache.dtype, torch.device(device))
    losses = []
    evaluated_offsets = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for offset in sample_offsets:
                sample_end = offset + sample_tokens
                if offset % sequence_length:
                    raise ValueError(
                        "Teacher cache validation offsets must begin on a sequence boundary"
                    )
                if offset < 0 or sample_end > cache.token_count:
                    raise ValueError(
                        "Teacher cache validation sample lies outside the cache"
                    )
                evaluated_offsets.append(offset)
                kl_sum = 0.0
                compared_tokens = 0
                capture_offset = (
                    offset // capture_batch_tokens
                ) * capture_batch_tokens
                while capture_offset < sample_end:
                    capture_tokens = min(
                        capture_batch_tokens,
                        cache.token_count - capture_offset,
                    )
                    batch = packed_tokens.batch(
                        capture_offset,
                        capture_tokens,
                        capture_batch_sequences,
                    )
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    with autocast_context(device, cache_dtype):
                        online_logits = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        ).logits.reshape(-1, head.weight.shape[0])
                        hidden = cache.batch(
                            capture_offset,
                            capture_tokens,
                        ).reshape(
                            *input_ids.shape,
                            cache.hidden_size,
                        ).to(device)
                        cached_logits = head(hidden).reshape(
                            -1, head.weight.shape[0]
                        )
                    overlap_start = max(offset, capture_offset)
                    overlap_end = min(sample_end, capture_offset + capture_tokens)
                    local_start = overlap_start - capture_offset
                    local_end = overlap_end - capture_offset
                    online_probabilities = torch.softmax(
                        online_logits[local_start:local_end].float()
                        / float(temperature),
                        dim=-1,
                    )
                    cached_log_probabilities = torch.log_softmax(
                        cached_logits[local_start:local_end].float()
                        / float(temperature),
                        dim=-1,
                    )
                    kl_sum += float(
                        (
                            F.kl_div(
                                cached_log_probabilities,
                                online_probabilities,
                                reduction="sum",
                            )
                            * (float(temperature) ** 2)
                        ).item()
                    )
                    compared_tokens += local_end - local_start
                    capture_offset += capture_tokens
                if compared_tokens != sample_tokens:
                    raise RuntimeError(
                        "Teacher cache validation did not cover the requested sample"
                    )
                losses.append(kl_sum / compared_tokens)
    finally:
        model.train(was_training)
    mean_kl = sum(losses) / len(losses)
    record = {
        "sample_offsets": evaluated_offsets,
        "sample_sequences": sample_sequences,
        "capture_batch_sequences": capture_batch_sequences,
        "sample_kls": losses,
        "mean_kl": mean_kl,
        "maximum_mean_kl": float(maximum_mean_kl),
        "passed": mean_kl <= float(maximum_mean_kl),
    }
    if not record["passed"]:
        sample_summary = ", ".join(f"{value:.8g}" for value in losses)
        raise ValueError(
            f"Cached teacher mean KL {mean_kl:.8g} exceeds "
            f"{float(maximum_mean_kl):.8g} with capture batch "
            f"{capture_batch_sequences}; sample KLs: [{sample_summary}]"
        )
    return record


def cache_teacher_logits(model, loader, max_batches, device, cache_dtype="float16"):
    """Cache dense-model logits on CPU for later knowledge-distillation recovery."""

    if max_batches < 1:
        raise ValueError("Teacher-cache max_batches must be positive")
    dtype = resolve_dtype(cache_dtype, torch.device(device))
    batches = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if batch_index >= max_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                batches.append(
                    TeacherBatch(
                        input_ids=batch["input_ids"].detach().cpu(),
                        attention_mask=batch["attention_mask"].detach().cpu(),
                        logits=output.logits.detach().to(dtype=dtype, device="cpu"),
                    )
                )
    finally:
        model.train(was_training)
    if not batches:
        raise ValueError("Teacher cache received no batches")
    return TeacherCache(tuple(batches))
