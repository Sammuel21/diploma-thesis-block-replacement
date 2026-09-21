"""Small, portable artifact operations for reproducible experiment stages."""

import hashlib
import json
import os
from pathlib import Path


def file_digest(path, byte_limit=None):
    """Hash a file, optionally restricting the digest to an exact prefix."""

    digest = hashlib.sha256()
    remaining = byte_limit
    with Path(path).open("rb") as stream:
        while remaining is None or remaining > 0:
            chunk = stream.read(1024 * 1024 if remaining is None else min(remaining, 1024 * 1024))
            if not chunk:
                if remaining:
                    raise ValueError("File is shorter than the requested hash prefix")
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return digest.hexdigest()


def content_digest(value):
    """Hash a JSON scientific contract without depending on dictionary order."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path, value):
    """Flush a complete JSON document before replacing its committed path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def contained_path(directory, relative):
    """Reject artifact references that escape their explicitly owned directory."""

    directory = Path(directory).resolve()
    result = (directory / relative).resolve()
    if not result.is_relative_to(directory):
        raise ValueError(f"Artifact reference escapes its directory: {relative}")
    return result
