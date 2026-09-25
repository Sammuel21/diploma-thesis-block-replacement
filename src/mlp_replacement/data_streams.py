"""Build finite memory-mapped token streams from ordered documents."""

import os
from pathlib import Path

from mlp_replacement.artifacts import file_digest


def build_finite_stream(
    records,
    tokenizer,
    path,
    token_count,
    text_column="text",
    on_progress=None,
):
    """Tokenize a finite, non-repeating document stream into signed int32."""

    import numpy as np

    path = Path(path)
    token_count = int(token_count)
    if path.exists():
        raise FileExistsError(path)
    if tokenizer.eos_token_id is None or token_count < 1:
        raise ValueError("Invalid packed stream contract")
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    documents = 0
    next_report = 25_000_000
    try:
        with temporary.open("wb") as output:
            for record in records:
                text = str(record.get(text_column) or "")
                if not text.strip():
                    continue
                ids = tokenizer(
                    text,
                    add_special_tokens=False,
                    return_attention_mask=False,
                ).input_ids
                if not ids:
                    continue
                document = np.asarray([*ids, tokenizer.eos_token_id], dtype=np.int32)
                selected = document[: token_count - written]
                selected.tofile(output)
                written += len(selected)
                documents += 1
                if written >= next_report:
                    if on_progress:
                        on_progress(written, documents)
                    next_report += 25_000_000
                if written == token_count:
                    break
            if written != token_count:
                raise ValueError(
                    f"Source exhausted at {written:,} tokens; repetition is forbidden"
                )
            output.flush()
            os.fsync(output.fileno())
        digest = file_digest(temporary)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": path.name,
        "token_count": written,
        "dtype": "int32",
        "sha256": digest,
        "documents_consumed": documents,
    }


def build_verified_stream(records, tokenizer, path, token_count, prefix_path, prefix_tokens, on_progress=None):
    """Tokenize documents once; verify old bytes before consuming the suffix."""

    import numpy as np

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if tokenizer.eos_token_id is None or token_count < prefix_tokens:
        raise ValueError("Invalid packed stream contract")
    expected_prefix = file_digest(prefix_path)
    if Path(prefix_path).stat().st_size != prefix_tokens * 4:
        raise ValueError("Historical prefix has an unexpected byte length")
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    documents = 0
    verified = False
    next_report = 25_000_000
    with temporary.open("wb") as output:
        for record in records:
            text = str(record.get("text") or "")
            if not text.strip():
                continue
            ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False).input_ids
            if not ids:
                continue
            document = np.asarray([*ids, tokenizer.eos_token_id], dtype=np.int32)
            selected = document[:token_count - written]
            selected.tofile(output)
            written += len(selected)
            documents += 1
            if not verified and written >= prefix_tokens:
                output.flush()
                if file_digest(temporary, prefix_tokens * 4) != expected_prefix:
                    raise ValueError("Regenerated stream differs from the historical 100M-token prefix")
                verified = True
            if written >= next_report:
                if on_progress:
                    on_progress(written, documents)
                next_report += 25_000_000
            if written == token_count:
                break
        if written != token_count or not verified:
            raise ValueError(f"Source exhausted at {written:,} tokens; repetition is forbidden")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    return {"path": path.name, "token_count": written, "dtype": "int32",
            "sha256": file_digest(path), "prefix_sha256": expected_prefix,
            "prefix_tokens": prefix_tokens, "documents_consumed": documents}
