"""Teacher KL on a frozen EOS-packed long-context token stream.

This is the shared diagnostic introduced by Quantization-1 and reused by
SwiGLU-8. The stream holds consecutive EOS-packed documents from one ordered
source; the score is temperature-1 teacher KL averaged over every position.
"""

import math

TEACHER_KL_PROTOCOL = "teacher-kl-t1-all-valid-positions-c4-shard0-24x8192-v1"


def build_teacher_kl_stream(source, tokenizer, path, sequence_count, sequence_length):
    """Freeze the first packed sequences of the configured streaming source."""

    from datasets import load_dataset

    from mlp_replacement.data_streams import build_finite_stream

    records = load_dataset(
        source["path"],
        data_files=source["data_file"],
        revision=source["revision"],
        split=source["split"],
        streaming=True,
    )
    return build_finite_stream(
        records,
        tokenizer,
        path,
        sequence_count * sequence_length,
        text_column=source["text_column"],
    )


def evaluate_teacher_kl(
    student,
    teacher,
    stream_path,
    record,
    sequence_length,
    protocol=TEACHER_KL_PROTOCOL,
    device="cuda",
):
    """Average online teacher KL over every position of a packed stream.

    Both models must use SDPA attention, the execution path under which the
    protocol was defined. The caller verifies the stream digest beforehand.
    """

    import torch

    from mlp_replacement.compression.recovery import online_distillation_loss
    from mlp_replacement.data import PackedTokenCache

    for role, model in (("student", student), ("teacher", teacher)):
        implementation = getattr(model.config, "_attn_implementation", None)
        if implementation != "sdpa":
            raise ValueError(
                f"Teacher KL requires SDPA attention; the {role} uses {implementation!r}"
            )
    cache = PackedTokenCache(
        stream_path, record["token_count"], sequence_length, record["sha256"]
    )
    total = 0.0
    count = 0
    student.eval()
    teacher.eval()
    with torch.inference_mode():
        for offset in range(0, cache.token_count, cache.sequence_length):
            loss, tokens = online_distillation_loss(
                student,
                teacher,
                cache.batch(offset, cache.sequence_length, 1),
                1.0,
                device,
                torch.bfloat16,
            )
            total += float(loss.item()) * tokens
            count += tokens
    if count != record["token_count"] or not math.isfinite(total):
        raise ValueError("KL coverage/finite-loss contract failed")
    return {
        "kl": total / count,
        "valid_positions": count,
        "protocol": protocol,
        "stream_sha256": record["sha256"],
    }
