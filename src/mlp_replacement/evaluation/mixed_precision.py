"""Established mixed-precision evaluation with FP32 metric reductions."""

import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F


def autocast_context(device):
    """Use the established mixed-precision evaluation context."""

    device = torch.device(device)
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def evaluate_lm_mixed(model, loader, device, max_batches):
    """Evaluate causal-LM loss with FP32 reductions."""

    was_training = model.training
    model.eval()
    total_nll = 0.0
    predicted_tokens = 0
    batches = 0
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if batch_index >= max_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with autocast_context(device):
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    ).logits
                labels = input_ids[:, 1:].contiguous()
                mask = attention_mask[:, 1:].bool()
                labels = labels.masked_fill(~mask, -100)
                nll = F.cross_entropy(
                    logits[:, :-1, :].float().contiguous().view(
                        -1, logits.shape[-1]
                    ),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                total_nll += float(nll.item())
                predicted_tokens += int(mask.sum().item())
                batches += 1
    finally:
        model.train(was_training)
    if predicted_tokens == 0:
        raise ValueError("Language-model evaluation contained no predicted tokens")
    loss = total_nll / predicted_tokens
    return {
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 709 else float("inf"),
        "predicted_tokens": predicted_tokens,
        "batches": batches,
    }


def evaluate_teacher_cache_mixed(model, teacher_cache, temperature, device):
    """Evaluate teacher KL and causal-LM loss on a fixed cache."""

    losses = []
    total_nll = 0.0
    predicted_tokens = 0
    model.eval()
    with torch.no_grad():
        for batch in teacher_cache.batches:
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            with autocast_context(device):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
            mask = attention_mask.bool()
            teacher_probabilities = torch.softmax(
                batch.logits.to(device=device, dtype=torch.float32)[mask]
                / temperature,
                dim=-1,
            )
            student_log_probabilities = torch.log_softmax(
                logits.float()[mask] / temperature, dim=-1
            )
            kl = F.kl_div(
                student_log_probabilities,
                teacher_probabilities,
                reduction="batchmean",
            ) * (temperature**2)
            losses.append(float(kl.item()))
            labels = input_ids[:, 1:].contiguous()
            valid = attention_mask[:, 1:].bool()
            labels = labels.masked_fill(~valid, -100)
            nll = F.cross_entropy(
                logits[:, :-1, :].float().contiguous().view(
                    -1, logits.shape[-1]
                ),
                labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_nll += float(nll.item())
            predicted_tokens += int(valid.sum().item())
    if not losses or predicted_tokens == 0:
        raise ValueError("Teacher-cache evaluation contained no valid tokens")
    loss = total_nll / predicted_tokens
    return {
        "teacher_kl": sum(losses) / len(losses),
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 709 else float("inf"),
        "predicted_tokens": predicted_tokens,
        "batches": len(losses),
    }


def evaluate_validation_kl_mixed(model, teacher_cache, temperature, device):
    """Measure fixed-cache teacher KL for a mixed-dtype recovery model."""

    losses = []
    model.eval()
    with torch.no_grad():
        for batch in teacher_cache.batches:
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            with autocast_context(device):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
            mask = attention_mask.bool()
            probabilities = torch.softmax(
                batch.logits.to(device=device, dtype=torch.float32)[mask]
                / temperature,
                dim=-1,
            )
            log_probabilities = torch.log_softmax(
                logits.float()[mask] / temperature, dim=-1
            )
            losses.append(
                float(
                    (
                        F.kl_div(
                            log_probabilities,
                            probabilities,
                            reduction="batchmean",
                        )
                        * (temperature**2)
                    ).item()
                )
            )
    if not losses:
        raise ValueError("Recovery-validation cache contained no batches")
    return sum(losses) / len(losses)
