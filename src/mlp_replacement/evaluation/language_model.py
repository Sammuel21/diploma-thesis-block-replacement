import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LanguageModelMetrics:
    """Record token-weighted language-model loss and perplexity."""

    loss: float
    perplexity: float
    predicted_tokens: int
    batches: int


def evaluate_language_model(model, loader, device, max_batches=None):
    """Evaluate causal language-model loss over all valid predicted tokens."""

    was_training = model.training
    model.eval()
    total_nll = 0.0
    predicted_tokens = 0
    batches = 0

    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                output = model(input_ids=input_ids, attention_mask=attention_mask)

                shift_logits = output.logits[:, :-1, :].float().contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                valid_mask = attention_mask[:, 1:].bool()
                shift_labels = shift_labels.masked_fill(~valid_mask, -100)

                nll = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                total_nll += float(nll.item())
                predicted_tokens += int(valid_mask.sum().item())
                batches += 1
    finally:
        model.train(was_training)

    if predicted_tokens == 0:
        raise ValueError("Language-model evaluation contained no predicted tokens")
    mean_loss = total_nll / predicted_tokens
    perplexity = math.exp(mean_loss) if mean_loss < 709 else float("inf")
    return LanguageModelMetrics(mean_loss, perplexity, predicted_tokens, batches)
