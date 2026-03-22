import torch
import torch.nn.functional as F
import numpy as np

# model/block scoring

def evaluate_lm(model, loader, max_batches: int, device: str):
    model.eval()
    losses = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            labels[batch["attention_mask"] == 0] = -100

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=labels,
            )
            losses.append(out.loss.detach().float().cpu())

    mean_loss = torch.stack(losses).mean().item()
    ppl = np.exp(mean_loss)
    return mean_loss, ppl


# block importance scoring

def bi_cosine_score(X, Y):
    cos = F.cosine_similarity(X, Y, dim=-1, eps=1e-8)
    score = 1.0 - cos
    score = score.mean().item()
    return float(score)