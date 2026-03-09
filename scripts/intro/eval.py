import torch
import numpy as np

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