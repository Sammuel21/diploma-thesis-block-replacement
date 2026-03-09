import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class LinearReplacement(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.proj(x)


def fit_linear_replacement(X: torch.Tensor, Y: torch.Tensor, hidden_size: int, device: str, cfg):
    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=cfg.replacement_batch_size, shuffle=True)

    replacement = LinearReplacement(hidden_size).to(device)
    optimizer = torch.optim.AdamW(replacement.parameters(), lr=cfg.replacement_lr)
    loss_fn = nn.MSELoss()

    replacement.train()
    for epoch in range(cfg.replacement_epochs):
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = replacement(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().item())

        print(f"[replacement train] epoch={epoch+1} mse={sum(losses)/len(losses):.6f}")

    replacement.eval()
    return replacement


def evaluate_block_mse(replacement, X_val: torch.Tensor, Y_val: torch.Tensor, device: str):
    replacement.eval()
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        pred = replacement(X_val.to(device))
        mse = loss_fn(pred, Y_val.to(device)).item()

    return mse