"""Held-out metrics for isolated MLP replacement operators."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class OperatorMetrics:
    """Record scale-aware and token-level operator approximation metrics."""

    mse: float
    relative_mse: float
    r2: float
    cosine_similarity: float
    norm_ratio: float
    median_token_relative_error: float
    p95_token_relative_error: float
    tokens: int


def module_dtype(module, fallback=torch.float32):
    """Resolve a module's calculation dtype even when it has no parameters."""

    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.dtype
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.dtype
    return fallback


def evaluate_operator(module, pairs, device, batch_size):
    """Evaluate an isolated replacement on held-out activation pairs."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    dataset = TensorDataset(pairs.inputs, pairs.targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dtype = module_dtype(module, pairs.inputs.dtype)
    target_mean = pairs.targets.float().mean(dim=0)

    squared_error = 0.0
    target_energy = 0.0
    centered_target_energy = 0.0
    element_count = 0
    cosine_total = 0.0
    norm_ratio_total = 0.0
    token_count = 0
    token_relative_errors = []
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device=device, dtype=dtype)
                targets = targets.to(device=device, dtype=dtype)
                predictions = module(inputs)
                errors = predictions.float() - targets.float()
                float_targets = targets.float()
                squared_error += float(errors.square().sum().item())
                target_energy += float(float_targets.square().sum().item())
                centered = float_targets - target_mean.to(device)
                centered_target_energy += float(centered.square().sum().item())
                element_count += errors.numel()

                cosine = F.cosine_similarity(
                    predictions.float(),
                    float_targets,
                    dim=-1,
                    eps=1e-8,
                )
                target_norm = float_targets.norm(dim=-1)
                prediction_norm = predictions.float().norm(dim=-1)
                relative_error = errors.norm(dim=-1) / target_norm.clamp_min(1e-8)
                cosine_total += float(cosine.sum().item())
                norm_ratio_total += float(
                    (prediction_norm / target_norm.clamp_min(1e-8)).sum().item()
                )
                token_count += int(targets.shape[0])
                token_relative_errors.append(relative_error.cpu())
    finally:
        module.train(was_training)

    if element_count == 0 or token_count == 0:
        raise ValueError("Operator evaluation received no activation pairs")
    token_errors = torch.cat(token_relative_errors)
    relative_mse = squared_error / target_energy if target_energy > 0 else 0.0
    r2 = (
        1.0 - squared_error / centered_target_energy
        if centered_target_energy > 0
        else 0.0
    )
    return OperatorMetrics(
        mse=squared_error / element_count,
        relative_mse=relative_mse,
        r2=r2,
        cosine_similarity=cosine_total / token_count,
        norm_ratio=norm_ratio_total / token_count,
        median_token_relative_error=float(token_errors.median().item()),
        p95_token_relative_error=float(torch.quantile(token_errors, 0.95).item()),
        tokens=token_count,
    )
