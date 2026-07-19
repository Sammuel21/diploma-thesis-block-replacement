from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .modules import make_replacement_operator


@dataclass(frozen=True)
class OperatorTrainingEpoch:
    """Record local fitting and validation MSE for one epoch."""

    epoch: int
    train_mse: float
    validation_mse: float
    learning_rate: float


@dataclass(frozen=True)
class OperatorFitResult:
    """Return the fitted operator together with its validation history."""

    module: nn.Module
    history: tuple[OperatorTrainingEpoch, ...]
    best_epoch: int
    best_validation_mse: float


def evaluate_operator_mse(module, pairs, device, batch_size):
    """Measure mean squared approximation error on held-out activation pairs."""

    dataset = TensorDataset(pairs.inputs, pairs.targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    parameter = next(module.parameters())
    squared_error = 0.0
    element_count = 0
    module.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=parameter.dtype)
            targets = targets.to(device=device, dtype=parameter.dtype)
            errors = module(inputs) - targets
            squared_error += float(errors.float().square().sum().item())
            element_count += errors.numel()
    if element_count == 0:
        raise ValueError("Operator evaluation received no activation pairs")
    return squared_error / element_count


def fit_replacement_operator(training_pairs, validation_pairs, config, device):
    """Fit one replacement operator and retain its best validation state."""

    if training_pairs.hidden_size != validation_pairs.hidden_size:
        raise ValueError("Training and validation hidden sizes differ")

    torch.manual_seed(config.seed)
    module = make_replacement_operator(training_pairs.hidden_size, config).to(device)
    dataset = TensorDataset(training_pairs.inputs, training_pairs.targets)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = None
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    history = []
    best_validation = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    parameter_dtype = next(module.parameters()).dtype

    for epoch in range(1, config.epochs + 1):
        module.train()
        squared_error = 0.0
        element_count = 0
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=parameter_dtype)
            targets = targets.to(device=device, dtype=parameter_dtype)
            errors = module(inputs) - targets
            loss = errors.float().square().mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(module.parameters(), config.gradient_clip_norm)
            optimizer.step()

            squared_error += float(errors.detach().float().square().sum().item())
            element_count += errors.numel()

        train_mse = squared_error / element_count
        validation_mse = evaluate_operator_mse(
            module, validation_pairs, device, config.batch_size
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(OperatorTrainingEpoch(epoch, train_mse, validation_mse, learning_rate))

        improved = validation_mse < best_validation - config.early_stopping_min_delta
        if improved:
            best_validation = validation_mse
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in module.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if scheduler is not None:
            scheduler.step()
        if (
            config.early_stopping_patience is not None
            and stale_epochs >= config.early_stopping_patience
        ):
            break

    if best_state is None:
        raise RuntimeError("Operator training did not produce a model state")
    module.load_state_dict(best_state)
    module.to(device)
    module.eval()
    return OperatorFitResult(module, tuple(history), best_epoch, best_validation)
