from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..evaluation.operator import module_dtype
from .modules import make_replacement_operator
from .modules import LinearReplacement


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
    calculation_dtype = module_dtype(module, pairs.inputs.dtype)
    squared_error = 0.0
    element_count = 0
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device=device, dtype=calculation_dtype)
                targets = targets.to(device=device, dtype=calculation_dtype)
                errors = module(inputs) - targets
                squared_error += float(errors.float().square().sum().item())
                element_count += errors.numel()
    finally:
        module.train(was_training)
    if element_count == 0:
        raise ValueError("Operator evaluation received no activation pairs")
    return squared_error / element_count


def fit_operator(module, training_pairs, validation_pairs, config, device):
    """Fit an already constructed operator and retain its best validation state."""

    if training_pairs.hidden_size != validation_pairs.hidden_size:
        raise ValueError("Training and validation hidden sizes differ")
    module = module.to(device)
    parameters = tuple(parameter for parameter in module.parameters() if parameter.requires_grad)
    if not parameters:
        validation_mse = evaluate_operator_mse(
            module,
            validation_pairs,
            device,
            config.batch_size,
        )
        module.eval()
        return OperatorFitResult(module, (), 0, validation_mse)

    torch.manual_seed(config.seed)
    dataset = TensorDataset(training_pairs.inputs, training_pairs.targets)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = None
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
        )

    history = []
    best_validation = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    calculation_dtype = module_dtype(module, training_pairs.inputs.dtype)

    for epoch in range(1, config.epochs + 1):
        module.train()
        squared_error = 0.0
        element_count = 0
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=calculation_dtype)
            targets = targets.to(device=device, dtype=calculation_dtype)
            errors = module(inputs) - targets
            loss = errors.float().square().mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm)
            optimizer.step()

            squared_error += float(errors.detach().float().square().sum().item())
            element_count += errors.numel()

        train_mse = squared_error / element_count
        validation_mse = evaluate_operator_mse(
            module,
            validation_pairs,
            device,
            config.batch_size,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            OperatorTrainingEpoch(
                epoch,
                train_mse,
                validation_mse,
                learning_rate,
            )
        )

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


def fit_replacement_operator(
    training_pairs,
    validation_pairs,
    config,
    device,
    intermediate_width=None,
):
    """Fit one replacement operator and retain its best validation state."""

    if training_pairs.hidden_size != validation_pairs.hidden_size:
        raise ValueError("Training and validation hidden sizes differ")

    torch.manual_seed(config.seed)
    module = make_replacement_operator(
        training_pairs.hidden_size,
        config,
        intermediate_width,
    ).to(device)
    return fit_operator(module, training_pairs, validation_pairs, config, device)


def fit_ridge_linear(training_pairs, ridge=1e-4, bias=False, device="cpu"):
    """Fit a dense linear regression baseline by a regularized normal equation.

    ``ridge`` is scaled by the mean diagonal of ``X.T @ X`` so it remains
    interpretable across activation scales.
    """

    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    calculation_device = torch.device(device)
    inputs = training_pairs.inputs.to(calculation_device, torch.float32)
    targets = training_pairs.targets.to(calculation_device, torch.float32)
    if bias:
        input_mean = inputs.mean(dim=0)
        target_mean = targets.mean(dim=0)
        solve_inputs = inputs - input_mean
        solve_targets = targets - target_mean
    else:
        input_mean = target_mean = None
        solve_inputs = inputs
        solve_targets = targets

    gram = solve_inputs.T @ solve_inputs
    diagonal_scale = gram.diagonal().mean().clamp_min(torch.finfo(gram.dtype).eps)
    regularized = gram + ridge * diagonal_scale * torch.eye(
        gram.shape[0],
        device=calculation_device,
        dtype=gram.dtype,
    )
    coefficients = torch.linalg.solve(regularized, solve_inputs.T @ solve_targets)
    module = LinearReplacement(training_pairs.hidden_size, bias=bias)
    with torch.no_grad():
        module.projection.weight.copy_(coefficients.T.cpu())
        if bias:
            fitted_bias = target_mean - input_mean @ coefficients
            module.projection.bias.copy_(fitted_bias.cpu())
    module.eval()
    return module
