from dataclasses import dataclass, replace
from time import perf_counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..evaluation.operator import evaluate_operator, module_dtype
from .modules import (
    GatedMLPReplacement,
    LinearReplacement,
    initialize_gated_mlp_from_teacher,
    make_replacement_operator,
    swiglu_neuron_importance_scores,
)


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


@dataclass(frozen=True)
class DetailedOperatorEpoch:
    """Record scale-aware held-out metrics for one FP32 local-fit epoch."""

    epoch: int
    updates: int
    train_mse: float
    validation_mse: float
    validation_nmse: float
    validation_cosine: float
    learning_rate: float
    elapsed_seconds: float


@dataclass(frozen=True)
class DetailedOperatorFitResult:
    """Return a best-validation FP32 operator and its detailed history."""

    module: nn.Module
    history: tuple[DetailedOperatorEpoch, ...]
    best_epoch: int
    best_validation_mse: float
    updates: int


@dataclass(frozen=True)
class GatedReconstructionResult:
    """Return both phases of output-aware SwiGLU reconstruction."""

    module: GatedMLPReplacement
    down_only_fit: DetailedOperatorFitResult
    full_fit: DetailedOperatorFitResult
    neuron_indices: tuple[int, ...]
    initial_mean_residual: tuple[float, ...]
    down_only_validation_mse: float
    down_only_validation_nmse: float
    down_only_validation_cosine: float


def initialize_gated_mlp_with_output_reconstruction(
    student,
    teacher,
    calibration_pairs,
    neuron_indices,
    down_only_config,
):
    """Warm-start a reduced SwiGLU by reconstructing its dense output.

    The supplied configuration is the established full local-fitting contract.
    The reconstruction phase derives an eight-epoch, patience-two variant,
    optimizes only the selected down projection and its residual-mean bias, and
    then performs the complete fit from that validation-best state.
    """

    if isinstance(calibration_pairs, (tuple, list)):
        if len(calibration_pairs) != 2:
            raise ValueError(
                "Calibration pairs must be one pair set or (training, validation)"
            )
        training_pairs, validation_pairs = calibration_pairs
    else:
        training_pairs = validation_pairs = calibration_pairs
    if not isinstance(student, GatedMLPReplacement):
        raise TypeError("student must be a GatedMLPReplacement")
    if student.down_projection.bias is None:
        raise ValueError("Output-aware reconstruction requires a down bias")
    indices = torch.as_tensor(
        neuron_indices,
        dtype=torch.long,
        device=teacher.gate_proj.weight.device,
    )
    if indices.ndim != 1 or indices.numel() != student.bottleneck_size:
        raise ValueError("neuron_indices must match the replacement width")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError("neuron_indices must be unique")

    with torch.no_grad():
        student.gate_projection.weight.copy_(
            teacher.gate_proj.weight.index_select(0, indices).to(
                student.gate_projection.weight
            )
        )
        student.up_projection.weight.copy_(
            teacher.up_proj.weight.index_select(0, indices).to(
                student.up_projection.weight
            )
        )
        student.down_projection.weight.copy_(
            teacher.down_proj.weight.index_select(1, indices).to(
                student.down_projection.weight
            )
        )
        if student.gate_projection.bias is not None:
            student.gate_projection.bias.zero_()
        if student.up_projection.bias is not None:
            student.up_projection.bias.zero_()
        student.down_projection.bias.zero_()

    calculation_device = torch.device(
        next(student.parameters()).device
    )
    batch_size = int(down_only_config.batch_size)
    residual_sum = torch.zeros(
        training_pairs.hidden_size,
        dtype=torch.float64,
    )
    token_count = 0
    student.eval()
    with torch.no_grad():
        for start in range(0, training_pairs.num_tokens, batch_size):
            inputs = training_pairs.inputs[start : start + batch_size].to(
                device=calculation_device,
                dtype=torch.float32,
            )
            targets = training_pairs.targets[start : start + batch_size].to(
                device=calculation_device,
                dtype=torch.float32,
            )
            residual_sum += (
                targets - student(inputs)
            ).sum(dim=0).detach().cpu().double()
            token_count += inputs.shape[0]
    if token_count == 0:
        raise ValueError("Output reconstruction received no calibration pairs")
    mean_residual = (residual_sum / token_count).float()
    with torch.no_grad():
        student.down_projection.bias.copy_(
            mean_residual.to(student.down_projection.bias)
        )

    for parameter in student.parameters():
        parameter.requires_grad = False
    student.down_projection.weight.requires_grad = True
    student.down_projection.bias.requires_grad = True
    warmup_config = replace(
        down_only_config,
        epochs=min(8, int(down_only_config.epochs)),
        early_stopping_patience=(
            2
            if down_only_config.early_stopping_patience is None
            else min(2, int(down_only_config.early_stopping_patience))
        ),
    )
    down_only_fit = fit_operator_fp32_detailed(
        student,
        training_pairs,
        validation_pairs,
        warmup_config,
        calculation_device,
    )
    down_only_validation = evaluate_operator(
        student,
        validation_pairs,
        calculation_device,
        down_only_config.batch_size,
    )

    for parameter in student.parameters():
        parameter.requires_grad = True
    full_fit = fit_operator_fp32_detailed(
        student,
        training_pairs,
        validation_pairs,
        down_only_config,
        calculation_device,
    )
    return GatedReconstructionResult(
        module=full_fit.module,
        down_only_fit=down_only_fit,
        full_fit=full_fit,
        neuron_indices=tuple(int(index) for index in indices.detach().cpu()),
        initial_mean_residual=tuple(float(value) for value in mean_residual),
        down_only_validation_mse=float(down_only_validation.mse),
        down_only_validation_nmse=float(down_only_validation.relative_mse),
        down_only_validation_cosine=float(
            down_only_validation.cosine_similarity
        ),
    )


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
    best_validation = evaluate_operator_mse(
        module,
        validation_pairs,
        device,
        config.batch_size,
    )
    best_epoch = 0
    best_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }
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
    module.zero_grad(set_to_none=True)
    module.to(device)
    module.eval()
    return OperatorFitResult(module, tuple(history), best_epoch, best_validation)


def fit_operator_fp32_detailed(module, training_pairs, validation_pairs, config, device):
    """Fit an operator with FP32 master weights and per-epoch NMSE/cosine."""

    from ..evaluation.operator import evaluate_operator

    if training_pairs.hidden_size != validation_pairs.hidden_size:
        raise ValueError("Training and validation hidden sizes differ")
    module = module.to(device=device, dtype=torch.float32)
    parameters = tuple(parameter for parameter in module.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("Detailed local fitting requires trainable parameters")
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
        foreach=False,
    )
    if config.scheduler != "constant":
        raise ValueError("Detailed SwiGLU fitting requires a constant schedule")

    initial = evaluate_operator(module, validation_pairs, device, config.batch_size)
    best_validation = initial.mse
    best_epoch = 0
    best_state = {
        name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()
    }
    history = []
    stale_epochs = 0
    updates = 0
    started = perf_counter()
    for epoch in range(1, config.epochs + 1):
        module.train()
        squared_error = 0.0
        element_count = 0
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            errors = module(inputs) - targets
            loss = errors.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm)
            optimizer.step()
            updates += 1
            squared_error += float(errors.detach().square().sum().item())
            element_count += errors.numel()
        metrics = evaluate_operator(module, validation_pairs, device, config.batch_size)
        history.append(
            DetailedOperatorEpoch(
                epoch=epoch,
                updates=updates,
                train_mse=squared_error / element_count,
                validation_mse=metrics.mse,
                validation_nmse=metrics.relative_mse,
                validation_cosine=metrics.cosine_similarity,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                elapsed_seconds=perf_counter() - started,
            )
        )
        if metrics.mse < best_validation - config.early_stopping_min_delta:
            best_validation = metrics.mse
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in module.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if (
            config.early_stopping_patience is not None
            and stale_epochs >= config.early_stopping_patience
        ):
            break

    module.load_state_dict(best_state)
    module.zero_grad(set_to_none=True)
    module.eval()
    return DetailedOperatorFitResult(
        module=module,
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_mse=best_validation,
        updates=updates,
    )


def fit_replacement_operator(
    training_pairs,
    validation_pairs,
    config,
    device,
    intermediate_width=None,
    teacher_module=None,
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
    if config.initialization == "importance_teacher_subset":
        if teacher_module is None:
            raise ValueError(
                "Teacher-derived initialization requires the teacher module"
            )
        importance_scores = swiglu_neuron_importance_scores(
            teacher_module,
            training_pairs.inputs,
            config.batch_size,
        )
        neuron_indices = torch.argsort(
            importance_scores,
            descending=True,
            stable=True,
        )[:module.bottleneck_size].sort().values
        initialize_gated_mlp_from_teacher(
            module,
            teacher_module,
            neuron_indices,
        )
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
