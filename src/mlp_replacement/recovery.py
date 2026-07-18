from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RecoveryConfig
from .model import resolve_dtype


@dataclass(frozen=True)
class TeacherBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    logits: torch.Tensor


@dataclass(frozen=True)
class TeacherCache:
    batches: tuple[TeacherBatch, ...]

    def __len__(self) -> int:
        return len(self.batches)


@dataclass(frozen=True)
class RecoveryEpoch:
    epoch: int
    train_kl: float
    validation_kl: float | None


@dataclass(frozen=True)
class RecoveryResult:
    history: tuple[RecoveryEpoch, ...]
    best_epoch: int | None


def cache_teacher_logits(
    model: nn.Module,
    loader,
    max_batches: int,
    device: torch.device | str,
    cache_dtype: str = "float16",
) -> TeacherCache:
    if max_batches < 1:
        raise ValueError("Teacher-cache max_batches must be positive")
    dtype = resolve_dtype(cache_dtype, torch.device(device))
    batches: list[TeacherBatch] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if batch_index >= max_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                batches.append(
                    TeacherBatch(
                        input_ids=batch["input_ids"].detach().cpu(),
                        attention_mask=batch["attention_mask"].detach().cpu(),
                        logits=output.logits.detach().to(dtype=dtype, device="cpu"),
                    )
                )
    finally:
        model.train(was_training)
    if not batches:
        raise ValueError("Teacher cache received no batches")
    return TeacherCache(tuple(batches))


def _distillation_loss(
    student: nn.Module,
    batch: TeacherBatch,
    temperature: float,
    device: torch.device | str,
) -> torch.Tensor:
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)
    teacher_logits = batch.logits.to(device=device, dtype=torch.float32)
    student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    mask = attention_mask.bool()
    teacher_probabilities = torch.softmax(teacher_logits[mask] / temperature, dim=-1)
    student_log_probabilities = torch.log_softmax(student_logits[mask] / temperature, dim=-1)
    return F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="batchmean",
    ) * (temperature**2)


def _mean_cache_loss(
    student: nn.Module,
    cache: TeacherCache,
    temperature: float,
    device: torch.device | str,
) -> float:
    losses = []
    student.eval()
    with torch.no_grad():
        for batch in cache.batches:
            losses.append(float(_distillation_loss(student, batch, temperature, device).item()))
    return sum(losses) / len(losses)


def _replacement_parameters(student: nn.Module, target_paths: Sequence[str]) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for path in target_paths:
        for parameter in student.get_submodule(path).parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
    if not parameters:
        raise ValueError("Recovery target paths contain no trainable parameters")
    return parameters


def recover_replacements(
    student: nn.Module,
    training_cache: TeacherCache,
    validation_cache: TeacherCache | None,
    target_paths: Sequence[str],
    config: RecoveryConfig,
    device: torch.device | str,
) -> RecoveryResult:
    if not config.enabled:
        return RecoveryResult((), None)

    original_flags = [(parameter, parameter.requires_grad) for parameter in student.parameters()]
    for parameter, _ in original_flags:
        parameter.requires_grad = False
    trainable = _replacement_parameters(student, target_paths)
    for parameter in trainable:
        parameter.requires_grad = True

    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    target_modules = [student.get_submodule(path) for path in target_paths]
    best_state: list[dict[str, torch.Tensor]] | None = None
    best_validation = float("inf")
    best_epoch: int | None = None
    stale_epochs = 0
    history: list[RecoveryEpoch] = []

    try:
        for epoch in range(1, config.epochs + 1):
            student.eval()
            for module in target_modules:
                module.train()
            losses = []
            for batch in training_cache.batches:
                loss = _distillation_loss(student, batch, config.temperature, device)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().item()))

            train_kl = sum(losses) / len(losses)
            validation_kl = (
                _mean_cache_loss(student, validation_cache, config.temperature, device)
                if validation_cache is not None
                else None
            )
            history.append(RecoveryEpoch(epoch, train_kl, validation_kl))

            monitored = validation_kl if validation_kl is not None else train_kl
            if monitored < best_validation - config.early_stopping_min_delta:
                best_validation = monitored
                best_epoch = epoch
                best_state = [
                    {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in module.state_dict().items()
                    }
                    for module in target_modules
                ]
                stale_epochs = 0
            else:
                stale_epochs += 1

            if (
                config.early_stopping_patience is not None
                and stale_epochs >= config.early_stopping_patience
            ):
                break

        if best_state is not None:
            for module, state in zip(target_modules, best_state, strict=True):
                module.load_state_dict(state)
        student.eval()
    finally:
        for parameter, requires_grad in original_flags:
            parameter.requires_grad = requires_grad

    return RecoveryResult(tuple(history), best_epoch)

