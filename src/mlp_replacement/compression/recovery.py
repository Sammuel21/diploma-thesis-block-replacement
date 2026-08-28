from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..model import resolve_dtype


@dataclass(frozen=True)
class TeacherBatch:
    """Store one recovery batch and its cached dense-model predictions."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    logits: torch.Tensor


@dataclass(frozen=True)
class TeacherCache:
    """Keep dense-model predictions available after its MLPs are replaced."""

    batches: tuple[TeacherBatch, ...]

    def __len__(self):
        return len(self.batches)


@dataclass(frozen=True)
class RecoveryEpoch:
    """Record training and validation KL divergence for one recovery epoch."""

    epoch: int
    train_kl: float
    validation_kl: float | None


@dataclass(frozen=True)
class RecoveryResult:
    """Return recovery history and the epoch whose replacement state was retained."""

    history: tuple[RecoveryEpoch, ...]
    best_epoch: int | None


def cache_teacher_logits(model, loader, max_batches, device, cache_dtype="float16"):
    """Cache dense-model logits on CPU for later knowledge-distillation recovery."""

    if max_batches < 1:
        raise ValueError("Teacher-cache max_batches must be positive")
    dtype = resolve_dtype(cache_dtype, torch.device(device))
    batches = []
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


def distillation_loss(student, batch, temperature, device):
    """Compute temperature-scaled KL divergence from teacher to student predictions."""

    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)
    teacher_logits = batch.logits.to(device=device, dtype=torch.float32)
    student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    mask = attention_mask.bool()
    teacher_probabilities = torch.softmax(teacher_logits[mask] / temperature, dim=-1)
    student_log_probabilities = torch.log_softmax(student_logits[mask] / temperature, dim=-1)
    return F.kl_div(student_log_probabilities, teacher_probabilities, reduction="batchmean") * (temperature**2)


def mean_cache_loss(student, cache, temperature, device):
    """Average KL divergence across a cached recovery-validation set."""

    losses = []
    student.eval()
    with torch.no_grad():
        for batch in cache.batches:
            losses.append(float(distillation_loss(student, batch, temperature, device).item()))
    return sum(losses) / len(losses)


def replacement_parameters(student, target_paths):
    """Collect unique parameters belonging to the replacement modules."""

    parameters = []
    seen = set()
    for path in target_paths:
        for parameter in student.get_submodule(path).parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
    if not parameters:
        raise ValueError("Recovery target paths contain no trainable parameters")
    return parameters


def recover_replacements(student, training_cache, validation_cache, target_paths, config, device):
    """Fine-tune only replacement modules against cached teacher predictions."""

    if not config.enabled:
        return RecoveryResult((), None)

    original_flags = [(parameter, parameter.requires_grad) for parameter in student.parameters()]
    for parameter, _ in original_flags:
        parameter.requires_grad = False
    trainable = replacement_parameters(student, target_paths)
    for parameter in trainable:
        parameter.requires_grad = True

    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    target_modules = [student.get_submodule(path) for path in target_paths]
    best_state = None
    best_validation = float("inf")
    best_epoch = None
    stale_epochs = 0
    history = []

    try:
        for epoch in range(1, config.epochs + 1):
            student.eval()
            for module in target_modules:
                module.train()
            losses = []
            for batch in training_cache.batches:
                loss = distillation_loss(student, batch, config.temperature, device)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().item()))

            train_kl = sum(losses) / len(losses)
            validation_kl = (
                mean_cache_loss(student, validation_cache, config.temperature, device)
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
