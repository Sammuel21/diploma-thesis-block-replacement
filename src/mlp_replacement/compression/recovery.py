"""Distillation objectives and established recovery optimization loops."""

import math
from dataclasses import dataclass
from time import perf_counter

import torch
import torch.nn.functional as F

from ..model import autocast_context
from .teacher_cache import (
    TeacherBatch,
    TeacherCache,
    TeacherHiddenShard,
    TeacherFinalHiddenCache,
    load_hidden_shard,
    sha256_file,
    tensor_sha256,
    atomic_json,
    load_teacher_hidden_cache,
    build_teacher_final_hidden_cache,
    validate_teacher_final_hidden_cache,
    cache_teacher_logits,
)


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


@dataclass(frozen=True)
class TokenRecoveryEvent:
    """Describe one completed optimizer boundary in a token-budget run."""

    tokens_seen: int
    optimizer_updates: int
    microbatches: int
    mean_train_kl: float
    elapsed_seconds: float
    requested_checkpoint_tokens: tuple[int, ...]
    mean_train_ce: float | None = None
    mean_train_loss: float | None = None
    learning_rates: tuple[float, ...] = ()


@dataclass(frozen=True)
class TokenRecoveryResult:
    """Record the final cursor and first-step numerical diagnostics."""

    tokens_seen: int
    optimizer_updates: int
    elapsed_seconds: float
    first_step: dict | None


def cached_hidden_distillation_loss(
    student,
    teacher_head,
    batch,
    teacher_hidden,
    temperature,
    ce_weight,
    device,
    autocast_dtype,
):
    """Compute recovery loss from cached final teacher hidden states."""

    if not 0.0 <= float(ce_weight) <= 1.0:
        raise ValueError("CE loss weight must lie within [0, 1]")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    hidden = teacher_hidden.reshape(
        *input_ids.shape,
        teacher_hidden.shape[-1],
    ).to(device)
    with torch.no_grad(), autocast_context(device, autocast_dtype):
        teacher_logits = teacher_head(hidden)
    with autocast_context(device, autocast_dtype):
        student_logits = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
    mask = attention_mask.bool()
    teacher_probabilities = torch.softmax(
        teacher_logits[mask].float() / float(temperature), dim=-1
    )
    student_log_probabilities = torch.log_softmax(
        student_logits[mask].float() / float(temperature), dim=-1
    )
    kl = F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="batchmean",
    ) * (float(temperature) ** 2)
    ce = None
    predicted_tokens = 0
    if float(ce_weight):
        labels = input_ids[:, 1:].contiguous()
        valid = attention_mask[:, 1:].bool()
        labels = labels.masked_fill(~valid, -100)
        ce = F.cross_entropy(
            student_logits[:, :-1, :].float().contiguous().view(
                -1, student_logits.shape[-1]
            ),
            labels.view(-1),
            ignore_index=-100,
        )
        predicted_tokens = int(valid.sum().item())
        loss = (1.0 - float(ce_weight)) * kl + float(ce_weight) * ce
    else:
        loss = kl
    return loss, kl, ce, int(mask.sum().item()), predicted_tokens


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
    for parameter, original_requires_grad in original_flags:
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
        optimizer.zero_grad(set_to_none=True)
        for parameter, requires_grad in original_flags:
            parameter.requires_grad = requires_grad

    return RecoveryResult(tuple(history), best_epoch)


def next_optimizer_boundary(requested_tokens, effective_batch_tokens, limit=None):
    """Round a requested milestone up to a complete optimizer update."""

    if requested_tokens < 0 or effective_batch_tokens < 1:
        raise ValueError("Token milestone and effective batch must be non-negative")
    boundary = math.ceil(requested_tokens / effective_batch_tokens) * effective_batch_tokens
    return min(boundary, limit) if limit is not None else boundary


def token_checkpoint_schedule(target_tokens, interval_tokens, effective_batch_tokens):
    """Map requested periodic checkpoints to actual optimizer boundaries."""

    if target_tokens < 1 or interval_tokens < 1:
        raise ValueError("Recovery target and checkpoint interval must be positive")
    requested = list(range(interval_tokens, target_tokens, interval_tokens))
    requested.append(target_tokens)
    grouped = {}
    for value in requested:
        actual = next_optimizer_boundary(value, effective_batch_tokens, target_tokens)
        grouped.setdefault(actual, []).append(value)
    return tuple(
        (actual, tuple(values)) for actual, values in sorted(grouped.items())
    )


def online_distillation_loss(student, teacher, batch, temperature, device, autocast_dtype):
    """Compute teacher-to-student KL without retaining an unbounded logits cache."""

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    with torch.no_grad(), autocast_context(device, autocast_dtype):
        teacher_logits = teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
    with autocast_context(device, autocast_dtype):
        student_logits = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
    mask = attention_mask.bool()
    teacher_probabilities = torch.softmax(
        teacher_logits[mask].float() / temperature, dim=-1
    )
    student_log_probabilities = torch.log_softmax(
        student_logits[mask].float() / temperature, dim=-1
    )
    loss = F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="batchmean",
    ) * (temperature**2)
    return loss, int(mask.sum().item())


def online_recovery_loss(
    student,
    teacher,
    batch,
    temperature,
    ce_weight,
    device,
    autocast_dtype,
):
    """Compute online teacher KL and an optional next-token CE component."""

    if not 0.0 <= float(ce_weight) <= 1.0:
        raise ValueError("CE loss weight must lie within [0, 1]")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    with torch.no_grad(), autocast_context(device, autocast_dtype):
        teacher_logits = teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
    with autocast_context(device, autocast_dtype):
        student_logits = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
    mask = attention_mask.bool()
    teacher_probabilities = torch.softmax(
        teacher_logits[mask].float() / float(temperature), dim=-1
    )
    student_log_probabilities = torch.log_softmax(
        student_logits[mask].float() / float(temperature), dim=-1
    )
    kl = F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="batchmean",
    ) * (float(temperature) ** 2)
    ce = None
    predicted_tokens = 0
    if float(ce_weight):
        labels = input_ids[:, 1:].contiguous()
        valid = attention_mask[:, 1:].bool()
        labels = labels.masked_fill(~valid, -100)
        ce = F.cross_entropy(
            student_logits[:, :-1, :].float().contiguous().view(
                -1, student_logits.shape[-1]
            ),
            labels.view(-1),
            ignore_index=-100,
        )
        predicted_tokens = int(valid.sum().item())
        loss = (1.0 - float(ce_weight)) * kl + float(ce_weight) * ce
    else:
        loss = kl
    return loss, kl, ce, int(mask.sum().item()), predicted_tokens


def token_learning_rate_multiplier(
    tokens_after_step,
    schedule_tokens,
    scheduler,
    warmup_fraction=0.0,
    final_lr_ratio=1.0,
):
    """Return a deterministic token-position learning-rate multiplier."""

    tokens_after_step = int(tokens_after_step)
    schedule_tokens = int(schedule_tokens)
    if schedule_tokens < 1 or not 0 < tokens_after_step <= schedule_tokens:
        raise ValueError("Learning-rate position lies outside the schedule")
    if scheduler == "constant":
        return 1.0
    if scheduler != "warmup_cosine":
        raise ValueError(f"Unsupported recovery scheduler: {scheduler}")
    warmup_fraction = float(warmup_fraction)
    final_lr_ratio = float(final_lr_ratio)
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("Warmup fraction must lie within [0, 1)")
    if not 0.0 <= final_lr_ratio <= 1.0:
        raise ValueError("Final LR ratio must lie within [0, 1]")
    progress = tokens_after_step / schedule_tokens
    if warmup_fraction and progress <= warmup_fraction:
        return progress / warmup_fraction
    decay_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return final_lr_ratio + (1.0 - final_lr_ratio) * cosine


def recover_trainable_by_tokens(
    student,
    teacher,
    parameter_groups,
    train_modules,
    batch_at,
    target_tokens,
    schedule_tokens,
    microbatch_tokens,
    accumulation_steps,
    temperature,
    ce_weight,
    scheduler,
    warmup_fraction,
    final_lr_ratio,
    device,
    autocast_dtype=torch.bfloat16,
    start_tokens=0,
    start_updates=0,
    elapsed_seconds=0.0,
    optimizer_state=None,
    checkpoint_schedule=(),
    on_checkpoint=None,
    teacher_hidden_at=None,
    teacher_head=None,
    optimizer_backend="single_tensor",
):
    """Recover an explicit set of parameter groups under a token budget."""

    target_tokens = int(target_tokens)
    schedule_tokens = int(schedule_tokens)
    microbatch_tokens = int(microbatch_tokens)
    accumulation_steps = int(accumulation_steps)
    if target_tokens < 1 or schedule_tokens < target_tokens:
        raise ValueError("Recovery and schedule token budgets are inconsistent")
    if microbatch_tokens < 1 or accumulation_steps < 1:
        raise ValueError("Recovery batching values must be positive")
    if not 0 <= int(start_tokens) <= target_tokens:
        raise ValueError("Recovery start cursor lies outside the target")
    if int(start_tokens) % microbatch_tokens:
        raise ValueError("Recovery cursor must lie on a complete microbatch")

    original_flags = [(parameter, parameter.requires_grad) for parameter in student.parameters()]
    for parameter, original_requires_grad in original_flags:
        parameter.requires_grad = False
    optimizer_groups = []
    trainable = []
    seen = set()
    group_names = []
    for values in parameter_groups:
        name = str(values["name"])
        parameters = list(values["parameters"])
        if not parameters:
            raise ValueError(f"Recovery parameter group {name!r} is empty")
        for parameter in parameters:
            if id(parameter) in seen:
                raise ValueError("Recovery parameter groups overlap")
            seen.add(id(parameter))
            parameter.requires_grad = True
            trainable.append(parameter)
        optimizer_groups.append(
            {
                "params": parameters,
                "lr": float(values["learning_rate"]),
                "weight_decay": float(values.get("weight_decay", 0.0)),
                "initial_lr": float(values["learning_rate"]),
            }
        )
        group_names.append(name)
    if not trainable:
        raise ValueError("Recovery received no trainable parameters")
    if optimizer_backend == "fused":
        optimizer = torch.optim.AdamW(optimizer_groups, fused=True)
    elif optimizer_backend == "foreach":
        optimizer = torch.optim.AdamW(optimizer_groups, foreach=True)
    elif optimizer_backend == "single_tensor":
        optimizer = torch.optim.AdamW(optimizer_groups, foreach=False)
    else:
        raise ValueError(f"Unsupported optimizer backend: {optimizer_backend}")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        optimizer_state.clear()

    schedule = {int(actual): tuple(requested) for actual, requested in checkpoint_schedule}
    cursor = int(start_tokens)
    updates = int(start_updates)
    first_step = None
    total_loss = 0.0
    total_kl = 0.0
    total_ce = 0.0
    total_loss_tokens = 0
    total_ce_tokens = 0
    started = perf_counter()
    callback_seconds = 0.0
    student.eval()
    if teacher_hidden_at is None:
        if teacher is None:
            raise ValueError("Online recovery requires a dense teacher")
        teacher.eval()
    elif teacher_head is None:
        raise ValueError("Cached-hidden recovery requires the frozen teacher head")
    if teacher_head is not None:
        teacher_head.eval()
    for module in train_modules:
        module.train()

    try:
        while cursor < target_tokens:
            remaining_microbatches = math.ceil(
                (target_tokens - cursor) / microbatch_tokens
            )
            microbatches = min(accumulation_steps, remaining_microbatches)
            microbatch_counts = []
            planned_cursor = cursor
            for microbatch_index in range(microbatches):
                count = min(microbatch_tokens, target_tokens - planned_cursor)
                microbatch_counts.append(count)
                planned_cursor += count
            planned_step_tokens = sum(microbatch_counts)
            multiplier = token_learning_rate_multiplier(
                planned_cursor,
                schedule_tokens,
                scheduler,
                warmup_fraction,
                final_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = float(group["initial_lr"]) * multiplier
            optimizer.zero_grad(set_to_none=True)
            first_parameter = trainable[0]
            before = first_parameter.detach().clone() if first_step is None else None
            step_loss = 0.0
            step_kl = 0.0
            step_ce = 0.0
            step_tokens = 0
            step_ce_tokens = 0
            for count in microbatch_counts:
                batch = batch_at(cursor, count)
                if teacher_hidden_at is None:
                    loss, kl, ce, valid_tokens, predicted_tokens = (
                        online_recovery_loss(
                            student,
                            teacher,
                            batch,
                            float(temperature),
                            float(ce_weight),
                            device,
                            autocast_dtype,
                        )
                    )
                else:
                    hidden = teacher_hidden_at(cursor, count)
                    loss, kl, ce, valid_tokens, predicted_tokens = (
                        cached_hidden_distillation_loss(
                            student,
                            teacher_head,
                            batch,
                            hidden,
                            float(temperature),
                            float(ce_weight),
                            device,
                            autocast_dtype,
                        )
                    )
                if valid_tokens != count:
                    raise ValueError(
                        "Packed recovery batches must contain only valid, unpadded tokens"
                    )
                (loss * valid_tokens / planned_step_tokens).backward()
                cursor += valid_tokens
                step_loss += float(loss.detach().item()) * valid_tokens
                step_kl += float(kl.detach().item()) * valid_tokens
                if ce is not None:
                    step_ce += float(ce.detach().item()) * predicted_tokens
                    step_ce_tokens += predicted_tokens
                step_tokens += valid_tokens
            grad_norm = None
            if first_step is None:
                norms = [
                    parameter.grad.detach().float().norm()
                    for parameter in trainable
                    if parameter.grad is not None
                ]
                grad_norm = torch.linalg.vector_norm(torch.stack(norms))
            optimizer.step()
            updates += 1
            total_loss += step_loss
            total_kl += step_kl
            total_ce += step_ce
            total_loss_tokens += step_tokens
            total_ce_tokens += step_ce_tokens
            if first_step is None:
                first_step = {
                    "gradient_l2_norm": float(grad_norm.item()),
                    "first_parameter_max_abs_update": float(
                        (first_parameter.detach() - before).abs().max().item()
                    ),
                    "first_parameter_dtype": str(first_parameter.dtype),
                    "optimizer_state_dtypes": sorted(
                        {
                            str(value.dtype)
                            for state in optimizer.state.values()
                            for value in state.values()
                            if isinstance(value, torch.Tensor)
                        }
                    ),
                    "microbatches": microbatches,
                    "valid_tokens": step_tokens,
                    "parameter_groups": group_names,
                }
                del before
            if cursor in schedule and on_checkpoint is not None:
                event = TokenRecoveryEvent(
                    tokens_seen=cursor,
                    optimizer_updates=updates,
                    microbatches=microbatches,
                    mean_train_kl=total_kl / total_loss_tokens,
                    mean_train_ce=(
                        total_ce / total_ce_tokens if total_ce_tokens else None
                    ),
                    mean_train_loss=total_loss / total_loss_tokens,
                    learning_rates=tuple(
                        float(group["lr"]) for group in optimizer.param_groups
                    ),
                    elapsed_seconds=(
                        elapsed_seconds
                        + perf_counter()
                        - started
                        - callback_seconds
                    ),
                    requested_checkpoint_tokens=schedule[cursor],
                )
                callback_started = perf_counter()
                on_checkpoint(event, optimizer, first_step)
                callback_seconds += perf_counter() - callback_started
                student.eval()
                for module in train_modules:
                    module.train()
    finally:
        optimizer.zero_grad(set_to_none=True)
        for parameter, requires_grad in original_flags:
            parameter.requires_grad = requires_grad
        student.eval()

    return TokenRecoveryResult(
        tokens_seen=cursor,
        optimizer_updates=updates,
        elapsed_seconds=(
            elapsed_seconds + perf_counter() - started - callback_seconds
        ),
        first_step=first_step,
    )


def recover_replacements_by_tokens(
    student,
    teacher,
    target_paths,
    batch_at,
    target_tokens,
    microbatch_tokens,
    accumulation_steps,
    learning_rate,
    weight_decay,
    temperature,
    device,
    autocast_dtype=torch.bfloat16,
    start_tokens=0,
    start_updates=0,
    elapsed_seconds=0.0,
    optimizer_state=None,
    checkpoint_schedule=(),
    on_checkpoint=None,
):
    """Recover FP32 replacement weights along one resumable token trajectory.

    Frozen model parameters stay in their model dtype.  Replacement parameters
    remain FP32 master weights and AdamW uses FP32 states; CUDA forward operations
    run under the requested autocast dtype.  ``batch_at`` receives an absolute
    token cursor, which makes a packed finite stream exactly resumable.
    """

    target_tokens = int(target_tokens)
    microbatch_tokens = int(microbatch_tokens)
    accumulation_steps = int(accumulation_steps)
    if target_tokens < 1 or microbatch_tokens < 1 or accumulation_steps < 1:
        raise ValueError("Recovery token and accumulation budgets must be positive")
    if not 0 <= start_tokens <= target_tokens:
        raise ValueError("Recovery start cursor lies outside the target")
    if start_tokens % microbatch_tokens:
        raise ValueError("Recovery cursor must lie on a complete microbatch")

    original_flags = [(parameter, parameter.requires_grad) for parameter in student.parameters()]
    for parameter, original_requires_grad in original_flags:
        parameter.requires_grad = False
    trainable = replacement_parameters(student, target_paths)
    if any(parameter.dtype != torch.float32 for parameter in trainable):
        raise ValueError("Token-budget recovery requires FP32 replacement parameters")
    for parameter in trainable:
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
        foreach=False,
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        optimizer_state.clear()

    schedule = {int(actual): tuple(requested) for actual, requested in checkpoint_schedule}
    cursor = int(start_tokens)
    updates = int(start_updates)
    first_step = None
    total_loss = 0.0
    total_loss_tokens = 0
    started = perf_counter()
    target_modules = [student.get_submodule(path) for path in target_paths]
    student.eval()
    teacher.eval()
    for module in target_modules:
        module.train()

    try:
        while cursor < target_tokens:
            remaining_microbatches = math.ceil(
                (target_tokens - cursor) / microbatch_tokens
            )
            microbatches = min(accumulation_steps, remaining_microbatches)
            microbatch_counts = []
            planned_cursor = cursor
            for microbatch_index in range(microbatches):
                count = min(microbatch_tokens, target_tokens - planned_cursor)
                microbatch_counts.append(count)
                planned_cursor += count
            planned_step_tokens = sum(microbatch_counts)
            optimizer.zero_grad(set_to_none=True)
            first_parameter = trainable[0]
            before = first_parameter.detach().clone() if first_step is None else None
            step_loss = 0.0
            step_tokens = 0
            for count in microbatch_counts:
                batch = batch_at(cursor, count)
                loss, valid_tokens = online_distillation_loss(
                    student,
                    teacher,
                    batch,
                    float(temperature),
                    device,
                    autocast_dtype,
                )
                if valid_tokens != count:
                    raise ValueError(
                        "Packed recovery batches must contain only valid, unpadded tokens"
                    )
                (loss * valid_tokens / planned_step_tokens).backward()
                cursor += valid_tokens
                step_loss += float(loss.detach().item()) * valid_tokens
                step_tokens += valid_tokens
            grad_norm = None
            if first_step is None:
                grad_norm = torch.linalg.vector_norm(
                    torch.stack(
                        [
                            parameter.grad.detach().float().norm()
                            for parameter in trainable
                            if parameter.grad is not None
                        ]
                    )
                )
            optimizer.step()
            updates += 1
            total_loss += step_loss
            total_loss_tokens += step_tokens
            if first_step is None:
                first_step = {
                    "gradient_l2_norm": float(grad_norm.item()),
                    "first_parameter_max_abs_update": float(
                        (first_parameter.detach() - before).abs().max().item()
                    ),
                    "replacement_parameter_dtype": str(first_parameter.dtype),
                    "optimizer_state_dtypes": sorted(
                        {
                            str(value.dtype)
                            for state in optimizer.state.values()
                            for value in state.values()
                            if isinstance(value, torch.Tensor)
                        }
                    ),
                    "microbatches": microbatches,
                    "valid_tokens": step_tokens,
                }
                del before
            if cursor in schedule and on_checkpoint is not None:
                event = TokenRecoveryEvent(
                    tokens_seen=cursor,
                    optimizer_updates=updates,
                    microbatches=microbatches,
                    mean_train_kl=total_loss / total_loss_tokens,
                    elapsed_seconds=elapsed_seconds + perf_counter() - started,
                    requested_checkpoint_tokens=schedule[cursor],
                )
                on_checkpoint(event, optimizer, first_step)
    finally:
        optimizer.zero_grad(set_to_none=True)
        for parameter, requires_grad in original_flags:
            parameter.requires_grad = requires_grad
        student.eval()

    return TokenRecoveryResult(
        tokens_seen=cursor,
        optimizer_updates=updates,
        elapsed_seconds=elapsed_seconds + perf_counter() - started,
        first_step=first_step,
    )
