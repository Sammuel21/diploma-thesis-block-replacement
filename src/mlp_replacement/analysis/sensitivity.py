"""Model-wide sensitivity of individually replaced MLP operators."""

from copy import deepcopy
from dataclasses import dataclass

import torch

from ..compression.recovery import TeacherBatch, distillation_loss, mean_cache_loss
from ..compression.surgery import temporary_replacement
from ..model import get_mlp_block, resolve_dtype


@dataclass(frozen=True)
class ReplacementSensitivity:
    """Record pre-recovery damage and post-recovery residual damage."""

    layer: int
    operator: str
    recovery_updates: int
    pre_recovery_kl: float
    post_recovery_kl: float

    @property
    def kl_reduction(self):
        """Return the amount of KL divergence removed by recovery."""

        return self.pre_recovery_kl - self.post_recovery_kl

    def to_dict(self):
        """Convert the record into one dataframe- or JSON-ready row."""

        return {
            "layer": self.layer,
            "operator": self.operator,
            "recovery_updates": self.recovery_updates,
            "pre_recovery_kl": self.pre_recovery_kl,
            "post_recovery_kl": self.post_recovery_kl,
            "kl_reduction": self.kl_reduction,
        }


def replacement_kl(
    model,
    layer_index,
    replacement,
    teacher_cache,
    temperature,
    device,
):
    """Evaluate model-wide KL with one replacement temporarily integrated."""

    with temporary_replacement(model, layer_index, replacement):
        return mean_cache_loss(
            model,
            teacher_cache,
            temperature,
            device,
        )


def evaluate_replacement_sensitivity(
    model,
    candidates_by_layer,
    recovery_loader,
    validation_cache,
    recovery_config,
    update_budget,
    device=None,
):
    """Evaluate and recover each candidate as an isolated block replacement.

    ``candidates_by_layer`` maps layer indices to mappings of operator names to
    locally fitted modules. Candidates at the same layer share each streamed
    teacher batch, while different layers remain independent experiments.
    """

    device = (
        torch.device(device)
        if device is not None
        else next(model.parameters()).device
    )
    teacher_dtype = resolve_dtype(recovery_config.cache_dtype, device)
    original_flags = [
        (parameter, parameter.requires_grad)
        for parameter in model.parameters()
    ]
    records = []

    try:
        for parameter, _ in original_flags:
            parameter.requires_grad = False
        model.eval()

        for layer_index, candidates in candidates_by_layer.items():
            block = get_mlp_block(model, layer_index)
            reference = next(block.module.parameters())
            recovered = {
                name: deepcopy(module).to(
                    device=reference.device,
                    dtype=reference.dtype,
                )
                for name, module in candidates.items()
            }
            pre_recovery = {
                name: replacement_kl(
                    model,
                    layer_index,
                    module,
                    validation_cache,
                    recovery_config.temperature,
                    device,
                )
                for name, module in recovered.items()
            }

            optimizers = {}
            for name, module in recovered.items():
                parameters = list(module.parameters())
                for parameter in parameters:
                    parameter.requires_grad = True
                if parameters:
                    optimizers[name] = torch.optim.AdamW(
                        parameters,
                        lr=recovery_config.learning_rate,
                        weight_decay=recovery_config.weight_decay,
                    )

            completed_updates = 0
            for update, batch in enumerate(recovery_loader, start=1):
                if update > update_budget:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.no_grad():
                    teacher_logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ).logits.detach().to(dtype=teacher_dtype)
                teacher_batch = TeacherBatch(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits=teacher_logits,
                )

                for name, optimizer in optimizers.items():
                    module = recovered[name]
                    module.train()
                    with temporary_replacement(
                        model,
                        layer_index,
                        module,
                    ):
                        loss = distillation_loss(
                            model,
                            teacher_batch,
                            recovery_config.temperature,
                            device,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()

                completed_updates = update
                del teacher_batch, teacher_logits

            if completed_updates < update_budget:
                raise RuntimeError(
                    f"Recovery loader provided {completed_updates} of "
                    f"{update_budget} requested updates"
                )

            for name, module in recovered.items():
                module.eval()
                post_recovery = replacement_kl(
                    model,
                    layer_index,
                    module,
                    validation_cache,
                    recovery_config.temperature,
                    device,
                )
                records.append(
                    ReplacementSensitivity(
                        layer=layer_index,
                        operator=name,
                        recovery_updates=update_budget,
                        pre_recovery_kl=pre_recovery[name],
                        post_recovery_kl=post_recovery,
                    )
                )
                module.to("cpu")
    finally:
        for parameter, requires_grad in original_flags:
            parameter.requires_grad = requires_grad

    return tuple(records)
