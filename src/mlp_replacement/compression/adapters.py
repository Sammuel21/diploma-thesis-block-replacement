"""Low-rank adapters used by model-recovery experiments."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .surgery import replace_submodule


class LoRALinear(nn.Module):
    """Add a trainable low-rank update to one frozen linear projection."""

    def __init__(self, base, rank, alpha, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRA adapters require torch.nn.Linear targets")
        if not 1 <= int(rank) <= min(base.in_features, base.out_features):
            raise ValueError("LoRA rank must fit both linear dimensions")
        if float(alpha) <= 0 or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("LoRA alpha and dropout are outside their valid ranges")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Parameter(
            torch.empty(self.rank, base.in_features, device=base.weight.device)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, self.rank, device=base.weight.device)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs):
        """Return the frozen base projection plus its low-rank update."""

        base_output = self.base(inputs)
        adapter_input = self.dropout(inputs).to(dtype=self.lora_a.dtype)
        update = F.linear(F.linear(adapter_input, self.lora_a), self.lora_b)
        return base_output + update.to(dtype=base_output.dtype) * self.scaling


def install_lora_adapters(model, target_paths, rank, alpha, dropout=0.0):
    """Wrap explicit linear-module paths and return their installed adapters."""

    adapters = {}
    for path in target_paths:
        target = model.get_submodule(path)
        if isinstance(target, LoRALinear):
            raise ValueError(f"LoRA adapter is already installed at {path}")
        adapter = LoRALinear(target, rank, alpha, dropout)
        replace_submodule(model, path, adapter)
        adapters[path] = adapter
    if not adapters:
        raise ValueError("LoRA installation received no target paths")
    return adapters


def lora_parameters(adapters):
    """Return only the trainable low-rank matrices from installed adapters."""

    parameters = []
    for adapter in adapters.values():
        parameters.extend((adapter.lora_a, adapter.lora_b))
    return parameters


def merge_lora_adapters(model, adapters):
    """Merge installed LoRA updates into their base linear projections.

    The update is accumulated in FP32 and cast once to the base weight dtype,
    matching the precision used by the exported inference model.  Wrappers are
    replaced in place and the returned records describe the merged paths.
    """

    records = []
    for path, adapter in adapters.items():
        if not isinstance(adapter, LoRALinear):
            raise TypeError(f"Expected a LoRALinear adapter at {path}")
        if model.get_submodule(path) is not adapter:
            raise ValueError(f"Model no longer owns the recorded adapter at {path}")
        base = adapter.base
        update = adapter.lora_b.detach().float() @ adapter.lora_a.detach().float()
        merged = base.weight.detach().float() + update * float(adapter.scaling)
        base.weight.data.copy_(merged.to(device=base.weight.device, dtype=base.weight.dtype))
        base.train(adapter.training)
        replace_submodule(model, path, base)
        records.append(
            {
                "path": path,
                "rank": adapter.rank,
                "alpha": adapter.alpha,
                "dropout": adapter.dropout.p,
            }
        )
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise RuntimeError("Unmerged LoRA wrappers remain in the inference model")
    return records
