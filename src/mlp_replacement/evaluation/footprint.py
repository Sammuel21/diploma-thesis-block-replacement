from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch.nn as nn


@dataclass(frozen=True)
class ParameterFootprint:
    parameters: int
    trainable_parameters: int
    theoretical_weight_bytes: int


def parameter_footprint(model: nn.Module) -> ParameterFootprint:
    parameters = tuple(model.parameters())
    return ParameterFootprint(
        parameters=sum(parameter.numel() for parameter in parameters),
        trainable_parameters=sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        theoretical_weight_bytes=sum(
            parameter.numel() * parameter.element_size() for parameter in parameters
        ),
    )


def serialized_checkpoint_bytes(path: str | Path) -> int:
    checkpoint = Path(path)
    if checkpoint.is_file():
        return checkpoint.stat().st_size
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return sum(file.stat().st_size for file in checkpoint.rglob("*") if file.is_file())

