from __future__ import annotations

import torch.nn as nn

from ..config import OperatorConfig


class LinearReplacement(nn.Module):
    def __init__(self, hidden_size: int, bias: bool = False):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, inputs):
        return self.projection(inputs)


def _activation(name: str) -> nn.Module:
    activations = {
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "relu": nn.ReLU,
    }
    try:
        return activations[name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


class BottleneckMLPReplacement(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        bottleneck_ratio: float,
        activation: str = "gelu",
        bias: bool = False,
    ):
        super().__init__()
        bottleneck_size = max(1, round(hidden_size * bottleneck_ratio))
        self.up_projection = nn.Linear(hidden_size, bottleneck_size, bias=bias)
        self.activation = _activation(activation)
        self.down_projection = nn.Linear(bottleneck_size, hidden_size, bias=bias)

    def forward(self, inputs):
        return self.down_projection(self.activation(self.up_projection(inputs)))


def make_replacement_operator(hidden_size: int, config: OperatorConfig) -> nn.Module:
    if config.kind == "linear":
        return LinearReplacement(hidden_size, bias=config.bias)
    if config.kind == "bottleneck_mlp":
        return BottleneckMLPReplacement(
            hidden_size=hidden_size,
            bottleneck_ratio=config.bottleneck_ratio,
            activation=config.activation,
            bias=config.bias,
        )
    raise ValueError(f"Unsupported replacement operator: {config.kind}")

