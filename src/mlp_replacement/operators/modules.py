import torch.nn as nn


class LinearReplacement(nn.Module):
    """Approximate an MLP sublayer with one hidden-size linear projection."""

    def __init__(self, hidden_size, bias=False):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, inputs):
        """Apply the learned linear approximation."""

        return self.projection(inputs)


def create_activation(name):
    """Construct the activation requested for a bottleneck replacement."""

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
    """Approximate an MLP with a smaller two-layer bottleneck network."""

    def __init__(self, hidden_size, bottleneck_ratio, activation="gelu", bias=False):
        super().__init__()
        bottleneck_size = max(1, round(hidden_size * bottleneck_ratio))
        self.up_projection = nn.Linear(hidden_size, bottleneck_size, bias=bias)
        self.activation = create_activation(activation)
        self.down_projection = nn.Linear(bottleneck_size, hidden_size, bias=bias)

    def forward(self, inputs):
        """Project through the reduced hidden representation and back."""

        return self.down_projection(self.activation(self.up_projection(inputs)))


def make_replacement_operator(hidden_size, config):
    """Construct the replacement architecture selected in the configuration."""

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
