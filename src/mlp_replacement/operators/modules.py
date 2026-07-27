import torch
import torch.nn as nn


class ZeroReplacement(nn.Module):
    """Return a parameter-free zero MLP contribution."""

    def forward(self, inputs):
        """Return zeros with the input shape and dtype."""

        return torch.zeros_like(inputs)


class MeanReplacement(nn.Module):
    """Return one calibration-set mean output for every token."""

    def __init__(self, mean_output):
        super().__init__()
        if mean_output.ndim != 1:
            raise ValueError("mean_output must be one-dimensional")
        self.register_buffer("mean_output", mean_output.detach().clone())

    def forward(self, inputs):
        """Broadcast the stored mean over all leading input dimensions."""

        return self.mean_output.to(
            device=inputs.device,
            dtype=inputs.dtype,
        ).expand_as(inputs)


class LinearReplacement(nn.Module):
    """Approximate an MLP sublayer with one hidden-size linear projection."""

    def __init__(self, hidden_size, bias=False):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, inputs):
        """Apply the learned linear approximation."""

        return self.projection(inputs)


class LowRankLinearReplacement(nn.Module):
    """Approximate an MLP output with a rank-constrained linear mapping."""

    def __init__(self, hidden_size, rank, bias=False):
        super().__init__()
        if not 1 <= rank <= hidden_size:
            raise ValueError("rank must lie within [1, hidden_size]")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.input_projection = nn.Linear(hidden_size, rank, bias=False)
        self.output_projection = nn.Linear(rank, hidden_size, bias=bias)

    def forward(self, inputs):
        """Apply the two factors of the low-rank mapping."""

        return self.output_projection(self.input_projection(inputs))


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


class GatedMLPReplacement(nn.Module):
    """Preserve a SwiGLU-style operator while reducing its intermediate width."""

    def __init__(self, hidden_size, bottleneck_size, bias=False):
        super().__init__()
        if bottleneck_size < 1:
            raise ValueError("bottleneck_size must be positive")
        self.bottleneck_size = int(bottleneck_size)
        self.gate_projection = nn.Linear(hidden_size, bottleneck_size, bias=bias)
        self.up_projection = nn.Linear(hidden_size, bottleneck_size, bias=bias)
        self.down_projection = nn.Linear(bottleneck_size, hidden_size, bias=bias)
        self.activation = nn.SiLU()

    def forward(self, inputs):
        """Apply the gated intermediate representation and down projection."""

        gate = self.activation(self.gate_projection(inputs))
        values = self.up_projection(inputs)
        return self.down_projection(gate * values)


class HybridReplacement(nn.Module):
    """Combine a low-rank linear map with a compact nonlinear residual."""

    def __init__(
        self,
        hidden_size,
        rank,
        bottleneck_size,
        activation="silu",
        bias=False,
    ):
        super().__init__()
        self.linear = LowRankLinearReplacement(hidden_size, rank, bias=bias)
        self.nonlinear = BottleneckMLPReplacement(
            hidden_size,
            bottleneck_ratio=bottleneck_size / hidden_size,
            activation=activation,
            bias=bias,
        )
        if self.nonlinear.up_projection.out_features != bottleneck_size:
            raise RuntimeError("Hybrid nonlinear width was rounded unexpectedly")
        self.rank = int(rank)
        self.bottleneck_size = int(bottleneck_size)

    def forward(self, inputs):
        """Add the linear component and learned nonlinear correction."""

        return self.linear(inputs) + self.nonlinear(inputs)


def linear_svd(linear):
    """Compute one reusable float32 SVD of a dense linear replacement."""

    if not isinstance(linear, LinearReplacement):
        raise TypeError("linear must be a LinearReplacement")
    weight = linear.projection.weight.detach().float()
    return torch.linalg.svd(weight, full_matrices=False)


def initialize_low_rank_from_svd(
    low_rank,
    left,
    singular_values,
    right,
    bias=None,
):
    """Initialize low-rank factors from a reusable dense-map SVD."""

    if not isinstance(low_rank, LowRankLinearReplacement):
        raise TypeError("low_rank must be a LowRankLinearReplacement")
    rank = low_rank.rank
    if singular_values.numel() < rank:
        raise ValueError("SVD does not contain enough singular components")
    roots = singular_values[:rank].sqrt()
    input_weight = roots[:, None] * right[:rank, :]
    output_weight = left[:, :rank] * roots[None, :]
    with torch.no_grad():
        low_rank.input_projection.weight.copy_(
            input_weight.to(low_rank.input_projection.weight)
        )
        low_rank.output_projection.weight.copy_(
            output_weight.to(low_rank.output_projection.weight)
        )
        if low_rank.output_projection.bias is not None:
            if bias is None:
                low_rank.output_projection.bias.zero_()
            else:
                low_rank.output_projection.bias.copy_(
                    bias.to(low_rank.output_projection.bias)
                )
    return low_rank


def initialize_low_rank_from_linear(low_rank, linear):
    """Initialize low-rank factors from a dense linear map's truncated SVD."""

    left, singular_values, right = linear_svd(linear)
    return initialize_low_rank_from_svd(
        low_rank,
        left,
        singular_values,
        right,
        linear.projection.bias,
    )


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
