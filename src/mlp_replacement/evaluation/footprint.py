from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParameterFootprint:
    """Record parameter counts and theoretical weight storage for a model."""

    parameters: int
    trainable_parameters: int
    theoretical_weight_bytes: int


def parameter_footprint(model):
    """Measure the model's structural parameter footprint in its current dtype."""

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


def serialized_checkpoint_bytes(path):
    """Measure the total byte size of a checkpoint file or directory."""

    checkpoint = Path(path)
    if checkpoint.is_file():
        return checkpoint.stat().st_size
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return sum(file.stat().st_size for file in checkpoint.rglob("*") if file.is_file())
