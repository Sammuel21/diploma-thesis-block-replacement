from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    name: str | None
    split: str
    text_column: str = "text"
    revision: str | None = None
    data_file: str | None = None
    streaming: bool = False


def _default_calibration_source() -> DatasetSpec:
    return DatasetSpec(
        path="allenai/c4",
        name=None,
        split="train",
        data_file="en/c4-train.00000-of-01024.json.gz",
    )


def _default_evaluation_source() -> DatasetSpec:
    return DatasetSpec(
        path="wikitext",
        name="wikitext-2-raw-v1",
        split="test",
    )


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "HuggingFaceTB/SmolLM2-1.7B"
    revision: str | None = None
    tokenizer_revision: str | None = None
    device: str = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    trust_remote_code: bool = False


@dataclass(frozen=True)
class DataConfig:
    calibration_source: DatasetSpec = field(default_factory=_default_calibration_source)
    evaluation_source: DatasetSpec = field(default_factory=_default_evaluation_source)
    sequence_length: int = 128
    batch_size: int = 2
    num_calibration_batches: int = 24
    num_operator_validation_batches: int = 24
    num_recovery_batches: int = 512
    num_recovery_validation_batches: int = 24
    num_evaluation_batches: int = 24
    seed: int = 21

    def __post_init__(self) -> None:
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        for name in (
            "num_calibration_batches",
            "num_operator_validation_batches",
            "num_recovery_batches",
            "num_recovery_validation_batches",
            "num_evaluation_batches",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True)
class CaptureConfig:
    storage_device: str = "cpu"
    storage_dtype: Literal["float32", "float16", "bfloat16"] = "float32"


@dataclass(frozen=True)
class SelectionConfig:
    strategy: Literal["manual", "first_k", "random_k", "top_k_bi"] = "manual"
    k: int = 1
    manual_indices: tuple[int, ...] = (3,)
    bi_order: Literal["asc", "desc"] = "asc"
    application_order: Literal["layer", "selection"] = "layer"
    protected_prefix: int = 1
    protected_suffix: int = 1
    seed: int = 21

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("selection k must be positive")
        if self.protected_prefix < 0 or self.protected_suffix < 0:
            raise ValueError("protected boundary counts cannot be negative")


@dataclass(frozen=True)
class OperatorConfig:
    kind: Literal["linear", "bottleneck_mlp"] = "linear"
    bottleneck_ratio: float = 0.25
    activation: Literal["gelu", "silu", "relu"] = "gelu"
    bias: bool = False
    epochs: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 2048
    weight_decay: float = 0.0
    scheduler: Literal["constant", "cosine"] = "constant"
    gradient_clip_norm: float | None = None
    early_stopping_patience: int | None = 2
    early_stopping_min_delta: float = 0.0
    seed: int = 21

    def __post_init__(self) -> None:
        if not 0.0 < self.bottleneck_ratio <= 1.0:
            raise ValueError("bottleneck_ratio must be in (0, 1]")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("operator epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid operator learning rate or weight decay")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive or None")


@dataclass(frozen=True)
class RecoveryConfig:
    enabled: bool = True
    epochs: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    temperature: float = 1.0
    cache_dtype: Literal["float32", "float16", "bfloat16"] = "float16"
    early_stopping_patience: int | None = 1
    early_stopping_min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("recovery epochs must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid recovery learning rate or weight decay")
        if self.temperature <= 0:
            raise ValueError("recovery temperature must be positive")


@dataclass(frozen=True)
class WorkflowConfig:
    strategy: Literal["one_shot", "iterative"] = "one_shot"
    iterative_teacher: Literal["dense", "previous"] = "dense"
    iterative_recovery_scope: Literal["current", "all_replacements"] = "current"


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    operator: OperatorConfig = field(default_factory=OperatorConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def experiment_config_from_dict(data: Mapping[str, Any]) -> ExperimentConfig:
    model = ModelConfig(**data.get("model", {}))

    data_values = dict(data.get("data", {}))
    if "calibration_source" in data_values:
        data_values["calibration_source"] = DatasetSpec(**data_values["calibration_source"])
    if "evaluation_source" in data_values:
        data_values["evaluation_source"] = DatasetSpec(**data_values["evaluation_source"])
    data_config = DataConfig(**data_values)

    selection_values = dict(data.get("selection", {}))
    if "manual_indices" in selection_values:
        selection_values["manual_indices"] = tuple(selection_values["manual_indices"])

    return ExperimentConfig(
        model=model,
        data=data_config,
        capture=CaptureConfig(**data.get("capture", {})),
        selection=SelectionConfig(**selection_values),
        operator=OperatorConfig(**data.get("operator", {})),
        recovery=RecoveryConfig(**data.get("recovery", {})),
        workflow=WorkflowConfig(**data.get("workflow", {})),
    )

