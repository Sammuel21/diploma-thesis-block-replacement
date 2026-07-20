from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class DatasetSpec:
    """Identifies one dataset split and the text field consumed by the workflow."""

    path: str
    name: str | None
    split: str
    text_column: str = "text"
    revision: str | None = None
    data_file: str | None = None
    streaming: bool = False


def default_calibration_source():
    """Return the default C4 shard used for fitting and recovery data."""

    return DatasetSpec(
        path="allenai/c4",
        name=None,
        split="train",
        data_file="en/c4-train.00000-of-01024.json.gz",
    )


def default_model_validation_source():
    """Return the WikiText-2 validation stream used to compare experiments."""

    return DatasetSpec(
        path="Salesforce/wikitext",
        name="wikitext-2-raw-v1",
        split="validation",
    )


def default_test_source():
    """Return the WikiText-2 test stream reserved for frozen final candidates."""

    return DatasetSpec(
        path="Salesforce/wikitext",
        name="wikitext-2-raw-v1",
        split="test",
    )


@dataclass(frozen=True)
class ModelConfig:
    """Controls model loading, numerical precision, and execution device."""

    model_id: str = "HuggingFaceTB/SmolLM2-1.7B"
    revision: str | None = None
    tokenizer_revision: str | None = None
    device: str = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    trust_remote_code: bool = False


@dataclass(frozen=True)
class DataConfig:
    """Defines the data sources and independent batch budgets for every stage."""

    calibration_source: DatasetSpec = field(default_factory=default_calibration_source)
    model_validation_source: DatasetSpec = field(default_factory=default_model_validation_source)
    test_source: DatasetSpec = field(default_factory=default_test_source)
    sequence_length: int = 128
    batch_size: int = 2
    num_calibration_batches: int = 24
    num_operator_validation_batches: int = 24
    num_recovery_batches: int = 512
    num_recovery_validation_batches: int = 24
    num_model_validation_batches: int | None = 24
    num_test_batches: int | None = 0
    seed: int = 21

    def __post_init__(self):
        """Reject data budgets that cannot form valid token batches."""

        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        for name in (
            "num_calibration_batches",
            "num_operator_validation_batches",
            "num_recovery_batches",
            "num_recovery_validation_batches",
            "num_model_validation_batches",
            "num_test_batches",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.num_model_validation_batches == 0:
            raise ValueError("num_model_validation_batches must be positive")


@dataclass(frozen=True)
class CaptureConfig:
    """Controls where captured MLP activations are stored and in what precision."""

    storage_device: str = "cpu"
    storage_dtype: Literal["float32", "float16", "bfloat16"] = "float32"


@dataclass(frozen=True)
class SelectionConfig:
    """Defines how replacement candidates are selected and ordered."""

    strategy: Literal["manual", "first_k", "random_k", "top_k_bi"] = "manual"
    k: int = 1
    manual_indices: tuple[int, ...] = (3,)
    bi_scope: Literal["transformer_layer", "mlp_sublayer"] = "transformer_layer"
    bi_order: Literal["asc", "desc"] = "asc"
    application_order: Literal["layer", "selection"] = "layer"
    protected_prefix: int = 1
    protected_suffix: int = 1
    seed: int = 21

    def __post_init__(self):
        """Validate the requested number of layers and protected boundaries."""

        if self.k < 1:
            raise ValueError("selection k must be positive")
        if self.protected_prefix < 0 or self.protected_suffix < 0:
            raise ValueError("protected boundary counts cannot be negative")


@dataclass(frozen=True)
class OperatorConfig:
    """Defines the replacement architecture and its local fitting procedure."""

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

    def __post_init__(self):
        """Validate operator capacity and optimization settings."""

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
    """Defines model-level knowledge-distillation recovery after replacement."""

    enabled: bool = True
    epochs: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    temperature: float = 1.0
    cache_dtype: Literal["float32", "float16", "bfloat16"] = "float16"
    early_stopping_patience: int | None = 1
    early_stopping_min_delta: float = 0.0

    def __post_init__(self):
        """Validate recovery optimization and distillation temperature."""

        if self.epochs < 1:
            raise ValueError("recovery epochs must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid recovery learning rate or weight decay")
        if self.temperature <= 0:
            raise ValueError("recovery temperature must be positive")


@dataclass(frozen=True)
class WorkflowConfig:
    """Select integration strategy; iterative replacement is deprecated."""

    strategy: Literal["one_shot", "iterative"] = "one_shot"


@dataclass(frozen=True)
class ExperimentConfig:
    """Groups the complete configuration of one replacement experiment."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    operator: OperatorConfig = field(default_factory=OperatorConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)

    def to_dict(self):
        """Convert the nested dataclass configuration into plain dictionaries."""

        return asdict(self)


def experiment_config_from_dict(data):
    """Reconstruct an experiment configuration from JSON-compatible data."""

    model = ModelConfig(**data.get("model", {}))

    data_values = dict(data.get("data", {}))
    if "calibration_source" in data_values:
        data_values["calibration_source"] = DatasetSpec(**data_values["calibration_source"])
    if "model_validation_source" in data_values:
        data_values["model_validation_source"] = DatasetSpec(**data_values["model_validation_source"])
    if "test_source" in data_values:
        data_values["test_source"] = DatasetSpec(**data_values["test_source"])
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
