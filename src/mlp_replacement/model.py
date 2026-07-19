import re
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BlockRef:
    """Connect a Transformer layer index with its discovered MLP module path."""

    index: int
    path: str
    module: nn.Module


@dataclass(frozen=True)
class LayerRef:
    """Connect a Transformer layer index with its complete layer module."""

    index: int
    path: str
    module: nn.Module


def resolve_device(requested):
    """Resolve an explicit or automatic Torch execution device."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def resolve_dtype(requested, device):
    """Resolve the requested numerical precision for the selected device."""

    if requested == "auto":
        if device.type != "cuda":
            return torch.float32
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return dtypes[requested]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {requested}") from exc


def load_model_and_tokenizer(config):
    """Load a causal language model and tokenizer using the experiment settings."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype, device)
    tokenizer_revision = config.tokenizer_revision or config.revision

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=tokenizer_revision,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.revision,
        torch_dtype=dtype,
        trust_remote_code=config.trust_remote_code,
    )
    model.to(device)
    model.eval()

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


LAYER_PATTERNS = (
    re.compile(r"(?:^|\.)layers\.(\d+)\.mlp$"),
    re.compile(r"(?:^|\.)h\.(\d+)\.mlp$"),
    re.compile(r"(?:^|\.)blocks\.(\d+)\.mlp$"),
)


def extract_layer_index(path):
    """Extract the numerical layer index from a known Transformer MLP path."""

    for pattern in LAYER_PATTERNS:
        match = pattern.search(path)
        if match:
            return int(match.group(1))
    return None


def discover_mlp_blocks(model):
    """Discover indexed MLP sublayers without assuming one model-specific root path."""

    candidates = [(name, module) for name, module in model.named_modules() if name.endswith(".mlp")]
    if not candidates:
        raise ValueError("No modules ending in '.mlp' were found in the model")

    parsed = [(extract_layer_index(path), path, module) for path, module in candidates]
    if all(index is not None for index, _, _ in parsed):
        parsed.sort(key=lambda item: int(item[0]))
    else:
        parsed = [(index, path, module) for index, (_, path, module) in enumerate(parsed)]

    refs = tuple(BlockRef(index=int(index), path=path, module=module) for index, path, module in parsed)
    indices = [ref.index for ref in refs]
    if len(indices) != len(set(indices)):
        raise ValueError(f"MLP topology contains duplicate layer indices: {indices}")
    return refs


def discover_transformer_layers(model):
    """Discover the complete Transformer layers that contain the model's MLPs."""

    refs = []
    for mlp_ref in discover_mlp_blocks(model):
        layer_path = mlp_ref.path.rsplit(".", 1)[0]
        refs.append(LayerRef(mlp_ref.index, layer_path, model.get_submodule(layer_path)))
    return tuple(refs)


def get_mlp_block(model, layer_index):
    """Return the discovered MLP reference for one layer index."""

    refs = {ref.index: ref for ref in discover_mlp_blocks(model)}
    try:
        return refs[layer_index]
    except KeyError as exc:
        raise ValueError(f"No MLP block found for layer index {layer_index}") from exc
