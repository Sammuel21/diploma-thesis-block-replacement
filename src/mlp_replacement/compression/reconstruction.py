"""Reconstruct replacement modules and preserve their exact tensor states."""

import torch

from ..model import discover_mlp_blocks, load_model_and_tokenizer
from ..operators import GatedMLPReplacement
from .surgery import replace_submodule


def replacement_state(model, paths):
    return {
        path: {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.get_submodule(path).state_dict().items()
        }
        for path in paths
    }


def load_replacement_state(model, state):
    for path, values in state.items():
        model.get_submodule(path).load_state_dict(values)


def load_operator(
    path,
    hidden_size,
    width,
    device="cpu",
    bias=False,
    down_bias=False,
):
    """Reconstruct one saved FP32 SwiGLU replacement."""

    module = GatedMLPReplacement(
        hidden_size,
        width,
        bias=bias,
        down_bias=down_bias,
    ).to(
        dtype=torch.float32
    )
    state = torch.load(path, map_location="cpu")
    module.load_state_dict(state)
    module.to(device=device, dtype=torch.float32)
    module.eval()
    return module


def build_swiglu_student(model_config, hidden_size, allocation_rows):
    """Load the dense model and insert ordered FP32 SwiGLU replacements."""

    student, tokenizer = load_model_and_tokenizer(model_config)
    blocks = {block.index: block for block in discover_mlp_blocks(student)}
    hidden = int(hidden_size)
    target_paths = []
    train_modules = []
    for row in allocation_rows:
        if row.get("retains_dense_module"):
            continue
        layer = int(row["layer"])
        module = GatedMLPReplacement(
            hidden,
            int(row["replacement_width"]),
            down_bias=bool(row.get("has_output_bias", False)),
        ).to(next(student.parameters()).device, dtype=torch.float32)
        replace_submodule(student, blocks[layer].path, module)
        target_paths.append(blocks[layer].path)
        train_modules.append(module)
    del tokenizer
    return student, target_paths, train_modules
