import torch
import torch.nn as nn

def list_mlp_paths(model):
    paths = []
    for name, module in model.named_modules():
        if name.endswith(".mlp"):
            paths.append(name)
    return paths


def choose_target_mlp_path(model, target_layer_idx: int):
    paths = list_mlp_paths(model)
    print("Candidate MLP paths:")
    for p in paths[:10]:
        print("  ", p)

    expected = f"model.layers.{target_layer_idx}.mlp"
    if expected in paths:
        return expected

    for p in paths:
        if f".{target_layer_idx}.mlp" in p:
            return p

    raise ValueError(f"Nepodarilo sa nájsť MLP pre layer_idx={target_layer_idx}")


def replace_submodule(root_model: nn.Module, path: str, new_module: nn.Module):
    parent_path, child_name = path.rsplit(".", 1)
    parent = root_model.get_submodule(parent_path)
    setattr(parent, child_name, new_module)