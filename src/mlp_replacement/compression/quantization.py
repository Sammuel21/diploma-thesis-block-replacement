"""Scoped weight-only PTQ and QAT using the pinned TorchAO recipes."""

import hashlib
from importlib.metadata import version

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def projection_paths(model, eligible_layers):
    """Resolve an exact set of MLP projections; never use substring filters."""

    import torch.nn as nn
    from mlp_replacement.model import discover_mlp_blocks

    refs = {ref.index: ref for ref in discover_mlp_blocks(model)}
    layers = tuple(eligible_layers)
    if len(layers) != len(set(layers)) or not set(layers).issubset(refs):
        raise ValueError("Eligible layers are duplicated or absent from the model")
    paths = tuple(f"{refs[index].path}.{name}" for index in layers for name in PROJECTIONS)
    for path in paths:
        module = model.get_submodule(path)
        if not isinstance(module, nn.Linear) or module.bias is not None:
            raise ValueError(f"Expected a bias-free MLP linear projection: {path}")
    return paths


def tensor_digest(value):
    """Hash ordinary tensor values without NumPy's BF16 dtype limitation."""

    import torch

    value = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_state_hashes(model, paths):
    eligible = {f"{path}.weight" for path in paths}
    return {name: tensor_digest(value) for name, value in model.state_dict().items()
            if name not in eligible}


def verify_frozen_state(model, paths, expected):
    if frozen_state_hashes(model, paths) != expected:
        raise ValueError("Quantization/recovery changed weights outside the eligible MLP scope")


def tensor_leaves(value):
    """Expose physical storage inside a TorchAO tensor subclass."""

    if hasattr(value, "__tensor_flatten__"):
        names, metadata = value.__tensor_flatten__()
        for name in names:
            child = getattr(value, name)
            if child is not None:
                yield from tensor_leaves(child)
    else:
        yield value


def storage_inventory(values):
    """Count each physical allocation once, including scales and padding."""

    storages = {}
    for value in values:
        for leaf in tensor_leaves(value):
            storage = leaf.untyped_storage()
            identity = (str(leaf.device), storage.data_ptr(), storage.nbytes())
            storages[identity] = storage.nbytes()
    return storages


def quantization_footprint(model, paths):
    return scoped_storage_footprint(model, {f"{path}.weight" for path in paths})


def scoped_storage_footprint(model, eligible_names):
    """Also support supplied structural MLPs with different parameter names."""

    weights = dict(model.named_parameters())
    eligible = storage_inventory(value for name, value in weights.items() if name in eligible_names)
    outside = storage_inventory(value for name, value in weights.items() if name not in eligible_names)
    if eligible.keys() & outside.keys():
        raise ValueError("Eligible MLP storage unexpectedly aliases a protected parameter")
    return {
        "logical_parameters": sum(value.numel() for value in weights.values()),
        "eligible_parameters": sum(weights[name].numel() for name in eligible_names),
        "weight_storage_bytes": sum({**eligible, **outside}.values()),
        "eligible_weight_storage_bytes": sum(eligible.values()),
        "outside_weight_storage_bytes": sum(outside.values()),
        "buffer_storage_bytes": sum(storage_inventory(model.buffers()).values()),
    }


def require_backend(settings, bits):
    """Reject incompatible environments rather than emitting a dense fallback."""

    import torch

    if str(torch.__version__) != settings["torch"] or torch.version.cuda != settings["cuda"]:
        raise RuntimeError("Quantization-1 requires its pinned PyTorch/CUDA environment")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Quantization-1 requires a CUDA GPU with BF16 support")
    if version("torchao").split("+")[0] != settings["torchao"]:
        raise RuntimeError("Install the pinned TorchAO release in the baseline environment")
    if bits == 4:
        if version("mslk") != settings["mslk"]:
            raise RuntimeError("Install the pinned CUDA MSLK wheel for INT4")
        import mslk  # Registers the packed INT4 CUDA operators.
        if not hasattr(torch.ops.mslk, "bf16i4bf16_rowwise"):
            raise RuntimeError("The installed MSLK wheel lacks the BF16/INT4 kernel")


def weight_only_recipe(bits):
    from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, PerRow
    from torchao.quantization.quantize_.workflows import Int4PackingFormat

    if bits == 8:
        return Int8WeightOnlyConfig(version=2, granularity=PerRow(), set_inductor_config=False)
    if bits == 4:
        return Int4WeightOnlyConfig(version=2, group_size=128,
                                   int4_packing_format=Int4PackingFormat.PLAIN,
                                   set_inductor_config=False)
    raise ValueError("Only INT8 and INT4 weight-only quantization are supported")


def recipe_record(bits):
    """Portable description of the pinned library configuration, not a quantizer."""

    common = {"version": 2, "weight_only": True, "set_inductor_config": False}
    if bits == 8:
        return {**common, "config": "Int8WeightOnlyConfig", "bits": 8,
                "mapping": "symmetric", "granularity": "PerRow"}
    if bits == 4:
        return {**common, "config": "Int4WeightOnlyConfig", "bits": 4,
                "mapping": "asymmetric", "group_size": 128, "input_axis": -1,
                "int4_packing_format": "PLAIN"}
    raise ValueError("Unsupported quantization recipe")


def verify_quantized_scope(model, paths, bits):
    from torchao.quantization.quantize_.workflows import Int4Tensor, Int8Tensor

    expected_type = Int4Tensor if bits == 4 else Int8Tensor
    expected_names = {f"{path}.weight" for path in paths}
    found = set()
    for name, value in model.named_parameters():
        if isinstance(value, (Int4Tensor, Int8Tensor)):
            if name not in expected_names or not isinstance(value, expected_type):
                raise ValueError(f"Unexpected quantized weight: {name}")
            expected_block = (1, 128 if bits == 4 else value.shape[-1])
            if tuple(value.block_size) != expected_block or str(value.dtype) != "torch.bfloat16":
                raise ValueError(f"Quantized weight has the wrong granularity or scale dtype: {name}")
            if bits == 4 and str(value.activation_dtype) != "torch.bfloat16":
                raise ValueError(f"INT4 weight changed activation precision: {name}")
            if bits == 8:
                zero_point = value.zero_point
                if value.act_quant_kwargs is not None or (zero_point is not None and zero_point.ne(0).any().item()):
                    raise ValueError(f"INT8 weight is not symmetric weight-only: {name}")
            found.add(name)
        elif name not in expected_names and str(value.dtype) != "torch.bfloat16":
            raise ValueError(f"Protected parameter is no longer BF16: {name}")
    if found != expected_names:
        raise ValueError(f"Incomplete quantization: {sorted(expected_names - found)}")


def apply_ptq(model, paths, bits):
    from torchao.quantization import quantize_

    names = set(paths)
    before = quantization_footprint(model, paths)
    dimensions = {path: (model.get_submodule(path).in_features, model.get_submodule(path).out_features,
                         tuple(model.get_submodule(path).weight.shape)) for path in paths}
    frozen = frozen_state_hashes(model, paths)
    quantize_(model, weight_only_recipe(bits), filter_fn=lambda module, name: name in names)
    verify_quantized_scope(model, paths, bits)
    verify_frozen_state(model, paths, frozen)
    after = quantization_footprint(model, paths)
    if before["logical_parameters"] != after["logical_parameters"]:
        raise ValueError("Quantization changed logical parameter count")
    if dimensions != {path: (model.get_submodule(path).in_features, model.get_submodule(path).out_features,
                             tuple(model.get_submodule(path).weight.shape)) for path in paths}:
        raise ValueError("Quantization changed a projection's logical dimensions")
    return {"before": before, "after": after, "converted_projections": len(paths)}


def prepare_int4_qat(model, paths):
    """Prepare library fake quantization with FP32 master weights only in scope."""

    import torch
    from torchao.quantization import quantize_
    from torchao.quantization.qat import QATConfig

    names = set(paths)
    dimensions = {path: tuple(model.get_submodule(path).weight.shape) for path in paths}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    quantize_(model, QATConfig(weight_only_recipe(4), step="prepare"),
              filter_fn=lambda module, name: name in names)
    for path in paths:
        module = model.get_submodule(path)
        if tuple(module.weight.shape) != dimensions[path]:
            raise ValueError("QAT preparation changed projection dimensions")
        # Native v2 Int4WeightFakeQuantizer is always active and has no
        # `enabled` attribute. Other library fake quantizers may expose one.
        if module.activation_fake_quantizer is not None or not getattr(module.weight_fake_quantizer, "enabled", True):
            raise ValueError("QAT must fake-quantize weights only, from the first update")
        module.weight.data = module.weight.data.to(torch.float32)
        module.weight.requires_grad_(True)
    expected = {f"{path}.weight" for path in paths}
    if {name for name, parameter in model.named_parameters() if parameter.requires_grad} != expected:
        raise ValueError("QAT gradient scope differs from the eligible MLP weights")
    return [model.get_submodule(path).weight for path in paths]


def convert_int4_qat(model, paths):
    import torch
    from torchao.quantization import quantize_
    from torchao.quantization.qat import QATConfig

    logical_parameters = sum(value.numel() for value in model.parameters())
    dimensions = {path: tuple(model.get_submodule(path).weight.shape) for path in paths}
    # The deployed PTQ/QAT recipes both use BF16 scales and BF16 activations.
    for path in paths:
        model.get_submodule(path).weight.data = model.get_submodule(path).weight.data.to(torch.bfloat16)
    names = set(paths)
    quantize_(model, QATConfig(weight_only_recipe(4), step="convert"),
              filter_fn=lambda module, name: name in names)
    verify_quantized_scope(model, paths, 4)
    if logical_parameters != sum(value.numel() for value in model.parameters()) or dimensions != {
        path: tuple(model.get_submodule(path).weight.shape) for path in paths
    }:
        raise ValueError("QAT conversion changed logical parameters or projection dimensions")
    return {"converted_projections": len(paths), "logical_parameters": logical_parameters}
