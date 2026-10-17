"""Scoped weight-only PTQ and QAT using the pinned TorchAO recipes."""

import hashlib
from importlib.metadata import version

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
REPLACEMENT_PROJECTIONS = ("gate_projection", "up_projection", "down_projection")
GROUP_SIZE = 128


def mlp_projection_names(module):
    """Name the gate, up, and down projections of a dense MLP or a SwiGLU replacement."""

    import torch.nn as nn

    for names in (PROJECTIONS, REPLACEMENT_PROJECTIONS):
        if all(isinstance(getattr(module, name, None), nn.Linear) for name in names):
            return names
    raise ValueError(f"Unsupported MLP module for weight quantization: {type(module).__name__}")


def projection_paths(model, eligible_layers):
    """Resolve an exact set of MLP projections; never use substring filters."""

    import torch.nn as nn
    from mlp_replacement.model import discover_mlp_blocks

    refs = {ref.index: ref for ref in discover_mlp_blocks(model)}
    layers = tuple(eligible_layers)
    if len(layers) != len(set(layers)) or not set(layers).issubset(refs):
        raise ValueError("Eligible layers are duplicated or absent from the model")
    paths = tuple(f"{refs[index].path}.{name}" for index in layers
                  for name in mlp_projection_names(refs[index].module))
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

    if bits == 8:
        return Int8WeightOnlyConfig(version=2, granularity=PerRow(), set_inductor_config=False)
    if bits == 4:
        from torchao.quantization.quantize_.workflows import Int4PackingFormat

        return Int4WeightOnlyConfig(version=2, group_size=128,
                                   int4_packing_format=Int4PackingFormat.PLAIN,
                                   set_inductor_config=False)
    if bits == 3:
        import torch
        from torchao.quantization import IntxWeightOnlyConfig, MappingType, PerGroup

        # INT4's grouping and mapping at 3 bits (decision log, 2026-10-17).
        return IntxWeightOnlyConfig(weight_dtype=torch.int3, granularity=PerGroup(GROUP_SIZE),
                                    mapping_type=MappingType.ASYMMETRIC, version=2)
    raise ValueError("Only INT8, INT4, and INT3 weight-only quantization are supported")


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
    if bits == 3:
        # IntxWeightOnlyConfig has no Inductor setting; it stores one int8 per value.
        return {"version": 2, "weight_only": True, "config": "IntxWeightOnlyConfig", "bits": 3,
                "weight_dtype": "int3", "mapping": "asymmetric", "group_size": GROUP_SIZE,
                "input_axis": -1, "intx_packing_format": "UNPACKED_TO_INT8",
                "qparams_algorithm": "AFFINE"}
    raise ValueError("Unsupported quantization recipe")


def verify_quantized_scope(model, paths, bits):
    import torch
    from torchao.quantization.quantize_.workflows import Int4Tensor, Int8Tensor, IntxUnpackedToInt8Tensor

    expected_type = {3: IntxUnpackedToInt8Tensor, 4: Int4Tensor, 8: Int8Tensor}[bits]
    expected_names = {f"{path}.weight" for path in paths}
    found = set()
    for name, value in model.named_parameters():
        if isinstance(value, (Int4Tensor, Int8Tensor, IntxUnpackedToInt8Tensor)):
            if name not in expected_names or not isinstance(value, expected_type):
                raise ValueError(f"Unexpected quantized weight: {name}")
            expected_block = (1, value.shape[-1] if bits == 8 else 128)
            if tuple(value.block_size) != expected_block or str(value.dtype) != "torch.bfloat16":
                raise ValueError(f"Quantized weight has the wrong granularity or scale dtype: {name}")
            if bits == 4 and str(value.activation_dtype) != "torch.bfloat16":
                raise ValueError(f"INT4 weight changed activation precision: {name}")
            if bits == 8:
                zero_point = value.zero_point
                if value.act_quant_kwargs is not None or (zero_point is not None and zero_point.ne(0).any().item()):
                    raise ValueError(f"INT8 weight is not symmetric weight-only: {name}")
            if bits == 3 and (
                value.target_dtype != torch.int3 or value.activation_quantization is not None
                or value.qdata.dtype != torch.int8 or value.scale.dtype != torch.bfloat16
                or value.zero_point.dtype != torch.int8
            ):
                raise ValueError(f"INT3 weight is not asymmetric weight-only with BF16 scales: {name}")
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


def qat_prepare_config(bits):
    """Fake-quantization config whose conversion matches the deployed PTQ recipe."""

    from torchao.quantization.qat import QATConfig

    if bits == 4:
        return QATConfig(weight_only_recipe(4), step="prepare")
    if bits == 3:
        import torch
        from torchao.quantization.qat import IntxFakeQuantizeConfig

        # TorchAO 0.17.0 infers fake quantization from IntxWeightOnlyConfig only for
        # symmetric mapping, so the asymmetric INT3 fake quantizer is stated explicitly.
        return QATConfig(
            weight_config=IntxFakeQuantizeConfig(torch.int3, group_size=GROUP_SIZE, is_symmetric=False),
            step="prepare",
        )
    raise ValueError("QAT is implemented for INT4 and INT3 only")


def prepare_qat(model, paths, bits, trainable_paths=None):
    """Fake-quantize every path; train FP32 masters of `trainable_paths` (default: all)."""

    import torch
    from torchao.quantization import quantize_

    trainable = tuple(paths if trainable_paths is None else trainable_paths)
    if not set(trainable) <= set(paths):
        raise ValueError("QAT can train only fake-quantized projections")
    names = set(paths)
    dimensions = {path: tuple(model.get_submodule(path).weight.shape) for path in paths}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    quantize_(model, qat_prepare_config(bits), filter_fn=lambda module, name: name in names)
    for path in paths:
        module = model.get_submodule(path)
        if tuple(module.weight.shape) != dimensions[path]:
            raise ValueError("QAT preparation changed projection dimensions")
        # Native v2 Int4WeightFakeQuantizer is always active and has no
        # `enabled` attribute. Other library fake quantizers may expose one.
        if module.activation_fake_quantizer is not None or not getattr(module.weight_fake_quantizer, "enabled", True):
            raise ValueError("QAT must fake-quantize weights only, from the first update")
    for path in trainable:
        module = model.get_submodule(path)
        module.weight.data = module.weight.data.to(torch.float32)
        module.weight.requires_grad_(True)
    expected = {f"{path}.weight" for path in trainable}
    if {name for name, parameter in model.named_parameters() if parameter.requires_grad} != expected:
        raise ValueError("QAT gradient scope differs from the selected MLP weights")
    return [model.get_submodule(path).weight for path in trainable]


def prepare_int4_qat(model, paths):
    """Prepare library fake quantization with FP32 master weights only in scope."""

    return prepare_qat(model, paths, 4)


def convert_qat(model, paths, bits):
    import torch
    from torchao.quantization import quantize_
    from torchao.quantization.qat import QATConfig

    logical_parameters = sum(value.numel() for value in model.parameters())
    dimensions = {path: tuple(model.get_submodule(path).weight.shape) for path in paths}
    # The deployed PTQ/QAT recipes both use BF16 scales and BF16 activations.
    for path in paths:
        model.get_submodule(path).weight.data = model.get_submodule(path).weight.data.to(torch.bfloat16)
    names = set(paths)
    quantize_(model, QATConfig(weight_only_recipe(bits), step="convert"),
              filter_fn=lambda module, name: name in names)
    verify_quantized_scope(model, paths, bits)
    if logical_parameters != sum(value.numel() for value in model.parameters()) or dimensions != {
        path: tuple(model.get_submodule(path).weight.shape) for path in paths
    }:
        raise ValueError("QAT conversion changed logical parameters or projection dimensions")
    return {"converted_projections": len(paths), "logical_parameters": logical_parameters}


def convert_int4_qat(model, paths):
    return convert_qat(model, paths, 4)


# Width alignment and storage arithmetic for grouped formats (Stacking-1)


def padded_width(width, multiple=GROUP_SIZE):
    return -(-int(width) // multiple) * multiple


def pad_gated_replacement(module, multiple=GROUP_SIZE):
    """Append zero neurons so the down projection's rows split into whole groups.

    A zero gate row and a zero up row give a zero intermediate value, so the
    appended neurons never change the output (decision log, 2026-10-17).
    Returns the new module, or the given one when no padding is needed.
    """

    import torch
    from mlp_replacement.operators import GatedMLPReplacement

    names = mlp_projection_names(module)
    if names != REPLACEMENT_PROJECTIONS:
        raise ValueError("Only SwiGLU replacement modules are padded")
    gate, up, down = (getattr(module, name) for name in names)
    if any(projection.bias is not None for projection in (gate, up, down)):
        raise ValueError("Padding is defined for bias-free SwiGLU replacements")
    width = gate.out_features
    target = padded_width(width, multiple)
    if target == width:
        return module
    padded = GatedMLPReplacement(gate.in_features, target).to(device=gate.weight.device, dtype=gate.weight.dtype)
    with torch.no_grad():
        for projection in (padded.gate_projection, padded.up_projection, padded.down_projection):
            projection.weight.zero_()
        padded.gate_projection.weight[:width].copy_(gate.weight)
        padded.up_projection.weight[:width].copy_(up.weight)
        padded.down_projection.weight[:, :width].copy_(down.weight)
    padded.train(module.training)
    return padded


def pad_replacements(model, paths, multiple=GROUP_SIZE):
    """Pad the SwiGLU replacements at `paths`; return one record per replacement."""

    from mlp_replacement.compression.surgery import replace_submodule

    records = []
    for path in paths:
        module = model.get_submodule(path)
        real = module.gate_projection.out_features
        replacement = pad_gated_replacement(module, multiple)
        if replacement is not module:
            replace_submodule(model, path, replacement)
        records.append({"path": path, "real_width": real,
                        "width": replacement.gate_projection.out_features})
    return records


def padded_slices(model, records):
    """Yield each padded neuron block as (weight, index) views for checks and masks."""

    for record in records:
        if record["width"] == record["real_width"]:
            continue
        module = model.get_submodule(record["path"])
        real = record["real_width"]
        yield module.gate_projection.weight, (slice(real, None), slice(None))
        yield module.up_projection.weight, (slice(real, None), slice(None))
        yield module.down_projection.weight, (slice(None), slice(real, None))


def padded_weights_are_zero(model, records):
    """True when every appended neuron still has exactly zero gate, up, and down weights."""

    return all(not weight.detach()[index].any().item() for weight, index in padded_slices(model, records))


def mask_padded_gradients(model, records):
    """Zero the gradients of appended neurons so training keeps them exactly zero."""

    handles = []
    for weight, index in padded_slices(model, records):
        def hook(grad, index=index):
            grad = grad.clone()
            grad[index] = 0
            return grad
        handles.append(weight.register_hook(hook))
    return handles


def calculated_weight_bytes(model, paths, bits):
    """Packed-layout storage arithmetic for the eligible projections; the rest as stored.

    INT8: one byte per weight plus a BF16 scale and an int8 zero point per output
    row, TorchAO's Int8Tensor layout. INT4 and INT3: `bits` per weight plus a
    16-bit scale and a 16-bit zero point per group of 128 input weights, the
    INT4 layout; INT3 bytes are calculated, never measured (decision log, 2026-10-17).
    """

    eligible_bits = 0
    for path in paths:
        out_features, in_features = (int(value) for value in model.get_submodule(path).weight.shape)
        if bits == 8:
            eligible_bits += 8 * out_features * in_features + 24 * out_features
        elif bits in (3, 4):
            if in_features % GROUP_SIZE:
                raise ValueError(f"Grouped quantization needs whole groups of {GROUP_SIZE}: {path}")
            eligible_bits += bits * out_features * in_features + 32 * out_features * (in_features // GROUP_SIZE)
        else:
            raise ValueError("Calculated bytes are defined for INT8, INT4, and INT3")
    if eligible_bits % 8:
        raise ValueError("Packed eligible weights do not fill whole bytes")
    eligible_bytes = eligible_bits // 8
    outside = scoped_storage_footprint(model, {f"{path}.weight" for path in paths})["outside_weight_storage_bytes"]
    return {"bits": bits, "eligible_weight_bytes": eligible_bytes, "outside_weight_bytes": outside,
            "weight_bytes": eligible_bytes + outside,
            "convention": "int8: 1 byte/weight + 3 bytes/row; int4/int3: bits/weight + 4 bytes/group of 128"}
