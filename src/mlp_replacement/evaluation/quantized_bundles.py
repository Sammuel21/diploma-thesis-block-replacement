"""Packed TorchAO bundles, kept separate from historical BF16 bundles."""

from pathlib import Path

from mlp_replacement.artifacts import contained_path, file_digest, read_json, write_json_atomic
from mlp_replacement.compression.continuation import save_tensor_atomic
from mlp_replacement.compression.quantization import (
    projection_paths, quantization_footprint, recipe_record, require_backend, tensor_digest, tensor_leaves,
    verify_quantized_scope,
)


FORMAT = "mlp-replacement-torchao-v1"


def state_digests(model):
    """Hash packed leaves as well as logical shapes and tensor classes."""

    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype),
               "class": f"{type(value).__module__}.{type(value).__name__}",
               "metadata": repr(value.__tensor_flatten__()[1]) if hasattr(value, "__tensor_flatten__") else None,
               "leaves": [tensor_digest(leaf) for leaf in tensor_leaves(value)]}
        for name, value in model.state_dict().items()
    }


def export_quantized_bundle(model, tokenizer, directory, paths, bits, backend, provenance):
    directory = Path(directory)
    staging = directory.with_name(directory.name + ".building")
    if directory.exists() or staging.exists():
        raise FileExistsError(f"Bundle or incomplete staging already exists: {directory}")
    verify_quantized_scope(model, paths, bits)
    staging.mkdir(parents=True)
    model.config.save_pretrained(staging)
    tokenizer.save_pretrained(staging / "tokenizer")
    weights = staging / "model.pt"
    save_tensor_atomic(weights, model.state_dict())
    manifest = {
        "schema_version": 1, "format": FORMAT, "bits": bits,
        "quantization_recipe": recipe_record(bits),
        "projection_paths": list(paths), "backend": backend,
        "provenance": provenance, "footprint": quantization_footprint(model, paths),
        "tensor_file_bytes": weights.stat().st_size, "state_digests": state_digests(model),
        "files": {str(path.relative_to(staging)): file_digest(path)
                  for path in sorted(staging.rglob("*")) if path.is_file()},
    }
    write_json_atomic(staging / "bundle.json", manifest)
    staging.rename(directory)
    return validate_quantized_bundle(directory)


def validate_quantized_bundle(directory):
    directory = Path(directory)
    manifest = read_json(directory / "bundle.json")
    if manifest.get("format") != FORMAT or manifest.get("bits") not in (4, 8):
        raise ValueError("Unsupported quantized inference bundle")
    if manifest.get("quantization_recipe") != recipe_record(manifest["bits"]):
        raise ValueError("Quantized bundle recipe differs from the pinned library recipe")
    for name, digest in manifest["files"].items():
        if file_digest(contained_path(directory, name)) != digest:
            raise ValueError(f"Bundle content changed: {name}")
    return manifest


def load_quantized_bundle(directory, device="cuda", attention_implementation="sdpa"):
    """Reload packed tensors directly, without a dense checkpoint or requantization."""

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    directory = Path(directory)
    manifest = validate_quantized_bundle(directory)
    require_backend(manifest["backend"], manifest["bits"])
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, torch_dtype=torch.bfloat16, attn_implementation=attention_implementation,
        )
    model.requires_grad_(False)
    state = torch.load(directory / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    del state
    expected_paths = projection_paths(model, manifest["provenance"]["scope"]["eligible_layers"])
    if tuple(manifest["projection_paths"]) != expected_paths or len(expected_paths) != 66:
        raise ValueError("Quantized bundle does not cover exactly the 66 eligible projections")
    # HF meta construction leaves nonpersistent rotary buffers on meta; they
    # are absent from state_dict and must be regenerated from the saved config.
    rotary = model.model.rotary_emb
    if any(value.is_meta for value in rotary.buffers()):
        model.model.rotary_emb = type(rotary)(config=config, device=device)
    if any(value.is_meta for value in model.buffers()):
        raise ValueError("Reload left an unsupported nonpersistent buffer on meta")
    if state_digests(model) != manifest["state_digests"]:
        raise ValueError("Packed model state changed during serialization round trip")
    verify_quantized_scope(model, manifest["projection_paths"], manifest["bits"])
    if quantization_footprint(model, manifest["projection_paths"]) != manifest["footprint"]:
        raise ValueError("Packed storage footprint changed during reload")
    verify_tied_embeddings(model)
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
    model.eval()
    return model, tokenizer, manifest


def verify_tied_embeddings(model):
    if model.config.tie_word_embeddings:
        if model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
            raise ValueError("Bundle lost tied embedding/head weights")


def load_inference_bundle(directory, device="cuda", attention_implementation="sdpa"):
    """Load either supported deployment format under one attention backend."""

    from mlp_replacement.evaluation.bundles import load_bundle

    manifest = read_json(Path(directory) / "bundle.json")
    if manifest.get("format") == FORMAT:
        return load_quantized_bundle(directory, device, attention_implementation)
    model, tokenizer, manifest = load_bundle(directory, device)
    model.requires_grad_(False)
    model.set_attn_implementation(attention_implementation)
    verify_tied_embeddings(model)
    return model, tokenizer, manifest
