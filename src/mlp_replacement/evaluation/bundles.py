"""Self-contained BF16 model bundles with explicit nonuniform MLP topology."""

import gc
import os
from pathlib import Path

from mlp_replacement.artifacts import contained_path, file_digest, read_json, write_json_atomic


def export_bundle(model, tokenizer, directory, replacements, provenance):
    """Export every inference tensor, deduplicating exact tensor aliases."""

    import torch
    from safetensors.torch import save_file

    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Bundle already exists: {directory}")
    staging = directory.with_name(directory.name + ".building")
    if staging.exists():
        raise FileExistsError(f"Incomplete bundle requires inspection: {staging}")
    staging.mkdir(parents=True)
    tensors, aliases, owners = {}, {}, {}
    parameter_names = {name for name, value in model.named_parameters(remove_duplicate=False)}
    for name, value in model.state_dict().items():
        identity = (value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype)
        if identity in owners:
            aliases[name] = owners[identity]
            continue
        owners[identity] = name
        dtype = torch.bfloat16 if name in parameter_names and value.is_floating_point() else value.dtype
        tensors[name] = value.detach().to(device="cpu", dtype=dtype).contiguous()
    model.config.save_pretrained(staging)
    tokenizer.save_pretrained(staging / "tokenizer")
    weights = staging / "model.safetensors"
    save_file(tensors, str(weights))
    parameter_count = sum(value.numel() for value in model.parameters())
    parameter_bytes = sum(value.numel() * (2 if value.is_floating_point() else value.element_size())
                          for value in model.parameters())
    # Keep model-defined buffer precision (e.g. FP32 rotary frequencies).
    # Nonpersistent buffers are regenerated from the saved base configuration.
    buffer_bytes = sum(value.numel() * value.element_size() for value in model.buffers())
    manifest = {
        "schema_version": 1, "format": "mlp-replacement-bf16-v1",
        "replacements": replacements, "aliases": aliases, "provenance": provenance,
        "parameters": parameter_count, "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes, "tensor_file_bytes": weights.stat().st_size,
        "files": {str(path.relative_to(staging)): file_digest(path)
                  for path in sorted(staging.rglob("*")) if path.is_file()},
    }
    write_json_atomic(staging / "bundle.json", manifest)
    staging.rename(directory)
    manifest["bundle_bytes"] = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    return manifest


def validate_bundle(directory):
    directory = Path(directory)
    manifest = read_json(directory / "bundle.json")
    if manifest.get("format") != "mlp-replacement-bf16-v1":
        raise ValueError("Unsupported inference bundle")
    for name, digest in manifest["files"].items():
        if file_digest(contained_path(directory, name)) != digest:
            raise ValueError(f"Bundle content changed: {name}")
    return manifest


def load_bundle(directory, device="cuda"):
    """Build topology locally; no dense checkpoint/network is needed on reload."""

    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from mlp_replacement.compression.surgery import replace_submodule
    from mlp_replacement.operators import GatedMLPReplacement

    directory = Path(directory)
    manifest = validate_bundle(directory)
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    for row in manifest["replacements"]:
        module = GatedMLPReplacement(
            config.hidden_size, row["width"], down_bias=row["down_bias"],
        ).to(dtype=torch.bfloat16)
        replace_submodule(model, row["path"], module)
    model.tie_weights()
    state = load_file(str(directory / "model.safetensors"))
    for alias, owner in manifest["aliases"].items():
        state[alias] = state[owner]
    model.load_state_dict(state, strict=True)
    for name, value in model.state_dict().items():
        if not torch.equal(value.cpu(), state[name]):
            raise ValueError(f"Bundle round-trip tensor mismatch: {name}")
    del state
    if sum(value.numel() for value in model.parameters()) != manifest["parameters"]:
        raise ValueError("Bundle parameter count changed on reload")
    if sum(value.numel() * value.element_size() for value in model.buffers()) != manifest["buffer_bytes"]:
        raise ValueError("Bundle buffer footprint changed on reload")
    if config.tie_word_embeddings and model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
        raise ValueError("Bundle lost tied embedding/head weights")
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
    model.eval().to(device)
    return model, tokenizer, manifest


def measure_resident_bundle(directory, device="cuda"):
    """Measure loading only; invoke in a fresh process, never after inference."""

    import psutil
    import torch

    process = psutil.Process(os.getpid())
    torch.cuda.init()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    before = {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
              "rss": process.memory_info().rss}
    model, tokenizer, manifest = load_bundle(directory, device)
    gc.collect()
    torch.cuda.synchronize()
    return {"gpu_allocated_delta_bytes": torch.cuda.memory_allocated() - before["allocated"],
            "gpu_reserved_delta_bytes": torch.cuda.memory_reserved() - before["reserved"],
            "host_rss_delta_bytes": process.memory_info().rss - before["rss"],
            "parameters": manifest["parameters"], "protocol": "fresh-process-model-only-no-forward"}
