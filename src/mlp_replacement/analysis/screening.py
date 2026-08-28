from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..capture import first_tensor
from ..model import discover_mlp_blocks, discover_transformer_layers


@dataclass(frozen=True)
class ScreeningResult:
    """Store one screening metric and its per-layer scores."""

    metric: str
    scores: dict[int, float]
    num_batches: int


def cosine_distance_score(inputs, outputs):
    """Measure mean token-level cosine distance between paired representations."""

    if inputs.shape != outputs.shape:
        raise ValueError(f"BI inputs and outputs must have the same shape: {inputs.shape} != {outputs.shape}")
    cosine = F.cosine_similarity(inputs.float(), outputs.float(), dim=-1, eps=1e-8)
    return float((1.0 - cosine).mean().item())


def select_screening_refs(refs, layer_indices):
    """Return requested screening references and reject unknown indices."""

    selected = set(layer_indices) if layer_indices is not None else {ref.index for ref in refs}
    target_refs = tuple(ref for ref in refs if ref.index in selected)
    missing = selected - {ref.index for ref in target_refs}
    if missing:
        raise ValueError(f"Cannot screen unknown Transformer layer indices: {sorted(missing)}")
    return target_refs


def run_screening_forward(model, loader, max_batches, device):
    """Run the shared no-gradient forward pass used by screening metrics."""

    was_training = model.training
    model.eval()
    batches = 0
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if batch_index >= max_batches:
                    break
                device_batch = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                model(**device_batch)
                batches += 1
    finally:
        model.train(was_training)
    if batches == 0:
        raise ValueError("Screening received no batches")
    return batches


def compute_module_input_output_scores(
    model,
    refs,
    metric,
    loader,
    max_batches,
    device,
    layer_indices=None,
):
    """Compute cosine distance directly across selected module boundaries."""

    target_refs = select_screening_refs(refs, layer_indices)
    sums = {ref.index: 0.0 for ref in target_refs}
    counts = {ref.index: 0 for ref in target_refs}
    pending = {ref.index: [] for ref in target_refs}
    handles = []

    for ref in target_refs:
        def pre_hook(_module, args, index=ref.index):
            """Remember the representation entering one screened module."""

            pending[index].append(first_tensor(args).detach())

        def post_hook(_module, _args, output, index=ref.index):
            """Accumulate cosine distance without retaining full activations."""

            if not pending[index]:
                raise RuntimeError(f"Screening output for layer {index} has no matching input")
            inputs = pending[index].pop()
            outputs = first_tensor(output).detach()
            if inputs.shape != outputs.shape:
                raise ValueError(
                    f"Screening inputs and outputs must have the same shape: "
                    f"{inputs.shape} != {outputs.shape}"
                )
            cosine = F.cosine_similarity(inputs.float(), outputs.float(), dim=-1, eps=1e-8)
            sums[index] += float((1.0 - cosine).sum().item())
            counts[index] += int(cosine.numel())

        handles.append(ref.module.register_forward_pre_hook(pre_hook))
        handles.append(ref.module.register_forward_hook(post_hook))

    try:
        batches = run_screening_forward(model, loader, max_batches, device)
    finally:
        for handle in handles:
            handle.remove()

    if any(pending[index] for index in pending):
        raise RuntimeError("Screening ended with unmatched module inputs")
    scores = {
        index: sums[index] / counts[index]
        for index in sums
        if counts[index] > 0
    }
    if len(scores) != len(target_refs):
        raise RuntimeError("Some selected modules were not executed during screening")
    return ScreeningResult(metric, scores, batches)


def compute_mlp_bi_scores(model, loader, max_batches, device, layer_indices=None):
    """Compute residual-aware MLP BI in one model pass.

    For each selected layer, the raw MLP update is subtracted from the complete
    layer output to reconstruct the residual stream immediately before the MLP
    addition. The score is the cosine distance between that residual and the
    complete layer output.

    This assumes the selected Transformer layer ends with ``residual + MLP``
    in evaluation mode, as in the thesis target model.
    """

    layer_refs = {ref.index: ref for ref in discover_transformer_layers(model)}
    mlp_refs = {ref.index: ref for ref in discover_mlp_blocks(model)}
    target_layers = select_screening_refs(tuple(layer_refs.values()), layer_indices)
    selected = tuple(ref.index for ref in target_layers)
    missing_mlp = set(selected) - set(mlp_refs)
    if missing_mlp:
        raise ValueError(f"Cannot find MLP modules for layers: {sorted(missing_mlp)}")

    sums = {index: 0.0 for index in selected}
    counts = {index: 0 for index in selected}
    pending_mlp_outputs = {index: [] for index in selected}
    handles = []

    for index in selected:
        def mlp_hook(_module, _args, output, layer_index=index):
            """Remember the raw MLP update before its residual addition."""

            pending_mlp_outputs[layer_index].append(first_tensor(output).detach())

        def layer_hook(_module, _args, output, layer_index=index):
            """Compare the residual stream immediately before and after the MLP."""

            if not pending_mlp_outputs[layer_index]:
                raise RuntimeError(
                    f"Layer {layer_index} output has no matching raw MLP output"
                )
            mlp_output = pending_mlp_outputs[layer_index].pop().float()
            layer_output = first_tensor(output).detach().float()
            if mlp_output.shape != layer_output.shape:
                raise ValueError(
                    f"MLP and layer outputs must have the same shape: "
                    f"{mlp_output.shape} != {layer_output.shape}"
                )
            residual_input = layer_output - mlp_output
            cosine = F.cosine_similarity(
                residual_input, layer_output, dim=-1, eps=1e-8
            )
            sums[layer_index] += float((1.0 - cosine).sum().item())
            counts[layer_index] += int(cosine.numel())

        handles.append(mlp_refs[index].module.register_forward_hook(mlp_hook))
        handles.append(layer_refs[index].module.register_forward_hook(layer_hook))

    try:
        batches = run_screening_forward(model, loader, max_batches, device)
    finally:
        for handle in handles:
            handle.remove()

    if any(pending_mlp_outputs[index] for index in pending_mlp_outputs):
        raise RuntimeError("MLP BI screening ended with unmatched raw MLP outputs")
    scores = {
        index: sums[index] / counts[index]
        for index in sums
        if counts[index] > 0
    }
    if len(scores) != len(selected):
        raise RuntimeError("Some selected MLP residual additions were not executed")
    return ScreeningResult("mlp_sublayer_bi", scores, batches)


def compute_raw_mlp_input_output_scores(
    model,
    loader,
    max_batches,
    device,
    layer_indices=None,
):
    """Compare normalized MLP inputs with raw MLP outputs.

    This preserves the former ``mlp_sublayer`` screening calculation as a
    separately named diagnostic. It is not residual-aware MLP BI.
    """

    return compute_module_input_output_scores(
        model,
        discover_mlp_blocks(model),
        "raw_mlp_input_output_cosine_distance",
        loader,
        max_batches,
        device,
        layer_indices,
    )


def compute_bi_scores(
    model,
    loader,
    max_batches,
    device,
    scope="transformer_layer",
    layer_indices=None,
):
    """Compute canonical Transformer-layer BI or residual-aware MLP BI.

    Canonical BI compares each complete Transformer layer's residual-stream
    input and output. The MLP scope compares the residual stream immediately
    before and after the MLP update is added.
    """

    if scope == "transformer_layer":
        return compute_module_input_output_scores(
            model,
            discover_transformer_layers(model),
            "transformer_layer_bi",
            loader,
            max_batches,
            device,
            layer_indices,
        )
    if scope == "mlp_sublayer":
        return compute_mlp_bi_scores(
            model, loader, max_batches, device, layer_indices
        )
    raise ValueError(f"Unsupported BI scope: {scope}")
