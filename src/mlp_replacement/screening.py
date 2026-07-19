from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .capture import first_tensor
from .model import discover_mlp_blocks, discover_transformer_layers


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


def compute_bi_scores(model, loader, max_batches, device, scope="transformer_layer", layer_indices=None):
    """Compute canonical layer BI or adapted MLP BI in one model pass.

    Canonical BI compares each complete Transformer layer's residual-stream
    input and output. The optional MLP scope compares the MLP sublayer input
    with its raw output and is reported separately as an adapted metric.
    """

    if scope == "transformer_layer":
        refs = discover_transformer_layers(model)
        metric = "transformer_layer_bi"
    elif scope == "mlp_sublayer":
        refs = discover_mlp_blocks(model)
        metric = "mlp_sublayer_bi"
    else:
        raise ValueError(f"Unsupported BI scope: {scope}")
    selected = set(layer_indices) if layer_indices is not None else {ref.index for ref in refs}
    target_refs = tuple(ref for ref in refs if ref.index in selected)
    missing = selected - {ref.index for ref in target_refs}
    if missing:
        raise ValueError(f"Cannot screen unknown Transformer layer indices: {sorted(missing)}")

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
                raise RuntimeError(f"BI output for layer {index} has no matching input")
            inputs = pending[index].pop()
            outputs = first_tensor(output).detach()
            if inputs.shape != outputs.shape:
                raise ValueError(
                    f"BI inputs and outputs must have the same shape: {inputs.shape} != {outputs.shape}"
                )
            cosine = F.cosine_similarity(inputs.float(), outputs.float(), dim=-1, eps=1e-8)
            sums[index] += float((1.0 - cosine).sum().item())
            counts[index] += int(cosine.numel())

        handles.append(ref.module.register_forward_pre_hook(pre_hook))
        handles.append(ref.module.register_forward_hook(post_hook))

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
        for handle in handles:
            handle.remove()
        model.train(was_training)

    if batches == 0:
        raise ValueError("BI screening received no batches")
    if any(pending[index] for index in pending):
        raise RuntimeError("BI screening ended with unmatched module inputs")
    scores = {
        index: sums[index] / counts[index]
        for index in sums
        if counts[index] > 0
    }
    if len(scores) != len(target_refs):
        raise RuntimeError("Some selected modules were not executed during BI screening")
    return ScreeningResult(metric, scores, batches)
