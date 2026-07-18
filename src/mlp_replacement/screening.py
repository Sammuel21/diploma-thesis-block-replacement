from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import BlockRef, discover_mlp_blocks


@dataclass(frozen=True)
class ScreeningResult:
    metric: str
    scores: dict[int, float]
    num_batches: int


def bi_cosine_score(inputs: torch.Tensor, outputs: torch.Tensor) -> float:
    if inputs.shape != outputs.shape:
        raise ValueError(f"BI inputs and outputs must have the same shape: {inputs.shape} != {outputs.shape}")
    cosine = F.cosine_similarity(inputs.float(), outputs.float(), dim=-1, eps=1e-8)
    return float((1.0 - cosine).mean().item())


def compute_bi_scores(
    model: nn.Module,
    loader,
    max_batches: int,
    device: torch.device | str,
    layer_indices: Sequence[int] | None = None,
) -> ScreeningResult:
    refs = discover_mlp_blocks(model)
    selected = set(layer_indices) if layer_indices is not None else {ref.index for ref in refs}
    target_refs: tuple[BlockRef, ...] = tuple(ref for ref in refs if ref.index in selected)
    missing = selected - {ref.index for ref in target_refs}
    if missing:
        raise ValueError(f"Cannot screen unknown MLP layer indices: {sorted(missing)}")

    sums = {ref.index: 0.0 for ref in target_refs}
    counts = {ref.index: 0 for ref in target_refs}
    pending: dict[int, list[torch.Tensor]] = {ref.index: [] for ref in target_refs}
    handles = []

    for ref in target_refs:
        def pre_hook(_module, args, index=ref.index):
            pending[index].append(args[0].detach())

        def post_hook(_module, _args, output, index=ref.index):
            inputs = pending[index].pop()
            outputs = output[0] if isinstance(output, (tuple, list)) else output
            cosine = F.cosine_similarity(inputs.float(), outputs.detach().float(), dim=-1, eps=1e-8)
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
    scores = {
        index: sums[index] / counts[index]
        for index in sums
        if counts[index] > 0
    }
    if len(scores) != len(target_refs):
        raise RuntimeError("Some MLP blocks were not executed during BI screening")
    return ScreeningResult("bi_cosine", scores, batches)

