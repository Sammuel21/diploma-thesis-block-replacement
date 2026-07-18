from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ActivationPairs:
    inputs: torch.Tensor
    targets: torch.Tensor

    @property
    def hidden_size(self) -> int:
        return int(self.inputs.shape[-1])

    @property
    def num_tokens(self) -> int:
        return int(self.inputs.shape[0])


def _first_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"Expected a tensor module output, received {type(value)!r}")


def collect_module_io(
    model: nn.Module,
    module_path: str,
    loader,
    max_batches: int,
    device: torch.device | str,
    storage_device: torch.device | str = "cpu",
    storage_dtype: torch.dtype = torch.float32,
) -> ActivationPairs:
    if max_batches < 1:
        raise ValueError("max_batches must be positive")

    module = model.get_submodule(module_path)
    input_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    pending: list[torch.Tensor] = []

    def pre_hook(_module, args):
        pending.append(_first_tensor(args).detach())

    def post_hook(_module, _args, output):
        if not pending:
            raise RuntimeError("Module output was observed without a matching input")
        inputs = pending.pop()
        targets = _first_tensor(output).detach()
        input_chunks.append(
            inputs.reshape(-1, inputs.shape[-1]).to(device=storage_device, dtype=storage_dtype)
        )
        target_chunks.append(
            targets.reshape(-1, targets.shape[-1]).to(device=storage_device, dtype=storage_dtype)
        )

    pre_handle = module.register_forward_pre_hook(pre_hook)
    post_handle = module.register_forward_hook(post_hook)
    was_training = model.training
    model.eval()
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
    finally:
        pre_handle.remove()
        post_handle.remove()
        model.train(was_training)

    if pending:
        raise RuntimeError("Activation capture ended with unmatched module inputs")
    if not input_chunks:
        raise ValueError(f"No activations were captured for {module_path}")

    return ActivationPairs(
        inputs=torch.cat(input_chunks, dim=0),
        targets=torch.cat(target_chunks, dim=0),
    )

