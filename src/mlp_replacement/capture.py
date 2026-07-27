from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActivationPairs:
    """Store corresponding MLP inputs and dense outputs for local fitting."""

    inputs: torch.Tensor
    targets: torch.Tensor

    @property
    def hidden_size(self):
        """Return the hidden dimension shared by inputs and targets."""

        return int(self.inputs.shape[-1])

    @property
    def num_tokens(self):
        """Return the number of flattened token representations."""

        return int(self.inputs.shape[0])


def first_tensor(value):
    """Extract the tensor carried directly or as the first tuple element."""

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"Expected a tensor module output, received {type(value)!r}")


def collect_module_io(model, module_path, loader, max_batches, device, storage_device="cpu", storage_dtype=torch.float32):
    """Capture token-level inputs and outputs for one model submodule."""

    if max_batches < 1:
        raise ValueError("max_batches must be positive")

    module = model.get_submodule(module_path)
    input_chunks = []
    target_chunks = []
    pending = []

    def pre_hook(_module, args):
        """Remember the input paired with the next observed module output."""

        pending.append(first_tensor(args).detach())

    def post_hook(_module, _args, output):
        """Move the paired module input and output into activation storage."""

        if not pending:
            raise RuntimeError("Module output was observed without a matching input")
        inputs = pending.pop()
        targets = first_tensor(output).detach()
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


def collect_modules_io(
    model,
    module_paths,
    loader,
    max_batches,
    device,
    storage_device="cpu",
    storage_dtype=torch.float32,
):
    """Capture corresponding inputs and outputs for several modules in one pass."""

    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    paths = tuple(dict.fromkeys(str(path) for path in module_paths))
    if not paths:
        raise ValueError("At least one module path is required")

    input_chunks = {path: [] for path in paths}
    target_chunks = {path: [] for path in paths}
    pending = {path: [] for path in paths}
    handles = []

    for path in paths:
        module = model.get_submodule(path)

        def pre_hook(_module, args, module_path=path):
            pending[module_path].append(first_tensor(args).detach())

        def post_hook(_module, _args, output, module_path=path):
            if not pending[module_path]:
                raise RuntimeError(
                    f"Module output for {module_path} has no matching input"
                )
            inputs = pending[module_path].pop()
            targets = first_tensor(output).detach()
            input_chunks[module_path].append(
                inputs.reshape(-1, inputs.shape[-1]).to(
                    device=storage_device,
                    dtype=storage_dtype,
                )
            )
            target_chunks[module_path].append(
                targets.reshape(-1, targets.shape[-1]).to(
                    device=storage_device,
                    dtype=storage_dtype,
                )
            )

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))

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
        for handle in handles:
            handle.remove()
        model.train(was_training)

    unmatched = [path for path in paths if pending[path]]
    if unmatched:
        raise RuntimeError(f"Activation capture ended with unmatched modules: {unmatched}")
    missing = [path for path in paths if not input_chunks[path]]
    if missing:
        raise ValueError(f"No activations were captured for modules: {missing}")
    return {
        path: ActivationPairs(
            inputs=torch.cat(input_chunks[path], dim=0),
            targets=torch.cat(target_chunks[path], dim=0),
        )
        for path in paths
    }
