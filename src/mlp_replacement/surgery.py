from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn

from .model import BlockRef, discover_mlp_blocks


@dataclass(frozen=True)
class ReplacementRecord:
    layer_index: int
    path: str
    original_parameters: int
    replacement_parameters: int

    @property
    def removed_parameters(self) -> int:
        return self.original_parameters - self.replacement_parameters


@dataclass(frozen=True)
class ReplacementManifest:
    records: tuple[ReplacementRecord, ...]

    @property
    def removed_parameters(self) -> int:
        return sum(record.removed_parameters for record in self.records)


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def replace_submodule(model: nn.Module, path: str, replacement: nn.Module) -> None:
    try:
        parent_path, child_name = path.rsplit(".", 1)
    except ValueError as exc:
        raise ValueError(f"Replacement path must include a parent module: {path}") from exc
    parent = model.get_submodule(parent_path)
    if child_name not in parent._modules:
        raise ValueError(f"{path} is not a registered child module")
    setattr(parent, child_name, replacement)


def _module_device_dtype(module: nn.Module, model: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters(), None)
    if parameter is None:
        parameter = next(model.parameters())
    return parameter.device, parameter.dtype


def apply_replacements(
    model: nn.Module,
    replacements: Mapping[int, nn.Module],
) -> ReplacementManifest:
    refs: dict[int, BlockRef] = {ref.index: ref for ref in discover_mlp_blocks(model)}
    unknown = set(replacements) - set(refs)
    if unknown:
        raise ValueError(f"Cannot replace unknown MLP layers: {sorted(unknown)}")

    records: list[ReplacementRecord] = []
    for layer_index, replacement in replacements.items():
        ref = refs[layer_index]
        device, dtype = _module_device_dtype(ref.module, model)
        replacement.to(device=device, dtype=dtype)
        record = ReplacementRecord(
            layer_index=layer_index,
            path=ref.path,
            original_parameters=count_parameters(ref.module),
            replacement_parameters=count_parameters(replacement),
        )
        replace_submodule(model, ref.path, replacement)
        records.append(record)
    return ReplacementManifest(tuple(records))

