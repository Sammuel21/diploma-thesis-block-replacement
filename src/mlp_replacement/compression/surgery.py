from contextlib import contextmanager
from dataclasses import dataclass

from ..model import discover_mlp_blocks


@dataclass(frozen=True)
class ReplacementRecord:
    """Describe one applied replacement and its parameter reduction."""

    layer_index: int
    path: str
    original_parameters: int
    replacement_parameters: int

    @property
    def removed_parameters(self):
        """Return the parameter count removed by this replacement."""

        return self.original_parameters - self.replacement_parameters


@dataclass(frozen=True)
class ReplacementManifest:
    """Group the structural changes made during one replacement step."""

    records: tuple[ReplacementRecord, ...]

    @property
    def removed_parameters(self):
        """Return the total parameter reduction across all recorded replacements."""

        return sum(record.removed_parameters for record in self.records)


def count_parameters(module, trainable_only=False):
    """Count all or only trainable parameters in a module."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def count_state_elements(module):
    """Count parameters and persistent buffers stored by one module."""

    return sum(value.numel() for value in module.state_dict().values())


def replace_submodule(model, path, replacement):
    """Replace one registered child module at a dotted model path."""

    try:
        parent_path, child_name = path.rsplit(".", 1)
    except ValueError as exc:
        raise ValueError(f"Replacement path must include a parent module: {path}") from exc
    parent = model.get_submodule(parent_path)
    if child_name not in parent._modules:
        raise ValueError(f"{path} is not a registered child module")
    setattr(parent, child_name, replacement)


def apply_replacements(model, replacements):
    """Cast and insert indexed MLP replacements while recording parameter changes."""

    refs = {ref.index: ref for ref in discover_mlp_blocks(model)}
    unknown = set(replacements) - set(refs)
    if unknown:
        raise ValueError(f"Cannot replace unknown MLP layers: {sorted(unknown)}")

    records = []
    for layer_index, replacement in replacements.items():
        ref = refs[layer_index]
        parameter = next(ref.module.parameters(), None)
        if parameter is None:
            parameter = next(model.parameters())
        replacement.to(device=parameter.device, dtype=parameter.dtype)
        record = ReplacementRecord(
            layer_index=layer_index,
            path=ref.path,
            original_parameters=count_parameters(ref.module),
            replacement_parameters=count_parameters(replacement),
        )
        replace_submodule(model, ref.path, replacement)
        records.append(record)
    return ReplacementManifest(tuple(records))


@contextmanager
def temporary_replacements(model, replacements):
    """Temporarily insert replacements and restore exact original modules.

    All target indices are validated before mutation. Restoration runs even if
    candidate evaluation raises an exception.
    """

    refs = {ref.index: ref for ref in discover_mlp_blocks(model)}
    unknown = set(replacements) - set(refs)
    if unknown:
        raise ValueError(f"Cannot replace unknown MLP layers: {sorted(unknown)}")
    originals = {index: refs[index].module for index in replacements}
    manifest = None
    try:
        manifest = apply_replacements(model, replacements)
        yield manifest
    finally:
        for index, original in originals.items():
            replace_submodule(model, refs[index].path, original)


@contextmanager
def temporary_replacement(model, layer_index, replacement):
    """Temporarily insert one MLP replacement and restore it afterward."""

    with temporary_replacements(model, {layer_index: replacement}) as manifest:
        yield manifest.records[0]
