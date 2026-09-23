"""Tensor ownership and temporary model mutation; no training or downloads."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn

from mlp_replacement.compression import reconstruction, surgery
from mlp_replacement.model import discover_mlp_blocks
from mlp_replacement.operators import GatedMLPReplacement


def small_model():
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module() for index in range(3)])
    for layer in model.layers:
        layer.mlp = nn.Linear(4, 4, bias=False).to(dtype=torch.float16)
    return model


class ReconstructionContracts(unittest.TestCase):
    def test_snapshot_is_independent_and_restore_is_scoped(self):
        model = small_model()
        original = model.layers[0].mlp.weight.detach().clone()
        untouched = model.layers[1].mlp.weight.detach().clone()
        state = reconstruction.replacement_state(model, ["layers.0.mlp"])
        self.assertEqual(list(state), ["layers.0.mlp"])
        self.assertEqual(list(state["layers.0.mlp"]), ["weight"])
        saved = state["layers.0.mlp"]["weight"]
        self.assertEqual(saved.dtype, torch.float16)
        self.assertEqual(saved.device.type, "cpu")
        self.assertNotEqual(saved.data_ptr(), model.layers[0].mlp.weight.data_ptr())
        with torch.no_grad():
            model.layers[0].mlp.weight.add_(1)
        self.assertTrue(torch.equal(saved, original))
        reconstruction.load_replacement_state(model, state)
        self.assertTrue(torch.equal(model.layers[0].mlp.weight, original))
        self.assertTrue(torch.equal(model.layers[1].mlp.weight, untouched))

    def test_fp32_temporary_insertion_restores_identity_on_failure(self):
        model = small_model()
        blocks = {ref.index: ref for ref in discover_mlp_blocks(model)}
        original = model.layers[0].mlp
        replacement = GatedMLPReplacement(4, 2).float()
        for fail in (False, True):
            try:
                with surgery.temporary_fp32_replacements(model, blocks, {0: replacement}):
                    self.assertIs(model.layers[0].mlp, replacement)
                    self.assertEqual(next(replacement.parameters()).dtype, torch.float32)
                    if fail:
                        raise RuntimeError("injected")
            except RuntimeError as error:
                self.assertEqual(str(error), "injected")
            self.assertIs(model.layers[0].mlp, original)
        with surgery.temporary_replacements(model, {0: replacement}):
            self.assertEqual(next(replacement.parameters()).dtype, torch.float16)
        self.assertIs(model.layers[0].mlp, original)

    def test_student_preserves_allocation_order_bias_and_dense_modules(self):
        model = small_model()
        dense = model.layers[1].mlp
        rows = [
            {"layer": 2, "replacement_width": 3, "has_output_bias": True},
            {"layer": 1, "retains_dense_module": True},
            {"layer": 0, "replacement_width": 2},
        ]
        config = object()
        with patch.object(reconstruction, "load_model_and_tokenizer", return_value=(model, object())) as load:
            student, paths, modules = reconstruction.build_swiglu_student(config, 4, rows)
        load.assert_called_once_with(config)
        self.assertIs(student, model)
        self.assertIs(model.layers[1].mlp, dense)
        self.assertEqual(paths, ["layers.2.mlp", "layers.0.mlp"])
        self.assertEqual([module.bottleneck_size for module in modules], [3, 2])
        self.assertIsNotNone(modules[0].down_projection.bias)
        self.assertIsNone(modules[1].down_projection.bias)
        self.assertTrue(all(parameter.dtype == torch.float32 for module in modules for parameter in module.parameters()))

    def test_saved_operator_is_fp32_and_evaluating(self):
        module = GatedMLPReplacement(4, 2, down_bias=True).float()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "operator.pt"
            torch.save(module.state_dict(), path)
            loaded = reconstruction.load_operator(path, 4, 2, down_bias=True)
        self.assertFalse(loaded.training)
        for name, value in loaded.state_dict().items():
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(torch.equal(value, module.state_dict()[name]))


if __name__ == "__main__":
    unittest.main()
