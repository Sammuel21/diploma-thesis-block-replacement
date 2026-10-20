"""Stacking-1 contracts: configuration, grid, padding, bytes, and INT3; no model download or GPU job."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn

from mlp_replacement.artifacts import read_json, write_json_atomic
from mlp_replacement.compression import quantization
from mlp_replacement.operators import GatedMLPReplacement
from workflows.runs.model.stacking import stacking_1 as runner

TORCHAO_AVAILABLE = importlib.util.find_spec("torchao") is not None


class DenseMLP(nn.Module):
    def __init__(self, hidden, width):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, width, bias=False)
        self.up_proj = nn.Linear(hidden, width, bias=False)
        self.down_proj = nn.Linear(width, hidden, bias=False)


class Layer(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp


class TinyModel(nn.Module):
    """Three indexed layers: a dense MLP and two SwiGLU replacements."""

    def __init__(self, hidden=128, widths=(256, 200, 128)):
        super().__init__()
        mlps = [DenseMLP(hidden, widths[0])] + [GatedMLPReplacement(hidden, width) for width in widths[1:]]
        self.layers = nn.ModuleList(Layer(mlp) for mlp in mlps)
        self.norm = nn.Parameter(torch.ones(hidden))


class ConfigurationContracts(unittest.TestCase):
    def setUp(self):
        self.settings, self.path = runner.load_settings(runner.DEFAULT_CONFIG)

    def test_shared_blocks_equal_swiglu8_and_quantization1(self):
        swiglu8 = read_json(ROOT / "workflows/configs/model/swiglu/swiglu-8.json")
        quantization1 = read_json(ROOT / "workflows/configs/model/baseline/quantization-1.json")
        for key in ("model", "data", "evaluation", "strategies", "recovery", "long_context"):
            self.assertEqual(self.settings[key], swiglu8[key])
        self.assertEqual(self.settings["backend"], quantization1["backend"])
        self.assertEqual(self.settings["scope"]["eligible_layers"], list(range(1, 23)))
        self.assertEqual(self.settings["scope"]["expected_projections"], 66)
        # The evaluation that makes every point comparable with SwiGLU-8 and Quantization-1.
        evaluation = self.settings["evaluation"]
        self.assertEqual((evaluation["contexts"], evaluation["strides"]), ([128, 2048, 8192], [64, 1024, 4096]))
        self.assertEqual(evaluation["tasks"], ["piqa", "arc_easy", "arc_challenge", "winogrande", "hellaswag"])
        self.assertEqual((evaluation["bootstrap_resamples"], self.settings["seed"]), (10000, 21))
        self.assertEqual(self.settings["long_context"]["sequence_length"], quantization1["data"]["sequence_length"])
        self.assertEqual(self.settings["recovery"]["target_tokens"], 100_000_000)

    def test_grid_has_the_decided_forty_eight_stacked_variants(self):
        table = runner.variant_table(self.settings)
        counts = {group: sum(spec["group"] == group for spec in table.values()) for group in runner.GROUPS}
        self.assertEqual(counts, {"a100m": 33, "a1b": 9, "b": 6})
        self.assertEqual(len(table), 48)
        # Quantization of the dense model alone is Quantization-2's baseline (decision log, 2026-10-20).
        self.assertEqual({spec["order"] for spec in table.values()}, {"A", "B"})
        self.assertFalse([name for name in table if name.startswith("a-slice") and "-1b-" in name])
        self.assertEqual({spec["bits"] for spec in table.values() if spec["order"] == "B"}, {4, 3})
        self.assertEqual(table["b-fitted-0.8-100m-int3"]["reference"], "fitted-0.8-100m")

    def test_pins_agree_with_swiglu9(self):
        swiglu9 = read_json(ROOT / "workflows/configs/model/swiglu/swiglu-9.json")["sources"]
        models = self.settings["sources"]["models"]
        self.assertEqual(self.settings["sources"]["swiglu8_prepare_sha256"], swiglu9["swiglu8_prepare_sha256"])
        for key, digest in swiglu9["swiglu8_runs_sha256"].items():
            arm, target = key.split(":")
            self.assertEqual(models[f"{arm}-{target}-100m"]["result_sha256"], digest)
        for target, digest in swiglu9["swiglu7_runs_sha256"].items():
            self.assertEqual(models[f"fitted-{target}-1b"]["result_sha256"], digest)
        self.assertEqual(models["fitted-0.8-1b"]["workflow"], "swiglu-9")

    def test_unpinned_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            changed = deepcopy(self.settings)
            changed["sources"]["models"]["fitted-0.8-1b"]["result_sha256"] = None
            write_json_atomic(path, changed)
            with self.assertRaisesRegex(ValueError, "not pinned consistently"):
                runner.load_settings(path)

    def test_changed_stage_lists_grid_or_shared_blocks_are_rejected(self):
        changes = [
            ("orders", lambda value: value["A"]["stages"].remove("pad"), "stage lists"),
            ("grid", lambda value: value["a1b"]["sources"].append("slice-0.2-1b"), "run grid"),
            ("evaluation", lambda value: value.update(bootstrap_resamples=1000), "evaluation"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for key, change, message in changes:
                changed = deepcopy(self.settings)
                change(changed[key])
                write_json_atomic(path, changed)
                with self.assertRaisesRegex(ValueError, message):
                    runner.load_settings(path)


class PaddingAndBytesContracts(unittest.TestCase):
    def test_projection_paths_name_dense_and_replacement_projections(self):
        model = TinyModel()
        paths = quantization.projection_paths(model, [0, 1])
        self.assertEqual(paths, (
            "layers.0.mlp.gate_proj", "layers.0.mlp.up_proj", "layers.0.mlp.down_proj",
            "layers.1.mlp.gate_projection", "layers.1.mlp.up_projection", "layers.1.mlp.down_projection",
        ))

    def test_padding_appends_zero_neurons_without_changing_the_output(self):
        torch.manual_seed(0)
        model = TinyModel()
        inputs = torch.randn(5, 128)
        before = [layer.mlp(inputs) for layer in model.layers[1:]]
        records = quantization.pad_replacements(model, ["layers.1.mlp", "layers.2.mlp"])
        self.assertEqual(records, [
            {"path": "layers.1.mlp", "real_width": 200, "width": 256},
            {"path": "layers.2.mlp", "real_width": 128, "width": 128},
        ])
        padded = model.layers[1].mlp
        self.assertTrue(padded.gate_projection(inputs)[:, 200:].eq(0).all())
        self.assertTrue(padded.up_projection(inputs)[:, 200:].eq(0).all())
        self.assertTrue(quantization.padded_weights_are_zero(model, records))
        for layer, reference in zip(model.layers[1:], before):
            torch.testing.assert_close(layer.mlp(inputs), reference, rtol=0, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "Only SwiGLU replacement"):
            quantization.pad_gated_replacement(model.layers[0].mlp)

    def test_gradient_mask_keeps_padded_neurons_at_zero(self):
        model = TinyModel()
        records = quantization.pad_replacements(model, ["layers.1.mlp"])
        handles = quantization.mask_padded_gradients(model, records)
        module = model.layers[1].mlp
        loss = sum(projection.weight.sum() for projection in
                   (module.gate_projection, module.up_projection, module.down_projection))
        loss.backward()
        self.assertTrue(module.gate_projection.weight.grad[200:].eq(0).all())
        self.assertTrue(module.down_projection.weight.grad[:, 200:].eq(0).all())
        self.assertTrue(module.gate_projection.weight.grad[:200].eq(1).all())
        for handle in handles:
            handle.remove()

    def test_calculated_bytes_reproduce_quantization1_and_the_design_savings(self):
        """66 projections of SmolLM2-1.7B; measured values are Quantization-1's stored footprints."""

        shapes = {f"{layer}.{name}": shape for layer in range(22)
                  for name, shape in (("gate", (8192, 2048)), ("up", (8192, 2048)), ("down", (2048, 8192)))}
        model = types.SimpleNamespace(get_submodule=lambda path: types.SimpleNamespace(
            weight=types.SimpleNamespace(shape=shapes[path])))
        outside = 1_208_160_256
        dense = 3_422_752_768
        with patch.object(quantization, "scoped_storage_footprint",
                          return_value={"outside_weight_storage_bytes": outside}):
            int8, int4, int3 = (quantization.calculated_weight_bytes(model, list(shapes), bits) for bits in (8, 4, 3))
        self.assertEqual(int8["eligible_weight_bytes"], 1_108_512_768)
        self.assertEqual(int4["eligible_weight_bytes"], 588_251_136)
        self.assertEqual(int4["weight_bytes"], 1_796_411_392)
        self.assertAlmostEqual(1 - int3["weight_bytes"] / dense, 0.516, places=3)
        self.assertEqual(int3["eligible_weight_bytes"], 1_107_296_256 * 13 // 32)

    def test_footprint_uses_calculated_bytes_only_for_int3(self):
        prepared = {"results": {"dense_evaluation": {"footprint": {"parameter_bytes": 1000, "parameters": 500}}}}
        settings = {"model": {"hidden_size": 2, "intermediate_size": 4}, "scope": {"eligible_layers": [1]},
                    "quantization": runner.QUANTIZATION}
        for bits, basis, weight_bytes in ((3, "calculated", 300), (4, "measured", 400)):
            context = types.SimpleNamespace(settings=settings, prepared=prepared,
                                            artifact={"results": {"padding": {"padding_weights": 6}}})
            manifest = {"bits": bits,
                        "footprint": {"logical_parameters": 506, "weight_storage_bytes": 400,
                                      "eligible_weight_storage_bytes": 40},
                        "calculated_bytes": {"weight_bytes": 300, "eligible_weight_bytes": 30}}
            record = runner.footprint_record(context, manifest)
            self.assertEqual((record["frontier_basis"], record["frontier_weight_bytes"]), (basis, weight_bytes))
            self.assertEqual(record["real_logical_parameters"], 500)
            self.assertAlmostEqual(record["whole_model_byte_saving"], 1 - weight_bytes / 1000)

    def test_quantization1_recipe_records_are_unchanged(self):
        common = {"version": 2, "weight_only": True, "set_inductor_config": False}
        self.assertEqual(quantization.recipe_record(8), {**common, "config": "Int8WeightOnlyConfig", "bits": 8,
                                                         "mapping": "symmetric", "granularity": "PerRow"})
        self.assertEqual(quantization.recipe_record(4), {**common, "config": "Int4WeightOnlyConfig", "bits": 4,
                                                         "mapping": "asymmetric", "group_size": 128,
                                                         "input_axis": -1, "int4_packing_format": "PLAIN"})
        self.assertEqual(quantization.recipe_record(3)["config"], "IntxWeightOnlyConfig")


@unittest.skipUnless(TORCHAO_AVAILABLE, "TorchAO is not installed")
class Int3Contracts(unittest.TestCase):
    def padded_bf16_model(self):
        torch.manual_seed(0)
        model = TinyModel()
        quantization.pad_replacements(model, ["layers.1.mlp", "layers.2.mlp"])
        return model.to(torch.bfloat16)

    def test_int3_ptq_keeps_padded_neurons_exactly_zero(self):
        from torchao.quantization import quantize_

        model = self.padded_bf16_model()
        paths = quantization.projection_paths(model, [0, 1, 2])
        quantize_(model, quantization.weight_only_recipe(3), filter_fn=lambda module, name: name in set(paths))
        quantization.verify_quantized_scope(model, paths, 3)
        module = model.layers[1].mlp
        self.assertTrue(module.gate_projection.weight.dequantize()[200:].eq(0).all())
        self.assertTrue(module.down_projection.weight.dequantize()[:, 200:].eq(0).all())
        inputs = torch.randn(3, 128, dtype=torch.bfloat16)
        self.assertTrue(module.gate_projection(inputs)[:, 200:].eq(0).all())

    def test_int3_qat_trains_only_the_selected_scope_and_converts(self):
        torch.manual_seed(0)
        model = TinyModel()
        records = quantization.pad_replacements(model, ["layers.1.mlp"])
        model.to(torch.bfloat16)
        paths = quantization.projection_paths(model, [0, 1, 2])
        trainable = [path for path in paths if path.startswith("layers.1.")]
        parameters = quantization.prepare_qat(model, paths, 3, trainable)
        self.assertEqual(len(parameters), 3)
        self.assertTrue(all(value.dtype == torch.float32 for value in parameters))
        self.assertEqual({name for name, value in model.named_parameters() if value.requires_grad},
                         {f"{path}.weight" for path in trainable})
        handles = quantization.mask_padded_gradients(model, records)
        optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=0.0)
        for unused_step in range(3):
            optimizer.zero_grad()
            model.layers[1].mlp(torch.randn(4, 128, dtype=torch.bfloat16).float()).pow(2).mean().backward()
            optimizer.step()
        self.assertTrue(quantization.padded_weights_are_zero(model, records))
        for handle in handles:
            handle.remove()
        quantization.convert_qat(model, paths, 3)
        self.assertTrue(model.layers[1].mlp.gate_projection.weight.dequantize()[200:].eq(0).all())

    def test_asymmetric_int3_cannot_use_the_inferred_qat_path(self):
        from torchao.quantization import quantize_
        from torchao.quantization.qat import QATConfig

        model = TinyModel().to(torch.bfloat16)
        with self.assertRaisesRegex(AssertionError, "symmetric"):
            quantize_(model, QATConfig(quantization.weight_only_recipe(3), step="prepare"))

    def test_packed_bundle_round_trip_restores_padded_replacement_topology(self):
        from torchao.quantization import quantize_
        from transformers import LlamaConfig, LlamaForCausalLM
        from mlp_replacement.compression.surgery import replace_submodule
        from mlp_replacement.evaluation import quantized_bundles

        torch.manual_seed(0)
        config = LlamaConfig(vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=24,
                             num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=True,
                             max_position_embeddings=64)
        model = LlamaForCausalLM(config).eval()
        # Parameters only: like a BF16 checkpoint, the rotary buffers stay FP32.
        for parameter in model.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)
        replace_submodule(model, "model.layers.5.mlp", GatedMLPReplacement(128, 200).to(torch.bfloat16))
        records = quantization.pad_replacements(model, ["model.layers.5.mlp"])
        paths = quantization.projection_paths(model, range(1, 23))
        quantize_(model, quantization.weight_only_recipe(3), filter_fn=lambda module, name: name in set(paths))
        inputs = torch.randint(0, 64, (1, 16))
        with torch.no_grad():
            expected = model(input_ids=inputs).logits

        class Tokenizer:
            def save_pretrained(self, path):
                Path(path).mkdir(parents=True)
                (Path(path) / "tokenizer.json").write_text("{}")

        replacements = [{**row, "down_bias": False} for row in records]
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(quantized_bundles, "require_backend"), \
                patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()):
            directory = Path(temporary) / "model"
            quantized_bundles.export_quantized_bundle(
                model, Tokenizer(), directory, paths, 3, {}, {"scope": {"eligible_layers": list(range(1, 23))}},
                replacements=replacements, calculated_bytes={"bits": 3})
            manifest = json.loads((directory / "bundle.json").read_text())
            self.assertEqual(manifest["replacements"], replacements)
            reloaded, unused_tokenizer, unused_manifest = quantized_bundles.load_quantized_bundle(directory, "cpu")
        self.assertIsInstance(reloaded.model.layers[5].mlp, GatedMLPReplacement)
        self.assertEqual(reloaded.model.layers[5].mlp.gate_projection.out_features, 256)
        with torch.no_grad():
            torch.testing.assert_close(reloaded(input_ids=inputs).logits, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
