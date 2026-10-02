"""Focused artifact/scope contracts; no model download or scientific job."""

import importlib.util
import csv
import json
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from contextlib import nullcontext

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mlp_replacement.artifacts import file_digest, read_json, write_json_atomic
from workflows.runs.model.baseline import quantization as runner

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TORCHAO_AVAILABLE = importlib.util.find_spec("torchao") is not None


def load_contract_module(name, relative_path):
    """Inspect dependency-free helpers without the package's eager Torch exports."""

    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if TORCH_AVAILABLE:
    from mlp_replacement.compression import quantization
    from mlp_replacement.compression.continuation import segment_schedule
else:
    quantization = load_contract_module("quantization_contract_impl", "src/mlp_replacement/compression/quantization.py")
    segment_schedule = load_contract_module("continuation_contract_impl", "src/mlp_replacement/compression/continuation.py").segment_schedule


class QuantizationArtifactContracts(unittest.TestCase):
    def setUp(self):
        self.settings = runner.load_settings(runner.DEFAULT_CONFIG)

    def test_frozen_reference_and_optional_qat(self):
        reference = read_json(ROOT / "workflows/configs/model/swiglu/swiglu-7.json")
        self.assertEqual(self.settings["model"], reference["model"])
        self.assertEqual(self.settings["evaluation"], reference["evaluation"])
        self.assertEqual(self.settings["scope"]["eligible_layers"], list(range(1, 23)))
        self.assertEqual(self.settings["scope"]["expected_projections"], 66)
        self.assertEqual(self.settings["variants"], ["dense-bf16", "ptq-int8", "ptq-int4"])
        self.assertIsNone(self.settings["qat"]["target_tokens"])
        self.assertEqual(self.settings["data"]["recovery_source"]["first_shard"], 1)
        self.assertIn("00000", self.settings["data"]["kl_source"]["data_file"])
        self.assertEqual(self.settings["data"]["kl_sequences"] * self.settings["data"]["sequence_length"], 196608)

    def test_invalid_qat_budgets_and_scope_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for invalid in (0, -1, True, 1.5):
                write_json_atomic(path, self.settings)
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    runner.load_settings(path, invalid)
            changed = deepcopy(self.settings)
            changed["scope"]["eligible_layers"] = list(range(23))
            write_json_atomic(path, changed)
            with self.assertRaisesRegex(ValueError, "1–22"):
                runner.load_settings(path)

    def test_packed_storage_deduplicates_aliases_and_includes_metadata(self):
        def leaf(pointer, byte_count):
            storage = types.SimpleNamespace(data_ptr=lambda: pointer, nbytes=lambda: byte_count)
            return types.SimpleNamespace(device="cpu", untyped_storage=lambda: storage)

        values = leaf(100, 256)
        scales = leaf(200, 32)
        offsets = leaf(300, 32)
        packed = types.SimpleNamespace(values=values, scales=scales, offsets=offsets,
                                       __tensor_flatten__=lambda: (["values", "scales", "offsets"], None))
        inventory = quantization.storage_inventory([packed, values, packed])
        self.assertEqual(sum(inventory.values()), 320)
        self.assertEqual(len(inventory), 3)

    def test_recovery_schedule_preserves_exact_endpoint(self):
        endpoint = 1_000_000_003
        schedule = segment_schedule(0, endpoint, [10_000_000, 25_000_000, endpoint], 8192)
        self.assertEqual(schedule[-1], (endpoint, (endpoint,)))
        self.assertTrue(all(actual % 8192 == 0 for actual, requests in schedule[:-1]))
        self.assertEqual(schedule[0][1], (10_000_000,))

    def test_prepared_token_extent_and_digest_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "tokens.int32"
            np.asarray([1, 2, 3], dtype=np.int32).tofile(path)
            record = {"path": path.name, "token_count": 3, "sha256": file_digest(path)}
            values = runner.verified_tokens(root, record)
            self.assertEqual(values.tolist(), [1, 2, 3])
            del values
            np.asarray([1, 2, 4], dtype=np.int32).tofile(path)
            with self.assertRaisesRegex(ValueError, "changed"):
                runner.verified_tokens(root, record)
            path.write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "changed"):
                runner.verified_tokens(root, record)

    def test_structural_reference_requires_matching_provenance_and_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "model"
            bundle.mkdir()
            manifest = {"provenance": {"workflow": "swiglu-7", "run_fingerprint": "original"},
                        "replacements": [{"path": "model.layers.1.mlp"}]}
            original = {"status": "completed", "run_fingerprint": "original",
                        "identity": {"target": 0.5},
                        "configuration": {"model": self.settings["model"], "allocation": self.settings["scope"]},
                        "results": {"recovery": {"tokens_seen": 1_000_000_000, "training_seconds": 500}}}
            write_json_atomic(bundle / "bundle.json", manifest)
            write_json_atomic(root / "result.json", original)
            result = runner.validate_reference(bundle, root / "result.json", self.settings)
            self.assertEqual(result["training_tokens"], 1_000_000_000)
            self.assertEqual(result["target"], 0.5)
            self.assertEqual(result["recovery_cost"]["training_seconds"], 500)
            original["run_fingerprint"] = "foreign"
            write_json_atomic(root / "result.json", original)
            with self.assertRaisesRegex(ValueError, "completed bundle"):
                runner.validate_reference(bundle, root / "result.json", self.settings)
            original["run_fingerprint"] = "original"
            original["configuration"]["model"] = {**self.settings["model"], "revision": "foreign"}
            write_json_atomic(root / "result.json", original)
            with self.assertRaisesRegex(ValueError, "revision"):
                runner.validate_reference(bundle, root / "result.json", self.settings)
            original["configuration"]["model"] = self.settings["model"]
            write_json_atomic(root / "result.json", original)
            manifest["replacements"] = [{"path": "model.layers.0.mlp"}]
            write_json_atomic(bundle / "bundle.json", manifest)
            with self.assertRaisesRegex(ValueError, "protected"):
                runner.validate_reference(bundle, root / "result.json", self.settings)

    def test_resume_rejects_changed_identity_and_configuration_without_overwriting(self):
        class ContractLog:
            def __init__(self, path, config):
                self.path = path
                self.data = {"status": "running"}

            def begin(self, stage):
                write_json_atomic(self.path, self.data)

        log_module = types.ModuleType("mlp_replacement.runlog")
        log_module.ExperimentLog = ContractLog
        log_module.environment_record = lambda: {"contract_fixture": True, "packages": {}}
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {"mlp_replacement.runlog": log_module}), \
             patch.object(runner, "source_hashes", return_value={"source": "fixed"}):
            output = Path(temporary) / "run"
            runner.start_artifact(output, self.settings, "run", {"variant": "ptq-int4"}, False)
            before = (output / "result.json").read_bytes()
            changed = deepcopy(self.settings)
            changed["qat"]["target_tokens"] = 1_000_000_000
            for configuration, identity in ((changed, {"variant": "ptq-int4"}),
                                            (self.settings, {"variant": "ptq-int8"})):
                with self.assertRaisesRegex(ValueError, "fingerprint changed"):
                    runner.start_artifact(output, configuration, "run", identity, True)
                self.assertEqual((output / "result.json").read_bytes(), before)
            with self.assertRaises(FileExistsError):
                runner.start_artifact(output, self.settings, "run", {"variant": "ptq-int4"}, False)
            artifact, log = runner.start_artifact(output, self.settings, "run", {"variant": "ptq-int4"}, True)
            self.assertEqual(artifact["status"], "running")
            self.assertEqual(len(log.data["attempts"]), 1)

    def test_report_keeps_full_quality_speed_bytes_and_pairing(self):
        quality = load_contract_module("quality_contract_impl", "src/mlp_replacement/evaluation/final_quality.py")
        dependencies = nullcontext() if TORCH_AVAILABLE else patch.dict(sys.modules, {
            "mlp_replacement.evaluation": types.ModuleType("mlp_replacement.evaluation"),
            "mlp_replacement.evaluation.final_quality": quality,
        })
        with tempfile.TemporaryDirectory() as temporary, dependencies:
            root = Path(temporary)
            paths = []
            for variant, eligible_bytes, accuracy in (("dense-bf16", 100, 1.0), ("ptq-int8", 54, 0.5)):
                output = root / variant
                output.mkdir()
                tasks = {}
                for task in self.settings["evaluation"]["tasks"]:
                    metric = self.settings["evaluation"]["primary_metrics"][task]
                    samples = [{"doc_id": i, "doc_hash": str(i), "prompt_hash": str(i),
                                "target_hash": str(i), metric: float(i == 0 or accuracy == 1.0)} for i in range(2)]
                    task_path = output / f"{task}.json"
                    write_json_atomic(task_path, {"results": {task: {f"{metric},none": accuracy}},
                                                  "samples": {task: samples}})
                    tasks[task] = {"path": task_path.name, "sha256": file_digest(task_path)}
                workloads = [{**workload, "peak_gpu_allocated_bytes": 1000, "median": {"time_to_first_token_ms": 10,
                               "mean_subsequent_token_latency_ms": 2, "generation_latency_ms": 520,
                               "output_tokens_per_second": 492, "peak_gpu_allocated_bytes": 1000}}
                             for workload in self.settings["runtime"]["workloads"]]
                result = {"footprint": {"logical_parameters": 100, "eligible_parameters": 50,
                           "weight_storage_bytes": eligible_bytes + 100, "eligible_weight_storage_bytes": eligible_bytes,
                           "tensor_file_bytes": eligible_bytes + 120, "bundle_bytes": eligible_bytes + 140},
                          "training_tokens": 0, "calibration_tokens": 0,
                          "evaluation": {"status": "completed", "protocol_fingerprint": "shared",
                              "kl": {"kl": 0.1}, "tasks": tasks,
                              "likelihood": {f"{split}-{context}": {"perplexity": 10, "predicted_tokens": 100}
                                             for split in ("validation", "test") for context in (128, 2048, 8192)}},
                          "runtime": {"execution_fingerprint": "same-gpu", "generation": {"workloads": workloads},
                                      "resident_memory": {"gpu_allocated_delta_bytes": 1000, "host_rss_delta_bytes": 2000}}}
                write_json_atomic(output / "result.json", {"workflow": "quantization-1", "status": "completed",
                                                           "identity": {"variant": variant}, "results": result})
                write_json_atomic(output / "run.json", {"workflow_gpu_hours_proxy": 0.1})
                paths.append(output / "result.json")
            report_dir = root / "report"
            report_dir.mkdir()
            artifact = {"results": {}}
            log = types.SimpleNamespace(begin=lambda stage: None)
            runner.report(report_dir, artifact, log, self.settings, paths)
            rows = read_json(report_dir / "comparison.json")["rows"]
            int8 = rows[1]
            self.assertAlmostEqual(int8["eligible_weight_byte_savings"], 0.46)
            self.assertEqual(int8["eligible_parameter_removal"], 0)
            self.assertEqual(len(int8["likelihood"]), 6)
            self.assertEqual(int8["tasks"]["piqa"]["student_minus_dense"], -0.5)
            self.assertEqual(int8["recovery_seconds"], 0.0)
            self.assertEqual(len(rows), 5)  # Three structural budgets are pending.
            self.assertIn("b1_p7936_output_tokens_per_second", (report_dir / "comparison.csv").read_text())
            with (report_dir / "comparison.csv").open(encoding="utf-8", newline="") as stream:
                headings = next(csv.reader(stream))
            self.assertEqual(len(headings), len(set(headings)))
            changed = read_json(paths[1])
            changed["results"]["evaluation"]["protocol_fingerprint"] = "historical-128"
            write_json_atomic(paths[1], changed)
            with self.assertRaisesRegex(ValueError, "protocols differ"):
                runner.report(report_dir, artifact, log, self.settings, paths)
            changed["results"]["evaluation"]["protocol_fingerprint"] = "shared"
            write_json_atomic(paths[1], changed)
            sample_path = paths[1].parent / "piqa.json"
            samples = read_json(sample_path)
            samples["samples"]["piqa"][0]["prompt_hash"] = "changed"
            write_json_atomic(sample_path, samples)
            changed["results"]["evaluation"]["tasks"]["piqa"]["sha256"] = file_digest(sample_path)
            write_json_atomic(paths[1], changed)
            with self.assertRaisesRegex(ValueError, "prompt_hash differs"):
                runner.report(report_dir, artifact, log, self.settings, paths)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in the local inspection environment")
class QuantizationTensorContracts(unittest.TestCase):
    def test_frozen_weights_and_shared_storage(self):
        import torch
        import torch.nn as nn

        model = nn.Module()
        model.eligible = nn.Linear(128, 128, bias=False).to(torch.bfloat16)
        model.protected = nn.Linear(128, 128, bias=False).to(torch.bfloat16)
        frozen = quantization.frozen_state_hashes(model, ("eligible",))
        before = quantization.quantization_footprint(model, ("eligible",))
        with torch.no_grad():
            model.eligible.weight.add_(1)
        quantization.verify_frozen_state(model, ("eligible",), frozen)
        self.assertEqual(before, quantization.quantization_footprint(model, ("eligible",)))
        with torch.no_grad():
            model.protected.weight.add_(1)
        with self.assertRaisesRegex(ValueError, "outside"):
            quantization.verify_frozen_state(model, ("eligible",), frozen)
        weight = model.eligible.weight
        self.assertEqual(sum(quantization.storage_inventory([weight, weight[:4]]).values()), weight.numel() * 2)

    @unittest.skipUnless(TORCHAO_AVAILABLE, "Pinned TorchAO is unavailable")
    def test_native_qat_gradient_scope_and_checkpoint_restoration(self):
        import torch
        import torch.nn as nn
        from mlp_replacement.compression.continuation import capture_rng, commit_single_checkpoint, restore_checkpoint, restore_rng

        def make_model():
            model = nn.Module()
            model.eligible = nn.Linear(128, 128, bias=False).to(torch.bfloat16)
            model.protected = nn.Linear(128, 128, bias=False).to(torch.bfloat16)
            return model

        model = make_model()
        parameters = quantization.prepare_int4_qat(model, ("eligible",))
        self.assertEqual([name for name, value in model.named_parameters() if value.requires_grad], ["eligible.weight"])
        self.assertEqual(parameters[0].dtype, torch.float32)
        self.assertIsNone(model.eligible.activation_fake_quantizer)
        self.assertEqual(model.eligible.weight_fake_quantizer.config.group_size, 128)
        optimizer = torch.optim.AdamW(parameters, lr=3e-5)
        parameters[0].grad = torch.ones_like(parameters[0])
        optimizer.step()
        with tempfile.TemporaryDirectory() as temporary:
            payload = {"run_fingerprint": "contract", "tokens_seen": 8192,
                       "eligible_state": model.eligible.state_dict(), "optimizer": optimizer.state_dict(), **capture_rng()}
            commit_single_checkpoint(temporary, payload)
            expected = torch.rand(5)
            restored, descriptor = restore_checkpoint(temporary, "contract")
            target = make_model()
            restored_parameters = quantization.prepare_int4_qat(target, ("eligible",))
            target.eligible.load_state_dict(restored["eligible_state"], strict=True)
            restored_optimizer = torch.optim.AdamW(restored_parameters, lr=3e-5)
            restored_optimizer.load_state_dict(restored["optimizer"])
            restore_rng(restored)
            self.assertTrue(torch.equal(torch.rand(5), expected))
            self.assertTrue(torch.equal(model.eligible.weight, target.eligible.weight))
            self.assertTrue(torch.equal(optimizer.state[parameters[0]]["exp_avg"], restored_optimizer.state[restored_parameters[0]]["exp_avg"]))
            with self.assertRaisesRegex(ValueError, "different experiment"):
                restore_checkpoint(temporary, "foreign")


if __name__ == "__main__":
    unittest.main()
