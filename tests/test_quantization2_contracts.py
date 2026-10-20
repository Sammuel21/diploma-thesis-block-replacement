"""Quantization-2 contracts: configuration, variant scope, stream pins, and the shared report."""

import csv
import importlib.util
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mlp_replacement.artifacts import content_digest, file_digest, read_json, write_json_atomic
from workflows.runs.model.baseline import quantization as runner

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
CONFIG = ROOT / "workflows/configs/model/baseline/quantization-2.json"
SWIGLU8_PREPARED = ROOT / "data/results/workflows/model/swiglu-8/prepare-001/result.json"


class ConfigurationContracts(unittest.TestCase):
    def setUp(self):
        self.settings = runner.load_settings(CONFIG)
        self.quantization1 = runner.load_settings(runner.DEFAULT_CONFIG)

    def test_reuses_quantization1_blocks_and_preparation(self):
        self.assertEqual(self.settings["workflow"], "quantization-2")
        for key in runner.SHARED_BLOCKS:
            self.assertEqual(self.settings[key], self.quantization1[key])
        # Equal preparation fingerprints let Quantization-2 load Quantization-1's prepare-001.
        self.assertEqual(content_digest(runner.preparation_settings(self.settings)),
                         content_digest(runner.preparation_settings(self.quantization1)))
        self.assertEqual(self.settings["variants"], ["ptq-int3"])
        self.assertEqual(self.settings["runtime"], {"attention_implementation": "sdpa"})
        self.assertEqual(self.settings["references"], [])
        self.assertTrue(self.settings["outputs"]["root"].endswith("/quantization-2"))

    def test_variants_are_scoped_per_workflow(self):
        self.assertEqual(runner.WORKFLOW_VARIANTS["quantization-1"],
                         ("dense-bf16", "ptq-int8", "ptq-int4", "qat-int4"))
        self.assertEqual(runner.WORKFLOW_VARIANTS["quantization-2"], ("ptq-int3", "qat-int3"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            changed = deepcopy(self.quantization1)
            changed["variants"] = ["ptq-int3"]
            write_json_atomic(path, changed)
            with self.assertRaisesRegex(ValueError, "Unknown default"):
                runner.load_settings(path)
            changed = deepcopy(self.settings)
            changed["variants"] = ["ptq-int3", "qat-int3"]
            write_json_atomic(path, changed)
            with self.assertRaisesRegex(ValueError, "QAT is optional"):
                runner.load_settings(path)

    def test_changed_shared_block_speed_or_missing_stream_pin_is_rejected(self):
        changes = [
            (lambda value: value["qat"].update(learning_rate=1e-4), "qat block"),
            (lambda value: value["evaluation"].update(bootstrap_resamples=1000), "evaluation block"),
            (lambda value: value["recovery_stream_sha256"].pop("1000000000"), "pins the 100M and 1B"),
            (lambda value: value["runtime"].update(workloads=[]), "no speed"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for change, message in changes:
                changed = deepcopy(self.settings)
                change(changed)
                write_json_atomic(path, changed)
                with self.assertRaisesRegex(ValueError, message):
                    runner.load_settings(path)

    def test_effective_configuration_with_a_qat_budget_reloads(self):
        # The fresh-process memory worker reloads the run's effective configuration.
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "effective-config.json"
            write_json_atomic(path, runner.load_settings(CONFIG, 1_000_000_000))
            self.assertEqual(runner.load_settings(path)["qat"]["target_tokens"], 1_000_000_000)

    def test_stream_pins_are_the_streams_quantization1_trained_on(self):
        pins = self.settings["recovery_stream_sha256"]
        # Quantization-1 recorded its 1B stream as identical to SwiGLU-7's 1B stream.
        page = (ROOT / "docs/experiments/model/baseline/quantization-1.md").read_text(encoding="utf-8")
        self.assertIn(pins["1000000000"], page)
        if SWIGLU8_PREPARED.is_file():
            stream = read_json(SWIGLU8_PREPARED)["results"]["stream"]
            self.assertEqual((stream["sha256"], stream["token_count"]), (pins["100000000"], 100_000_000))

    def test_quantization2_refuses_its_own_preparation_and_foreign_variants(self):
        with tempfile.TemporaryDirectory() as temporary:
            for arguments, message in (
                (["prepare"], "reuses Quantization-1's prepare-001"),
                (["run", "--prepared", "unused.json", "--variant", "ptq-int4"], "does not define variant"),
            ):
                argv = ["quantization", *arguments, "--config", str(CONFIG), "--output-dir", temporary]
                with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, message):
                    runner.main()


class ReportContracts(unittest.TestCase):
    def test_report_places_int3_at_calculated_bytes_without_speed(self):
        settings = runner.load_settings(CONFIG)
        quality =importlib.util.module_from_spec(importlib.util.spec_from_file_location(
            "quality_contract_impl", ROOT / "src/mlp_replacement/evaluation/final_quality.py"))
        quality.__spec__.loader.exec_module(quality)
        dependencies = nullcontext() if TORCH_AVAILABLE else patch.dict(sys.modules, {
            "mlp_replacement.evaluation": types.ModuleType("mlp_replacement.evaluation"),
            "mlp_replacement.evaluation.final_quality": quality,
        })
        with tempfile.TemporaryDirectory() as temporary, dependencies:
            root = Path(temporary)
            paths = []
            rows = (("quantization-1", "dense-bf16", 0, 1.0), ("quantization-2", "qat-int3", 1_000_000_000, 0.5))
            for workflow, variant, tokens, accuracy in rows:
                output = root / variant
                output.mkdir()
                tasks = {}
                for task in settings["evaluation"]["tasks"]:
                    metric = settings["evaluation"]["primary_metrics"][task]
                    samples = [{"doc_id": i, "doc_hash": str(i), "prompt_hash": str(i), "target_hash": str(i),
                                metric: float(i == 0 or accuracy == 1.0)} for i in range(2)]
                    task_path = output / f"{task}.json"
                    write_json_atomic(task_path, {"results": {task: {f"{metric},none": accuracy}},
                                                  "samples": {task: samples}})
                    tasks[task] = {"path": task_path.name, "sha256": file_digest(task_path)}
                dense = variant == "dense-bf16"
                eligible = 200 if dense else 103
                result = {
                    "footprint": {"logical_parameters": 100, "eligible_parameters": 50,
                                  "weight_storage_bytes": eligible + 100, "eligible_weight_storage_bytes": eligible,
                                  "tensor_file_bytes": eligible + 120, "bundle_bytes": eligible + 140},
                    "training_tokens": tokens, "calibration_tokens": 0,
                    "recovery": {"elapsed_seconds": 10.0},
                    "evaluation": {"status": "completed", "protocol_fingerprint": "shared", "kl": {"kl": 0.1},
                                   "tasks": tasks,
                                   "likelihood": {f"{split}-{context}": {"perplexity": 10, "predicted_tokens": 100}
                                                  for split in ("validation", "test") for context in (128, 2048, 8192)}},
                    "runtime": ({"execution_fingerprint": "dense-gpu", "generation": {"workloads": []},
                                 "resident_memory": {"gpu_allocated_delta_bytes": 300, "host_rss_delta_bytes": 1}}
                                if dense else
                                {"speed": None, "resident_memory": {"gpu_allocated_delta_bytes": 203,
                                                                    "host_rss_delta_bytes": 1}}),
                }
                if not dense:
                    # 3.25 bits per weight against the 8.25-bit container: 103 measured, 41 calculated.
                    result["calculated_footprint"] = {"bits": 3, "eligible_weight_bytes": 41,
                                                      "outside_weight_bytes": 100, "weight_bytes": 141}
                write_json_atomic(output / "result.json", {"workflow": workflow, "status": "completed",
                                                           "identity": {"variant": variant}, "results": result})
                write_json_atomic(output / "run.json", {"workflow_gpu_hours_proxy": 0.1})
                paths.append(output / "result.json")
            report_dir = root / "report"
            report_dir.mkdir()
            log = types.SimpleNamespace(begin=lambda stage: None)
            runner.report(report_dir, {"results": {}}, log, settings, paths)
            dense_row, int3 = read_json(report_dir / "comparison.json")["rows"]
            self.assertEqual((dense_row["frontier_basis"], dense_row["whole_model_frontier_byte_savings"]),
                             ("measured", 0.0))
            self.assertEqual(int3["variant"], "qat-int3-1000M")
            self.assertEqual((int3["workflow"], int3["frontier_basis"], int3["frontier_weight_bytes"]),
                             ("quantization-2", "calculated", 141))
            self.assertAlmostEqual(int3["whole_model_frontier_byte_savings"], 1 - 141 / 300)
            self.assertAlmostEqual(int3["whole_model_weight_byte_savings"], 1 - 203 / 300)
            self.assertFalse(int3["runtime_comparable_to_dense"])
            self.assertEqual(int3["tasks"]["piqa"]["student_minus_dense"], -0.5)
            with (report_dir / "comparison.csv").open(encoding="utf-8", newline="") as stream:
                headings = next(csv.reader(stream))
            self.assertIn("whole_model_frontier_byte_savings", headings)
            self.assertFalse([name for name in headings if "tokens_per_second" in name])


if __name__ == "__main__":
    unittest.main()
