"""Speed-1 contracts: configuration, measurement order, input pinning, and summary; no GPU."""

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mlp_replacement.artifacts import file_digest
from workflows.runs.model.baseline import speed as runner

JOB = ROOT / "workflows/jobs/perun/speed_1.sbatch"


def write_config(directory, settings):
    path = Path(directory) / "speed.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    return path


def runtime_record(label, ttft, tokens_per_second, fingerprint="same", device="GPU-1"):
    workloads = []
    for batch_size, prompt in ((1, 2048), (4, 2048), (1, 7936)):
        median = {"time_to_first_token_ms": ttft * prompt / 2048,
                  "output_tokens_per_second": tokens_per_second * batch_size,
                  "decode_tokens_per_second": tokens_per_second * batch_size * 1.01,
                  "peak_gpu_allocated_bytes": 4 * 2**30 * batch_size}
        workloads.append({"batch_size": batch_size, "prompt_tokens": prompt,
                          "generated_tokens": 256, "median": median})
    return {"bundle_manifest_sha256": label, "execution_fingerprint": fingerprint,
            "execution": {"device_uuid": device},
            "resident_memory": {"gpu_allocated_delta_bytes": 3 * 2**30},
            "generation": {"workloads": workloads}}


class SpeedConfiguration(unittest.TestCase):
    def setUp(self):
        self.settings = runner.load_settings(runner.DEFAULT_CONFIG)

    def test_decided_models_and_passes(self):
        labels = [model["label"] for model in self.settings["models"]]
        self.assertEqual(labels[:4], ["dense-bf16", "ptq-int8", "ptq-int4", "qat-int4-1b"])
        self.assertEqual(labels[4:], [f"swiglu-8-fitted-{target}" for target in
                                      ("0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.75", "0.8")])
        self.assertEqual(self.settings["passes"], ["forward", "reverse"])
        self.assertEqual(self.settings["reference_label"], "dense-bf16")
        self.assertEqual(self.settings["measurement"]["protocol_config"],
                         "workflows/configs/model/baseline/quantization-1.json")

    def test_reverse_pass_measures_every_model_in_reverse_order(self):
        plan = runner.measurement_plan(self.settings)
        labels = [model["label"] for model in self.settings["models"]]
        self.assertEqual(len(plan), 2 * len(labels))
        self.assertEqual([label for name, label in plan[:len(labels)]], labels)
        self.assertEqual([label for name, label in plan[len(labels):]], labels[::-1])
        self.assertEqual({name for name, label in plan[len(labels):]}, {"reverse"})

    def test_rejects_unsafe_or_ambiguous_configuration(self):
        cases = []
        duplicate = deepcopy(self.settings)
        duplicate["models"][1]["label"] = "dense-bf16"
        cases.append(duplicate)
        escaping = deepcopy(self.settings)
        escaping["models"][0]["bundle"] = "../elsewhere/model"
        cases.append(escaping)
        absolute = deepcopy(self.settings)
        absolute["models"][0]["bundle"] = str(ROOT / "model")
        cases.append(absolute)
        unpinned = deepcopy(self.settings)
        unpinned["models"][0]["bundle_manifest_sha256"] = "unknown"
        cases.append(unpinned)
        missing_reference = deepcopy(self.settings)
        missing_reference["reference_label"] = "teacher"
        cases.append(missing_reference)
        with tempfile.TemporaryDirectory() as directory:
            for settings in cases:
                with self.subTest(settings=settings["models"][0]), self.assertRaises(ValueError):
                    runner.load_settings(write_config(directory, settings))

    def test_inputs_are_checked_against_pinned_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = deepcopy(self.settings)
            settings["models"] = settings["models"][:2]
            prepared = root / settings["measurement"]["prepared"]
            prepared.parent.mkdir(parents=True)
            prepared.write_text("{}", encoding="utf-8")
            settings["measurement"]["prepared_sha256"] = file_digest(prepared)
            for model in settings["models"]:
                manifest = root / model["bundle"] / "bundle.json"
                manifest.parent.mkdir(parents=True)
                manifest.write_text(json.dumps({"label": model["label"]}), encoding="utf-8")
                model["bundle_manifest_sha256"] = file_digest(manifest)
            resolved, bundles = runner.verify_inputs(settings, root)
            self.assertEqual(resolved, prepared.resolve())
            self.assertEqual(set(bundles), {"dense-bf16", "ptq-int8"})
            (root / settings["models"][1]["bundle"] / "bundle.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.verify_inputs(settings, root)


class SpeedSummary(unittest.TestCase):
    def setUp(self):
        settings = runner.load_settings(runner.DEFAULT_CONFIG)
        settings["models"] = settings["models"][:2]
        self.settings = settings

    def test_means_pass_medians_and_reports_spread_and_ratio(self):
        measurements = {
            ("forward", "dense-bf16"): runtime_record("dense", 20.0, 160.0),
            ("reverse", "dense-bf16"): runtime_record("dense", 20.0, 156.0),
            ("forward", "ptq-int8"): runtime_record("int8", 25.0, 120.0),
            ("reverse", "ptq-int8"): runtime_record("int8", 25.0, 120.0),
        }
        summary = runner.summarize(self.settings, measurements)
        dense = summary["models"]["dense-bf16"]["workloads"]["b1-p2048-g256"]
        self.assertAlmostEqual(dense["output_tokens_per_second"]["value"], 158.0)
        self.assertAlmostEqual(dense["output_tokens_per_second"]["pass_spread"], 4.0 / 158.0)
        int8 = summary["models"]["ptq-int8"]["workloads"]["b1-p2048-g256"]
        self.assertAlmostEqual(int8["output_tokens_per_second"]["relative_to_reference"], 120.0 / 158.0)
        self.assertAlmostEqual(int8["time_to_first_token_ms"]["relative_to_reference"], 1.25)
        self.assertTrue(summary["single_execution_fingerprint"])
        self.assertAlmostEqual(summary["max_timing_pass_spread"], 4.0 / 158.0)

    def test_flags_measurements_from_different_executions(self):
        measurements = {
            ("forward", "dense-bf16"): runtime_record("dense", 20.0, 160.0),
            ("reverse", "dense-bf16"): runtime_record("dense", 20.0, 160.0),
            ("forward", "ptq-int8"): runtime_record("int8", 25.0, 120.0, "other", "GPU-2"),
            ("reverse", "ptq-int8"): runtime_record("int8", 25.0, 120.0),
        }
        summary = runner.summarize(self.settings, measurements)
        self.assertFalse(summary["single_execution_fingerprint"])
        self.assertEqual(summary["device_uuids"], ["GPU-1", "GPU-2"])


class SpeedJob(unittest.TestCase):
    def test_job_is_one_gpu_without_private_values(self):
        source = JOB.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:1", source)
        self.assertIn("-m workflows.runs.model.baseline.speed", source)
        self.assertIn("Speed-1 takes no array", source)
        self.assertNotIn("--account", source)
        self.assertNotIn("--qos", source)


if __name__ == "__main__":
    unittest.main()
