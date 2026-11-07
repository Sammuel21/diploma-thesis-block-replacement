"""Focused CPU/static contracts for Heterogeneous-2; no pretrained model or dataset is loaded."""

import copy
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from mlp_replacement.artifacts import read_json, write_json_atomic
from mlp_replacement.compression.teacher_cache import cache_teacher_logits
from mlp_replacement.data import PackedTokenCache
from mlp_replacement.model import discover_mlp_blocks
from mlp_replacement.operators import DenseHybridReplacement, GatedMLPReplacement
from workflows.runs.model.heterogenous import heterogeneous_1 as h1
from workflows.runs.model.heterogenous import heterogeneous_2 as h
from workflows.runs.model.swiglu.swiglu_7 import desired_requests
from workflows.runs.model.swiglu.swiglu_8 import AcceptanceFailure

from mlp_replacement.compression.continuation import segment_schedule

CONFIG = ROOT / "workflows/configs/model/heterogenous/heterogeneous-2.json"
SBATCH = ROOT / "workflows/jobs/perun/heterogenous/heterogeneous_2.sbatch"
READING = {"bar": 0.035, "decision_tokens": 100, "shrink_from_tokens": 50, "high_learning_rate": 0.0001}


def record(at_50, at_100, kl=None):
    return {"curve": {50: at_50, 100: at_100}, "kl_8192": at_100 if kl is None else kl}


class StepTwoRule(unittest.TestCase):
    def test_the_bar_itself_counts_as_level(self):
        self.assertEqual(h.band(0.035, 0.035), "level")
        self.assertEqual(h.band(-0.035, 0.035), "level")
        self.assertEqual(h.band(0.0351, 0.035), "worse")
        self.assertEqual(h.band(-0.0351, 0.035), "better")

    def test_better_and_level_go_ahead_however_the_gap_moved(self):
        swiglu = record(1.0, 1.0)
        self.assertTrue(h.step_two_reading(record(1.0, 0.90), swiglu, READING)["go_to_1b"])
        level = h.step_two_reading(record(0.90, 1.02), swiglu, READING)
        self.assertEqual(level["band"], "level")
        self.assertTrue(level["go_to_1b"])

    def test_worse_goes_ahead_only_while_clearly_catching_up(self):
        swiglu = record(1.0, 1.0)
        catching_up = h.step_two_reading(record(1.20, 1.10), swiglu, READING)
        self.assertAlmostEqual(catching_up["shrink"], 0.10)
        self.assertTrue(catching_up["go_to_1b"])
        self.assertFalse(h.step_two_reading(record(1.12, 1.10), swiglu, READING)["go_to_1b"])
        self.assertFalse(h.step_two_reading(record(1.05, 1.10), swiglu, READING)["go_to_1b"])

    def test_the_gap_is_positive_when_forced_is_worse(self):
        self.assertAlmostEqual(h.gap(1.10, 1.00), 0.10)
        self.assertAlmostEqual(h.gap(0.90, 1.00), -0.10)

    def test_the_8k_cross_check_flags_another_band_and_leaves_the_decision_alone(self):
        swiglu = record(1.0, 1.0, kl=1.0)
        flagged = h.step_two_reading(record(1.0, 1.02, kl=1.10), swiglu, READING)
        self.assertEqual((flagged["band"], flagged["band_8k"]), ("level", "worse"))
        self.assertTrue(flagged["flag_8k_disagrees"])
        self.assertTrue(flagged["go_to_1b"])
        self.assertFalse(h.step_two_reading(record(1.0, 1.02, kl=1.03), swiglu, READING)["flag_8k_disagrees"])


class LearningRateReading(unittest.TestCase):
    def test_the_design_example_favours_new_blocks(self):
        reading = h.rate_reading(record(0.7, 0.60), record(0.6, 0.50), record(0.6, 0.52), record(0.6, 0.52), READING)
        self.assertAlmostEqual(reading["gap_at_recipe_rate"], 0.60 / 0.52 - 1)
        self.assertAlmostEqual(reading["gap_at_better_rates"], 0.50 / 0.52 - 1)
        self.assertAlmostEqual(reading["shift_toward_new_blocks"], 0.1923, places=4)
        self.assertTrue(reading["favours_new_blocks"])
        self.assertTrue(reading["forced_prefers_high_rate"])
        self.assertFalse(reading["swiglu_prefers_high_rate"])

    def test_a_rate_that_helps_both_equally_changes_nothing_but_flags_swiglus_recipe(self):
        reading = h.rate_reading(record(0.7, 0.60), record(0.6, 0.54), record(0.6, 0.52), record(0.5, 0.468), READING)
        self.assertAlmostEqual(reading["shift_toward_new_blocks"], 0.0)
        self.assertFalse(reading["favours_new_blocks"])
        self.assertTrue(reading["flag_swiglu_recipe_limit"])

    def test_a_rate_that_hurts_swiglu_cannot_flatter_the_forced_model(self):
        reading = h.rate_reading(record(0.7, 0.60), record(0.6, 0.59), record(0.6, 0.52), record(0.8, 0.70), READING)
        self.assertLess(0.59 / 0.70 - 1, 0)
        self.assertGreater(reading["gap_at_better_rates"], 0)
        self.assertFalse(reading["favours_new_blocks"])

    def test_a_favouring_check_decides_the_target_on_its_better_rates(self):
        forced, forced_high = record(0.7, 0.60), record(0.6, 0.50)
        swiglu, swiglu_high = record(0.6, 0.52), record(0.6, 0.52)
        entry = h.target_entry(forced, swiglu, forced_high, swiglu_high, READING)
        self.assertFalse(entry["at_recipe_rate"]["go_to_1b"])
        self.assertEqual(entry["decision"], entry["rate_check"]["step_two_at_better_rates"])
        self.assertTrue(entry["decision"]["go_to_1b"])

    def test_a_neutral_check_keeps_the_recipe_rate_decision(self):
        forced, swiglu = record(0.7, 0.60), record(0.6, 0.52)
        entry = h.target_entry(forced, swiglu, record(0.7, 0.59), record(0.6, 0.52), READING)
        self.assertEqual(entry["decision"], entry["at_recipe_rate"])

    def test_a_target_without_a_rate_check_reads_at_the_recipe_rate(self):
        entry = h.target_entry(record(0.7, 0.60), record(0.6, 0.52), None, None, READING)
        self.assertIsNone(entry["rate_check"])
        self.assertEqual(entry["decision"], entry["at_recipe_rate"])


class CurveCheck(unittest.TestCase):
    def setUp(self):
        self.recorded = [
            {"actual_tokens": 0, "requested_tokens": [0], "recovery_validation_kl": 1.0},
            {"actual_tokens": 10, "requested_tokens": [8], "recovery_validation_kl": 0.8},
            {"actual_tokens": 20, "requested_tokens": [16], "recovery_validation_kl": 0.6},
        ]

    def test_values_inside_the_limit_pass_and_token_0_is_not_recounted(self):
        history = copy.deepcopy(self.recorded)
        history[2]["recovery_validation_kl"] = 0.605
        checks = h.curve_checks(history, self.recorded, 0.5, 0.5, 0.01)
        self.assertEqual([row.get("tokens") for row in checks], [10, 20, None])
        self.assertTrue(all(row["passed"] for row in checks))

    def test_one_value_beyond_the_limit_fails_the_curve(self):
        history = copy.deepcopy(self.recorded)
        history[1]["recovery_validation_kl"] = 0.82
        self.assertFalse(all(row["passed"] for row in h.curve_checks(history, self.recorded, 0.5, 0.5, 0.01)))

    def test_the_8k_kl_counts_as_part_of_the_curve(self):
        history = copy.deepcopy(self.recorded)
        self.assertFalse(all(row["passed"] for row in h.curve_checks(history, self.recorded, 0.52, 0.5, 0.01)))


class TrainingContracts(unittest.TestCase):
    """Drive the real loop and scorers on a tiny random Llama, on the CPU, with two different blocks."""

    def setUp(self):
        torch.manual_seed(0)
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)
        config._attn_implementation = "sdpa"
        self.student = LlamaForCausalLM(config).eval()
        self.teacher = copy.deepcopy(self.student)
        self.blocks = {row.index: row for row in discover_mlp_blocks(self.student)}
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        folder = Path(self.directory.name)
        stream = folder / "tokens.int32"
        (numpy.arange(64, dtype=numpy.int32) % 32).tofile(stream)
        length = 8
        settings, reference, unused_path = h.load_settings(CONFIG)
        reference["recovery"] = {
            **reference["recovery"], "sequence_length": length, "effective_batch_tokens": length,
            "optimizer_backend": "single_tensor", "target_tokens": 40, "validation_interval_tokens": 16,
            "monitoring_ppl_tokens": [8], "segment_endpoints": [40],
        }
        self.reference = reference
        self.original_recovery = copy.deepcopy(reference["recovery"])
        batches = [{"input_ids": torch.arange(length).reshape(1, length),
                    "attention_mask": torch.ones(1, length, dtype=torch.long)}]
        cached = cache_teacher_logits(self.teacher, batches, 1, "cpu", "float32")
        selection = {"token_count": 64, "sha256": "stream", "protocol": "test"}
        self.artifact = h.start_artifact(settings, {"name": "test"}, CONFIG)
        self.task = SimpleNamespace(
            settings=reference,
            artifact=self.artifact,
            output_dir=folder,
            context=SimpleNamespace(model=self.student, blocks=self.blocks, device=torch.device("cpu")),
            teacher=self.teacher,
            selection_cache=cached,
            selection_path=stream,
            selection_record=selection,
            length=length,
            cache=PackedTokenCache(stream, 64, length, "stream"),
            validation_cache=cached,
            kl_path=stream,
            kl_record=selection,
        )
        swiglu = GatedMLPReplacement(16, 4)
        hybrid = DenseHybridReplacement(16, GatedMLPReplacement(16, 4))
        self.modules = {(0, "swiglu", 4): swiglu, (1, "linear_swiglu", 4): hybrid}
        rows = [
            {"layer": 0, "replacement_width": 4, "retains_dense_module": False,
             "replacement_parameters": sum(value.numel() for value in swiglu.parameters())},
            {"layer": 1, "operator": "linear_swiglu", "replacement_width": 4, "retains_dense_module": False,
             "replacement_parameters": sum(value.numel() for value in hybrid.parameters())},
        ]
        self.point = {"name": "forced-8", "rows": rows, "predicted_teacher_kl": 1.0}
        self.point.update({metric: value for metric, value in h1.score_point(
            self.task, self.modules, self.point, cached
        ).items() if metric in h.TOKEN_ZERO_METRICS})
        schedule = segment_schedule(0, 40, desired_requests(reference), length)
        self.recorded = [
            {"actual_tokens": 0, "requested_tokens": [0], "recovery_validation_kl": self.point["monitoring_kl"]},
            *({"actual_tokens": actual, "requested_tokens": list(requested), "recovery_validation_kl": 1.0}
              for actual, requested in schedule),
        ]
        self.run_row = {"name": "forced-0.8-lr1e-4", "target": 0.8, "model": "forced-8", "learning_rate": 0.01}

    def weights(self):
        return {name: value.detach().clone() for name, value in self.student.named_parameters()}

    def test_a_token_0_mismatch_stops_before_any_training(self):
        self.point["monitoring_kl"] *= 1.001
        start = {key: {name: value.detach().clone() for name, value in module.named_parameters()}
                 for key, module in self.modules.items()}
        with self.assertRaisesRegex(AcceptanceFailure, "token-0"):
            h.train_model(self.task, self.run_row, self.point, self.modules, self.recorded)
        self.assertNotIn("recovery", self.artifact["results"])
        for key, module in self.modules.items():
            for name, value in module.named_parameters():
                self.assertTrue(torch.equal(value, start[key][name]), name)

    def test_swiglus_own_model_must_also_match_swiglu8s_recorded_token_0(self):
        self.recorded[0]["recovery_validation_kl"] *= 1.001
        swiglu_run = {**self.run_row, "model": "swiglu-8"}
        with self.assertRaisesRegex(AcceptanceFailure, "token-0"):
            h.train_model(self.task, swiglu_run, self.point, self.modules, self.recorded)

    def test_a_checkpoint_schedule_unlike_swiglu8s_stops_before_training(self):
        self.recorded[1]["actual_tokens"] += 8
        with self.assertRaisesRegex(AcceptanceFailure, "schedule"):
            h.train_model(self.task, self.run_row, self.point, self.modules, self.recorded)
        self.assertNotIn("recovery", self.artifact["results"])

    def test_a_wrong_trainable_count_stops_before_training(self):
        self.point["rows"][1]["replacement_parameters"] += 1
        with self.assertRaisesRegex(AcceptanceFailure, "trainable-parameters"):
            h.train_model(self.task, self.run_row, self.point, self.modules, self.recorded)
        self.assertNotIn("recovery", self.artifact["results"])

    def test_the_run_keeps_swiglu8s_checkpoints_and_ends_with_the_8k_kl(self):
        h.train_model(self.task, self.run_row, self.point, self.modules, self.recorded)
        results = self.artifact["results"]
        history = results["recovery"]["validation_history"]
        self.assertEqual([row["actual_tokens"] for row in history], [row["actual_tokens"] for row in self.recorded])
        self.assertEqual(results["recovery"]["tokens_seen"], 40)
        self.assertEqual(history[0]["recovery_validation_kl"], self.point["monitoring_kl"])
        # Student and teacher start identical, so a KL of zero would mean a teacher MLP came back.
        self.assertTrue(all(row["recovery_validation_kl"] > 0 for row in history))
        self.assertGreater(results["teacher_kl_8192"]["kl"], 0)

    def test_the_runs_rate_reaches_the_optimizer_and_the_settings_stay_unchanged(self):
        h.train_model(self.task, self.run_row, self.point, self.modules, self.recorded)
        history = self.artifact["results"]["recovery"]["validation_history"]
        self.assertTrue(all(row["learning_rates"] == [0.01] for row in history))
        self.assertEqual(self.reference["recovery"], self.original_recovery)

    def test_only_the_inserted_blocks_train_and_the_teacher_mlps_return(self):
        before = self.weights()
        start = {key: {name: value.detach().clone() for name, value in module.named_parameters()}
                 for key, module in self.modules.items()}
        h.train_model(self.task, self.run_row, self.point, self.modules, self.recorded)
        for layer in (0, 1):
            self.assertIs(self.student.get_submodule(self.blocks[layer].path), self.blocks[layer].module)
        for name, value in self.student.named_parameters():
            self.assertTrue(torch.equal(value, before[name]), name)
        for key, module in self.modules.items():
            self.assertFalse(all(torch.equal(value, start[key][name]) for name, value in module.named_parameters()))

    def test_a_non_finite_score_stops_the_run(self):
        with mock.patch.object(h, "evaluate_validation_kl_mixed", return_value=float("nan")):
            with self.assertRaises(FloatingPointError):
                h.train_model(self.task, self.run_row, self.point, self.modules, self.recorded)


class ControlOutcome(unittest.TestCase):
    """The control's result stays complete when it leaves the limit; only the job fails."""

    def run_control(self, control_kl):
        settings, reference, unused_path = h.load_settings(CONFIG)
        recorded = [
            {"actual_tokens": 0, "requested_tokens": [0], "recovery_validation_kl": 1.0},
            {"actual_tokens": 10, "requested_tokens": [10], "recovery_validation_kl": 0.8},
        ]
        swiglu8 = {"results": {"recovery": {"validation_history": recorded},
                               "evaluation": {"teacher_kl_8192": {"kl": 0.5}}}}
        assembled = {"results": {"points": [{"name": "swiglu-8", "rows": []}]}}

        def trained(task, run, point, modules, history):
            task.artifact["results"]["recovery"] = {
                "validation_history": [recorded[0], {**recorded[1], "recovery_validation_kl": control_kl}],
            }
            task.artifact["results"]["teacher_kl_8192"] = {"kl": 0.5}

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        output = Path(directory.name) / "output"
        args = SimpleNamespace(task_index=0, work_dir=Path(directory.name) / "work", output_dir=output,
                               swiglu8_root=Path("unused"), heterogeneous1_root=Path("unused"))
        task = SimpleNamespace(prepared={"results": {"width_curves": {"rows": []}}})
        with mock.patch.object(h, "pinned_json", side_effect=[assembled, swiglu8]), \
                mock.patch.object(h, "open_task", side_effect=lambda *values: setattr(task, "artifact", values[3]) or task), \
                mock.patch.object(h, "completed_table_rows", return_value=([], {})), \
                mock.patch.object(h, "fit_needed"), mock.patch.object(h, "attach_inputs"), \
                mock.patch.object(h, "train_model", side_effect=trained):
            try:
                h.train(args, settings, reference, CONFIG)
            except AcceptanceFailure as error:
                return read_json(output / "result.json"), error
        return read_json(output / "result.json"), None

    def test_a_curve_inside_the_limit_ends_normally(self):
        result, error = self.run_control(0.801)
        self.assertIsNone(error)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["results"]["acceptance"]["curve"]["passed"])

    def test_a_curve_outside_the_limit_fails_the_job_and_keeps_the_complete_result(self):
        result, error = self.run_control(0.82)
        self.assertRegex(str(error), "1% limit")
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["results"]["acceptance"]["curve"]["passed"])


class SummaryContracts(unittest.TestCase):
    def setUp(self):
        self.settings, self.reference, unused_path = h.load_settings(CONFIG)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for index, run in enumerate(self.settings["runs"]):
            self.write_run(run, passed=True)

    def history(self, at_50, at_100):
        return [
            {"requested_tokens": [50_000_000], "recovery_validation_kl": at_50},
            {"requested_tokens": [100_000_000], "recovery_validation_kl": at_100},
        ]

    def write_run(self, run, passed, code="same"):
        folder = self.root / h.run_output_name(run["name"])
        folder.mkdir(parents=True, exist_ok=True)
        forced = run["model"] == "forced-8"
        artifact = {
            "workflow": h.WORKFLOW, "status": "completed", "code_hashes": {"file": code},
            "updated_at_utc": "2026-11-06T00:00:00+00:00",
            "results": {
                "recovery": {"validation_history": self.history(0.66 if forced else 0.6, 0.63 if forced else 0.6)},
                "teacher_kl_8192": {"kl": 0.65 if forced else 0.6},
                "acceptance": {"curve": {"passed": passed, "checks": [
                    {"check": "curve", "relative_difference": 0.002, "passed": passed},
                ]}},
            },
        }
        write_json_atomic(folder / "result.json", artifact)

    def summarize(self):
        recorded = {"results": {"recovery": {"validation_history": self.history(0.6, 0.6)},
                                "evaluation": {"teacher_kl_8192": {"kl": 0.6}}}}
        args = SimpleNamespace(results_root=self.root, swiglu8_root=Path("unused"))
        with mock.patch.object(h, "pinned_json", return_value=recorded):
            h.summarize(args, self.settings, self.reference)
        return read_json(self.root / "summary-001" / "result.json")["results"]

    def test_every_forced_target_gets_a_decision_and_only_80_a_rate_check(self):
        results = self.summarize()
        self.assertEqual(sorted(results["targets"]), ["0.2", "0.5", "0.8"])
        self.assertIsNone(results["targets"]["0.2"]["rate_check"])
        self.assertIsNone(results["targets"]["0.5"]["rate_check"])
        self.assertIsNotNone(results["targets"]["0.8"]["rate_check"])
        self.assertTrue(results["control"]["passed"])

    def test_a_rerun_with_the_same_inputs_is_safe(self):
        self.assertEqual(self.summarize(), self.summarize())

    def test_a_flagged_control_stays_readable_and_is_reported(self):
        self.write_run(self.settings["runs"][0], passed=False)
        self.assertFalse(self.summarize()["control"]["passed"])

    def test_runs_on_other_code_are_refused(self):
        self.write_run(self.settings["runs"][3], passed=True, code="other")
        with self.assertRaisesRegex(ValueError, "code"):
            self.summarize()


class ConfigurationContracts(unittest.TestCase):
    def setUp(self):
        self.settings, self.reference, unused_path = h.load_settings(CONFIG)
        self.runs = self.settings["runs"]

    def test_the_configuration_holds_the_decided_six_runs(self):
        recipe = self.reference["recovery"]["learning_rate"]
        high = self.settings["reading"]["high_learning_rate"]
        self.assertEqual(recipe, 3e-05)
        self.assertEqual(high, 1e-04)
        self.assertEqual(
            [(run["model"], run["target"], run["learning_rate"]) for run in self.runs],
            [("swiglu-8", 0.8, recipe), ("forced-8", 0.2, recipe), ("forced-8", 0.5, recipe),
             ("forced-8", 0.8, recipe), ("forced-8", 0.8, high), ("swiglu-8", 0.8, high)],
        )

    def test_the_recipe_is_swiglu8s_and_the_bar_is_the_decided_one(self):
        swiglu8 = read_json(ROOT / "workflows/configs/model/swiglu/swiglu-8.json")
        self.assertEqual(self.reference["recovery"], swiglu8["recovery"])
        self.assertEqual(self.settings["reading"]["bar"], 0.035)

    def test_a_second_control_or_a_control_out_of_place_is_refused(self):
        for runs in (list(reversed(self.runs)), [*self.runs, {**self.runs[0], "name": "another"}]):
            changed = {**self.settings, "runs": runs}
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "config.json"
                write_json_atomic(path, changed)
                with self.assertRaisesRegex(ValueError, "control"):
                    h.load_settings(path)

    def test_a_run_without_a_pinned_assembly_is_refused(self):
        changed = copy.deepcopy(self.settings)
        changed["runs"].append({"name": "forced-0.3", "target": 0.3, "model": "forced-8", "learning_rate": 3e-05})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            write_json_atomic(path, changed)
            with self.assertRaisesRegex(ValueError, "pinned"):
                h.load_settings(path)


class JobContracts(unittest.TestCase):
    def setUp(self):
        self.source = SBATCH.read_text(encoding="utf-8")
        self.settings, unused_reference, unused_path = h.load_settings(CONFIG)

    def test_the_job_repeats_the_configurations_run_order(self):
        names = re.search(r"RUN_NAMES=\(([^)]*)\)", self.source).group(1).split()
        self.assertEqual(names, [run["name"] for run in self.settings["runs"]])

    def test_job_output_names_match_the_runner(self):
        self.assertEqual(h.run_output_name("forced-0.2"), "forced-0.2-run-001")
        self.assertIn('OUTPUT_NAME="${RUN_NAMES[$TASK_ID]}-run-001"', self.source)

    def test_the_job_checks_the_assemblies_the_configuration_pins(self):
        targets = re.search(r"for TARGET in ([^;]*);", self.source).group(1).split()
        self.assertEqual(targets, sorted(self.settings["sources"]["heterogeneous1_assemble_sha256"]))


if __name__ == "__main__":
    unittest.main()
