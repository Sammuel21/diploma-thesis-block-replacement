"""Focused CPU/static contracts for SwiGLU-9; no model or dataset is loaded."""

import re
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from mlp_replacement.artifacts import read_json, write_json_atomic
from mlp_replacement.capture import ActivationPairs
from mlp_replacement.operators import GatedMLPReplacement
from workflows.runs.model.swiglu import swiglu_8, swiglu_9

CONFIG = ROOT / "workflows/configs/model/swiglu/swiglu-9.json"
SBATCH = ROOT / "workflows/jobs/perun/swiglu_9.sbatch"
WIDTHS = (2048, 4096)
COMBINATIONS = [(layer, width) for layer in swiglu_9.LAYERS for width in WIDTHS]
CHECKPOINTS = [0, 10_000_000, *range(25_000_000, 1_000_000_001, 25_000_000)]


def reference_block_check(values=None):
    """SwiGLU-8's block check: c1e-3-128 at three seeds, {seed: (kl_128, kl_8192)} per combination."""

    values = values or {21: (1.0, 1.0), 22: (1.0, 1.0), 23: (1.0, 1.0)}
    fits = [
        {"setting": "c1e-3-128", "seed": seed, "layer": layer, "replacement_width": width,
         "kl_128": short, "kl_8192": long, "local_nmse": 0.3, "local_mse": 0.1}
        for layer, width in COMBINATIONS for seed, (short, long) in values.items()
    ]
    return {
        "status": "completed",
        "results": {
            "adoption": {"selected_recipe": "c1e-3-128"},
            "fits": fits,
            "slices": [{"layer": layer, "replacement_width": width, "validation_128": {"nmse": 0.5}}
                       for layer, width in COMBINATIONS],
            "long_context_data": {
                "audit": {"local_window_records_digest": "w"},
                **{name: {"sha256": name} for name in
                   ("kl_stream", "calibration_stream", "validation_stream", "selection_stream")},
            },
            "ranking_digests": {str(layer): f"rank-{layer}" for layer in swiglu_9.LAYERS},
        },
    }


def part_a_rows(values=None):
    """Single-seed Part A rows; {setting: (kl_128, kl_8192)}, default 1.0 at both contexts."""

    values = values or {}
    return [
        {"setting": setting, "seed": 21, "layer": layer, "replacement_width": width, "width_ratio": width / 8192,
         "kl_128": values.get(setting, (1.0, 1.0))[0], "kl_8192": values.get(setting, (1.0, 1.0))[1],
         "best_epoch": 3, "history": [], "local_nmse": 0.3, "local_mse": 0.1}
        for setting in swiglu_9.PART_A_SETTINGS for layer, width in COMBINATIONS
    ]


def block_check_tasks(settings):
    rows = part_a_rows()
    passed = {"checks": [], "passed": True}
    return [
        {
            "workflow": "swiglu-9", "command": "block-check", "status": "completed",
            "identity": {"kind": "block-check-task", "task_index": index, "layer": layer},
            "configuration": deepcopy(settings), "configuration_sha256": "c" * 64, "code_hashes": {"runner": "h"},
            "environment": {"python": "3.12.14", "packages": {"torch": "2.11.0+cu128"}, "cuda_runtime": "12.8",
                            "gpu": "NVIDIA H200", "platform": f"node-{index}"},
            "created_at_utc": f"2026-10-18T0{index}:00:00+00:00", "updated_at_utc": "2026-10-18T12:00:00+00:00",
            "run_fingerprint": f"{index:064d}",
            "results": {
                "long_context_data": {"kl_stream": {"sha256": "k" * 64}},
                "calibration_extension": {"record_indices_digest": "e", "windows": 21504},
                "ranking_digest": f"rank-{layer}",
                "fits": [row for row in rows if row["layer"] == layer],
                "slices": [{"layer": layer, "replacement_width": width} for width in WIDTHS],
                "l2_sp_calibration": {str(width): {"alpha": {}} for width in WIDTHS},
                "acceptance": {name: passed for name in ("D1", "D2", "P5", "A1", "A2-A3")},
            },
        }
        for index, layer in enumerate(swiglu_9.LAYERS)
    ]


def history(kl, ppl_points=(0, 10_000_000, 25_000_000, 50_000_000, 100_000_000, 250_000_000, 500_000_000,
                            1_000_000_000)):
    """A monitoring history with one row per SwiGLU-7 checkpoint; kl(tokens) gives the KL."""

    rows = []
    for tokens in CHECKPOINTS:
        row = {"actual_tokens": tokens, "requested_tokens": [tokens], "recovery_validation_kl": kl(tokens)}
        if tokens in ppl_points:
            row["wikitext_validation"] = {"perplexity": 20.0}
        rows.append(row)
    return rows


def evaluation(kl, ppl=8.0, accuracy=0.6):
    return {
        "teacher_kl_8192": {"kl": kl},
        "likelihood": {"test-8192": {"perplexity": ppl}},
        "comparison_to_dense": {"tasks": {"piqa": {"student": accuracy}}, "macro_accuracy": accuracy},
    }


class SwiGLU9Contracts(unittest.TestCase):
    def setUp(self):
        self.settings, unused_path = swiglu_9.load_settings(CONFIG)

    def test_configuration_fixes_the_decided_grid_and_reuses_swiglu7_and_swiglu8(self):
        settings = self.settings
        grid = settings["part_a"]["settings"]
        self.assertEqual(len(grid), 12)
        self.assertEqual(len(swiglu_9.NEW_SETTINGS), 11)
        self.assertEqual(swiglu_9.REPRODUCTION_SETTING, "mse-393k")
        self.assertEqual({(row["loss"], row["calibration_pairs"]) for row in grid.values()},
                         {(loss, pairs) for loss in ("mse", "l2sp-weak", "l2sp-strong")
                          for pairs in (393_216, 786_432, 1_572_864, 3_145_728)})
        self.assertTrue(all(pairs % 256 == 0 for pairs in swiglu_9.CALIBRATION_SIZES))
        self.assertEqual(settings["part_a"]["extension"]["windows"] * 128 + 393_216, 3_145_728)
        self.assertEqual(settings["part_a"]["losses"], {"mse": None, "l2sp-weak": 0.1, "l2sp-strong": 1.0})
        swiglu8 = read_json(ROOT / "workflows/configs/model/swiglu/swiglu-8.json")
        self.assertEqual(settings["part_a"]["recipe"], swiglu8["block_check"]["settings"]["c1e-3-128"])
        swiglu7 = read_json(ROOT / "workflows/configs/model/swiglu/swiglu-7.json")
        self.assertEqual(settings["recovery"], swiglu7["recovery"])
        self.assertEqual(settings["recovery"]["target_tokens"], 1_000_000_000)
        self.assertEqual(settings["recovery"]["validation_interval_tokens"], 25_000_000)
        self.assertEqual([(run["arm"], run["target"]) for run in settings["part_b"]["runs"]], list(swiglu_9.RUNS))
        self.assertEqual(settings["part_b"]["verdict"]["gap_bar"], 0.05)

    def test_configuration_rejects_changed_contracts(self):
        changes = (
            lambda value: value["part_a"]["losses"].update({"l2sp-weak": 0.2}),
            lambda value: value["part_a"]["settings"]["mse-786k"].update(calibration_pairs=700_000),
            lambda value: value["part_a"]["calibration_sizes"].pop(),
            lambda value: value["recovery"].update(target_tokens=100_000_000),
            lambda value: value["part_b"]["verdict"].update(gap_bar=0.02),
            lambda value: value["sources"]["swiglu8_runs_sha256"].update({"slice:0.5": "0" * 64}) or
            value["sources"]["swiglu8_runs_sha256"].pop("slice:0.8"),
            lambda value: value["sources"]["swiglu7_runs_sha256"].update({"0.2": "1" * 64}),
            lambda value: value["acceptance"].update(reproduction_review_relative=0.02),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for index, change in enumerate(changes):
                changed = deepcopy(self.settings)
                change(changed)
                write_json_atomic(path, changed)
                with self.subTest(change=index), self.assertRaises(ValueError):
                    swiglu_9.load_settings(path)

    def test_block_check_runs_one_array_task_per_layer(self):
        self.assertEqual([swiglu_9.block_check_layer(index) for index in range(3)], [3, 11, 19])
        self.assertEqual(swiglu_9.block_check_output_name(2), "block-check-001-task-02")
        for index in (-1, 3):
            with self.assertRaises(ValueError):
                swiglu_9.block_check_layer(index)

    def test_l2_sp_strength_keeps_one_over_one_plus_ratio_of_the_fit_displacement(self):
        gain, distance = 0.03, 2.0
        for ratio in (0.1, 1.0):
            alpha = swiglu_9.l2_sp_alpha(ratio, gain, distance)
            self.assertAlmostEqual(alpha * distance / 2, ratio * gain)
            # Minimize gain * (1 - t)^2 + (alpha / 2) * distance * t^2 along the fit direction.
            steps = torch.linspace(0, 1, 100_001, dtype=torch.float64)
            objective = gain * (1 - steps) ** 2 + 0.5 * alpha * distance * steps ** 2
            self.assertAlmostEqual(float(steps[objective.argmin()]), 1 / (1 + ratio), places=4)
        for bad in ((0.1, 0.0, 1.0), (0.1, 0.1, 0.0), (0.1, -0.1, 1.0)):
            with self.assertRaises(ValueError):
                swiglu_9.l2_sp_alpha(*bad)

    def test_calibration_sizes_are_views_of_one_capture(self):
        inputs = torch.arange(40, dtype=torch.float32).reshape(10, 4)
        pairs = ActivationPairs(inputs=inputs, targets=inputs + 1)
        prefix = swiglu_9.calibration_prefix(pairs, 6)
        self.assertEqual(prefix.num_tokens, 6)
        self.assertEqual(prefix.inputs.data_ptr(), inputs.data_ptr())
        self.assertTrue(torch.equal(prefix.targets, inputs[:6] + 1))
        with self.assertRaises(ValueError):
            swiglu_9.calibration_prefix(pairs, 11)

    def test_squared_distance_sums_every_parameter(self):
        torch.manual_seed(0)
        start = GatedMLPReplacement(8, 4)
        moved = deepcopy(start)
        with torch.no_grad():
            moved.down_projection.weight.add_(0.5)
        self.assertAlmostEqual(swiglu_9.squared_distance(moved, start), 0.25 * 8 * 4, places=6)
        self.assertEqual(swiglu_9.squared_distance(start, start), 0.0)

    def test_data_identity_and_reproduction_checks_compare_with_swiglu8(self):
        reference = reference_block_check()
        long_data = deepcopy(reference["results"]["long_context_data"])
        audit = {"windows": 21504, "document_disjoint": True}
        self.assertTrue(all(row["passed"] for row in swiglu_9.data_identity_checks(
            long_data, audit, reference, self.settings)))
        long_data["selection_stream"]["sha256"] = "other"
        failed = [row["metric"] for row in swiglu_9.data_identity_checks(
            long_data, {**audit, "document_disjoint": False}, reference, self.settings) if not row["passed"]]
        self.assertEqual(failed, ["selection_stream_sha256", "extension_document_disjoint"])
        slices = [{"layer": 3, "replacement_width": 2048, "validation_128": {"nmse": 0.50003}}]
        self.assertTrue(swiglu_9.slice_checks(slices, reference, 1e-4)[0]["passed"])
        rows = [{"setting": "mse-393k", "layer": 3, "replacement_width": 2048, "local_nmse": 0.3,
                 "local_mse": 0.1, "kl_128": 1.0005, "kl_8192": 1.002},
                {"setting": "mse-786k", "layer": 3, "replacement_width": 2048, "local_nmse": 9.0,
                 "local_mse": 9.0, "kl_128": 9.0, "kl_8192": 9.0}]
        checks = swiglu_9.reproduction_checks(rows, reference, 1e-3)
        self.assertEqual([(row["check"], row["metric"], row["passed"]) for row in checks],
                         [("A2", "local_nmse", True), ("A2", "local_mse", True),
                          ("A3", "kl_128", True), ("A3", "kl_8192", False)])

    def test_adoption_uses_the_swiglu8_seed_mean_and_twice_its_seed_spread(self):
        # Reference mean 1.0 at both contexts with sample sd 0.02 at 8K: seed noise 4%.
        reference = reference_block_check({21: (1.0, 1.0), 22: (1.0, 1.02), 23: (1.0, 0.98)})
        values = {
            "mse-3146k": (0.97, 0.90),           # 10% better at 8K: qualifies and wins
            "l2sp-weak-786k": (1.0, 0.94),       # 6% better: qualifies
            "l2sp-strong-393k": (1.08, 0.80),    # 20% better at 8K, but 8% worse at 128
            "l2sp-weak-393k": (1.0, 0.965),      # 3.5% better: below 5% and below the noise
        }
        decision = swiglu_9.adoption_decision(part_a_rows(values), reference, self.settings)
        self.assertEqual(sorted(decision["qualifying"]), ["l2sp-weak-786k", "mse-3146k"])
        self.assertEqual(decision["selected_setting"], "mse-3146k")
        self.assertAlmostEqual(decision["settings"]["mse-3146k"]["mean_seed_noise"], 0.04)
        self.assertIn("worse", decision["settings"]["l2sp-strong-393k"]["reasons"][0])
        self.assertEqual(len(decision["settings"]["l2sp-weak-393k"]["reasons"]), 2)
        self.assertNotIn("mse-393k", decision["settings"])
        nothing = swiglu_9.adoption_decision(part_a_rows(), reference, self.settings)
        self.assertEqual((nothing["selected_setting"], nothing["qualifying"]), ("mse-393k", []))
        incomplete = [row for row in part_a_rows()
                      if (row["setting"], row["layer"], row["replacement_width"]) != ("mse-786k", 3, 2048)]
        with self.assertRaisesRegex(ValueError, "lacks mse-786k"):
            swiglu_9.adoption_decision(incomplete, reference, self.settings)
        missing_seed = reference_block_check({21: (1.0, 1.0), 22: (1.0, 1.0)})
        with self.assertRaisesRegex(ValueError, "lacks seeds"):
            swiglu_9.adoption_decision(part_a_rows(), missing_seed, self.settings)

    def test_merge_joins_identical_layer_tasks_and_refuses_differences(self):
        reference = reference_block_check()
        records = [{"output": swiglu_9.block_check_output_name(index)} for index in range(3)]
        merged = swiglu_9.merge_block_check(block_check_tasks(self.settings), self.settings, records, reference)
        self.assertEqual(len(merged["results"]["fits"]), 72)
        self.assertEqual(merged["results"]["adoption"]["selected_setting"], "mse-393k")
        self.assertEqual(set(merged["results"]["stage_two_indicator"]), set(swiglu_9.PART_A_SETTINGS))
        self.assertEqual(merged["results"]["ranking_digests"], {"3": "rank-3", "11": "rank-11", "19": "rank-19"})
        changes = {
            "different extension": lambda tasks: tasks[1]["results"]["calibration_extension"].update(windows=1),
            "different software": lambda tasks: tasks[2]["environment"]["packages"].update(torch="2.12.0"),
            "unfinished task": lambda tasks: tasks[0].update(status="failed"),
            "missing fit": lambda tasks: tasks[1]["results"]["fits"].pop(),
            "failed check": lambda tasks: tasks[2]["results"]["acceptance"]["A2-A3"].update(passed=False),
            "missing calibration": lambda tasks: tasks[0]["results"]["l2_sp_calibration"].pop("4096"),
            "missing task": lambda tasks: tasks.pop(),
        }
        for name, change in changes.items():
            tasks = block_check_tasks(self.settings)
            change(tasks)
            with self.subTest(change=name), self.assertRaises(ValueError):
                swiglu_9.merge_block_check(tasks, self.settings, records, reference)

    def test_verdict_follows_the_logged_five_percent_rule(self):
        cases = (
            (0.03, 0.02, "closed"),
            (-0.04, 0.02, "closed"),
            (0.05, 0.02, "still_matters"),
            (0.12, 0.049, "still_matters"),
            (0.04, 0.06, "cannot_tell"),
            (0.06, 0.06, "cannot_tell"),
            (0.09, 0.06, "still_matters"),
            (-0.09, 0.06, "outside_logged_rule"),
        )
        for gap, noise, expected in cases:
            with self.subTest(gap=gap, noise=noise):
                self.assertEqual(swiglu_9.part_b_verdict(gap, noise, 0.05), expected)
        self.assertAlmostEqual(swiglu_9.relative_gap(0.11, 0.10), 0.1)

    def test_twin_noise_covers_the_21_checkpoints_from_500m_and_the_final_kl(self):
        twin = history(lambda tokens: 0.2 * (1.03 if tokens == 700_000_000 else 1.0))
        reference = history(lambda tokens: 0.2 * (1.5 if tokens == 400_000_000 else 1.0))
        noise = swiglu_9.twin_noise(twin, reference, 0.101, 0.100, [500_000_000, 1_000_000_000], 25_000_000)
        self.assertEqual(len(noise["points"]), 22)
        self.assertEqual(noise["points"][0]["tokens"], 500_000_000)
        self.assertEqual(noise["points"][-2]["tokens"], 1_000_000_000)
        self.assertAlmostEqual(noise["value"], 0.03)
        self.assertEqual(noise["largest_at"]["tokens"], 700_000_000)
        final = swiglu_9.twin_noise(twin, reference, 0.105, 0.100, [500_000_000, 1_000_000_000], 25_000_000)
        self.assertEqual(final["largest_at"]["metric"], "teacher_kl_8192")
        with self.assertRaisesRegex(ValueError, "monitoring row"):
            swiglu_9.twin_noise(twin[:-1], reference, 0.1, 0.1, [500_000_000, 1_000_000_000], 25_000_000)

    def test_gap_curve_closing_point_and_token_equivalent(self):
        fitted = history(lambda tokens: 1.0 / (1 + tokens / 1e8))
        sliced = history(lambda tokens: 1.3 / (1 + tokens / 1e8) if tokens < 600_000_000 else 1.02 / (1 + tokens / 1e8))
        curve = swiglu_9.gap_curve(fitted, sliced)
        self.assertEqual(len(curve), len(CHECKPOINTS))
        self.assertAlmostEqual(curve[0]["gap"], 0.3)
        self.assertEqual(swiglu_9.closing_point(curve, 0.05), 600_000_000)
        self.assertIsNone(swiglu_9.closing_point(curve[:20], 0.05))
        equivalent = {row["fitted_tokens"]: row for row in swiglu_9.token_equivalent(curve)}
        # Slice-only reaches the fitted KL at 100M (0.5) between 150M (0.520) and 175M (0.473).
        self.assertGreater(equivalent[100_000_000]["slice_tokens"], 150_000_000)
        self.assertLess(equivalent[100_000_000]["slice_tokens"], 175_000_000)
        self.assertIsNone(equivalent[1_000_000_000]["slice_tokens"])
        self.assertIsNone(equivalent[0]["ratio"])
        shifted = deepcopy(sliced)
        shifted[3]["actual_tokens"] += 8192
        with self.assertRaisesRegex(ValueError, "different token counts"):
            swiglu_9.gap_curve(fitted, shifted)

    def test_reproduction_against_swiglu8_stops_at_token_zero_and_ignores_rows_after_100m(self):
        reference = [row for row in history(lambda tokens: 0.2) if row["actual_tokens"] <= 100_000_000
                     and row["actual_tokens"] in (0, 10_000_000, 25_000_000, 50_000_000, 75_000_000, 100_000_000)]
        rows = history(lambda tokens: 0.2 * (1.02 if tokens == 50_000_000 else 1.0))
        artifact = {"results": {"recovery": {"validation_history": rows}}}
        stages = []
        original = swiglu_8.persist
        swiglu_8.persist = lambda output_dir, artifact, stage: stages.append(stage)
        try:
            swiglu_9.check_against_swiglu8(artifact, None, reference, {"acceptance": self.settings["acceptance"]})
            acceptance = artifact["results"]["acceptance"]
            self.assertTrue(acceptance["C1"]["passed"])
            self.assertEqual(len(acceptance["C2"]["checks"]), 9)  # KL at 5 points and PPL at 4
            self.assertEqual([row["passed"] for row in acceptance["C2"]["checks"] if not row["passed"]], [False])
            rows[0]["recovery_validation_kl"] = 0.21
            with self.assertRaisesRegex(swiglu_8.AcceptanceFailure, "C1 failed"):
                swiglu_9.check_against_swiglu8(artifact, None, reference, {"acceptance": self.settings["acceptance"]})
        finally:
            swiglu_8.persist = original

    def test_environment_difference_lists_only_changed_fields(self):
        reference = {"python": "3.12.14", "gpu": "NVIDIA H200", "packages": {"torch": "2.11.0", "lm_eval": "0.4.13"}}
        current = {"python": "3.12.14", "gpu": "NVIDIA H200", "packages": {"torch": "2.11.0", "lm_eval": "0.4.14"}}
        self.assertEqual(swiglu_9.environment_difference(current, reference),
                         {"packages": {"lm_eval": {"value": "0.4.14", "reference": "0.4.13"}}})
        self.assertEqual(swiglu_9.environment_difference(reference, reference), {})

    def test_part_b_summary_applies_the_verdict_per_target(self):
        settings = self.settings
        passed = {"checks": [], "passed": True}

        def run(arm, target, final, kl):
            return {
                "workflow": "swiglu-9", "command": "train", "status": "completed", "configuration": deepcopy(settings),
                "configuration_sha256": "c" * 64, "code_hashes": {"runner": "h"}, "run_fingerprint": f"{arm}{target}",
                "updated_at_utc": "2026-10-19T00:00:00+00:00",
                "identity": {"arm": arm, "target": target, "role": swiglu_9.run_role(arm, target)},
                "results": {"recovery": {"validation_history": history(kl)}, "evaluation": evaluation(final),
                            "acceptance": {"C1": passed, "B1": passed,
                                           "C2": {"checks": [{"check": "C2", "passed": False, "tokens": 10_000_000}],
                                                  "passed": False}}},
            }

        runs = {
            ("fitted", 0.5): run("fitted", 0.5, 0.101, lambda tokens: 0.2 * 1.01),
            ("fitted", 0.8): run("fitted", 0.8, 0.30, lambda tokens: 0.3),
            ("slice", 0.2): run("slice", 0.2, 0.0612, lambda tokens: 0.061),
            ("slice", 0.5): run("slice", 0.5, 0.115, lambda tokens: 0.23),
            ("slice", 0.8): run("slice", 0.8, 0.33, lambda tokens: 0.33),
        }
        reference = {"workflow": "swiglu-9", "command": "reference-kl", "status": "completed",
                     "configuration": deepcopy(settings), "configuration_sha256": "c" * 64,
                     "code_hashes": {"runner": "h"}, "updated_at_utc": "2026-10-19T01:00:00+00:00",
                     "results": {"targets": {"0.2": {"teacher_kl_8192": {"kl": 0.06}},
                                             "0.5": {"teacher_kl_8192": {"kl": 0.100}}}}}
        swiglu7 = {0.2: {"results": {"recovery": {"validation_history": history(lambda tokens: 0.06)},
                                     "evaluation": evaluation(0.06)}},
                   0.5: {"results": {"recovery": {"validation_history": history(lambda tokens: 0.2)},
                                     "evaluation": evaluation(0.100, ppl=8.08, accuracy=0.59)}}}
        summary = swiglu_9.summarize_part_b(runs, reference, swiglu7, settings)
        results = summary["results"]
        self.assertAlmostEqual(results["noise"]["value"], 0.01)
        self.assertTrue(results["noise_valid"])
        self.assertEqual({target: row["verdict"] for target, row in results["targets"].items()},
                         {"0.2": "closed", "0.5": "still_matters", "0.8": "still_matters"})
        self.assertAlmostEqual(results["targets"]["0.5"]["gap"], 0.15)
        self.assertEqual(results["targets"]["0.2"]["fitted_source"], "swiglu-7 S7-0-target-0.2-run-001")
        self.assertEqual(results["targets"]["0.8"]["fitted_source"], "fitted-target-0.8-1b-run-001")
        self.assertAlmostEqual(results["twin"]["gap_with_twin_at_0.5"], 0.115 / 0.101 - 1)
        self.assertAlmostEqual(results["twin"]["final_evaluation_differences"]["task_accuracy"]["piqa"]["difference"],
                               0.01)
        self.assertEqual(len(results["flagged_for_review"]), 5)
        changed = deepcopy(runs)
        changed[("slice", 0.2)]["code_hashes"] = {"runner": "other"}
        with self.assertRaisesRegex(ValueError, "differ"):
            swiglu_9.summarize_part_b(changed, reference, swiglu7, settings)
        missing = dict(runs)
        missing.pop(("slice", 0.8))
        with self.assertRaisesRegex(ValueError, "all five"):
            swiglu_9.summarize_part_b(missing, reference, swiglu7, settings)

    def test_summary_command_writes_once_and_refuses_unfinished_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reference-kl-001.perun-lock").mkdir()
            with self.assertRaisesRegex(ValueError, "did not finish"):
                swiglu_9.finished_outputs(root, ["reference-kl-001"])
            with self.assertRaises(FileNotFoundError):
                swiglu_9.finished_outputs(root, ["slice-target-0.2-1b-run-001"])
            artifact = {"schema_version": 1, "workflow": "swiglu-9", "command": "summarize-part-b",
                        "status": "completed", "configuration": {}, "results": {"value": 1}}
            output = root / "part-b-summary-001"
            swiglu_9.write_once(output, artifact, "part_b_summary")
            swiglu_9.write_once(output, deepcopy(artifact), "part_b_summary")
            with self.assertRaisesRegex(ValueError, "different"):
                swiglu_9.write_once(output, {**artifact, "results": {"value": 2}}, "part_b_summary")

    def test_runs_have_roles_and_unique_outputs(self):
        self.assertEqual(swiglu_9.run_role("fitted", 0.5), "noise_twin")
        self.assertEqual({swiglu_9.run_role(arm, target) for arm, target in swiglu_9.RUNS[1:]}, {"new"})
        with self.assertRaises(ValueError):
            swiglu_9.run_role("fitted", 0.2)   # reused from SwiGLU-7, not rerun
        names = [swiglu_9.train_output_name(arm, target) for arm, target in swiglu_9.RUNS]
        self.assertEqual(len(set(names)), 5)
        self.assertIn("slice-target-0.8-1b-run-001", names)

    def test_perun_job_maps_layers_and_runs_like_the_runner(self):
        source = SBATCH.read_text(encoding="utf-8")
        arms = re.search(r"^RUN_ARMS=\(([^)]*)\)", source, flags=re.MULTILINE).group(1).split()
        targets = re.search(r"^RUN_TARGETS=\(([^)]*)\)", source, flags=re.MULTILINE).group(1).split()
        self.assertEqual(list(zip(arms, (float(value) for value in targets))), list(swiglu_9.RUNS))
        references = re.search(r"^REFERENCE_TARGETS=\(([^)]*)\)", source, flags=re.MULTILINE).group(1).split()
        self.assertEqual([float(value) for value in references], list(swiglu_9.REUSED_FITTED_TARGETS))
        self.assertIn("(( TASK_ID >= 0 && TASK_ID <= 2 ))", source)
        self.assertIn("(( TASK_ID >= 0 && TASK_ID <= 4 ))", source)
        self.assertIn('OUTPUT_NAME="${ARM}-target-${TARGET}-1b-run-001"', source)
        self.assertIn("OUTPUT_NAME=\"$(printf 'block-check-001-task-%02d' \"$TASK_ID\")\"", source)
        self.assertIn("--exclude='prepared/recovery-tokens.int32'", source)
        self.assertIn("-m workflows.runs.model.swiglu.swiglu_9", source)
        self.assertIn('PROJECT_RESULTS_ROOT="$PROJECT_ROOT/perun-results/swiglu-9"', source)
        self.assertIn("trap finalize_job EXIT", source)
        self.assertIn("rsync -rcn", source)
        self.assertTrue(callable(swiglu_9.main))


if __name__ == "__main__":
    unittest.main()
