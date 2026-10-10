"""Focused CPU/static contracts for SwiGLU-8; no model or dataset is loaded."""

import re
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from mlp_replacement.artifacts import read_json, write_json_atomic
from mlp_replacement.data_streams import build_finite_stream
from workflows.runs.model.swiglu import swiglu_7, swiglu_8

CONFIG = ROOT / "workflows/configs/model/swiglu/swiglu-8.json"
SBATCH = ROOT / "workflows/jobs/perun/swiglu_8.sbatch"
COMBINATIONS = [(layer, width) for layer in (3, 11, 19) for width in (2048, 4096)]


class CharacterTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens, return_attention_mask):
        return types.SimpleNamespace(input_ids=[ord(word[0]) for word in text.split()])


def block_rows(values):
    """Build block-check rows from {recipe: (kl_128, kl_8192)} per combination."""

    return [
        {"recipe": recipe, "layer": layer, "replacement_width": width,
         "width_ratio": width / 8192, "kl_128": short, "kl_8192": long}
        for recipe, (short, long) in values.items()
        for layer, width in COMBINATIONS
    ]


class SwiGLU8Contracts(unittest.TestCase):
    def setUp(self):
        self.settings, unused_path = swiglu_8.load_settings(CONFIG)
        self.neutral = {recipe: (1.0, 1.0) for recipe in self.settings["block_check"]["recipes"]}
        self.neutral["cur-s22"] = (1.02, 1.02)

    def test_configuration_fixes_the_decided_design(self):
        settings = self.settings
        self.assertEqual(settings["targets"], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8])
        self.assertEqual(settings["arms"], ["fitted", "slice"])
        self.assertEqual(len(settings["block_check"]["recipes"]), 9)
        self.assertEqual(settings["recovery"]["target_tokens"], 100_000_000)
        self.assertEqual(settings["recovery"]["learning_rate"], 3e-5)
        self.assertEqual(settings["allocation"]["width_ratios"][:2], [0.0625, 0.125])
        self.assertEqual(settings["acceptance"]["fit_relative"], 1e-3)
        self.assertEqual(settings["long_context"]["calibration_sequences"] * 8192, 393_216)
        reference = read_json(ROOT / "workflows/configs/model/swiglu/swiglu-7.json")
        for key in ("model", "data", "calibration", "local_fitting", "evaluation"):
            self.assertEqual(settings[key], reference[key])

    def test_configuration_rejects_changed_contracts(self):
        changes = (
            ("block_check", "recipes", "c3e-4", "learning_rate", 0.0005),
            ("acceptance", "fit_relative", None, None, 0.01),
            ("part_b", "fitting_recipe", None, None, "cos1e-3"),
            ("recovery", "learning_rate", None, None, 1e-4),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for first, second, third, fourth, value in changes:
                changed = deepcopy(self.settings)
                target = changed[first]
                if third is None:
                    target[second] = value
                else:
                    target[second][third][fourth] = value
                write_json_atomic(path, changed)
                with self.subTest(change=(first, second, third)), self.assertRaises(ValueError):
                    swiglu_8.load_settings(path)

    def test_adoption_keeps_the_current_recipe_when_nothing_qualifies(self):
        decision = swiglu_8.adoption_decision(block_rows(self.neutral), self.settings)
        self.assertEqual(decision["selected_recipe"], "cur-s21")
        self.assertEqual(decision["qualifying"], [])

    def test_adoption_decides_at_8k_and_guards_at_128(self):
        values = dict(self.neutral)
        values["c3e-4"] = (1.0, 0.90)   # about 11% better at 8K: qualifies
        values["cos3e-4"] = (1.0, 0.85)  # about 16% better: qualifies and wins
        values["ols"] = (1.2, 0.80)      # better at 8K, but about 19% worse at 128
        values["c1e-4"] = (0.80, 0.97)   # better at 128 only: below 5% at 8K
        decision = swiglu_8.adoption_decision(block_rows(values), self.settings)
        self.assertEqual(sorted(decision["qualifying"]), ["c3e-4", "cos3e-4"])
        self.assertEqual(decision["selected_recipe"], "cos3e-4")
        self.assertIn("worse", decision["recipes"]["ols"]["reasons"][0])
        self.assertFalse(decision["recipes"]["c1e-4"]["qualifies"])

    def test_adoption_requires_more_than_the_seed_variation(self):
        values = dict(self.neutral)
        values["cur-s22"] = (1.3, 1.3)   # seed variation about 26%
        values["c3e-4"] = (1.0, 1.0)     # about 13% below the seed mean only
        decision = swiglu_8.adoption_decision(block_rows(values), self.settings)
        self.assertIn("seed-to-seed", decision["recipes"]["c3e-4"]["reasons"][0])
        self.assertEqual(decision["selected_recipe"], "cur-s21")

    def test_stage_two_indicator_detects_early_best_with_falling_training_error(self):
        def row(layer, best, train):
            history = [{"phase": "full", "epoch": index + 1, "train_mse": value}
                       for index, value in enumerate(train)]
            return {"recipe": "cur-s21", "layer": layer, "width_ratio": 0.5,
                    "best_epoch": best, "history": history}

        rows = [row(3, 2, [5, 4, 3, 2, 1]), row(11, 1, [5, 4, 3, 2]), row(19, 20, [5] * 23)]
        result = swiglu_8.stage_two_indicator(rows, "cur-s21")
        self.assertEqual(result["overfitting_layers"], [3, 11])
        self.assertTrue(result["overfitting_remains"])
        self.assertFalse(swiglu_8.stage_two_indicator(rows, "ols")["applicable"])

    def test_acceptance_comparisons_use_relative_tolerances_and_stop_on_failure(self):
        passed = swiglu_8.comparison("A2", 1.0005, 1.0, 1e-3, metric="local_nmse")
        failed = swiglu_8.comparison("A2", 1.002, 1.0, 1e-3, metric="local_nmse")
        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])
        reference = {(3, 2048): {"initial_local_nmse": 0.4733, "local_nmse": 0.3397, "local_mse": 0.1275}}
        curves = {(3, 2048): {"raw_teacher_kl": 0.0269}}
        slices = [{"layer": 3, "replacement_width": 2048, "validation_128": {"nmse": 0.4733}}]
        fits = [{"recipe": "cur-s21", "layer": 3, "replacement_width": 2048,
                 "local_nmse": 0.3397, "local_mse": 0.1275, "kl_128": 0.0270}]
        self.assertTrue(all(row["passed"] for row in swiglu_8.slice_checks(slices, reference, 1e-4)))
        checks = swiglu_8.current_recipe_checks(fits, reference, curves, 1e-3)
        self.assertEqual([row["passed"] for row in checks], [True, True, False])
        with tempfile.TemporaryDirectory() as directory:
            artifact = {"schema_version": 1, "workflow": "swiglu-8", "command": "block-check",
                        "status": "running", "configuration": {}, "results": {}}
            with self.assertRaises(swiglu_8.AcceptanceFailure):
                swiglu_8.enforce(artifact, Path(directory), "A2-A3", checks)
            recorded = read_json(Path(directory) / "result.json")["results"]["acceptance"]["A2-A3"]
            self.assertFalse(recorded["passed"])

    def test_both_arms_share_one_selection_fingerprint(self):
        rankings = {1: torch.arange(8192).flip(0), 2: torch.arange(8192)}
        allocation = [
            {"layer": 1, "replacement_width": 512, "retains_dense_module": False},
            {"layer": 2, "replacement_width": 8192, "retains_dense_module": True},
        ]
        first = swiglu_8.selection_record(allocation, rankings)
        self.assertEqual(first, swiglu_8.selection_record(deepcopy(allocation), rankings))
        self.assertEqual(sorted(first["index_digests"]), ["1"])
        changed = deepcopy(allocation)
        changed[0]["replacement_width"] = 1024
        self.assertNotEqual(first["fingerprint"], swiglu_8.selection_record(changed, rankings)["fingerprint"])
        self.assertEqual(swiglu_8.selected_indices(rankings[1], 3).tolist(), [8189, 8190, 8191])

    def test_local_window_records_follow_partition_order(self):
        sizes = {name: count * 2 for name, count in self.settings["data"]["partition_batches"].items()}
        indices = list(range(sum(sizes.values())))
        partitions = swiglu_8.partition_record_indices(self.settings, indices)
        self.assertEqual({name: len(values) for name, values in partitions.items()}, sizes)
        self.assertEqual(partitions["calibration"][0], 0)
        with self.assertRaises(ValueError):
            swiglu_8.partition_record_indices(self.settings, indices[:-1])

    def test_fitting_streams_are_document_disjoint_from_kl_and_local_windows(self):
        records = [{"text": text} for text in (
            "a b c", "d e", " ", "f g h", "i j", "k l", "m n o", "p q", "r s", "t u", "v w",
        )]
        tokenizer = CharacterTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kl = build_finite_stream(iter(records), tokenizer, root / "kl.int32", 8)
            last = swiglu_8.replay_kl_records(records, tokenizer, np.fromfile(root / "kl.int32", dtype=np.int32))
            self.assertEqual((kl["documents_consumed"], last), (3, 3))
            built = swiglu_8.build_fitting_streams(
                records, tokenizer, last + 1, {5}, {"calibration": root / "c.int32", "validation": root / "v.int32"},
                2, {"calibration": 2, "validation": 1},
            )
            calibration = np.fromfile(root / "c.int32", dtype=np.int32).tolist()
            validation = np.fromfile(root / "v.int32", dtype=np.int32).tolist()
        self.assertEqual(calibration, [ord("i"), ord("j"), 0, ord("m")])
        self.assertEqual(built["calibration"]["first_record"], 4)
        self.assertEqual(built["calibration"]["excluded_records_skipped"], 1)
        self.assertEqual(validation, [ord("p"), ord("q")])
        self.assertGreater(built["validation"]["first_record"], built["calibration"]["last_record"])
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            swiglu_8.replay_kl_records(records[1:], tokenizer, np.asarray([97, 98, 99, 0], dtype=np.int32))

    def test_monitoring_checks_pair_rows_by_requested_tokens(self):
        reference = [
            {"actual_tokens": 0, "requested_tokens": [0], "recovery_validation_kl": 0.2,
             "wikitext_validation": {"perplexity": 20.0}},
            {"actual_tokens": 75_005_952, "requested_tokens": [75_000_000], "recovery_validation_kl": 0.1},
        ]
        row = deepcopy(reference[1])
        row["recovery_validation_kl"] = 0.1002
        self.assertEqual([check["check"] for check in swiglu_8.monitoring_checks(reference[0], reference,
                                                                                    swiglu_8.ACCEPTANCE)],
                         ["B1", "B1"])
        self.assertFalse(swiglu_8.monitoring_checks(row, reference, swiglu_8.ACCEPTANCE)[0]["passed"])
        row["wikitext_validation"] = {"perplexity": 10.0}
        with self.assertRaisesRegex(ValueError, "WikiText"):
            swiglu_8.monitoring_checks(row, reference, swiglu_8.ACCEPTANCE)

    def test_swiglu7_persist_keeps_its_run_record_and_names_the_artifact_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for workflow in ("swiglu-7", "swiglu-8"):
                artifact = {"schema_version": 1, "workflow": workflow, "command": "train",
                            "status": "running", "configuration": {"seed": 21}, "results": {}}
                swiglu_7.persist(root, artifact, "stage")
                run = read_json(root / "run.json")
                self.assertEqual((run["workflow"], run["schema_version"], run["stage"]),
                                 (workflow, 1, "stage"))

    def test_perun_job_maps_sixteen_tasks_and_stages_without_rsyncignore(self):
        source = SBATCH.read_text(encoding="utf-8")
        arms = re.search(r"^ARMS=\(([^)]*)\)", source, flags=re.MULTILINE).group(1).split()
        targets = re.search(r"^TARGETS=\(([^)]*)\)", source, flags=re.MULTILINE).group(1).split()
        self.assertEqual(arms, list(swiglu_8.ARMS))
        self.assertEqual([float(value) for value in targets], list(swiglu_8.TARGETS))
        self.assertIn('ARM="${ARMS[$((TASK_ID / 8))]}"', source)
        self.assertIn('TARGET="${TARGETS[$((TASK_ID % 8))]}"', source)
        self.assertIn("-m workflows.runs.model.swiglu.swiglu_8", source)
        self.assertIn('PROJECT_RESULTS_ROOT="$PROJECT_ROOT/perun-results/swiglu-8"', source)
        self.assertIn("trap finalize_job EXIT", source)
        self.assertIn("rsync -rcn", source)
        self.assertNotIn(".rsyncignore", source)
        self.assertTrue(callable(swiglu_8.main))


if __name__ == "__main__":
    unittest.main()
