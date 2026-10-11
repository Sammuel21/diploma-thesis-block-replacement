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
NOTEBOOK = ROOT / "notebooks/model/swiglu/swiglu-8.ipynb"
COMBINATIONS = [(layer, width) for layer in (3, 11, 19) for width in (2048, 4096)]


class CharacterTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens, return_attention_mask):
        return types.SimpleNamespace(input_ids=[ord(word[0]) for word in text.split()])


def block_rows(settings, values):
    """Build fit rows from {setting: [(kl_128, kl_8192) per seed]} at every combination.

    Settings without values score 1.0 at both contexts for every seed.
    """

    rows = []
    for setting in settings["block_check"]["settings"]:
        seeds = swiglu_8.expected_seeds(setting, settings)
        for seed, (short, long) in zip(seeds, values.get(setting, [(1.0, 1.0)] * len(seeds))):
            for layer, width in COMBINATIONS:
                rows.append({"setting": setting, "seed": seed, "layer": layer, "replacement_width": width,
                             "width_ratio": width / 8192, "kl_128": short, "kl_8192": long})
    return rows


def block_check_tasks(settings):
    """Build one completed synthetic task artifact per Part A setting."""

    rows = block_rows(settings, {})
    tasks = []
    for index, setting in enumerate(settings["block_check"]["settings"]):
        anchor = setting == swiglu_8.CURRENT_SETTING
        passed = {"checks": [], "passed": True}
        tasks.append({
            "workflow": "swiglu-8", "command": "block-check", "status": "completed",
            "identity": {"kind": "block-check-task", "task_index": index, "setting": setting},
            "configuration": deepcopy(settings), "configuration_sha256": "c" * 64, "code_hashes": {"runner": "h"},
            "environment": {"python": "3.12.14", "packages": {"torch": "2.11.0+cu128"}, "cuda_runtime": "12.8",
                            "gpu": "NVIDIA H200", "platform": f"node-{index}"},
            "created_at_utc": f"2026-10-12T0{index % 10}:00:00+00:00", "updated_at_utc": "2026-10-12T12:00:00+00:00",
            "run_fingerprint": f"{index:064d}",
            "results": {
                "long_context_data": {"kl_stream": {"sha256": "k" * 64}, "selection_stream": {"sha256": "s" * 64}},
                "ranking_digests": {"3": "r", "11": "r", "19": "r"},
                "fits": [row for row in rows if row["setting"] == setting],
                **({"slices": [{"layer": layer, "replacement_width": width} for layer, width in COMBINATIONS],
                    "acceptance": {"P5": passed, "A1": passed, "A2-A3": passed}} if anchor else {"slices": []}),
            },
        })
    return tasks


class SwiGLU8Contracts(unittest.TestCase):
    def setUp(self):
        self.settings, unused_path = swiglu_8.load_settings(CONFIG)

    def test_block_check_runs_one_array_task_per_setting(self):
        grid = list(self.settings["block_check"]["settings"])
        self.assertEqual([swiglu_8.block_check_setting(self.settings, index) for index in range(len(grid))], grid)
        self.assertEqual(swiglu_8.block_check_setting(self.settings, 0), swiglu_8.CURRENT_SETTING)
        self.assertEqual(swiglu_8.block_check_output_name(3), "block-check-001-task-03")
        for index in (-1, len(grid)):
            with self.assertRaises(ValueError):
                swiglu_8.block_check_setting(self.settings, index)

    def test_merge_joins_identical_tasks_into_the_part_a_result(self):
        tasks = block_check_tasks(self.settings)
        records = [{"output": swiglu_8.block_check_output_name(index), "result_sha256": "x" * 64}
                   for index in range(len(tasks))]
        merged = swiglu_8.merge_block_check(tasks, self.settings, records)
        self.assertEqual((merged["command"], merged["status"]), ("block-check", "completed"))
        self.assertEqual(len(merged["results"]["fits"]), 246)
        self.assertEqual(set(merged["results"]["acceptance"]), {"P5", "A1", "A2-A3"})
        self.assertEqual(merged["results"]["adoption"]["selected_recipe"], swiglu_8.CURRENT_SETTING)
        self.assertEqual([task["setting"] for task in merged["results"]["tasks"]],
                         list(self.settings["block_check"]["settings"]))
        self.assertEqual(merged["created_at_utc"], "2026-10-12T00:00:00+00:00")

    def test_merge_command_writes_once_and_is_safe_to_rerun(self):
        tasks = block_check_tasks(self.settings)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, task in enumerate(tasks):
                write_json_atomic(root / swiglu_8.block_check_output_name(index) / "result.json", task)
            args = types.SimpleNamespace(results_root=root, output_dir=None)
            swiglu_8.merge_block_check_command(args, self.settings)
            merged = read_json(root / "block-check-001" / "result.json")
            self.assertEqual(len(merged["results"]["fits"]), 246)
            self.assertTrue((root / "block-check-001" / "run.json").is_file())
            swiglu_8.merge_block_check_command(args, self.settings)   # identical: no change
            tasks[3]["results"]["fits"][0]["kl_8192"] = 0.5
            write_json_atomic(root / swiglu_8.block_check_output_name(3) / "result.json", tasks[3])
            with self.assertRaisesRegex(ValueError, "different merged block check"):
                swiglu_8.merge_block_check_command(args, self.settings)
            (root / f"{swiglu_8.block_check_output_name(5)}.perun-lock").mkdir()
            with self.assertRaisesRegex(ValueError, "did not finish"):
                swiglu_8.merge_block_check_command(args, self.settings)

    def test_merge_refuses_tasks_that_differ_or_are_missing(self):
        records = [{} for unused in range(15)]
        changes = {
            "different data": lambda tasks: tasks[5]["results"]["long_context_data"]["kl_stream"].update(sha256="z"),
            "different rankings": lambda tasks: tasks[7]["results"]["ranking_digests"].update({"3": "other"}),
            "different software": lambda tasks: tasks[2]["environment"]["packages"].update(torch="2.12.0"),
            "unfinished task": lambda tasks: tasks[9].update(status="failed"),
            "missing fits": lambda tasks: tasks[4]["results"]["fits"].pop(),
            "failed anchor check": lambda tasks: tasks[0]["results"]["acceptance"]["A1"].update(passed=False),
            "missing task": lambda tasks: tasks.pop(),
        }
        for name, change in changes.items():
            tasks = block_check_tasks(self.settings)
            change(tasks)
            with self.subTest(change=name), self.assertRaises(ValueError):
                swiglu_8.merge_block_check(tasks, self.settings, records)

    def test_configuration_fixes_the_decided_design(self):
        settings = self.settings
        self.assertEqual(settings["targets"], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8])
        self.assertEqual(settings["arms"], ["fitted", "slice"])
        self.assertEqual(len(settings["block_check"]["settings"]), 15)
        self.assertEqual(settings["block_check"]["seeds"], [21, 22, 23])
        self.assertEqual(settings["long_context"]["selection_sequences"], settings["long_context"]["kl_sequences"])
        self.assertEqual(settings["recovery"]["target_tokens"], 100_000_000)
        self.assertEqual(settings["recovery"]["learning_rate"], 3e-5)
        self.assertEqual(settings["allocation"]["width_ratios"][:2], [0.0625, 0.125])
        self.assertEqual(settings["acceptance"]["fit_relative"], 1e-3)
        self.assertEqual(settings["long_context"]["calibration_sequences"] * 8192, 393_216)
        reference = read_json(ROOT / "workflows/configs/model/swiglu/swiglu-7.json")
        for key in ("model", "data", "calibration", "local_fitting", "evaluation"):
            self.assertEqual(settings[key], reference[key])

    def test_part_a_grid_crosses_context_schedule_and_rate(self):
        grid = self.settings["block_check"]["settings"]
        gradient = {name: value for name, value in grid.items()
                    if value["method"] == "gradient" and value["batch_size"] == 2048}
        self.assertEqual(
            {(value["calibration_context"], value["scheduler"], value["learning_rate"]) for value in gradient.values()},
            {(context, scheduler, rate) for context in (128, 8192) for scheduler in ("constant", "cosine")
             for rate in (1e-3, 3e-4, 1e-4)},
        )
        for value in gradient.values():
            expected = (32, 32) if value["scheduler"] == "cosine" else (64, 3)
            self.assertEqual((value["max_epochs"], value["early_stopping_patience"]), expected)
        current = grid[swiglu_8.CURRENT_SETTING]
        self.assertEqual({key: current[key] for key in ("learning_rate", "scheduler", "batch_size", "max_epochs",
                                                        "early_stopping_patience")},
                         {key: self.settings["local_fitting"][key] for key in ("learning_rate", "scheduler",
                                                                                "batch_size", "max_epochs",
                                                                                "early_stopping_patience")})
        self.assertEqual(grid["c1e-3-128-b256"], {**current, "batch_size": 256})
        self.assertEqual({grid[name]["calibration_context"] for name in ("ols-128", "ols-8k")}, {128, 8192})
        self.assertEqual(swiglu_8.expected_seeds("ols-8k", self.settings), [None])
        self.assertEqual(sum(len(swiglu_8.expected_seeds(name, self.settings)) for name in grid) * 6, 246)

    def test_settings_reach_the_fitting_view_without_changing_the_context(self):
        context = types.SimpleNamespace(settings=deepcopy(self.settings))
        view = swiglu_8.recipe_context(context, swiglu_8.PART_A_SETTINGS["cos3e-4-8k"], 22)
        self.assertEqual(
            [view.settings["local_fitting"][key] for key in ("learning_rate", "scheduler", "batch_size",
                                                              "max_epochs", "early_stopping_patience")],
            [0.0003, "cosine", 2048, 32, 32],
        )
        self.assertEqual(view.settings["seed"], 22)
        self.assertEqual(context.settings, self.settings)
        self.assertEqual(swiglu_8.recipe_context(context, swiglu_8.PART_A_SETTINGS["ols-8k"], None).settings,
                         self.settings)
        self.assertEqual(swiglu_8.fit_context_kind("c1e-3-128", 21), "dense")
        self.assertEqual(swiglu_8.fit_context_kind("c1e-3-128", 22), "c1e-3-128-s22")
        self.assertEqual(swiglu_8.fit_context_kind("ols-8k", None), "ols-8k")

    def test_configuration_rejects_changed_contracts(self):
        changes = (
            ("block_check", "settings", "c3e-4-128", "learning_rate", 0.0005),
            ("block_check", "seeds", None, None, [21, 22]),
            ("long_context", "selection_sequences", None, None, 12),
            ("acceptance", "fit_relative", None, None, 0.01),
            ("part_b", "fitting_recipe", None, None, "cos1e-3-128"),
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

    def test_adoption_keeps_the_current_setting_when_nothing_qualifies(self):
        decision = swiglu_8.adoption_decision(block_rows(self.settings, {}), self.settings)
        self.assertEqual(decision["selected_recipe"], "c1e-3-128")
        self.assertEqual(decision["qualifying"], [])

    def test_adoption_compares_seed_means_decides_at_8k_and_guards_at_128(self):
        values = {
            "c3e-4-128": [(1.0, 0.90)] * 3,   # 10% better at 8K: qualifies
            "cos3e-4-8k": [(1.0, 0.85)] * 3,  # 15% better: qualifies and wins
            "ols-128": [(1.2, 0.80)],         # better at 8K, but 20% worse at 128
            "c1e-4-128": [(0.80, 0.97)] * 3,  # better at 128 only: below 5% at 8K
        }
        decision = swiglu_8.adoption_decision(block_rows(self.settings, values), self.settings)
        self.assertEqual(sorted(decision["qualifying"]), ["c3e-4-128", "cos3e-4-8k"])
        self.assertEqual(decision["selected_recipe"], "cos3e-4-8k")
        self.assertIn("worse", decision["settings"]["ols-128"]["reasons"][0])
        self.assertFalse(decision["settings"]["c1e-4-128"]["qualifies"])

    def test_adoption_requires_more_than_the_seed_noise(self):
        values = {
            "c1e-3-128": [(1.0, 1.0), (1.0, 1.2), (1.0, 0.8)],     # mean 1.0, sample sd 0.2
            "c3e-4-128": [(1.0, 0.82), (1.0, 0.92), (1.0, 0.72)],  # 18% better, sd 0.1: noise 30%
            "c1e-3-8k": [(1.0, 0.6)] * 3,                           # 40% better, noise 20%
        }
        decision = swiglu_8.adoption_decision(block_rows(self.settings, values), self.settings)
        noisy = decision["settings"]["c3e-4-128"]
        self.assertAlmostEqual(noisy["mean_reduction"]["8192"], 0.18)
        self.assertAlmostEqual(noisy["mean_seed_noise"], 0.3)
        self.assertEqual(noisy["reasons"], ["mean reduction is not above the seed noise"])
        self.assertEqual(decision["selected_recipe"], "c1e-3-8k")

    def test_adoption_refuses_a_setting_with_missing_seeds(self):
        rows = [row for row in block_rows(self.settings, {})
                if not (row["setting"] == "c3e-4-8k" and row["seed"] == 23)]
        with self.assertRaisesRegex(ValueError, "lacks seeds"):
            swiglu_8.adoption_decision(rows, self.settings)

    def test_stage_two_indicator_counts_a_layer_when_most_seeds_overfit(self):
        def row(layer, seed, best, train):
            history = [{"phase": "full", "epoch": index + 1, "train_mse": value}
                       for index, value in enumerate(train)]
            return {"setting": "c1e-3-128", "seed": seed, "layer": layer, "width_ratio": 0.5,
                    "best_epoch": best, "history": history}

        overfit = (2, [5, 4, 3, 2, 1])
        settled = (20, [5] * 23)
        pattern = {3: (overfit, overfit, overfit), 11: (overfit, overfit, settled), 19: (overfit, settled, settled)}
        rows = [row(layer, seed, *case) for layer, cases in pattern.items()
                for seed, case in zip((21, 22, 23), cases)]
        result = swiglu_8.stage_two_indicator(rows, "c1e-3-128")
        self.assertEqual(result["overfitting_layers"], [3, 11])
        self.assertTrue(result["overfitting_remains"])
        least_squares = [{"setting": "ols-128", "seed": None, "layer": 3, "width_ratio": 0.5, "best_epoch": 0,
                          "history": []}]
        self.assertFalse(swiglu_8.stage_two_indicator(least_squares, "ols-128")["applicable"])

    def test_acceptance_comparisons_use_relative_tolerances_and_stop_on_failure(self):
        passed = swiglu_8.comparison("A2", 1.0005, 1.0, 1e-3, metric="local_nmse")
        failed = swiglu_8.comparison("A2", 1.002, 1.0, 1e-3, metric="local_nmse")
        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])
        reference = {(3, 2048): {"initial_local_nmse": 0.4733, "local_nmse": 0.3397, "local_mse": 0.1275}}
        curves = {(3, 2048): {"raw_teacher_kl": 0.0269}}
        slices = [{"layer": 3, "replacement_width": 2048, "validation_128": {"nmse": 0.4733}}]
        anchor = {"layer": 3, "replacement_width": 2048, "local_nmse": 0.3397, "local_mse": 0.1275,
                  "kl_128": 0.0270}
        fits = [{"setting": "c1e-3-128", "seed": 21, **anchor},
                {"setting": "c1e-3-128", "seed": 22, **anchor, "local_nmse": 0.5}]
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

    def test_8k_streams_are_document_disjoint_from_kl_local_windows_and_each_other(self):
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
                records, tokenizer, last + 1, {5},
                {"calibration": root / "c.int32", "validation": root / "v.int32", "selection": root / "s.int32"},
                2, {"calibration": 2, "validation": 1, "selection": 1},
            )
            calibration = np.fromfile(root / "c.int32", dtype=np.int32).tolist()
            validation = np.fromfile(root / "v.int32", dtype=np.int32).tolist()
            selection = np.fromfile(root / "s.int32", dtype=np.int32).tolist()
        self.assertEqual(calibration, [ord("i"), ord("j"), 0, ord("m")])
        self.assertEqual(built["calibration"]["first_record"], 4)
        self.assertEqual(built["calibration"]["excluded_records_skipped"], 1)
        self.assertEqual(validation, [ord("p"), ord("q")])
        self.assertEqual(selection, [ord("r"), ord("s")])
        self.assertEqual(list(built), ["calibration", "validation", "selection"])
        self.assertGreater(built["validation"]["first_record"], built["calibration"]["last_record"])
        self.assertGreater(built["selection"]["first_record"], built["validation"]["last_record"])
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
        last_task = len(swiglu_8.PART_A_SETTINGS) - 1
        self.assertIn(f"(( TASK_ID >= 0 && TASK_ID <= {last_task} ))", source)
        self.assertIn("OUTPUT_NAME=\"$(printf 'block-check-001-task-%02d' \"$TASK_ID\")\"", source)
        self.assertIn('--task-index "$TASK_ID"', source)
        self.assertIn("-m workflows.runs.model.swiglu.swiglu_8", source)
        self.assertIn('PROJECT_RESULTS_ROOT="$PROJECT_ROOT/perun-results/swiglu-8"', source)
        self.assertIn("trap finalize_job EXIT", source)
        self.assertIn("rsync -rcn", source)
        self.assertNotIn(".rsyncignore", source)
        self.assertTrue(callable(swiglu_8.main))

    def test_notebook_is_load_only_and_uses_the_decided_interpretation(self):
        notebook = read_json(NOTEBOOK)
        code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        self.assertNotRegex(code, r"(?m)^\s*(import|from)\s+(torch|transformers|mlp_replacement)\b")
        self.assertTrue(all(
            cell.get("execution_count") is None and cell.get("outputs") == []
            for cell in notebook["cells"] if cell["cell_type"] == "code"
        ))
        constants = dict(re.findall(r"(?m)^(GAP_THRESHOLD|STABILITY_THRESHOLD|TARGETS|ARMS) = (.+)$", code))
        self.assertEqual(float(constants["GAP_THRESHOLD"]), 0.02)
        self.assertEqual(float(constants["STABILITY_THRESHOLD"]), 0.01)
        self.assertEqual(eval(constants["TARGETS"]), swiglu_8.TARGETS)
        self.assertEqual(eval(constants["ARMS"]), swiglu_8.ARMS)


if __name__ == "__main__":
    unittest.main()
