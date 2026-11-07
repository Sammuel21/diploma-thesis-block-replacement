"""Focused CPU/static contracts for Recovery-1; no pretrained model or dataset is loaded."""

import copy
import json
import math
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

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from mlp_replacement.artifacts import file_digest, read_json
from mlp_replacement.compression.continuation import recover_exact_segment, segment_schedule
from mlp_replacement.compression.recovery import recover_trainable_by_tokens, token_learning_rate_multiplier
from workflows.runs.model.recovery import recovery_1 as r
from workflows.runs.model.swiglu.swiglu_7 import parameter_names

CONFIG = ROOT / "workflows/configs/model/recovery/recovery-1.json"
SBATCH = ROOT / "workflows/jobs/perun/recovery/recovery_1.sbatch"
LOCAL = ROOT / "data/results/workflows/model"
RULE = {"bar": 0.035, "reference_requests": [75, 80, 90, 100]}


def row(tokens, selection, legacy=None):
    return {"actual_tokens": tokens, "requested_tokens": [tokens], "selection_kl": selection,
            "legacy_kl": selection if legacy is None else legacy}


def trunk(values, legacy=None):
    tokens = [0, 75, 80, 90, 100]
    legacy = values if legacy is None else legacy
    return [row(token, value, other) for token, value, other in zip(tokens, values, legacy)]


def branches(values, legacy=None):
    legacy = values if legacy is None else legacy
    return {str(percent): {"history": [row(100, value, other)]}
            for percent, value, other in zip((10, 20, 50), values, legacy)}


class MultiplierContracts(unittest.TestCase):
    def test_stable_until_the_decay_start_then_one_minus_sqrt_to_the_final_ratio(self):
        args = dict(schedule_tokens=100, scheduler="wsd_one_minus_sqrt", final_lr_ratio=0.0, decay_start_tokens=60)
        self.assertEqual(token_learning_rate_multiplier(1, **args), 1.0)
        self.assertEqual(token_learning_rate_multiplier(60, **args), 1.0)
        self.assertAlmostEqual(token_learning_rate_multiplier(70, **args), 1.0 - math.sqrt(0.25))
        self.assertAlmostEqual(token_learning_rate_multiplier(100, **args), 0.0)
        ratio = token_learning_rate_multiplier(100, **{**args, "final_lr_ratio": 0.1})
        self.assertAlmostEqual(ratio, 0.1)

    def test_refuses_warmup_a_missing_start_and_a_start_on_another_schedule(self):
        base = dict(schedule_tokens=100, scheduler="wsd_one_minus_sqrt", decay_start_tokens=60)
        with self.assertRaises(ValueError):
            token_learning_rate_multiplier(10, warmup_fraction=0.01, **base)
        with self.assertRaises(ValueError):
            token_learning_rate_multiplier(10, **{**base, "decay_start_tokens": None})
        with self.assertRaises(ValueError):
            token_learning_rate_multiplier(10, **{**base, "decay_start_tokens": 100})
        with self.assertRaises(ValueError):
            token_learning_rate_multiplier(10, 100, "constant", decay_start_tokens=60)

    def test_existing_schedules_are_unchanged(self):
        self.assertEqual(token_learning_rate_multiplier(5, 100, "constant"), 1.0)
        cosine = token_learning_rate_multiplier(50, 100, "warmup_cosine", 0.0, 0.1)
        self.assertAlmostEqual(cosine, 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * 0.5)))


def tiny_model(seed):
    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=16)
    return LlamaForCausalLM(config).eval()


TOKENS = torch.randint(0, 32, (64,), generator=torch.Generator().manual_seed(5))


def batch_at(offset, count):
    ids = TOKENS[offset:offset + count].view(1, count)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def mlp_groups(model):
    parameters = list(model.model.layers[0].mlp.parameters())
    return [{"name": "replacements", "parameters": parameters, "parameter_names": parameter_names(model, parameters),
             "parameter_count": sum(value.numel() for value in parameters), "learning_rate": 1e-2,
             "weight_decay": 0.0}], [model.model.layers[0].mlp]


def mlp_state(model):
    return [value.detach().clone() for value in model.model.layers[0].mlp.parameters()]


def loop_arguments(model, teacher):
    groups, modules = mlp_groups(model)
    return dict(
        student=model, teacher=teacher,
        parameter_groups=[{key: group[key] for key in ("name", "parameters", "learning_rate", "weight_decay")}
                          for group in groups],
        train_modules=modules, batch_at=batch_at, microbatch_tokens=4, accumulation_steps=1,
        temperature=1.0, ce_weight=0.0, device=torch.device("cpu"), autocast_dtype=torch.bfloat16,
        optimizer_backend="single_tensor",
    )


class TrunkAndBranchContracts(unittest.TestCase):
    """The branched decay must equal one uninterrupted stable-decay run on the same data."""

    budget = 40

    def task(self, model, teacher, output_dir):
        groups, modules = mlp_groups(model)
        recorded = {"results": {"recovery": {"validation_history": [
            {"requested_tokens": [value], "actual_tokens": value, "recovery_validation_kl": 1.0}
            for value in (0, 10, 20, 30, 40)]}}}
        settings = {"budget_tokens": self.budget, "rows": {"trunk_requests": [10, 20, 30, 40]},
                    "schedule": {"scheduler": "wsd_one_minus_sqrt", "final_lr_ratio": 0.0, "decay_percent": [50]},
                    "acceptance": {"review_relative": 0.01}}
        recipe = {"effective_batch_tokens": 4, "sequence_length": 4, "microbatch_sequences": 1,
                  "gradient_accumulation_steps": 1, "temperature": 1.0, "ce_weight": 0.0, "scheduler": "constant",
                  "final_lr_ratio": 1.0, "optimizer_backend": "single_tensor"}
        artifact = {"schema_version": 1, "workflow": "recovery-1", "command": "train", "status": "running",
                    "configuration": settings, "results": {"trunk": {"history": [row(0, 1.0)]}}}
        return SimpleNamespace(settings=settings, recipe=recipe, budget=self.budget, model=model, teacher=teacher,
                               groups=groups, train_modules=modules, batch_at=batch_at, device=torch.device("cpu"),
                               family={"name": "toy", "kind": "order-b"}, recorded={"recorded_100m": recorded},
                               artifact=artifact, output_dir=output_dir)

    def fake_score(self, task, tokens, requested, updates, seconds, mean_train_kl, learning_rates):
        value = float(sum(value.detach().sum() for value in task.model.model.layers[0].mlp.parameters()))
        return {"actual_tokens": tokens, "requested_tokens": list(requested), "optimizer_updates": updates,
                "learning_rates": list(learning_rates), "selection_kl": value, "legacy_kl": value}

    def test_branch_from_the_trunk_snapshot_equals_an_uninterrupted_decay(self):
        teacher = tiny_model(1)
        student = tiny_model(2)
        reference = copy.deepcopy(student)
        settings = {"budget_tokens": self.budget, "schedule": {"decay_percent": [50]}}
        starts = r.decay_starts(settings, {"effective_batch_tokens": 4})
        self.assertEqual(starts[50], {"requested": 20, "tokens": 20, "midpoint": 30})
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(r, "score", self.fake_score):
            task = self.task(student, teacher, Path(directory))
            snapshots = r.train_trunk(task, starts)
            constant_end = mlp_state(student)
            r.train_branch(task, 50, starts[50], snapshots.pop(50))
            branch = task.artifact["results"]["branches"]["50"]
        recover_trainable_by_tokens(
            target_tokens=self.budget, schedule_tokens=self.budget, scheduler="wsd_one_minus_sqrt",
            warmup_fraction=0.0, final_lr_ratio=0.0, decay_start_tokens=20, **loop_arguments(reference, teacher))
        for left, right in zip(mlp_state(student), mlp_state(reference)):
            self.assertTrue(torch.equal(left, right))
        self.assertFalse(all(torch.equal(left, right) for left, right in zip(constant_end, mlp_state(student))))
        # The midpoint request (30) rounds up to the update boundary at 32, as in the real schedule.
        rates = {row["actual_tokens"]: row["learning_rates"][0] for row in branch["history"]}
        self.assertEqual(sorted(rates), [32, 40])
        self.assertAlmostEqual(rates[32], 1e-2 * (1.0 - math.sqrt((32 - 20) / (40 - 20))))
        self.assertEqual(rates[40], 0.0)

    def test_the_direct_constant_call_equals_the_segment_path_the_recorded_runs_used(self):
        teacher = tiny_model(1)
        direct = tiny_model(3)
        segment = copy.deepcopy(direct)
        schedule = segment_schedule(0, self.budget, [10, 20, 30, 40], 4)
        recover_trainable_by_tokens(target_tokens=self.budget, schedule_tokens=self.budget, scheduler="constant",
                                    warmup_fraction=0.0, final_lr_ratio=1.0, start_tokens=0,
                                    checkpoint_schedule=schedule, on_checkpoint=None,
                                    **loop_arguments(direct, teacher))
        recover_exact_segment(origin=0, end=self.budget, cursor=0, schedule=schedule, batch_at=batch_at,
                              on_checkpoint=lambda *unused: None, scheduler="constant", warmup_fraction=0.0,
                              final_lr_ratio=1.0,
                              **{key: value for key, value in loop_arguments(segment, teacher).items()
                                 if key != "batch_at"})
        for left, right in zip(mlp_state(direct), mlp_state(segment)):
            self.assertTrue(torch.equal(left, right))


class ConfigurationContracts(unittest.TestCase):
    def setUp(self):
        self.settings, self.stacking, self.dense, unused_path = r.load_settings(CONFIG)

    def test_decay_starts_sit_on_update_boundaries_and_on_trunk_rows(self):
        recipe = self.stacking["recovery"]
        starts = r.decay_starts(self.settings, recipe)
        schedule = dict(segment_schedule(0, 100_000_000, self.settings["rows"]["trunk_requests"], 8192))
        self.assertEqual({percent: row["tokens"] for percent, row in starts.items()},
                         {10: 90_005_504, 20: 80_003_072, 50: 50_003_968})
        for row in starts.values():
            self.assertEqual(row["tokens"] % 8192, 0)
            self.assertIn(row["tokens"], schedule)
            self.assertEqual(schedule[row["tokens"]], (row["requested"],))

    def test_the_recipe_is_the_constant_3e5_of_the_recorded_runs(self):
        recipe = self.stacking["recovery"]
        self.assertEqual((recipe["learning_rate"], recipe["scheduler"], recipe["effective_batch_tokens"]),
                         (3e-05, "constant", 8192))
        self.assertEqual(self.settings["schedule"]["warmup_fraction"], 0.0)
        self.assertEqual(self.settings["schedule"]["final_lr_ratio"], 0.0)

    def refused(self, change):
        settings = read_json(CONFIG)
        change(settings)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recovery-1.json"
            path.write_text(json.dumps(settings), encoding="utf-8")
            with self.assertRaises(ValueError):
                r.load_settings(path)

    def test_refuses_a_new_peak_warmup_reordered_families_or_a_missing_reference_row(self):
        self.refused(lambda value: value["schedule"].update(learning_rate=1e-4))
        self.refused(lambda value: value["schedule"].update(warmup_fraction=0.01))
        self.refused(lambda value: value["families"].reverse())
        self.refused(lambda value: value["rows"].update(trunk_requests=[10_000_000, 100_000_000]))
        self.refused(lambda value: value["streams"].update(selection_sha256="0" * 63))

    def test_pins_match_the_local_artifacts(self):
        checked = 0
        for family in self.settings["families"]:
            for key in ("recorded_100m", "constant_1b", "selection_token_zero"):
                pin = family[key]
                path = None if pin is None else LOCAL / pin["workflow"] / pin["run"] / "result.json"
                if path is not None and path.is_file():
                    self.assertEqual(file_digest(path), pin["result_sha256"], path)
                    checked += 1
        prepared = LOCAL / "swiglu-8/prepare-001/result.json"
        block = LOCAL / "swiglu-8/block-check-001/result.json"
        if not (prepared.is_file() and block.is_file()) or not checked:
            self.skipTest("Local result copies are not present")
        streams = self.settings["streams"]
        results = read_json(prepared)["results"]
        self.assertEqual(results["stream"]["sha256"], streams["recovery_sha256"])
        self.assertEqual(results["long_context_data"]["kl_stream"]["sha256"], streams["final_kl_sha256"])
        self.assertEqual(results["monitoring_evaluation"]["sha256"], streams["monitoring_sha256"])
        selection = read_json(block)["results"]["long_context_data"]["selection_stream"]
        self.assertEqual(selection["sha256"], streams["selection_sha256"])

    def test_job_array_order_matches_the_families(self):
        text = SBATCH.read_text(encoding="utf-8")
        names = re.search(r"FAMILY_NAMES=\(([^)]*)\)", text).group(1).split()
        self.assertEqual(names, [family["name"] for family in self.settings["families"]])
        self.assertIn("workflows.runs.model.recovery.recovery_1 train", text)
        self.assertNotIn("\r\n", text)


class WinnerRuleContracts(unittest.TestCase):
    def test_no_decay_beyond_the_bar_keeps_constant(self):
        decision = r.family_decision(trunk([2.0, 1.0, 1.0, 1.0, 1.0]), branches([0.97, 0.98, 1.1]), RULE)
        self.assertIsNone(decision["decay_percent"])
        self.assertEqual([row["qualifies"] for row in decision["candidates"]], [False, False, False])

    def test_the_shortest_decay_within_the_bar_of_the_best_qualifier_wins(self):
        decision = r.family_decision(trunk([2.0, 1.0, 1.0, 1.0, 1.0]), branches([0.95, 0.93, 0.90]), RULE)
        self.assertEqual(decision["decay_percent"], 20)
        decision = r.family_decision(trunk([2.0, 1.0, 1.0, 1.0, 1.0]), branches([0.95, 0.99, 0.99]), RULE)
        self.assertEqual(decision["decay_percent"], 10)

    def test_a_constant_spike_does_not_lower_the_bar(self):
        decision = r.family_decision(trunk([2.0, 1.0, 1.01, 1.02, 1.5]), branches([0.98, 0.98, 0.98]), RULE)
        self.assertEqual(decision["reference"], 1.0)
        self.assertEqual(decision["reference_tokens"], 75)
        self.assertIsNone(decision["decay_percent"])

    def test_a_missing_reference_row_is_refused(self):
        rows = [row for row in trunk([2.0, 1.0, 1.0, 1.0, 1.0]) if row["actual_tokens"] != 80]
        with self.assertRaises(ValueError):
            r.family_decision(rows, branches([0.9, 0.9, 0.9]), RULE)

    def test_token_equivalent_interpolates_and_reports_never(self):
        curve = [(0, 1.0), (25, 0.5), (50, 0.4)]
        self.assertAlmostEqual(r.token_equivalent(curve, 0.45), 37.5)
        self.assertEqual(r.token_equivalent(curve, 1.2), 0.0)
        self.assertIsNone(r.token_equivalent(curve, 0.3))

    def test_summary_holds_flagged_families_and_writes_only_after_review(self):
        settings = read_json(CONFIG)
        trunk_rows = [row(tokens, 1.0) for tokens in (0, 10_000_000, 75_000_000, 80_000_000, 90_000_000, 100_000_000)]
        for value, tokens in zip(trunk_rows, (0, 10_000_000, 75_000_000, 80_000_000, 90_000_000, 100_000_000)):
            value["requested_tokens"] = [tokens]
        family_branches = {str(percent): {"history": [row(100_000_000, value)]}
                           for percent, value in ((10, 0.99), (20, 0.95), (50, 0.94))}
        constant = {"results": {"recovery": {
            "history": [{"tokens_seen": 0, "kl": {"kl": 2.0}}, {"tokens_seen": 500, "kl": {"kl": 0.9}}],
            "validation_history": [{"actual_tokens": 0, "recovery_validation_kl": 2.0},
                                   {"actual_tokens": 500, "recovery_validation_kl": 0.9}],
        }}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, family in enumerate(settings["families"]):
                passed = index != 1
                artifact = {"schema_version": 1, "workflow": "recovery-1", "command": "train", "status": "completed",
                            "configuration": settings, "code_hashes": {"same": "code"},
                            "updated_at_utc": "2026-11-08T00:00:00+00:00",
                            "results": {"trunk": {"history": trunk_rows}, "branches": family_branches,
                                        "acceptance": {"trunk-curve": {"passed": passed, "checks": [
                                            {"passed": passed, "relative_difference": 0.004 if passed else 0.02}]}}}}
                (root / r.output_name(family)).mkdir()
                (root / r.output_name(family) / "result.json").write_text(json.dumps(artifact), encoding="utf-8")
            arguments = SimpleNamespace(results_root=root, source_root=root, accept_flags=[])
            with mock.patch.object(r, "pinned_json", return_value=constant), mock.patch("builtins.print"):
                r.summarize(arguments, settings)
                self.assertFalse((root / "summary-001").exists())
                arguments.accept_flags = ["b-0.2-int4"]
                r.summarize(arguments, settings)
            summary = read_json(root / "summary-001" / "result.json")["results"]
        self.assertEqual(summary["accepted_flags"], ["b-0.2-int4"])
        families = summary["families"]
        self.assertEqual(families["dense-int3"]["decay_percent"], 20)
        self.assertTrue(families["b-0.2-int4"]["flags_accepted"])
        self.assertEqual(families["swiglu-0.2"]["noise_sample"], 0.004)
        self.assertAlmostEqual(families["dense-int3"]["token_equivalent"]["20"], (2.0 - 0.95) / 1.1 * 500)

    def test_flags_cover_fidelity_and_an_old_metric_that_disagrees(self):
        selection = r.family_decision(trunk([2.0, 1.0, 1.0, 1.0, 1.0]), branches([0.9, 1.0, 1.0]), RULE)
        legacy = r.family_decision(trunk([2.0, 1.0, 1.0, 1.0, 1.0]), branches([0.99, 1.0, 1.0]), RULE,
                                   metric="legacy_kl")
        artifact = {"results": {"acceptance": {"trunk-curve": {"passed": False, "checks": [
            {"passed": False, "relative_difference": 0.02}]}}}}
        flags = r.family_flags(artifact, selection, legacy)
        self.assertEqual([flag["flag"] for flag in flags], ["trunk-curve", "legacy-disagrees"])


if __name__ == "__main__":
    unittest.main()
