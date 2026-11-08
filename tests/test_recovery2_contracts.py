"""Focused CPU/static contracts for Recovery-2; no pretrained model or dataset is loaded."""

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
from mlp_replacement.compression.recovery import recover_trainable_by_tokens
from workflows.runs.model.recovery import recovery_2 as r
from workflows.runs.model.swiglu.swiglu_7 import cpu_tree, parameter_names
from workflows.runs.model.swiglu.swiglu_8 import AcceptanceFailure

CONFIG = ROOT / "workflows/configs/model/recovery/recovery-2.json"
SBATCH = ROOT / "workflows/jobs/perun/recovery/recovery_2.sbatch"
LOCAL = ROOT / "data/results/workflows/model"
M = 1_000_000
RULE = {"bar": 0.035, "reference_requests": [750 * M, 800 * M, 900 * M, 1000 * M]}


def row(tokens, selection, legacy=None):
    return {"actual_tokens": tokens, "requested_tokens": [tokens], "selection_kl": selection,
            "legacy_kl": selection if legacy is None else legacy}


def trunk(values):
    return [row(tokens * M, value) for tokens, value in zip((0, 750, 800, 900, 1000), values)]


def branches(values):
    return {str(percent): {"history": [row(1000 * M, value)]} for percent, value in zip((20, 50), values)}


def tiny_model(seed):
    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=16)
    return LlamaForCausalLM(config).eval()


TOKENS = torch.randint(0, 32, (64,), generator=torch.Generator().manual_seed(5))
LR = 1e-2


def batch_at(offset, count):
    ids = TOKENS[offset:offset + count].view(1, count)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def mlp_groups(model):
    parameters = list(model.model.layers[0].mlp.parameters())
    return [{"name": "replacements", "parameters": parameters, "parameter_names": parameter_names(model, parameters),
             "parameter_count": sum(value.numel() for value in parameters), "learning_rate": LR,
             "weight_decay": 0.0}], [model.model.layers[0].mlp]


def mlp_state(model):
    return [value.detach().clone() for value in model.model.layers[0].mlp.parameters()]


def loop_arguments(model, teacher):
    groups, modules = mlp_groups(model)
    return dict(
        student=model, teacher=teacher,
        parameter_groups=[{key: group[key] for key in ("name", "parameters", "learning_rate", "weight_decay")}
                          for group in groups],
        train_modules=modules, microbatch_tokens=4, accumulation_steps=1,
        temperature=1.0, ce_weight=0.0, device=torch.device("cpu"), autocast_dtype=torch.bfloat16,
        optimizer_backend="single_tensor",
    )


class LayoutContracts(unittest.TestCase):
    """The trunk must repeat the recorded layout, and a branch an uninterrupted decay in it.

    Budget 40, update 4 tokens: the segment end at 18 leaves a 2-token update, as 100M
    leaves 256 tokens after the last full 8,192-token update.
    """

    budget = 40

    def task(self, model, teacher, output_dir, endpoints):
        groups, modules = mlp_groups(model)
        recorded = {"results": {"recovery": {"validation_history": [
            {"requested_tokens": [value], "actual_tokens": value, "recovery_validation_kl": 1.0}
            for value in (0, 10, 18, 20, 30, 40)]}}}
        settings = {"budget_tokens": self.budget, "rows": {"interval_tokens": 10},
                    "schedule": {"scheduler": "wsd_one_minus_sqrt", "final_lr_ratio": 0.0, "decay_percent": [50]},
                    "acceptance": {"review_relative": 0.01}}
        recipe = {"effective_batch_tokens": 4, "sequence_length": 4, "microbatch_sequences": 1,
                  "gradient_accumulation_steps": 1, "temperature": 1.0, "ce_weight": 0.0, "scheduler": "constant",
                  "final_lr_ratio": 1.0, "optimizer_backend": "single_tensor", "monitoring_ppl_tokens": [],
                  "segment_endpoints": endpoints}
        artifact = {"schema_version": 1, "workflow": "recovery-2", "command": "train", "status": "running",
                    "configuration": settings, "results": {"trunk": {"history": [row(0, 1.0)]}}}
        return SimpleNamespace(settings=settings, recipe=recipe, budget=self.budget, model=model, teacher=teacher,
                               groups=groups, train_modules=modules, batch_at=batch_at, device=torch.device("cpu"),
                               family={"name": "toy", "kind": "order-b", "segment_endpoints": endpoints},
                               recorded={"recorded_1b": recorded}, artifact=artifact, output_dir=output_dir)

    def fake_score(self, task, tokens, requested, updates, seconds, mean_train_kl, learning_rates):
        value = float(sum(value.detach().sum() for value in task.model.model.layers[0].mlp.parameters()))
        return {"actual_tokens": tokens, "requested_tokens": list(requested), "optimizer_updates": updates,
                "learning_rates": list(learning_rates), "selection_kl": value, "legacy_kl": value}

    def run_screen(self, student, teacher, endpoints):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(r, "score", self.fake_score):
            task = self.task(student, teacher, Path(directory), endpoints)
            starts = r.decay_starts(task.settings, task.recipe, task.family)
            snapshots = r.train_trunk(task, starts)
            trunk_end = mlp_state(student)
            r.train_branch(task, 50, starts[50], snapshots.pop(50))
        return task, starts, trunk_end

    def recorded_path(self, model, teacher, endpoints, decay_start=None):
        """The SwiGLU-7 / Stacking-1 path: one recover_exact_segment per segment, optimizer state carried."""

        requests = [10, 18, 20, 30, 40]
        origin, updates, carried = 0, 0, None
        for end in endpoints:
            kept = {}

            def keep(event, optimizer, first_step, end=end, kept=kept):
                if event.tokens_seen == end:
                    kept["state"] = cpu_tree(optimizer.state_dict())

            arguments = loop_arguments(model, teacher)
            if decay_start is not None and end == endpoints[-1]:
                # The same segment with the decay, uninterrupted: constant to the start, then 1-sqrt.
                result = recover_trainable_by_tokens(
                    batch_at=lambda offset, count, origin=origin: batch_at(origin + offset, count),
                    target_tokens=end - origin, schedule_tokens=end - origin, scheduler="wsd_one_minus_sqrt",
                    warmup_fraction=0.0, final_lr_ratio=0.0, decay_start_tokens=decay_start - origin,
                    start_updates=updates, optimizer_state=carried, **arguments)
            else:
                result = recover_exact_segment(
                    origin=origin, end=end, cursor=origin, schedule=segment_schedule(origin, end, requests, 4),
                    batch_at=batch_at, on_checkpoint=keep, scheduler="constant", warmup_fraction=0.0,
                    final_lr_ratio=1.0, start_updates=updates, optimizer_state=carried, **arguments)
            origin, updates, carried = end, result.optimizer_updates, kept.get("state")

    def test_decay_starts_follow_each_layout(self):
        settings = {"budget_tokens": self.budget, "schedule": {"decay_percent": [50]}}
        recipe = {"effective_batch_tokens": 4}
        self.assertEqual(r.decay_starts(settings, recipe, {"segment_endpoints": [40]})[50],
                         {"requested": 20, "origin": 0, "tokens": 20})
        self.assertEqual(r.decay_starts(settings, recipe, {"segment_endpoints": [18, 40]})[50],
                         {"requested": 20, "origin": 18, "tokens": 22})

    def test_trunk_equals_the_recorded_segment_path(self):
        ends = {}
        for endpoints in ([40], [18, 40]):
            with self.subTest(endpoints=endpoints):
                teacher, student = tiny_model(1), tiny_model(2)
                reference = copy.deepcopy(student)
                task, starts, trunk_end = self.run_screen(student, teacher, endpoints)
                self.recorded_path(reference, teacher, endpoints)
                for left, right in zip(trunk_end, mlp_state(reference)):
                    self.assertTrue(torch.equal(left, right))
                actual = [value["actual_tokens"] for value in task.artifact["results"]["trunk"]["history"]]
                self.assertEqual(actual, [0, 12, 20, 32, 40] if endpoints == [40] else [0, 12, 18, 22, 30, 40])
                ends[len(endpoints)] = trunk_end
        # The two layouts cut the data differently, so the check above can tell them apart.
        self.assertFalse(all(torch.equal(left, right) for left, right in zip(ends[1], ends[2])))

    def test_branch_from_the_snapshot_equals_an_uninterrupted_decay(self):
        for endpoints in ([40], [18, 40]):
            with self.subTest(endpoints=endpoints):
                teacher, student = tiny_model(1), tiny_model(3)
                reference = copy.deepcopy(student)
                task, starts, trunk_end = self.run_screen(student, teacher, endpoints)
                start = starts[50]["tokens"]
                self.recorded_path(reference, teacher, endpoints, decay_start=start)
                for left, right in zip(mlp_state(student), mlp_state(reference)):
                    self.assertTrue(torch.equal(left, right))
                self.assertFalse(all(torch.equal(left, right) for left, right in zip(trunk_end, mlp_state(student))))
                # Each row's rate is the 1-sqrt formula in absolute tokens, whatever the segment origin.
                history = task.artifact["results"]["branches"]["50"]["history"]
                self.assertEqual(history[-1]["actual_tokens"], self.budget)
                for value in history:
                    expected = LR * (1.0 - math.sqrt((value["actual_tokens"] - start) / (self.budget - start)))
                    self.assertAlmostEqual(value["learning_rates"][0], expected, delta=1e-12)


def recorded_rows(family, recorded):
    """(request, actual tokens, updates or None) for every row of a recorded 1B run."""

    recovery = recorded["results"]["recovery"]
    if family["kind"] == "dense-qat":
        return [(int(value["requested_tokens"][-1]), int(value["tokens_seen"]), None) for value in recovery["history"]]
    return [(int(value["requested_tokens"][-1]), int(value["actual_tokens"]), int(value["optimizer_updates"]))
            for value in recovery["validation_history"]]


class ConfigurationContracts(unittest.TestCase):
    def setUp(self):
        self.settings, self.stacking, self.dense, unused_path = r.load_settings(CONFIG)
        self.recipe = self.stacking["recovery_1b"]

    def test_decay_starts_per_family(self):
        expected = {"dense-int3": {20: 800_006_144, 50: 500_006_912},
                    "b-0.2-int4": {20: 800_006_400, 50: 500_007_168},
                    "swiglu-0.2": {20: 800_006_400, 50: 500_007_168}}
        for family in self.settings["families"]:
            starts = r.decay_starts(self.settings, self.recipe, family)
            self.assertEqual({percent: value["tokens"] for percent, value in starts.items()}, expected[family["name"]])

    def test_planned_rows_reproduce_every_recorded_1b_row(self):
        requests = r.trunk_requests(self.settings, self.recipe)
        self.assertEqual(len(requests), 41)
        checked = 0
        for family in self.settings["families"]:
            pin = family["recorded_1b"]
            path = LOCAL / pin["workflow"] / pin["run"] / "result.json"
            if not path.is_file():
                continue
            planned, origin, updates = {}, 0, 0
            for end in family["segment_endpoints"]:
                for actual, requested in segment_schedule(origin, end, requests, 8192):
                    for value in requested:
                        planned[value] = (actual, updates + -(-(actual - origin) // 8192))
                updates += -(-(end - origin) // 8192)
                origin = end
            rows = recorded_rows(family, read_json(path))
            self.assertEqual(sorted(value[0] for value in rows), [0, *requests])
            for request, actual, recorded_updates in rows:
                if request == 0:
                    continue
                self.assertEqual(planned[request][0], actual, (family["name"], request))
                if recorded_updates is not None:
                    self.assertEqual(planned[request][1], recorded_updates, (family["name"], request))
            starts = r.decay_starts(self.settings, self.recipe, family)
            self.assertTrue({value["tokens"] for value in starts.values()} <= {value[1] for value in rows})
            if family["kind"] != "dense-qat":
                self.assertIn((100_000_000, 100_000_000, 12_208), rows)
            checked += 1
        if not checked:
            self.skipTest("Local result copies are not present")

    def refused(self, change):
        settings = read_json(CONFIG)
        change(settings)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recovery-2.json"
            path.write_text(json.dumps(settings), encoding="utf-8")
            with self.assertRaises(ValueError):
                r.load_settings(path)

    def test_refuses_a_changed_recipe_layout_rule_or_pin(self):
        self.refused(lambda value: value["schedule"].update(learning_rate=1e-4))
        self.refused(lambda value: value["schedule"].update(warmup_fraction=0.01))
        self.refused(lambda value: value["schedule"].update(decay_percent=[10, 20, 50]))
        self.refused(lambda value: value["families"].reverse())
        self.refused(lambda value: value["families"][1].update(segment_endpoints=[1_000_000_000]))
        self.refused(lambda value: value["families"][0].update(segment_endpoints=[100_000_000, 1_000_000_000]))
        self.refused(lambda value: value["families"][2]["recovery_1"].update(run="swiglu-0.2-run-002"))
        self.refused(lambda value: value["rule"].update(reference="lowest"))
        self.refused(lambda value: value["rule"].update(reference_requests=[760_000_000, 800_000_000]))
        self.refused(lambda value: value["rows"].update(interval_tokens=50_000_000))
        self.refused(lambda value: value["streams"].update(recovery_sha256="0" * 63))
        self.refused(lambda value: value["sources"].update(swiglu7_prepare_sha256="0" * 64))

    def test_pins_match_the_local_artifacts(self):
        checked = 0
        for family in self.settings["families"]:
            for key in ("recorded_1b", "recovery_1"):
                pin = family[key]
                path = LOCAL / pin["workflow"] / pin["run"] / "result.json"
                if path.is_file():
                    self.assertEqual(file_digest(path), pin["result_sha256"], path)
                    checked += 1
        prepared = LOCAL / "swiglu-8/prepare-001/result.json"
        block = LOCAL / "swiglu-8/block-check-001/result.json"
        swiglu7 = LOCAL / "swiglu-7/prepare-001/result.json"
        if not (prepared.is_file() and block.is_file() and swiglu7.is_file()) or not checked:
            self.skipTest("Local result copies are not present")
        streams, sources = self.settings["streams"], self.settings["sources"]
        self.assertEqual(file_digest(swiglu7), sources["swiglu7_prepare_sha256"])
        stream = read_json(swiglu7)["results"]["stream"]
        self.assertEqual((stream["sha256"], stream["token_count"], stream["sequence_length"]),
                         (streams["recovery_sha256"], 1_000_000_000, 8192))
        results = read_json(prepared)["results"]
        self.assertEqual(results["long_context_data"]["kl_stream"]["sha256"], streams["final_kl_sha256"])
        self.assertEqual(results["monitoring_evaluation"]["sha256"], streams["monitoring_sha256"])
        selection = read_json(block)["results"]["long_context_data"]["selection_stream"]
        self.assertEqual(selection["sha256"], streams["selection_sha256"])

    def test_job_array_order_and_request(self):
        text = SBATCH.read_text(encoding="utf-8")
        names = re.search(r"FAMILY_NAMES=\(([^)]*)\)", text).group(1).split()
        self.assertEqual(names, [family["name"] for family in self.settings["families"]])
        self.assertIn("workflows.runs.model.recovery.recovery_2 train", text)
        self.assertIn("#SBATCH --time=30:00:00", text)
        self.assertIn("swiglu-7/prepare-001/prepared/recovery-tokens.int32", text)
        for family in self.settings["families"]:
            self.assertIn(f"recovery-1/{family['name']}-run-001/result.json", text)
        self.assertNotIn("\r\n", text)


class StartGateContracts(unittest.TestCase):
    """Token 0 must reproduce the recorded 1B run and Recovery-1's selection value."""

    def task(self, output_dir, earlier_selection):
        model = tiny_model(4)
        groups, modules = mlp_groups(model)
        recorded = {"results": {"recovery": {"validation_history": [
            {"requested_tokens": [0], "actual_tokens": 0, "recovery_validation_kl": 0.25}]}}}
        earlier = {"results": {"trunk": {"history": [{"actual_tokens": 0, "selection_kl": earlier_selection}]}}}
        artifact = {"schema_version": 1, "workflow": "recovery-2", "command": "train", "status": "running",
                    "configuration": {}, "results": {}}
        return SimpleNamespace(
            settings={"acceptance": {"exact_relative": 1e-4, "teacher_self_kl_absolute": 1e-6}},
            family={"name": "toy", "kind": "order-b"}, recorded={"recorded_1b": recorded, "recovery_1": earlier},
            groups=groups, model=model, teacher=model, selection=None, length=8, artifact=artifact,
            output_dir=output_dir)

    def check(self, earlier_selection):
        score = {"actual_tokens": 0, "requested_tokens": [0], "selection_kl": 0.2, "legacy_kl": 0.25}
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(r, "stream_kl", return_value=0.0), \
                mock.patch.object(r, "score", return_value=score):
            task = self.task(Path(directory), earlier_selection)
            r.check_start(task)
            return task.artifact["results"]["acceptance"]

    def test_matching_recovery_1_passes_and_a_different_value_stops(self):
        acceptance = self.check(0.2)
        self.assertTrue(acceptance["selection-token-0"]["passed"])
        self.assertTrue(acceptance["token-0"]["passed"])
        with self.assertRaises(AcceptanceFailure):
            self.check(0.2 * 1.001)


class WinnerRuleContracts(unittest.TestCase):
    def test_no_decay_beyond_the_bar_keeps_constant(self):
        decision = r.family_decision(trunk([2.0, 1.0, 1.0, 1.0, 1.0]), branches([0.97, 0.98]), RULE)
        self.assertIsNone(decision["decay_percent"])

    def test_the_shorter_decay_wins_within_the_bar_of_the_better(self):
        values = trunk([2.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(r.family_decision(values, branches([0.93, 0.91]), RULE)["decay_percent"], 20)
        self.assertEqual(r.family_decision(values, branches([0.95, 0.90]), RULE)["decay_percent"], 50)
        self.assertEqual(r.family_decision(values, branches([0.95, 0.99]), RULE)["decay_percent"], 20)

    def test_the_median_ignores_one_spike_or_dip_and_the_lowest_is_reported_only(self):
        spike = r.family_decision(trunk([2.0, 1.0, 1.0, 1.0, 5.0]), branches([0.98, 0.98]), RULE)
        self.assertEqual(spike["reference"], 1.0)
        self.assertIsNone(spike["decay_percent"])
        dip = trunk([2.0, 1.0, 1.2, 0.8, 1.0])
        median = r.family_decision(dip, branches([0.95, 0.90]), RULE)
        lowest = r.family_decision(dip, branches([0.95, 0.90]), RULE, reference="lowest")
        self.assertEqual((median["reference"], median["decay_percent"]), (1.0, 50))
        self.assertEqual((lowest["reference"], lowest["decay_percent"]), (0.8, None))
        self.assertEqual(median["reference_rows"], {str(750 * M): 1.0, str(800 * M): 1.2,
                                                    str(900 * M): 0.8, str(1000 * M): 1.0})

    def test_a_missing_reference_row_is_refused(self):
        rows = [value for value in trunk([2.0, 1.0, 1.0, 1.0, 1.0]) if value["actual_tokens"] != 800 * M]
        with self.assertRaises(ValueError):
            r.family_decision(rows, branches([0.9, 0.9]), RULE)

    def test_summary_holds_flagged_families_and_writes_only_after_review(self):
        settings = read_json(CONFIG)
        trunk_rows = trunk([2.0, 1.0, 1.0, 1.0, 1.0])
        family_branches = branches([0.95, 0.94])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, family in enumerate(settings["families"]):
                passed = index != 1
                artifact = {"schema_version": 1, "workflow": "recovery-2", "command": "train", "status": "completed",
                            "configuration": settings, "code_hashes": {"same": "code"},
                            "updated_at_utc": "2026-11-09T00:00:00+00:00",
                            "results": {"trunk": {"history": trunk_rows}, "branches": family_branches,
                                        "acceptance": {"trunk-curve": {"passed": passed, "checks": [
                                            {"passed": passed, "relative_difference": 0.004 if passed else 0.02}]}}}}
                (root / r.output_name(family)).mkdir()
                (root / r.output_name(family) / "result.json").write_text(json.dumps(artifact), encoding="utf-8")
            arguments = SimpleNamespace(results_root=root, accept_flags=[])
            with mock.patch("builtins.print"):
                r.summarize(arguments, settings)
                self.assertFalse((root / "summary-001").exists())
                arguments.accept_flags = ["b-0.2-int4"]
                r.summarize(arguments, settings)
            summary = read_json(root / "summary-001" / "result.json")["results"]
        self.assertEqual(summary["accepted_flags"], ["b-0.2-int4"])
        families = summary["families"]
        self.assertEqual(families["dense-int3"]["decay_percent"], 20)
        self.assertEqual(families["dense-int3"]["lowest_of_four"]["decay_percent"], 20)
        self.assertTrue(families["b-0.2-int4"]["flags_accepted"])
        self.assertEqual(families["swiglu-0.2"]["noise_sample"], 0.004)


if __name__ == "__main__":
    unittest.main()
