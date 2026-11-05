"""Focused CPU/static contracts for Heterogeneous-1; no pretrained model or dataset is loaded."""

import copy
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from mlp_replacement.artifacts import read_json, write_json_atomic
from mlp_replacement.capture import ActivationPairs
from mlp_replacement.compression.allocation import (
    allocate_discrete_swiglu_widths,
    build_operator_curve,
    build_swiglu_width_curve,
)
from mlp_replacement.compression.teacher_cache import cache_teacher_logits
from mlp_replacement.data import PackedTokenCache
from mlp_replacement.model import discover_mlp_blocks
from mlp_replacement.operators import (
    BottleneckMLPReplacement,
    DenseHybridReplacement,
    GatedMLPReplacement,
    LinearReplacement,
    MeanReplacement,
    ZeroReplacement,
    fit_ridge_linear,
)
from workflows.runs.model.heterogenous import heterogeneous_1 as h

CONFIG = ROOT / "workflows/configs/model/heterogenous/heterogeneous-1.json"
SBATCH = ROOT / "workflows/jobs/perun/heterogenous/heterogeneous_1.sbatch"
SWIGLU8_PREPARE = ROOT / "data/results/workflows/model/swiglu-8/prepare-001/result.json"
SIZES = (0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
HIDDEN = 2048
MLP_UNITS = 3 * 8192


def swiglu_curve(layer, costs, hidden=4, original=8):
    evaluations = {width: {"teacher_kl": cost} for width, cost in costs.items()}
    return build_swiglu_width_curve(layer, original, evaluations, hidden)


def single_item(layer, operator, units, cost, hidden=4, original=8):
    item = {"width": 0, "parameters": units * hidden, "teacher_kl": cost}
    return build_operator_curve(layer, operator, original, [item])


def small_menu():
    costs = {2: 0.3, 4: 0.1, 8: 0.0}
    return {
        layer: (*swiglu_curve(layer, costs), *single_item(layer, "zero", 0, zero_cost))
        for layer, zero_cost in ((1, 0.5), (2, 0.25), (3, 0.9))
    }


class Teacher(torch.nn.Module):
    """A SwiGLU block exposing the projections teacher initialization expects."""

    def __init__(self, hidden, width):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, width, bias=False)
        self.up_proj = torch.nn.Linear(hidden, width, bias=False)
        self.down_proj = torch.nn.Linear(width, hidden, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, inputs):
        return self.down_proj(self.act_fn(self.gate_proj(inputs)) * self.up_proj(inputs))


def fitting_setup(hidden=8, width=16, tokens=256):
    torch.manual_seed(0)
    teacher = Teacher(hidden, width)
    inputs = torch.randn(tokens, hidden)
    with torch.no_grad():
        pairs = ActivationPairs(inputs=inputs, targets=teacher(inputs))
    settings = {
        "seed": 21,
        "model": {"hidden_size": hidden},
        "local_fitting": {
            "batch_size": 64, "learning_rate": 0.001, "weight_decay": 0.0, "max_epochs": 2,
            "scheduler": "constant", "early_stopping_patience": 3, "early_stopping_min_delta": 0.0,
        },
    }
    context = SimpleNamespace(
        settings=settings,
        device=torch.device("cpu"),
        blocks={3: SimpleNamespace(module=teacher, path="mlp")},
    )
    return context, pairs, torch.arange(width)


def training_mse(module, pairs):
    with torch.no_grad():
        return float((module(pairs.inputs) - pairs.targets).square().mean())


class AllocatorContracts(unittest.TestCase):
    def test_swiglu_only_rows_carry_no_operator_key(self):
        curves = {1: swiglu_curve(1, {2: 0.5, 4: 0.2, 8: 0.0}), 2: swiglu_curve(2, {2: 0.4, 4: 0.3, 8: 0.0})}
        solved = allocate_discrete_swiglu_widths(curves, 0.5, 4)
        self.assertTrue(all("operator" not in row for row in solved.rows))

    def test_a_mixed_menu_names_its_operator_and_leftover_goes_to_swiglu(self):
        curves = {
            1: (*swiglu_curve(1, {2: 0.5, 4: 0.2, 8: 0.0}), *single_item(1, "linear", 9, 0.1)),
            2: swiglu_curve(2, {2: 0.5, 4: 0.3, 8: 0.0}),
        }
        solved = allocate_discrete_swiglu_widths(curves, 0.5, 4)
        first, second = solved.rows
        self.assertEqual((first["operator"], first["replacement_parameters"]), ("linear", 36))
        self.assertEqual((solved.boundary_layer, second["replacement_width"]), (2, 5))

    def test_leftover_budget_never_widens_a_non_swiglu_item(self):
        curves = {
            layer: (*swiglu_curve(layer, {2: 0.5, 4: 0.2, 8: 0.0}), *single_item(layer, "linear", 9, 0.1))
            for layer in (1, 2)
        }
        solved = allocate_discrete_swiglu_widths(curves, 0.5, 4)
        self.assertIsNone(solved.boundary_layer)
        self.assertEqual([row["replacement_parameters"] for row in solved.rows], [36, 36])

    def test_smoothing_stays_within_one_family_and_nothing_is_the_dense_module(self):
        points = build_operator_curve(1, "linear_plain_mlp", 8, [
            {"width": 2, "parameters": 8, "teacher_kl": 0.2},
            {"width": 8, "parameters": 16, "teacher_kl": 0.4},
            {"width": 12, "parameters": 24, "teacher_kl": 0.1},
        ])
        self.assertAlmostEqual(points[0].monotone_teacher_kl, 0.3)
        self.assertAlmostEqual(points[1].monotone_teacher_kl, 0.3)
        self.assertFalse(any(point.retains_dense_module for point in points))

    @unittest.skipUnless(SWIGLU8_PREPARE.is_file(), "local SwiGLU-8 preparation record is absent")
    def test_check_1_reproduces_swiglu8s_recorded_solves(self):
        settings, unused_path = h.load_settings(CONFIG)
        prepared = read_json(SWIGLU8_PREPARE)
        checks = h.allocator_checks(prepared["results"]["width_curves"]["rows"], prepared, settings)
        self.assertEqual(len(checks), 2 * len(h.TARGETS))
        self.assertTrue(all(row["passed"] for row in checks), [row for row in checks if not row["passed"]])


class PriceContracts(unittest.TestCase):
    def test_table_widths_land_on_the_decided_sizes(self):
        settings, unused_path = h.load_settings(CONFIG)
        widths = settings["cost_tables"]["widths"]
        self.assertEqual(
            [h.item_units("plain_mlp", width, HIDDEN) for width in widths["plain_mlp"]],
            [round(size * MLP_UNITS) for size in SIZES],
        )
        self.assertEqual(
            [h.item_units("linear_plain_mlp", width, HIDDEN) for width in widths["linear_plain_mlp"]],
            [round(size * MLP_UNITS) for size in SIZES[1:]],
        )
        # Whole neurons leave linear + SwiGLU one budget unit (2,048 parameters) below each size.
        self.assertEqual(
            [h.item_units("linear_swiglu", width, HIDDEN) for width in widths["linear_swiglu"]],
            [round(size * MLP_UNITS) - 1 for size in SIZES[1:]],
        )

    def test_each_module_holds_exactly_its_price(self):
        hidden = 8
        modules = {
            ("zero", 0): ZeroReplacement(),
            ("mean", 0): MeanReplacement(torch.zeros(hidden)),
            ("linear", 0): LinearReplacement(hidden),
            ("swiglu", 4): GatedMLPReplacement(hidden, 4),
            ("plain_mlp", 6): BottleneckMLPReplacement(hidden, 6 / hidden, activation="silu"),
            ("linear_swiglu", 4): DenseHybridReplacement(hidden, GatedMLPReplacement(hidden, 4)),
            ("linear_plain_mlp", 6): DenseHybridReplacement(
                hidden, BottleneckMLPReplacement(hidden, 6 / hidden, activation="silu")
            ),
        }
        for (operator, width), module in modules.items():
            self.assertEqual(h.check_parameters(module, operator, width, hidden),
                             hidden * h.item_units(operator, width, hidden))
        with self.assertRaises(h.AcceptanceFailure):
            h.check_parameters(LinearReplacement(hidden, bias=True), "linear", 0, hidden)

    def test_the_full_menu_prices_every_item(self):
        settings, unused_path = h.load_settings(CONFIG)
        layers = settings["allocation"]["eligible_layers"]
        curve_rows = [
            {"layer": layer, "replacement_width": width, "raw_teacher_kl": 0.0 if width == 8192 else 1.0 / width}
            for layer in layers for width in (512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192)
        ]
        table = [
            {"operator": operator, "layer": layer, "replacement_width": width, "raw_teacher_kl": 0.1,
             "replacement_parameters": HIDDEN * h.item_units(operator, width, HIDDEN)}
            for layer in layers
            for operator, widths in (*settings["cost_tables"]["widths"].items(), ("zero", [0]), ("mean", [0]),
                                     ("swiglu", [256]))
            for width in widths
        ]
        menu = h.build_menu(curve_rows, table, settings, (*h.FIXED_OPERATORS, *h.FITTED_OPERATORS))
        self.assertTrue(all(len(points) == 35 for points in menu.values()))
        self.assertTrue(all(row["passed"] for row in h.menu_price_checks(menu, HIDDEN)))


class FittingContracts(unittest.TestCase):
    def test_a_zero_linear_part_reproduces_the_nonlinear_part_exactly(self):
        nonlinear = GatedMLPReplacement(8, 4)
        hybrid = DenseHybridReplacement(8, nonlinear)
        torch.nn.init.zeros_(hybrid.linear.weight)
        inputs = torch.randn(5, 8)
        self.assertTrue(torch.equal(hybrid(inputs), nonlinear(inputs)))

    def test_the_residual_start_beats_the_slice_and_a_fit_of_the_whole_output(self):
        context, pairs, ranking = fitting_setup()
        start = h.nonlinear_start(context, "linear_swiglu", 3, 12, ranking, 21)
        whole = DenseHybridReplacement(8, start)
        with torch.no_grad():
            whole.linear.weight.copy_(fit_ridge_linear(pairs, ridge=1e-4).projection.weight)
        residual = h.hybrid_start(context, "linear_swiglu", 3, 12, ranking, 21, pairs, 1e-4)
        self.assertLess(training_mse(residual, pairs), training_mse(start, pairs))
        self.assertLess(training_mse(residual, pairs), training_mse(whole, pairs))

    def test_seeds_change_random_starts_reproducibly(self):
        context, unused_pairs, ranking = fitting_setup()
        first = h.nonlinear_start(context, "plain_mlp", 3, 6, ranking, 21)
        again = h.nonlinear_start(context, "plain_mlp", 3, 6, ranking, 21)
        other = h.nonlinear_start(context, "plain_mlp", 3, 6, ranking, 22)
        self.assertTrue(torch.equal(first.up_projection.weight, again.up_projection.weight))
        self.assertFalse(torch.equal(first.up_projection.weight, other.up_projection.weight))
        self.assertIsInstance(first.activation, torch.nn.SiLU)

    def test_joint_and_linear_first_share_a_start_and_differ_only_in_freezing(self):
        context, pairs, ranking = fitting_setup()
        start = h.hybrid_start(context, "linear_plain_mlp", 3, 6, ranking, 21, pairs, 1e-4)

        def fit(setting_id, setting):
            context, pairs, ranking = fitting_setup()
            setting = {**setting, "max_epochs": 2, "batch_size": 64}
            return h.fit_new_operator(context, "linear_plain_mlp", 3, 6, setting_id, setting, 21, ranking,
                                      pairs, pairs, 1e-4)[0]

        frozen = fit(h.LINEAR_FIRST, h.recipe_grid("linear_plain_mlp")[h.LINEAR_FIRST])
        joint = fit(h.REFERENCE_SETTING, h.GRADIENT_SETTINGS[h.REFERENCE_SETTING])
        self.assertTrue(torch.equal(frozen.linear.weight, start.linear.weight))
        self.assertFalse(torch.equal(frozen.nonlinear.up_projection.weight, start.nonlinear.up_projection.weight))
        self.assertFalse(torch.equal(joint.linear.weight, start.linear.weight))

    def test_ridge_precision_is_optional_and_the_module_stays_float32(self):
        unused_context, pairs, unused_ranking = fitting_setup()
        single = fit_ridge_linear(pairs, ridge=1e-4).projection.weight
        double = fit_ridge_linear(pairs, ridge=1e-4, dtype=torch.float64).projection.weight
        self.assertEqual(double.dtype, torch.float32)
        self.assertTrue(torch.allclose(single, double, atol=1e-4))


class DecisionContracts(unittest.TestCase):
    def test_forcing_ranks_layers_by_extra_damage_and_excludes_every_swiglu_item(self):
        forcing = h.forced_points(small_menu(), 0.5, 4, (1, 2))
        self.assertEqual(forcing["order"], [2, 1, 3])
        first, second = forcing["points"]
        self.assertEqual((first["forced_layers"], first["non_swiglu_layers"]), ([2], [2]))
        self.assertEqual((second["forced_layers"], second["non_swiglu_layers"]), ([1, 2], [1, 2]))
        self.assertLess(first["predicted_teacher_kl"], second["predicted_teacher_kl"])
        self.assertEqual([point.operator for point in h.restricted(small_menu(), {1})[1]], ["zero"])

    def test_the_swap_rule_follows_the_design_example_and_its_margin(self):
        before = {"swiglu": 0.017, "plain": 0.030}
        self.assertTrue(h.ranking_swap(before, {"swiglu": 0.015, "plain": 0.014}, 0.05)["swap"])
        self.assertFalse(h.ranking_swap(before, {"swiglu": 0.0150, "plain": 0.0149}, 0.05)["swap"])

    def test_the_swap_table_compares_equal_sizes_only(self):
        def run(operator, width, size, before, after):
            return {"layer": 3, "size": size, "operator": operator, "replacement_width": width,
                    "history": [{"kl_8192": before}, {"kl_8192": after}]}

        rows = h.swap_table([
            run("swiglu", 683, "8.33", 0.05, 0.04), run("linear", 0, "8.33", 0.06, 0.03),
            run("swiglu", 1024, "12.5", 0.02, 0.01), run("plain_mlp", 1536, "12.5", 0.03, 0.02),
        ], 0.05, "kl_8192")
        self.assertEqual([(row["size"], row["swap"]) for row in rows], [("12.5", False), ("8.33", True)])

    def test_assembly_starts_with_swiglu8_and_skips_repeats(self):
        dense = {"layer": 2, "replacement_width": 8192, "retains_dense_module": True}
        swiglu = {"layer": 1, "replacement_width": 1024, "retains_dense_module": False}
        hybrid = {"layer": 1, "replacement_width": 0, "retains_dense_module": False, "operator": "linear"}
        forced = {"layer": 2, "replacement_width": 0, "retains_dense_module": False, "operator": "zero"}
        entry = {
            "swiglu_with_256": {"changed": False, "rows": [swiglu, dense], "predicted_teacher_kl": 0.2},
            "full_menu": {"non_swiglu_layers": [1], "rows": [hybrid, dense], "predicted_teacher_kl": 0.1},
            "forcing": {"points": [
                {"forced_count": 1, "rows": [hybrid, dense], "predicted_teacher_kl": 0.1},
                {"forced_count": 2, "rows": [hybrid, forced], "predicted_teacher_kl": 0.3},
            ]},
        }
        recorded = {"allocation": [swiglu, dense], "solver": {"predicted_teacher_kl": 0.2}}
        names = [point["name"] for point in h.assemble_points(entry, recorded)]
        self.assertEqual(names, ["swiglu-8", "hybrid", "forced-2"])


def fit_row(operator, setting, seed, layer, width, kl=1.0, best_epoch=5, epochs=8):
    return {
        "operator": operator, "setting": setting, "seed": seed, "layer": layer, "replacement_width": width,
        "kl_128": kl, "kl_8192": kl, "best_epoch": best_epoch, "epochs_completed": epochs,
        "history": [{"phase": "full", "train_mse": 1.0 - 0.01 * epoch} for epoch in range(epochs)],
    }


def select_artifact(settings, index, item, better=None):
    operator, layer, width = item["operator"], item["layer"], item["width"]
    seeds = settings["recipe_selection"]["seeds"]
    fits = [
        fit_row(operator, name, seed, layer, width, kl=0.9 if name == better else 1.0 + 0.001 * (seed or 0))
        for name, seed in h.grid_runs(h.recipe_grid(operator), seeds)
    ]
    acceptance = {"data-identity": {"passed": True}, "check-6": {"passed": True}, "check-2": {"passed": True}}
    if operator == "swiglu":
        fits.append(fit_row(operator, h.REFERENCE_SETTING, 21, layer, 683))
        acceptance["check-3"] = {"passed": True}
    if operator in h.HYBRIDS:
        acceptance["check-5"] = {"passed": True}
    return {
        "workflow": h.WORKFLOW, "command": "select", "status": "completed",
        "identity": {"kind": "select-task", "task_index": index, **item},
        "configuration": settings, "code_hashes": {"file": "digest"}, "environment": {},
        "created_at_utc": "2026-11-05T00:00:00+00:00", "updated_at_utc": "2026-11-05T01:00:00+00:00",
        "results": {"long_context_data": {"stream": "digest"}, "acceptance": acceptance, "fits": fits,
                    "anchor": fit_row("swiglu", h.REFERENCE_SETTING, 21, layer, 2048)},
    }


class SelectionContracts(unittest.TestCase):
    def setUp(self):
        self.settings, unused_path = h.load_settings(CONFIG)
        self.items = h.select_tasks(self.settings)

    def tasks(self, better=None):
        return [
            select_artifact(self.settings, index, item, (better or {}).get(item["operator"]))
            for index, item in enumerate(self.items)
        ]

    def merge(self, tasks):
        return h.merge_select(tasks, self.settings, [{"output": "task"} for unused_task in tasks])

    def test_the_task_map_and_grids_cover_every_decided_fit(self):
        self.assertEqual(len(self.items), 24)
        self.assertEqual(self.items[7], {"operator": "plain_mlp", "layer": 3, "width": 3072})
        self.assertEqual(len(h.recipe_grid("plain_mlp")), 13)
        self.assertEqual(set(h.recipe_grid("linear")) - set(h.GRADIENT_SETTINGS), set(h.RIDGE_SETTINGS))
        self.assertEqual(set(h.recipe_grid("linear_swiglu")) - set(h.GRADIENT_SETTINGS), {h.LINEAR_FIRST})
        seeds = self.settings["recipe_selection"]["seeds"]
        self.assertEqual(h.grid_runs(h.recipe_grid("swiglu"), seeds)[0], (h.REFERENCE_SETTING, 21))
        fits = sum(
            len(h.grid_runs(h.recipe_grid(item["operator"]), seeds)) + (item["operator"] == "swiglu")
            for item in self.items
        )
        self.assertEqual(fits, 981)

    def test_each_operator_gets_its_own_adoption_decision(self):
        merged = self.merge(self.tasks({"linear": "ridge-128", "plain_mlp": "c3e-4-128"}))
        adoption = merged["results"]["adoption"]
        self.assertEqual(adoption["linear"]["selected_recipe"], "ridge-128")
        self.assertEqual(adoption["plain_mlp"]["selected_recipe"], "c3e-4-128")
        self.assertEqual(adoption["linear_swiglu"]["selected_recipe"], h.REFERENCE_SETTING)

    def test_swiglu_683_stays_out_of_the_adoption_rule(self):
        merged = self.merge(self.tasks())
        rows = merged["results"]["adoption"]["swiglu"]["reference_seed_summaries"]
        self.assertEqual({row["width"] for row in rows}, {1024})
        self.assertIn(683, {row["replacement_width"] for row in merged["results"]["fits"]})

    def test_a_missing_fit_or_check_stops_the_merge_but_a_pinned_recipe_does_not(self):
        tasks = self.tasks()
        tasks[5]["results"]["fits"].pop()
        with self.assertRaises(ValueError):
            self.merge(tasks)
        tasks = self.tasks()
        tasks[12]["results"]["acceptance"].pop("check-5")
        with self.assertRaises(ValueError):
            self.merge(tasks)
        tasks = self.tasks()
        pinned = {**self.settings, "recipes": {"select_sha256": "a" * 64, "settings": {}}}
        self.assertEqual(h.merge_select(tasks, pinned, [{} for unused_task in tasks])["status"], "completed")

    def test_the_indicator_votes_at_the_largest_tested_size_in_two_layers(self):
        rows = [
            fit_row("plain_mlp", "x", seed, layer, width, best_epoch=2 if width == 3072 else 20, epochs=25)
            for layer in (3, 11, 19) for width in (1536, 3072) for seed in (21, 22, 23)
        ]
        result = h.overfitting_indicator(rows, "x")
        self.assertEqual((result["width"], result["overfitting_remains"]), (3072, True))
        one_layer = [
            fit_row("plain_mlp", "x", seed, layer, 3072, best_epoch=2 if layer == 3 else 20, epochs=25)
            for layer in (3, 11, 19) for seed in (21, 22, 23)
        ]
        self.assertFalse(h.overfitting_indicator(one_layer, "x")["overfitting_remains"])

    def test_later_stages_refuse_unreviewed_recipes(self):
        unreviewed = {**self.settings, "recipes": {"select_sha256": None, "settings": {"linear": None}}}
        with self.assertRaises(ValueError):
            h.load_selection(ROOT, unreviewed)

    def test_swiglu_keeps_the_decided_recipe_and_closed_forms_have_no_seed(self):
        settings = {**self.settings, "recipes": {"settings": {"linear": "ridge-8k"}}}
        self.assertEqual(h.operator_recipe(settings, "swiglu")[0], h.REFERENCE_SETTING)
        self.assertIsNone(h.operator_recipe(settings, "linear")[2])
        self.assertTrue(h.needs_long_pairs(settings, ["swiglu", "linear"]))


class TableContracts(unittest.TestCase):
    """A rerun under other settings must not mix into the knapsack's menu."""

    def setUp(self):
        self.unpinned, unused_path = h.load_settings(CONFIG)
        self.pinned = {**self.unpinned, "recipes": {"select_sha256": "a" * 64, "settings": {"linear": "ridge-128"}}}
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.write("baseline-table-001", "baseline-table", self.unpinned)
        for layer in self.unpinned["allocation"]["eligible_layers"]:
            self.write(h.cost_output_name(layer), "cost-table", self.pinned)

    def write(self, name, command, configuration):
        write_json_atomic(self.root / name / "result.json", {
            "workflow": h.WORKFLOW, "command": command, "status": "completed",
            "configuration": configuration, "results": {"table": [{"operator": "zero"}]},
        })

    def test_the_baseline_table_may_predate_the_pinned_recipes(self):
        rows, inputs = h.completed_table_rows(self.root, self.pinned)
        self.assertEqual((len(rows), len(inputs)), (23, 23))

    def test_a_cost_table_under_other_recipes_is_refused(self):
        other = {**self.pinned, "recipes": {"select_sha256": "b" * 64, "settings": {"linear": "c1e-3-128"}}}
        self.write(h.cost_output_name(5), "cost-table", other)
        with self.assertRaises(ValueError):
            h.completed_table_rows(self.root, self.pinned)

    def test_a_table_under_another_configuration_is_refused(self):
        self.write("baseline-table-001", "baseline-table", {**self.unpinned, "seed": 22})
        with self.assertRaises(ValueError):
            h.completed_table_rows(self.root, self.pinned)


class RecoveryContracts(unittest.TestCase):
    """Drive stage D's real loop and scorers on a tiny random Llama, on the CPU."""

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
        stream = Path(self.directory.name) / "tokens.int32"
        (numpy.arange(64, dtype=numpy.int32) % 32).tofile(stream)
        length = 8
        settings, unused_path = h.load_settings(CONFIG)
        settings["recovery"] = {
            **settings["recovery"], "sequence_length": length, "effective_batch_tokens": length,
            "optimizer_backend": "single_tensor", "learning_rate": 0.01,
        }
        settings["single_block_recovery"] = {"tokens": 40, "checkpoints": [8, 24, 40]}
        batches = [{"input_ids": torch.arange(length).reshape(1, length),
                    "attention_mask": torch.ones(1, length, dtype=torch.long)}]
        self.task = SimpleNamespace(
            settings=settings,
            context=SimpleNamespace(model=self.student, blocks=self.blocks, device=torch.device("cpu")),
            teacher=self.teacher,
            selection_cache=cache_teacher_logits(self.teacher, batches, 1, "cpu", "float32"),
            selection_path=stream,
            selection_record={"token_count": 64, "sha256": "stream", "protocol": "test"},
            length=length,
        )
        self.cache = PackedTokenCache(stream, 64, length, "stream")
        self.module = DenseHybridReplacement(16, GatedMLPReplacement(16, 4))

    def test_the_operator_stays_inserted_through_every_checkpoint(self):
        run = h.recover_item(self.task, self.cache, 1, self.module)
        self.assertEqual([row["tokens"] for row in run["history"]], [0, 8, 24, 40])
        self.assertEqual(run["tokens_seen"], 40)
        # Student and teacher start identical, so a KL of zero would mean the teacher MLP came back.
        self.assertTrue(all(row["kl_128"] > 0 and row["kl_8192"] > 0 for row in run["history"]))

    def test_only_the_inserted_operator_trains_and_the_teacher_mlp_returns(self):
        before = {name: value.detach().clone() for name, value in self.student.named_parameters()}
        start = {name: value.detach().clone() for name, value in self.module.named_parameters()}
        flags = {name: value.requires_grad for name, value in self.student.named_parameters()}
        h.recover_item(self.task, self.cache, 1, self.module)
        self.assertIs(self.student.get_submodule(self.blocks[1].path), self.blocks[1].module)
        for name, value in self.student.named_parameters():
            self.assertTrue(torch.equal(value, before[name]), name)
            self.assertEqual(value.requires_grad, flags[name], name)
        self.assertFalse(all(torch.equal(value, start[name]) for name, value in self.module.named_parameters()))


class ConfigurationContracts(unittest.TestCase):
    def test_every_recovery_group_holds_equal_sizes_and_its_extra_swiglu_widths(self):
        settings, unused_path = h.load_settings(CONFIG)
        groups = settings["single_block_recovery"]["size_groups"]
        members = [tuple(member) for values in groups.values() for member in values]
        tested = {(operator, width) for operator, widths in settings["recipe_selection"]["sizes"].items()
                  for width in widths}
        self.assertEqual(len(members), 10)
        self.assertEqual(set(members) - tested, {("swiglu", 683), ("swiglu", 2048)})
        for values in groups.values():
            shares = [h.item_units(operator, width, HIDDEN) / MLP_UNITS for operator, width in values]
            self.assertLess(max(shares) - min(shares), 2 / MLP_UNITS)


class JobContracts(unittest.TestCase):
    def setUp(self):
        self.source = SBATCH.read_text(encoding="utf-8")
        self.settings, unused_path = h.load_settings(CONFIG)

    def test_job_arrays_repeat_the_runners_task_maps(self):
        bounds = [int(value) for value in re.findall(r"array_index (\d+)", self.source)]
        self.assertEqual(bounds, [
            len(h.select_tasks(self.settings)) - 1,
            len(self.settings["allocation"]["eligible_layers"]) - 1,
            len(self.settings["recipe_selection"]["layers"]) - 1,
            len(h.TARGETS) - 1,
        ])
        layers = re.search(r"RECOVERY_LAYERS=\(([^)]*)\)", self.source).group(1).split()
        targets = re.search(r"ASSEMBLE_TARGETS=\(([^)]*)\)", self.source).group(1).split()
        self.assertEqual([int(value) for value in layers], self.settings["recipe_selection"]["layers"])
        self.assertEqual([float(value) for value in targets], list(h.TARGETS))

    def test_job_output_names_match_the_runner(self):
        self.assertEqual(h.select_output_name(7), "select-001-task-07")
        self.assertEqual(h.cost_output_name(1), "cost-table-001-layer-01")
        self.assertEqual(h.recovery_output_name(3), "recovery-001-layer-03")
        self.assertEqual(h.assemble_output_name(0.75), "assemble-001-target-0.75")
        for pattern in ("select-001-task-%02d", "cost-table-001-layer-%02d", "recovery-001-layer-%02d",
                        "assemble-001-target-${ASSEMBLE_TARGETS"):
            self.assertIn(pattern, self.source)


if __name__ == "__main__":
    unittest.main()
