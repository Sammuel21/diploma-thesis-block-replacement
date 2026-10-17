"""Contracts for SwiGLU local-fitting schedules, least-squares refits, and sampling."""

import math
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from mlp_replacement.capture import ActivationPairs
from mlp_replacement.config import OperatorConfig
from mlp_replacement.data import sample_partitioned_windows
from mlp_replacement.operators import (
    GatedMLPReplacement,
    fit_operator_fp32_detailed,
    refit_down_projection_least_squares,
)


def pairs(seed, tokens=64, hidden=8):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(tokens, hidden, generator=generator)
    return ActivationPairs(inputs=inputs, targets=torch.randn(tokens, hidden, generator=generator))


def fit(scheduler, epochs=5, batch_size=16, **options):
    torch.manual_seed(0)
    module = GatedMLPReplacement(8, 4)
    config = OperatorConfig(
        kind="swiglu", initialization="importance_teacher_subset", epochs=epochs,
        learning_rate=0.01, batch_size=batch_size, scheduler=scheduler, early_stopping_patience=None,
    )
    return fit_operator_fp32_detailed(module, pairs(1), pairs(2), config, "cpu", **options)


def distance_from_start(result):
    torch.manual_seed(0)
    start = GatedMLPReplacement(8, 4).state_dict()
    return sum(float((value - start[name]).square().sum()) for name, value in result.module.state_dict().items())


class LocalFittingContracts(unittest.TestCase):
    def test_constant_schedule_keeps_the_learning_rate(self):
        history = fit("constant").history
        self.assertEqual([epoch.learning_rate for epoch in history], [0.01] * 5)

    def test_cosine_schedule_decays_once_per_epoch_over_the_epoch_budget(self):
        history = fit("cosine").history
        expected = [0.01 * (1 + math.cos(math.pi * epoch / 5)) / 2 for epoch in range(5)]
        for observed, value in zip((epoch.learning_rate for epoch in history), expected):
            self.assertAlmostEqual(observed, value, places=12)

    def test_unknown_schedule_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "constant or cosine"):
            fit("linear")

    def test_zero_l2_sp_strength_is_the_plain_mse_fit(self):
        plain = fit("constant")
        explicit = fit("constant", l2_sp_alpha=0.0)

        def without_timing(result):
            return [{**vars(epoch), "elapsed_seconds": None} for epoch in result.history]

        self.assertEqual(without_timing(plain), without_timing(explicit))
        for name, value in plain.module.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.module.state_dict()[name]))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            fit("constant", l2_sp_alpha=-1.0)

    def test_l2_sp_pulls_the_fit_toward_its_starting_point(self):
        # AdamW's normalized steps keep a strongly penalized fit about one step from w0,
        # so the check is a monotone pull rather than an exact return.
        distances = [distance_from_start(fit("constant", l2_sp_alpha=alpha)) for alpha in (0.0, 0.1, 1.0, 10.0)]
        self.assertEqual(distances, sorted(distances, reverse=True))
        self.assertEqual(len(set(distances)), len(distances))
        self.assertLess(distances[-1], 0.05 * distances[0])

    def test_l2_sp_history_records_the_data_mse_only(self):
        # One full-batch epoch records the MSE at the starting weights, before any update.
        plain = fit("constant", epochs=1, batch_size=64)
        regularized = fit("constant", epochs=1, batch_size=64, l2_sp_alpha=10.0)
        self.assertEqual(plain.history[0].train_mse, regularized.history[0].train_mse)
        torch.manual_seed(0)
        start = GatedMLPReplacement(8, 4)
        training = pairs(1)
        with torch.no_grad():
            expected = float((start(training.inputs) - training.targets).square().mean())
        self.assertAlmostEqual(plain.history[0].train_mse, expected, places=6)

    def test_least_squares_recovers_an_exactly_representable_down_projection(self):
        torch.manual_seed(3)
        reference = GatedMLPReplacement(8, 4).double()
        inputs = torch.randn(256, 8, dtype=torch.float64)
        with torch.no_grad():
            targets = reference(inputs)
        student = GatedMLPReplacement(8, 4)
        with torch.no_grad():
            student.gate_projection.weight.copy_(reference.gate_projection.weight)
            student.up_projection.weight.copy_(reference.up_projection.weight)
            student.down_projection.weight.zero_()
        record = refit_down_projection_least_squares(
            student, ActivationPairs(inputs=inputs.float(), targets=targets.float()), "cpu", 64
        )
        self.assertEqual(record["calibration_pairs"], 256)
        self.assertGreaterEqual(record["gram_condition_number"], 1.0)
        self.assertTrue(torch.allclose(student.down_projection.weight.double(),
                                       reference.down_projection.weight, atol=1e-4))
        with self.assertRaises(ValueError):
            refit_down_projection_least_squares(GatedMLPReplacement(8, 4, down_bias=True),
                                                pairs(4), "cpu", 64)


class Tokenizer:
    def __call__(self, text, return_tensors, add_special_tokens):
        ids = torch.tensor([[int(value) for value in text.split()]])
        return types.SimpleNamespace(input_ids=ids)


class WindowSamplingContracts(unittest.TestCase):
    def test_excluded_records_are_skipped_and_none_keeps_the_sampling_unchanged(self):
        records = [{"text": " ".join(str(100 * index + offset) for offset in range(6))}
                   for index in range(40)]
        sizes = {"calibration": 6}
        plain = sample_partitioned_windows(records, Tokenizer(), sizes, 4, 21)
        unchanged = sample_partitioned_windows(records, Tokenizer(), sizes, 4, 21, excluded_records=None)
        for first, second in zip(plain["calibration"], unchanged["calibration"]):
            self.assertTrue(torch.equal(first, second))
        excluded = set(range(20))
        indices = []
        sample_partitioned_windows(records, Tokenizer(), {"extension": 10}, 4, 21, record_indices=indices,
                                   excluded_records=excluded)
        self.assertEqual(len(set(indices)), 10)
        self.assertFalse(set(indices) & excluded)
        with self.assertRaises(RuntimeError):
            sample_partitioned_windows(records, Tokenizer(), {"extension": 21}, 4, 21, excluded_records=excluded)

    def test_record_indices_follow_windows_without_changing_them(self):
        records = [{"text": " ".join(str(100 * index + offset) for offset in range(6))}
                   for index in range(12)]
        sizes = {"calibration": 3, "validation": 2}
        plain = sample_partitioned_windows(records, Tokenizer(), sizes, 4, 21)
        indices = []
        recorded = sample_partitioned_windows(records, Tokenizer(), sizes, 4, 21, record_indices=indices)
        windows = [*recorded["calibration"], *recorded["validation"]]
        self.assertEqual(len(indices), 5)
        self.assertEqual(len(set(indices)), 5)
        for name in sizes:
            for first, second in zip(plain[name], recorded[name]):
                self.assertTrue(torch.equal(first, second))
        for index, window in zip(indices, windows):
            self.assertEqual(int(window[0]) // 100, index)


if __name__ == "__main__":
    unittest.main()
