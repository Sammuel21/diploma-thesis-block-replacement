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


def fit(scheduler, epochs=5):
    torch.manual_seed(0)
    module = GatedMLPReplacement(8, 4)
    config = OperatorConfig(
        kind="swiglu", initialization="importance_teacher_subset", epochs=epochs,
        learning_rate=0.01, batch_size=16, scheduler=scheduler, early_stopping_patience=None,
    )
    return fit_operator_fp32_detailed(module, pairs(1), pairs(2), config, "cpu")


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


class WindowSamplingContracts(unittest.TestCase):
    def test_record_indices_follow_windows_without_changing_them(self):
        class Tokenizer:
            def __call__(self, text, return_tensors, add_special_tokens):
                ids = torch.tensor([[int(value) for value in text.split()]])
                return types.SimpleNamespace(input_ids=ids)

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
