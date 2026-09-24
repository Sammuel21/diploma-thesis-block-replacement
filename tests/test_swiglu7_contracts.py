"""Focused CPU/static contracts for SwiGLU-7; no model or dataset is loaded."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn

from mlp_replacement.compression.adapters import (
    LoRALinear,
    install_lora_adapters,
    merge_lora_adapters,
)
from mlp_replacement.compression import continuation
from mlp_replacement.data import PackedTokenCache
from workflows.runs.model.swiglu import swiglu_7


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 3, bias=True)

    def forward(self, values):
        return self.projection(values)


class SwiGLU7Contracts(unittest.TestCase):
    def test_configuration_fixes_the_declared_grid_and_recipe(self):
        settings, path = swiglu_7.load_settings(
            ROOT / "workflows/configs/model/swiglu/swiglu-7.json"
        )
        self.assertEqual(path, ROOT / "workflows/configs/model/swiglu/swiglu-7.json")
        self.assertEqual(settings["targets"], [0.2, 0.3, 0.4, 0.5])
        self.assertEqual(list(settings["strategies"]), ["S7-0", "S7-1", "S7-2"])
        self.assertEqual(settings["recovery"]["learning_rate"], 3e-5)
        self.assertEqual(settings["recovery"]["sequence_length"], 8192)
        self.assertEqual(settings["recovery"]["effective_batch_tokens"], 8192)
        self.assertEqual(settings["recovery"]["target_tokens"], 1_000_000_000)
        self.assertEqual(
            settings["recovery"]["segment_endpoints"],
            [100_000_000, 1_000_000_000],
        )
        self.assertEqual(settings["preparation"]["branch_recovery"]["sequence_length"], 128)
        self.assertEqual(settings["evaluation"]["contexts"], [128, 2048, 8192])

    def test_exact_endpoint_may_use_one_short_final_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.int32"
            np.arange(16, dtype=np.int32).tofile(path)
            cache = PackedTokenCache(path, 16, 8, "fixture")
            with self.assertRaisesRegex(ValueError, "complete sequences"):
                cache.batch(8, 5, 1)
            batch = cache.batch(8, 5, 1, allow_partial_sequence=True)
            self.assertEqual(tuple(batch["input_ids"].shape), (1, 5))
            self.assertTrue(torch.equal(batch["input_ids"][0], torch.arange(8, 13)))

    def test_notebook_is_load_only_and_has_no_fabricated_outputs(self):
        notebook = json.loads(
            (ROOT / "notebooks/model/swiglu/swiglu-7.ipynb").read_text(encoding="utf-8")
        )
        self.assertEqual(notebook["nbformat"], 4)
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertNotIn("import torch", code)
        self.assertNotIn("transformers", code)
        self.assertTrue(
            all(
                cell.get("execution_count") is None and cell.get("outputs") == []
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
        )

    def test_directory_contract_rejects_overlap_and_occupied_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "non-nested"):
                swiglu_7.resolve_storage(root / "same", root / "same", False)
            with self.assertRaisesRegex(ValueError, "non-nested"):
                swiglu_7.resolve_storage(root / "parent", root / "parent/output", False)
            occupied = root / "occupied"
            occupied.mkdir()
            (occupied / "existing.txt").write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "non-empty"):
                swiglu_7.resolve_storage(root / "work", occupied, False)

    def test_single_checkpoint_replaces_only_after_new_descriptor_commits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for tokens in (128, 256):
                continuation.commit_single_checkpoint(
                    root,
                    {
                        "run_fingerprint": "run",
                        "tokens_seen": tokens,
                        "value": torch.tensor([tokens]),
                    },
                )
            records = continuation.valid_checkpoint_records(root, "run")
            self.assertEqual([row["tokens_seen"] for row in records], [256])
            self.assertEqual(len(list(root.glob("checkpoint-*.pt"))), 1)
            self.assertEqual(len(list(root.glob("checkpoint-*.json"))), 1)

            with patch.object(continuation, "write_json_atomic", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    continuation.commit_single_checkpoint(
                        root,
                        {
                            "run_fingerprint": "run",
                            "tokens_seen": 384,
                            "value": torch.tensor([384]),
                        },
                    )
            restored, record = continuation.restore_checkpoint(root, "run")
            self.assertEqual(record["tokens_seen"], 256)
            self.assertEqual(restored["tokens_seen"], 256)
            continuation.commit_single_checkpoint(
                root,
                {
                    "run_fingerprint": "run",
                    "tokens_seen": 512,
                    "value": torch.tensor([512]),
                },
            )
            self.assertEqual(
                {path.name for path in root.glob("checkpoint-*")},
                {
                    "checkpoint-000000000512.pt",
                    "checkpoint-000000000512.json",
                },
            )

    def test_lora_merge_preserves_eval_output_and_removes_wrapper(self):
        torch.manual_seed(7)
        model = TinyModel().eval()
        adapters = install_lora_adapters(model, ["projection"], rank=2, alpha=4, dropout=0.0)
        with torch.no_grad():
            adapters["projection"].lora_b.copy_(torch.randn_like(adapters["projection"].lora_b))
        values = torch.randn(5, 4)
        expected = model(values)
        records = merge_lora_adapters(model, adapters)
        actual = model(values)
        self.assertEqual(records[0]["path"], "projection")
        self.assertIsInstance(model.projection, nn.Linear)
        self.assertFalse(any(isinstance(module, LoRALinear) for module in model.modules()))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_optimizer_expansion_preserves_replacement_moments_only(self):
        source = {
            "state": {
                0: {"step": torch.tensor(4.0), "exp_avg": torch.tensor([1.0])},
                1: {"step": torch.tensor(4.0), "exp_avg": torch.tensor([2.0])},
            },
            "param_groups": [
                {
                    "params": [0, 1],
                    "lr": 3e-5,
                    "initial_lr": 3e-5,
                    "weight_decay": 0.0,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                }
            ],
        }
        groups = [
            {
                "parameters": [nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))],
                "learning_rate": 3e-5,
                "weight_decay": 0.0,
            },
            {
                "parameters": [nn.Parameter(torch.zeros(1))],
                "learning_rate": 3e-5,
                "weight_decay": 0.0,
            },
        ]
        expanded = swiglu_7.optimizer_state_for_groups(source, groups)
        self.assertEqual(expanded["param_groups"][0]["params"], [0, 1])
        self.assertEqual(expanded["param_groups"][1]["params"], [2])
        self.assertEqual(set(expanded["state"]), {0, 1})
        self.assertNotIn(2, expanded["state"])


if __name__ == "__main__":
    unittest.main()
