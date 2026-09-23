"""Serialization and durability contracts, independent of scientific runs."""

import hashlib
import math
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlp_replacement import artifacts


@dataclass
class Metric:
    value: float


class ArtifactContracts(unittest.TestCase):
    def test_normalization_and_fingerprint_bytes(self):
        value = {"z": (Path("asset.pt"), Metric(2.5)), "a": [math.nan, math.inf, -math.inf], "unicode": "λ"}
        normalized = {"z": ["asset.pt", {"value": 2.5}], "a": ["nan", "inf", "-inf"], "unicode": "λ"}
        self.assertEqual(artifacts.json_value(value), normalized)
        encoded = '{"a":["nan","inf","-inf"],"unicode":"λ","z":["asset.pt",{"value":2.5}]}'.encode("utf-8")
        self.assertEqual(artifacts.fingerprint(value), hashlib.sha256(encoded).hexdigest())
        self.assertEqual(artifacts.fingerprint(value), artifacts.fingerprint(dict(reversed(list(value.items())))))
        with self.assertRaises(ValueError):
            artifacts.content_digest({"value": math.nan})

    def test_legacy_and_durable_json_remain_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "legacy.json"
            durable = Path(directory) / "durable.json"
            text = '{\n  "value": "inf",\n  "label": "λ"\n}'
            with patch.object(artifacts.os, "fsync", wraps=os.fsync) as sync:
                artifacts.atomic_json(legacy, {"value": math.inf, "label": "λ"})
                sync.assert_not_called()
                artifacts.write_json_atomic(durable, {"value": "inf", "label": "λ"})
                self.assertEqual(sync.call_count, 1)
            self.assertEqual(legacy.read_bytes(), text.replace("\n", os.linesep).encode("utf-8"))
            self.assertEqual(durable.read_bytes(), (text + "\n").replace("\n", os.linesep).encode("utf-8"))
            self.assertFalse(legacy.with_suffix(".json.tmp").exists())
            self.assertFalse(durable.with_suffix(".json.tmp").exists())
            before = durable.read_bytes()
            with self.assertRaises(ValueError):
                artifacts.write_json_atomic(durable, {"value": math.nan})
            self.assertEqual(durable.read_bytes(), before)

    def test_file_hash_and_prefix_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            path.write_bytes(b"abc\x00def")
            self.assertEqual(artifacts.sha256_file(path), hashlib.sha256(b"abc\x00def").hexdigest())
            self.assertEqual(artifacts.file_digest(path), artifacts.sha256_file(path))
            self.assertEqual(artifacts.file_digest(path, 3), hashlib.sha256(b"abc").hexdigest())
            with self.assertRaises(ValueError):
                artifacts.file_digest(path, 8)

    def test_torch_writer_durability_and_payload(self):
        import torch
        from mlp_replacement.compression.continuation import save_tensor_atomic
        from mlp_replacement.runlog import json_value

        self.assertIs(json_value, artifacts.json_value)
        with tempfile.TemporaryDirectory() as directory:
            payload = {"weights": torch.arange(6, dtype=torch.float32)}
            for writer, sync_count in ((artifacts.atomic_torch_save, 0), (save_tensor_atomic, 1)):
                path = Path(directory) / "state.pt"
                with patch.object(artifacts.os, "fsync", wraps=os.fsync) as sync:
                    writer(path, payload)
                    self.assertEqual(sync.call_count, sync_count)
                self.assertTrue(torch.equal(torch.load(path, weights_only=True)["weights"], payload["weights"]))
                self.assertFalse(path.with_suffix(".pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
