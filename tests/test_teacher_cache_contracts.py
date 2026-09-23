"""CPU tensor/file invariants; no model fitting or recovery is executed."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from mlp_replacement.compression import recovery, teacher_cache


class TeacherCacheContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.values = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        shards = []
        for index, start in enumerate((0, 4)):
            path = self.root / f"shard-{index}.pt"
            torch.save(self.values[start : start + 4].clone(), path)
            shards.append({
                "token_start": start, "token_end": start + 4,
                "file": path.name, "sha256": teacher_cache.sha256_file(path),
            })
        self.manifest = {
            "fingerprint": "contract", "token_count": 8, "sequence_length": 2,
            "hidden_size": 2, "dtype": "float32", "token_fingerprint": "tokens",
            "head_fingerprint": "head", "shards": shards,
        }
        self.write_manifest(self.manifest)

    def tearDown(self):
        teacher_cache.load_hidden_shard.cache_clear()
        self.temporary.cleanup()

    def write_manifest(self, manifest):
        teacher_cache.atomic_json(self.root / "manifest.json", manifest)

    def test_reads_and_release_keep_cache_contract(self):
        cache = teacher_cache.load_teacher_hidden_cache(self.root, "contract")
        self.assertTrue(torch.equal(cache.batch(0, 2), self.values[:2]))
        self.assertTrue(torch.equal(cache.batch(2, 4), self.values[2:6]))
        self.assertEqual(teacher_cache.load_hidden_shard.cache_info().maxsize, 2)
        self.assertEqual(teacher_cache.load_hidden_shard.cache_info().currsize, 2)
        self.assertEqual(cache.batch(0, 2).dtype, torch.float32)
        cache.release()
        self.assertEqual(teacher_cache.load_hidden_shard.cache_info().currsize, 0)

    def test_invalid_ranges_are_rejected(self):
        cache = teacher_cache.load_teacher_hidden_cache(self.root)
        for offset, count in ((-1, 1), (0, 0), (7, 2)):
            with self.subTest(offset=offset, count=count), self.assertRaises(ValueError):
                cache.batch(offset, count)

    def test_manifest_fingerprint_and_coverage_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "fingerprint differs"):
            teacher_cache.load_teacher_hidden_cache(self.root, "foreign")
        for field, value in (("token_start", 5), ("token_end", 4)):
            manifest = copy.deepcopy(self.manifest)
            manifest["shards"][1][field] = value
            self.write_manifest(manifest)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "coverage"):
                teacher_cache.load_teacher_hidden_cache(self.root)

    def test_changed_missing_and_unexpected_files_are_rejected(self):
        path = self.root / "shard-0.pt"
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            teacher_cache.load_teacher_hidden_cache(self.root)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            teacher_cache.load_teacher_hidden_cache(self.root)
        torch.save(self.values[:4].clone(), path)
        self.manifest["shards"][0]["sha256"] = teacher_cache.sha256_file(path)
        self.write_manifest(self.manifest)
        (self.root / "unexpected.txt").write_text("extra", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            teacher_cache.load_teacher_hidden_cache(self.root)

    def test_legacy_exports_keep_class_and_cache_identity(self):
        for name in (
            "TeacherBatch", "TeacherCache", "TeacherHiddenShard", "TeacherFinalHiddenCache",
            "load_hidden_shard", "load_teacher_hidden_cache", "cache_teacher_logits",
            "build_teacher_final_hidden_cache", "validate_teacher_final_hidden_cache",
        ):
            self.assertIs(getattr(recovery, name), getattr(teacher_cache, name))


if __name__ == "__main__":
    unittest.main()
