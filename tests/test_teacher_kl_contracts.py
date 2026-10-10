"""Contracts for the shared long-context teacher-KL diagnostic."""

import copy
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlp_replacement.artifacts import file_digest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None


def load_teacher_kl_module():
    """Load the module without the evaluation package's eager Torch imports."""

    spec = importlib.util.spec_from_file_location(
        "teacher_kl_contract_impl", ROOT / "src/mlp_replacement/evaluation/teacher_kl.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CharacterTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens, return_attention_mask):
        return types.SimpleNamespace(input_ids=[ord(word[0]) for word in text.split()])


class TeacherKLStreamContracts(unittest.TestCase):
    def test_protocol_identifier_is_frozen(self):
        module = load_teacher_kl_module()
        self.assertEqual(
            module.TEACHER_KL_PROTOCOL,
            "teacher-kl-t1-all-valid-positions-c4-shard0-24x8192-v1",
        )

    def test_stream_packs_the_first_documents_of_the_pinned_streaming_source(self):
        module = load_teacher_kl_module()
        calls = []
        records = [{"text": "a b"}, {"text": " "}, {"text": "c d e"}, {"text": "f g"}, {"text": "h"}]

        def load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return iter(records)

        source = {
            "path": "allenai/c4",
            "split": "train",
            "text_column": "text",
            "revision": "pinned",
            "data_file": "en/c4-train.00000-of-01024.json.gz",
        }
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            sys.modules, {"datasets": types.SimpleNamespace(load_dataset=load_dataset)}
        ):
            path = Path(temporary) / "kl.int32"
            record = module.build_teacher_kl_stream(source, CharacterTokenizer(), path, 2, 4)
            values = np.fromfile(path, dtype=np.int32).tolist()
        self.assertEqual(
            calls,
            [(("allenai/c4",), {"data_files": source["data_file"], "revision": "pinned",
                                "split": "train", "streaming": True})],
        )
        self.assertEqual(values, [97, 98, 0, 99, 100, 101, 0, 102])
        self.assertEqual(record["token_count"], 8)
        self.assertEqual(record["documents_consumed"], 3)


@unittest.skipUnless(TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE, "PyTorch and Transformers are required")
class TeacherKLEvaluationContracts(unittest.TestCase):
    def setUp(self):
        from mlp_replacement.evaluation import teacher_kl

        self.teacher_kl = teacher_kl
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "kl.int32"
        np.arange(16, dtype=np.int32).tofile(self.path)
        self.record = {"token_count": 16, "sha256": file_digest(self.path)}

    def tearDown(self):
        self.directory.cleanup()

    def tiny_model(self, seed, attention="sdpa"):
        import torch
        from transformers import AutoModelForCausalLM, LlamaConfig

        torch.manual_seed(seed)
        config = LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
        )
        return AutoModelForCausalLM.from_config(config, attn_implementation=attention).eval()

    def evaluate(self, student, teacher, record=None):
        return self.teacher_kl.evaluate_teacher_kl(
            student, teacher, self.path, record or self.record, 8, device="cpu"
        )

    def test_teacher_against_itself_scores_zero_over_every_position(self):
        teacher = self.tiny_model(0)
        result = self.evaluate(copy.deepcopy(teacher), teacher)
        self.assertEqual(set(result), {"kl", "valid_positions", "protocol", "stream_sha256"})
        self.assertLess(abs(result["kl"]), 1e-6)
        self.assertEqual(result["valid_positions"], 16)
        self.assertEqual(result["protocol"], self.teacher_kl.TEACHER_KL_PROTOCOL)
        self.assertEqual(result["stream_sha256"], self.record["sha256"])

    def test_different_student_scores_positive_kl(self):
        self.assertGreater(self.evaluate(self.tiny_model(1), self.tiny_model(0))["kl"], 1e-4)

    def test_non_sdpa_attention_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SDPA"):
            self.evaluate(self.tiny_model(0, "eager"), self.tiny_model(0))

    def test_stream_without_whole_sequences_is_rejected(self):
        np.arange(12, dtype=np.int32).tofile(self.path)
        record = {"token_count": 12, "sha256": file_digest(self.path)}
        model = self.tiny_model(0)
        with self.assertRaises(ValueError):
            self.evaluate(copy.deepcopy(model), model, record)


if __name__ == "__main__":
    unittest.main()
