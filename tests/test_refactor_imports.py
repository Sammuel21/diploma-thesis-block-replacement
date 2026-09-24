"""Import compatibility checks; no model loading, evaluation, or recovery."""

import ast
import importlib
import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
WORKFLOW = "workflows.runs.model.swiglu."


class RefactorImports(unittest.TestCase):
    def test_swiglu5_explicit_exports_resolve_to_original_owners(self):
        facade = importlib.import_module(WORKFLOW + "swiglu_5")
        source = ROOT / "workflows/runs/model/swiglu/swiglu_5.py"
        for node in ast.parse(source.read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.ImportFrom):
                continue
            module = importlib.import_module("." * node.level + node.module, facade.__package__)
            for alias in node.names:
                self.assertIs(getattr(facade, alias.asname or alias.name), getattr(module, alias.name))

    def test_legacy_shared_helpers_and_autocast_signatures(self):
        shared = importlib.import_module(WORKFLOW + "shared")
        s3 = importlib.import_module(WORKFLOW + "swiglu_3_calibration_recovery")
        mixed = importlib.import_module("mlp_replacement.evaluation.mixed_precision")
        recovery = importlib.import_module("mlp_replacement.compression.recovery")
        model = importlib.import_module("mlp_replacement.model")
        for name in ("autocast_context", "evaluate_lm_mixed", "evaluate_teacher_cache_mixed",
                     "evaluate_validation_kl_mixed"):
            self.assertIs(getattr(shared, name), getattr(mixed, name))
        self.assertIs(s3.evaluate_lm_mixed, mixed.evaluate_lm_mixed)
        self.assertIs(s3.evaluate_validation_kl_mixed, mixed.evaluate_validation_kl_mixed)
        self.assertIsNot(s3.evaluate_teacher_cache_mixed, mixed.evaluate_teacher_cache_mixed)
        self.assertIs(recovery.autocast_context, model.autocast_context)
        self.assertEqual(list(inspect.signature(mixed.autocast_context).parameters), ["device"])
        self.assertEqual(list(inspect.signature(model.autocast_context).parameters), ["device", "dtype"])
        for owner, names in (
            ("artifacts", ("sha256_file", "fingerprint", "atomic_json", "atomic_torch_save")),
            ("config", ("deep_merge", "make_model_config")),
            ("compression.reconstruction", ("load_operator",)),
        ):
            module = importlib.import_module("mlp_replacement." + owner)
            for name in names:
                self.assertIs(getattr(shared, name), getattr(module, name))

    def test_cli_and_swiglu6_imports_without_loading_models(self):
        for name in ("swiglu_5_search", "swiglu_5_confirmation", "swiglu_6_prepare",
                     "swiglu_6_recovery", "swiglu_6_evaluate", "swiglu_7"):
            module = importlib.import_module(WORKFLOW + name)
            self.assertTrue(callable(module.main))


if __name__ == "__main__":
    unittest.main()
