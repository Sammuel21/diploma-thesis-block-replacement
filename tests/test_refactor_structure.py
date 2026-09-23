"""Read-only structural checks against the preserved refactor baseline.

This module needs only Python's standard library and the local Git history.
It never imports an experiment or runs a scientific workflow.
"""

import ast
import copy
import re
import subprocess
import unittest
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "dc11a031a07b217e1661adf0cb1f29b1ccb28393"
SOURCE = "src/mlp_replacement/"
WORKFLOW = "workflows/runs/model/swiglu/"
S5 = WORKFLOW + "swiglu_5.py"
PACKAGE = WORKFLOW + "swiglu5/"
RECONSTRUCTION = SOURCE + "compression/reconstruction.py"
MODULES = ("context", "fitting", "candidates", "recovery", "search", "confirmation")
SHARED_OWNERS = {
    **dict.fromkeys(("sha256_file", "fingerprint", "atomic_json", "atomic_torch_save"),
                    SOURCE + "artifacts.py"),
    **dict.fromkeys(("deep_merge", "make_model_config"), SOURCE + "config.py"),
    **dict.fromkeys(("autocast_context", "evaluate_lm_mixed", "evaluate_teacher_cache_mixed",
                     "evaluate_validation_kl_mixed"), SOURCE + "evaluation/mixed_precision.py"),
    "relative_to_root": "workflows/runs/model/common.py",
    "load_operator": RECONSTRUCTION,
    "replacement_state": RECONSTRUCTION,
    "load_replacement_state": RECONSTRUCTION,
}


def git(*arguments):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *arguments], cwd=ROOT
    ).decode("utf-8")


@lru_cache(maxsize=None)
def baseline_source(path):
    return git("show", f"{BASELINE}:{path}")


def current_source(path):
    return (ROOT / path).read_text(encoding="utf-8")


def definitions(source):
    return {
        node.name: node for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
    }


def canonical(node):
    """Ignore docstrings and the documented function-local Torch import only."""
    node = copy.deepcopy(node)
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        node.body.pop(0)
    if node.name == "atomic_torch_save":
        node.body = [item for item in node.body if not (
            isinstance(item, ast.Import) and ast.dump(item) == ast.dump(ast.parse("import torch").body[0])
        )]
    return ast.dump(node, include_attributes=False)


def moved_owner(path, name):
    if path == SOURCE + "compression/recovery.py":
        if name == "autocast_context":
            return SOURCE + "model.py"
        if name == "sha256_file":
            return SOURCE + "artifacts.py"
        return SOURCE + "compression/teacher_cache.py"
    if path == SOURCE + "runlog.py":
        return SOURCE + "artifacts.py"
    if path == S5:
        if name in ("replacement_state", "load_replacement_state"):
            return RECONSTRUCTION
        if name == "temporary_fp32_replacements":
            return SOURCE + "compression/surgery.py"
        for module in MODULES:
            target = PACKAGE + module + ".py"
            if name in definitions(current_source(target)):
                return target
        raise AssertionError(f"Missing SwiGLU-5 definition: {name}")
    return SHARED_OWNERS[name]


class RefactorStructure(unittest.TestCase):
    def test_all_maintained_python_and_contract_checks_parse(self):
        for directory in ("src", "workflows", "tests"):
            for path in (ROOT / directory).rglob("*.py"):
                with self.subTest(path=path.relative_to(ROOT)):
                    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_original_definitions_are_preserved_at_their_owners(self):
        adaptations = {
            (S5, "blank_candidate_student"),
            (WORKFLOW + "swiglu_6_recovery.py", "run_recovery"),
            (WORKFLOW + "swiglu_6_evaluate.py", "export_cohort_model"),
        }
        for path in git("ls-tree", "-r", "--name-only", BASELINE, "src", "workflows").splitlines():
            if not path.endswith(".py"):
                continue
            before = baseline_source(path)
            if path == WORKFLOW + "swiglu_3_calibration_recovery.py":
                before = re.sub(r"\bmodule_state\b", "values", before)
            current = definitions(current_source(path))
            for name, original in definitions(before).items():
                if (path, name) in adaptations:
                    continue
                owner = path if name in current else moved_owner(path, name)
                with self.subTest(path=path, name=name, owner=owner):
                    actual = definitions(current_source(owner))[name]
                    self.assertEqual(canonical(original), canonical(actual))

    def test_student_extraction_is_only_an_explicit_input_adaptation(self):
        before = ast.get_source_segment(baseline_source(S5),
                                       definitions(baseline_source(S5))["blank_candidate_student"])
        before = before.replace(
            "blank_candidate_student(context, model_config, candidate)",
            "build_swiglu_student(model_config, hidden_size, allocation_rows)",
        ).replace('context.settings["model"]["hidden_size"]', "hidden_size")
        before = before.replace('candidate["allocation"]', "allocation_rows")
        expected = definitions(before)["build_swiglu_student"]
        actual = definitions(current_source(RECONSTRUCTION))["build_swiglu_student"]
        self.assertEqual(canonical(expected), canonical(actual))
        wrapper = definitions(current_source(PACKAGE + "candidates.py"))["blank_candidate_student"]
        expected_wrapper = ast.parse('''def blank_candidate_student(context, model_config, candidate):
    return build_swiglu_student(model_config, context.settings["model"]["hidden_size"], candidate["allocation"])
''').body[0]
        self.assertEqual(canonical(expected_wrapper), canonical(wrapper))

    def test_cli_definitions_constants_and_failures_are_unchanged(self):
        for filename in ("swiglu_5_search.py", "swiglu_5_confirmation.py", "swiglu_6_prepare.py"):
            path = WORKFLOW + filename
            self.assertEqual(ast.dump(ast.parse(baseline_source(path))),
                             ast.dump(ast.parse(current_source(path))))
        for filename in ("swiglu_6_recovery.py", "swiglu_6_evaluate.py"):
            path = WORKFLOW + filename
            old = definitions(baseline_source(path))
            new = definitions(current_source(path))
            for name in ("main",):
                self.assertEqual(canonical(old[name]), canonical(new[name]))
        assignments = lambda text: [ast.dump(node) for node in ast.parse(text).body
                                    if isinstance(node, ast.Assign)]
        self.assertEqual(assignments(baseline_source(S5)),
                         assignments(current_source(PACKAGE + "context.py")))

    def test_swiglu6_only_adapts_reconstruction_inputs(self):
        changes = {
            "swiglu_6_recovery.py": (
                "run_recovery",
                'context = SimpleNamespace(settings=settings)\n'
                '    student, target_paths, train_modules = blank_candidate_student(context, model_config, candidate)',
                'student, target_paths, train_modules = build_swiglu_student('
                'model_config, settings["model"]["hidden_size"], candidate["allocation"])',
            ),
            "swiglu_6_evaluate.py": (
                "export_cohort_model",
                'model, paths, modules = blank_candidate_student(SimpleNamespace(settings=settings), model_config,\n'
                '                                                        {"allocation": row["allocation"]})',
                'model, paths, modules = build_swiglu_student('
                'model_config, settings["model"]["hidden_size"], row["allocation"])',
            ),
        }
        for filename, (name, old, new) in changes.items():
            source = baseline_source(WORKFLOW + filename)
            self.assertEqual(source.count(old), 1)
            source = source.replace(old, new).replace(
                'from .swiglu_5 import blank_candidate_student, load_replacement_state',
                'from mlp_replacement.compression.reconstruction import build_swiglu_student, load_replacement_state',
            )
            self.assertEqual(canonical(definitions(source)[name]),
                             canonical(definitions(current_source(WORKFLOW + filename))[name]))

    def test_dependency_boundaries(self):
        allowed = {
            "context": set(), "fitting": {"context"},
            "candidates": {"context", "fitting"},
            "recovery": {"context", "candidates"},
            "search": {"context", "fitting", "candidates", "recovery"},
            "confirmation": {"context", "fitting", "candidates", "recovery"},
        }
        for module in MODULES:
            for node in ast.walk(ast.parse(current_source(PACKAGE + module + ".py"))):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn("swiglu_5", node.module or "")
                    if node.level == 1:
                        self.assertIn(node.module, allowed[module])
        for path in (ROOT / "src").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith("workflows"))
                elif isinstance(node, ast.Import):
                    self.assertFalse(any(alias.name.startswith("workflows") for alias in node.names))
        for filename in ("swiglu_6_recovery.py", "swiglu_6_evaluate.py"):
            text = current_source(WORKFLOW + filename)
            self.assertNotIn("swiglu_5", text)
            self.assertNotIn("swiglu5", text)
        for node in ast.walk(ast.parse(current_source(SOURCE + "compression/teacher_cache.py"))):
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn("recovery", node.module or "")
        facade = ast.parse(current_source(S5))
        exported = {alias.asname or alias.name for node in facade.body
                    if isinstance(node, ast.ImportFrom) for alias in node.names}
        self.assertNotIn("*", exported)
        self.assertTrue(set(definitions(baseline_source(S5))) <= exported)


if __name__ == "__main__":
    unittest.main()
