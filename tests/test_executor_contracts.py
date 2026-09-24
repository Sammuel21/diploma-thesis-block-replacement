"""Static contracts for local and Perun process launchers."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "workflows" / "jobs" / "local" / "run_model.sh"
PERUN = ROOT / "workflows" / "jobs" / "perun" / "run_model.sbatch"


def workflow_modules(source):
    return dict(re.findall(
        r"^    ([a-z0-9-]+)\)\n        WORKFLOW_MODULE=\"([^\"]+)\"",
        source,
        flags=re.MULTILINE,
    ))


class ExecutorContracts(unittest.TestCase):
    def test_local_and_perun_launch_the_same_existing_modules(self):
        expected = {
            "compression-baseline": "workflows.runs.model.baseline.compression",
            "swiglu": "workflows.runs.model.swiglu.swiglu_initial",
            "swiglu-2": "workflows.runs.model.swiglu.swiglu_2_allocation",
            "swiglu-7": "workflows.runs.model.swiglu.swiglu_7",
        }
        self.assertEqual(workflow_modules(LOCAL.read_text(encoding="utf-8")), expected)
        self.assertEqual(workflow_modules(PERUN.read_text(encoding="utf-8")), expected)

    def test_existing_entries_keep_the_artifact_contract(self):
        for path in (LOCAL, PERUN):
            source = path.read_text(encoding="utf-8")
            self.assertIn('STORAGE_CONTRACT="artifact"', source)
            case_body = source[source.index('case "$WORKFLOW_NAME" in'):source.index("esac")]
            self.assertEqual(
                re.findall(
                    r'^\s+STORAGE_CONTRACT="directories"$',
                    case_body,
                    flags=re.MULTILINE,
                ),
                ['        STORAGE_CONTRACT="directories"'],
            )
            historical = case_body[:case_body.index("    swiglu-7)")]
            self.assertNotIn('STORAGE_CONTRACT="directories"', historical)
            self.assertIn('"--output"', source)

    def test_perun_directory_contract_uses_only_job_scratch_paths(self):
        source = PERUN.read_text(encoding="utf-8")
        self.assertIn('${TMPDIR:?Perun scratch did not define TMPDIR}/mlp-replacement/', source)
        self.assertIn('${RESULTS_DIR:?Perun scratch did not define RESULTS_DIR}/', source)
        self.assertIn('"$TMPDIR"/mlp-replacement/*', source)
        self.assertIn('run_model.sbatch owns --work-dir', source)
        self.assertIn("trap cleanup_work_dir EXIT", source)

    def test_local_directory_contract_uses_bounded_defaults(self):
        source = LOCAL.read_text(encoding="utf-8")
        self.assertIn('WORK_DIR="data/work/${WORKFLOW_NAME}/${RUN_ID}"', source)
        self.assertIn("MLP_REPLACEMENT_RUN_ID must be one safe path component", source)
        self.assertIn(
            'OUTPUT_DIR="data/results/workflows/model/${WORKFLOW_NAME}/${RUN_ID}"',
            source,
        )
        self.assertIn('data/work/"$WORKFLOW_NAME"/*', source)
        self.assertIn("trap cleanup_work_dir EXIT", source)

    def test_rsync_ignore_is_narrow(self):
        lines = (ROOT / ".rsyncignore").read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines, ["/tmp/mlp-replacement/"])


if __name__ == "__main__":
    unittest.main()
