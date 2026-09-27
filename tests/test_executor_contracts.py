"""Static contracts for local and Perun process launchers."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "workflows" / "jobs" / "local" / "run_model.sh"
PERUN = ROOT / "workflows" / "jobs" / "perun" / "run_model.sbatch"
PERUN_SWIGLU_7 = ROOT / "workflows" / "jobs" / "perun" / "swiglu_7.sbatch"


def workflow_modules(source):
    return dict(re.findall(
        r"^    ([a-z0-9-]+)\)\n        WORKFLOW_MODULE=\"([^\"]+)\"",
        source,
        flags=re.MULTILINE,
    ))


class ExecutorContracts(unittest.TestCase):
    def test_local_and_perun_launch_the_same_existing_modules(self):
        historical = {
            "compression-baseline": "workflows.runs.model.baseline.compression",
            "swiglu": "workflows.runs.model.swiglu.swiglu_initial",
            "swiglu-2": "workflows.runs.model.swiglu.swiglu_2_allocation",
        }
        self.assertEqual(
            workflow_modules(LOCAL.read_text(encoding="utf-8")),
            {
                **historical,
                "swiglu-7": "workflows.runs.model.swiglu.swiglu_7",
            },
        )
        self.assertEqual(workflow_modules(PERUN.read_text(encoding="utf-8")), historical)
        self.assertIn(
            "workflows.runs.model.swiglu.swiglu_7",
            PERUN_SWIGLU_7.read_text(encoding="utf-8"),
        )

    def test_existing_entries_keep_the_artifact_contract(self):
        local_source = LOCAL.read_text(encoding="utf-8")
        local_case = local_source[
            local_source.index('case "$WORKFLOW_NAME" in'):local_source.index("esac")
        ]
        self.assertEqual(
            re.findall(
                r'^\s+STORAGE_CONTRACT="directories"$',
                local_case,
                flags=re.MULTILINE,
            ),
            ['        STORAGE_CONTRACT="directories"'],
        )
        historical = local_case[:local_case.index("    swiglu-7)")]
        self.assertNotIn('STORAGE_CONTRACT="directories"', historical)

        perun_source = PERUN.read_text(encoding="utf-8")
        self.assertIn('STORAGE_CONTRACT="artifact"', perun_source)
        self.assertNotIn('STORAGE_CONTRACT="directories"', perun_source)
        self.assertIn('"--output"', perun_source)
        self.assertIn("Use workflows/jobs/perun/swiglu_7.sbatch", perun_source)

    def test_perun_swiglu7_uses_manual_bounded_scratch_and_project_stageout(self):
        source = PERUN_SWIGLU_7.read_text(encoding="utf-8")
        self.assertNotIn(".activate_scratch", source)
        self.assertIn('SCRATCH_ROOT="/mnt/scratch/${USER:', source)
        self.assertIn('PROJECT_RESULTS_ROOT="$PROJECT_ROOT/perun-results/swiglu-7"', source)
        self.assertIn('"$SCRATCH_ROOT"/job_*', source)
        self.assertIn("trap finalize_job EXIT", source)
        self.assertIn("rsync -a --delete --exclude='*.tmp'", source)
        self.assertIn("OUTPUT_READY_FOR_STAGEOUT", source)

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
