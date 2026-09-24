#!/bin/bash
#SBATCH --job-name=mlp-model-workflow
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Supply --account and --qos to sbatch; see README.md.
#SBATCH --partition=gpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --export=ALL

if ! source .activate_scratch; then
    echo "Perun scratch activation failed; submit this job from the repository root." >&2
    exit 1
fi
set -euo pipefail

if (( $# < 1 )); then
    echo "Usage: sbatch [Slurm options] $0 WORKFLOW [WORKFLOW_ARGUMENTS...]" >&2
    echo "Workflows: compression-baseline, swiglu, swiglu-2, swiglu-7" >&2
    exit 2
fi

WORKFLOW_NAME="$1"
shift
STORAGE_CONTRACT="artifact"
case "$WORKFLOW_NAME" in
    compression-baseline)
        WORKFLOW_MODULE="workflows.runs.model.baseline.compression"
        ;;
    swiglu)
        WORKFLOW_MODULE="workflows.runs.model.swiglu.swiglu_initial"
        ;;
    swiglu-2)
        WORKFLOW_MODULE="workflows.runs.model.swiglu.swiglu_2_allocation"
        ;;
    swiglu-7)
        WORKFLOW_MODULE="workflows.runs.model.swiglu.swiglu_7"
        STORAGE_CONTRACT="directories"
        ;;
    *)
        echo "Unsupported model workflow: $WORKFLOW_NAME" >&2
        exit 2
        ;;
esac

if [[ "$STORAGE_CONTRACT" == "artifact" ]]; then
    HAS_OUTPUT=0
    for argument in "$@"; do
        if [[ "$argument" == "--output" || "$argument" == --output=* ]]; then
            HAS_OUTPUT=1
            break
        fi
    done
    if (( HAS_OUTPUT == 0 )); then
        set -- "$@" --output "data/results/perun/model/${WORKFLOW_NAME}-${SLURM_JOB_ID}.json"
    fi
else
    WORK_DIR="${TMPDIR:?Perun scratch did not define TMPDIR}/mlp-replacement/${WORKFLOW_NAME}-${SLURM_JOB_ID}"
    HAS_OUTPUT_DIR=0
    for argument in "$@"; do
        if [[ "$argument" == "--work-dir" || "$argument" == --work-dir=* ]]; then
            echo "run_model.sbatch owns --work-dir; it must remain below TMPDIR." >&2
            exit 2
        fi
        if [[ "$argument" == "--output-dir" || "$argument" == --output-dir=* ]]; then
            HAS_OUTPUT_DIR=1
        fi
    done
    if (( HAS_OUTPUT_DIR == 0 )); then
        OUTPUT_DIR="${RESULTS_DIR:?Perun scratch did not define RESULTS_DIR}/${WORKFLOW_NAME}-${SLURM_JOB_ID}"
        set -- "$@" --output-dir "$OUTPUT_DIR"
    fi
    case "$WORK_DIR" in
        "$TMPDIR"/mlp-replacement/*) ;;
        *)
            echo "Refusing unsafe temporary path: $WORK_DIR" >&2
            exit 2
            ;;
    esac
    cleanup_work_dir() {
        rm -rf -- "$WORK_DIR"
    }
    trap cleanup_work_dir EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    set -- "$@" --work-dir "$WORK_DIR"
fi

PYTHON_BIN="${MLP_REPLACEMENT_PYTHON:-python3}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Job: ${SLURM_JOB_ID} on ${SLURM_NODELIST}"
echo "Checkout directory: $PWD"
echo "Workflow: $WORKFLOW_NAME"
echo "Module: $WORKFLOW_MODULE"
echo "Python: $PYTHON_BIN"
if [[ "$STORAGE_CONTRACT" == "directories" ]]; then
    echo "Temporary workflow directory: $WORK_DIR"
    if (( HAS_OUTPUT_DIR == 0 )); then
        echo "Durable output directory: $OUTPUT_DIR"
    else
        echo "Durable output directory: forwarded --output-dir"
    fi
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

"$PYTHON_BIN" -m "$WORKFLOW_MODULE" "$@"
