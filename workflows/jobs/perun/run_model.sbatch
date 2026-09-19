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
    echo "Workflows: compression-baseline, swiglu, swiglu-2" >&2
    exit 2
fi

WORKFLOW_NAME="$1"
shift
case "$WORKFLOW_NAME" in
    compression-baseline)
        WORKFLOW_MODULE="workflows.runs.model.baseline.compression"
        ;;
    swiglu)
        WORKFLOW_MODULE="workflows.runs.model.swiglu.initial"
        ;;
    swiglu-2)
        WORKFLOW_MODULE="workflows.runs.model.swiglu.allocation"
        ;;
    *)
        echo "Unsupported model workflow: $WORKFLOW_NAME" >&2
        exit 2
        ;;
esac

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

PYTHON_BIN="${MLP_REPLACEMENT_PYTHON:-python3}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Job: ${SLURM_JOB_ID} on ${SLURM_NODELIST}"
echo "Work directory: $PWD"
echo "Workflow: $WORKFLOW_NAME"
echo "Module: $WORKFLOW_MODULE"
echo "Python: $PYTHON_BIN"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

"$PYTHON_BIN" -m "$WORKFLOW_MODULE" "$@"
