#!/bin/bash
#SBATCH --job-name=swiglu-7
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Supply --account and --qos to sbatch; see README.md.
#SBATCH --partition=gpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --export=ALL

if ! source .activate_scratch; then
    echo "Perun scratch activation failed; submit this job from the repository root." >&2
    exit 1
fi
set -euo pipefail

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    echo "Usage: sbatch $0 prepare" >&2
    echo "   or: sbatch --array=0-11[%N] $0 train PREPARED_RESULT" >&2
    exit 2
fi
shift

PYTHON_BIN="${MLP_REPLACEMENT_PYTHON:-python3}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

TASK_SUFFIX="${SLURM_ARRAY_TASK_ID:-single}"
WORK_DIR="${TMPDIR:?Perun scratch did not define TMPDIR}/mlp-replacement/swiglu-7-${SLURM_JOB_ID}-${TASK_SUFFIX}"
case "$WORK_DIR" in
    "$TMPDIR"/mlp-replacement/swiglu-7-*) ;;
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

case "$MODE" in
    prepare)
        if [[ -n "${SLURM_ARRAY_TASK_ID:-}" || $# -ne 0 ]]; then
            echo "Preparation is one non-array job with no positional arguments." >&2
            exit 2
        fi
        OUTPUT_DIR="${RESULTS_DIR:?Perun scratch did not define RESULTS_DIR}/swiglu-7/prepare-001"
        RUN_ARGUMENTS=(prepare --work-dir "$WORK_DIR" --output-dir "$OUTPUT_DIR")
        ;;
    train)
        if [[ $# -ne 1 || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
            echo "Training requires PREPARED_RESULT and --array=0-11." >&2
            exit 2
        fi
        TASK_ID="$SLURM_ARRAY_TASK_ID"
        if (( TASK_ID < 0 || TASK_ID > 11 )); then
            echo "SwiGLU-7 array index must be between 0 and 11." >&2
            exit 2
        fi
        PREPARED_RESULT="$1"
        OUTPUT_ROOT="${RESULTS_DIR:?Perun scratch did not define RESULTS_DIR}/swiglu-7"
        STRATEGIES=(S7-0 S7-1 S7-2)
        TARGETS=(0.2 0.3 0.4 0.5)
        STRATEGY="${STRATEGIES[$((TASK_ID / 4))]}"
        TARGET="${TARGETS[$((TASK_ID % 4))]}"
        OUTPUT_DIR="$OUTPUT_ROOT/${STRATEGY}-target-${TARGET}-run-001"
        RUN_ARGUMENTS=(
            train
            --prepared "$PREPARED_RESULT"
            --strategy "$STRATEGY"
            --target "$TARGET"
            --work-dir "$WORK_DIR"
            --output-dir "$OUTPUT_DIR"
        )
        ;;
    *)
        echo "Unsupported SwiGLU-7 mode: $MODE" >&2
        exit 2
        ;;
esac

echo "Job: ${SLURM_JOB_ID} on ${SLURM_NODELIST}"
echo "Checkout directory: $PWD"
echo "Mode: $MODE"
echo "Python: $PYTHON_BIN"
echo "Temporary workflow directory: $WORK_DIR"
echo "Durable output directory: $OUTPUT_DIR"
if [[ "$MODE" == "train" ]]; then
    echo "Array mapping: task=$TASK_ID strategy=$STRATEGY target=$TARGET"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

"$PYTHON_BIN" -u -m workflows.runs.model.swiglu.swiglu_7 "${RUN_ARGUMENTS[@]}"
