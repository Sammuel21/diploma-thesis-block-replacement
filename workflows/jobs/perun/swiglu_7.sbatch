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

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:?Slurm did not define SLURM_SUBMIT_DIR}"
SCRATCH_USER="${SLURM_JOB_USER:-${USER:?Neither Slurm nor the shell defined a username}}"
if [[ ! "$SCRATCH_USER" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Refusing unsafe scratch username: $SCRATCH_USER" >&2
    exit 2
fi

JOB_SCRATCH_DIR="/mnt/scratch/$SCRATCH_USER/mlp-replacement/job-$SLURM_JOB_ID"
export SCRATCH_DIR="$JOB_SCRATCH_DIR"
export TMPDIR="$JOB_SCRATCH_DIR/tmp"
export RESULTS_DIR="$JOB_SCRATCH_DIR/results"
STAGE_OUT_DIR="$SUBMIT_DIR/results_job_$SLURM_JOB_ID"

case "$JOB_SCRATCH_DIR" in
    "/mnt/scratch/$SCRATCH_USER/mlp-replacement/job-$SLURM_JOB_ID") ;;
    *)
        echo "Refusing unsafe job scratch path: $JOB_SCRATCH_DIR" >&2
        exit 2
        ;;
esac

mkdir -p "$TMPDIR" "$RESULTS_DIR"
cd "$SUBMIT_DIR"

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    echo "Usage: sbatch $0 prepare" >&2
    echo "   or: sbatch --array=0-11[%N] $0 train PREPARED_RESULT" >&2
    exit 2
fi
shift

PYTHON_BIN="${MLP_REPLACEMENT_PYTHON:-python3}"
export PYTHONPATH="$SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
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
stage_out() {
    local job_status=$?
    local sync_status
    trap - EXIT
    set +e
    mkdir -p "$STAGE_OUT_DIR"
    rsync -a "$RESULTS_DIR/" "$STAGE_OUT_DIR/"
    sync_status=$?
    if (( sync_status == 0 )); then
        rm -rf -- "$JOB_SCRATCH_DIR"
    else
        echo "Stage-out failed; scratch retained at $JOB_SCRATCH_DIR" >&2
    fi
    if (( job_status == 0 && sync_status != 0 )); then
        job_status=$sync_status
    fi
    exit "$job_status"
}
trap stage_out EXIT
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
echo "Checkout directory: $SUBMIT_DIR"
echo "Job scratch directory: $JOB_SCRATCH_DIR"
echo "Stage-out directory: $STAGE_OUT_DIR"
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
