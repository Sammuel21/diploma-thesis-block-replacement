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
#SBATCH --signal=B:TERM@1800
#SBATCH --export=ALL

set -euo pipefail

for REQUIRED_COMMAND in rsync realpath cmp nvidia-smi; do
    if ! command -v "$REQUIRED_COMMAND" >/dev/null 2>&1; then
        echo "Required command is unavailable: $REQUIRED_COMMAND" >&2
        exit 2
    fi
done

PYTHON_BIN="${MLP_REPLACEMENT_PYTHON:?Export MLP_REPLACEMENT_PYTHON before submission}"
if [[ "$PYTHON_BIN" != /* ]]; then
    echo "MLP_REPLACEMENT_PYTHON must be an absolute path: $PYTHON_BIN" >&2
    exit 2
fi
PYTHON_BIN="$(realpath "$PYTHON_BIN")"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter is not executable: $PYTHON_BIN" >&2
    exit 2
fi
PYTHON_ENV_PREFIX="$(dirname "$(dirname "$PYTHON_BIN")")"
case ":${LD_LIBRARY_PATH:-}:" in
    *":$PYTHON_ENV_PREFIX/lib:"*) ;;
    *) export LD_LIBRARY_PATH="$PYTHON_ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac

HF_HOME="${HF_HOME:?Export a persistent PROJECT HF_HOME before submission}"
if [[ "$HF_HOME" != /* || ! -d "$HF_HOME" || ! -w "$HF_HOME" ]]; then
    echo "HF_HOME must be an existing writable absolute directory: $HF_HOME" >&2
    exit 2
fi
HF_HOME="$(realpath "$HF_HOME")"
export HF_HOME

usage() {
    echo "Usage: sbatch $0 prepare [--resume]" >&2
    echo "   or: sbatch --array=0-11[%N] $0 train [--resume] PREPARED_RESULT" >&2
}

if (( $# < 1 )); then
    usage
    exit 2
fi

MODE="$1"
shift
RESUME=0
if [[ "${1:-}" == "--resume" ]]; then
    RESUME=1
    shift
fi

TASK_SUFFIX="${SLURM_ARRAY_TASK_ID:-single}"
case "$MODE" in
    prepare)
        if [[ -n "${SLURM_ARRAY_TASK_ID:-}" || $# -ne 0 ]]; then
            echo "Preparation is one non-array job with no positional arguments." >&2
            exit 2
        fi
        OUTPUT_NAME="prepare-001"
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
        PREPARED_SOURCE="$1"
        if [[ ! -f "$PREPARED_SOURCE" ]]; then
            echo "Prepared result does not exist: $PREPARED_SOURCE" >&2
            exit 2
        fi
        PREPARED_SOURCE="$(realpath "$PREPARED_SOURCE")"
        STRATEGIES=(S7-0 S7-1 S7-2)
        TARGETS=(0.2 0.3 0.4 0.5)
        STRATEGY="${STRATEGIES[$((TASK_ID / 4))]}"
        TARGET="${TARGETS[$((TASK_ID % 4))]}"
        OUTPUT_NAME="${STRATEGY}-target-${TARGET}-run-001"
        ;;
    *)
        echo "Unsupported SwiGLU-7 mode: $MODE" >&2
        usage
        exit 2
        ;;
esac

SUBMIT_DIR="$(realpath "${SLURM_SUBMIT_DIR:?Slurm did not define SLURM_SUBMIT_DIR}")"
SCRATCH_ROOT="/mnt/scratch/${USER:?Slurm did not export USER}"
PROJECT_ROOT="$(dirname "$SUBMIT_DIR")"
PROJECT_RESULTS_ROOT="$PROJECT_ROOT/perun-results/swiglu-7"
PERSISTENT_OUTPUT_DIR="$PROJECT_RESULTS_ROOT/$OUTPUT_NAME"
LOCK_DIR="${PERSISTENT_OUTPUT_DIR}.perun-lock"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    JOB_KEY="${SLURM_ARRAY_JOB_ID:?}_${SLURM_ARRAY_TASK_ID}"
else
    JOB_KEY="${SLURM_JOB_ID:?}"
fi
SCRATCH_DIR="$SCRATCH_ROOT/job_${JOB_KEY}"

test -f "$SUBMIT_DIR/workflows/jobs/perun/swiglu_7.sbatch"
test -f "$SUBMIT_DIR/.rsyncignore"
test -d "$SCRATCH_ROOT"
test -w "$SCRATCH_ROOT"
if [[ "$(dirname "$PROJECT_ROOT")" != "/mnt/project" ]]; then
    echo "Repository must be a direct child of a PERUN PROJECT directory." >&2
    exit 2
fi
test -w "$PROJECT_ROOT"
case "$HF_HOME" in
    "$PROJECT_ROOT"/*) ;;
    *)
        echo "SwiGLU-7 requires HF_HOME inside the active PROJECT directory." >&2
        exit 2
        ;;
esac
if [[ "$MODE" == "train" ]] \
    && [[ "$PREPARED_SOURCE" != "$PROJECT_RESULTS_ROOT/prepare-001/result.json" ]]; then
    echo "Training requires the canonical SwiGLU-7 preparation result." >&2
    echo "Expected: $PROJECT_RESULTS_ROOT/prepare-001/result.json" >&2
    exit 2
fi

case "$SCRATCH_DIR" in
    "$SCRATCH_ROOT"/job_*) ;;
    *)
        echo "Refusing unsafe scratch path: $SCRATCH_DIR" >&2
        exit 2
        ;;
esac
case "$PERSISTENT_OUTPUT_DIR" in
    "$PROJECT_RESULTS_ROOT"/*) ;;
    *)
        echo "Refusing unsafe persistent output path: $PERSISTENT_OUTPUT_DIR" >&2
        exit 2
        ;;
esac

if [[ -e "$SCRATCH_DIR" ]]; then
    echo "Refusing to reuse existing scratch: $SCRATCH_DIR" >&2
    exit 2
fi
if (( RESUME == 1 )); then
    if [[ ! -f "$PERSISTENT_OUTPUT_DIR/result.json" ]]; then
        echo "Resume requires $PERSISTENT_OUTPUT_DIR/result.json" >&2
        exit 2
    fi
elif [[ -e "$PERSISTENT_OUTPUT_DIR" ]]; then
    echo "Output already exists; pass --resume only for the same run: $PERSISTENT_OUTPUT_DIR" >&2
    exit 2
fi

nvidia-smi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Slurm did not expose a CUDA device to the job." >&2
    exit 2
fi
"$PYTHON_BIN" -c \
    'import torch, transformers, datasets, lm_eval, safetensors, psutil; assert torch.cuda.is_available(), "PyTorch cannot access CUDA"; print("environment preflight: OK", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)'

mkdir -p "$PROJECT_RESULTS_ROOT"
test -w "$PROJECT_RESULTS_ROOT"
if ! mkdir "$LOCK_DIR"; then
    echo "Output is locked by another or interrupted job: $LOCK_DIR" >&2
    exit 2
fi
{
    printf 'job_id=%s\n' "$SLURM_JOB_ID"
    printf 'array_task_id=%s\n' "$TASK_SUFFIX"
    printf 'scratch_dir=%s\n' "$SCRATCH_DIR"
    printf 'output_dir=%s\n' "$PERSISTENT_OUTPUT_DIR"
} > "$LOCK_DIR/owner.txt"

if ! mkdir "$SCRATCH_DIR"; then
    rm -f "$LOCK_DIR/owner.txt"
    rmdir "$LOCK_DIR"
    exit 1
fi
printf 'created_by_job=%s\n' "$SLURM_JOB_ID" > "$SCRATCH_DIR/.swiglu_7_owner"

export SCRATCH_DIR
export TMPDIR="$SCRATCH_DIR/tmp"
export RESULTS_DIR="$SCRATCH_DIR/results"
OUTPUT_DIR="$RESULTS_DIR/swiglu-7/$OUTPUT_NAME"
WORK_DIR="$TMPDIR/mlp-replacement/swiglu-7-${OUTPUT_NAME}-${TASK_SUFFIX}"
PYTHON_PID=""
OUTPUT_READY_FOR_STAGEOUT=0

finalize_job() {
    exit_code=$?
    trap - EXIT TERM INT
    set +e
    stageout_code=0
    scratch_removed=0

    if (( OUTPUT_READY_FOR_STAGEOUT == 1 )) && [[ -f "$OUTPUT_DIR/result.json" ]]; then
        mkdir -p "$PERSISTENT_OUTPUT_DIR"
        rsync -a --delete --exclude='*.tmp' \
            "$OUTPUT_DIR/" "$PERSISTENT_OUTPUT_DIR/"
        stageout_code=$?
        if (( stageout_code == 0 )) \
            && ! cmp -s "$OUTPUT_DIR/result.json" "$PERSISTENT_OUTPUT_DIR/result.json"; then
            echo "Stage-out verification failed for result.json" >&2
            stageout_code=1
        fi
    elif (( OUTPUT_READY_FOR_STAGEOUT == 0 )); then
        echo "Output stage-in did not complete; refusing to overwrite the persistent copy." >&2
        stageout_code=1
    else
        echo "No result.json was available to stage out." >&2
        stageout_code=1
    fi

    if (( stageout_code == 0 )); then
        if [[ -f "$SCRATCH_DIR/.swiglu_7_owner" ]] \
            && [[ "$(<"$SCRATCH_DIR/.swiglu_7_owner")" == "created_by_job=$SLURM_JOB_ID" ]]; then
            if rm -rf -- "$SCRATCH_DIR" && [[ ! -e "$SCRATCH_DIR" ]]; then
                scratch_removed=1
            else
                echo "Could not remove verified scratch directory: $SCRATCH_DIR" >&2
                stageout_code=1
            fi
        else
            echo "Scratch marker verification failed; preserving $SCRATCH_DIR" >&2
            stageout_code=1
        fi
    fi

    if (( stageout_code == 0 )); then
        rm -f "$LOCK_DIR/owner.txt"
        if ! rmdir "$LOCK_DIR"; then
            echo "Could not remove completed-job lock: $LOCK_DIR" >&2
            stageout_code=1
        fi
        echo "Persistent output: $PERSISTENT_OUTPUT_DIR"
    else
        if (( scratch_removed == 0 )); then
            echo "Stage-out incomplete; scratch preserved at $SCRATCH_DIR" >&2
        else
            echo "Persistent output was verified and scratch was cleaned." >&2
        fi
        echo "Lock preserved at $LOCK_DIR" >&2
    fi

    if (( exit_code == 0 && stageout_code != 0 )); then
        exit "$stageout_code"
    fi
    exit "$exit_code"
}

terminate_job() {
    signal_exit="$1"
    echo "Termination requested; stopping Python before stage-out." >&2
    if [[ -n "$PYTHON_PID" ]] && kill -0 "$PYTHON_PID" 2>/dev/null; then
        kill -TERM "$PYTHON_PID" 2>/dev/null || true
        wait "$PYTHON_PID" 2>/dev/null || true
    fi
    exit "$signal_exit"
}

trap finalize_job EXIT
trap 'terminate_job 143' TERM
trap 'terminate_job 130' INT

rsync -a \
    --exclude='.git/' \
    --exclude='.env' \
    --exclude='local/' \
    --exclude='*.out' \
    --exclude='*.err' \
    --exclude='results_job_*/' \
    --exclude-from="$SUBMIT_DIR/.rsyncignore" \
    "$SUBMIT_DIR/" "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"
mkdir -p "$TMPDIR" "$RESULTS_DIR"

if (( RESUME == 1 )); then
    mkdir -p "$OUTPUT_DIR"
    rsync -a --delete "$PERSISTENT_OUTPUT_DIR/" "$OUTPUT_DIR/"
fi
OUTPUT_READY_FOR_STAGEOUT=1
export PYTHONPATH="$SCRATCH_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

RESUME_ARGUMENTS=()
if (( RESUME == 1 )); then
    RESUME_ARGUMENTS=(--resume)
fi

case "$MODE" in
    prepare)
        RUN_ARGUMENTS=(
            prepare
            --work-dir "$WORK_DIR"
            --output-dir "$OUTPUT_DIR"
            "${RESUME_ARGUMENTS[@]}"
        )
        ;;
    train)
        PREPARED_INPUT_DIR="$SCRATCH_DIR/inputs/prepared"
        mkdir -p "$PREPARED_INPUT_DIR"
        rsync -a "$(dirname "$PREPARED_SOURCE")/" "$PREPARED_INPUT_DIR/"
        PREPARED_RESULT="$PREPARED_INPUT_DIR/$(basename "$PREPARED_SOURCE")"
        test -f "$PREPARED_RESULT"
        RUN_ARGUMENTS=(
            train
            --prepared "$PREPARED_RESULT"
            --strategy "$STRATEGY"
            --target "$TARGET"
            --work-dir "$WORK_DIR"
            --output-dir "$OUTPUT_DIR"
            "${RESUME_ARGUMENTS[@]}"
        )
        ;;
esac

echo "Job: ${SLURM_JOB_ID} on ${SLURM_NODELIST}"
echo "Submit directory: $SUBMIT_DIR"
echo "Scratch directory: $SCRATCH_DIR"
echo "Mode: $MODE"
echo "Resume: $RESUME"
echo "Python: $PYTHON_BIN"
echo "Temporary workflow directory: $WORK_DIR"
echo "Persistent output directory: $PERSISTENT_OUTPUT_DIR"
if [[ "$MODE" == "train" ]]; then
    echo "Array mapping: task=$TASK_ID strategy=$STRATEGY target=$TARGET"
    echo "Staged prepared result: $PREPARED_RESULT"
fi

"$PYTHON_BIN" -u -m workflows.runs.model.swiglu.swiglu_7 "${RUN_ARGUMENTS[@]}" &
PYTHON_PID=$!
wait "$PYTHON_PID"
PYTHON_PID=""
