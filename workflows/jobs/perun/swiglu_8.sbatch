#!/bin/bash
#SBATCH --job-name=swiglu-8
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Supply --account, --qos, and the stage's planned --time to sbatch; see README.md.
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@1800
#SBATCH --export=ALL

set -euo pipefail

usage() {
    echo "Usage: sbatch $0 block-check [--resume]" >&2
    echo "   or: sbatch $0 prepare [--resume] BLOCK_CHECK_RESULT" >&2
    echo "   or: sbatch --array=0-15[%N] $0 train [--resume] PREPARED_RESULT" >&2
}

for REQUIRED_COMMAND in rsync realpath cmp nvidia-smi; do
    command -v "$REQUIRED_COMMAND" >/dev/null 2>&1 || {
        echo "Required command is unavailable: $REQUIRED_COMMAND" >&2
        exit 2
    }
done
(( $# >= 1 )) || { usage; exit 2; }
MODE="$1"
shift
RESUME=0
if [[ "${1:-}" == "--resume" ]]; then
    RESUME=1
    shift
fi

PYTHON_BIN="${MLP_REPLACEMENT_PYTHON:?Export MLP_REPLACEMENT_PYTHON before submission}"
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || { echo "MLP_REPLACEMENT_PYTHON must be an absolute executable" >&2; exit 2; }
PYTHON_BIN="$(realpath "$PYTHON_BIN")"
PYTHON_ENV_PREFIX="$(dirname "$(dirname "$PYTHON_BIN")")"
export LD_LIBRARY_PATH="$PYTHON_ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SUBMIT_DIR="$(realpath "${SLURM_SUBMIT_DIR:?Submit from the repository root}")"
test -f "$SUBMIT_DIR/workflows/jobs/perun/swiglu_8.sbatch"
PROJECT_ROOT="$(dirname "$SUBMIT_DIR")"
[[ "$(dirname "$PROJECT_ROOT")" == "/mnt/project" ]] || { echo "Repository must be a direct child of a PERUN PROJECT directory." >&2; exit 2; }
test -w "$PROJECT_ROOT"
HF_HOME="${HF_HOME:?Export a persistent PROJECT HF_HOME before submission}"
[[ "$HF_HOME" == /* && -d "$HF_HOME" && -w "$HF_HOME" ]] || exit 2
HF_HOME="$(realpath "$HF_HOME")"
[[ "$HF_HOME" == "$PROJECT_ROOT"/* ]] || { echo "SwiGLU-8 requires HF_HOME inside the active PROJECT directory." >&2; exit 2; }
export HF_HOME

PROJECT_RESULTS_ROOT="$PROJECT_ROOT/perun-results/swiglu-8"
SWIGLU7_ROOT="$PROJECT_ROOT/perun-results/swiglu-7"
test -f "$SWIGLU7_ROOT/prepare-001/result.json"
ARMS=(fitted slice)
TARGETS=(0.2 0.3 0.4 0.5 0.6 0.7 0.75 0.8)
case "$MODE" in
    block-check)
        [[ -z "${SLURM_ARRAY_TASK_ID:-}" && $# -eq 0 ]] || { usage; exit 2; }
        OUTPUT_NAME="block-check-001"
        ;;
    prepare)
        [[ -z "${SLURM_ARRAY_TASK_ID:-}" && $# -eq 1 ]] || { usage; exit 2; }
        INPUT_SOURCE="$(realpath "$1")"
        [[ "$INPUT_SOURCE" == "$PROJECT_RESULTS_ROOT/block-check-001/result.json" ]] || {
            echo "Preparation requires $PROJECT_RESULTS_ROOT/block-check-001/result.json" >&2
            exit 2
        }
        OUTPUT_NAME="prepare-001"
        ;;
    train)
        [[ -n "${SLURM_ARRAY_TASK_ID:-}" && $# -eq 1 ]] || { usage; exit 2; }
        TASK_ID="$SLURM_ARRAY_TASK_ID"
        (( TASK_ID >= 0 && TASK_ID <= 15 )) || { echo "SwiGLU-8 array index must be 0-15." >&2; exit 2; }
        INPUT_SOURCE="$(realpath "$1")"
        [[ "$INPUT_SOURCE" == "$PROJECT_RESULTS_ROOT/prepare-001/result.json" ]] || {
            echo "Training requires $PROJECT_RESULTS_ROOT/prepare-001/result.json" >&2
            exit 2
        }
        ARM="${ARMS[$((TASK_ID / 8))]}"
        TARGET="${TARGETS[$((TASK_ID % 8))]}"
        OUTPUT_NAME="${ARM}-target-${TARGET}-run-001"
        ;;
    *)
        usage
        exit 2
        ;;
esac

PERSISTENT_OUTPUT_DIR="$PROJECT_RESULTS_ROOT/$OUTPUT_NAME"
LOCK_DIR="${PERSISTENT_OUTPUT_DIR}.perun-lock"
SCRATCH_ROOT="/mnt/scratch/${USER:?Slurm did not export USER}"
JOB_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?}}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
SCRATCH_DIR="$SCRATCH_ROOT/job_${JOB_KEY}"
test -d "$SCRATCH_ROOT" && test -w "$SCRATCH_ROOT"
[[ "$SCRATCH_DIR" == "$SCRATCH_ROOT"/job_* && ! -e "$SCRATCH_DIR" ]] || { echo "Refusing unsafe or existing scratch: $SCRATCH_DIR" >&2; exit 2; }
if (( RESUME )); then
    test -f "$PERSISTENT_OUTPUT_DIR/result.json" || { echo "Resume requires $PERSISTENT_OUTPUT_DIR/result.json" >&2; exit 2; }
elif [[ -e "$PERSISTENT_OUTPUT_DIR" ]]; then
    echo "Output exists; pass --resume only for the same run: $PERSISTENT_OUTPUT_DIR" >&2
    exit 2
fi

nvidia-smi
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "Slurm did not expose a CUDA device to the job." >&2; exit 2; }
"$PYTHON_BIN" -c \
    'from importlib.metadata import version; from packaging.version import Version; import torch, transformers, datasets, lm_eval, safetensors, psutil; from lm_eval.models.huggingface import HFLM; accelerate_version = version("accelerate"); assert Version(accelerate_version) >= Version("0.26.0"), f"accelerate>=0.26.0 is required, found {accelerate_version}"; assert torch.cuda.is_available(), "PyTorch cannot access CUDA"; print("environment preflight: OK", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), "accelerate", accelerate_version, flush=True)'

mkdir -p "$PROJECT_RESULTS_ROOT"
mkdir "$LOCK_DIR" || { echo "Output is locked by another or interrupted job: $LOCK_DIR" >&2; exit 2; }
printf 'job_id=%s\narray_task_id=%s\nscratch_dir=%s\n' \
    "$SLURM_JOB_ID" "${SLURM_ARRAY_TASK_ID:-single}" "$SCRATCH_DIR" > "$LOCK_DIR/owner.txt"
mkdir "$SCRATCH_DIR"
printf '%s\n' "$SLURM_JOB_ID" > "$SCRATCH_DIR/.swiglu_8_owner"
export SCRATCH_DIR
export TMPDIR="$SCRATCH_DIR/tmp"
OUTPUT_DIR="$SCRATCH_DIR/results/$OUTPUT_NAME"
WORK_DIR="$TMPDIR/swiglu-8-$OUTPUT_NAME"
PYTHON_PID=""
OUTPUT_READY=0

finalize_job() {
    exit_code=$?
    trap - EXIT TERM INT
    set +e
    stageout_code=1
    if (( OUTPUT_READY )) && [[ -f "$OUTPUT_DIR/result.json" ]]; then
        mkdir -p "$PERSISTENT_OUTPUT_DIR"
        rsync -a --delete --exclude='*.tmp' "$OUTPUT_DIR/" "$PERSISTENT_OUTPUT_DIR/"
        stageout_code=$?
        if (( stageout_code == 0 )); then
            cmp -s "$OUTPUT_DIR/result.json" "$PERSISTENT_OUTPUT_DIR/result.json" || stageout_code=1
            # Verify every durable byte before scratch is removed.
            comparison="$(rsync -rcn --delete --exclude='*.tmp' --out-format='%i %n' "$OUTPUT_DIR/" "$PERSISTENT_OUTPUT_DIR/")"
            [[ $? -eq 0 && -z "$comparison" ]] || stageout_code=1
        fi
    fi
    if (( stageout_code == 0 )) && [[ "$SCRATCH_DIR" == "$SCRATCH_ROOT"/job_* ]] \
       && [[ -f "$SCRATCH_DIR/.swiglu_8_owner" ]] \
       && [[ "$(<"$SCRATCH_DIR/.swiglu_8_owner")" == "$SLURM_JOB_ID" ]]; then
        rm -rf -- "$SCRATCH_DIR" || stageout_code=1
        if (( stageout_code == 0 )); then
            rm -f "$LOCK_DIR/owner.txt"
            rmdir "$LOCK_DIR" || stageout_code=1
            echo "Verified persistent output: $PERSISTENT_OUTPUT_DIR"
        fi
    else
        stageout_code=1
        echo "Stage-out incomplete; scratch and lock preserved: $SCRATCH_DIR $LOCK_DIR" >&2
    fi
    (( exit_code != 0 || stageout_code == 0 )) || exit_code=1
    exit "$exit_code"
}
terminate_job() {
    echo "Termination requested; stopping Python before stage-out." >&2
    if [[ -n "$PYTHON_PID" ]] && kill -0 "$PYTHON_PID" 2>/dev/null; then
        kill -TERM "$PYTHON_PID" 2>/dev/null || true
        wait "$PYTHON_PID" 2>/dev/null || true
    fi
    exit "$1"
}
trap finalize_job EXIT
trap 'terminate_job 143' TERM
trap 'terminate_job 130' INT

# Stage only executable code and configurations.
for CODE_PATH in src workflows/runs workflows/configs workflows/__init__.py; do
    mkdir -p "$SCRATCH_DIR/$(dirname "$CODE_PATH")"
    rsync -a --exclude='__pycache__/' "$SUBMIT_DIR/$CODE_PATH" "$SCRATCH_DIR/$(dirname "$CODE_PATH")/"
done
mkdir -p "$TMPDIR" "$SCRATCH_DIR/results" "$SCRATCH_DIR/inputs"
if (( RESUME )); then
    mkdir -p "$OUTPUT_DIR"
    rsync -a --delete "$PERSISTENT_OUTPUT_DIR/" "$OUTPUT_DIR/"
fi
OUTPUT_READY=1

RESUME_ARGUMENTS=()
if (( RESUME )); then
    RESUME_ARGUMENTS=(--resume)
fi
COMMON_ARGUMENTS=(--work-dir "$WORK_DIR" --output-dir "$OUTPUT_DIR" --swiglu7-root "$SWIGLU7_ROOT")
case "$MODE" in
    block-check)
        RUN_ARGUMENTS=(block-check "${COMMON_ARGUMENTS[@]}")
        ;;
    prepare)
        rsync -a "$INPUT_SOURCE" "$SCRATCH_DIR/inputs/block-check.json"
        RUN_ARGUMENTS=(prepare "${COMMON_ARGUMENTS[@]}" --block-check "$SCRATCH_DIR/inputs/block-check.json")
        ;;
    train)
        # Stage the shared prepared inputs and only this task's start file.
        PREPARED_INPUT="$SCRATCH_DIR/inputs/prepared"
        mkdir -p "$PREPARED_INPUT/prepared/starts"
        rsync -a --exclude='prepared/starts/' "$(dirname "$INPUT_SOURCE")/" "$PREPARED_INPUT/"
        rsync -a "$(dirname "$INPUT_SOURCE")/prepared/starts/${ARM}-target-${TARGET}.pt" "$PREPARED_INPUT/prepared/starts/"
        RUN_ARGUMENTS=(train "${COMMON_ARGUMENTS[@]}" --prepared "$PREPARED_INPUT/result.json"
                       --arm "$ARM" --target "$TARGET")
        ;;
esac

cd "$SCRATCH_DIR"
export PYTHONPATH="$SCRATCH_DIR/src:$SCRATCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
echo "Job: ${SLURM_JOB_ID} on ${SLURM_NODELIST:-unknown}; mode=$MODE output=$PERSISTENT_OUTPUT_DIR"
[[ "$MODE" != "train" ]] || echo "Array mapping: task=$TASK_ID arm=$ARM target=$TARGET"
"$PYTHON_BIN" -u -m workflows.runs.model.swiglu.swiglu_8 "${RUN_ARGUMENTS[@]}" "${RESUME_ARGUMENTS[@]}" &
PYTHON_PID=$!
wait "$PYTHON_PID"
PYTHON_PID=""
