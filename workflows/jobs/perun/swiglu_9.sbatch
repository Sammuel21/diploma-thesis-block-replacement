#!/bin/bash
#SBATCH --job-name=swiglu-9
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Supply --account, --qos, and the stage's planned --time and --mem to sbatch; see README.md.
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --signal=B:TERM@1800
#SBATCH --export=ALL

set -euo pipefail

usage() {
    echo "Usage: sbatch --array=0-2 $0 block-check [--resume]" >&2
    echo "   or: sbatch --array=0-4 $0 train [--resume]" >&2
    echo "   or: sbatch $0 reference-kl [--resume]" >&2
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
(( $# == 0 )) || { usage; exit 2; }

PYTHON_BIN="${MLP_REPLACEMENT_PYTHON:?Export MLP_REPLACEMENT_PYTHON before submission}"
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || { echo "MLP_REPLACEMENT_PYTHON must be an absolute executable" >&2; exit 2; }
PYTHON_BIN="$(realpath "$PYTHON_BIN")"
PYTHON_ENV_PREFIX="$(dirname "$(dirname "$PYTHON_BIN")")"
export LD_LIBRARY_PATH="$PYTHON_ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SUBMIT_DIR="$(realpath "${SLURM_SUBMIT_DIR:?Submit from the repository root}")"
test -f "$SUBMIT_DIR/workflows/jobs/perun/swiglu_9.sbatch"
PROJECT_ROOT="$(dirname "$SUBMIT_DIR")"
[[ "$(dirname "$PROJECT_ROOT")" == "/mnt/project" ]] || { echo "Repository must be a direct child of a PERUN PROJECT directory." >&2; exit 2; }
test -w "$PROJECT_ROOT"
HF_HOME="${HF_HOME:?Export a persistent PROJECT HF_HOME before submission}"
[[ "$HF_HOME" == /* && -d "$HF_HOME" && -w "$HF_HOME" ]] || exit 2
HF_HOME="$(realpath "$HF_HOME")"
[[ "$HF_HOME" == "$PROJECT_ROOT"/* ]] || { echo "SwiGLU-9 requires HF_HOME inside the active PROJECT directory." >&2; exit 2; }
export HF_HOME

PROJECT_RESULTS_ROOT="$PROJECT_ROOT/perun-results/swiglu-9"
SWIGLU7_ROOT="$PROJECT_ROOT/perun-results/swiglu-7"
SWIGLU8_ROOT="$PROJECT_ROOT/perun-results/swiglu-8"
SWIGLU8_PREPARED="$SWIGLU8_ROOT/prepare-001"
test -f "$SWIGLU7_ROOT/prepare-001/result.json"
test -f "$SWIGLU8_PREPARED/result.json"
# Part B array order; the runner's RUNS constant repeats it.
RUN_ARMS=(fitted fitted slice slice slice)
RUN_TARGETS=(0.5 0.8 0.2 0.5 0.8)
REFERENCE_TARGETS=(0.2 0.5)
case "$MODE" in
    block-check)
        # One array task per Part A layer (3, 11, 19); merge-block-check joins them afterwards.
        [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { usage; exit 2; }
        TASK_ID="$SLURM_ARRAY_TASK_ID"
        (( TASK_ID >= 0 && TASK_ID <= 2 )) || { echo "Block-check array index must be 0-2." >&2; exit 2; }
        test -f "$SWIGLU8_ROOT/block-check-001/result.json"
        OUTPUT_NAME="$(printf 'block-check-001-task-%02d' "$TASK_ID")"
        ;;
    train)
        [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { usage; exit 2; }
        TASK_ID="$SLURM_ARRAY_TASK_ID"
        (( TASK_ID >= 0 && TASK_ID <= 4 )) || { echo "SwiGLU-9 training array index must be 0-4." >&2; exit 2; }
        ARM="${RUN_ARMS[$TASK_ID]}"
        TARGET="${RUN_TARGETS[$TASK_ID]}"
        test -f "$SWIGLU7_ROOT/prepare-001/prepared/recovery-tokens.int32"
        test -f "$SWIGLU8_PREPARED/prepared/starts/${ARM}-target-${TARGET}.pt"
        OUTPUT_NAME="${ARM}-target-${TARGET}-1b-run-001"
        ;;
    reference-kl)
        [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || { usage; exit 2; }
        for REFERENCE_TARGET in "${REFERENCE_TARGETS[@]}"; do
            test -f "$SWIGLU7_ROOT/S7-0-target-${REFERENCE_TARGET}-run-001/model/bundle.json"
        done
        OUTPUT_NAME="reference-kl-001"
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
printf '%s\n' "$SLURM_JOB_ID" > "$SCRATCH_DIR/.swiglu_9_owner"
export SCRATCH_DIR
export TMPDIR="$SCRATCH_DIR/tmp"
OUTPUT_DIR="$SCRATCH_DIR/results/$OUTPUT_NAME"
WORK_DIR="$TMPDIR/swiglu-9-$OUTPUT_NAME"
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
       && [[ -f "$SCRATCH_DIR/.swiglu_9_owner" ]] \
       && [[ "$(<"$SCRATCH_DIR/.swiglu_9_owner")" == "$SLURM_JOB_ID" ]]; then
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
COMMON_ARGUMENTS=(--work-dir "$WORK_DIR" --output-dir "$OUTPUT_DIR" --swiglu8-root "$SWIGLU8_ROOT")
PREPARED_INPUT="$SCRATCH_DIR/inputs/swiglu-8-prepare"
case "$MODE" in
    block-check)
        RUN_ARGUMENTS=(block-check "${COMMON_ARGUMENTS[@]}" --swiglu7-root "$SWIGLU7_ROOT" --task-index "$TASK_ID")
        ;;
    train)
        # Stage SwiGLU-8's shared prepared inputs without its starts and 100M stream, this
        # task's start, and SwiGLU-7's 1B recovery stream.
        mkdir -p "$PREPARED_INPUT/prepared/starts" "$SCRATCH_DIR/inputs/swiglu-7-prepare/prepared"
        rsync -a --exclude='prepared/starts/' --exclude='prepared/recovery-tokens.int32' \
            "$SWIGLU8_PREPARED/" "$PREPARED_INPUT/"
        rsync -a "$SWIGLU8_PREPARED/prepared/starts/${ARM}-target-${TARGET}.pt" "$PREPARED_INPUT/prepared/starts/"
        rsync -a "$SWIGLU7_ROOT/prepare-001/prepared/recovery-tokens.int32" \
            "$SCRATCH_DIR/inputs/swiglu-7-prepare/prepared/"
        RUN_ARGUMENTS=(train "${COMMON_ARGUMENTS[@]}" --swiglu7-root "$SWIGLU7_ROOT"
                       --prepared "$PREPARED_INPUT/result.json"
                       --recovery-stream-root "$SCRATCH_DIR/inputs/swiglu-7-prepare"
                       --arm "$ARM" --target "$TARGET")
        ;;
    reference-kl)
        # Stage SwiGLU-8's preparation record and KL stream, and the two SwiGLU-7 runs.
        mkdir -p "$PREPARED_INPUT/prepared"
        rsync -a "$SWIGLU8_PREPARED/result.json" "$PREPARED_INPUT/"
        rsync -a "$SWIGLU8_PREPARED/prepared/kl-tokens.int32" "$PREPARED_INPUT/prepared/"
        for REFERENCE_TARGET in "${REFERENCE_TARGETS[@]}"; do
            REFERENCE_RUN="S7-0-target-${REFERENCE_TARGET}-run-001"
            mkdir -p "$SCRATCH_DIR/inputs/swiglu-7/$REFERENCE_RUN"
            rsync -a "$SWIGLU7_ROOT/$REFERENCE_RUN/result.json" "$SWIGLU7_ROOT/$REFERENCE_RUN/model" \
                "$SCRATCH_DIR/inputs/swiglu-7/$REFERENCE_RUN/"
        done
        RUN_ARGUMENTS=(reference-kl "${COMMON_ARGUMENTS[@]}" --swiglu7-root "$SCRATCH_DIR/inputs/swiglu-7"
                       --prepared "$PREPARED_INPUT/result.json")
        ;;
esac

cd "$SCRATCH_DIR"
export PYTHONPATH="$SCRATCH_DIR/src:$SCRATCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
echo "Job: ${SLURM_JOB_ID} on ${SLURM_NODELIST:-unknown}; mode=$MODE output=$PERSISTENT_OUTPUT_DIR"
[[ "$MODE" != "train" ]] || echo "Array mapping: task=$TASK_ID arm=$ARM target=$TARGET"
[[ "$MODE" != "block-check" ]] || echo "Block-check task: $TASK_ID (Part A layer in configuration order)"
"$PYTHON_BIN" -u -m workflows.runs.model.swiglu.swiglu_9 "${RUN_ARGUMENTS[@]}" "${RESUME_ARGUMENTS[@]}" &
PYTHON_PID=$!
wait "$PYTHON_PID"
PYTHON_PID=""
