#!/bin/bash
#SBATCH --job-name=recovery-2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Supply --account, --qos, and the planned --time and --mem to sbatch; see the README.
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --export=ALL

set -euo pipefail

usage() {
    echo "Usage: RECOVERY2_SELECTION_STREAM=/mnt/project/.../long-context-selection.int32 sbatch --array=0-2 $0 train" >&2
}

(( $# == 1 )) && [[ "$1" == "train" ]] || { usage; exit 2; }
for REQUIRED_COMMAND in rsync realpath cmp nvidia-smi; do
    command -v "$REQUIRED_COMMAND" >/dev/null 2>&1 || {
        echo "Required command is unavailable: $REQUIRED_COMMAND" >&2
        exit 2
    }
done

PYTHON_BIN="${MLP_REPLACEMENT_PYTHON:?Export MLP_REPLACEMENT_PYTHON before submission}"
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || { echo "MLP_REPLACEMENT_PYTHON must be an absolute executable" >&2; exit 2; }
PYTHON_BIN="$(realpath "$PYTHON_BIN")"
PYTHON_ENV_PREFIX="$(dirname "$(dirname "$PYTHON_BIN")")"
export LD_LIBRARY_PATH="$PYTHON_ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SUBMIT_DIR="$(realpath "${SLURM_SUBMIT_DIR:?Submit from the repository root}")"
test -f "$SUBMIT_DIR/workflows/jobs/perun/recovery/recovery_2.sbatch"
PROJECT_ROOT="$(dirname "$SUBMIT_DIR")"
[[ "$(dirname "$PROJECT_ROOT")" == "/mnt/project" ]] || { echo "Repository must be a direct child of a PERUN PROJECT directory." >&2; exit 2; }
test -w "$PROJECT_ROOT"
HF_HOME="${HF_HOME:?Export a persistent PROJECT HF_HOME before submission}"
[[ "$HF_HOME" == /* && -d "$HF_HOME" && -w "$HF_HOME" ]] || exit 2
HF_HOME="$(realpath "$HF_HOME")"
[[ "$HF_HOME" == "$PROJECT_ROOT"/* ]] || { echo "Recovery-2 requires HF_HOME inside the active PROJECT directory." >&2; exit 2; }
export HF_HOME
# The 8K selection stream is read in place from the copy the researcher located; the runner checks its sha256.
SELECTION_STREAM="${RECOVERY2_SELECTION_STREAM:?Export RECOVERY2_SELECTION_STREAM before submission}"
SELECTION_STREAM="$(realpath "$SELECTION_STREAM")"
[[ "$SELECTION_STREAM" == "$PROJECT_ROOT"/* && -f "$SELECTION_STREAM" ]] || { echo "The selection stream must be a file inside the active PROJECT directory." >&2; exit 2; }

RESULTS_ROOT="$PROJECT_ROOT/perun-results"
PROJECT_RESULTS_ROOT="$RESULTS_ROOT/recovery-2"
SWIGLU8_ROOT="$RESULTS_ROOT/swiglu-8"
# Array order; the configuration's family list repeats it.
FAMILY_NAMES=(dense-int3 b-0.2-int4 swiglu-0.2)
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { usage; exit 2; }
TASK_ID="$SLURM_ARRAY_TASK_ID"
(( TASK_ID >= 0 && TASK_ID < ${#FAMILY_NAMES[@]} )) || { echo "Array index must be 0-$(( ${#FAMILY_NAMES[@]} - 1 ))." >&2; exit 2; }
OUTPUT_NAME="${FAMILY_NAMES[$TASK_ID]}-run-001"
# Every input is read in place; the runner checks each against its sha256 pin.
for INPUT in swiglu-8/prepare-001/result.json swiglu-8/prepare-001/prepared/kl-tokens.int32 \
             swiglu-8/prepare-001/prepared/local-evaluation-batches.pt swiglu-8/block-check-001/result.json \
             swiglu-8/fitted-target-0.2-run-001/result.json \
             swiglu-7/prepare-001/result.json swiglu-7/prepare-001/prepared/recovery-tokens.int32 \
             quantization-2/qat-int3-1b-002/result.json stacking-1/b-fitted-0.2-1b-int4-001/result.json \
             swiglu-7/S7-0-target-0.2-run-001/result.json \
             recovery-1/dense-int3-run-001/result.json recovery-1/b-0.2-int4-run-001/result.json \
             recovery-1/swiglu-0.2-run-001/result.json; do
    test -f "$RESULTS_ROOT/$INPUT" || { echo "Missing pinned input: $RESULTS_ROOT/$INPUT" >&2; exit 2; }
done

PERSISTENT_OUTPUT_DIR="$PROJECT_RESULTS_ROOT/$OUTPUT_NAME"
LOCK_DIR="${PERSISTENT_OUTPUT_DIR}.perun-lock"
SCRATCH_ROOT="/mnt/scratch/${USER:?Slurm did not export USER}"
JOB_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?}}_$TASK_ID"
SCRATCH_DIR="$SCRATCH_ROOT/job_${JOB_KEY}"
test -d "$SCRATCH_ROOT" && test -w "$SCRATCH_ROOT"
[[ "$SCRATCH_DIR" == "$SCRATCH_ROOT"/job_* && ! -e "$SCRATCH_DIR" ]] || { echo "Refusing unsafe or existing scratch: $SCRATCH_DIR" >&2; exit 2; }
[[ ! -e "$PERSISTENT_OUTPUT_DIR" ]] || { echo "Output exists; move it aside before a rerun: $PERSISTENT_OUTPUT_DIR" >&2; exit 2; }

nvidia-smi
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "Slurm did not expose a CUDA device to the job." >&2; exit 2; }
"$PYTHON_BIN" -c \
    'import torch, transformers, datasets, torchao; assert torch.cuda.is_available(), "PyTorch cannot access CUDA"; print("environment preflight: OK", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)'

mkdir -p "$PROJECT_RESULTS_ROOT"
mkdir "$LOCK_DIR" || { echo "Output is locked by another or interrupted job: $LOCK_DIR" >&2; exit 2; }
printf 'job_id=%s\narray_task_id=%s\nscratch_dir=%s\n' \
    "$SLURM_JOB_ID" "$TASK_ID" "$SCRATCH_DIR" > "$LOCK_DIR/owner.txt"
mkdir "$SCRATCH_DIR"
printf '%s\n' "$SLURM_JOB_ID" > "$SCRATCH_DIR/.recovery_2_owner"
export SCRATCH_DIR
export TMPDIR="$SCRATCH_DIR/tmp"
OUTPUT_DIR="$SCRATCH_DIR/results/$OUTPUT_NAME"
WORK_DIR="$TMPDIR/recovery-2-$OUTPUT_NAME"
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
       && [[ -f "$SCRATCH_DIR/.recovery_2_owner" ]] \
       && [[ "$(<"$SCRATCH_DIR/.recovery_2_owner")" == "$SLURM_JOB_ID" ]]; then
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

# Stage only executable code and configurations; pinned inputs are read in place from PROJECT.
for CODE_PATH in src workflows/runs workflows/configs workflows/__init__.py; do
    mkdir -p "$SCRATCH_DIR/$(dirname "$CODE_PATH")"
    rsync -a --exclude='__pycache__/' "$SUBMIT_DIR/$CODE_PATH" "$SCRATCH_DIR/$(dirname "$CODE_PATH")/"
done
mkdir -p "$TMPDIR" "$SCRATCH_DIR/results"
OUTPUT_READY=1

cd "$SCRATCH_DIR"
export PYTHONPATH="$SCRATCH_DIR/src:$SCRATCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
echo "Job: ${SLURM_JOB_ID} on ${SLURM_NODELIST:-unknown}; task=$TASK_ID family=${FAMILY_NAMES[$TASK_ID]} output=$PERSISTENT_OUTPUT_DIR"
"$PYTHON_BIN" -u -m workflows.runs.model.recovery.recovery_2 train \
    --task-index "$TASK_ID" --work-dir "$WORK_DIR" --output-dir "$OUTPUT_DIR" \
    --prepared "$SWIGLU8_ROOT/prepare-001/result.json" --source-root "$RESULTS_ROOT" \
    --selection-stream "$SELECTION_STREAM" &
PYTHON_PID=$!
wait "$PYTHON_PID"
PYTHON_PID=""
