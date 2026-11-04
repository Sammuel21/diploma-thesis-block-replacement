#!/bin/bash
#SBATCH --job-name=heterogeneous-1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Supply --account, --qos, and the stage's planned --time and --mem to sbatch; see the README.
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --export=ALL

set -euo pipefail

usage() {
    echo "Usage: sbatch --array=0-23 $0 select" >&2
    echo "   or: sbatch $0 baseline-table" >&2
    echo "   or: sbatch --array=0-21 $0 cost-table" >&2
    echo "   or: sbatch --array=0-2 $0 recover" >&2
    echo "   or: sbatch --partition=cpu_short --gres=none --cpus-per-task=4 --mem=16G $0 solve" >&2
    echo "   or: sbatch --array=0-7 $0 assemble" >&2
}

(( $# == 1 )) || { usage; exit 2; }
MODE="$1"
for REQUIRED_COMMAND in rsync realpath cmp; do
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
test -f "$SUBMIT_DIR/workflows/jobs/perun/heterogenous/heterogeneous_1.sbatch"
PROJECT_ROOT="$(dirname "$SUBMIT_DIR")"
[[ "$(dirname "$PROJECT_ROOT")" == "/mnt/project" ]] || { echo "Repository must be a direct child of a PERUN PROJECT directory." >&2; exit 2; }
test -w "$PROJECT_ROOT"
HF_HOME="${HF_HOME:?Export a persistent PROJECT HF_HOME before submission}"
[[ "$HF_HOME" == /* && -d "$HF_HOME" && -w "$HF_HOME" ]] || exit 2
HF_HOME="$(realpath "$HF_HOME")"
[[ "$HF_HOME" == "$PROJECT_ROOT"/* ]] || { echo "Heterogeneous-1 requires HF_HOME inside the active PROJECT directory." >&2; exit 2; }
export HF_HOME

PROJECT_RESULTS_ROOT="$PROJECT_ROOT/perun-results/heterogeneous-1"
SWIGLU8_ROOT="$PROJECT_ROOT/perun-results/swiglu-8"
test -f "$SWIGLU8_ROOT/prepare-001/result.json"
test -f "$SWIGLU8_ROOT/block-check-001/result.json"
# Array orders; the runner's task maps repeat them.
RECOVERY_LAYERS=(3 11 19)
ASSEMBLE_TARGETS=(0.2 0.3 0.4 0.5 0.6 0.7 0.75 0.8)

array_index() {
    [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { usage; exit 2; }
    (( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID <= $1 )) || { echo "$MODE array index must be 0-$1." >&2; exit 2; }
    TASK_ID="$SLURM_ARRAY_TASK_ID"
}
single_job() {
    [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || { usage; exit 2; }
}
TASK_ID=""
case "$MODE" in
    select)
        array_index 23
        OUTPUT_NAME="$(printf 'select-001-task-%02d' "$TASK_ID")"
        ;;
    baseline-table)
        single_job
        OUTPUT_NAME="baseline-table-001"
        ;;
    cost-table)
        # Task i is eligible layer i + 1.
        array_index 21
        OUTPUT_NAME="$(printf 'cost-table-001-layer-%02d' "$(( TASK_ID + 1 ))")"
        ;;
    recover)
        array_index 2
        OUTPUT_NAME="$(printf 'recovery-001-layer-%02d' "${RECOVERY_LAYERS[$TASK_ID]}")"
        test -f "$SWIGLU8_ROOT/prepare-001/prepared/recovery-tokens.int32"
        ;;
    solve)
        single_job
        OUTPUT_NAME="solve-001"
        ;;
    assemble)
        array_index 7
        OUTPUT_NAME="assemble-001-target-${ASSEMBLE_TARGETS[$TASK_ID]}"
        test -f "$PROJECT_RESULTS_ROOT/solve-001/result.json"
        test -f "$SWIGLU8_ROOT/prepare-001/prepared/local-evaluation-batches.pt"
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
[[ ! -e "$PERSISTENT_OUTPUT_DIR" ]] || { echo "Output exists; move it aside before a rerun: $PERSISTENT_OUTPUT_DIR" >&2; exit 2; }

if [[ "$MODE" != "solve" ]]; then
    command -v nvidia-smi >/dev/null 2>&1 || { echo "Required command is unavailable: nvidia-smi" >&2; exit 2; }
    nvidia-smi
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "Slurm did not expose a CUDA device to the job." >&2; exit 2; }
    "$PYTHON_BIN" -c \
        'import torch, transformers, datasets; assert torch.cuda.is_available(), "PyTorch cannot access CUDA"; print("environment preflight: OK", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)'
fi

mkdir -p "$PROJECT_RESULTS_ROOT"
mkdir "$LOCK_DIR" || { echo "Output is locked by another or interrupted job: $LOCK_DIR" >&2; exit 2; }
printf 'job_id=%s\narray_task_id=%s\nscratch_dir=%s\n' \
    "$SLURM_JOB_ID" "${SLURM_ARRAY_TASK_ID:-single}" "$SCRATCH_DIR" > "$LOCK_DIR/owner.txt"
mkdir "$SCRATCH_DIR"
printf '%s\n' "$SLURM_JOB_ID" > "$SCRATCH_DIR/.heterogeneous_1_owner"
export SCRATCH_DIR
export TMPDIR="$SCRATCH_DIR/tmp"
OUTPUT_DIR="$SCRATCH_DIR/results/$OUTPUT_NAME"
WORK_DIR="$TMPDIR/heterogeneous-1-$OUTPUT_NAME"
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
       && [[ -f "$SCRATCH_DIR/.heterogeneous_1_owner" ]] \
       && [[ "$(<"$SCRATCH_DIR/.heterogeneous_1_owner")" == "$SLURM_JOB_ID" ]]; then
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

RUN_ARGUMENTS=("$MODE" --work-dir "$WORK_DIR" --output-dir "$OUTPUT_DIR" --swiglu8-root "$SWIGLU8_ROOT")
[[ -z "$TASK_ID" ]] || RUN_ARGUMENTS+=(--task-index "$TASK_ID")
case "$MODE" in
    cost-table|recover|solve|assemble)
        RUN_ARGUMENTS+=(--results-root "$PROJECT_RESULTS_ROOT")
        ;;
esac

cd "$SCRATCH_DIR"
export PYTHONPATH="$SCRATCH_DIR/src:$SCRATCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
echo "Job: ${SLURM_JOB_ID} on ${SLURM_NODELIST:-unknown}; mode=$MODE output=$PERSISTENT_OUTPUT_DIR"
"$PYTHON_BIN" -u -m workflows.runs.model.heterogenous.heterogeneous_1 "${RUN_ARGUMENTS[@]}" &
PYTHON_PID=$!
wait "$PYTHON_PID"
PYTHON_PID=""
