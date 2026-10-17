#!/bin/bash
#SBATCH --job-name=stacking-1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Supply --account, --qos, and the group's planned --time to sbatch; see the PERUN runbook.
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@1800
#SBATCH --export=ALL

set -euo pipefail

usage() {
    echo "Usage: sbatch --array=0-32 $0 group a100m [--resume]" >&2
    echo "   or: sbatch --array=0-8 $0 group a1b [--resume]" >&2
    echo "   or: sbatch --array=0-5 $0 group b [--resume]" >&2
    echo "   or: sbatch $0 variant VARIANT_ID [--resume]" >&2
}

for REQUIRED_COMMAND in rsync realpath cmp nvidia-smi; do
    command -v "$REQUIRED_COMMAND" >/dev/null 2>&1 || {
        echo "Required command is unavailable: $REQUIRED_COMMAND" >&2
        exit 2
    }
done
(( $# >= 2 )) || { usage; exit 2; }
MODE="$1"
SELECTOR="$2"
shift 2
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
test -f "$SUBMIT_DIR/workflows/jobs/perun/stacking_1.sbatch"
PROJECT_ROOT="$(dirname "$SUBMIT_DIR")"
[[ "$(dirname "$PROJECT_ROOT")" == "/mnt/project" ]] || { echo "Repository must be a direct child of a PERUN PROJECT directory." >&2; exit 2; }
test -w "$PROJECT_ROOT"
HF_HOME="${HF_HOME:?Export a persistent PROJECT HF_HOME before submission}"
[[ "$HF_HOME" == /* && -d "$HF_HOME" && -w "$HF_HOME" ]] || exit 2
HF_HOME="$(realpath "$HF_HOME")"
[[ "$HF_HOME" == "$PROJECT_ROOT"/* ]] || { echo "Stacking-1 requires HF_HOME inside the active PROJECT directory." >&2; exit 2; }
export HF_HOME

PROJECT_RESULTS_ROOT="$PROJECT_ROOT/perun-results/stacking-1"
# The swiglu-7, swiglu-8, and swiglu-9 outputs; the runner verifies every source by sha256.
SOURCES_ROOT="$PROJECT_ROOT/perun-results"
SWIGLU8_PREPARED="$SOURCES_ROOT/swiglu-8/prepare-001"
test -f "$SWIGLU8_PREPARED/result.json"
RUNNER=(env PYTHONPATH="$SUBMIT_DIR/src:$SUBMIT_DIR" "$PYTHON_BIN" -m workflows.runs.model.stacking.stacking_1)
case "$MODE" in
    group)
        # Array order comes from the runner's grid, so this script holds no variant list.
        [[ "$SELECTOR" == a100m || "$SELECTOR" == a1b || "$SELECTOR" == b ]] || { usage; exit 2; }
        [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { usage; exit 2; }
        mapfile -t VARIANTS < <(cd "$SUBMIT_DIR" && "${RUNNER[@]}" list-variants --group "$SELECTOR")
        TASK_ID="$SLURM_ARRAY_TASK_ID"
        (( TASK_ID >= 0 && TASK_ID < ${#VARIANTS[@]} )) || { echo "Array index must be 0-$(( ${#VARIANTS[@]} - 1 ))." >&2; exit 2; }
        VARIANT="${VARIANTS[$TASK_ID]}"
        ;;
    variant)
        [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || { usage; exit 2; }
        VARIANT="$SELECTOR"
        ;;
    *)
        usage
        exit 2
        ;;
esac
mapfile -t INPUTS < <(cd "$SUBMIT_DIR" && "${RUNNER[@]}" inputs --variant "$VARIANT" --prepared "$SWIGLU8_PREPARED/result.json")
(( ${#INPUTS[@]} > 0 )) || { echo "No input list for variant $VARIANT" >&2; exit 2; }
OUTPUT_NAME="${VARIANT}-001"

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
# Importing torchao loads MSLK's CUDA library, which needs the GPU driver.
"$PYTHON_BIN" -c \
    'from importlib.metadata import version; from packaging.version import Version; import torch, transformers, datasets, lm_eval, safetensors, psutil; from lm_eval.models.huggingface import HFLM; assert version("torchao").split("+")[0] == "0.17.0", "torchao 0.17.0 is required, found " + version("torchao"); assert version("mslk").startswith("1.1.0"), "mslk 1.1.0 is required, found " + version("mslk"); assert Version(version("accelerate")) >= Version("0.26.0"), "accelerate>=0.26.0 is required"; import torchao; assert torch.cuda.is_available(), "PyTorch cannot access CUDA"; print("environment preflight: OK", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), "torchao", version("torchao"), "mslk", version("mslk"), flush=True)'

mkdir -p "$PROJECT_RESULTS_ROOT"
mkdir "$LOCK_DIR" || { echo "Output is locked by another or interrupted job: $LOCK_DIR" >&2; exit 2; }
printf 'job_id=%s\narray_task_id=%s\nscratch_dir=%s\n' \
    "$SLURM_JOB_ID" "${SLURM_ARRAY_TASK_ID:-single}" "$SCRATCH_DIR" > "$LOCK_DIR/owner.txt"
mkdir "$SCRATCH_DIR"
printf '%s\n' "$SLURM_JOB_ID" > "$SCRATCH_DIR/.stacking_1_owner"
export SCRATCH_DIR
export TMPDIR="$SCRATCH_DIR/tmp"
OUTPUT_DIR="$SCRATCH_DIR/results/$OUTPUT_NAME"
WORK_DIR="$TMPDIR/stacking-1-$OUTPUT_NAME"
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
       && [[ -f "$SCRATCH_DIR/.stacking_1_owner" ]] \
       && [[ "$(<"$SCRATCH_DIR/.stacking_1_owner")" == "$SLURM_JOB_ID" ]]; then
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
mkdir -p "$TMPDIR" "$SCRATCH_DIR/results"
# Stage SwiGLU-8's preparation record, the files this variant reads, and its source run.
PREPARED_INPUT="$SCRATCH_DIR/inputs/swiglu-8-prepare"
SOURCE_INPUT="$SCRATCH_DIR/inputs/sources"
mkdir -p "$PREPARED_INPUT" "$SOURCE_INPUT"
rsync -a "$SWIGLU8_PREPARED/result.json" "$PREPARED_INPUT/"
for INPUT in "${INPUTS[@]}"; do
    KIND="${INPUT%% *}"
    RELATIVE="${INPUT#* }"
    case "$KIND" in
        prepared)
            mkdir -p "$PREPARED_INPUT/$(dirname "$RELATIVE")"
            rsync -a "$SWIGLU8_PREPARED/$RELATIVE" "$PREPARED_INPUT/$RELATIVE"
            ;;
        source)
            mkdir -p "$SOURCE_INPUT/$RELATIVE"
            rsync -a "$SOURCES_ROOT/$RELATIVE/result.json" "$SOURCES_ROOT/$RELATIVE/model" "$SOURCE_INPUT/$RELATIVE/"
            ;;
        reference)
            mkdir -p "$SOURCE_INPUT/$(dirname "$RELATIVE")"
            rsync -a "$SOURCES_ROOT/$RELATIVE" "$SOURCE_INPUT/$RELATIVE"
            ;;
        *)
            echo "Unknown input kind: $INPUT" >&2
            exit 2
            ;;
    esac
done
if (( RESUME )); then
    mkdir -p "$OUTPUT_DIR"
    rsync -a --delete "$PERSISTENT_OUTPUT_DIR/" "$OUTPUT_DIR/"
fi
OUTPUT_READY=1

RESUME_ARGUMENTS=()
if (( RESUME )); then
    RESUME_ARGUMENTS=(--resume)
fi
cd "$SCRATCH_DIR"
export PYTHONPATH="$SCRATCH_DIR/src:$SCRATCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
echo "Job: ${SLURM_JOB_ID} on ${SLURM_NODELIST:-unknown}; variant=$VARIANT output=$PERSISTENT_OUTPUT_DIR"
[[ "$MODE" != group ]] || echo "Array mapping: group=$SELECTOR task=$TASK_ID variant=$VARIANT"
"$PYTHON_BIN" -u -m workflows.runs.model.stacking.stacking_1 run --variant "$VARIANT" \
    --prepared "$PREPARED_INPUT/result.json" --source-root "$SOURCE_INPUT" \
    --work-dir "$WORK_DIR" --output-dir "$OUTPUT_DIR" "${RESUME_ARGUMENTS[@]}" &
PYTHON_PID=$!
wait "$PYTHON_PID"
PYTHON_PID=""
