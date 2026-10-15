#!/bin/bash
#SBATCH --job-name=speed-1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Supply account/QoS at submission. One job on one GPU, so every speed is comparable.
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --export=ALL

set -euo pipefail
if (( $# < 1 )); then
    echo "Usage: sbatch $0 OUTPUT_NAME [--resume]" >&2
    exit 2
fi
OUTPUT_NAME="$1"
shift
if [[ ! "$OUTPUT_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "OUTPUT_NAME must be a single safe directory name" >&2
    exit 2
fi
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "Speed-1 takes no array: all models share one GPU" >&2; exit 2; }
RESUME=0
for ARGUMENT in "$@"; do
    case "$ARGUMENT" in
        --resume) RESUME=1;;
        *) echo "Unsupported argument: $ARGUMENT (the configuration is fixed)" >&2; exit 2;;
    esac
done
PYTHON_BIN="${MLP_REPLACEMENT_PYTHON:?Export an absolute Python interpreter path}"
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || exit 2
PYTHON_BIN="$(realpath "$PYTHON_BIN")"
PYTHON_ENV_PREFIX="$(dirname "$(dirname "$PYTHON_BIN")")"
export LD_LIBRARY_PATH="$PYTHON_ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SUBMIT_DIR="$(realpath "${SLURM_SUBMIT_DIR:?Submit from the repository root}")"
test -f "$SUBMIT_DIR/workflows/jobs/perun/speed_1.sbatch"
PROJECT_DIR="$(realpath "${PERUN_PROJECT:-$(dirname "$SUBMIT_DIR")}")"
SCRATCH_ROOT="/mnt/scratch/${USER:?}"
test -d "$SCRATCH_ROOT" && test -w "$SCRATCH_ROOT"
test -w "$PROJECT_DIR"
HF_HOME="${HF_HOME:?Export the persistent PROJECT model/dataset cache}"
[[ "$HF_HOME" == /* && -d "$HF_HOME" && -w "$HF_HOME" ]] || exit 2
HF_HOME="$(realpath "$HF_HOME")"
[[ "$HF_HOME" == "$PROJECT_DIR"/* ]] || exit 2
export HF_HOME
# The measured bundles are read in place; Speed-1 never writes below them.
SOURCE_ROOT="$PROJECT_DIR/perun-results"
test -d "$SOURCE_ROOT/quantization-1" && test -d "$SOURCE_ROOT/swiglu-8"
PROJECT_RESULTS="$SOURCE_ROOT/speed-1"
PERSISTENT_OUTPUT="$PROJECT_RESULTS/$OUTPUT_NAME"
LOCK_DIR="${PERSISTENT_OUTPUT}.perun-lock"
SCRATCH_DIR="$SCRATCH_ROOT/job_${SLURM_JOB_ID:?}"
[[ "$SCRATCH_DIR" == "$SCRATCH_ROOT"/job_* && ! -e "$SCRATCH_DIR" ]] || exit 2
if (( RESUME )); then
    test -f "$PERSISTENT_OUTPUT/result.json"
elif [[ -e "$PERSISTENT_OUTPUT" ]]; then
    echo "Output exists; use explicit compatible --resume" >&2
    exit 2
fi
# Same environment check as Quantization-1: the INT8/INT4 bundles need TorchAO and MSLK.
nvidia-smi
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "Slurm did not expose a CUDA device to the job." >&2; exit 2; }
PREFLIGHT='from importlib.metadata import version; import torch, transformers, safetensors, psutil, torchao; assert version("torchao").split("+")[0] == "0.17.0", "torchao 0.17.0 is required, found " + version("torchao"); assert version("mslk").startswith("1.1.0"), "mslk 1.1.0 is required, found " + version("mslk"); assert torch.cuda.is_available(), "PyTorch cannot access CUDA"; print("environment preflight: OK", torch.__version__, "torchao", version("torchao"), "CUDA device:", torch.cuda.get_device_name(0), flush=True)'
"$PYTHON_BIN" -c "$PREFLIGHT"
mkdir -p "$PROJECT_RESULTS"
mkdir "$LOCK_DIR"
printf 'job_id=%s\n' "$SLURM_JOB_ID" > "$LOCK_DIR/owner.txt"
mkdir "$SCRATCH_DIR"
printf '%s\n' "$SLURM_JOB_ID" > "$SCRATCH_DIR/.speed_1_owner"
export SCRATCH_DIR
export TMPDIR="$SCRATCH_DIR/tmp"
OUTPUT_DIR="$SCRATCH_DIR/results/$OUTPUT_NAME"
PYTHON_PID=""
OUTPUT_READY=0

finalize_job() {
    exit_code=$?
    trap - EXIT TERM INT
    set +e
    stageout_code=1
    if (( OUTPUT_READY )) && [[ -f "$OUTPUT_DIR/result.json" ]]; then
        mkdir -p "$PERSISTENT_OUTPUT"
        rsync -a --delete --exclude='*.tmp' "$OUTPUT_DIR/" "$PERSISTENT_OUTPUT/"
        stageout_code=$?
        if (( stageout_code == 0 )); then
            cmp -s "$OUTPUT_DIR/result.json" "$PERSISTENT_OUTPUT/result.json" || stageout_code=1
            comparison="$(rsync -rcn --delete --exclude='*.tmp' --out-format='%i %n' "$OUTPUT_DIR/" "$PERSISTENT_OUTPUT/")"
            [[ $? -eq 0 && -z "$comparison" ]] || stageout_code=1
        fi
    fi
    if (( stageout_code == 0 )) && [[ "$SCRATCH_DIR" == "$SCRATCH_ROOT"/job_* ]] \
       && [[ -f "$SCRATCH_DIR/.speed_1_owner" ]] \
       && [[ "$(<"$SCRATCH_DIR/.speed_1_owner")" == "$SLURM_JOB_ID" ]]; then
        rm -rf -- "$SCRATCH_DIR" || stageout_code=1
        if (( stageout_code == 0 )); then
            rm -f "$LOCK_DIR/owner.txt"
            rmdir "$LOCK_DIR" || stageout_code=1
            echo "Verified persistent output: $PERSISTENT_OUTPUT"
        fi
    else
        stageout_code=1
        echo "Stage-out incomplete; scratch and lock preserved: $SCRATCH_DIR" >&2
    fi
    (( exit_code != 0 || stageout_code == 0 )) || exit_code=1
    exit "$exit_code"
}
terminate_job() {
    if [[ -n "$PYTHON_PID" ]] && kill -0 "$PYTHON_PID" 2>/dev/null; then
        kill -TERM "$PYTHON_PID" 2>/dev/null || true
        wait "$PYTHON_PID" 2>/dev/null || true
    fi
    exit "$1"
}
trap finalize_job EXIT
trap 'terminate_job 143' TERM
trap 'terminate_job 130' INT

# Stage only executable code/configs, independently of the historical .rsyncignore.
for CODE_PATH in src workflows/runs workflows/configs workflows/__init__.py; do
    mkdir -p "$SCRATCH_DIR/$(dirname "$CODE_PATH")"
    rsync -a --exclude='__pycache__/' "$SUBMIT_DIR/$CODE_PATH" "$SCRATCH_DIR/$(dirname "$CODE_PATH")/"
done
mkdir -p "$TMPDIR" "$SCRATCH_DIR/results"
if (( RESUME )); then
    mkdir -p "$OUTPUT_DIR"
    rsync -a "$PERSISTENT_OUTPUT/" "$OUTPUT_DIR/"
fi
OUTPUT_READY=1
RUN_ARGUMENTS=()
(( RESUME == 0 )) || RUN_ARGUMENTS+=(--resume)
cd "$SCRATCH_DIR"
export PYTHONPATH="$SCRATCH_DIR/src:$SCRATCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
"$PYTHON_BIN" -u -m workflows.runs.model.baseline.speed \
    --source-root "$SOURCE_ROOT" --output-dir "$OUTPUT_DIR" ${RUN_ARGUMENTS[@]+"${RUN_ARGUMENTS[@]}"} &
PYTHON_PID=$!
wait "$PYTHON_PID"
PYTHON_PID=""
