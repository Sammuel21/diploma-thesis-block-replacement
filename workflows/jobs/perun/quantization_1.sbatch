#!/bin/bash
#SBATCH --job-name=quantization-1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Supply account/QoS and a measured wall-time estimate at submission.
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@1800
#SBATCH --export=ALL

set -euo pipefail
if (( $# < 2 )); then
    echo "Usage: sbatch $0 prepare|run|evaluate OUTPUT_NAME [runner arguments]" >&2
    exit 2
fi
MODE="$1"
OUTPUT_NAME="$2"
shift 2
case "$MODE" in prepare|run|evaluate) ;; *) echo "Unsupported mode: $MODE" >&2; exit 2;; esac
if [[ ! "$OUTPUT_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "OUTPUT_NAME must be a single safe directory name" >&2
    exit 2
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Quantization-1 uses one explicitly selected experiment per job" >&2
    exit 2
fi
PYTHON_BIN="${MLP_REPLACEMENT_PYTHON:?Export an absolute Python interpreter path}"
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || exit 2
PYTHON_BIN="$(realpath "$PYTHON_BIN")"
PYTHON_ENV_PREFIX="$(dirname "$(dirname "$PYTHON_BIN")")"
export LD_LIBRARY_PATH="$PYTHON_ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SUBMIT_DIR="$(realpath "${SLURM_SUBMIT_DIR:?Submit from the repository root}")"
test -f "$SUBMIT_DIR/workflows/jobs/perun/quantization_1.sbatch"
PROJECT_DIR="$(realpath "${PERUN_PROJECT:-$(dirname "$SUBMIT_DIR")}")"
SCRATCH_ROOT="/mnt/scratch/${USER:?}"
test -d "$SCRATCH_ROOT" && test -w "$SCRATCH_ROOT"
test -w "$PROJECT_DIR"
HF_HOME="${HF_HOME:?Export the persistent PROJECT model/dataset cache}"
[[ "$HF_HOME" == /* && -d "$HF_HOME" && -w "$HF_HOME" ]] || exit 2
HF_HOME="$(realpath "$HF_HOME")"
[[ "$HF_HOME" == "$PROJECT_DIR"/* ]] || exit 2
export HF_HOME
PROJECT_RESULTS="$PROJECT_DIR/perun-results/quantization-1"
PERSISTENT_OUTPUT="$PROJECT_RESULTS/$OUTPUT_NAME"
LOCK_DIR="${PERSISTENT_OUTPUT}.perun-lock"
SCRATCH_DIR="$SCRATCH_ROOT/job_${SLURM_JOB_ID:?}"
[[ "$SCRATCH_DIR" == "$SCRATCH_ROOT"/job_* && ! -e "$SCRATCH_DIR" ]] || exit 2
RESUME=0
for ARGUMENT in "$@"; do
    [[ "$ARGUMENT" != "--resume" ]] || RESUME=1
    case "$ARGUMENT" in --output-dir|--output-dir=*|--work-dir|--work-dir=*)
        echo "Launcher owns work/output paths; use OUTPUT_NAME" >&2; exit 2;; esac
done
if (( RESUME )); then
    test -f "$PERSISTENT_OUTPUT/result.json"
elif [[ -e "$PERSISTENT_OUTPUT" ]]; then
    echo "Output exists; use explicit compatible --resume" >&2
    exit 2
fi
mkdir -p "$PROJECT_RESULTS"
mkdir "$LOCK_DIR"
printf 'job_id=%s\n' "$SLURM_JOB_ID" > "$LOCK_DIR/owner.txt"
mkdir "$SCRATCH_DIR"
printf '%s\n' "$SLURM_JOB_ID" > "$SCRATCH_DIR/.quantization_1_owner"
export SCRATCH_DIR
export TMPDIR="$SCRATCH_DIR/tmp"
export RESULTS_DIR="$SCRATCH_DIR/results"
OUTPUT_DIR="$RESULTS_DIR/$OUTPUT_NAME"
WORK_DIR="$TMPDIR/quantization-1"
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
            # Verify every durable byte, not just the summary, before removing scratch.
            comparison="$(rsync -rcn --delete --exclude='*.tmp' --out-format='%i %n' "$OUTPUT_DIR/" "$PERSISTENT_OUTPUT/")"
            [[ $? -eq 0 && -z "$comparison" ]] || stageout_code=1
        fi
    fi
    if (( stageout_code == 0 )) && [[ "$SCRATCH_DIR" == "$SCRATCH_ROOT"/job_* ]] \
       && [[ -f "$SCRATCH_DIR/.quantization_1_owner" ]] \
       && [[ "$(<"$SCRATCH_DIR/.quantization_1_owner")" == "$SLURM_JOB_ID" ]]; then
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
mkdir -p "$WORK_DIR" "$RESULTS_DIR" "$SCRATCH_DIR/inputs"
if (( RESUME )); then
    mkdir -p "$OUTPUT_DIR"
    rsync -a "$PERSISTENT_OUTPUT/" "$OUTPUT_DIR/"
fi
OUTPUT_READY=1
RUN_ARGUMENTS=()
while (( $# )); do
    ARGUMENT="$1"
    shift
    case "$ARGUMENT" in
        --prepared)
            SOURCE="${1:?--prepared requires a result.json}"
            shift
            [[ "$SOURCE" == /* ]] || SOURCE="$SUBMIT_DIR/$SOURCE"
            SOURCE="$(realpath "$SOURCE")"
            mkdir -p "$SCRATCH_DIR/inputs/prepared"
            rsync -a "$(dirname "$SOURCE")/" "$SCRATCH_DIR/inputs/prepared/"
            RUN_ARGUMENTS+=(--prepared "$SCRATCH_DIR/inputs/prepared/$(basename "$SOURCE")")
            ;;
        --bundle)
            SOURCE="${1:?--bundle requires a directory}"
            shift
            [[ "$SOURCE" == /* ]] || SOURCE="$SUBMIT_DIR/$SOURCE"
            SOURCE="$(realpath "$SOURCE")"
            mkdir -p "$SCRATCH_DIR/inputs/reference/model"
            rsync -a "$SOURCE/" "$SCRATCH_DIR/inputs/reference/model/"
            if [[ -f "$(dirname "$SOURCE")/result.json" ]]; then
                rsync -a "$(dirname "$SOURCE")/result.json" "$SCRATCH_DIR/inputs/reference/"
            fi
            RUN_ARGUMENTS+=(--bundle "$SCRATCH_DIR/inputs/reference/model")
            ;;
        --config|--reference-result)
            SOURCE="${1:?Input option requires a file}"
            shift
            [[ "$SOURCE" == /* ]] || SOURCE="$SUBMIT_DIR/$SOURCE"
            TARGET="$SCRATCH_DIR/inputs/${ARGUMENT#--}.json"
            rsync -a "$SOURCE" "$TARGET"
            RUN_ARGUMENTS+=("$ARGUMENT" "$TARGET")
            ;;
        --prepared=*|--bundle=*|--config=*|--reference-result=*)
            echo "Use separate option/value arguments for staged inputs" >&2; exit 2;;
        *) RUN_ARGUMENTS+=("$ARGUMENT");;
    esac
done
cd "$SCRATCH_DIR"
export PYTHONPATH="$SCRATCH_DIR/src:$SCRATCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
nvidia-smi
"$PYTHON_BIN" -u -m workflows.runs.model.baseline.quantization "$MODE" \
    --output-dir "$OUTPUT_DIR" --work-dir "$WORK_DIR" "${RUN_ARGUMENTS[@]}" &
PYTHON_PID=$!
wait "$PYTHON_PID"
PYTHON_PID=""
