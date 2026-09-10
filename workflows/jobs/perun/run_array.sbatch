#!/bin/bash
#SBATCH --job-name=mlp-array
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# Supply --account and --qos to sbatch; see README.md.
#SBATCH --partition=gpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --export=ALL

if ! source .activate_scratch; then
    echo "Perun scratch activation failed; submit this job from the repository root." >&2
    exit 1
fi
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
    echo "Usage: sbatch --array=START-END [Slurm options] $0 MANIFEST [OUTPUT_ROOT]" >&2
    exit 2
fi

MANIFEST_PATH="$1"
if [[ ! -f "$MANIFEST_PATH" ]]; then
    echo "Manifest does not exist in the staged repository: $MANIFEST_PATH" >&2
    exit 2
fi

ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?Submit this file with sbatch --array=START-END}"
mapfile -t CONFIG_PATHS < <(
    sed -e 's/\r$//' \
        -e 's/^[[:space:]]*//' \
        -e 's/[[:space:]]*$//' \
        -e '/^#/d' \
        -e '/^$/d' \
        "$MANIFEST_PATH"
)

if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= ${#CONFIG_PATHS[@]} )); then
    echo "Array index $ARRAY_INDEX is outside manifest range 0-$((${#CONFIG_PATHS[@]} - 1))" >&2
    exit 2
fi

CONFIG_PATH="${CONFIG_PATHS[$ARRAY_INDEX]}"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Configuration does not exist in the staged repository: $CONFIG_PATH" >&2
    exit 2
fi

CONFIG_NAME="$(basename "$CONFIG_PATH")"
CONFIG_STEM="${CONFIG_NAME%.*}"
OUTPUT_ROOT="${2:-data/results/perun/arrays/${SLURM_ARRAY_JOB_ID}}"
OUTPUT_PATH="${OUTPUT_ROOT}/${CONFIG_STEM}-${SLURM_ARRAY_JOB_ID}_${ARRAY_INDEX}.json"
PYTHON_BIN="${MLP_REPLACEMENT_PYTHON:-python3}"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Array job: ${SLURM_ARRAY_JOB_ID}, task: $ARRAY_INDEX, node: ${SLURM_NODELIST}"
echo "Work directory: $PWD"
echo "Configuration: $CONFIG_PATH"
echo "Output: $OUTPUT_PATH"
echo "Python: $PYTHON_BIN"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

"$PYTHON_BIN" -m workflows.runs.run_experiment \
    "$CONFIG_PATH" \
    --output "$OUTPUT_PATH"
