#!/bin/bash
#SBATCH --job-name=mlp-experiment
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
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
    echo "Usage: sbatch [Slurm options] $0 CONFIG_JSON [OUTPUT_JSON]" >&2
    exit 2
fi

CONFIG_PATH="$1"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Configuration does not exist in the staged repository: $CONFIG_PATH" >&2
    exit 2
fi

CONFIG_NAME="$(basename "$CONFIG_PATH")"
CONFIG_STEM="${CONFIG_NAME%.*}"
OUTPUT_PATH="${2:-data/results/perun/${CONFIG_STEM}-${SLURM_JOB_ID}.json}"
PYTHON_BIN="${MLP_REPLACEMENT_PYTHON:-python3}"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Job: ${SLURM_JOB_ID} on ${SLURM_NODELIST}"
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
