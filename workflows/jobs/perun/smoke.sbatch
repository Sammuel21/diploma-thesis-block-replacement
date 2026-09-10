#!/bin/bash
#SBATCH --job-name=mlp-smoke
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Supply --account and --qos to sbatch; see README.md.
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --export=ALL

if ! source .activate_scratch; then
    echo "Perun scratch activation failed; submit this job from the repository root." >&2
    exit 1
fi
set -euo pipefail

PYTHON_BIN="${MLP_REPLACEMENT_PYTHON:-python3}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Job: ${SLURM_JOB_ID} on ${SLURM_NODELIST}"
echo "Work directory: $PWD"
echo "Python: $PYTHON_BIN"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

"$PYTHON_BIN" -m workflows.runs.run_experiment \
    configs/experiments/perun-smoke-linear.json \
    --output "data/results/perun/smoke-${SLURM_JOB_ID}.json"
