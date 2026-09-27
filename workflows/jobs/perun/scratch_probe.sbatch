#!/bin/bash
#SBATCH --job-name=perun-scratch-probe
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Supply --account and --qos to sbatch; see README.md.
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --export=ALL

set -eo pipefail

source .activate_scratch

set -u

: "${SCRATCH_DIR:?Perun scratch did not define SCRATCH_DIR}"
: "${TMPDIR:?Perun scratch did not define TMPDIR}"
: "${RESULTS_DIR:?Perun scratch did not define RESULTS_DIR}"

test "$PWD" = "$SCRATCH_DIR"
test -d "$TMPDIR"
test -d "$RESULTS_DIR"

PROBE_RESULT="$RESULTS_DIR/perun-scratch-probe.txt"
{
    printf 'status=ok\n'
    printf 'job_id=%s\n' "$SLURM_JOB_ID"
    printf 'node=%s\n' "$SLURM_NODELIST"
    printf 'submit_dir=%s\n' "$SLURM_SUBMIT_DIR"
    printf 'scratch_dir=%s\n' "$SCRATCH_DIR"
    printf 'tmpdir=%s\n' "$TMPDIR"
    printf 'results_dir=%s\n' "$RESULTS_DIR"
} > "$PROBE_RESULT"

printf 'PERUN automatic scratch probe passed.\n'
printf 'Probe result: %s\n' "$PROBE_RESULT"
