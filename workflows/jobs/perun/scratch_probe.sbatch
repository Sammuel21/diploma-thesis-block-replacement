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

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:?Slurm did not define SLURM_SUBMIT_DIR}"
SCRATCH_ROOT="/mnt/scratch/${USER:?Slurm did not export USER}"
SCRATCH_DIR="$SCRATCH_ROOT/job_${SLURM_JOB_ID}"
PROJECT_RESULTS_ROOT="$(dirname "$SUBMIT_DIR")/perun-results"
STAGEOUT_DIR="$PROJECT_RESULTS_ROOT/job_${SLURM_JOB_ID}"

test -f "$SUBMIT_DIR/workflows/jobs/perun/scratch_probe.sbatch"
test -d "$SCRATCH_ROOT"
test -w "$SCRATCH_ROOT"
test -w "$(dirname "$SUBMIT_DIR")"

case "$SCRATCH_DIR" in
    "$SCRATCH_ROOT"/job_*) ;;
    *)
        echo "Refusing unsafe scratch path: $SCRATCH_DIR" >&2
        exit 2
        ;;
esac

if [[ -e "$SCRATCH_DIR" || -e "$STAGEOUT_DIR" ]]; then
    echo "Refusing to reuse an existing probe directory." >&2
    exit 2
fi

mkdir "$SCRATCH_DIR"
printf 'created_by_job=%s\n' "$SLURM_JOB_ID" > "$SCRATCH_DIR/.probe_owner"

preserve_scratch_on_failure() {
    exit_code=$?
    if (( exit_code != 0 )); then
        echo "Probe failed; scratch preserved at $SCRATCH_DIR" >&2
    fi
}
trap preserve_scratch_on_failure EXIT

rsync -a \
    --exclude='.git/' \
    --exclude='*.out' \
    --exclude='*.err' \
    --exclude='results_job_*/' \
    --exclude-from="$SUBMIT_DIR/.rsyncignore" \
    "$SUBMIT_DIR/" "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"
export SCRATCH_DIR
export TMPDIR="$SCRATCH_DIR/tmp"
export RESULTS_DIR="$SCRATCH_DIR/results"
mkdir -p "$TMPDIR" "$RESULTS_DIR"

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

mkdir -p "$PROJECT_RESULTS_ROOT"
mkdir "$STAGEOUT_DIR"
rsync -a "$RESULTS_DIR/" "$STAGEOUT_DIR/"
test -f "$STAGEOUT_DIR/perun-scratch-probe.txt"

if [[ ! -f "$SCRATCH_DIR/.probe_owner" ]] \
    || [[ "$(<"$SCRATCH_DIR/.probe_owner")" != "created_by_job=$SLURM_JOB_ID" ]]; then
    echo "Refusing to clean an unverified scratch directory." >&2
    exit 2
fi
rm -rf -- "$SCRATCH_DIR"
trap - EXIT

printf 'PERUN manual scratch probe passed.\n'
printf 'Persistent result: %s\n' "$STAGEOUT_DIR/perun-scratch-probe.txt"
