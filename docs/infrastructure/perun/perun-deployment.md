---
metadata_version: 1
title: TUKE Perun Deployment Blueprint
type: procedure
category: infrastructure/perun
status: active
created: 2026-09-28
modified: 2026-09-28
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---

# TUKE Perun Deployment Blueprint

## Purpose

Workflow-neutral commands for deploying one repository job to PERUN. Each
experiment document owns its job script, arguments, result path, optional array
range, resume behavior, and job-specific dependency checks.

Run cluster commands on a PERUN login node unless a section says **local
computer**. Replace placeholder values only in the shell; never commit account,
QoS, username, credential, or token values.

## 1. Set up the login session

```bash
export PERUN_ACCOUNT="your-project-account"
export PERUN_QOS="your-project-qos"
export PERUN_PROJECT="/mnt/project/$PERUN_ACCOUNT"
export PERUN_SCRATCH="/mnt/scratch/$USER"
export PERUN_LOG_DIR="$PERUN_PROJECT/perun-job-logs"
export PERUN_REPOSITORY="diploma-thesis-block-replacement"

cd "$PERUN_PROJECT/$PERUN_REPOSITORY"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlp-replacement

case ":${LD_LIBRARY_PATH:-}:" in
  *":$CONDA_PREFIX/lib:"*) ;;
  *) export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac

export MLP_REPLACEMENT_PYTHON="$(command -v python)"
export HF_HOME="$PERUN_PROJECT/huggingface-cache"
mkdir -p "$HF_HOME" "$PERUN_LOG_DIR"
```

## 2. Update the repository

Pull only from a clean checkout:

```bash
git status --short
git pull --ff-only origin llm-wiki
git log -1 --oneline
```

## 3. Configure the job

Set these values from the experiment guide:

```bash
export JOB_SCRIPT="workflows/jobs/perun/replace-me.sbatch"
export JOB_NAME="replace-with-slurm-job-name"
export RESULT_RELATIVE_PATH="perun-results/replace-with-workflow/run-id"

JOB_ARGS=()
```

Add positional arguments only when the job requires them:

```bash
JOB_ARGS=("first-argument" "second-argument")
```

## 4. Validate before submission

```bash
test -f "$JOB_SCRIPT"
test -w "$PERUN_PROJECT" && echo "PROJECT: writable"
test -w "$PERUN_SCRATCH" && echo "SCRATCH: writable"

"$MLP_REPLACEMENT_PYTHON" --version
"$MLP_REPLACEMENT_PYTHON" -m pip check
bash -n "$JOB_SCRIPT"
```

Run any additional dependency, model, dataset, configuration, or artifact
checks required by the experiment guide.

## 5. Submit one job

```bash
JOB_SUBMISSION=$(sbatch --parsable \
  --account="$PERUN_ACCOUNT" \
  --qos="$PERUN_QOS" \
  --output="$PERUN_LOG_DIR/%x_%j.out" \
  --error="$PERUN_LOG_DIR/%x_%j.err" \
  "$JOB_SCRIPT" "${JOB_ARGS[@]}")

export JOB_ID="${JOB_SUBMISSION%%;*}"
echo "JOB_ID=$JOB_ID"
```

## 6. Submit an array when required

Skip this section unless the experiment defines independent array tasks.

```bash
export ARRAY_SPEC="0-3%4"

JOB_SUBMISSION=$(sbatch --parsable \
  --account="$PERUN_ACCOUNT" \
  --qos="$PERUN_QOS" \
  --array="$ARRAY_SPEC" \
  --output="$PERUN_LOG_DIR/%x_%A_%a.out" \
  --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
  "$JOB_SCRIPT" "${JOB_ARGS[@]}")

export JOB_ID="${JOB_SUBMISSION%%;*}"
echo "JOB_ID=$JOB_ID"
```

The value after `%` is only the maximum concurrent array-task count. Each job
script still determines the resources requested by one task.

## 7. Monitor

```bash
squeue -j "$JOB_ID"
squeue -r -j "$JOB_ID"
scontrol show job "$JOB_ID"
```

Inspect scheduler logs:

```bash
ls -lt "$PERUN_LOG_DIR/${JOB_NAME}"_*.out "$PERUN_LOG_DIR/${JOB_NAME}"_*.err
tail -n 100 "$PERUN_LOG_DIR/${JOB_NAME}_${JOB_ID}.out"
tail -n 100 "$PERUN_LOG_DIR/${JOB_NAME}_${JOB_ID}.err"
```

Array-task log names may include the array and task IDs. Use the `ls` command
to obtain their exact names.

## 8. Verify persistent results

```bash
export RESULT_PATH="$PERUN_PROJECT/$RESULT_RELATIVE_PATH"

test -e "$RESULT_PATH"
find "$RESULT_PATH" -maxdepth 3 \
  \( -name result.json -o -name run.json \) -print
du -sh "$RESULT_PATH"
```

Inspect structured status when the workflow writes `result.json`:

```bash
"$MLP_REPLACEMENT_PYTHON" -c 'import json, os; path = os.path.join(os.environ["RESULT_PATH"], "result.json"); result = json.load(open(path)); print("status:", result["status"]); print("path:", path)'
```

Do not treat disappearance from `squeue` or the presence of output files as
proof of scientific completion. Use the workflow's structured status and
validation contract.

## 9. Retrieve one result

Run on the **local computer** while connected to the PERUN VPN:

```bash
export PERUN_USER="your-perun-username"
export PERUN_ACCOUNT="your-project-account"
export RESULT_RELATIVE_PATH="perun-results/replace-with-workflow/run-id"
export LOCAL_RESULT="./perun-results/run-id"

mkdir -p "$LOCAL_RESULT"

rsync -av --partial --progress \
  "$PERUN_USER@login01.perun.tuke.sk:/mnt/project/$PERUN_ACCOUNT/$RESULT_RELATIVE_PATH/" \
  "$LOCAL_RESULT/"
```

Retrieve scheduler logs separately:

```bash
export JOB_NAME="replace-with-slurm-job-name"
mkdir -p ./perun-job-logs

rsync -av \
  --include="${JOB_NAME}_*.out" \
  --include="${JOB_NAME}_*.err" \
  --exclude='*' \
  "$PERUN_USER@login01.perun.tuke.sk:/mnt/project/$PERUN_ACCOUNT/perun-job-logs/" \
  ./perun-job-logs/
```

## 10. Handle failure

Capture the controller record before it expires:

```bash
scontrol show job "$JOB_ID"
ls -lt "$PERUN_LOG_DIR/${JOB_NAME}"_*.out "$PERUN_LOG_DIR/${JOB_NAME}"_*.err
```

Then inspect `run.json`, `result.json`, scheduler stderr, persistent locks, and
surviving SCRATCH. Use only the resume command defined by the experiment; some
jobs are resumable and others must restart with a new output identity.

## 11. Verify cleanup

These commands inspect state and do not delete anything:

```bash
find "$PERUN_PROJECT/perun-results" \
  -type d -name '*.perun-lock' -print

find "$PERUN_SCRATCH" -maxdepth 1 -type d \
  -name "job_${JOB_ID}*" -print
```

No output means no matching lock or job SCRATCH directory remains. Inspect any
remaining lock and its `owner.txt` before attempting recovery. Do not manually
delete SCRATCH as a routine cleanup step.

## References

- [PERUN operating contract](perun.md)
- [PERUN command cheat sheet](perun-cheatsheet.md)
- [Current PERUN status](perun-status.md)
- [PERUN experiment log](perun-log.md)

