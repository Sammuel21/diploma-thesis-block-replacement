---
metadata_version: 1
title: TUKE Perun Command Cheat Sheet
type: reference
category: infrastructure
status: active
created: 2026-09-27
modified: 2026-09-27
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---

# TUKE Perun Command Cheat Sheet

Current state:

- copy data between PROJECT and SCRATCH manually;
- keep durable results in PROJECT;
- PERUN provisions `/mnt/scratch/$USER` in advance;
- `.activate_scratch` is unavailable as of 2026-09-27;
- `sacct` is disabled for regular users;
- SwiGLU-7 uses its dedicated manual-staging launcher.

## 1. Connect

Run on your local computer:

```bash
export PERUN_USER="your-perun-username"
ssh "$PERUN_USER@login01.perun.tuke.sk"
```

## 2. Prepare the session and enter the repository

Run after every login:

```bash
export PERUN_ACCOUNT="your-project-account"
export PERUN_QOS="your-project-qos"
export PERUN_PROJECT="/mnt/project/$PERUN_ACCOUNT"
export PERUN_SCRATCH="/mnt/scratch/$USER"

cd "$PERUN_PROJECT/diploma-thesis-block-replacement"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlp-replacement

case ":${LD_LIBRARY_PATH:-}:" in
  *":$CONDA_PREFIX/lib:"*) ;;
  *) export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac

export MLP_REPLACEMENT_PYTHON="$(command -v python)"
export HF_HOME="$PERUN_PROJECT/huggingface-cache"
mkdir -p "$HF_HOME"

pwd
python --version
```

Expected repository path:

```text
/mnt/project/<project-account>/diploma-thesis-block-replacement
```

## 3. Update the repository

Check for local changes:

```bash
git status --short
git log -1 --oneline
```

Pull only when the checkout is clean:

```bash
git pull --ff-only origin llm-wiki
git log -1 --oneline
```

## 4. Check PERUN

Check storage and permissions:

```bash
perunfsusage
ls -ld "$PERUN_PROJECT" "$PERUN_SCRATCH"
test -w "$PERUN_PROJECT" && echo "PROJECT: writable"
test -w "$PERUN_SCRATCH" && echo "SCRATCH: writable"
```

If the user SCRATCH directory is missing or not writable, contact PERUN
support. Do not try to create `/mnt/scratch/$USER` yourself.

Check partitions and existing jobs:

```bash
sinfo -s
squeue -u "$USER"
```

Check the project association and QoS:

```bash
scontrol show assoc_mgr \
  users="$USER" accounts="$PERUN_ACCOUNT" flags=assoc

scontrol show assoc_mgr \
  qos="$PERUN_QOS" flags=qos
```

## 5. Select and validate a job

Example: manual scratch pipeline probe:

```bash
export JOB_FILE="workflows/jobs/perun/scratch_probe.sbatch"
export JOB_NAME="perun-scratch-probe"
```

Check its shell syntax:

```bash
bash -n "$JOB_FILE"
```

Ask Slurm whether it can be scheduled without running it:

```bash
sbatch --test-only \
  --account="$PERUN_ACCOUNT" \
  --qos="$PERUN_QOS" \
  "$JOB_FILE"
```

## 6. Submit the job

Submit and capture the job ID:

```bash
SUBMISSION=$(sbatch --parsable \
  --account="$PERUN_ACCOUNT" \
  --qos="$PERUN_QOS" \
  "$JOB_FILE")

export JOB_ID="${SUBMISSION%%;*}"
echo "JOB_ID=$JOB_ID"
```

Confirm it entered the queue:

```bash
squeue -j "$JOB_ID"
```

## 7. Monitor the job

Check its state:

```bash
squeue -j "$JOB_ID"
```

Show its allocation, node, and failure reason:

```bash
scontrol show job "$JOB_ID"
```

Follow standard output:

```bash
tail -f "${JOB_NAME}_${JOB_ID}.out"
```

Follow standard error:

```bash
tail -f "${JOB_NAME}_${JOB_ID}.err"
```

Use `Ctrl+C` to stop following a log. This does not cancel the job.

## 8. Inspect a completed job

Read the final logs:

```bash
tail -n 100 "${JOB_NAME}_${JOB_ID}.out"
tail -n 100 "${JOB_NAME}_${JOB_ID}.err"
```

Inspect the persistent result directory:

```bash
find "$PERUN_PROJECT/perun-results/job_${JOB_ID}" \
  -maxdepth 5 -type f -print
```

Inspect structured workflow results:

```bash
find "$PERUN_PROJECT/perun-results/job_${JOB_ID}" \
  \( -name 'result.json' -o -name 'run.json' \) \
  -print
```

Check output size:

```bash
du -sh "$PERUN_PROJECT/perun-results/job_${JOB_ID}"
```

## 9. Cancel a job

Cancel one job or an entire array:

```bash
scancel "$JOB_ID"
```

Confirm it left the queue:

```bash
squeue -j "$JOB_ID"
```

## 10. Run SwiGLU-7

Submit preparation from the repository root:

```bash
PREP_SUBMISSION=$(sbatch --parsable \
  --account="$PERUN_ACCOUNT" \
  --qos="$PERUN_QOS" \
  workflows/jobs/perun/swiglu_7.sbatch prepare)

export PREP_JOB="${PREP_SUBMISSION%%;*}"
echo "PREP_JOB=$PREP_JOB"
```

The verified preparation output is:

```bash
export S7_PREPARED="$PERUN_PROJECT/perun-results/swiglu-7/prepare-001/result.json"
test -f "$S7_PREPARED"
```

Submit the 12 fixed trajectories with at most four running concurrently:

```bash
TRAIN_SUBMISSION=$(sbatch --parsable \
  --account="$PERUN_ACCOUNT" \
  --qos="$PERUN_QOS" \
  --array=0-11%4 \
  workflows/jobs/perun/swiglu_7.sbatch train "$S7_PREPARED")

export TRAIN_ARRAY="${TRAIN_SUBMISSION%%;*}"
echo "TRAIN_ARRAY=$TRAIN_ARRAY"
```

Resume preparation:

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  workflows/jobs/perun/swiglu_7.sbatch prepare --resume
```

Resume one failed array task, for example task 6:

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --array=6 \
  workflows/jobs/perun/swiglu_7.sbatch train --resume "$S7_PREPARED"
```

Inspect SwiGLU-7 persistent outputs:

```bash
find "$PERUN_PROJECT/perun-results/swiglu-7" \
  -maxdepth 3 \( -name 'result.json' -o -name 'run.json' \) -print
```

The scheduler logs remain in the repository as
`swiglu-7_<job-id>.out` and `swiglu-7_<job-id>.err`.

## 11. Manual staging inside a job

The user SCRATCH directory must already exist. Create only a unique child
directory for the job, then copy the checkout into it:

```bash
JOB_SCRATCH="/mnt/scratch/${USER:?}/job_${SLURM_JOB_ID:?}"
mkdir "$JOB_SCRATCH"

rsync -a \
  --exclude='.git/' \
  --exclude='*.out' \
  --exclude='*.err' \
  "$SLURM_SUBMIT_DIR/" "$JOB_SCRATCH/"

cd "$JOB_SCRATCH"
```

Copy durable results back to PROJECT:

```bash
JOB_SCRATCH="/mnt/scratch/${USER:?}/job_${SLURM_JOB_ID:?}"
RESULT_DEST="$PERUN_PROJECT/perun-results/job_${SLURM_JOB_ID:?}"
mkdir -p "$RESULT_DEST"
rsync -a "$JOB_SCRATCH/results/" "$RESULT_DEST/"
```

Verify the PROJECT copy before deleting anything from SCRATCH.

## 12. Environment checks

Confirm the interpreter:

```bash
echo "CONDA_PREFIX=$CONDA_PREFIX"
command -v python
python --version
```

Check required packages:

```bash
python -c 'from importlib.metadata import version; import torch, transformers, datasets, lm_eval, huggingface_hub, safetensors, psutil; print("torch:", torch.__version__, "CUDA:", torch.version.cuda); print("transformers:", transformers.__version__); print("datasets:", datasets.__version__); print("lm_eval:", version("lm_eval")); print("imports: OK")'

python -m pip check
```

## 13. Troubleshooting

Confirm identity and project membership:

```bash
id
getent passwd "$(id -u)"
groups
```

Print important paths without printing tokens:

```bash
echo "PERUN_PROJECT=${PERUN_PROJECT:-<unset>}"
echo "PERUN_SCRATCH=${PERUN_SCRATCH:-<unset>}"
echo "CONDA_PREFIX=${CONDA_PREFIX:-<unset>}"
echo "MLP_REPLACEMENT_PYTHON=${MLP_REPLACEMENT_PYTHON:-<unset>}"
echo "HF_HOME=${HF_HOME:-<unset>}"
```

Capture a failed job before its controller record expires:

```bash
scontrol show job "$JOB_ID"
cat "${JOB_NAME}_${JOB_ID}.out"
cat "${JOB_NAME}_${JOB_ID}.err"
```

For job history or official consumption statistics, contact PERUN support.
Regular users cannot use `sacct`.

If a SwiGLU-7 `*.perun-lock` remains, inspect its `owner.txt` and the recorded
SCRATCH directory before resubmitting. The lock means stage-out did not finish;
do not delete it until `result.json` has been copied to PROJECT and verified.

## References

- [Project PERUN guide](perun.md)
- [Current project status](perun-status.md)
- [Experiment log](perun-log.md)
- [Official storage guide](https://wiki.perun.tuke.sk/perun/System_overview/storage/)
- [Official partition guide](https://wiki.perun.tuke.sk/slurm/partitions/)
- Support: `hpc@helpdesk.tuke.sk`
