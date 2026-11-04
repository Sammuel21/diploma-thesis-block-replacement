---
title: PERUN Runbook
status: active
created: 2026-10-11
modified: 2026-11-04
---

# PERUN Runbook

Copy-paste commands for running the project's experiments on PERUN, from
login to copying results home. It replaces the earlier cheat sheet and
deployment template.

Each block is labelled:

- **Safe to re-run.** It only reads, checks, or sets up the shell. Run it as
  often as you like.
- **Submits a job. Run once.** It starts GPU work. Each such block refuses to
  submit when the output already exists or a matching job is queued. The job
  scripts also refuse an existing output or lock, so an accidental second
  submission exits at once; it only wastes a queue slot.
- **Deliberate.** It changes or removes files. Run it only when the text says
  so, after reading the listing that precedes it.

Your private values (login, account, QoS, branch) are never in git. They live
in `local/perun/runs/session.md` on your PC and in `~/.perun-private.sh` on
PERUN. Record each job ID in `local/perun/runs/<experiment>.md`.

The rules behind these commands are in the [PERUN guide](perun.md). Resource
planning is in [perun-resources.md](perun-resources.md), and the job history is
in [perun-log.md](perun-log.md) and [perun-status.md](perun-status.md).

**What you submit, in total: 5 submissions.** SwiGLU-8 has three: the block
check, prepare, and the training array of 16 runs (section 3). Quantization-1
has two: prepare, and the five model runs (section 4). The second is one block
with two `sbatch` lines, because its tasks need different time limits. The two
experiments are independent and can run at the same time. SwiGLU-9
(section 9) adds three more, submitted back to back in one block: the five 1B
training runs, the three-task block check, and the reference KL. Stacking-1
(section 10) adds three more, submitted back to back in one block: order B,
order A at 100M, and order A at 1B, 48 tasks in total. Quantization-2
(section 11) adds three single runs on Quantization-1's launcher, after
Stacking-1 has finished. Heterogeneous-1 (section 12) adds five sittings, each
after a review of the one before.

**Official PERUN rules these commands follow**, checked against the
[PERUN wiki](https://wiki.hpc.tuke.sk/perun/perun/) on 2026-10-11:

- HOME holds code, configurations, and environments; it is not for large data
  or computation.
- Jobs compute on SCRATCH and copy results to PROJECT before they end. SCRATCH
  is deleted 60 days after last access, and nothing is copied automatically.
- Every submission sets `--account` and `--qos`.
- GPU partitions are only for jobs that use a GPU, with the GPU requested
  explicitly (`--gres=gpu:1`). Quantization-1's `prepare` uses no GPU and goes
  to a CPU partition.
- Wall times stay close to the expected run time, because inflated limits
  lengthen queue waits. `gpu_short` allows at most 2 days.
- No `--nodelist` and no long-running interactive jobs.
- Transfers between your PC and PERUN run from your PC.
- Miniconda lives in `~/miniconda3`, with one environment per project and
  nothing installed into `base`.

## 0. One-time setup

### 0.1 Private settings on PERUN

**Deliberate; run once.** Running it again overwrites the file. Fill in the
three values from `local/perun/runs/session.md` before pasting.

```bash
cat > ~/.perun-private.sh <<'EOF'
export PERUN_ACCOUNT="<project-account>"
export PERUN_QOS="<qos>"
export PERUN_BRANCH="<branch checked out on PERUN>"
EOF
chmod 600 ~/.perun-private.sh
```

### 0.2 Add the Quantization-1 packages to the project environment

Both experiments use the one project environment, `mlp-replacement`. It
already has the SwiGLU-7 software stack that Quantization-1 pins (PyTorch
2.11.0+cu128, Transformers 5.14.1, Datasets 5.0.0, LM Evaluation Harness
0.4.13). Quantization-1 adds only TorchAO 0.17.0 and MSLK 1.1.0+cu128.

The install uses `--no-deps`, so pip cannot upgrade or replace an existing
package. The block saves the package list before the first install and
compares it with the list afterwards. Only the two new packages may differ,
because SwiGLU-8's reproduction checks need the rest exactly as SwiGLU-7
recorded it. Run it before your first SwiGLU-8 or Quantization-1 submission,
so every job uses the same environment.

**Safe to re-run.** The "before" files are written only on the first run, and
pip skips packages that are already installed.

```bash
source ~/.perun-private.sh
cd "/mnt/project/$PERUN_ACCOUNT/diploma-thesis-block-replacement"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlp-replacement
perunfsusage   # HOME needs free space for the two packages
[[ -e ~/conda-mlp-replacement-before-q1.yml ]] || conda env export -n mlp-replacement > ~/conda-mlp-replacement-before-q1.yml
[[ -e ~/pip-mlp-replacement-before-q1.txt ]] || python -m pip freeze > ~/pip-mlp-replacement-before-q1.txt
python -m pip install --no-cache-dir --no-deps \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  "torchao==0.17.0" "mslk==1.1.0+cu128"
python -m pip freeze > ~/pip-mlp-replacement-after-q1.txt
CHANGED=$(diff ~/pip-mlp-replacement-before-q1.txt ~/pip-mlp-replacement-after-q1.txt | grep -E '^[<>]' | grep -vE '^> (torchao|mslk)==')
if [[ -n "$CHANGED" ]]; then
    echo "STOP: the install changed other packages:"; echo "$CHANGED"
else
    echo "OK: no existing package changed. Added:"
    grep -E '^(torchao|mslk)==' ~/pip-mlp-replacement-after-q1.txt
fi
python -m pip check
```

On STOP, submit nothing and paste the output. The saved
`~/conda-mlp-replacement-before-q1.yml` can rebuild the original environment.

### 0.3 Check the environment

**Safe to re-run.** Run 1.2 and 1.3 first. The packages listed are the ones the
Quantization-1 job checks before it starts. Login nodes have no GPU driver.
Importing `torchao` there fails with `libcuda.so.1: cannot open shared object
file`, because it loads MSLK's CUDA library. So this check reads the TorchAO
and MSLK versions without importing them. Each GPU job imports them and checks
CUDA on its node before it takes its output lock.

```bash
python -c 'from importlib.metadata import version; import torch, transformers, datasets, lm_eval, safetensors, psutil; from lm_eval.models.huggingface import HFLM; print("mlp-replacement OK:", torch.__version__, "transformers", transformers.__version__, "accelerate", version("accelerate"), "triton", version("triton"), "torchao", version("torchao"), "mslk", version("mslk"))'
python -m pip check
```

### 0.4 Save the environment specification

**Safe to re-run;** it refreshes the file. The PERUN Conda guide recommends
keeping an export of each environment in HOME, so a broken environment can be
rebuilt.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda env export -n mlp-replacement > ~/conda-mlp-replacement.yml
ls -l ~/conda-*.yml
```

## 1. Every login

### 1.1 Connect (on your PC)

Connect the PERUN VPN first. Use your login from
`local/perun/runs/session.md`.

```bash
ssh "<login>@login01.perun.tuke.sk"
```

### 1.2 Session

**Safe to re-run.**

```bash
source ~/.perun-private.sh
export PERUN_PROJECT="/mnt/project/$PERUN_ACCOUNT"
export PERUN_SCRATCH="/mnt/scratch/$USER"
export PERUN_RESULTS="$PERUN_PROJECT/perun-results"
export PERUN_LOG_DIR="$PERUN_PROJECT/perun-job-logs"
export HF_HOME="$PERUN_PROJECT/huggingface-cache"
export PERUN_BASE_LD_LIBRARY_PATH="${PERUN_BASE_LD_LIBRARY_PATH-${LD_LIBRARY_PATH:-}}"
mkdir -p "$HF_HOME" "$PERUN_LOG_DIR"
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
source ~/miniconda3/etc/profile.d/conda.sh
pwd
```

### 1.3 Activate the environment

**Safe to re-run.** Both experiments use `mlp-replacement`. Jobs use the
Python that `MLP_REPLACEMENT_PYTHON` points to *at submission*, so run this
before submitting.

```bash
conda activate mlp-replacement
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${PERUN_BASE_LD_LIBRARY_PATH:+:$PERUN_BASE_LD_LIBRARY_PATH}"
export MLP_REPLACEMENT_PYTHON="$(command -v python)"
echo "Jobs will use: $MLP_REPLACEMENT_PYTHON"
```

### 1.4 Update the repository

**Safe to re-run.** `git status` must print nothing; if it lists files, stop
and sort them out first.

Do not pull while one of your jobs is still waiting in the queue. A job copies
the code when it *starts*, not when you submit it. A resume also refuses code
that changed since the original run.

```bash
squeue -u "$USER"
git status --short
git pull --ff-only origin "$PERUN_BRANCH"
git log -1 --oneline
```

## 2. Status at a glance

Run 1.2 and 1.3 first; these use the environment's `python`.

### 2.1 Everything at once

**Safe to re-run.** A running job's result lives on SCRATCH and reaches
PROJECT only when the job ends.

```bash
squeue -u "$USER" -o "%.10i %.18j %.9T %.10M %.11l %R"
python - <<'EOF'
import json, os
from pathlib import Path

def show(result, label):
    data = json.loads(result.read_text())
    run = result.with_name("run.json")
    run_data = json.loads(run.read_text()) if run.exists() else {}
    stage = data.get("current_stage") or run_data.get("stage") or "-"
    progress = run_data.get("progress") or ""
    print(f"  {label:44} {data.get('status', '?'):10} {stage:26} {str(progress)[:50]} {(data.get('error') or '')[:80]}")

results = Path(os.environ["PERUN_RESULTS"])
print("Finished or stopped (PROJECT):")
for experiment in ("swiglu-8", "quantization-1", "swiglu-9"):
    for result in sorted((results / experiment).glob("*/result.json")):
        show(result, f"{experiment}/{result.parent.name}")
print("Running now (SCRATCH):")
for result in sorted(Path(os.environ["PERUN_SCRATCH"]).glob("job_*/results/*/result.json")):
    show(result, f"{result.parents[2].name}/{result.parent.name}")
EOF
find "$PERUN_RESULTS" -maxdepth 2 -name '*.perun-lock' -print   # any line here: see 5.2
ls -t "$PERUN_LOG_DIR" | head -n 6
gpu-status   # GPU use of your running jobs; `gpu-status <job-id>` shows per-GPU detail
```

### 2.2 Read one result

**Safe to re-run.** Set `RUN` to the output you want, for example
`swiglu-8/block-check-001`, `swiglu-8/prepare-001` or
`swiglu-8/fitted-target-0.2-run-001`.

```bash
export RUN=swiglu-8/block-check-001
python - <<'EOF'
import json, os
result = json.load(open(os.path.join(os.environ["PERUN_RESULTS"], os.environ["RUN"], "result.json")))
print("status:", result.get("status"), "| error:", result.get("error"))
for group, summary in result.get("results", {}).get("acceptance", {}).items():
    print(f"  acceptance {group}: {'passed' if summary['passed'] else 'FAILED'}")
adoption = result.get("results", {}).get("adoption")
if adoption:
    selected = adoption.get("selected_recipe", adoption.get("selected_setting"))   # SwiGLU-8 or SwiGLU-9
    print("selected setting:", selected, "| qualifying:", adoption["qualifying"])
    print("stage-2 indicator:", result["results"].get("stage_two_indicator"))
EOF
sha256sum "$PERUN_RESULTS/$RUN/result.json"
```

### 2.3 Cancel a job

**Deliberate.** It stops the job; the job copies its partial output to PROJECT
on the way out.

```bash
scancel "replace-with-job-id"
```

## 3. SwiGLU-8

Design and decisions: [swiglu-8.md](../../experiments/model/swiglu/swiglu-8.md).
Every submission needs the researcher's go.

Three submissions, the same shape as SwiGLU-7 plus the block check in front:

| Step | Block | Requested time | Expected | Before the next step |
| --- | --- | --- | --- | --- |
| Pre-submission checks (G2) | 3.1 | - | minutes | All checks pass |
| **Submission 1:** block check, 15 tasks on 15 GPUs (one per setting) | 3.2 | 6 h per task | 1-3 h per task, about 2-3 h in total | Merge (3.4); A1-A3 pass; you choose the setting and set `part_b` |
| **Submission 2:** prepare | 3.5 | 24 h | 3-8 h | P1-P5 pass |
| **Submission 3:** training array, 16 runs on 16 GPUs | 3.6 | 6 h per run | 1.2-1.5 h per run, plus evaluation | - |

Training task map: tasks 0-7 are the fitted arm and 8-15 the slice-only arm,
each at targets 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75 and 0.8 in that order. Task
11, for example, is slice-only at 0.5. Tasks 0-3 (fitted, 20-50%) also compare
themselves with SwiGLU-7 at every monitoring point. They stop only on a
token-0 mismatch (B1). Later differences (B2) are recorded with their exact
values and reviewed against 1% afterwards (decision log, 2026-10-12).

Run 1.2, 1.3 and 1.4 first, and 0.2 once before the first submission.

### 3.1 Pre-submission checks (G2)

**Safe to re-run.** Every line should print OK or a value, never MISMATCH or
DIFFERS.

```bash
# SwiGLU-7 inputs exist and match the sha256 values pinned in the configuration.
python - <<'EOF'
import hashlib, json, os
sources = json.load(open("workflows/configs/model/swiglu/swiglu-8.json"))["sources"]
root = os.path.join(os.environ["PERUN_RESULTS"], "swiglu-7")
expected = {"prepare-001/result.json": sources["swiglu7_prepare_sha256"]}
expected.update({f"S7-0-target-{target}-run-001/result.json": digest
                 for target, digest in sources["swiglu7_runs_sha256"].items()})
for relative, digest in expected.items():
    path = os.path.join(root, relative)
    if not os.path.isfile(path):
        print("MISSING ", relative)
        continue
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    print("OK      " if sha.hexdigest() == digest else "MISMATCH", relative)
EOF
ls -l "$PERUN_RESULTS/swiglu-7/prepare-001/prepared/recovery-tokens.int32"

# The environment matches the one SwiGLU-7 recorded.
PYTHONPATH=src python - <<'EOF'
import json, os
from mlp_replacement.runlog import environment_record
recorded = json.load(open(os.path.join(os.environ["PERUN_RESULTS"], "swiglu-7/prepare-001/result.json")))["environment"]
current = environment_record()
print("python  ", "OK" if recorded["python"] == current["python"] else f"DIFFERS {recorded['python']} -> {current['python']}")
for name, version in sorted(recorded["packages"].items()):
    now = current["packages"].get(name)
    print(f"{name:8}", "OK" if now == version else f"DIFFERS {version} -> {now}")
EOF

# Storage, launcher syntax, and whether Slurm would accept the job (nothing is submitted).
perunfsusage                    # SCRATCH needs about 2 GB per block-check task
bash -n workflows/jobs/perun/swiglu_8.sbatch && echo "launcher syntax OK"
sbatch --test-only --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=06:00:00 --array=0-14 \
  workflows/jobs/perun/swiglu_8.sbatch block-check
ls "$PERUN_RESULTS/swiglu-8" 2>/dev/null || echo "no SwiGLU-8 outputs yet"
grep -A2 '"part_b"' workflows/configs/model/swiglu/swiglu-8.json   # both null before the block check
```

The 100M stream check (P4) needs no value from you: the job compares its
stream with SwiGLU-7's file listed above.

### 3.2 Submit the block check

**Submits a job. Run once.** One command starts 15 tasks, one per setting.
Task 0 is SwiGLU-7's setting and also runs the reproduction checks (P5,
A1-A3).

```bash
EXISTING=$(find "$PERUN_RESULTS/swiglu-8" -maxdepth 1 -name 'block-check-001*' 2>/dev/null)
if [[ -n "$EXISTING" ]] || squeue -u "$USER" -h -n swiglu-8 | grep -q .; then
    echo "STOP: block-check outputs exist or a swiglu-8 job is queued:"; echo "$EXISTING"
else
    JOB=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=06:00:00 --array=0-14 \
        --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
        workflows/jobs/perun/swiglu_8.sbatch block-check)
    echo "block-check array: ${JOB%%;*}   <- record it in local/perun/runs/swiglu-8.md"
fi
```

### 3.3 Watch the block check

**Safe to re-run.** Each task first loads the model and builds its data
(about 20-40 minutes), then runs its setting's fits. Task 0 checks the
unfitted slices (A1) and the SwiGLU-7 reproduction (A2-A3) and stops on a
mismatch with `AcceptanceFailure` in its `.err` file.

```bash
ARRAY=$(ls -t "$PERUN_LOG_DIR"/swiglu-8_*_0.out | head -n 1 | sed -E 's/.*swiglu-8_([0-9]+)_0\.out/\1/')
squeue -u "$USER" -n swiglu-8
echo "fits finished: $(cat "$PERUN_LOG_DIR"/swiglu-8_"$ARRAY"_*.out 2>/dev/null | grep -c '^setting=') of 246"
for TASK in $(seq 0 14); do
    LAST=$(grep '^setting=' "$PERUN_LOG_DIR/swiglu-8_${ARRAY}_${TASK}.out" 2>/dev/null | tail -n 1)
    FAILED=$(grep -l -E 'AcceptanceFailure|Traceback' "$PERUN_LOG_DIR/swiglu-8_${ARRAY}_${TASK}.err" 2>/dev/null)
    echo "task $TASK: ${LAST:-no fit finished yet}${FAILED:+   <- check its .err}"
done
gpu-status
```

### 3.4 Merge the block check and set Part B

**Safe to re-run.** When all 15 tasks have finished (`squeue` shows none),
merge them on the login node; it needs no GPU and reads only JSON. The merge
checks that every task used identical data and neuron rankings, joins the
fits, computes the adoption decision, and writes
`swiglu-8/block-check-001/result.json`. Running it again changes nothing.

```bash
PYTHONPATH=src python -m workflows.runs.model.swiglu.swiglu_8 merge-block-check \
  --results-root "$PERUN_RESULTS/swiglu-8"
```

It prints the selected setting, the stage-2 indicator, and the sha256 for
`part_b`. If it reports a missing or unfinished task, check that task's log,
resume it (3.7), and merge again.

Then copy the results to your PC (section 6) and review the notebook's Part A
and "why" sections.

Then, **on your PC**, set `part_b` in
`workflows/configs/model/swiglu/swiglu-8.json` to the selected setting and the
sha256 printed by the merge:

```json
"part_b": {
  "fitting_recipe": "<selected setting, e.g. c1e-3-128>",
  "block_check_sha256": "<sha256 of block-check-001/result.json>"
},
```

Commit and push from your PC. Then pull on PERUN (1.4) and check:

```bash
grep -A2 '"part_b"' workflows/configs/model/swiglu/swiglu-8.json
```

If the stage-2 indicator reports overfitting, stage 2 needs its own decision
before preparation.

### 3.5 Submit preparation

**Submits a job. Run once.**

```bash
BLOCK="$PERUN_RESULTS/swiglu-8/block-check-001/result.json"
OUT="$PERUN_RESULTS/swiglu-8/prepare-001"
if [[ -e "$OUT" || -e "$OUT.perun-lock" ]] || squeue -u "$USER" -h -n swiglu-8 | grep -q .; then
    echo "STOP: preparation exists, is locked, or a swiglu-8 job is queued"
else
    JOB=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=24:00:00 \
        --output="$PERUN_LOG_DIR/%x_%j.out" --error="$PERUN_LOG_DIR/%x_%j.err" \
        workflows/jobs/perun/swiglu_8.sbatch prepare "$BLOCK")
    echo "prepare job: ${JOB%%;*}   <- record it in local/perun/runs/swiglu-8.md"
fi
```

Watch it with the 3.3 commands; width-curve lines start with `curve`. When it
ends, read it with 2.2 and `RUN=swiglu-8/prepare-001`. P1-P5 must all pass.

### 3.6 Submit the training array

**Submits a job. Run once.** One command starts all 16 runs; Slurm runs as
many at once as your QoS allows and queues the rest.

```bash
export S8_PREPARED="$PERUN_RESULTS/swiglu-8/prepare-001/result.json"
EXISTING=$(find "$PERUN_RESULTS/swiglu-8" -maxdepth 1 \( -name 'fitted-target-*' -o -name 'slice-target-*' \) 2>/dev/null)
if [[ -n "$EXISTING" ]]; then
    echo "STOP: these outputs already exist:"; echo "$EXISTING"
else
    JOB=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=06:00:00 --array=0-15 \
        --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
        workflows/jobs/perun/swiglu_8.sbatch train "$S8_PREPARED")
    echo "training array: ${JOB%%;*}   <- record it in local/perun/runs/swiglu-8.md"
fi
```

**Safe to re-run:** watch all 16 runs. Each line shows a task's latest
monitoring point. An `AcceptanceFailure` in tasks 0-3 means the SwiGLU-7
reproduction failed; stop and investigate before trusting any result.

```bash
ARRAY=$(ls -t "$PERUN_LOG_DIR"/swiglu-8_*_0.out | head -n 1 | sed -E 's/.*swiglu-8_([0-9]+)_0\.out/\1/')
squeue -u "$USER" -n swiglu-8
for TASK in $(seq 0 15); do
    LAST=$(grep '^arm=' "$PERUN_LOG_DIR/swiglu-8_${ARRAY}_${TASK}.out" 2>/dev/null | tail -n 1)
    FAILED=$(grep -l 'AcceptanceFailure' "$PERUN_LOG_DIR/swiglu-8_${ARRAY}_${TASK}.err" 2>/dev/null)
    echo "task $TASK: ${LAST:-not started}${FAILED:+   <- AcceptanceFailure}"
done
```

### 3.7 Resume a stopped stage

**Submits a job. Run once, after the cause is understood.** Resume continues
the same run from its last saved point. It refuses changed code,
configuration, or environment, and needs the run's `result.json` on PROJECT
with no lock (see 5.2).

```bash
# One block-check task; replace 7 with the task number (0-14):
sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=06:00:00 --array=7 \
  --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
  workflows/jobs/perun/swiglu_8.sbatch block-check --resume

# Preparation:
sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=24:00:00 \
  --output="$PERUN_LOG_DIR/%x_%j.out" --error="$PERUN_LOG_DIR/%x_%j.err" \
  workflows/jobs/perun/swiglu_8.sbatch prepare --resume "$PERUN_RESULTS/swiglu-8/block-check-001/result.json"

# One training task; replace 6 with the task number from the map above:
sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=06:00:00 --array=6 \
  --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
  workflows/jobs/perun/swiglu_8.sbatch train --resume "$PERUN_RESULTS/swiglu-8/prepare-001/result.json"
```

### 3.8 Rerun training tasks from token 0

**Deliberate, then submits a job. Run once, after the cause is understood**,
for example when code changed and a resume would be refused. The job script
always writes `<arm>-target-<t>-run-001`. The block therefore first renames
each stopped output, keeping it as evidence, then submits only those tasks.
It refuses if a task's output is locked or still running. Set `TASKS` and
`SUFFIX` first.

```bash
TASKS="0 1 2 3"          # task numbers from the map above
SUFFIX="b2-stopped"      # why the first attempt was set aside
TARGETS=(0.2 0.3 0.4 0.5 0.6 0.7 0.75 0.8); ARMS=(fitted slice)
ROOT="$PERUN_RESULTS/swiglu-8"
if squeue -u "$USER" -h -n swiglu-8 | grep -q .; then
    echo "STOP: a swiglu-8 job is still queued or running"
else
    OK=1
    for T in $TASKS; do
        NAME="${ARMS[$((T / 8))]}-target-${TARGETS[$((T % 8))]}-run-001"
        if [[ -e "$ROOT/$NAME.perun-lock" || -e "$ROOT/$NAME-$SUFFIX" ]]; then
            echo "STOP: $NAME is locked or already set aside"; OK=0
        fi
    done
    if (( OK )); then
        for T in $TASKS; do
            NAME="${ARMS[$((T / 8))]}-target-${TARGETS[$((T % 8))]}-run-001"
            [[ ! -d "$ROOT/$NAME" ]] || { mv "$ROOT/$NAME" "$ROOT/$NAME-$SUFFIX" && echo "set aside: $NAME-$SUFFIX"; }
        done
        JOB=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=06:00:00 \
            --array="$(echo $TASKS | tr ' ' ',')" \
            --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
            workflows/jobs/perun/swiglu_8.sbatch train "$ROOT/prepare-001/result.json")
        echo "rerun array: ${JOB%%;*}"
    fi
fi
```

## 4. Quantization-1

Design: [quantization-1.md](../../experiments/model/baseline/quantization-1.md).
Quantization-1 uses the same `mlp-replacement` environment as SwiGLU-8, with
TorchAO and MSLK added once by 0.2; 0.3 checks it. Run 1.2 and 1.3 before any
submission.

Two submissions:

1. **Prepare.** It freezes the evaluation data and uses no GPU, so it goes to
   a CPU partition. The launcher checks for a GPU only in the other modes.
2. **The five models.** One block starts all five, each as its own task of
   `run-array`. It contains two `sbatch` lines, because one array can have
   only one time limit: tasks 0-3 get 8 hours and task 4 gets 36 hours.

| Task | Model | Output | Expected run time | Time limit |
| --- | --- | --- | --- | --- |
| 0 | Original BF16 (control) | `dense-001` | 0.5-2 h | 8 h |
| 1 | INT8 | `int8-001` | 0.5-2 h | 8 h |
| 2 | INT4 | `int4-001` | 0.5-2 h | 8 h |
| 3 | INT4 retrained on 100M tokens | `qat-int4-100m-001` | 1.5-3 h | 8 h |
| 4 | INT4 retrained on 1B tokens | `qat-int4-1b-001` | 13-20 h; about 25 h if fake quantization doubles the cost | 36 h |

Quantization-1 has never run, so these run times are estimates. The
[experiment guide](../../experiments/model/baseline/quantization-1.md) gives
their basis. Speed is reported only as raw measurements, because the tasks run
on different GPUs (decision of 2026-10-11).

### 4.1 Check that Slurm accepts the requests

**Safe to re-run;** nothing is submitted. Run 1.2 and 1.3 first.

```bash
sbatch --test-only --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  --partition=cpu_short --gres=none --cpus-per-task=4 --mem=32G --time=01:00:00 \
  workflows/jobs/perun/quantization_1.sbatch prepare prepare-001
sbatch --test-only --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=08:00:00 --array=0-3 \
  workflows/jobs/perun/quantization_1.sbatch run-array --prepared "$PERUN_RESULTS/quantization-1/prepare-001/result.json"
sbatch --test-only --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=36:00:00 --array=4 \
  workflows/jobs/perun/quantization_1.sbatch run-array --prepared "$PERUN_RESULTS/quantization-1/prepare-001/result.json"
```

If the first command is rejected because of `--gres=none` or the QoS, stop
and paste the message.

### 4.2 Submit preparation

**Submits a job. Run once.**

```bash
OUT="$PERUN_RESULTS/quantization-1/prepare-001"
if [[ -e "$OUT" || -e "$OUT.perun-lock" ]] || squeue -u "$USER" -h -n quantization-1 | grep -q .; then
    echo "STOP: preparation exists, is locked, or a quantization-1 job is queued"
else
    JOB=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
        --partition=cpu_short --gres=none --cpus-per-task=4 --mem=32G --time=01:00:00 \
        --output="$PERUN_LOG_DIR/%x_%j.out" --error="$PERUN_LOG_DIR/%x_%j.err" \
        workflows/jobs/perun/quantization_1.sbatch prepare prepare-001)
    echo "quantization-1 prepare job: ${JOB%%;*}   <- record it in local/perun/runs/quantization-1.md"
fi
```

When it ends, read it with 2.2 and `RUN=quantization-1/prepare-001`.

### 4.3 Submit the five models

**Submits jobs. Run once,** after preparation completed. It submits two
arrays: tasks 0-3 (8 hours) and task 4 (36 hours).

```bash
export Q1_PREPARED="$PERUN_RESULTS/quantization-1/prepare-001/result.json"
EXISTING=$(find "$PERUN_RESULTS/quantization-1" -maxdepth 1 \( -name 'dense-001*' -o -name 'int8-001*' -o -name 'int4-001*' -o -name 'qat-int4-*' \) 2>/dev/null)
if [[ -n "$EXISTING" ]]; then
    echo "STOP: these outputs already exist:"; echo "$EXISTING"
else
    JOB=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=08:00:00 --array=0-3 \
        --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
        workflows/jobs/perun/quantization_1.sbatch run-array --prepared "$Q1_PREPARED")
    echo "quantization-1 tasks 0-3: ${JOB%%;*}   <- record it in local/perun/runs/quantization-1.md"
    JOB=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=36:00:00 --array=4 \
        --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
        workflows/jobs/perun/quantization_1.sbatch run-array --prepared "$Q1_PREPARED")
    echo "quantization-1 task 4:   ${JOB%%;*}   <- record it in local/perun/runs/quantization-1.md"
fi
```

### 4.4 Watch the five models

**Safe to re-run.** It shows each task's newest log, including a resumed
one. Section 2.1 also shows each running task's current stage. In the first
hour of tasks 3 and 4, the training progress lines show the real QAT speed.

```bash
squeue -u "$USER" -n quantization-1
for TASK in 0 1 2 3 4; do
    LOG=$(ls -t "$PERUN_LOG_DIR"/quantization-1_*_"$TASK".out 2>/dev/null | head -n 1)
    LAST=$([[ -n "$LOG" ]] && tail -n 1 "$LOG")
    FAILED=$([[ -n "$LOG" ]] && grep -l 'Traceback' "${LOG%.out}.err" 2>/dev/null)
    echo "task $TASK: ${LAST:-not started}${FAILED:+   <- check ${FAILED}}"
done
gpu-status
```

### 4.5 Resume one model task

**Submits a job. Run once, after the cause is understood.** Set `TASK` to the
task number from the table above; the block picks its time limit.

```bash
TASK=4
LIMIT=$([[ "$TASK" == 4 ]] && echo 36:00:00 || echo 08:00:00)
sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time="$LIMIT" --array="$TASK" \
  --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
  workflows/jobs/perun/quantization_1.sbatch run-array --prepared "$PERUN_RESULTS/quantization-1/prepare-001/result.json" --resume
```

### 4.6 Rerun INT4 QAT 100M as `-002`

**The end-to-end validation of the INT4 QAT fix,** submitted first. Submit
10.6 after this run's start check passes. The fix and its checks are in
[quantization-1.md](../../experiments/model/baseline/quantization-1.md#int4-qat-conversion-gap-2026-11-01).
Run 1.2, 1.3, and 1.4 first: the checkout must contain the fix commit.

**Pre-submission checks. Safe to re-run.** The `git log` line must show the
fix commit, the `--test-only` line must be accepted, and `qat-int4-100m-002`
must not exist.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
Q1_PREPARED="$PERUN_RESULTS/quantization-1/prepare-001/result.json"
git log -1 --oneline
bash -n workflows/jobs/perun/quantization_1.sbatch && echo "launcher syntax OK"
sbatch --test-only --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=04:00:00 \
  workflows/jobs/perun/quantization_1.sbatch run qat-int4-100m-002 --prepared "$Q1_PREPARED" \
  --variant qat-int4 --qat-tokens 100000000
find "$PERUN_RESULTS/quantization-1" -maxdepth 1 -name 'qat-int4-100m-002*' | grep . || echo "no -002 output yet"
squeue -u "$USER"
```

**Submits a job. Run once,** after the checks are clean. It writes the job ID
to `$PERUN_LOG_DIR/quantization-1-int4-rerun_submission.txt`.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
Q1_PREPARED="$PERUN_RESULTS/quantization-1/prepare-001/result.json"
EXISTING=$(find "$PERUN_RESULTS/quantization-1" -maxdepth 1 -name 'qat-int4-100m-002*')
if [[ -n "$EXISTING" ]]; then
    echo "STOP: the -002 output or lock already exists:"; echo "$EXISTING"
else
    QINT4=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=04:00:00 \
        --output="$PERUN_LOG_DIR/%x_%j.out" --error="$PERUN_LOG_DIR/%x_%j.err" \
        workflows/jobs/perun/quantization_1.sbatch run qat-int4-100m-002 --prepared "$Q1_PREPARED" \
        --variant qat-int4 --qat-tokens 100000000)
    printf 'QINT4=%s\n' "${QINT4%%;*}" | tee "$PERUN_LOG_DIR/quantization-1-int4-rerun_submission.txt"
    echo "<- record this in local/perun/runs/quantization-1.md"
fi
```

**Start check. Safe to re-run,** about 10-20 minutes after the job starts.
The `mismatched` list must be empty for all 66 projections; then submit 10.6.
If the run stops instead, its `.err` file says "INT4 fake-quantized weights
differ": stop and tell the assistant.

While the job runs, its record is on scratch; it is copied to
`$PERUN_RESULTS` only when the job ends.

```bash
source "$PERUN_LOG_DIR/quantization-1-int4-rerun_submission.txt"
squeue -j "$QINT4"
python -c "import json; r = json.load(open('$PERUN_SCRATCH/job_$QINT4/results/qat-int4-100m-002/result.json')); c = r['results'].get('int4_fake_quantization_check'); print(r['status'], 'check:', c and (c['projections'], c['mismatched']))"
tail -n 5 "$PERUN_LOG_DIR/quantization-1_$QINT4.err"
```

**After it finishes** (about 2 h): a mismatch at conversion stops the run
with "Deployed INT4 weights differ" before export. The two lines below must
print `completed`, `[]`, and the same stream sha256 twice (same tokens as
`-001`). Then do sections 7 and 6.

```bash
python -c "import json; r = json.load(open('$PERUN_RESULTS/quantization-1/qat-int4-100m-002/result.json')); print(r['status'], r['results']['conversion'].get('int4_fake_equals_deployed', {}).get('mismatched'))"
for RUN in qat-int4-100m-001 qat-int4-100m-002; do python -c "import json; print('$RUN', json.load(open('$PERUN_RESULTS/quantization-1/$RUN/result.json'))['results']['recovery_stream']['sha256'])"; done
```

### 4.7 Rerun INT4 QAT 1B as `-002`

**After 4.6 completed with an empty conversion mismatch list** (decision log,
2026-11-01). Same checkout as 4.6. The pre-checks follow 4.6, with the 1B
output, budget, and limit.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
Q1_PREPARED="$PERUN_RESULTS/quantization-1/prepare-001/result.json"
git log -1 --oneline
sbatch --test-only --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=24:00:00 \
  workflows/jobs/perun/quantization_1.sbatch run qat-int4-1b-002 --prepared "$Q1_PREPARED" \
  --variant qat-int4 --qat-tokens 1000000000
find "$PERUN_RESULTS/quantization-1" -maxdepth 1 -name 'qat-int4-1b-002*' | grep . || echo "no 1B -002 output yet"
```

**Submits a job. Run once,** after the checks are clean.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
Q1_PREPARED="$PERUN_RESULTS/quantization-1/prepare-001/result.json"
EXISTING=$(find "$PERUN_RESULTS/quantization-1" -maxdepth 1 -name 'qat-int4-1b-002*')
if [[ -n "$EXISTING" ]]; then
    echo "STOP: the 1B -002 output or lock already exists:"; echo "$EXISTING"
else
    QINT4B=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=24:00:00 \
        --output="$PERUN_LOG_DIR/%x_%j.out" --error="$PERUN_LOG_DIR/%x_%j.err" \
        workflows/jobs/perun/quantization_1.sbatch run qat-int4-1b-002 --prepared "$Q1_PREPARED" \
        --variant qat-int4 --qat-tokens 1000000000)
    printf 'QINT4B=%s\n' "${QINT4B%%;*}" | tee "$PERUN_LOG_DIR/quantization-1-int4-1b-rerun_submission.txt"
fi
```

**Watch** with the start check of 4.6, using `$QINT4B` and
`qat-int4-1b-002`. After it finishes, use the two "after it finishes" lines of
4.6 with `qat-int4-1b-001` and `qat-int4-1b-002`. A stopped run resumes with
4.6's `run` line plus `--resume`, from the same checkout.

## 5. When a job fails

### 5.1 Capture the record

**Safe to re-run.** Do this soon after the failure: Slurm forgets finished jobs
after a while, and `sacct` is not available to users.

```bash
JOB="replace-with-job-id"   # from squeue or local/perun/runs/<experiment>.md
scontrol show job "$JOB" | grep -E 'JobId|JobName|JobState|ExitCode|Reason|RunTime|TimeLimit|StartTime|EndTime|NodeList|AllocTRES'
tail -n 40 "$PERUN_LOG_DIR"/*_"$JOB"*.err
```

Then read the run's result with 2.2. Its `error` field names the failing step.

### 5.2 A leftover lock

A `<run>.perun-lock` directory means the job could not finish copying its
output to PROJECT. While the lock exists, new submissions and resumes of that
run refuse to start.

**Safe to re-run:** look first.

```bash
OUT="$PERUN_RESULTS/swiglu-8/block-check-001"   # the locked run
cat "$OUT.perun-lock/owner.txt"
SCR=$(sed -n 's/^scratch_dir=//p' "$OUT.perun-lock/owner.txt")
SCR="${SCR:-$PERUN_SCRATCH/job_$(sed -n 's/^job_id=//p' "$OUT.perun-lock/owner.txt")}"
echo "job scratch: $SCR"
ls -la "$SCR/results/$(basename "$OUT")/" 2>&1 | head
ls -la "$OUT/" 2>&1 | head
```

**Deliberate**, only if the listing shows a `result.json` in the job scratch:
copy it to PROJECT, verify the copy, then release the lock. The `rsync -rcn`
line must print nothing.

```bash
rsync -a --exclude='*.tmp' "$SCR/results/$(basename "$OUT")/" "$OUT/"
rsync -rcn --exclude='*.tmp' --out-format='%i %n' "$SCR/results/$(basename "$OUT")/" "$OUT/"
rm "$OUT.perun-lock/owner.txt" && rmdir "$OUT.perun-lock"
```

**Deliberate**, only if neither the job scratch nor PROJECT has a
`result.json`. The job died before writing anything, for example at an
environment error, so there is nothing to keep:

```bash
rm "$OUT.perun-lock/owner.txt" && rmdir "$OUT.perun-lock"
[[ "$SCR" == "$PERUN_SCRATCH"/job_* ]] && rm -r "$SCR"
```

When in doubt, paste the listing to the assistant before deleting anything.

## 6. Copy results to your PC

The notebooks read a reduced copy: JSON results, model manifests, and
scheduler logs. Model weights, prepared data, and raw per-example benchmark
records stay on PROJECT.

### 6.1 On PERUN: build the analysis archive

**Safe to re-run;** it refreshes the archive.

```bash
EXPERIMENT=swiglu-8   # or quantization-1, swiglu-9, or stacking-1
STAGE="$PERUN_PROJECT/transfer/$EXPERIMENT"
mkdir -p "$STAGE/logs/perun-job-logs"
rsync -a --prune-empty-dirs \
  --exclude='*.perun-lock/' --exclude='prepared/' --exclude='benchmarks/' --exclude='tokenizer*' \
  --include='*/' --include='*.json' --include='*.md' --exclude='*' \
  "$PERUN_RESULTS/$EXPERIMENT/" "$STAGE/"
rsync -a "$PERUN_LOG_DIR"/"$EXPERIMENT"_* "$STAGE/logs/perun-job-logs/"
tar -czf "$PERUN_PROJECT/transfer/$EXPERIMENT-analysis.tar.gz" -C "$PERUN_PROJECT/transfer" "$EXPERIMENT"
ls -lh "$PERUN_PROJECT/transfer/$EXPERIMENT-analysis.tar.gz"
```

### 6.2 On your PC: download and unpack

**Safe to re-run.** Run in Git Bash from the repository root, with the VPN
connected. The values come from `local/perun/runs/session.md`.

```bash
PERUN_LOGIN="<login>"
PERUN_ACCOUNT="<project-account>"
EXPERIMENT=swiglu-8   # or quantization-1, swiglu-9, or stacking-1
scp "$PERUN_LOGIN@login01.perun.tuke.sk:/mnt/project/$PERUN_ACCOUNT/transfer/$EXPERIMENT-analysis.tar.gz" local/perun/
mkdir -p data/results/workflows/model
tar -xzf "local/perun/$EXPERIMENT-analysis.tar.gz" -C data/results/workflows/model
ls data/results/workflows/model/"$EXPERIMENT"
```

## 7. After every job

**Safe to re-run.** Collect the facts for the job log, then paste the output to
the assistant, which writes the [perun-log.md](perun-log.md) entry and updates
[perun-status.md](perun-status.md).

```bash
JOB="replace-with-job-id"   # from squeue or local/perun/runs/<experiment>.md
RUN=swiglu-8/block-check-001   # the run this job produced
scontrol show job "$JOB" | grep -E 'JobId|JobName|JobState|ExitCode|RunTime|TimeLimit|SubmitTime|StartTime|EndTime|NodeList|AllocTRES'
du -sh "$PERUN_RESULTS/$RUN"
perunfsusage
```

## 8. Troubleshooting

**Safe to re-run.**

```bash
# Who am I, and which project groups do I belong to?
id; groups

# Which values does this shell use? No secrets are printed.
for NAME in PERUN_ACCOUNT PERUN_QOS PERUN_BRANCH PERUN_PROJECT PERUN_SCRATCH PERUN_LOG_DIR HF_HOME MLP_REPLACEMENT_PYTHON; do
    printf '%-24s %s\n' "$NAME" "${!NAME:-<unset>}"
done

# Your account and QoS, as the PERUN wiki shows them, then their limits and the partitions:
sacctmgr show user "$USER" withassoc format=account,qos
scontrol show assoc_mgr users="$USER" accounts="$PERUN_ACCOUNT" flags=assoc
scontrol show assoc_mgr qos="$PERUN_QOS" flags=qos
sinfo -s
```

If `/mnt/scratch/$USER` is missing or not writable, contact PERUN support at
`hpc@helpdesk.tuke.sk`; do not create it yourself. If a job cannot download a
model or dataset on its compute node, download it on the login node with the
same `HF_HOME` first.

## 9. SwiGLU-9

Design, acceptance checks, and resource plan:
[swiglu-9.md](../../experiments/model/swiglu/swiglu-9.md). Every submission
needs the researcher's go.

Three submissions, independent of each other, sent back to back in one block
(9.2). Send them while at most one or two of your jobs are running: with four
running, PERUN refused every new `sbatch` (perun-status.md).

| Step | Block | Requested per task | Expected per task | Before using the results |
| --- | --- | --- | --- | --- |
| Pre-submission checks | 9.1 | - | minutes | Every line OK |
| **Submission 1:** 1B training, 5 tasks | 9.2 | 14 h, 64 GB | 9.5-11 h | C1 (and B1 for the twin) pass at token 0 |
| **Submission 2:** block check, 3 tasks | 9.2 | 6 h, 128 GB | 0.8-2.7 h | D1, D2, P5, A1, A2-A3 pass; merge (9.4) |
| **Submission 3:** reference KL, 1 job | 9.2 | 1 h, 64 GB | 0.1-0.3 h | P5 passes |
| Part B summary | 9.5 | login node | minutes | Review the flagged B2/C2 values |

Training task map: 0 fitted 0.5 (the noise twin), 1 fitted 0.8, 2 slice-only
0.2, 3 slice-only 0.5, 4 slice-only 0.8. Block-check task map: 0, 1, 2 are
layers 3, 11, 19.

Run 1.2, 1.3, and 1.4 first.

### 9.1 Pre-submission checks

**Safe to re-run.** Every line should print OK or a value, never MISSING,
MISMATCH, or DIFFERS.

```bash
# Every pinned SwiGLU-7 and SwiGLU-8 input exists and matches the configuration.
python - <<'EOF'
import hashlib, json, os
sources = json.load(open("workflows/configs/model/swiglu/swiglu-9.json"))["sources"]
expected = {
    "swiglu-7/prepare-001/result.json": sources["swiglu7_prepare_sha256"],
    "swiglu-8/prepare-001/result.json": sources["swiglu8_prepare_sha256"],
    "swiglu-8/block-check-001/result.json": sources["swiglu8_block_check_sha256"],
}
expected.update({f"swiglu-7/S7-0-target-{target}-run-001/result.json": digest
                 for target, digest in sources["swiglu7_runs_sha256"].items()})
expected.update({f"swiglu-8/{key.replace(':', '-target-')}-run-001/result.json": digest
                 for key, digest in sources["swiglu8_runs_sha256"].items()})
for relative, digest in expected.items():
    path = os.path.join(os.environ["PERUN_RESULTS"], relative)
    if not os.path.isfile(path):
        print("MISSING ", relative)
        continue
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    print("OK      " if sha.hexdigest() == digest else "MISMATCH", relative)
EOF

# The large inputs the jobs stage; the stream must be 4000000000 bytes.
stat -c '%s %n' "$PERUN_RESULTS/swiglu-7/prepare-001/prepared/recovery-tokens.int32"
ls "$PERUN_RESULTS"/swiglu-7/S7-0-target-0.{2,5}-run-001/model/bundle.json
ls "$PERUN_RESULTS"/swiglu-8/prepare-001/prepared/starts/{fitted-target-0.5,fitted-target-0.8,slice-target-0.2,slice-target-0.5,slice-target-0.8}.pt
ls "$PERUN_RESULTS/swiglu-8/prepare-001/prepared/kl-tokens.int32"

# The environment matches the one SwiGLU-8's runs recorded (the twin compares across it).
PYTHONPATH=src python - <<'EOF'
import json, os
from mlp_replacement.runlog import environment_record
recorded = json.load(open(os.path.join(os.environ["PERUN_RESULTS"], "swiglu-8/fitted-target-0.5-run-001/result.json")))["environment"]
current = environment_record()
print("python  ", "OK" if recorded["python"] == current["python"] else f"DIFFERS {recorded['python']} -> {current['python']}")
for name, version in sorted(recorded["packages"].items()):
    now = current["packages"].get(name)
    print(f"{name:12}", "OK" if now == version else f"DIFFERS {version} -> {now}")
EOF

# Storage, launcher syntax, and whether Slurm would accept the jobs (nothing is submitted).
perunfsusage    # PROJECT needs about 12 GB; SCRATCH up to 25 GB per training task
bash -n workflows/jobs/perun/swiglu_9.sbatch && echo "launcher syntax OK"
sbatch --test-only --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=14:00:00 --mem=64G --array=0-4 \
  workflows/jobs/perun/swiglu_9.sbatch train
sbatch --test-only --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=06:00:00 --mem=128G --array=0-2 \
  workflows/jobs/perun/swiglu_9.sbatch block-check
sbatch --test-only --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=01:00:00 --mem=64G \
  workflows/jobs/perun/swiglu_9.sbatch reference-kl
ls "$PERUN_RESULTS/swiglu-9" 2>/dev/null || echo "no SwiGLU-9 outputs yet"
squeue -u "$USER"   # submit 9.2 only while at most one or two of your jobs run
```

### 9.2 Submit the three jobs

**Submits jobs. Run once.** It writes the three job IDs to
`$PERUN_LOG_DIR/swiglu-9_submission.txt`, which 9.3 reads. If one of the three
IDs prints empty, that `sbatch` was refused: submit only that line again once
fewer jobs run.

```bash
ROOT="$PERUN_RESULTS/swiglu-9"
EXISTING=$(find "$ROOT" -mindepth 1 -maxdepth 1 2>/dev/null)
if [[ -n "$EXISTING" ]] || squeue -u "$USER" -h -n swiglu-9 | grep -q .; then
    echo "STOP: SwiGLU-9 outputs exist or a swiglu-9 job is queued:"; echo "$EXISTING"
else
    TRAIN=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=14:00:00 --mem=64G \
        --array=0-4 --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
        workflows/jobs/perun/swiglu_9.sbatch train)
    BLOCK=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=06:00:00 --mem=128G \
        --array=0-2 --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
        workflows/jobs/perun/swiglu_9.sbatch block-check)
    REFERENCE=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=01:00:00 --mem=64G \
        --output="$PERUN_LOG_DIR/%x_%j.out" --error="$PERUN_LOG_DIR/%x_%j.err" \
        workflows/jobs/perun/swiglu_9.sbatch reference-kl)
    printf 'TRAIN=%s\nBLOCK=%s\nREFERENCE=%s\n' "${TRAIN%%;*}" "${BLOCK%%;*}" "${REFERENCE%%;*}" \
        | tee "$PERUN_LOG_DIR/swiglu-9_submission.txt"
    echo "<- record these in local/perun/runs/swiglu-9.md"
fi
```

### 9.3 Watch the jobs

**Safe to re-run.** Each training task checks token 0 against SwiGLU-8 (C1),
and the twin also against SwiGLU-7 (B1); a mismatch stops it with
`AcceptanceFailure` in its `.err` file. Each block-check task stops the same
way on D1, D2, P5, A1, or A2-A3. Stop and investigate before trusting any
result.

```bash
source "$PERUN_LOG_DIR/swiglu-9_submission.txt"
squeue -u "$USER" -n swiglu-9
RUNS=("fitted 0.5 (twin)" "fitted 0.8" "slice 0.2" "slice 0.5" "slice 0.8")
for TASK in 0 1 2 3 4; do
    LAST=$(grep '^arm=' "$PERUN_LOG_DIR/swiglu-9_${TRAIN}_${TASK}.out" 2>/dev/null | tail -n 1)
    FAILED=$(grep -l -E 'AcceptanceFailure|Traceback' "$PERUN_LOG_DIR/swiglu-9_${TRAIN}_${TASK}.err" 2>/dev/null)
    echo "train $TASK, ${RUNS[$TASK]}: ${LAST:-not started}${FAILED:+   <- check its .err}"
done
LAYERS=(3 11 19)
for TASK in 0 1 2; do
    DONE=$(grep -c '^setting=' "$PERUN_LOG_DIR/swiglu-9_${BLOCK}_${TASK}.out" 2>/dev/null)
    FAILED=$(grep -l -E 'AcceptanceFailure|Traceback' "$PERUN_LOG_DIR/swiglu-9_${BLOCK}_${TASK}.err" 2>/dev/null)
    echo "block check, layer ${LAYERS[$TASK]}: ${DONE:-0} of 24 fits${FAILED:+   <- check its .err}"
done
grep 'S7-0 target=' "$PERUN_LOG_DIR/swiglu-9_${REFERENCE}.out" 2>/dev/null || echo "reference KL: not finished"
gpu-status
```

### 9.4 Merge the block check

**Safe to re-run.** When the three block-check tasks have finished, merge
them on the login node; it needs no GPU and reads only JSON. It checks that
the tasks used identical data, joins the 72 fits, applies the adoption rule
against SwiGLU-8's three seeds, and writes `swiglu-9/block-check-001`.
Running it again changes nothing.

```bash
PYTHONPATH=src python -m workflows.runs.model.swiglu.swiglu_9 merge-block-check \
  --results-root "$PERUN_RESULTS/swiglu-9" --swiglu8-root "$PERUN_RESULTS/swiglu-8"
```

An adopted setting applies only to later experiments, never to Part B.

### 9.5 Summarize Part B

**Safe to re-run.** When all five training runs and the reference KL are on
PROJECT, apply the decided verdict on the login node. It prints the twin
noise, then g and the verdict per target, and writes
`swiglu-9/part-b-summary-001`.

```bash
PYTHONPATH=src python -m workflows.runs.model.swiglu.swiglu_9 summarize-part-b \
  --results-root "$PERUN_RESULTS/swiglu-9" --swiglu7-root "$PERUN_RESULTS/swiglu-7"
```

Its `flagged_for_review` list holds every B2 and C2 value above 1%. Review
them before any Part B result is used, as in SwiGLU-8. Then copy the results
home (section 6, `EXPERIMENT=swiglu-9`).

### 9.6 Resume a stopped task

**Submits a job. Run once, after the cause is understood.** Resume continues
the same run from its last saved point. It refuses changed code,
configuration, or environment, and needs the run's `result.json` on PROJECT
with no lock (see 5.2).

```bash
# One training task; replace 3 with the task number (0-4):
sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=14:00:00 --mem=64G --array=3 \
  --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
  workflows/jobs/perun/swiglu_9.sbatch train --resume

# One block-check task; replace 1 with the task number (0-2):
sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=06:00:00 --mem=128G --array=1 \
  --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
  workflows/jobs/perun/swiglu_9.sbatch block-check --resume

# The reference KL:
sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=01:00:00 --mem=64G \
  --output="$PERUN_LOG_DIR/%x_%j.out" --error="$PERUN_LOG_DIR/%x_%j.err" \
  workflows/jobs/perun/swiglu_9.sbatch reference-kl --resume
```

## 10. Stacking-1

Design, checks, and resource plan:
[stacking-1.md](../../experiments/model/stacking/stacking-1.md). Every
submission needs the researcher's go.

Stacking-1 runs from the usual checkout, `diploma-thesis-block-replacement`
(decision log, 2026-10-20). All 48 tasks were submitted on 2026-10-20 and
finished on 2026-10-21 (10.1-10.2 are kept as the record of that submission).
The three order-B INT3 outputs (`-001`) are invalid and are rerun as `-002`
with the fixed INT3 quantizer (10.5). The same rule applies to the reruns:
once they are submitted, do not update the checkout until every rerun has
*finished*, because a job copies the code when it starts and a resume
refuses changed code.

| Step | Block | Requested per task | Expected per task | Before using the results |
| --- | --- | --- | --- | --- |
| Update and pre-submission checks (done 2026-10-20) | 10.1 | - | minutes | Every line OK; your queue is empty |
| Submit all 48 tasks: order B (6), order A at 100M (33), order A at 1B (9) (done 2026-10-20) | 10.2 | 6 h (B) or 2 h (A), 128 GB | 1.5-3 h (B), 0.3-0.5 h (A) | K1, K2, K4, K5 (order A); K3, K4, K5 (order B) |
| Rerun the three order-B INT3 tasks as `-002` (stage 1, with 11.5) | 10.5 | 3 h, 128 GB | 1.2-1.6 h | K3, K5, K6 (K4 covers INT8 and INT4 only) |
| Order B at 1B: six tasks in one array (`b1b`, `-001`) | 10.7 | 18 h, 128 GB | 10-12.5 h | K3, K4 (INT4), K5, K6, K7 (`K7-start` and `K7-review` for INT3 only) |

At the original submission, PERUN accepted at most 48 submitted tasks,
counting each array task, and refused every new `sbatch` while 4 of your jobs
ran. Stacking-1 is exactly 48 tasks, so all three arrays were submitted back
to back while nothing of yours was queued, with a 10-minute delayed start, as
for SwiGLU-9. Both limits have since changed (perun-status.md, 2026-10-25):
50 queued tasks, 20 running jobs, and no "4 running jobs" submit refusal.

Task maps come from the runner, not the job script:
`python -m workflows.runs.model.stacking.stacking_1 list-variants --group b`
(or `a100m`, `a1b`, `b1b`) prints the variants in array order.

Run 1.2, 1.3, and 1.4 first.

### 10.1 Pre-submission checks

**Safe to re-run.** Every line should print OK or a value, never MISSING or
MISMATCH. The group counts must be 6, 33, and 9, every `--test-only` line must
be accepted, and `squeue` must list no job.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
# Every pinned SwiGLU-7, -8, and -9 input exists and matches the configuration.
python - <<'EOF'
import hashlib, json, os
config = json.load(open("workflows/configs/model/stacking/stacking-1.json"))
root = os.environ["PERUN_RESULTS"]
def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
expected = {"swiglu-8/prepare-001/result.json": config["sources"]["swiglu8_prepare_sha256"]}
for name, model in config["sources"]["models"].items():
    run = f"{model['workflow']}/{model['run']}"
    expected[f"{run}/result.json"] = model["result_sha256"]
    expected[f"{run}/model/bundle.json"] = model["bundle_sha256"]
for relative, digest in expected.items():
    path = os.path.join(root, relative)
    print("MISSING " if not os.path.isfile(path) else "OK      " if sha(path) == digest else "MISMATCH", relative)
EOF

# The large inputs order B stages; the stream must be 400000000 bytes.
stat -c '%s %n' "$PERUN_RESULTS/swiglu-8/prepare-001/prepared/recovery-tokens.int32"
ls "$PERUN_RESULTS"/swiglu-8/prepare-001/prepared/starts/fitted-target-0.{2,5,8}.pt
ls "$PERUN_RESULTS/swiglu-8/prepare-001/prepared/kl-tokens.int32" \
   "$PERUN_RESULTS/swiglu-8/prepare-001/prepared/local-evaluation-batches.pt"

# Software, the variant lists (6, 33, 9), launcher syntax, and whether Slurm would accept the jobs.
pip show torch torchao mslk 2>/dev/null | grep -E '^(Name|Version)'
for GROUP in b a100m a1b; do
    echo "$GROUP: $(PYTHONPATH=src python -m workflows.runs.model.stacking.stacking_1 list-variants --group $GROUP | wc -l)"
done
bash -n workflows/jobs/perun/stacking_1.sbatch && echo "launcher syntax OK"
sbatch --test-only --begin=now+10minutes --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=06:00:00 \
  --array=0-5 workflows/jobs/perun/stacking_1.sbatch group b
sbatch --test-only --begin=now+10minutes --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=02:00:00 \
  --array=0-32 workflows/jobs/perun/stacking_1.sbatch group a100m
sbatch --test-only --begin=now+10minutes --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=02:00:00 \
  --array=0-8 workflows/jobs/perun/stacking_1.sbatch group a1b
perunfsusage    # PROJECT needs about 70-100 GB
ls "$PERUN_RESULTS/stacking-1" 2>/dev/null || echo "no Stacking-1 outputs yet"
git log -1 --oneline
squeue -u "$USER"   # must list no job: 48 tasks are the whole submission limit
```

### 10.2 Submit all 48 tasks

*Original submission, done on 2026-10-20. Its STOP guard now always fires
because Stacking-1 outputs exist; reruns use 10.5.*

**Submits jobs. Run once,** after 10.1 is clean. It writes the three job IDs
to `$PERUN_LOG_DIR/stacking-1_submission.txt`. If an ID prints empty, that
`sbatch` was refused: tell the assistant before submitting it again.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
EXISTING=$(find "$PERUN_RESULTS/stacking-1" -mindepth 1 -maxdepth 1 2>/dev/null)
if [[ -n "$EXISTING" ]] || squeue -u "$USER" -h | grep -q .; then
    echo "STOP: Stacking-1 outputs exist or a job of yours is queued:"; echo "$EXISTING"
else
    B=$(sbatch --parsable --begin=now+10minutes --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
        --time=06:00:00 --array=0-5 \
        --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
        workflows/jobs/perun/stacking_1.sbatch group b)
    A100M=$(sbatch --parsable --begin=now+10minutes --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
        --time=02:00:00 --array=0-32 \
        --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
        workflows/jobs/perun/stacking_1.sbatch group a100m)
    A1B=$(sbatch --parsable --begin=now+10minutes --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
        --time=02:00:00 --array=0-8 \
        --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
        workflows/jobs/perun/stacking_1.sbatch group a1b)
    printf 'B=%s\nA100M=%s\nA1B=%s\n' "${B%%;*}" "${A100M%%;*}" "${A1B%%;*}" \
        | tee "$PERUN_LOG_DIR/stacking-1_submission.txt"
    echo "<- record these in local/perun/runs/stacking-1.md"
fi
```

Order-B task map: 0 fitted 0.2 INT4, 1 fitted 0.2 INT3, 2 fitted 0.5 INT4,
3 fitted 0.5 INT3, 4 fitted 0.8 INT4, 5 fitted 0.8 INT3. The order-A maps run
source by source, each with INT8, INT4, and INT3: `a100m` tasks 0-23 are the
fitted 20, 30, 40, 50, 60, 70, 75, and 80% models and 24-32 the slice-only
20, 50, and 80% models; `a1b` tasks 0-8 are the fitted 20, 50, and 80% models.

### 10.3 Watch the jobs

**Safe to re-run.** Each task prints its variant at the start. A failed check
(K1-K5) stops the task with `AcceptanceFailure` in its `.err` file; stop and
investigate before trusting any result.

```bash
squeue -u "$USER" -n stacking-1 -o "%.18i %.10T %.10M %.11l %R"
for FILE in "$PERUN_LOG_DIR"/stacking-1_*.out; do
    VARIANT=$(grep -m1 -o 'variant=[^ ]*' "$FILE")
    LAST=$(grep '^variant=.*tokens=' "$FILE" | tail -n 1)
    DONE=$(grep -l 'Verified persistent output' "$FILE")
    FAILED=$(grep -l -E 'AcceptanceFailure|Traceback' "${FILE%.out}.err" 2>/dev/null)
    echo "$(basename "$FILE"): ${VARIANT:-not started}${LAST:+ | $LAST}${DONE:+ | done}${FAILED:+   <- check its .err}"
done
echo "done: $(grep -l 'Verified persistent output' "$PERUN_LOG_DIR"/stacking-1_*.out 2>/dev/null | wc -l) of 48"
gpu-status
```

### 10.4 Resume a stopped task

**Submits a job. Run once, after the cause is understood.** A finished bundle
is reused; an order-B task continues from its last 25M-token checkpoint.
Resume refuses changed code, configuration, or environment, and needs the
run's `result.json` on PROJECT with no lock (see 5.2).

The `-001` tasks have all finished and cannot be resumed after the INT3 fix
(their code fingerprint changed). For an order-B INT3 rerun task, keep
`--attempt 002` before `--resume`:

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
# One order-B INT3 rerun task; replace the index (1, 3, or 5):
sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=03:00:00 --array=3 \
  --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
  workflows/jobs/perun/stacking_1.sbatch group b --attempt 002 --resume
```

Right after the last task finishes, run section 7 for the job IDs
(scheduler records expire), then copy the results home (section 6,
`EXPERIMENT=stacking-1`).

### 10.5 Rerun the order-B INT3 tasks as `-002`

**Stage 1 of the INT3 QAT reruns,** submitted in the same sitting as 11.5,
after the researcher's go. The fixed INT3 quantizer and its checks are in
[stacking-1.md](../../experiments/model/stacking/stacking-1.md#int3-qat-fix-2026-10-25).
Run 1.2, 1.3, and 1.4 first: the checkout must contain the fix commit.

**Pre-submission checks. Safe to re-run.** The `git log` line must show the
fix commit. The `list-variants` lines must print
`b-fitted-0.2-100m-int3`, `b-fitted-0.5-100m-int3`, and
`b-fitted-0.8-100m-int3`, in that order. The `--test-only` line must be
accepted. No `-002` output may exist yet, and `squeue` should list no job.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
git log -1 --oneline
# Array tasks 1, 3, and 5 of group b are the three INT3 variants:
PYTHONPATH=src python -m workflows.runs.model.stacking.stacking_1 list-variants --group b | sed -n '2p;4p;6p'
bash -n workflows/jobs/perun/stacking_1.sbatch && echo "launcher syntax OK"
find "$PERUN_RESULTS/stacking-1" -maxdepth 1 -name 'b-fitted-*-100m-int3-002*' | grep . || echo "no -002 outputs yet"
sbatch --test-only --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=03:00:00 --array=1,3,5 \
  workflows/jobs/perun/stacking_1.sbatch group b --attempt 002
squeue -u "$USER"
```

**Submits jobs. Run once,** after the checks are clean. It writes the job ID
to `$PERUN_LOG_DIR/stacking-1-int3-rerun_submission.txt`. If the ID prints
empty, `sbatch` was refused: tell the assistant before submitting again.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
EXISTING=$(find "$PERUN_RESULTS/stacking-1" -maxdepth 1 -name 'b-fitted-*-100m-int3-002*')
if [[ -n "$EXISTING" ]]; then
    echo "STOP: an order-B INT3 -002 output or lock already exists:"; echo "$EXISTING"
else
    BINT3=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=03:00:00 --array=1,3,5 \
        --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
        workflows/jobs/perun/stacking_1.sbatch group b --attempt 002)
    printf 'BINT3=%s\n' "${BINT3%%;*}" | tee "$PERUN_LOG_DIR/stacking-1-int3-rerun_submission.txt"
    echo "<- record this in local/perun/runs/stacking-1.md"
fi
```

**Watch** with 10.3. If training is not shown the deployed INT3 weights, the
task stops right after QAT preparation with `AcceptanceFailure` and `K6-start`
in its `.err` file. A mismatch at conversion stops it with "Deployed INT3
weights differ" before export. Either way, stop and tell the assistant.

### 10.6 Rerun the order-B INT4 tasks as `-002`

**After 4.6's start check passed.** The INT4 fix is in
[quantization-1.md](../../experiments/model/baseline/quantization-1.md#int4-qat-conversion-gap-2026-11-01).
Same checkout as 4.6.

**Pre-submission checks. Safe to re-run.** The `list-variants` lines must
print `b-fitted-0.2-100m-int4`, `b-fitted-0.5-100m-int4`, and
`b-fitted-0.8-100m-int4`, in that order. The `--test-only` line must be
accepted, and no `-002` INT4 output may exist yet.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
git log -1 --oneline
# Array tasks 0, 2, and 4 of group b are the three INT4 variants:
PYTHONPATH=src python -m workflows.runs.model.stacking.stacking_1 list-variants --group b | sed -n '1p;3p;5p'
find "$PERUN_RESULTS/stacking-1" -maxdepth 1 -name 'b-fitted-*-100m-int4-002*' | grep . || echo "no -002 outputs yet"
sbatch --test-only --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=03:00:00 --array=0,2,4 \
  workflows/jobs/perun/stacking_1.sbatch group b --attempt 002
```

**Submits jobs. Run once,** after the checks are clean. It writes the job ID
to `$PERUN_LOG_DIR/stacking-1-int4-rerun_submission.txt`.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
EXISTING=$(find "$PERUN_RESULTS/stacking-1" -maxdepth 1 -name 'b-fitted-*-100m-int4-002*')
if [[ -n "$EXISTING" ]]; then
    echo "STOP: an order-B INT4 -002 output or lock already exists:"; echo "$EXISTING"
else
    BINT4=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=03:00:00 --array=0,2,4 \
        --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
        workflows/jobs/perun/stacking_1.sbatch group b --attempt 002)
    printf 'BINT4=%s\n' "${BINT4%%;*}" | tee "$PERUN_LOG_DIR/stacking-1-int4-rerun_submission.txt"
    echo "<- record this in local/perun/runs/stacking-1.md"
fi
```

**Watch** with 10.3. A start mismatch stops a task right after QAT
preparation with `AcceptanceFailure` and `K6-start` in its `.err` file; a
mismatch at conversion stops it with "Deployed INT4 weights differ". Either
way, stop and tell the assistant.

### 10.7 Order B at 1B (`b1b`)

Design, checks K7, and resource plan:
[stacking-1.md](../../experiments/model/stacking/stacking-1.md#order-b-at-1b-plan-2026-11-01).
Six tasks in one array, all running at once, about 10-12.5 h each with an
18 h limit. Submit only after the researcher's go. 4.6 and 10.6 run from
`b59da5c`, which does not contain the 1B code. Update the checkout (1.4) only
after their jobs have *finished*, because a resume refuses changed code. The
check below confirms that the updated checkout still contains the INT4 fix.
Do not update the checkout again until all six tasks have *finished*.

Task map: 0 fitted 0.2 INT4, 1 fitted 0.2 INT3, 2 fitted 0.5 INT4, 3 fitted
0.5 INT3, 4 fitted 0.8 INT4, 5 fitted 0.8 INT3.

**Pre-submission checks. Safe to re-run.** Run 1.2, 1.3, and 1.4 first. Every
hash line must print OK (the 4 GB stream takes about a minute). The stream
size must be 4000000000. `list-variants` must print the six variants in
task-map order. The `--test-only` line must be accepted, and no 1B output
may exist yet.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
git log -1 --oneline
git merge-base --is-ancestor b59da5c HEAD && echo "INT4 fix included" || echo "MISSING the INT4 fix commit"
# Every pinned 1B input exists and matches the configuration.
python - <<'EOF'
import hashlib, json, os
config = json.load(open("workflows/configs/model/stacking/stacking-1.json"))
sources = config["sources"]
root = os.environ["PERUN_RESULTS"]
def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
expected = {"swiglu-8/prepare-001/result.json": sources["swiglu8_prepare_sha256"],
            "swiglu-7/prepare-001/result.json": sources["swiglu7_prepare_sha256"]}
for target in ("0.2", "0.5", "0.8"):
    model = sources["models"][f"fitted-{target}-100m"]
    expected[f"{model['workflow']}/{model['run']}/result.json"] = model["result_sha256"]
for twin in sources["b1b_twins"].values():
    expected[f"stacking-1/{twin['run']}/result.json"] = twin["result_sha256"]
preparation = os.path.join(root, "swiglu-7/prepare-001/result.json")
if os.path.isfile(preparation):
    stream = json.load(open(preparation))["results"]["stream"]
    expected[f"swiglu-7/prepare-001/{stream['path']}"] = stream["sha256"]
for relative, digest in expected.items():
    path = os.path.join(root, relative)
    print("MISSING " if not os.path.isfile(path) else "OK      " if sha(path) == digest else "MISMATCH", relative)
EOF
stat -c '%s %n' "$PERUN_RESULTS/swiglu-7/prepare-001/prepared/recovery-tokens.int32"
ls "$PERUN_RESULTS"/swiglu-8/prepare-001/prepared/starts/fitted-target-0.{2,5,8}.pt
pip show torch torchao mslk 2>/dev/null | grep -E '^(Name|Version)'
PYTHONPATH=src python -m workflows.runs.model.stacking.stacking_1 list-variants --group b1b
bash -n workflows/jobs/perun/stacking_1.sbatch && echo "launcher syntax OK"
find "$PERUN_RESULTS/stacking-1" -maxdepth 1 -name 'b-fitted-*-1b-*' | grep . || echo "no 1B outputs yet"
sbatch --test-only --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=18:00:00 --array=0-5 \
  workflows/jobs/perun/stacking_1.sbatch group b1b
squeue -u "$USER"
perunfsusage    # PROJECT needs about 10 GB; SCRATCH about 20 GB per task
```

**Submits jobs. Run once,** after the checks are clean. It writes the job ID
to `$PERUN_LOG_DIR/stacking-1-b1b_submission.txt`. If the ID prints empty,
`sbatch` was refused: tell the assistant before submitting again.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
EXISTING=$(find "$PERUN_RESULTS/stacking-1" -maxdepth 1 -name 'b-fitted-*-1b-*')
if [[ -n "$EXISTING" ]]; then
    echo "STOP: an order-B 1B output or lock already exists:"; echo "$EXISTING"
else
    B1B=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=18:00:00 --array=0-5 \
        --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
        workflows/jobs/perun/stacking_1.sbatch group b1b)
    printf 'B1B=%s\n' "${B1B%%;*}" | tee "$PERUN_LOG_DIR/stacking-1-b1b_submission.txt"
    echo "<- record this in local/perun/runs/stacking-1.md"
fi
```

**Watch. Safe to re-run.** A failed stopping check (K3-K7) ends a task with
`AcceptanceFailure` in its `.err` file: stop and tell the assistant.
`K7-start` runs in the first minutes, before the first update. `K7-review`
never stops a task. A flagged value (more than 1% from the 100M twin, at
10-100M) must be investigated before any result is used; tell the assistant.

```bash
B1B=$(sed -n 's/^B1B=//p' "$PERUN_LOG_DIR/stacking-1-b1b_submission.txt")
squeue -j "$B1B" -o "%.18i %.10T %.10M %.11l %R"
for FILE in "$PERUN_LOG_DIR"/stacking-1_"$B1B"_*.out; do
    LAST=$(grep '^variant=.*tokens=' "$FILE" | tail -n 1)
    DONE=$(grep -l 'Verified persistent output' "$FILE")
    FAILED=$(grep -l -E 'AcceptanceFailure|Traceback' "${FILE%.out}.err" 2>/dev/null)
    echo "$(basename "$FILE"): ${LAST:-starting}${DONE:+ | done}${FAILED:+   <- check its .err}"
done
# Checks of the running tasks (their result.json is on SCRATCH until the task ends).
python - <<'EOF'
import json, os
from pathlib import Path
for result in sorted(Path(os.environ["PERUN_SCRATCH"]).glob("job_*/results/b-fitted-*-1b-*/result.json")):
    acceptance = json.loads(result.read_text()).get("results", {}).get("acceptance", {})
    flags = [f"{row['tokens']:,} {row['metric']} {row['relative_difference']:.2%}"
             for row in acceptance.get("K7-review", {}).get("checks", []) if not row["passed"]]
    print(result.parent.name, {group: summary["passed"] for group, summary in acceptance.items()},
          "| K7-review flags:", flags or "none")
EOF
gpu-status
```

**Resume a stopped task** only after the cause is understood, as in 10.4. A
resume continues from the last 25M-token checkpoint and refuses changed code.
Replace the index:

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=18:00:00 --array=3 \
  --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
  workflows/jobs/perun/stacking_1.sbatch group b1b --resume
```

Right after the last task finishes, run section 7 for the job ID (scheduler
records expire), then copy the results home (section 6,
`EXPERIMENT=stacking-1`).

## 11. Quantization-2

Design, checks, and resource plan:
[quantization-2.md](../../experiments/model/baseline/quantization-2.md).
Every submission needs the researcher's go.

Quantization-2 runs on Quantization-1's launcher with its own configuration
and writes to `perun-results/quantization-2/`. It reuses Quantization-1's
`prepare-001`, so it has no preparation step. The three original runs were
submitted on 2026-10-21 (jobs 123646-123648) after Stacking-1 had finished;
11.1-11.2 are kept as the record of that submission. Both INT3 QAT runs are
invalid. The 100M run was rerun as `-002` with the fixed INT3 quantizer
(11.5); the 1B rerun was skipped on 2026-10-26 and reopened on 2026-11-03
(11.6).

| Run | Output | Expected | Time limit | Status |
| --- | --- | --- | --- | --- |
| INT3 PTQ | `int3-001` | 0.3-0.5 h | 2 h | Valid |
| INT3 QAT 100M | `qat-int3-100m-001` | 1.6-3.5 h | 6 h | Invalid (INT3 QAT incident) |
| INT3 QAT 1B | `qat-int3-1b-001` | 16-32 h | 36 h | Invalid (INT3 QAT incident) |
| INT3 QAT 100M, rerun (stage 1, 11.5) | `qat-int3-100m-002` | 1.7-2.2 h | 4 h | Completed 2026-10-26 (job 132434) |
| INT3 QAT 1B, rerun (stage 2, 11.6) | `qat-int3-1b-002` | 14-17 h | 24 h | Reopened (decision log, 2026-11-03) |

Run 1.2, 1.3, and 1.4 first.

### 11.1 Pre-submission checks

*Original submission, done on 2026-10-21. "No Quantization-2 output may exist
yet" applied to that submission only; reruns use 11.5 and 11.6.*

**Safe to re-run.** The settings line must print `quantization-2 ['ptq-int3']`,
every `--test-only` line must be accepted, and no Quantization-2 output may
exist yet.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
CONFIG=workflows/configs/model/baseline/quantization-2.json
Q1_PREPARED="$PERUN_RESULTS/quantization-1/prepare-001/result.json"
PYTHONPATH=src python -c "from workflows.runs.model.baseline import quantization as q; s = q.load_settings('$CONFIG'); print(s['workflow'], s['variants'])"
ls -l "$Q1_PREPARED" "$PERUN_RESULTS"/quantization-1/prepare-001/prepared/
bash -n workflows/jobs/perun/quantization_1.sbatch && echo "launcher syntax OK"
sbatch --test-only --job-name=quantization-2 --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=02:00:00 \
  workflows/jobs/perun/quantization_1.sbatch run int3-001 --config "$CONFIG" --prepared "$Q1_PREPARED" --variant ptq-int3
sbatch --test-only --job-name=quantization-2 --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=36:00:00 \
  workflows/jobs/perun/quantization_1.sbatch run qat-int3-1b-001 --config "$CONFIG" --prepared "$Q1_PREPARED" \
  --variant qat-int3 --qat-tokens 1000000000
perunfsusage    # PROJECT needs about 7 GB
ls "$PERUN_RESULTS/quantization-2" 2>/dev/null || echo "no Quantization-2 outputs yet"
git log -1 --oneline
squeue -u "$USER"   # Stacking-1 must have finished
```

### 11.2 Submit the three runs

*Original submission, done on 2026-10-21 (jobs 123646-123648). Its STOP guard
now always fires because Quantization-2 outputs exist.*

**Submits jobs. Run once,** after 11.1 is clean. It writes the three job IDs
to `$PERUN_LOG_DIR/quantization-2_submission.txt`.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
CONFIG=workflows/configs/model/baseline/quantization-2.json
Q1_PREPARED="$PERUN_RESULTS/quantization-1/prepare-001/result.json"
COMMON=(--parsable --job-name=quantization-2 --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS"
        --output="$PERUN_LOG_DIR/%x_%j.out" --error="$PERUN_LOG_DIR/%x_%j.err")
EXISTING=$(find "$PERUN_RESULTS/quantization-2" -mindepth 1 -maxdepth 1 2>/dev/null)
if [[ -n "$EXISTING" ]] || squeue -u "$USER" -h -n quantization-2 | grep -q .; then
    echo "STOP: Quantization-2 outputs exist or a quantization-2 job is queued:"; echo "$EXISTING"
else
    PTQ=$(sbatch "${COMMON[@]}" --time=02:00:00 workflows/jobs/perun/quantization_1.sbatch \
        run int3-001 --config "$CONFIG" --prepared "$Q1_PREPARED" --variant ptq-int3)
    QAT100M=$(sbatch "${COMMON[@]}" --time=06:00:00 workflows/jobs/perun/quantization_1.sbatch \
        run qat-int3-100m-001 --config "$CONFIG" --prepared "$Q1_PREPARED" --variant qat-int3 --qat-tokens 100000000)
    QAT1B=$(sbatch "${COMMON[@]}" --time=36:00:00 workflows/jobs/perun/quantization_1.sbatch \
        run qat-int3-1b-001 --config "$CONFIG" --prepared "$Q1_PREPARED" --variant qat-int3 --qat-tokens 1000000000)
    printf 'PTQ=%s\nQAT100M=%s\nQAT1B=%s\n' "${PTQ%%;*}" "${QAT100M%%;*}" "${QAT1B%%;*}" \
        | tee "$PERUN_LOG_DIR/quantization-2_submission.txt"
    echo "<- record these in local/perun/runs/quantization-2.md"
fi
```

### 11.3 Watch the runs

**Safe to re-run.** A stream that differs from Quantization-1's stops the QAT
run with "QAT stream differs" in its `.err` file before training starts.

```bash
source "$PERUN_LOG_DIR/quantization-2_submission.txt"
squeue -u "$USER" -n quantization-2 -o "%.12i %.9T %.10M %.11l %R"
for NAME in PTQ QAT100M QAT1B; do
    JOB="${!NAME}"
    LAST=$(tail -n 1 "$PERUN_LOG_DIR/quantization-2_$JOB.out" 2>/dev/null)
    DONE=$(grep -l 'Verified persistent output' "$PERUN_LOG_DIR/quantization-2_$JOB.out" 2>/dev/null)
    FAILED=$(grep -l 'Traceback' "$PERUN_LOG_DIR/quantization-2_$JOB.err" 2>/dev/null)
    echo "$NAME ($JOB): ${LAST:-not started}${DONE:+ | done}${FAILED:+   <- check its .err}"
done
gpu-status
```

While a run is going, its current stage is in the scratch copy of its
`result.json` (section 2.1).

### 11.4 Resume a stopped run

**Submits a job. Run once, after the cause is understood.** Needs the run's
`result.json` on PROJECT and no lock (see 5.2). The `-001` runs cannot be
resumed after the INT3 fix (their code fingerprint changed). Replace the
output name, token budget, and limit for the rerun being resumed (100M: 4 h;
1B: 24 h).

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
sbatch --parsable --job-name=quantization-2 --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=04:00:00 \
  --output="$PERUN_LOG_DIR/%x_%j.out" --error="$PERUN_LOG_DIR/%x_%j.err" \
  workflows/jobs/perun/quantization_1.sbatch run qat-int3-100m-002 \
  --config workflows/configs/model/baseline/quantization-2.json \
  --prepared "$PERUN_RESULTS/quantization-1/prepare-001/result.json" \
  --variant qat-int3 --qat-tokens 100000000 --resume
```

Right after the last run finishes, run section 7 for its job IDs, then
copy the results home (section 6, `EXPERIMENT=quantization-2`). The
comparison report is built on the PC (see the experiment page).

### 11.5 Rerun INT3 QAT 100M as `-002`

**Stage 1 of the INT3 QAT reruns,** submitted in the same sitting as 10.5,
after the researcher's go. The fixed INT3 quantizer and its checks are in
[quantization-2.md](../../experiments/model/baseline/quantization-2.md#int3-qat-fix-2026-10-25).
Run 1.2, 1.3, and 1.4 first: the checkout must contain the fix commit.

**Pre-submission checks. Safe to re-run.** The `git log` line must show the
fix commit, the settings line must print `quantization-2 ['ptq-int3']`, the
`--test-only` line must be accepted, and `qat-int3-100m-002` must not exist.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
CONFIG=workflows/configs/model/baseline/quantization-2.json
Q1_PREPARED="$PERUN_RESULTS/quantization-1/prepare-001/result.json"
git log -1 --oneline
PYTHONPATH=src python -c "from workflows.runs.model.baseline import quantization as q; s = q.load_settings('$CONFIG'); print(s['workflow'], s['variants'])"
bash -n workflows/jobs/perun/quantization_1.sbatch && echo "launcher syntax OK"
sbatch --test-only --job-name=quantization-2 --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=04:00:00 \
  workflows/jobs/perun/quantization_1.sbatch run qat-int3-100m-002 --config "$CONFIG" --prepared "$Q1_PREPARED" \
  --variant qat-int3 --qat-tokens 100000000
find "$PERUN_RESULTS/quantization-2" -maxdepth 1 -name 'qat-int3-100m-002*' | grep . || echo "no 100M -002 output yet"
squeue -u "$USER"
```

**Submits a job. Run once,** after the checks are clean. It writes the job ID
to `$PERUN_LOG_DIR/quantization-2-int3-rerun_submission.txt`.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
CONFIG=workflows/configs/model/baseline/quantization-2.json
Q1_PREPARED="$PERUN_RESULTS/quantization-1/prepare-001/result.json"
EXISTING=$(find "$PERUN_RESULTS/quantization-2" -maxdepth 1 -name 'qat-int3-100m-002*')
if [[ -n "$EXISTING" ]]; then
    echo "STOP: the 100M -002 output or lock already exists:"; echo "$EXISTING"
else
    QAT100M=$(sbatch --parsable --job-name=quantization-2 --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
        --time=04:00:00 --output="$PERUN_LOG_DIR/%x_%j.out" --error="$PERUN_LOG_DIR/%x_%j.err" \
        workflows/jobs/perun/quantization_1.sbatch run qat-int3-100m-002 --config "$CONFIG" \
        --prepared "$Q1_PREPARED" --variant qat-int3 --qat-tokens 100000000)
    printf 'QAT100M=%s\n' "${QAT100M%%;*}" | tee "$PERUN_LOG_DIR/quantization-2-int3-rerun_submission.txt"
    echo "<- record this in local/perun/runs/quantization-2.md"
fi
```

**Watch** with 11.3, replacing its first line with
`source "$PERUN_LOG_DIR/quantization-2-int3-rerun_submission.txt"` (only
`QAT100M` is set, so PTQ and QAT1B print "not started"). If training is not shown the deployed INT3 weights, the run
stops right after QAT preparation with "INT3 fake-quantized weights differ"
in its `.err` file. A mismatch at conversion stops it with "Deployed INT3
weights differ" before export. Either way, stop and tell the assistant.

### 11.6 Rerun INT3 QAT 1B as `-002` (stage 2)

*Reopened on 2026-11-03 (decision log, 2026-11-03), after being skipped on
2026-10-26. Plan: Quantization-2 "INT3 QAT 1B rerun". Run 1.2, 1.3, and 1.4
first; no code change is needed.*

**Pre-submission checks. Safe to re-run.** The checkout must contain
`b59da5c`, the settings line must print `quantization-2 ['ptq-int3']`, the
`--test-only` line must be accepted, and `qat-int3-1b-002` must not exist.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
CONFIG=workflows/configs/model/baseline/quantization-2.json
Q1_PREPARED="$PERUN_RESULTS/quantization-1/prepare-001/result.json"
git log -1 --oneline
git merge-base --is-ancestor b59da5c HEAD && echo "checkout contains b59da5c"
PYTHONPATH=src python -c "from workflows.runs.model.baseline import quantization as q; s = q.load_settings('$CONFIG'); print(s['workflow'], s['variants'])"
bash -n workflows/jobs/perun/quantization_1.sbatch && echo "launcher syntax OK"
sbatch --test-only --job-name=quantization-2 --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=24:00:00 \
  workflows/jobs/perun/quantization_1.sbatch run qat-int3-1b-002 --config "$CONFIG" --prepared "$Q1_PREPARED" \
  --variant qat-int3 --qat-tokens 1000000000
find "$PERUN_RESULTS/quantization-2" -maxdepth 1 -name 'qat-int3-1b-002*' | grep . || echo "no 1B -002 output yet"
perunfsusage
squeue -u "$USER"
```

**Submits a job. Run once,** after the checks are clean. It appends the job ID
to `$PERUN_LOG_DIR/quantization-2-int3-rerun_submission.txt`.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
CONFIG=workflows/configs/model/baseline/quantization-2.json
Q1_PREPARED="$PERUN_RESULTS/quantization-1/prepare-001/result.json"
EXISTING=$(find "$PERUN_RESULTS/quantization-2" -maxdepth 1 -name 'qat-int3-1b-002*')
if [[ -n "$EXISTING" ]]; then
    echo "STOP: the 1B -002 output or lock already exists:"; echo "$EXISTING"
else
    QAT1B=$(sbatch --parsable --job-name=quantization-2 --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
        --time=24:00:00 --output="$PERUN_LOG_DIR/%x_%j.out" --error="$PERUN_LOG_DIR/%x_%j.err" \
        workflows/jobs/perun/quantization_1.sbatch run qat-int3-1b-002 --config "$CONFIG" \
        --prepared "$Q1_PREPARED" --variant qat-int3 --qat-tokens 1000000000)
    printf 'QAT1B=%s\n' "${QAT1B%%;*}" | tee -a "$PERUN_LOG_DIR/quantization-2-int3-rerun_submission.txt"
    echo "<- record this in local/perun/runs/quantization-2.md"
fi
```

**Start check. Safe to re-run,** about 30 minutes after the job starts, and
again about 2 h after it starts, for the 100M row. The start check must show 66 projections
and 0 mismatched, and the token-0 KL must be near 0.2227. The first-step
gradient norm should be near 0.56; it appears with the 10M row. The 10-100M rows
compare with `qat-int3-100m-002`, which trained on the same first tokens; gaps
of up to about 4% are noise. The 24-batch monitoring perplexity is not a gate.

```bash
python - <<'EOF'
import json, os
from pathlib import Path

twin = {0: 0.22268, 10: 0.09793, 25: 0.0885, 50: 0.0819, 75: 0.07878, 100: 0.07954}
found = sorted(Path(os.environ["PERUN_SCRATCH"]).glob("job_*/results/qat-int3-1b-002/result.json"))
if not found:
    print("no running copy on SCRATCH yet")
for path in found:
    data = json.loads(path.read_text())
    results = data.get("results", {})
    recovery = results.get("recovery", {})
    check = results.get("int3_fake_quantization_check")
    print(path.parts[-4], data.get("status"), data.get("current_stage"), "| error:", data.get("error"))
    print("stream:", results.get("recovery_stream", {}).get("sha256", "-")[:8])
    print("start check:", f"{check['projections']} projections, {len(check['mismatched'])} mismatched" if check else "not run yet")
    print("first-step gradient norm:", (recovery.get("first_step") or {}).get("gradient_l2_norm"))
    for row in recovery.get("history", []):
        tokens, kl = row["tokens_seen"] // 1_000_000, row["kl"]["kl"]
        reference = twin.get(tokens)
        print(f"{tokens:>6}M KL {kl:.5f}" + (f"  twin {reference:.5f}  {kl / reference - 1:+.2%}" if reference else ""))
EOF
```

**Watch** with 11.3, replacing its first line as in 11.5. The stops are also
those of 11.5.

## 12. Heterogeneous-1

Design, checks, and resource plan:
[heterogeneous-1.md](../../experiments/model/heterogenous/heterogeneous-1.md).
Every submission needs the researcher's go. Everything goes to
`perun-results/heterogeneous-1/`.

| Step | Block | Requested per task | Expected per task | Before the next step |
| --- | --- | --- | --- | --- |
| Update and pre-submission checks | 12.1 | - | minutes | Every line OK |
| **Sitting 1:** recipe selection (24 tasks) and baseline table (1 job) | 12.2 | 4 h and 3 h, 64 GB | 1.2-2.4 h and 1.2-1.7 h | data-identity, check-6, check-2 pass in every task; the first linear fits are near their expected KL (12.3) |
| Merge and review | 12.4 | login node | minutes | You pin `recipes` in the configuration |
| **Sitting 2:** cost tables (22 tasks) and recovery (3 tasks) | 12.5 | 3 h, 64 GB | 0.7-1.8 h | check-8 passes |
| **Sitting 3:** knapsack solve (1 CPU job) | 12.6 | `cpu_short`, 2 h, 16 GB | 0.3-0.7 h | check-1 and check-4 pass |
| **Sitting 4:** assembly (8 tasks) | 12.7 | 6 h, 128 GB | 0.2-1.5 h | check-7 and check-8 pass |
| Summary | 12.8 | login node | minutes | - |

Do not update the checkout while a job of yours is queued or running, including
`qat-int3-1b-002`: a job copies the code when it starts, and a resume refuses
changed code. Update it between sittings only for the `recipes` commit in 12.4.
There is no resume in this experiment. To rerun a failed task, move its output
aside first, because the job refuses an existing output.

Run 1.2 and 1.3 first.

### 12.1 Pre-submission checks

**Safe to re-run, except the `git pull`, which needs an empty queue.** Every hash
line must print OK, `squeue` must list no job when you pull, every `--test-only`
line must be accepted, and no Heterogeneous-1 output may exist yet.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
squeue -u "$USER"   # must list no job before the pull below
git status --short
git pull --ff-only origin "$PERUN_BRANCH"
git log -1 --oneline
# Every pinned SwiGLU-8 input exists and matches the configuration.
python - <<'EOF'
import hashlib, json, os
sources = json.load(open("workflows/configs/model/heterogenous/heterogeneous-1.json"))["sources"]
expected = {
    "swiglu-8/prepare-001/result.json": sources["swiglu8_prepare_sha256"],
    "swiglu-8/block-check-001/result.json": sources["swiglu8_block_check_sha256"],
}
expected.update({f"swiglu-8/fitted-target-{target}-run-001/result.json": digest
                 for target, digest in sources["swiglu8_fitted_runs_sha256"].items()})
for relative, digest in expected.items():
    path = os.path.join(os.environ["PERUN_RESULTS"], relative)
    if not os.path.isfile(path):
        print("MISSING ", relative)
        continue
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    print("OK      " if sha.hexdigest() == digest else "MISMATCH", relative)
EOF
ls "$PERUN_RESULTS/swiglu-8/prepare-001/prepared/recovery-tokens.int32" \
   "$PERUN_RESULTS/swiglu-8/prepare-001/prepared/local-evaluation-batches.pt"

# The environment matches the one SwiGLU-8's block check recorded.
PYTHONPATH=src python - <<'EOF'
import json, os
from mlp_replacement.runlog import environment_record
recorded = json.load(open(os.path.join(os.environ["PERUN_RESULTS"], "swiglu-8/block-check-001/result.json")))["environment"]
current = environment_record()
print("python  ", "OK" if recorded["python"] == current["python"] else f"DIFFERS {recorded['python']} -> {current['python']}")
for name, version in sorted(recorded["packages"].items()):
    now = current["packages"].get(name)
    print(f"{name:12}", "OK" if now == version else f"DIFFERS {version} -> {now}")
EOF

# Configuration, launcher syntax, and whether Slurm would accept the jobs (nothing is submitted).
PYTHONPATH=src python -c "from workflows.runs.model.heterogenous import heterogeneous_1 as h; s, p = h.load_settings(h.DEFAULT_CONFIG); print(s['workflow'], len(h.select_tasks(s)), 'select tasks')"
JOB=workflows/jobs/perun/heterogenous/heterogeneous_1.sbatch
bash -n "$JOB" && echo "launcher syntax OK"
sbatch --test-only --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=04:00:00 --mem=64G --array=0-23 "$JOB" select
sbatch --test-only --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=03:00:00 --mem=64G "$JOB" baseline-table
perunfsusage    # PROJECT needs under 1 GB; SCRATCH at most 5 GB per task
ls "$PERUN_RESULTS/heterogeneous-1" 2>/dev/null || echo "no Heterogeneous-1 outputs yet"
```

### 12.2 Sitting 1: recipe selection and the baseline table

**Submits jobs. Run once,** after 12.1 is clean. It writes the job IDs to
`$PERUN_LOG_DIR/heterogeneous-1_submission.txt`. If an ID prints empty, that `sbatch`
was refused: tell the assistant before submitting it again.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
JOB=workflows/jobs/perun/heterogenous/heterogeneous_1.sbatch
EXISTING=$(find "$PERUN_RESULTS/heterogeneous-1" -mindepth 1 -maxdepth 1 2>/dev/null)
if [[ -n "$EXISTING" ]] || squeue -u "$USER" -h -n heterogeneous-1 | grep -q .; then
    echo "STOP: Heterogeneous-1 outputs exist or a heterogeneous-1 job is queued:"; echo "$EXISTING"
else
    SELECT=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=04:00:00 --mem=64G \
        --array=0-23 --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
        "$JOB" select)
    BASELINE=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=03:00:00 --mem=64G \
        --output="$PERUN_LOG_DIR/%x_%j.out" --error="$PERUN_LOG_DIR/%x_%j.err" \
        "$JOB" baseline-table)
    printf 'SELECT=%s\nBASELINE=%s\n' "${SELECT%%;*}" "${BASELINE%%;*}" \
        | tee "$PERUN_LOG_DIR/heterogeneous-1_submission.txt"
    echo "<- record these in local/perun/runs/heterogeneous-1.md"
fi
```

Task map: 0-2 SwiGLU 1024 at layers 3, 11, 19 (40 fits each). 3-5 linear (41 fits).
6-11 plain MLP (39 fits), 12-17 linear + SwiGLU, and 18-23 linear + plain MLP
(42 fits), each at layers 3, 11, 19 with both sizes per layer. Every task also fits
one SwiGLU anchor first, so the watch counts below read one higher. Up to 20 run at
once, so the last tasks wait for a free slot.

### 12.3 Watch sitting 1

**Safe to re-run.** Each task checks its data, the teacher against itself, and the
SwiGLU anchor in its first minutes; a mismatch stops it with `AcceptanceFailure` in
its `.err` file. Stop and tell the assistant before trusting any result.

```bash
source "$PERUN_LOG_DIR/heterogeneous-1_submission.txt"
squeue -u "$USER" -n heterogeneous-1 -o "%.18i %.10T %.10M %.11l %R"
for TASK in $(seq 0 23); do
    LOG="$PERUN_LOG_DIR/heterogeneous-1_${SELECT}_${TASK}"
    DONE=$(grep -c '^operator=' "$LOG.out" 2>/dev/null)
    FINISHED=$(grep -l 'Verified persistent output' "$LOG.out" 2>/dev/null)
    FAILED=$(grep -l -E 'AcceptanceFailure|Traceback' "$LOG.err" 2>/dev/null)
    echo "select $TASK: ${DONE:-0} fits${FINISHED:+ | done}${FAILED:+   <- check its .err}"
done
grep -c '^table operator=' "$PERUN_LOG_DIR/heterogeneous-1_${BASELINE}.out" 2>/dev/null   # 66 when done
python - <<'EOF'
import json, os
from pathlib import Path
for result in sorted(Path(os.environ["PERUN_SCRATCH"]).glob("job_*/results/select-001-task-*/result.json")):
    data = json.loads(result.read_text())
    acceptance = data.get("results", {}).get("acceptance", {})
    print(result.parent.name, data.get("status"), {group: row["passed"] for group, row in acceptance.items()},
          (data.get("error") or "")[:120])
EOF
gpu-status
```

**Early sanity gate, about 15 minutes after tasks 3 and 4 start. Safe to re-run.** The
SwiGLU checks cannot test a new operator, so the first linear fit gives a reference:
`kl128` should be near 0.076 in task 3 (layer 3) and 0.045 in task 4 (layer 11). If
either is more than a factor of 2 away, stop and tell the assistant.

```bash
source "$PERUN_LOG_DIR/heterogeneous-1_submission.txt"
for TASK in 3 4; do
    grep -m1 '^operator=linear ' "$PERUN_LOG_DIR/heterogeneous-1_${SELECT}_${TASK}.out"
done
```

### 12.4 Merge and review

**Safe to re-run.** When the 24 tasks have finished, merge them on the login node. It
checks that they used identical configuration, code, software, and data, applies the
adoption rule per operator, and writes `heterogeneous-1/select-001`.

```bash
PYTHONPATH=src python -m workflows.runs.model.heterogenous.heterogeneous_1 merge-select \
  --results-root "$PERUN_RESULTS/heterogeneous-1"
```

It prints each operator's selected recipe, the overfitting indicator, and the sha256
for `recipes.select_sha256`. If it prints `STOP`, the indicator fired: plan the
follow-up before going on. Then copy the results home (section 6,
`EXPERIMENT=heterogeneous-1`) and review the adoption table.

Then, **on your PC**, fill in `recipes` in
`workflows/configs/model/heterogenous/heterogeneous-1.json`:

```json
"recipes": {
  "select_sha256": "<sha256 printed by the merge>",
  "settings": {
    "linear": "<selected recipe>",
    "plain_mlp": "<selected recipe>",
    "linear_swiglu": "<selected recipe>",
    "linear_plain_mlp": "<selected recipe>"
  }
},
```

Commit and push from your PC, then pull on PERUN (1.4) once your queue is empty.

### 12.5 Sitting 2: cost tables and recovery

**Submits jobs. Run once,** after 12.4. The block refuses while any heterogeneous-1
job is queued, including the baseline table. If a job stops with "Pin the reviewed
recipes", the checkout lacks the 12.4 commit.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
JOB=workflows/jobs/perun/heterogenous/heterogeneous_1.sbatch
EXISTING=$(find "$PERUN_RESULTS/heterogeneous-1" -maxdepth 1 \( -name 'cost-table-*' -o -name 'recovery-*' \))
if [[ -n "$EXISTING" ]] || squeue -u "$USER" -h -n heterogeneous-1 | grep -q .; then
    echo "STOP: outputs exist or a heterogeneous-1 job is queued:"; echo "$EXISTING"
else
    COST=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=03:00:00 --mem=64G \
        --array=0-21 --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
        "$JOB" cost-table)
    RECOVER=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=03:00:00 --mem=64G \
        --array=0-2 --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
        "$JOB" recover)
    printf 'COST=%s\nRECOVER=%s\n' "${COST%%;*}" "${RECOVER%%;*}" \
        | tee -a "$PERUN_LOG_DIR/heterogeneous-1_submission.txt"
fi
```

### 12.6 Sitting 3: the knapsack solve

**Submits a job. Run once,** after every cost-table task finished. It uses no GPU.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
SOLVE=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
    --partition=cpu_short --gres=none --cpus-per-task=4 --mem=16G --time=02:00:00 \
    --output="$PERUN_LOG_DIR/%x_%j.out" --error="$PERUN_LOG_DIR/%x_%j.err" \
    workflows/jobs/perun/heterogenous/heterogeneous_1.sbatch solve)
printf 'SOLVE=%s\n' "${SOLVE%%;*}" | tee -a "$PERUN_LOG_DIR/heterogeneous-1_submission.txt"
```

### 12.7 Sitting 4: assembly

**Submits jobs. Run once,** after the solve completed. Task i is target i of 20, 30, 40,
50, 60, 70, 75, 80%. Each task first rebuilds SwiGLU-8's own model and stops unless it
reproduces the recorded token-0 KL.

```bash
cd "$PERUN_PROJECT/diploma-thesis-block-replacement"
ASSEMBLE=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=06:00:00 --mem=128G \
    --array=0-7 --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
    workflows/jobs/perun/heterogenous/heterogeneous_1.sbatch assemble)
printf 'ASSEMBLE=%s\n' "${ASSEMBLE%%;*}" | tee -a "$PERUN_LOG_DIR/heterogeneous-1_submission.txt"
```

### 12.8 Summary

**Safe to re-run.** When recovery and assembly have finished, on the login node:

```bash
PYTHONPATH=src python -m workflows.runs.model.heterogenous.heterogeneous_1 summarize \
  --results-root "$PERUN_RESULTS/heterogeneous-1"
```

It prints the hybrid targets, the ranking swaps, and whether Heterogeneous-2's forced
runs are required. Right after each sitting, run section 7 for its job IDs (scheduler
records expire), and copy the results home (section 6, `EXPERIMENT=heterogeneous-1`).
