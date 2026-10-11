---
metadata_version: 1
title: PERUN Runbook
type: procedure
category: infrastructure/perun
status: active
created: 2026-10-11
modified: 2026-10-11
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
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
has two: prepare, and the model array of 5 runs (section 4). The two
experiments are independent and can run at the same time.

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

### 0.2 Quantization-1 environment

**Safe to re-run.** It creates the environment only if it is missing, and pip
skips what is already installed. Use a Conda environment, not a `venv`: the job
scripts resolve the interpreter's real path, which bypasses a `venv`.

```bash
source ~/.perun-private.sh
cd "/mnt/project/$PERUN_ACCOUNT/diploma-thesis-block-replacement"
source ~/miniconda3/etc/profile.d/conda.sh
perunfsusage   # HOME needs several GB free for PyTorch and its CUDA libraries
conda env list | grep -qE '^quantization-1[[:space:]]' || conda create -y -n quantization-1 python=3.12
conda activate quantization-1
python -m pip install --no-cache-dir -r workflows/environments/quantization-1-requirements.txt
python -m pip check
conda deactivate
```

### 0.3 Check both environments

**Safe to re-run.** Run 1.2 first. Login nodes have no GPU; each job checks
CUDA on its GPU node before it takes its output lock.

After 1.3 (SwiGLU-8):

```bash
python -c 'from importlib.metadata import version; import torch, transformers, datasets, lm_eval; from lm_eval.models.huggingface import HFLM; print("mlp-replacement OK:", torch.__version__, "transformers", transformers.__version__, "accelerate", version("accelerate"))'
python -m pip check
```

After 1.3 (Quantization-1):

```bash
python -c 'from importlib.metadata import version; import torch, torchao, transformers, datasets, lm_eval; from lm_eval.models.huggingface import HFLM; print("quantization-1 OK:", torch.__version__, "torchao", version("torchao"), "mslk", version("mslk"))'
python -m pip check
```

### 0.4 Save the environment specifications

**Safe to re-run;** it refreshes the files. The PERUN Conda guide recommends
keeping an export of each environment in HOME, so a broken environment can be
rebuilt.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda env export -n mlp-replacement > ~/conda-mlp-replacement.yml
conda env list | grep -qE '^quantization-1[[:space:]]' && conda env export -n quantization-1 > ~/conda-quantization-1.yml
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

### 1.3 Choose the environment

**Safe to re-run.** Jobs use the Python that `MLP_REPLACEMENT_PYTHON` points
to *at submission*. Run the matching block before submitting, and again
whenever you switch experiments.

SwiGLU-8:

```bash
conda activate mlp-replacement
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${PERUN_BASE_LD_LIBRARY_PATH:+:$PERUN_BASE_LD_LIBRARY_PATH}"
export MLP_REPLACEMENT_PYTHON="$(command -v python)"
echo "Jobs will use: $MLP_REPLACEMENT_PYTHON"
```

Quantization-1:

```bash
conda activate quantization-1
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

Run 1.2 and either 1.3 block first; these use the environment's `python`.

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
for experiment in ("swiglu-8", "quantization-1"):
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
    print("selected setting:", adoption["selected_recipe"], "| qualifying:", adoption["qualifying"])
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
11, for example, is slice-only at 0.5. Tasks 0-3 (fitted, 20-50%) also check
the SwiGLU-7 reproduction at every monitoring point and stop themselves on a
mismatch.

Run 1.2, 1.3 (SwiGLU-8) and 1.4 first.

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

## 4. Quantization-1

Design: [quantization-1.md](../../experiments/model/baseline/quantization-1.md).
The environment setup and check are in 0.2 and 0.3. Select the environment
with 1.3 (Quantization-1) before any submission.

Two submissions:

1. **Prepare.** It freezes the evaluation data and uses no GPU, so it goes to
   a CPU partition. The launcher checks for a GPU only in the other modes.
2. **Model array.** One command, `run-array` with `--array=0-4`, starts the
   five models, each as its own task:

| Task | Model | Output |
| --- | --- | --- |
| 0 | Original BF16 (control) | `dense-001` |
| 1 | INT8 | `int8-001` |
| 2 | INT4 | `int4-001` |
| 3 | INT4 retrained on 100M tokens | `qat-int4-100m-001` |
| 4 | INT4 retrained on 1B tokens | `qat-int4-1b-001` |

The blocks below use the proposed deployment choices: prepare on `cpu_short`
for 1 hour, and one 36-hour limit for the model array, sized for the 1B QAT
run. Speed is measured per run; comparisons across different GPUs are marked
incompatible, as designed. Submit only after the researcher approves these
choices.

### 4.1 Check that Slurm accepts both requests

**Safe to re-run;** nothing is submitted. Run 1.2 and 1.3 (Quantization-1)
first.

```bash
sbatch --test-only --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  --partition=cpu_short --gres=none --cpus-per-task=4 --mem=32G --time=01:00:00 \
  workflows/jobs/perun/quantization_1.sbatch prepare prepare-001
sbatch --test-only --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=36:00:00 --array=0-4 \
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

### 4.3 Submit the model array

**Submits a job. Run once,** after preparation completed.

```bash
export Q1_PREPARED="$PERUN_RESULTS/quantization-1/prepare-001/result.json"
EXISTING=$(find "$PERUN_RESULTS/quantization-1" -maxdepth 1 \( -name 'dense-001*' -o -name 'int8-001*' -o -name 'int4-001*' -o -name 'qat-int4-*' \) 2>/dev/null)
if [[ -n "$EXISTING" ]]; then
    echo "STOP: these outputs already exist:"; echo "$EXISTING"
else
    JOB=$(sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=36:00:00 --array=0-4 \
        --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
        workflows/jobs/perun/quantization_1.sbatch run-array --prepared "$Q1_PREPARED")
    echo "quantization-1 model array: ${JOB%%;*}   <- record it in local/perun/runs/quantization-1.md"
fi
```

### 4.4 Watch the model array

**Safe to re-run.** Section 2.1 also shows each running task's current stage.

```bash
ARRAY=$(ls -t "$PERUN_LOG_DIR"/quantization-1_*_0.out | head -n 1 | sed -E 's/.*quantization-1_([0-9]+)_0\.out/\1/')
squeue -u "$USER" -n quantization-1
for TASK in 0 1 2 3 4; do
    LAST=$(tail -n 1 "$PERUN_LOG_DIR/quantization-1_${ARRAY}_${TASK}.out" 2>/dev/null)
    FAILED=$(grep -l 'Traceback' "$PERUN_LOG_DIR/quantization-1_${ARRAY}_${TASK}.err" 2>/dev/null)
    echo "task $TASK: ${LAST:-not started}${FAILED:+   <- check its .err}"
done
gpu-status
```

### 4.5 Resume one model task

**Submits a job. Run once, after the cause is understood.** Replace 4 with
the task number from the table above.

```bash
sbatch --parsable --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --time=36:00:00 --array=4 \
  --output="$PERUN_LOG_DIR/%x_%A_%a.out" --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
  workflows/jobs/perun/quantization_1.sbatch run-array --prepared "$PERUN_RESULTS/quantization-1/prepare-001/result.json" --resume
```

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
EXPERIMENT=swiglu-8   # or quantization-1
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
EXPERIMENT=swiglu-8   # or quantization-1
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
