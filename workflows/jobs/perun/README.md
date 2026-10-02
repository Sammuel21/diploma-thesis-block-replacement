# TUKE Perun Jobs

These files adapt the maintained Python runner to TUKE Perun's Slurm execution
environment. Before changing or using them, read the canonical
[Perun infrastructure guide](../../../docs/infrastructure/perun/perun.md). That guide
owns the broader access, storage, environment, scratch, artifact, and
validation assumptions. This README explains only the tracked job files.

[`heterogenous/`](heterogenous/README.md) reserves launchers for the
heterogeneous operator class. It contains directory scaffolding only; no
Slurm job or resource request is defined yet.

## Files

| File | Purpose |
| --- | --- |
| `scratch_probe.sbatch` | Performs a five-minute CPU-only check of manual PROJECT-to-SCRATCH staging and PROJECT stage-out without loading Python or requesting a GPU. |
| `smoke.sbatch` | Runs `perun-smoke-linear.json` with minimal budgets to check the real model, data, GPU, workflow, and result path. It is an infrastructure check, not thesis evidence. |
| `run_experiment.sbatch` | Runs one supplied JSON configuration in one isolated Python process on one GPU. |
| `run_array.sbatch` | Maps a manifest of JSON configurations onto independent one-GPU Slurm array tasks. It does not distribute one experiment across GPUs. |
| `run_model.sbatch` | Runs the historical allow-listed model-wide workflows that still depend on the unavailable automatic-scratch helper. It is not the SwiGLU-7 launcher. |
| `swiglu_7.sbatch` | Runs the independent SwiGLU-7 preparation or maps array tasks `0-11` onto its fixed three-scope/four-target grid. |

`smoke.sbatch` defaults to `gpu_short`, 48 GB of CPU memory, and one hour. The
generic and array launchers default to `gpu_long`, 64 GB, and 72 hours.
`run_model.sbatch` defaults to the documented four-day `gpu_long` ceiling
because the migrated studies contain long local-fitting loops. These are
starting values, not measured requirements. Options passed to `sbatch` may
override them.

`swiglu_7.sbatch` requests one GPU, eight CPUs, 128 GB RAM, and 48 hours on
`gpu_short`. It is the only supported PERUN launcher for SwiGLU-7 new runs and
resumes.

## Manual scratch probe

PERUN support confirmed on 2026-09-27 that `.activate_scratch` is currently
unavailable and that staging is manual. Before allocating a GPU, verify the
manual pipeline from a repository checkout directly below its PROJECT root:

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  workflows/jobs/perun/scratch_probe.sbatch
```

The probe uses `cpu_short`, one CPU, 1 GB RAM, and a five-minute limit. It
copies the checkout to `/mnt/scratch/$USER/job_<job-id>`, runs there, copies a
small result to the sibling PROJECT directory
`perun-results/job_<job-id>/`, verifies the persistent copy, and removes only
its marker-verified scratch directory. It preserves scratch if the probe
fails.

## Submission inputs

On a Perun login node, inspect and export the account and QoS assigned to the
current user:

```bash
export PERUN_ACCOUNT="your-project-account"
export PERUN_QOS="your-project-qos"
scontrol show assoc_mgr users="$USER" accounts="$PERUN_ACCOUNT" flags=assoc
scontrol show assoc_mgr qos="$PERUN_QOS" flags=qos
```

Activate the project environment and export its absolute interpreter path:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlp-replacement
export MLP_REPLACEMENT_PYTHON="$(command -v python)"
export HF_HOME="/mnt/project/$PERUN_ACCOUNT/huggingface-cache"
export PERUN_LOG_DIR="/mnt/project/$PERUN_ACCOUNT/perun-job-logs"
mkdir -p "$HF_HOME" "$PERUN_LOG_DIR"
```

SwiGLU-7 uses the Hugging Face model adapter from the pinned evaluation
harness. In a clean environment, install that backend rather than only the
base package:

```bash
python -m pip install 'lm_eval[hf]==0.4.13'
```

Before submission, verify the exact lazy-loaded adapter import. A successful
top-level `import lm_eval` does not prove that its optional `accelerate`
dependency is installed:

```bash
python -c 'from importlib.metadata import version; from packaging.version import Version; from lm_eval.models.huggingface import HFLM; accelerate_version = version("accelerate"); assert Version(accelerate_version) >= Version("0.26.0"); print("HFLM: OK; accelerate:", accelerate_version)'
python -m pip check
```

The job files request export of the submission environment. They intentionally
do not hardcode an environment name or path. The interpreter must remain
accessible from compute nodes and must not be a `.venv/` inside the staged
repository.

Run `sbatch` from the repository root. The dedicated SwiGLU-7 launcher copies
that checkout to a unique `/mnt/scratch/$USER/job_<job-key>` directory. Use a
clean checkout so untracked runtime data does not waste staging time or scratch
capacity. Keep installed environments and the reusable Hugging Face cache in
persistent storage outside the checkout.

## Smoke job

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  workflows/jobs/perun/smoke.sbatch
```

This uses `configs/experiments/perun-smoke-linear.json`. It still loads the
real model and datasets, so a successful result also confirms cache or network
access from the compute allocation.

## One experiment

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  workflows/jobs/perun/run_experiment.sbatch \
  configs/experiments/smollm2-one-shot-linear.json
```

An optional second script argument chooses the structured JSON output path:

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  workflows/jobs/perun/run_experiment.sbatch \
  configs/experiments/smollm2-one-shot-linear.json \
  data/results/perun/my-run.json
```

Slurm resource options can be overridden without modifying the tracked file:

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  --partition=gpu_short --time=12:00:00 --mem=48G \
  workflows/jobs/perun/run_experiment.sbatch \
  configs/experiments/smollm2-one-shot-linear.json
```

## Experiment array

Create a plain-text manifest with one repository-relative JSON configuration
path per non-empty line. Lines beginning with `#` are ignored:

```text
configs/experiments/operator-linear-seed-1.json
configs/experiments/operator-linear-seed-2.json
configs/experiments/operator-swiglu-seed-1.json
```

For a three-line manifest, submit array indices `0-2`. The optional `%2`
limits concurrent tasks to two:

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --array=0-2%2 \
  workflows/jobs/perun/run_array.sbatch configs/experiments/perun-array.txt
```

The array range is supplied at submission time because a Slurm directive
cannot infer it from the manifest. An out-of-range task fails before loading a
model.

## Model-wide notebook workflows

The model launcher maps a short allow-listed name to a Python module. It does
not contain selection, fitting, allocation, recovery, or evaluation logic.
The runner's default configuration and stage are used when no further
arguments are supplied:

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  workflows/jobs/perun/run_model.sbatch compression-baseline

sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  workflows/jobs/perun/run_model.sbatch swiglu

sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  workflows/jobs/perun/run_model.sbatch swiglu-2
```

The first two default to their optimized stage and require their historical
result as a control. These `data/results/` artifacts are git-ignored, so a
fresh Perun clone does not contain them. Transfer the required artifact into
the tree before submission, pass its staged path explicitly, or use `--stage
all` to regenerate both sections in dependency order. To regenerate only a
historical artifact:

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  workflows/jobs/perun/run_model.sbatch swiglu --stage historical
```

Arguments after the workflow name are forwarded to the Python entry point.
Use that to select another config, reference artifact, stage, or output path.
If `--output` is omitted, the launcher supplies the unique path
`data/results/perun/model/<workflow>-<job-id>.json`. The runner writes failure
state to the sibling `<workflow>-<job-id>.run.json` file.

The launcher requests one GPU because none of these Python modules implements
distributed execution. `swiglu-2` is especially compute intensive; do not
split it into independent array tasks unless its promotion and finalist
dependencies are first redesigned explicitly.

## SwiGLU-7 production workflow

The launcher creates disposable work and live output below its per-job SCRATCH
directory. It manually copies the live output to the stable identity path
`PROJECT/perun-results/swiglu-7/<run-id>` on normal completion, Python failure,
`TERM`, or `INT`. It verifies `result.json` before removing scratch. If copy or
verification fails, it preserves both scratch and a PROJECT lock for manual
recovery instead of risking the persistent copy.

SwiGLU-7 is the first entry using this contract. Its preparation is independent:
it regenerates the fixed allocation curves, four fitted starts, recovery data,
and evaluation references from pinned model/dataset sources. No SwiGLU-5 or
SwiGLU-6 runtime artifact is staged.

Prepare once with the dedicated job:

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  --output="$PERUN_LOG_DIR/%x_%j.out" \
  --error="$PERUN_LOG_DIR/%x_%j.err" \
  workflows/jobs/perun/swiglu_7.sbatch prepare
```

After a verified stage-out, preparation is already stored at
`$PERUN_PROJECT/perun-results/swiglu-7/prepare-001`. Then submit the fixed grid;
`%12` allows all 12 independent one-GPU tasks to run concurrently when the
scheduler has capacity:

```bash
export S7_PREPARED="$PERUN_PROJECT/perun-results/swiglu-7/prepare-001/result.json"

sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  --array=0-11%12 \
  --output="$PERUN_LOG_DIR/%x_%A_%a.out" \
  --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
  workflows/jobs/perun/swiglu_7.sbatch train "$S7_PREPARED"
```

Tasks `0-3`, `4-7`, and `8-11` map respectively to S7-0, S7-1, and S7-2;
within each group the targets are 0.2, 0.3, 0.4, and 0.5. Each task writes a
separate identity-named directory below its own `$RESULTS_DIR`. Do not point
checkpoint-heavy live output at slow persistent NFS.

Resume preparation or one failed array element from its persistent PROJECT
output:

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  --output="$PERUN_LOG_DIR/%x_%j.out" \
  --error="$PERUN_LOG_DIR/%x_%j.err" \
  workflows/jobs/perun/swiglu_7.sbatch prepare --resume

sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --array=5 \
  --output="$PERUN_LOG_DIR/%x_%A_%a.out" \
  --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
  workflows/jobs/perun/swiglu_7.sbatch train --resume "$S7_PREPARED"
```

Task 5 is S7-1 at target 0.3. Replace it with the failed task index. Never run
two jobs for the same identity simultaneously. A lock left after an incomplete
stage-out requires inspection and manual recovery before resubmission.

Use task 0 as the first native-8K resource measurement when scheduler policy
does not permit releasing the full array immediately. Do not extrapolate the
earlier 128-token runtime without accounting for 8K attention.
See the [SwiGLU-7 experiment guide](../../../docs/experiments/model/swiglu/swiglu-7.md)
for prerequisites, resume commands, allocation mapping, estimated cost, and
result handling.

## Results and logs

[quantization_1.sbatch](quantization_1.sbatch) submits one explicitly selected
Quantization-1 prepare/run/evaluate job. It stages supplied inputs, locks the
chosen output, and verifies durable stage-out before scratch cleanup. It uses
the separate pinned quantization environment and does not rely on
`.rsyncignore`. See the
[Quantization-1 guide](../../../docs/experiments/model/baseline/quantization-1.md)
for submission, resource-planning, and resume requirements.

The documented submission commands override the tracked fallback paths and
write single-job logs as `%x_%j` and array-task logs as `%x_%A_%a` below
`$PERUN_LOG_DIR`. Each identity-named PROJECT output contains the
structured `result.json` and `run.json`; completed training outputs also retain
raw evaluation records and the final `model/` bundle. Incomplete training
outputs retain one verified checkpoint. These files are copied to PROJECT, not
HOME, and SCRATCH is never the only durable copy after a verified stage-out.
