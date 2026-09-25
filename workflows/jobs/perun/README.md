# TUKE Perun Jobs

These files adapt the maintained Python runner to TUKE Perun's Slurm execution
environment. Before changing or using them, read the canonical
[Perun infrastructure guide](../../../docs/infrastructure/perun.md). That guide
owns the broader access, storage, environment, scratch, artifact, and
validation assumptions. This README explains only the tracked job files.

## Files

| File | Purpose |
| --- | --- |
| `smoke.sbatch` | Runs `perun-smoke-linear.json` with minimal budgets to check the real model, data, GPU, workflow, and result path. It is an infrastructure check, not thesis evidence. |
| `run_experiment.sbatch` | Runs one supplied JSON configuration in one isolated Python process on one GPU. |
| `run_array.sbatch` | Maps a manifest of JSON configurations onto independent one-GPU Slurm array tasks. It does not distribute one experiment across GPUs. |
| `run_model.sbatch` | Runs one allow-listed model-wide workflow and forwards its Python CLI arguments. Existing entries retain their historical `--output` interface; directory-contract entries receive Perun work/output directories. |
| `swiglu_7.sbatch` | Runs the independent SwiGLU-7 preparation or maps array tasks `0-11` onto its fixed three-scope/four-target grid. |

`smoke.sbatch` defaults to `gpu_short`, 48 GB of CPU memory, and one hour. The
generic and array launchers default to `gpu_long`, 64 GB, and 72 hours.
`run_model.sbatch` defaults to the documented four-day `gpu_long` ceiling
because the migrated studies contain long local-fitting loops. These are
starting values, not measured requirements. Options passed to `sbatch` may
override them.

`swiglu_7.sbatch` requests one GPU, eight CPUs, 128 GB RAM, and 48 hours on
`gpu_long`. It is the preferred new-run launcher for SwiGLU-7; `run_model.sbatch`
remains the explicit single-run/resume interface.

## Submission inputs

On a Perun login node, inspect and export the account and QoS assigned to the
current user:

```bash
sacctmgr show user "$USER" withassoc format=account,qos
export PERUN_ACCOUNT="your-project-account"
export PERUN_QOS="your-project-qos"
```

Activate the project environment and export its absolute interpreter path:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlp-replacement
export MLP_REPLACEMENT_PYTHON="$(command -v python)"
```

The job files request export of the submission environment. They intentionally
do not hardcode an environment name or path. The interpreter must remain
accessible from compute nodes and must not be a `.venv/` inside the staged
repository.

Run `sbatch` from the repository root. All configuration arguments below are
repository-relative so they remain valid after automatic scratch activation.

Use a clean Perun checkout as the submission directory. The prolog copies the
whole submitted directory, including untracked runtime data, so do not keep the
local historical `data/results/` archive or earlier `results_job_*` directories
inside it. Transfer only the prerequisite artifacts required by the submitted
workflow. Keep installed environments and reusable Hugging Face caches in
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

## Future long-running workflows

A new long-running runner opts into the directory storage contract when its
allow-list entry sets `STORAGE_CONTRACT="directories"`. The launcher then
supplies:

```text
--work-dir   $TMPDIR/mlp-replacement/<workflow>-<job-id>
--output-dir $RESULTS_DIR/<workflow>-<job-id>
```

The launcher owns the work-directory path and rejects a forwarded
`--work-dir`. It removes only that exact per-job directory on normal exit,
Python failure, `TERM`, or `INT`. The repository-root `.rsyncignore` also
excludes `/tmp/mlp-replacement/` from epilog synchronization. These two
protections keep disposable files out of persistent storage even though Perun
stages new and modified scratch files back after the job.

An explicit `--output-dir` may be forwarded for resume. Stage only the prior
output directory into the submitted checkout and pass it with `--resume`; do
not copy a complete historical results tree. The Python runner validates the
checkpoint and owns durable cleanup. A completed run retains its structured
results and final model but removes resumable optimizer state.

SwiGLU-7 is the first entry using this contract. Its preparation is independent:
it regenerates the fixed allocation curves, four fitted starts, recovery data,
and evaluation references from pinned model/dataset sources. No SwiGLU-5 or
SwiGLU-6 runtime artifact is staged.

Prepare once with the dedicated job:

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  workflows/jobs/perun/swiglu_7.sbatch prepare
```

After stage-out, verify the complete
`results/swiglu-7/prepare-001` directory and move it to stable PROJECT storage.
Then submit the fixed grid; `%4` is only a scheduler concurrency limit:

```bash
export S7_PREPARED="/project/path/swiglu-7/prepare-001/result.json"

sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  --array=0-11%4 \
  workflows/jobs/perun/swiglu_7.sbatch train "$S7_PREPARED"
```

Tasks `0-3`, `4-7`, and `8-11` map respectively to S7-0, S7-1, and S7-2;
within each group the targets are 0.2, 0.3, 0.4, and 0.5. Each task writes a
separate identity-named directory below its own `$RESULTS_DIR`. Do not point
checkpoint-heavy live output at slow persistent NFS.

Use task 0 as the first native-8K resource measurement when scheduler policy
does not permit releasing the full array immediately. Do not extrapolate the
earlier 128-token runtime without accounting for 8K attention.
See the [SwiGLU-7 experiment guide](../../../docs/experiments/model/swiglu/swiglu-7.md)
for prerequisites, resume commands, allocation mapping, estimated cost, and
result handling.

## Results

The runners write their structured JSON paths inside the staged repository.
The `.out` and `.err` files contain console output and tracebacks. The generic
runner stores failure state in its run JSON. Model-wide migrations store their
notebook-compatible science artifact separately and record progress or failure
in a sibling `.run.json` sidecar.

Current official Perun pages disagree on whether the synchronized
`results_job_<job-id>/` directory appears under HOME or beside the submit
directory. Inspect both after the first intended scientific run, append the job
and observed behavior to the [Perun experiment
log](../../../docs/infrastructure/perun-log.md), and update the [Perun project
status](../../../docs/infrastructure/perun-status.md). Verify hashes before
moving the compact output to its canonical project location or deleting a
redundant stage-out directory. Do not rely on job scratch for persistent results.
