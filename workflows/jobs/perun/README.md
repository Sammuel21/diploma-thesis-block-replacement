# TUKE Perun Jobs

These files adapt the maintained Python runner to TUKE Perun's Slurm and
automatic-scratch environment. They contain resource requests, argument
handling, and process launch commands only; scientific experiment choices stay
in JSON files under `configs/experiments/`.

## What each file does

| File | Purpose |
| --- | --- |
| `smoke.sbatch` | Runs the tiny `perun-smoke-linear.json` configuration to check the repository, Python environment, model/data access, GPU, workflow, and result writing end to end. It is an infrastructure check, not thesis evidence. |
| `run_experiment.sbatch` | Runs one supplied JSON configuration in one isolated Python process on one GPU. Use this for an ordinary experiment. |
| `run_array.sbatch` | Maps a text manifest of JSON configurations onto a Slurm job array. Each array task runs one independent configuration on one GPU; it does not distribute one experiment over several GPUs. |
| `README.md` | Documents the Perun-specific assumptions, submission commands, scratch behavior, and artifact locations. |

The default resources are starting points. `smoke.sbatch` uses `gpu_short` for
one hour. The other launchers use `gpu_long` for up to three days because the
fitting and recovery stages may be long. Slurm options supplied to `sbatch`
can override the values in a file without editing the tracked template.

## One-time setup

Run these commands on a Perun login node after cloning or updating the
repository. First inspect the account and QoS assigned to your user:

```bash
sacctmgr show user "$USER" withassoc format=account,qos
export PERUN_ACCOUNT="your-project-account"
export PERUN_QOS="your-project-qos"
```

Activate the Conda or virtual environment containing this project's
dependencies, then export its absolute Python path:

```bash
conda activate mlp-replacement
export MLP_REPLACEMENT_PYTHON="$(command -v python)"
```

The job files deliberately do not name or activate a particular environment,
because that name and location are user-specific. Slurm exports the selected
Python path into the job. The path must remain accessible from the compute
node; do not point it at a `.venv/` inside the repository because Perun's
automatic staging excludes hidden directories.

Submit from the repository root. Perun copies the directory from which
`sbatch` is called into job-local scratch, so submitting from a nested folder
would omit the rest of the repository.

## Smoke job

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  workflows/jobs/perun/smoke.sbatch
```

This invokes `configs/experiments/perun-smoke-linear.json` and writes a small
structured run record. It still loads the real model and datasets, so model
and dataset access must already work on Perun.

## One experiment

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  workflows/jobs/perun/run_experiment.sbatch \
  configs/experiments/smollm2-one-shot-linear.json
```

An optional second argument chooses the JSON output path:

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  workflows/jobs/perun/run_experiment.sbatch \
  configs/experiments/smollm2-one-shot-linear.json \
  data/results/perun/my-run.json
```

For example, a shorter resource override can be submitted as:

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  --partition=gpu_short --time=12:00:00 --mem=48G \
  workflows/jobs/perun/run_experiment.sbatch \
  configs/experiments/smollm2-one-shot-linear.json
```

## Experiment array

Create a plain-text manifest with one repository-relative JSON configuration
path per non-empty line. Lines beginning with `#` are ignored. For example:

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

The array range is intentionally supplied at submission time because Slurm
directives cannot infer it from the manifest. An out-of-range task fails before
loading a model.

## Scratch and results

Each launcher executes `source .activate_scratch`, the helper created by
Perun's prolog. The helper moves the process into the copied repository on
job-local Lustre scratch. The Python runner writes its JSON beneath
`data/results/perun/` by default. Perun's epilog then synchronizes the scratch
tree to `~/results_job_<job-id>/` and cleans the job scratch directory.

The `.out` and `.err` files contain console output and tracebacks. The JSON run
record is the machine-readable artifact intended for later notebook loading.
For a failed Python workflow, the runner attempts to store failure details in
that JSON before re-raising the exception.
