# Maintained Experiment Workflows

`workflows/` is the execution layer above the reusable scientific code in
`src/mlp_replacement/`. It contains process entry points and scheduler files;
model replacement, fitting, recovery, and evaluation logic stays in `src/`.

```text
workflows/
|-- AGENTS.md                scoped workflow and Perun requirements
|-- configs/                 explicit workflow choices and budgets
|   `-- model/               model-wide notebook-equivalent settings
|-- runs/                    Python process entry points
|   `-- model/               model-wide notebook migrations
`-- jobs/
    |-- local/               direct Linux-host launchers
    `-- perun/               TUKE Perun Slurm submission files
```

## Infrastructure prerequisite

Before designing or modifying a workflow intended for TUKE Perun, read the
canonical [Perun infrastructure guide](../docs/infrastructure/perun.md). It
records the submission, storage, environment, scratch, artifact, and validation
requirements that migrated workflows must respect. Scoped implementation-agent
instructions are in [`AGENTS.md`](AGENTS.md).

## Run one experiment

From the repository root, add `src/` to `PYTHONPATH` and pass one experiment
configuration to the runner:

```powershell
$env:PYTHONPATH = "src"
python -m workflows.runs.run_experiment configs/experiments/smollm2-one-shot-linear.json
```

An explicit output path is optional:

```powershell
python -m workflows.runs.run_experiment `
  configs/experiments/smollm2-one-shot-linear.json `
  --output data/results/run-001.json
```

Every execution writes one atomic JSON run record containing the resolved
configuration, environment, completed-stage metrics, final result, and failure
details when execution raises an exception. Model weights, captured activation
tensors, and teacher logits are not embedded in that record.

One experiment per process is deliberate. Process exit releases the model,
activation data, and CUDA allocator state between runs; a scheduler or job
array can then coordinate independent processes.

## Model-wide notebook migrations

The maintained model workflows are grouped by experiment class under
[`runs/model/`](runs/model/): homogeneous SwiGLU studies under `swiglu/` and
model baselines under `baseline/`. Their explicit choices use the same class
layout under [`configs/model/`](configs/model/). See the
[`runs/model` README](runs/model/README.md) for stage, dependency, artifact,
and validation details.

SwiGLU-5's context, fitting, candidate construction, recovery adaptation,
search, and confirmation are separated under `runs/model/swiglu/swiglu5/`.
The original CLI modules and `swiglu_5.py` compatibility imports remain available.
Shared evaluation and model reconstruction belong to `src/mlp_replacement/`;
historical artifact adaptation stays in the workflow layer. See the model-runner
README for the source-snapshot and continuation boundary of this refactor.

The older `runs/run_experiment.py` remains the generic configuration-driven
entry point for one standard replacement experiment. It is not an alias for
the multi-policy model-study runners.

For executor-specific commands, see [`jobs/local/README.md`](jobs/local/README.md)
and [`jobs/perun/README.md`](jobs/perun/README.md).

## Storage contract for long workflows

SwiGLU-1 through SwiGLU-6 retain their historical CLI and artifact layouts.
SwiGLU-7 and later long-running workflows use two operational paths without
adding those paths to their scientific configuration:

- `--work-dir` owns disposable files and is removed after the process;
- `--output-dir` owns `result.json`, `run.json`, a resumable checkpoint while
  training is active, and the final inference bundle.

The paths may be repository-relative or absolute. A runner must reject equal
or nested work/output locations, refuse a non-empty output directory for a new
run, and ensure durable artifacts contain no references to the work directory.
On successful completion it validates the final artifact before removing the
optimizer checkpoint. A failed run retains its latest verified checkpoint.

Executor launchers only bind these two paths and start the shared Python
module. They do not own scientific preparation, recovery, or evaluation.
