# Maintained Experiment Workflows

`workflows/` is the execution layer above the reusable scientific code in
`src/mlp_replacement/`. It contains process entry points and scheduler files;
model replacement, fitting, recovery, and evaluation logic stays in `src/`.

```text
workflows/
|-- runs/                    Python process entry points
`-- jobs/
    `-- perun/               TUKE Perun Slurm submission files
```

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

## Current migration boundary

`runs/run_experiment.py` is the maintained configuration-driven workflow that
already existed as a pipeline entry point. Notebook-specific workflows such as
the evolving SwiGLU studies have not yet been claimed as equivalent Python
jobs. They should be migrated here only when their inputs, outputs, and artifact
contracts are defined well enough to preserve notebook parity.

For TUKE Perun submission and artifact-retrieval instructions, see
[`jobs/perun/README.md`](jobs/perun/README.md).
