# Maintained Experiment Pipelines

Pipelines are thin process entry points. Scientific logic belongs to
`src/mlp_replacement/`; a pipeline loads configuration and resources, invokes
one workflow, and emits the result.

Add `src/` to `PYTHONPATH`, then run one JSON configuration:

```powershell
$env:PYTHONPATH = "src"
python -m pipelines.run_experiment path/to/experiment.json
```

Every execution writes one atomic JSON record under `data/results/`. Supply an
explicit path when useful:

```powershell
python -m pipelines.run_experiment path/to/experiment.json --output data/results/run-001.json
```

The file contains the resolved configuration, environment, completed stage
metrics, final result, and failure details when execution raises an exception.
It is updated after major workflow stages. Model weights, activation tensors,
and teacher logits are not written into the JSON log.

One experiment per process is deliberate: terminating the process releases
model, activation, and CUDA allocator state reliably. Search orchestration,
resumption, and multi-run scheduling remain outside the single-run workflow.
