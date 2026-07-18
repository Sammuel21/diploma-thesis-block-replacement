# Maintained Experiment Pipelines

Pipelines are thin process entry points. Scientific logic belongs to
`src/mlp_replacement/`; a pipeline loads configuration and resources, invokes
one workflow, and emits the result.

Add `src/` to `PYTHONPATH`, then run one JSON configuration:

```powershell
$env:PYTHONPATH = "src"
python -m pipelines.run_experiment path/to/experiment.json
```

One experiment per process is deliberate: terminating the process releases
model, activation, and CUDA allocator state reliably. Search orchestration,
resumption, and persistent experiment logging will be added with the artifact
contract rather than embedded in the scientific functions.
