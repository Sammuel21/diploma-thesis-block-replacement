# Model-wide workflow runners

These modules are the headless execution counterparts of the retained
notebooks under `notebooks/model/`. They run the scientific computation and
write notebook-compatible JSON; notebooks remain the explanatory and analysis
frontend.

| Notebook | Python module | Configuration | Default stage |
| --- | --- | --- | --- |
| `compression-baseline.ipynb` | `workflows.runs.model.compression_baseline` | `workflows/configs/model/compression-baseline.json` | optimized |
| `swiglu.ipynb` | `workflows.runs.model.swiglu` | `workflows/configs/model/swiglu.json` | optimized |
| `swiglu-2.ipynb` | `workflows.runs.model.swiglu_2` | `workflows/configs/model/swiglu-2.json` | complete search |
| `swiglu-3.ipynb` | `workflows.runs.model.swiglu_3` | `workflows/configs/model/swiglu-3.json` | calibration, sparsity, and recovery |

The two older notebooks contain historical and optimized sections. Their
runners expose `--stage historical`, `--stage optimized`, and `--stage all`.
The optimized stage is the default because it matches the notebooks' current
execution switches and consumes a historical artifact as a control.
`swiglu-2` consumes the optimized `swiglu` schema-3 artifact.

The existing artifacts under `data/results/` are local and git-ignored. A
fresh clone on Perun will not contain them. Before an optimized-only job,
transfer its prerequisite artifact into the submitted repository tree or pass
another staged path with `--historical-artifact` / `--reference-artifact`.
Alternatively, `--stage all` regenerates both sections of
`compression-baseline` or `swiglu` in dependency order, at substantially
higher compute cost. `swiglu-2` always needs a completed optimized `swiglu`
artifact.

Run from the repository root with `src/` on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m workflows.runs.model.compression_baseline --stage optimized
python -m workflows.runs.model.swiglu --stage optimized
python -m workflows.runs.model.swiglu_2
python -m workflows.runs.model.swiglu_3
```

Every runner accepts `--config` and `--output`. The two staged runners also
accept `--historical-artifact` and `--historical-output`; `swiglu-2` accepts
`--reference-artifact`. Repository-relative paths are resolved from the
repository root, including inside Perun's staged copy.

An output path is never overwritten. Each science artifact has a sibling
`*.run.json` sidecar that is initialized before model loading and records the
current stage, completed stage summaries, environment, and any exception raised
after sidecar initialization. Preflight failures, such as a missing prerequisite
artifact or an occupied output path, are reported through stderr before a new
sidecar is created. The science JSON retains the source notebook's schema and
table keys so a loading-only notebook can consume it without rerunning the fit.

The `swiglu` optimized path and every `swiglu-2` fitting phase capture a
bounded group of MLP blocks, keep activation tensors on CPU in the model's
native dtype, and release each group before capturing the next. Fitted
operators remain resident on CPU only as long as later integrated evaluation
requires them. No activation cache is written to disk.

## Validation boundary

The migration preserves the notebook inputs, partition order, operators,
allocation equations, metrics, recovery procedure, and artifact keys by code
inspection. It is not yet empirically parity-validated: this workstation has
no usable project Python environment or GPU, and the repository has not yet
recorded a Perun smoke run. Compare a runner artifact against the
corresponding notebook artifact before treating numerical equivalence as
established thesis evidence.

`swiglu-3` also accepts `--resume` with the same explicit `--output` and
`--smoke` for checked-in reduced budgets. Its long recovery writes incremental
current/best state plus retained 10M and 100M milestone states beneath the
output's sibling asset directory. See
[`swiglu-3.md`](../../../docs/experiments/model/swiglu-3.md) for background
launch, resume, and artifact semantics.
