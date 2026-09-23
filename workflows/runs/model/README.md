# Model-wide workflow runners

These modules are the headless execution counterparts of the retained
notebooks under `notebooks/model/`. They run the scientific computation and
write notebook-compatible JSON; notebooks remain the explanatory and analysis
frontend.

## Implementation ownership

The `swiglu/swiglu5/` package separates experiment context and source validation
(`context.py`), local fitting/allocation adaptation (`fitting.py`), candidate
assembly (`candidates.py`), recovery profiling/adaptation (`recovery.py`), and
the two execution paths (`search.py`, `confirmation.py`). Both context-preparation
functions live in `context.py`; `run_search` and `run_confirmation` live in their
respective execution modules. Lower-level modules never import the execution
modules or the compatibility facade.

`swiglu/swiglu_5.py` explicitly re-exports the original project-defined API.
The two existing CLI modules retain their arguments, defaults, and failure
handling. `swiglu/shared.py` retains historical source adaptation and old helper
imports; shared evaluation, configuration, artifact writing, and reconstruction
are owned by the maintained package. SwiGLU-6 uses package reconstruction without
depending on the SwiGLU-5 implementation. Older runners retain their own loops
and distinct helper variants.

Existing runs must use their original source snapshot, which may predate the
refactor baseline. SwiGLU-6's source hashes remain part of its fingerprint;
changed imports and new source files intentionally invalidate reuse under the
old fingerprint. Refactored runs require new outputs. Checkpoint migration and
changes to resume eligibility are outside this refactor.

The [implementation ledger](../../../plans/PLAN-behavior-preserving-refactor.md)
records baseline hashes, reversible stage commits, comparisons, and pending
checks. In the configured research Python environment, run the focused contracts
from the repository root:

```bash
python -m unittest discover -s tests -p 'test_*contracts.py' -v
python -m unittest discover -s tests -p 'test_refactor*.py' -v
```

These checks cover source structure, imports, small tensor/file invariants, and
read-only artifact interpretation. They do not launch scientific jobs. Numerical
equivalence remains pending the next intended scientific run, with existing
tolerances unchanged.

## Entry points

| Notebook | Python module | Configuration | Default stage |
| --- | --- | --- | --- |
| `baseline/compression-baseline.ipynb` | `workflows.runs.model.baseline.compression` | `workflows/configs/model/baseline/compression.json` | optimized |
| `swiglu/swiglu.ipynb` | `workflows.runs.model.swiglu.swiglu_initial` | `workflows/configs/model/swiglu/swiglu-initial.json` | optimized |
| `swiglu/swiglu-2.ipynb` | `workflows.runs.model.swiglu.swiglu_2_allocation` | `workflows/configs/model/swiglu/swiglu-2-allocation.json` | complete search |
| `swiglu/swiglu-3.ipynb` | `workflows.runs.model.swiglu.swiglu_3_calibration_recovery` | `workflows/configs/model/swiglu/swiglu-3-calibration-recovery.json` | calibration, sparsity, and recovery |
| `swiglu/swiglu-4.ipynb` | `workflows.runs.model.swiglu.swiglu_4_recovery_analysis` | `workflows/configs/model/swiglu/swiglu-4-recovery-analysis.json` | completed global-recovery analysis |
| `swiglu/swiglu-5.ipynb` | `workflows.runs.model.swiglu.swiglu_5_search` | `workflows/configs/model/swiglu/swiglu-5-search.json` | bounded search |
| `swiglu/swiglu-5.ipynb` | `workflows.runs.model.swiglu.swiglu_5_confirmation` | `workflows/configs/model/swiglu/swiglu-5-confirmation.json` | gated 100M confirmation |

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
python -m workflows.runs.model.baseline.compression --stage optimized
python -m workflows.runs.model.swiglu.swiglu_initial --stage optimized
python -m workflows.runs.model.swiglu.swiglu_2_allocation
python -m workflows.runs.model.swiglu.swiglu_3_calibration_recovery
python -m workflows.runs.model.swiglu.swiglu_4_recovery_analysis
python -m workflows.runs.model.swiglu.swiglu_5_search \
  --config workflows/configs/model/swiglu/swiglu-5-search.json \
  --source data/results/workflows/model/swiglu-3/run-001.json \
  --output data/results/workflows/model/swiglu-5/search/<unique>.json
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
inspection. SwiGLU-3 and SwiGLU-4 have completed darthmachinus artifacts; older
runner migrations still require notebook-versus-runner comparison before
numerical equivalence is established thesis evidence. Perun execution remains
a separate infrastructure validation boundary.

`swiglu-3` also accepts `--resume` with the same explicit `--output` and
`--smoke` for checked-in reduced budgets. Its long recovery writes incremental
current/best state plus retained 10M and 100M milestone states beneath the
output's sibling asset directory. See
[`swiglu-3.md`](../../../docs/experiments/model/swiglu/swiglu-3.md) for background
launch, resume, and artifact semantics.

`swiglu-4` accepts `--source`, `--resume`, and `--smoke`. It reuses the exact
50% `swiglu-3` operators and packed token stream, keeps only temporary resume
checkpoints during execution, and removes them after a successful run. See
[`swiglu-4.md`](../../../docs/experiments/model/swiglu/swiglu-4.md) for its tournament,
LoRA scopes, reporting contract, and nohup commands.

`swiglu-5-search` accepts `--source`, `--output`, and a pre-recovery `--resume`
against the original failed output. Resume validates retained operator hashes,
reuses completed width curves and initial candidates, and restarts the first
unfinished candidate stage. `--allow-over-budget` records but does not enforce
the configured runtime projection. There is no smoke mode. Its low-storage
policy retains only live tournament continuation states and the two 5M winners.
`swiglu-5-confirmation` additionally requires one
explicit `--target 0.2` or `--target 0.5` and a completed `--search-artifact`;
it creates only final and best replacement-weight snapshots, and references
the retained 5M search endpoint when that remains best. Implementing
confirmation does not authorize launching it before the search is analyzed. See
[`swiglu-5.md`](../../../docs/experiments/model/swiglu/swiglu-5.md).
