# Maintained Source Package

`mlp_replacement/` is the maintained implementation of the thesis workflow.
The historical MVP remains under `scripts/intro/` and `notebooks/mvp/` and is
not imported by this package.

The package follows a functional-core design. Configuration and result objects
are immutable dataclasses, model mutation is isolated in `compression/surgery.py`,
and reusable experiment orchestration is isolated in
`compression/workflows.py`. Crash-aware run logging and atomic artifact
persistence are provided by `runlog.py` and the maintained workflow layer.

## Package layout

Reusable code is grouped by the research operation it performs. This is a
maintainability decision, not a scientific claim, and the layout is not a
final public API.

| Area | Modules | Responsibility |
| --- | --- | --- |
| `operators/` | `modules.py`, `training.py` | Replacement architectures and local fitting |
| `compression/` | `allocation.py`, `selection.py`, `surgery.py`, `recovery.py`, `continuation.py`, `reconstruction.py`, `teacher_cache.py`, `adapters.py`, `quantization.py`, `workflows.py` | Width budgets, model mutation, recovery and checkpoint contracts, state reconstruction, teacher targets, LoRA, library-backed quantization, and reusable replacement flows |
| `analysis/` | `activations.py`, `screening.py`, `sensitivity.py` | Activation geometry, BI-style screening, and isolated replacement sensitivity |
| `evaluation/` | `operator.py`, `language_model.py`, `footprint.py`, `final_quality.py`, `teacher_kl.py`, `bundles.py`, `mixed_precision.py`, `generation.py`, `quantized_bundles.py` | Local fidelity, pinned likelihood and tasks, long-context teacher KL, footprint, inference bundles, dtype handling, and controlled generation measurements |
| Package root | `config.py`, `data.py`, `data_streams.py`, `model.py`, `capture.py`, `artifacts.py`, `runlog.py` | Configuration, finite streams, model loading, activation capture, artifact hashes, and JSON run records |

The typical flow is below. It lists responsibilities, not one universal job
sequence; curve-based allocation, for example, fits candidate widths before
choosing the joint topology.

1. Root configuration, model, and data modules prepare the experiment.
2. `analysis/screening.py` may produce model- or MLP-level screening scores.
3. `compression/selection.py` resolves the selected layer indices.
4. `capture.py` and `operators/training.py` fit local replacement operators.
5. `compression/surgery.py` integrates the fitted modules.
6. Recovery helpers and the selected runner apply the declared teacher-logit
   objective, trainable scope, stream, and checkpoint protocol.
7. `evaluation/` records quality and footprint outcomes.

Experiment-specific phases live in the runners under `../workflows/runs/`.
Exact continuation (`compression/continuation.py`) needs complete durable
states, not the reduced analysis exports; a hard node loss or an uncatchable
kill can bypass the final stage-out. Notebook imports are coupled to these module paths and must be checked when a
module moves. A future reusable model-compression baseline orchestrator would
belong at `compression/baselines.py`; none exists yet.

Scoped implementation instructions are in [`AGENTS.md`](AGENTS.md); the fuller
code and workflow standard is in
[`../docs/agents/maintained-code-and-workflows.md`](../docs/agents/maintained-code-and-workflows.md).

The exploratory block-study notebooks additionally use maintained helpers for
activation spectra and PCA reconstruction, richer local operator metrics,
constant/low-rank/gated/hybrid replacements, multi-module activation capture,
and exception-safe temporary replacement. Notebook-specific plots and result
tables remain in `notebooks/block/`; raw activations are not persisted.

## Reusable operations and compatibility

- `compression/teacher_cache.py` owns teacher-logit and sharded final-hidden
  caches. `compression/recovery.py` owns losses, schedules, and optimization,
  and re-exports the moved cache symbols for existing callers.
- `compression/reconstruction.py` owns replacement-state snapshots, restoration,
  saved-operator loading, and ordered SwiGLU student construction. Workflows
  remain responsible for extracting allocation records from historical artifacts.
- `compression/surgery.py` keeps casting replacement and FP32 temporary
  insertion as separate operations.
- `evaluation/mixed_precision.py` owns the shared mixed-precision evaluators
  and their one-argument autocast helper. The two-argument recovery autocast
  helper lives in `model.py` and remains importable from recovery.
- `artifacts.py` owns JSON normalization, normalized legacy fingerprints and
  writers, and the separate strict/durable operations. `runlog.json_value`
  remains a compatibility export. `config.py` owns matching configuration helpers.

New callers should import the owning module directly. Compatibility exports
preserve old import paths; they do not unify distinct evaluator, writer, or
recovery behavior. Refactor evidence and pending execution checks are tracked in
[`../plans/PLAN-behavior-preserving-refactor.md`](../plans/PLAN-behavior-preserving-refactor.md).
