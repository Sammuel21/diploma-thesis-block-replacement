# Maintained Source Package

`mlp_replacement/` is the maintained implementation of the thesis workflow.
The historical MVP remains under `scripts/intro/` and `notebooks/mvp/` and is
not imported by this package.

The package follows a functional-core design. Configuration and result objects
are immutable dataclasses, model mutation is isolated in `compression/surgery.py`,
and reusable experiment orchestration is isolated in
`compression/workflows.py`. Crash-aware run logging and atomic artifact
persistence are provided by `runlog.py` and the maintained workflow layer.

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
