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
