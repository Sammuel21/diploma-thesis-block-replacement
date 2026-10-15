---
id: implementation-maintained-mlp-replacement-package
title: Maintained MLP Replacement Package
summary: Pointer to the READMEs that map the maintained scientific code, workflow runners, and executor-specific jobs.
type: implementation
status: draft
created: 2026-08-28
updated: 2026-10-15

epistemic:
  role: mixed
  confidence: high
  verification:
    - experiment-backed

sources: []
related:
  - "[[decision-working-experiment-code-standards]]"
  - "[[experiment-initial-block-compression-study]]"
  - "[[method-discrete-width-curve-allocation]]"
  - "[[synthesis-homogeneous-swiglu-findings]]"
supersedes: []
superseded_by: []
---

# Maintained MLP Replacement Package

## Overview

**Pointer page.** This node connects methods and experiments in the wiki graph
to the maintained code. The code itself is documented next to it; this page
does not repeat module lists, which would go stale.

- [`src/README.md`](../../../src/README.md): package layout (`operators/`,
  `compression/`, `analysis/`, `evaluation/`, package root), the typical data
  flow, and compatibility exports.
- [`workflows/README.md`](../../../workflows/README.md): process entry points,
  one experiment per process, and the storage contract for long workflows.
- [`workflows/runs/model/README.md`](../../../workflows/runs/model/README.md):
  model-wide runners, including the SwiGLU family and its validation boundary.
- [`workflows/jobs/perun/README.md`](../../../workflows/jobs/perun/README.md):
  PERUN job scripts, staging, and the scratch contract.

Implementation presence is not a scientific finding. Experiment results are in
`docs/experiments/`.

## Relationships

- [[decision-working-experiment-code-standards]] governs when notebook logic
  should be promoted into the maintained package.
- [[experiment-initial-block-compression-study]],
  [[experiment-baseline-operator-analysis]], and
  [[experiment-swiglu-operator-design-progression]] use the operator,
  sensitivity, recovery, and evaluation components.
- [[method-discrete-width-curve-allocation]] is implemented by
  `compression/allocation.py`; experiment-specific fitting and orchestration
  remain in the SwiGLU runners.
- [[synthesis-homogeneous-swiglu-findings]] records the attribution boundary of
  the model-level studies run with the maintained runners.

## Sources

No registered literature source is cited. Until 2026-10-15 this page held the
package map itself; that content moved to `src/README.md` (git history keeps
the earlier version).
