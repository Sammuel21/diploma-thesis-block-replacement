---
id: experiment-baseline-operator-analysis
title: Baseline Operator Analysis
summary: Design, hypotheses, and metric provenance of the calibration, capacity, recovery, and singleton-sensitivity analyses; results live in docs.
type: experiment
status: draft
created: 2026-08-18
updated: 2026-10-15

epistemic:
  role: mixed
  confidence: medium
  verification:
    - experiment-backed

sources:
  - source_id: src-grafting-2025
    locator: "Sections 3.2-3.3"
    relation: contextualizes
    note: "Prior-work context for staged local fitting and integrated recovery; the project baseline is not a reproduction of self-grafting."

related:
  - "[[experiment-initial-block-compression-study]]"
  - "[[experiment-swiglu-operator-design-progression]]"
  - "[[method-block-importance]]"
  - "[[comparison-recovery-objective-trainable-scope]]"
  - "[[concept-replacement-error-propagation]]"
supersedes: []
superseded_by: []
---

# Baseline Operator Analysis

## Overview

**Mixed design and single-run evidence.** `baseline-testing.ipynb` defines the
fixed six-condition, layer-11 reference. `baseline-experiments.ipynb` extends
it with calibration-data, SwiGLU-width, recovery-budget, and full-depth
singleton-sensitivity analyses. Grafting provides prior-work context for local
operator fitting followed by integrated recovery, but this project baseline is
not a reproduction of its self-grafting control. [src-grafting-2025, Sections
3.2-3.3]

## Objective and Hypotheses

The notebook asks how calibration quantity affects local fit, how reduced
SwiGLU width trades footprint against quality, how equal replacement-only
recovery affects widths, and whether local distillability predicts the global
effect of replacing one block.

**Researcher hypotheses.** More data or width may reduce local error without a
proportional model-quality gain; layer position and operator family may govern
how residual error propagates. These propositions are not established beyond
the recorded run.

## Configuration

The recorded configuration (SmolLM2-1.7B, seed 21, layer 11, layers 1-22 for
the depth study) is in the configuration appendix of
[`baseline-experiments.md`](../../../docs/experiments/block/baseline/baseline-experiments.md).

A fixed local epoch count means a larger calibration set also produces more
optimizer updates. The current scaling curve therefore combines data quantity
with optimization exposure; it is not an equal-update calibration-only
comparison.

## Procedure

1. At layer 11, fit linear, affine, and 0.50-width SwiGLU candidates across
   calibration sizes and record held-out local metrics and model
   loss/perplexity.
2. At the largest calibration size, compare SwiGLU ratios 0.25-0.90 against
   the linear references.
3. Recover each fitted width with the same ordered teacher-logit batches and
   report common cumulative update checkpoints. Only replacement parameters
   are trainable; this is not full-model retraining.
4. Fit linear, affine, and 0.25/0.50/0.75 SwiGLU independently at each eligible
   layer and measure teacher-to-student KL with one replacement active.
5. Compare held-out local relative MSE with pre-recovery KL; retain their ratio
   only as a secondary within-protocol diagnostic.

## Metrics and Project Definitions

The study uses three quantities. All are project definitions, and none
requires an external citation:

- local relative error $E_i^{\mathrm{local}}$, a project definition;
- pre-recovery singleton KL $K_i^{\mathrm{pre}}$, a project evaluation
  definition in standard KL notation;
- the ratio $R_i=K_i^{\mathrm{pre}}/(E_i^{\mathrm{local}}+\varepsilon)$, a
  project-proposed diagnostic with no universal threshold.

The full equations and their reporting rules are in the Results section of
[`baseline-experiments.md`](../../../docs/experiments/block/baseline/baseline-experiments.md).

## Direct Results and Interpretation

**Empirical findings** from one configured run (artifact
`baseline-experiments.json`, schema 2) are recorded in the Results section of
[`baseline-experiments.md`](../../../docs/experiments/block/baseline/baseline-experiments.md).
In short, local relative MSE did not rank pre-recovery KL reliably across
depth, and pre-recovery KL is operator-conditioned rather than a universal
block-importance score.

## Inputs, Artifacts, and Reproducibility

- [`baseline-experiments.ipynb`](../../../notebooks/block/baseline/baseline-experiments.ipynb)
- [`baseline-testing.ipynb`](../../../notebooks/block/baseline/baseline-testing.ipynb)
- [`src/mlp_replacement/compression/recovery.py`](../../../src/mlp_replacement/compression/recovery.py)
- Artifact contents and reproducibility status: the Results section of
  [`baseline-experiments.md`](../../../docs/experiments/block/baseline/baseline-experiments.md)

## Limitations

Design limitations are listed here; result-specific limits are in the docs
page.

- One model and seed do not establish stability.
- Calibration quantity is confounded with update count.
- Operator families are not parameter-matched.
- Calibration and width tables measure local error on C4, while KL uses
  WikiText-2. The full-depth comparison measures both on the same WikiText-2
  batches. *Corrected 2026-10-15 from the notebook version that produced the
  artifact.*
- Every full-depth row replaces one block independently; simultaneous errors
  may interact.

## Relationships

- [[experiment-initial-block-compression-study]] owns the fixed reference and
  broader stage map.
- [[experiment-swiglu-operator-design-progression]] extends the same evaluation
  conventions to more operator families.
- [[comparison-recovery-objective-trainable-scope]] defines the recovery
  objective and trainable scope.
- [[concept-replacement-error-propagation]] motivates the local-to-global and
  multi-block distinction.
- [[method-block-importance]] records forward-only screening alternatives.

## Sources

- `src-grafting-2025` - Sections 3.2-3.3, contextual comparison only
- Project-generated evidence -
  `data/results/notebook-block-study/baseline-experiments.json`, schema version
  2, created 2026-08-18 and checked 2026-08-20
