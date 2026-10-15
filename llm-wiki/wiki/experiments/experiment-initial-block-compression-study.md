---
id: experiment-initial-block-compression-study
title: Initial Block-Compression Study
summary: Defines the working activation, practical single-block replacement, and multi-block degradation studies; links the separate Quantization-1 baseline.
type: experiment
status: draft
created: 2026-07-27
updated: 2026-10-15

epistemic:
  role: mixed
  confidence: not-assessed
  verification:
    - unverified

sources: []
related:
  - "[[method-block-importance]]"
  - "[[method-hybrid-operator-replacement]]"
  - "[[experiment-swiglu-operator-design-progression]]"
  - "[[experiment-baseline-operator-analysis]]"
  - "[[concept-replacement-error-propagation]]"
  - "[[decision-primary-compression-evaluation-scope]]"
  - "[[implementation-compute-environments]]"
  - "[[implementation-maintained-mlp-replacement-package]]"
  - "[[decision-working-experiment-code-standards]]"
  - "[[synthesis-homogeneous-swiglu-findings]]"
supersedes: []
superseded_by: []
---

# Initial Block-Compression Study

## Overview

**Working umbrella design.** This page connects the initial block-level
notebooks; it does not duplicate their numerical evidence. The design of the
artifact-backed singleton studies is on [[experiment-baseline-operator-analysis]]
and [[experiment-swiglu-operator-design-progression]]; their results are in
`docs/experiments/block/`. Activation analysis and the separate generic
multi-block interaction stage remain unverified here. The completed homogeneous
model-level SwiGLU family is recorded in `docs/experiments/model/swiglu/`, with
its attribution in [[synthesis-homogeneous-swiglu-findings]].

## Objective and Hypotheses

The study progresses through three questions:

| Stage | Experimental unit | Main question |
| --- | --- | --- |
| Activation analysis | One untouched MLP | Do captured activations show descriptive low-dimensional structure? |
| Single-block replacement | One fitted substitute at a time | Which simple controls and operator families preserve local and model behavior? |
| Multi-block interaction | Several simultaneous substitutes | How do selection, replacement count, and propagated errors compose? |

**Researcher hypotheses.** Post-gating activations may be compressible; a
reduced-width SwiGLU may outperform removal and constant controls; local error,
BI, and global KL may capture different aspects of replaceability; and damage
from simultaneous replacements may be non-additive. These are not established
results.

The separate [Quantization-1 baseline](../../../docs/experiments/model/baseline/quantization-1.md)
(uniform eligible-MLP weight-only INT8/INT4) is complete; its results are in
that doc. It measures stored-byte reduction, not parameter removal, and is not
a fourth required stage of this initial study. The doc owns its library
attribution and project-defined protocol.

## Configuration and Inputs

Current shared defaults are SmolLM2-1.7B, sequence length 128, seed 21, layer
11 for the initial single-block studies, and protected first and final layers
for depth-wide studies. Detailed data, fitting, width, and recovery settings
are owned by the two downstream experiment pages.

Primary inputs are:

- [`activation-analysis.ipynb`](../../../notebooks/block/analysis/activation-analysis.ipynb)
- [`baseline-testing.ipynb`](../../../notebooks/block/baseline/baseline-testing.ipynb)
- [`block-interaction.ipynb`](../../../notebooks/model/interaction/block-interaction.ipynb)
- [`operator.ipynb`](../../../notebooks/block/operator/operator.ipynb)
- [`src/mlp_replacement/`](../../../src/mlp_replacement/)

Results are written under `data/results/notebook-block-study/`; raw activation
tensors are intentionally not preserved as experiment artifacts.

## Procedure and Metrics

1. Capture MLP input, post-gating activation, and output at one untouched
   block; report covariance spectra, effective rank, and held-out
   reconstruction error.
2. At layer 11, compare the original MLP with zero, mean, dense linear, dense
   affine, and a $0.5d_{\mathrm{ff}}$ SwiGLU. Report exact footprint, local
   fidelity, and WikiText-2 loss/perplexity without recovery.
3. Extend to controlled capacity, operator-family, layer, and recovery studies
   without changing the fixed reference definitions.
4. Measure simultaneous replacements separately and compare observed damage
   with singleton expectations rather than assuming additivity.

Replacement metrics include parameter count, theoretical weight bytes,
serialized size where applicable, local MSE/relative MSE/cosine similarity,
model loss/perplexity, teacher-student KL, and a declared multi-block
interaction statistic. Activation measurements are descriptive evidence and
do not determine a deployable replacement width by themselves.

## Direct Results and Interpretation

No numerical results are duplicated here. The checked single-block findings
are recorded in [[experiment-baseline-operator-analysis]] and
[[experiment-swiglu-operator-design-progression]]. Their KL profiles describe
one fitted operator replacing one block in isolation; they do not predict
simultaneous replacement behavior.

## Limitations and Reproducibility

- The initial activation and fixed-baseline notebooks emphasize one model and
  layer 11.
- The $0.5d_{\mathrm{ff}}$ reference is practical, not optimal or minimal.
- The historical dense-linear degradation notebook is a limited diagnostic,
  not the final interaction protocol.
- The multi-block operator set, subset construction, order, and recovery
  controls have not yet been frozen.
- Quantization-1 is a separate baseline; its completed results are in
  [`quantization-1.md`](../../../docs/experiments/model/baseline/quantization-1.md).
- Latency and downstream benchmark claims remain outside this initial study.

Status: draft and mixed. Downstream singleton pages are experiment-backed for
their recorded artifacts; activation analysis, reusable-runner parity,
quantization execution, and the separate generic multi-block stage remain
unverified. Completed homogeneous findings have their own evidence boundary.

## Relationships

- [[experiment-baseline-operator-analysis]] owns calibration, width, recovery,
  and baseline singleton-sensitivity evidence.
- [[experiment-swiglu-operator-design-progression]] owns the broader generic
  operator comparison and remaining branch/topology proposals.
- [[synthesis-homogeneous-swiglu-findings]] owns the completed model-level
  homogeneous progression and the control for later experiment classes.
- [[concept-replacement-error-propagation]] motivates the transition from
  singleton profiles to simultaneous-replacement experiments.
- [[method-block-importance]] separates forward-only screening signals from
  operator-conditioned KL sensitivity.
- [[decision-primary-compression-evaluation-scope]] defines the primary
  footprint-quality evaluation boundary.
- [[implementation-maintained-mlp-replacement-package]] records where reusable
  operations from these notebooks belong in the maintained source tree.

## Sources

No registered literature source is cited directly. This page is a
collaborative project map; related pages retain the relevant literature and
artifact provenance.
