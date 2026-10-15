---
id: method-discrete-width-curve-allocation
title: Discrete Width-Curve Allocation
summary: Allocates homogeneous SwiGLU widths by minimizing a summed singleton-KL surrogate under a shared eligible-MLP parameter budget.
type: method
status: review
created: 2026-10-02
updated: 2026-10-15

epistemic:
  role: mixed
  confidence: medium
  verification:
    - source-checked
    - experiment-backed

sources:
  - source_id: src-modegpt-2025
    locator: "Section 3.3; Equations 10-11"
    relation: contextualizes
    note: "Contrast with importance-to-sparsity softmax allocation, not the source of the project's singleton-KL objective."
related:
  - "[[method-global-to-local-operator-budget-allocation]]"
  - "[[concept-replacement-error-propagation]]"
  - "[[implementation-maintained-mlp-replacement-package]]"
  - "[[synthesis-homogeneous-swiglu-findings]]"
supersedes: []
superseded_by: []
---

# Discrete Width-Curve Allocation

## Overview

**Project method and empirically supported control.** SwiGLU-5 introduced a
width-dependent allocation instead of mapping one score per layer to a width.
SwiGLU-7 independently rebuilds this method from the pinned teacher and data;
it needs no earlier run artifacts. This is one fixed allocation procedure,
not a tournament over allocation strategies. It still optimizes which width
each layer receives.

## Procedure and Objective

1. Fit each eligible MLP at retained widths 25%, 37.5%, 50%, 62.5%, 75%, and
   87.5%; the 100% anchor retains the original dense module exactly.
2. Insert each fitted module alone into the dense teacher and measure T=1
   teacher-to-student KL on the allocation-selection partition. This is model
   sensitivity to a **singleton replacement**, not local activation MSE.
3. Apply non-increasing isotonic regression to each width/KL curve. Preserve
   raw measurements alongside the pooled costs; monotonicity is a modeling
   constraint, not an observed guarantee.
4. Solve a multiple-choice knapsack: choose one width per eligible layer to
   minimize the sum of curve costs within the retained-parameter budget.
5. If possible, spend the residual budget on one intermediate, whole-neuron
   boundary width using the best interpolated marginal gain, then fit that
   selected width. Report realized parameters and any unspent budget.

Let $E$ be the eligible layers, $K_i$ their measured width choices,
$c_i(k)$ the monotonicized singleton KL, and $p_i(k)$ retained parameters:

$$
\min_{k_i\in K_i}\sum_{i\in E}c_i(k_i)
\quad\text{subject to}\quad
\sum_{i\in E}p_i(k_i)\leq B.
$$

**Project-proposed surrogate formalization.** The additive KL objective is a
project modeling choice, not a theorem about composed-model KL or an equation
borrowed from MoDeGPT. The knapsack and isotonic procedures are standard
algorithms, not claimed inventions; this formulation needs no external
citation to establish its project definition. For bias-free SwiGLU,
$p_i(k)=3dk$ is standard dimension accounting. Output-biased candidates add
$d$ parameters. The implementation handles integer budget realization.

Here the target $s$ means **eligible-MLP parameters removed**, so the retained
budget is approximately $(1-s)$ of their dense total. S7 targets 20/30/40/50%
removal retain 80/70/60/50%, not 20/30/40/50%. Protected layers, attention,
embeddings, and norms are outside this allocation denominator.

## Evidence and Prior-Work Boundary

**Prior work.** MoDeGPT maps BI scores to nonuniform sparsity through an
entropy-regularized negative-softmax rule. It does not define the singleton-KL
width-curve objective above. [src-modegpt-2025, Section 3.3, Equations 10-11]
The earlier project adaptation remains documented in
[[method-global-to-local-operator-budget-allocation]].

**Empirical finding.** In the
[SwiGLU-5 search](../../../data/results/workflows/model/swiglu-5/search/run-001.json),
discrete-width candidates improved pre-recovery quality and equal-recipe
short-budget recovery over the legacy ranked allocator. The C2 confirmations
at [20%](../../../data/results/workflows/model/swiglu-5/confirmation/run-001-target-0.2.json)
and [50%](../../../data/results/workflows/model/swiglu-5/confirmation/run-001-target-0.5.json)
preserved useful quality through 100M tokens. The 20% selection-label defect
and its limits are recorded in
[`swiglu-5.md`](../../../docs/experiments/model/swiglu/swiglu-5.md).

## Limitations

The solver minimizes the **grid surrogate**, not the true jointly replaced
model's loss. Singleton errors can interact, isotonic pooling removes measured
non-monotonicity, and a boundary cost is interpolated rather than independently
measured at selection time. The boundary extension is a heuristic, not a
global continuous optimum. Fresh local fitting and integrated evaluation remain
necessary; evidence from one model and seed does not establish transfer or
literature novelty.

## Relationships

- [[synthesis-homogeneous-swiglu-findings]] records why this became the
  homogeneous control and where the evidence ends.
- [[method-global-to-local-operator-budget-allocation]] preserves the earlier
  softmax-cap proposal; heterogeneous caps are not validated by these runs.
- [[concept-replacement-error-propagation]] explains the additive surrogate's
  main limitation.
- [[implementation-maintained-mlp-replacement-package]] locates the reusable
  allocator and experiment-specific orchestration.

## Sources

- `src-modegpt-2025` - Section 3.3, Equations 10-11; contrasting prior method.
- Project evidence - linked SwiGLU-5 search and confirmations, checked
  2026-10-02; [SwiGLU-7 preparation](../../../data/results/workflows/model/swiglu-7/prepare-001/result.json).
- Observed implementation -
  [allocation.py](../../../src/mlp_replacement/compression/allocation.py),
  `build_swiglu_width_curve` and `allocate_discrete_swiglu_widths`; fixed recipe
  in [swiglu-7.json](../../../workflows/configs/model/swiglu/swiglu-7.json).
