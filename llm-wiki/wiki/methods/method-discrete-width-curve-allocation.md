---
id: method-discrete-width-curve-allocation
title: Discrete Width-Curve Allocation
summary: Allocates homogeneous SwiGLU widths by minimizing a summed singleton-KL surrogate under a shared eligible-MLP parameter budget.
type: method
status: review
created: 2026-10-02
updated: 2026-10-17

epistemic:
  role: mixed
  confidence: medium
  verification:
    - source-checked
    - experiment-backed

sources:
  - source_id: src-spdy-2022
    locator: "Section 3.1, Eq. (1); Section 3.2, Eqs. (2)-(3), Algorithm 1"
    relation: motivates
    note: "Inspiration for the allocation formulation: exact dynamic program over discrete per-layer options under a summed budget with additive per-layer error. Locators from an automated HTML fetch, not yet confirmed in the PDF."
  - source_id: src-ziplm-2023
    locator: "Sections 3.1-3.3"
    relation: motivates
    note: "Inspiration as the closest prior application to structured FFN width: per-layer database with layer-wise reconstruction, SPDY-based search, then distillation. Locators from an automated HTML fetch, not yet confirmed in the PDF."
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

The allocation formulation is not a project invention. It solves the same
additive, budget-constrained per-layer choice as SPDY's dynamic program, and
ZipLM applies that kind of allocation to structured FFN width
[src-spdy-2022, Sections 3.1-3.2; src-ziplm-2023, Section 3.2]. The researcher
records both as inspirations. The project-specific parts are the choice set
(widths of locally fitted SwiGLU replacements), the per-option cost (measured
singleton teacher KL, smoothed by isotonic regression), and the parameter
budget. The working note
[Width allocation as an integer program](../../../docs/methodology/width-allocation-integer-program.md)
gives the full formulation and the evidence.

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

**Equation status.** The constrained problem is a multiple-choice knapsack,
a standard integer program, and the implementation solves it exactly with the
standard dynamic program over budget units of $d$ parameters. As a per-layer
compression allocation it follows SPDY's formulation, which is cited
[src-spdy-2022, Section 3.1, Eq. (1); Section 3.2, Eqs. (2)-(3)]. Choosing
$c_i(k)$ as the monotonicized singleton teacher KL, and therefore treating
joint model KL as the sum of singleton KLs, is a **project-proposed modeling
choice**. It is not a theorem about composed-model KL and is not borrowed from
MoDeGPT. Isotonic regression is a standard algorithm; a textbook reference is
not yet registered. For bias-free SwiGLU, $p_i(k)=3dk$ is standard dimension
accounting. Output-biased candidates add $d$ parameters. The implementation
handles integer budget realization.

Here the target $s$ means **eligible-MLP parameters removed**, so the retained
budget is approximately $(1-s)$ of their dense total. S7 targets 20/30/40/50%
removal retain 80/70/60/50%, not 20/30/40/50%. Protected layers, attention,
embeddings, and norms are outside this allocation denominator.

## Evidence and Prior-Work Boundary

**Prior work.** SPDY chooses one sparsity level per layer from a discrete
set, minimizing summed per-layer error under a summed runtime budget, and
solves this exactly with a dynamic program over discretized time. It names
error additivity as a strong assumption. Its per-layer error is a parametric
curve whose coefficients an outer random search tunes against the loss of the
stitched model. [src-spdy-2022, Sections 3.1-3.3] ZipLM prunes attention heads
and FFN intermediate neurons with a layer-wise reconstruction update, stores
several pruned versions per layer, chooses levels with an SPDY-based search
under a runtime budget, and then distills. [src-ziplm-2023, Sections 3.1-3.3]

**Synthesis.** Relative to these, the project allocator uses a parameter
budget instead of runtime, a directly measured model-level cost per option
instead of a parametric or layer-local error, isotonic smoothing, and a single
dynamic-program solve with no outer search. Its homogeneous use (teacher
neuron selection plus local fitting) is close to structured width pruning
with layer-wise reconstruction, so the word "replacement" does not by itself
separate it from ZipLM.

MoDeGPT maps BI scores to nonuniform sparsity through an
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

**Empirical finding.** The additive objective underestimates measured joint
damage. Offline comparisons of the allocator's predicted sum with the joint
model's token-0 KL in SwiGLU-7 and SwiGLU-8, and the block-interaction
study's window residuals, both show super-additive damage. The numbers and
their caveats are in the
[width-allocation note](../../../docs/methodology/width-allocation-integer-program.md#5-the-additive-singleton-kl-objective).
Whether the sum still ranks allocations at one budget correctly has not been
tested.

## Limitations

The solver minimizes the **grid surrogate**, not the true jointly replaced
model's loss. Singleton errors can interact, isotonic pooling removes measured
non-monotonicity, and a boundary cost is interpolated rather than independently
measured at selection time. The boundary extension is a heuristic, not a
global continuous optimum. Fresh local fitting and integrated evaluation remain
necessary; evidence from one model and seed does not establish transfer or
literature novelty.

**Open question.** Later work in the same research line replaces the additive
dynamic program with whole-model search. Those papers (EvoPress, DarwinLM)
are not registered, so this page makes no claim from them; the
width-allocation note records them as unregistered leads.

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

- `src-spdy-2022` - Section 3.1, Eq. (1); Section 3.2, Eqs. (2)-(3),
  Algorithm 1; Section 3.3; inspiration for the allocation formulation.
- `src-ziplm-2023` - Sections 3.1-3.3; inspiration and closest prior
  application to structured FFN width.
- `src-modegpt-2025` - Section 3.3, Equations 10-11; contrasting prior method.
- Project evidence - linked SwiGLU-5 search and confirmations, checked
  2026-10-02; [SwiGLU-7 preparation](../../../data/results/workflows/model/swiglu-7/prepare-001/result.json).
- Observed implementation -
  [allocation.py](../../../src/mlp_replacement/compression/allocation.py),
  `build_swiglu_width_curve` and `allocate_discrete_swiglu_widths`; fixed recipe
  in [swiglu-7.json](../../../workflows/configs/model/swiglu/swiglu-7.json).
