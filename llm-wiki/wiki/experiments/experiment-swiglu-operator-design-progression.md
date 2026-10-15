---
id: experiment-swiglu-operator-design-progression
title: SwiGLU Operator-Design Progression
summary: Design, prior-work boundary, and accounting of the generic singleton operator comparison (results in docs), plus remaining branch/topology proposals.
type: experiment
status: draft
created: 2026-08-11
updated: 2026-10-15

epistemic:
  role: mixed
  confidence: medium
  verification:
    - experiment-backed

sources:
  - source_id: src-minitron-2024
    locator: "Section 2.2; Section 4.2; Tables 13-14"
    relation: contextualizes
    note: "Prior work on forward-only activation importance for width components."
  - source_id: src-modegpt-2025
    locator: "Section 2.2, Equation 1; Sections 3.1-3.2; Algorithm 1; Appendix A"
    relation: contextualizes
    note: "Prior work on structure-preserving gated-MLP channel selection and down-projection reconstruction."
  - source_id: src-grafting-2025
    locator: "Sections 3.1-3.3 and 4.1; Tables 1-3"
    relation: contextualizes
    note: "Prior work on local operator fitting, integrated recovery, and variable-width replacement comparisons."
  - source_id: src-phase-transitions-compression-2026
    locator: "Overview of compression techniques; When compression becomes catastrophic; Eq. 5; Criticality-aware compression framework"
    relation: contextualizes
    note: "Supporting Perspective motivating capacity curves; its universal PTP and orthogonality claims are not adopted."

related:
  - "[[experiment-initial-block-compression-study]]"
  - "[[experiment-baseline-operator-analysis]]"
  - "[[concept-replacement-error-propagation]]"
  - "[[method-block-importance]]"
  - "[[method-hybrid-operator-replacement]]"
  - "[[source-summary-phase-transitions-compression-2026]]"
  - "[[synthesis-homogeneous-swiglu-findings]]"
supersedes: []
superseded_by: []
---

# SwiGLU Operator-Design Progression

## Overview and Terminology

**Mixed design and single-run evidence.** The generic whole-MLP stage has been
executed over eligible layers 1-22. This page preserves that singleton study.
Aligned teacher-neuron initialization and per-layer width allocation were
subsequently tested in the completed homogeneous family,
[[synthesis-homogeneous-swiglu-findings]]. Arbitrary branch simplification,
projection factorization, and learned nested topology remain proposals. The
rankings below apply only to this page's original model, data, seed, local-fit
budget, isolated replacements, and recovery protocol.

- **Tier I operator:** project-local shorthand for a generic whole-MLP
  substitute selected without teacher-specific internal decisions.
- **Structure-aware operator:** retains or compresses selected teacher SwiGLU
  branches, matrices, or aligned channels.
- **Teacher-tailored operator:** selects topology, ranks, widths, or retained
  components from measurements of a particular teacher block.

A hybrid may be generic when shared across blocks or teacher-tailored when its
parts are selected from block-specific evidence. These are project terms, not
established literature categories.

## Teacher Structure and Nested Direction

For normalized input $x$, model width $d$, and intermediate width
$m=d_{\mathrm{ff}}$, the configured teacher is represented as

$$
f(x)=W_d\left[\operatorname{SiLU}(W_gx)\odot W_ux\right].
$$

**Researcher direction.** A nested candidate may replace compatible branch
operators while preserving their interfaces:

$$
\widehat f(x)=\widehat D\left(\widehat G(x)\odot\widehat U(x)\right).
$$

The first expression is standard gated-MLP architecture notation, not an
original thesis operator; MoDeGPT describes the gated form in Section 2.2,
Equation 1. [src-modegpt-2025, Section 2.2, Equation 1] SiLU is the configured
teacher activation. The second is a project-proposed nested-interface
formalization and needs no citation to define the proposal. Gate and value
outputs must share an intermediate width; the down path returns width $d$.
Replacing SiLU alone saves no material weight storage; compression must alter
a parameterized branch or aligned channels.

Intermediate channel $i$ spans row $i$ of $W_g$, row $i$ of $W_u$, and column
$i$ of $W_d$. Candidate forward-only diagnostics include post-gating activity
and projected contribution magnitude, but a diagnostic becomes an importance
estimator only if it predicts held-out ablation or model-quality effects.

## Prior-Work Boundary

- Minitron uses activation statistics for structured width pruning; it does not
  learn a distinct nested replacement for each MLP. [src-minitron-2024,
  Section 2.2; Section 4.2]
- MoDeGPT selects gated-MLP intermediate channels and reconstructs the down
  projection, preserving the gated family rather than searching arbitrary
  nested branches. [src-modegpt-2025, Sections 3.1-3.2; Algorithm 1]
- Grafting supplies the local-fit then integrated-recovery pattern and
  variable-width comparisons, but its reported setting is diffusion
  transformers. [src-grafting-2025, Sections 3.1-3.3 and 4.1]
- The Phase Transitions Perspective motivates capacity curves and combined
  compression axes; this project does not adopt its proposed universal
  transition or orthogonality claims as axioms.
  [src-phase-transitions-compression-2026, "When compression becomes
  catastrophic"]

**Synthesis.** Combining block-specific component diagnostics with a learned
nested topology remains a project proposal whose novelty requires a dedicated
literature search.

## Objective, Stages, and Hypotheses

| Stage | Intervention | Purpose |
| --- | --- | --- |
| 0. Fixed references | Original, zero, mean, linear, affine, fixed SwiGLU | Establish common controls. |
| 1. Generic whole-MLP | Factorized linear, compact MLP, reduced SwiGLU, hybrid | Compare operator families. |
| 2. Structure-aware | Channel selection, projection factorization, branch simplification | Test teacher-topology retention. |
| 3. Teacher-tailored | Block-specific components, ranks, widths, or topology | Test evidence-guided customization. |
| 4. Recovery | Equal replacement-only recovery for shortlisted candidates | Test whether integration changes the ordering. |

**Researcher hypotheses; unverified.** At matched footprint, preserving useful
teacher structure may beat a teacher-agnostic substitute; importance-guided
aligned-channel selection may beat random selection; and a teacher-tailored
candidate may improve on the best generic family before or after equal
recovery. The executed run tests only Stage 1.

## Operator Accounting and Comparison Rules

Let $r_L$ be linear rank and $r_N$ nonlinear width:

| Candidate | Form | Bias-free parameters |
| --- | --- | ---: |
| Dense linear | $Ax$ | $d^2$ |
| Dense affine | $Ax+b$ | $d^2+d$ |
| Low-rank affine | $U(Vx)+b$ | $2dr_L+d$ |
| Compact MLP | $W_2\phi(W_1x)$ | $2dr_N$ |
| Reduced SwiGLU | $W_d[\operatorname{SiLU}(W_gx)\odot W_ux]$ | $3dr_N$ |
| Low-rank + MLP | $U(Vx)+W_2\phi(W_1x)$ | $2d(r_L+r_N)$ |
| Low-rank + SwiGLU | $U(Vx)+G_{\mathrm{SwiGLU}}(x)$ | $2dr_L+3dr_N$ |

These are standard dimension counts and require no citation. Exact experiments
must report realized parameters and bytes. Three comparisons serve different
purposes: matched-footprint architecture controls, within-family capacity
curves, and later cross-family Pareto analysis. Equal parameter count does not
imply equal expressivity: at $r_L=d/2$, $U(Vx)$ stores $d^2$ parameters but is
still rank-constrained. This is a project comparison rule, not prior-work
prescription.

## Configuration, Inputs, and Procedure

The recorded configuration and the ten candidates are in
[`operator.md`](../../../docs/experiments/block/operator/operator.md)
(operator design space and configuration appendix).

Each candidate was fitted independently from dense-teacher activations and
inserted at one layer at a time. The comparison records local fidelity,
realized footprint, pre-recovery KL, and post-recovery KL under the same
protocol. Learned fits report their stopping behavior; closed-form fits report
their solver and regularization.

Inputs are [`operator.ipynb`](../../../notebooks/block/operator/operator.ipynb),
[`analysis/sensitivity.py`](../../../src/mlp_replacement/analysis/sensitivity.py),
and the baseline conventions in
[[experiment-baseline-operator-analysis]]. The draft baseline runner is not yet
the validated executor of these notebook results.

## Direct Results and Interpretation

**Empirical findings** from one configured Stage 1 run (artifact
`operator-experiments.json`, schema 1) are recorded in the Results section of
[`operator.md`](../../../docs/experiments/block/operator/operator.md). In short,
under this budget a nonlinear operator beat the linear references, and good
local approximation did not imply a globally safe replacement.

Interpretation must separate architecture from footprint, local fit from model
effect, singleton sensitivity from intrinsic layer importance, and
pre-recovery ranking from repairability.

## Planned Extensions

The remaining branch/factorization/nested-topology proposals are unverified;
teacher-subset homogeneous widths are already studied separately. Future
internal interventions should compare importance-guided choices with random
and simple activation controls at matched retained width, and evaluate
shortlisted candidates at matched or nearest-feasible footprint. Capacity
boundaries should be called degradation knees or operational boundaries unless
a repeatable sharp transition is demonstrated.

Factorization, replacement, and quantization mechanisms must first be measured
separately; their errors are not assumed additive or orthogonal.

## Limitations and Reproducibility

Result-specific limits and the artifact contents are in
[`operator.md`](../../../docs/experiments/block/operator/operator.md). Design
limits:

- One model, seed, and data partition do not establish stability or transfer.
- Branch and channel effects may interact, so singleton component scores need
  not compose.
- Every result in this artifact replaces one MLP independently; it does not
  measure simultaneous cross-family interactions. Separate homogeneous
  multi-block evidence does not answer that broader question.

Status: draft and experiment-backed for Stage 1 only.

## Relationships

- [[experiment-initial-block-compression-study]] supplies the umbrella stage
  map and fixed references.
- [[experiment-baseline-operator-analysis]] supplies calibration, width,
  recovery, and singleton-sensitivity conventions.
- [[concept-replacement-error-propagation]] motivates the next multi-block
  interaction stage.
- [[method-block-importance]] distinguishes forward-only screening from
  operator-conditioned KL sensitivity.
- [[method-hybrid-operator-replacement]] defines the generic hybrid family.
- [[source-summary-phase-transitions-compression-2026]] records the capacity-
  curve source and qualifications.
- [[synthesis-homogeneous-swiglu-findings]] records the later completed
  teacher-initialized, width-allocated homogeneous family without changing
  this historical singleton artifact.

## Sources

- `src-minitron-2024` - Section 2.2, Section 4.2, Tables 13-14
- `src-modegpt-2025` - Section 2.2, Equation 1; Sections 3.1-3.2,
  Algorithm 1, Appendix A
- `src-grafting-2025` - Sections 3.1-3.3 and 4.1, Tables 1-3
- `src-phase-transitions-compression-2026` - overview, proposed transition
  modeling, and criticality-aware framework; supplement pending
- Project-generated evidence -
  `data/results/notebook-block-study/operator-experiments.json`, schema version
  1, created and checked 2026-08-20
