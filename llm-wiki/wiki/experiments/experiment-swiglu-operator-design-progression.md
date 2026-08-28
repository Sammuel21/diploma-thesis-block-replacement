---
id: experiment-swiglu-operator-design-progression
title: SwiGLU Operator-Design Progression
summary: Records an artifact-backed full-depth comparison of generic whole-MLP substitutes and the planned progression toward teacher-tailored SwiGLU compression.
type: experiment
status: draft
created: 2026-08-11
updated: 2026-08-20

authorship:
  created_by: collaborative
  contributors:
    - researcher
    - llm

epistemic:
  role: mixed
  confidence: medium
  verification:
    - experiment-backed

scope:
  topics:
    - activation-analysis
    - capacity-sweep
    - compression-boundary
    - hybrid-operator
    - mlp-block-replacement
    - neuron-importance
    - operator-comparison
    - replacement-sensitivity
    - structured-pruning
    - swiglu
    - teacher-tailored-operator
  granularities:
    - neuron
    - mlp-block
    - transformer-layer
    - model
    - cross-level
  pipeline_stages:
    - screening
    - selection
    - replacement
    - integration
    - evaluation
    - analysis

sources:
  - source_id: src-minitron-2024
    locator: "Section 2.2; Section 4.2; Tables 13-14"
    relation: contextualizes
    note: "Prior work on forward-only activation importance for width components."
  - source_id: src-modegpt-2025
    locator: "Sections 3.1-3.2; Algorithm 1; Appendix A"
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
supersedes: []
superseded_by: []
---

# SwiGLU Operator-Design Progression

## Overview and Terminology

**Mixed design and single-run evidence.** The generic whole-MLP stage has been
executed over eligible layers 1-22. Structure-aware and teacher-tailored
operators remain planned. The recorded rankings apply only to the configured
model, data, seed, local-fit budget, isolated replacements, and recovery
protocol.

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

These are explanatory/project formalizations and require no citation. Gate and
value outputs must have the same intermediate width, while the down path must
return width $d$. Replacing SiLU alone saves no material weight storage;
compression must alter a parameterized branch or aligned channels.

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

| Component | Recorded setting |
| --- | --- |
| Model | SmolLM2-1.7B, revision `effd688a12921b4cc83e3312b6feb579f70f9c71` |
| Layers | 1-22; first and final excluded |
| Candidates | zero, mean, linear, affine, rank-$0.5d$ factorized linear, width-$0.5d$ compact MLP, SwiGLU 0.25/0.50/0.75, rank/width-$0.25d$ hybrid |
| Data | 48 C4 calibration, 24 C4 local-validation, and 24 WikiText-2 KL batches |
| Learned fit | at most 64 epochs; minibatch 2,048; learning rate $10^{-3}$; patience 3 |
| Recovery | 64 streamed updates; temperature 1; learning rate $10^{-5}$; replacement parameters only |
| Seed | 21 |

Each candidate was fitted independently from dense-teacher activations and
inserted at one layer at a time. The comparison records local fidelity,
realized footprint, pre-recovery KL, and post-recovery KL under the same
protocol. Learned fits report their stopping behavior; closed-form fits report
their solver and regularization.

Inputs are [`operator.ipynb`](../../../notebooks/block/operator.ipynb),
[`analysis/sensitivity.py`](../../../src/mlp_replacement/analysis/sensitivity.py),
and the baseline conventions in
[[experiment-baseline-operator-analysis]]. The draft baseline runner is not yet
the validated executor of these notebook results.

## Direct Results and Interpretation

The following are **empirical findings from one configured run** in
[`operator-experiments.json`](../../../data/results/notebook-block-study/operator-experiments.json),
schema version 1. Values average over independent singleton replacements at
layers 1-22.

| Operator | Original-MLP parameters | Mean pre-KL | Mean post-KL |
| --- | ---: | ---: | ---: |
| zero | 0% | 0.7354 | 0.7354 |
| mean | 0% | 0.7365 | 0.7365 |
| linear | 8.33% | 0.5960 | 0.5962 |
| affine | 8.34% | 0.5912 | 0.5907 |
| rank-$0.5d$ factorized linear | 8.33% | 0.5985 | 0.5979 |
| width-$0.5d$ compact MLP | 8.33% | 0.3466 | 0.3371 |
| rank/width-$0.25d$ hybrid | 8.33% | 0.4956 | 0.4804 |
| 0.25-width SwiGLU | 25% | 0.2007 | 0.1943 |
| 0.50-width SwiGLU | 50% | 0.1766 | 0.1608 |
| 0.75-width SwiGLU | 75% | 0.1684 | 0.1413 |

- In the approximately-$d^2$ group, the compact MLP had the lowest post-KL at
  18 of 22 layers and beat dense linear at 21 of 22. This is evidence for a
  nonlinearity benefit under this budget, not a universal family ranking.
- Wider SwiGLU generally reduced KL, but post-recovery width order was
  non-monotonic at three layers.
- Mean learned-operator post-KL was concentrated at layers 22 (2.4374), 1
  (2.2194), and 7 (1.7966); the next layer mean was 0.1965.
- Learned-row local relative MSE and pre-KL had Spearman association -0.09;
  good local approximation did not imply globally safe replacement.
- Pre/post layer rankings were nearly unchanged under 64 recovery updates
  (per-family Spearman 0.993-1.000), making pre-KL a useful ranking proxy only
  under this protocol.
- At layer 22, zero and mean had lower KL than every learned substitute. This
  is an anomaly requiring confirmation.

Interpretation must separate architecture from footprint, local fit from model
effect, singleton sensitivity from intrinsic layer importance, and
pre-recovery ranking from repairability.

## Planned Extensions

Structure-aware and teacher-tailored stages remain unverified. They should
change one internal intervention at a time, compare any importance-guided
choice with random and simple activation controls at the same retained width,
and evaluate shortlisted candidates at matched or nearest-feasible footprint.
Capacity boundaries should be called degradation knees or operational
boundaries unless a repeatable sharp transition is demonstrated.

Factorization, replacement, and quantization mechanisms must first be measured
separately; their errors are not assumed additive or orthogonal.

## Limitations and Reproducibility

- One model, seed, and data partition do not establish stability or transfer.
- Many learned fits reached the 64-epoch ceiling, so comparisons are
  fixed-budget rather than demonstrated-convergence results.
- The artifact records KL but not full-model loss/perplexity for every case.
- Zero and mean have no trainable parameters; their recorded 64 recovery
  updates denote shared-stream traversal, not optimizer updates.
- Branch and channel effects may interact, so singleton component scores need
  not compose.
- Every result replaces one MLP independently; multi-block behavior remains
  unknown.

Status: draft and experiment-backed for Stage 1 only. The checked local,
ignored schema-1 artifact contains 220 unique fitting rows and 220 matching
sensitivity rows with verified `pre - post` KL arithmetic. It excludes fitted
weights, activations, logits, optimizer state, histories, perplexity, and
hardware telemetry. No independent seed or model replication exists.

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

## Sources

- `src-minitron-2024` - Section 2.2, Section 4.2, Tables 13-14
- `src-modegpt-2025` - Sections 3.1-3.2, Algorithm 1, Appendix A
- `src-grafting-2025` - Sections 3.1-3.3 and 4.1, Tables 1-3
- `src-phase-transitions-compression-2026` - overview, proposed transition
  modeling, and criticality-aware framework; supplement pending
- Project-generated evidence -
  `data/results/notebook-block-study/operator-experiments.json`, schema version
  1, created and checked 2026-08-20
