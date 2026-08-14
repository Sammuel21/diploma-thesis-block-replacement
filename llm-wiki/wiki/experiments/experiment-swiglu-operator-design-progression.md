---
id: experiment-swiglu-operator-design-progression
title: SwiGLU Operator-Design Progression
summary: Defines a staged experiment from generic whole-MLP substitutes to structure-aware and teacher-tailored SwiGLU compression.
type: experiment
status: draft
created: 2026-08-11
updated: 2026-08-13

authorship:
  created_by: collaborative
  contributors:
    - researcher
    - llm

epistemic:
  role: mixed
  confidence: not-assessed
  verification:
    - unverified

scope:
  topics:
    - activation-analysis
    - capacity-sweep
    - compression-boundary
    - hybrid-operator
    - mlp-block-replacement
    - neuron-importance
    - operator-comparison
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
    note: "Supporting Perspective motivating capacity curves and combined-axis analysis; its universal PTP and orthogonality claims are not adopted."

related:
  - "[[experiment-initial-block-compression-study]]"
  - "[[method-hybrid-operator-replacement]]"
  - "[[source-summary-phase-transitions-compression-2026]]"
supersedes: []
superseded_by: []
---

# SwiGLU Operator-Design Progression

## Overview and Status

**Collaborative experiment design; unverified.** This page turns the operator
classes sketched in `operator.ipynb` and the researcher's nested-operator idea
into a staged single-block study. It begins with generic drop-in substitutes,
then retains and modifies parts of the teacher SwiGLU, and only afterwards
constructs block-specific operators from component evidence.

No experiment currently establishes that a teacher-tailored operator is
smaller or better than a generic reduced-width SwiGLU. The progression is a
research plan, not a ranking of methods.

## Terminology

The following project terms distinguish two independent design choices:

- **Tier I operator:** project-local shorthand for a generic whole-MLP
  substitute whose family and dimensions are selected without teacher-specific
  internal component decisions. This is not established literature terminology.
- **Whole-MLP replacement:** substitute the complete MLP contribution through
  its existing model-width input-output interface.
- **Structure-aware compression:** preserve some of the teacher's SwiGLU
  branches or matrices while pruning, factorizing, or replacing selected
  internal components.
- **Teacher-tailored operator:** choose the architecture, retained components,
  ranks, or widths from measurements of a particular teacher block.

"Teacher-tailored" is preferred over "advanced" because it describes the
source of the design decision. "Personalized operator" remains an informal
synonym, not a claimed standard literature term.

These categories are not mutually exclusive. A hybrid is a generic operator
class when every block receives the same architecture, but becomes
teacher-tailored when its branch sizes or topology are selected separately
from block-specific evidence.

## Teacher SwiGLU as a Composite Operator

For the configured teacher block, omit biases and write the normalized MLP
input as $x$, model width as $d$, and teacher intermediate width as $m$:

$$
\begin{aligned}
g(x) &= \operatorname{SiLU}(W_gx), \\
u(x) &= W_ux, \\
z(x) &= g(x)\odot u(x), \\
f(x) &= W_dz(x).
\end{aligned}
$$

Here $g,u,z\in\mathbb{R}^{m}$ and $f(x)\in\mathbb{R}^{d}$. This is an
explanatory formalization of the configured teacher interface, not a new
scientific method, and requires no citation.

### What nested replacement means

**Researcher direction.** A compatible nested substitute can replace branch
operators while preserving their interfaces:

$$
\widehat f(x)=\widehat D\left(\widehat G(x)\odot\widehat U(x)\right).
$$

This project-proposed family requires no citation. It expresses composition,
not a claim that the displayed factorization is optimal or identifiable.

The interfaces impose concrete constraints:

- $\widehat G$ and $\widehat U$ must produce the same intermediate width for
  the elementwise product;
- $\widehat D$ must accept that width and return model width $d$;
- changing intermediate width requires coordinated changes to the gate path,
  value path, and down path; and
- replacing only the parameter-free SiLU function does not materially reduce
  stored weights. Compression must normally alter its surrounding gate branch
  or another parameterized component.

Consequently, "replace the nonlinear part" should specify whether it means the
SiLU function alone, the complete gate branch $\operatorname{SiLU}(W_gx)$, or
the gated interaction $z(x)$. These are different interventions.

## Aligned SwiGLU Neurons

Intermediate coordinate $i$ is an aligned channel across three parameter
surfaces: row $i$ of $W_g$, row $i$ of $W_u$, and column $i$ of $W_d$. Its
contribution before summation is

$$
c_i(x)=W_d[:,i]z_i(x),
\qquad
f(x)=\sum_{i=1}^{m}c_i(x).
$$

This is an explanatory decomposition derived from the teacher equation and
requires no citation. Removing or resizing a channel must update all three
aligned surfaces; independently deleting only one row or column would break
the intended channel interface.

Candidate diagnostics include:

- gate activation magnitude, variance, or near-zero frequency;
- post-gating magnitude or variance of $z_i$;
- projected contribution magnitude $\lVert c_i(x)\rVert$;
- held-out local error after ablating one channel or a selected channel set;
- downstream model-loss change after an internal intervention; and
- gradient-based sensitivity as a more expensive optional comparison.

The first three are forward-only descriptive signals. None is automatically
"importance" until it predicts a declared ablation or model-quality outcome.
Gate saturation or low gate frequency alone can describe behavior without
showing that the channel is dispensable.

**Project-proposed diagnostic.** One initial contribution score could be

$$
S_i^{\mathrm{contrib}}
=
\mathbb{E}_{x\sim\mathcal{C}}
\left[\lVert c_i(x)\rVert_2^2\right].
$$

This score is a researcher-project definition and requires no citation. It
must be compared with actual held-out ablation effects before being treated as
a useful selector. Scores for individual channels also do not predict the
joint error of removing a set because their output contributions can interact
when summed.

## Prior-Work Boundary

- Minitron uses forward-pass activation statistics to rank MLP neurons and
  other width components for structured pruning. It does not learn a nested
  replacement architecture for each individual MLP block.
  [src-minitron-2024, Section 2.2; Section 4.2]
- MoDeGPT is closer to internal SwiGLU surgery: for its gated-MLP module, it
  selects post-gating intermediate channels using calibration correlations and
  ridge leverage scores, then recomputes the down projection. It preserves the
  gated operator family rather than searching arbitrary nested branches.
  [src-modegpt-2025, Sections 3.1-3.2; Algorithm 1]
- Grafting fits replacement operators through local activation regression and
  then repairs the integrated model. Its variable-width MLP comparisons
  motivate a controlled operator screen, but its evidence is from diffusion
  transformers and does not establish a teacher-tailored causal-LLM design.
  [src-grafting-2025, Sections 3.1-3.3 and 4.1]
- The Phase Transitions Perspective organizes pruning, quantization, and
  low-rank decomposition as structural, numerical, and algebraic redundancy
  axes and fits compression-quality curves to proposed transition points. It
  motivates broad capacity sweeps, but does not establish a universal boundary
  for learned MLP replacement. [src-phase-transitions-compression-2026,
  "Overview of compression techniques"; "When compression becomes
  catastrophic"]

**Synthesis.** Existing project sources therefore support activation-guided
width pruning, internal structured decomposition, and whole-operator fitting
as neighboring approaches. Combining component diagnostics with a learned,
block-specific nested topology remains a thesis proposal whose literature
novelty has not been established by a systematic search.

## Objective

The experiment asks whether progressively using more of the teacher's internal
structure improves the footprint-quality trade-off over generic whole-MLP
replacement at one block.

The immediate questions are:

1. Which elementary whole-MLP operator families are credible references at
   comparable actual parameter counts?
2. Which SwiGLU branches or aligned channels are sensitive to simplification?
3. Does retaining and selectively compressing teacher structure outperform a
   complete substitute at matched footprint?
4. Can component evidence choose a better block-specific operator than a
   uniform architecture rule?
5. Does each Tier I family degrade smoothly with reduced capacity, or is there
   a repeatable knee beyond which model quality deteriorates rapidly?

## Staged Design Space

This is a research progression, not a universal complexity hierarchy:

| Stage | Intervention | Purpose |
| --- | --- | --- |
| 0. Controls and fixed reference | Original, zero, mean, dense linear, affine, and fixed reduced-width SwiGLU | Establish degradation and practical reference levels. |
| 1. Generic whole-MLP operators | Low-rank affine, compact ungated MLP, reduced-width SwiGLU, and hybrid | Compare broad operator families through the same drop-in interface. |
| 2. Structure-aware internal operators | Aligned channel pruning, projection factorization, constant or compressed gate path, and selected branch replacement | Determine whether preserving teacher topology is beneficial. |
| 3. Teacher-tailored composition | Choose retained channels, branch family, rank, or width from block-specific diagnostics | Test whether measured internal structure supports a better custom design. |
| 4. Model-level recovery | Apply equal recovery only to shortlisted candidates | Determine whether pre-recovery advantages survive lightweight integration repair. |

Stage 1 should precede Stage 3. Otherwise a complex custom construction could
appear successful without showing that it improves upon a simpler operator at
the same footprint.

## Operator-Family Accounting

Let $r_L\leq d$ be a linear rank and let $r_N$ be a nonlinear intermediate
width. Unlike a matrix rank, $r_N$ need not be smaller than $d$; it is normally
interpreted relative to the teacher width $m=d_{\mathrm{ff}}$.

| Candidate | Representative form | Bias-free parameter scale |
| --- | --- | ---: |
| Dense linear | $Ax$ | $d^2$ |
| Dense affine | $Ax+b$ | $d^2+d$ |
| Low-rank affine | $U(Vx)+b$ | $2dr_L+d$ |
| Compact ungated MLP | $W_2\phi(W_1x)$ | $2dr_N$ |
| Reduced-width SwiGLU | $W_d[\operatorname{SiLU}(W_gx)\odot W_ux]$ | $3dr_N$ |
| Low-rank plus ungated correction | $U(Vx)+W_2\phi(W_1x)$ | $2d(r_L+r_N)$ |
| Low-rank plus SwiGLU correction | $U(Vx)+G_{\mathrm{SwiGLU}}(x)$ | $2dr_L+3dr_N$ |
| Internal teacher factorization | Factorize selected $W_g$, $W_u$, or $W_d$ | design-specific |

These are standard matrix-dimension counts and require no citation. Exact
experiments must count configured biases, buffers, state elements, weight
bytes, and serialized size rather than relying only on the formulas.

## Tier I Research Axes

**Project experiment design; unverified.**

The first operator study should vary one scientific axis at a time rather than
run the full Cartesian product. The following axes answer different questions:

| Axis | Question | First controlled comparison |
| --- | --- | --- |
| Operator family | Does topology matter beyond footprint? | Compare dense linear, affine, low-rank, compact MLP, reduced SwiGLU, and hybrid candidates at the nearest feasible matched parameter count. |
| Capacity | How does each family fail as it becomes smaller? | Sweep nonlinear width $r_N$, linear rank $r_L$, or another family-specific size variable from mild to aggressive compression. |
| Calibration data | How sample-efficient is local fitting? | Vary activation-pair count while fixing optimizer-update count; study more updates as a separate optimization-budget axis. |
| Layer | Does the ranking transfer across depth? | Begin at layer 11, then repeat shortlisted settings at a small set of internal layers while leaving protected edge layers untouched. |
| Recovery | Does a local advantage survive integration repair? | Report every shortlisted operator before recovery and after the same model-level recovery protocol. |
| Randomness | Is a near-boundary result stable? | Repeat only shortlisted and knee-adjacent learned candidates across seeds and calibration splits. |
| Composition | Do distinct compression mechanisms interact? | Combine replacement with factorization or quantization only after their separate curves are understood. |

Width retention is not parameter retention across families. For example, dense
$Ax$ has no hidden-width control, low-rank linear capacity is controlled by
$r_L$, and reduced SwiGLU capacity is controlled by $r_N$ relative to
$d_{\mathrm{ff}}$. Family comparisons must therefore use realized parameter
count or bytes on the footprint axis rather than equating their raw width or
rank fractions.

### Coarse-to-boundary procedure

1. On one fixed internal block, hold the data split, fitting budget, and
   evaluation protocol constant and collect a coarse capacity curve for each
   Tier I family.
2. Define the **operational boundary** before examining held-out results: the
   smallest feasible candidate satisfying declared local-fidelity and
   model-quality tolerances.
3. Add measurements only around an observed bend or decision boundary; do not
   spend equal compute densely sampling obviously safe or collapsed regions.
4. Recheck the boundary on selected internal layers, calibration splits, and
   seeds before treating it as a stable property.
5. Apply equal model-level recovery to the shortlist and test whether the
   pre-recovery ordering remains.

The operational-boundary rule above is a **project-proposed experimental
definition** and requires no citation. The source-derived piecewise PTP model
would require citation if adopted. Until a sharp change is observed and shown
to be robust, use "degradation knee" or "operational boundary" rather than
claiming a phase transition. [src-phase-transitions-compression-2026,
"Quantitative phase transition modeling," Eq. 5]

### Mechanism axes are not automatically independent

**Synthesis.** A whole-MLP replacement is primarily structural; a low-rank
replacement also invokes algebraic compression; quantizing a replacement is
numerical; and a hybrid can mix these mechanisms. The Perspective's taxonomy
is therefore useful for describing interventions, but its claim that the axes
are orthogonal and their errors additive is not an axiom for this project.
Interaction must be measured through paired single-method and combined-method
controls. [src-phase-transitions-compression-2026, "Theoretical
orthogonality"]

## Research Hypotheses

**Researcher hypotheses; unverified.** The following propositions are suitable
for falsification:

- At matched actual parameter count, a structure-aware internal intervention
  retains better local and model quality than a teacher-agnostic whole-MLP
  substitute.
- Importance-guided aligned-channel selection outperforms random and uniform
  channel removal under the same retained width.
- Blocks whose outputs are poorly approximated by dense affine maps benefit
  more from retained gating or nonlinear branch capacity.
- A teacher-tailored operator improves over the best generic candidate before
  recovery, but that ordering may change after equal model-level recovery.

None of these hypotheses currently has project evidence.

## Configuration and Inputs

The initial scope remains deliberately narrow:

- model: `HuggingFaceTB/SmolLM2-1.7B`;
- first target: zero-based MLP layer 11;
- teacher input-output interface: the frozen native block implementation;
- calibration and held-out activation splits: shared across candidates;
- complete-model validation batches: shared across candidates;
- first comparison: no model-level recovery; and
- primary comparison basis: actual parameters and weight bytes, with
  serialized size recorded when candidates are materialized.

Relevant project inputs are:

- [`operator.ipynb`](../../../notebooks/block/operator.ipynb), which records
  the elementary operator classes;
- [`activation-analysis.ipynb`](../../../notebooks/block/activation-analysis.ipynb),
  which provides descriptive activation geometry;
- [`baseline-testing.ipynb`](../../../notebooks/block/baseline-testing.ipynb),
  which provides the fixed single-block reference protocol; and
- [`baseline-experiments.ipynb`](../../../notebooks/block/baseline-experiments.ipynb),
  which begins calibration-budget and reduced-width SwiGLU sweeps; and
- [`src/mlp_replacement/baselines.py`](../../../src/mlp_replacement/baselines.py),
  the draft runner intended to produce the fixed reference rows from a
  caller-prepared model, block, activation pairs, and validation loader.

The runner supplies baseline calculation only. It does not own model or data
loading, activation capture, experimental candidate fitting, plotting,
recovery, or artifact writing. It has not yet been integrated into these
notebooks or runtime-validated, so experiments must not claim shared-runner
comparability until that integration and parity check are complete.

## Procedure and Controls

1. Reuse the fixed single-block capture and evaluation protocol rather than
   changing data, layer, and metric conventions for each operator. Once the
   reusable baseline runner passes parity validation, call it within the same
   prepared experiment context and compare additional candidates with its
   standardized rows.
2. Screen generic whole-MLP candidates at a small declared set of actual
   parameter budgets. Do not infer an operator-family advantage from
   unmatched footprints.
3. Analyze one internal intervention at a time: aligned channel selection,
   gate-path simplification, or projection factorization. Keep untouched
   teacher components frozen so the changed component is identifiable.
4. Compare every importance-guided selection with random selection and a
   simple activation-magnitude reference at the same retained width.
5. Construct a teacher-tailored candidate only from rules declared before its
   held-out model evaluation. Compare it with the best generic and
   structure-aware candidates at a reconciled footprint.
6. Apply model-level recovery only after the pre-recovery comparison. Give
   shortlisted candidates equal recovery data, loss, trainable scope,
   optimizer, and step budget.

Learned candidates should report initialization seed, realized optimizer
steps, and early-stopping behavior. Closed-form candidates should report their
calibration size, numerical precision, regularization, and solver.

## Metrics

Local measurements:

- held-out MSE and relative MSE;
- cosine similarity and output-norm ratio;
- token-relative error summaries;
- component or channel ablation error; and
- train-validation behavior for learned nested components.

Model and footprint measurements:

- validation loss, perplexity, and deltas from the untouched teacher;
- replacement and removed parameter counts;
- theoretical weight bytes and actual serialized size; and
- optional controlled memory measurements under the project's primary
  evaluation decision.

Component scores should additionally be evaluated by rank correlation or
selection overlap with held-out ablation outcomes. A visually plausible score
distribution is not evidence that the score selects compressible components.

## Direct Results and Interpretation

No direct results are recorded. Notebook output, exploratory conversation, and
the presence of an implementation class do not by themselves make this page
experiment-backed.

Future interpretation must distinguish:

- a generic operator-family effect from a parameter-budget effect;
- a structure-preservation effect from additional teacher weight reuse;
- an importance-selection effect from the capacity of the resulting operator;
- local activation fidelity from complete-model quality; and
- pre-recovery ranking from ranking after equal model-level recovery.

## Limitations and Open Questions

- "Native MLP" is model-specific. SwiGLU is the configured target here, not a
  universal architecture used by every modern language model.
- Internal interventions have different feasible parameter grids, so exact
  footprint matching may require nearest-feasible comparisons.
- Branches are coupled by multiplication. An isolated gate or value statistic
  may misrepresent the importance of their product.
- Individual neuron contributions can cancel or reinforce after the down
  projection; singleton scores need not compose under multi-neuron removal.
- A learned nested branch may duplicate behavior already present in retained
  teacher components.
- A block-specific design can overfit its calibration distribution and may not
  transfer to other layers or models.
- The novelty of teacher-tailored nested replacement relative to broader
  pruning, decomposition, and neural architecture literature remains
  unresolved and requires a dedicated literature search before a novelty
  claim.

## Artifacts and Reproducibility Status

Status: planned and unverified. No final operator grid, component score,
parameter budgets, training schedule, or result schema has been frozen.

Intended artifacts should include one JSON row per candidate with its exact
topology, retained teacher components, dimensions, parameter accounting,
calibration budget, fitting history, local metrics, model metrics, and recovery
status. Raw captured activations should remain local rather than being promoted
as wiki evidence.

## Relationships

- [[experiment-initial-block-compression-study]] supplies the fixed baseline
  and single-block evaluation conventions from which this study proceeds.
- [[method-hybrid-operator-replacement]] defines one generic whole-MLP family
  that may later receive teacher-tailored branch sizes.
- [[method-minitron-activation-based-importance]] provides prior-work context
  for forward-only width-component scoring.
- [[method-modegpt-modular-decomposition]] provides prior-work context for
  structure-preserving internal gated-MLP compression.
- [[method-two-stage-operator-grafting]] provides the local-fit and equal
  integrated-recovery comparison pattern.
- [[source-summary-phase-transitions-compression-2026]] records the supporting
  capacity-curve framework and the limits on treating its PTP and
  orthogonality claims as established facts.

## Sources

- `src-minitron-2024` - Section 2.2, Section 4.2, Tables 13-14
- `src-modegpt-2025` - Sections 3.1-3.2, Algorithm 1, Appendix A
- `src-grafting-2025` - Sections 3.1-3.3 and 4.1, Tables 1-3
- `src-phase-transitions-compression-2026` - overview, phase-transition
  modeling, and criticality-aware compression framework; supplement pending
