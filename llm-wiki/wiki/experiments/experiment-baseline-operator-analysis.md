---
id: experiment-baseline-operator-analysis
title: Baseline Operator Analysis
summary: Defines the exploratory calibration, capacity, recovery, and replacement-sensitivity analyses built on the fixed single-block baseline.
type: experiment
status: draft
created: 2026-08-18
updated: 2026-08-18

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
    - baseline-evaluation
    - calibration-scaling
    - capacity-sweep
    - compression-recovery
    - knowledge-distillation
    - mlp-block-replacement
    - operator-comparison
    - replacement-error-propagation
  granularities:
    - mlp-block
    - transformer-layer
    - model
    - cross-level
  pipeline_stages:
    - replacement
    - integration
    - recovery
    - evaluation
    - analysis

sources:
  - source_id: src-grafting-2025
    locator: "Sections 3.2-3.3"
    relation: contextualizes
    note: "Prior-work context for self-grafting and staged local fitting followed by integrated recovery; the project baseline is not the same control."

related:
  - "[[experiment-initial-block-compression-study]]"
  - "[[experiment-swiglu-operator-design-progression]]"
  - "[[comparison-recovery-objective-trainable-scope]]"
  - "[[concept-replacement-error-propagation]]"
supersedes: []
superseded_by: []
---

# Baseline Operator Analysis

## Overview and Epistemic Boundary

**Collaborative experiment design; unverified.**
`baseline-testing.ipynb` defines the fixed six-condition single-block baseline.
`baseline-experiments.ipynb` builds exploratory analyses around that reference:
it varies calibration data, reduced-SwiGLU capacity, layer position, and
replacement-only recovery budget, then compares local approximation with
model-level consequences. It does not replace the fixed baseline protocol.

Grafting's self-grafting control inserts a randomly initialized operator of
the same type and size before local fitting and integrated recovery.
[src-grafting-2025, Sections 3.2-3.3] **Synthesis.** The present notebook
borrows the idea that a common, deliberately simple reference helps interpret
later operator designs. It is not a reproduction of self-grafting: the project
compares changed or reduced operator architectures in an autoregressive LLM.

This page records methodology, rationale, observed implementation state, and
planned extensions. It does not promote notebook output to empirical findings.

## Objective

The notebook is intended to answer four connected questions:

1. How does the amount of local calibration data affect an operator's held-out
   fit and the quality of the model after one replacement?
2. How does reduced-SwiGLU width trade block footprint against local error and
   model loss or perplexity?
3. Under an equal replacement-only teacher-logit recovery budget, how do
   different reduced widths recover?
4. Across model depth, does local operator distillability predict the global
   effect of inserting that operator?

The first three questions extend the single layer-11 baseline. The fourth
changes the unit of analysis to independently fitted, single-block
replacements across eligible layers. Simultaneous multi-block interaction is
not measured by the current KL-divergence analysis.

## Research Questions and Hypotheses

**Researcher questions.** These are candidate thesis questions, not
source-derived claims:

- Does additional calibration data improve held-out local approximation after
  optimization exposure is controlled?
- Does increasing reduced-SwiGLU width improve local fit monotonically, and
  does the same ordering hold for model loss or perplexity?
- Does model-level teacher-logit recovery change the ranking of widths?
- Does held-out local relative MSE predict pre-recovery model-level KL across
  layers and operator families?
- Does pre-recovery replacement sensitivity predict post-recovery quality or
  repairability?

**Researcher hypotheses.** More local data or more width is expected to reduce
local approximation error, but neither change is assumed to improve
model-level quality proportionally. Replacement depth and operator family may
change how a remaining local error propagates. These propositions remain
unverified until tested under controlled runs.

## Configuration

The current exploratory defaults observed in the notebook are:

| Component | Current setting |
| --- | --- |
| Model | `HuggingFaceTB/SmolLM2-1.7B` |
| Main single-block target | zero-based layer 11 |
| Selected depth comparison | layers 3, 11, and 19 |
| Full-depth eligibility | all layers except the first and last |
| Sequence length and batch size | 128 tokens and 2 sequences |
| Local calibration budgets | 8, 16, 32, and 48 batches |
| Operator-validation budget | 24 batches |
| Model-validation and KL-evaluation budget | 24 WikiText-2 validation batches |
| Linear references | ridge-fitted dense linear `Ax` and affine `Ax+b`, ridge `1e-4` |
| Reduced-SwiGLU widths | 0.25, 0.50, 0.75, and 0.90 of teacher `d_ff` |
| Local SwiGLU fit | activation MSE, at most 64 epochs, batch size 2,048, patience 3 |
| Recovery checkpoints | 0, 64, 128, 256, 512, and 1,024 optimizer updates |
| Recovery | forward KL at temperature 1, AdamW at `1e-5`, replacement parameters only |
| Seed | 21 |

The C4 calibration, operator-validation, and recovery windows are sampled as
document-disjoint partitions. WikiText-2 validation is kept separate for
complete-model evaluation. This describes current project data construction,
not a claim that these budgets are sufficient.

## Budget Terminology

The notebook uses several quantities that should not all be called simply
"compute budget":

| Budget | What changes | Current interpretation |
| --- | --- | --- |
| Calibration-data budget | C4 activation pairs used for local fitting | Data quantity; with a fixed epoch limit it also changes optimizer-update count |
| Operator-validation budget | Held-out C4 activation pairs | Local model selection and generalization measurement; no parameter updates |
| Model-validation or KL-evaluation budget | WikiText-2 sequences | Read-only model-level measurement; no parameter updates |
| Recovery budget | Teacher-logit KD optimizer updates and consumed C4 token positions | Model-level objective with replacement-only parameter updates |

One local-fit epoch means one pass through all activation pairs. Therefore,
with local minibatch size 2,048, the number of updates per epoch is

$$
U_{\mathrm{epoch}}
=
\left\lceil
\frac{N_{\mathrm{activation\ pairs}}}{2048}
\right\rceil.
$$

**Standard counting identity; no citation required.** Consequently, the
current calibration scaling experiment changes both the number of distinct
activation pairs and the maximum number of optimizer updates. It is a useful
combined data-and-training-exposure curve, but not yet an equal-update test of
calibration data alone.

## Procedure

### 1. Calibration-data scaling

At layer 11, fit dense linear, dense affine, and 50%-width SwiGLU substitutes
for every calibration budget. Reuse the same held-out operator-validation and
model-validation sets. Record the SwiGLU train and validation histories, local
metrics, model loss, and perplexity. Plot the linear references separately
when their error scale obscures the SwiGLU curve.

### 2. Reduced-SwiGLU capacity sweep

At the maximum local calibration budget, fit widths 0.25, 0.50, 0.75, and 0.90
of the teacher intermediate width. Compare held-out local relative MSE and
complete-model perplexity with the maximum-budget dense linear and affine
references. Report exact intermediate width and relative block parameters for
every candidate. Width is a capacity variable, not an importance score.

### 3. Recovery trajectories

Starting from every locally fitted SwiGLU width, integrate the substitute at
layer 11 and optimize final-logit teacher-to-student KL. Evaluate model loss
and perplexity at common cumulative update checkpoints. All original model
parameters remain frozen; only the active replacement module is updated. This
must be named **model-level teacher-logit KD with replacement-only updates**,
not full-model end-to-end retraining. See
[[comparison-recovery-objective-trainable-scope]].

### 4. Pre-recovery replacement-sensitivity profile

Fit linear, affine, and 0.25/0.50/0.75-width SwiGLU substitutes independently
at each eligible layer. Insert one candidate at a time and measure forward KL
from the untouched teacher to that single-replacement student. First inspect
layers 3, 11, and 19, then plot the same operator comparisons across all
eligible depths while visually marking those selected layers.

### 5. Global-to-local analysis

Compare each candidate's held-out local relative MSE with its pre-recovery
model-level KL. Use a scatter plot as the primary view because it preserves
both measurements. A ratio may be retained as a secondary within-operator
ranking diagnostic.

## Metrics and Project Definitions

**Project metric definition; no external citation required.** For original MLP
`f_i`, fitted substitute `g_i`, and held-out teacher input `h_i`, local
relative MSE is

$$
E_i^{\mathrm{local}}
=
\frac{
\mathbb{E}\left[
\left\|g_i(h_i)-f_i(h_i)\right\|_2^2
\right]
}{
\mathbb{E}\left[
\left\|f_i(h_i)\right\|_2^2
\right]
+\varepsilon
}.
$$

The normalization supports comparisons across layers whose target-output
scales differ.

**Project evaluation definition using standard KL notation; no external
citation required.** Let `S_i` be the model with only layer `i`'s MLP replaced
after local fitting. Its pre-recovery global divergence is

$$
K_i^{\mathrm{pre}}
=
\mathbb{E}_{x}\left[
D_{\mathrm{KL}}\left(
p_T(\cdot\mid x)
\parallel
p_{S_i}(\cdot\mid x)
\right)
\right].
$$

The implementation averages forward KL over valid token positions and then
over evaluation batches.

**Project-proposed diagnostic; no external citation required.** The current
notebook defines

$$
R_i
=
\frac{K_i^{\mathrm{pre}}}
{E_i^{\mathrm{local}}+\varepsilon}.
$$

`R_i` asks how much global divergence is associated with one unit of remaining
local relative error. It is unbounded and has no universal high/low threshold.
It should be ranked within a fixed operator family and shared evaluation
protocol, with the two-axis scatter retained to prevent a ratio from hiding
whether its numerator or denominator caused the value.

Additional recorded metrics are exact parameters, theoretical weight bytes,
local MSE, cosine similarity, language-model loss, perplexity, and deltas from
the untouched model.

## Teacher-Logit Storage and Planned Streaming

The current recovery implementation materializes every teacher-logit batch on
CPU before fitting any width. Its approximate logit storage is

$$
M_{\mathrm{cache}}
=
N_{\mathrm{batches}}BT|V|b,
$$

where `B` is batch size, `T` is sequence length, `|V|` is vocabulary size, and
`b` is bytes per cached logit. **Standard tensor-storage accounting; no
citation required.** With 1,024 batches, batch size 2, sequence length 128,
the current model vocabulary, and float16 logits, the recovery cache alone is
approximately 24 GiB of system RAM. Input IDs and attention masks add only a
small amount relative to the logits. This storage is not the recovery budget;
it is one materialization strategy for the selected recovery batches.

**Planned implementation.** A shared streaming recovery loop would process
all width candidates in lockstep:

1. run the untouched teacher once for the current recovery batch;
2. keep only that batch's logits;
3. update each width candidate once against the same logits;
4. discard the logits and continue to the next batch.

This preserves one teacher forward per recovery batch and equal ordered data
across candidates while reducing the large cache to approximately one batch.
A chunked variant can retain, for example, 32 or 64 teacher batches at a time
and may be simpler to integrate. Processing one candidate at a time without a
shared cache would also lower memory, but would recompute the teacher for every
candidate and therefore increase runtime.

The 24-batch teacher cache used for repeated read-only KL evaluation is much
smaller and heavily reused. The present design therefore plans to stream or
chunk the large recovery-training cache, not automatically remove the small
evaluation cache.

## Implementation Status

| Analysis or mechanism | Repository state | Evidence status |
| --- | --- | --- |
| Calibration-data scaling | Implemented and has notebook output | Not promoted or reviewed |
| Single-layer SwiGLU width sweep | Implemented and has notebook output | Not promoted or reviewed |
| Replacement-only recovery curves | Implemented and has notebook output | Not promoted or reviewed |
| Selected-layer pre-recovery KL | Implemented and has notebook output | Not promoted or reviewed |
| Full eligible-depth pre-recovery KL | Implemented and has notebook output | Not promoted or reviewed |
| Global-to-local scatter and ratio | Code present; remote execution pending | Unverified |
| Single end-of-run JSON artifact | Code present; remote execution pending | Unverified |
| Post-recovery KL profiles and repairability comparison | Planned; not implemented | Research idea |
| Independent calibration-size and update-count control | Planned; not implemented | Experimental control |
| Calibration/validation/recovery response surfaces | Placeholder only | Visualization idea |
| Multi-block KL interactions | Placeholder only | Out of current single-block scope |
| Pareto selection across footprint and quality | Deferred until candidate data are stable | Planned analysis |
| Shared streaming or chunked recovery | Planned; not implemented | Memory optimization |
| Stage-level recompute/load controls | Discussed; not implemented | Workflow idea |

An executed cell or plot does not establish a finding. Promotion requires a
preserved artifact, configuration review, finite-metric checks, and a written
interpretation tied to that artifact.

## Inputs and Artifact Contract

- [`baseline-experiments.ipynb`](../../../notebooks/block/baseline-experiments.ipynb)
- [`baseline-testing.ipynb`](../../../notebooks/block/baseline-testing.ipynb)
- [`src/mlp_replacement/baselines.py`](../../../src/mlp_replacement/baselines.py)
- [`src/mlp_replacement/recovery.py`](../../../src/mlp_replacement/recovery.py)
- planned local, ignored result:
  `data/results/notebook-block-study/baseline-experiments.json`

The current artifact cell performs one save at the end and overwrites the
latest file. It records schema version, UTC timestamp, model and data
configuration, operator-training and recovery configuration, selected and
excluded layers, sweep values, baseline language-model metrics, and the result
tables produced by completed analyses. It intentionally excludes activations,
logits, replacement weights, optimizer state, hardware telemetry, and plots.

This single-save policy is a current workflow decision, not a final artifact
versioning design. Partial-run checkpointing or one artifact per analysis may
be introduced if notebook stages become independently reusable.

## Direct Results

No direct results are recorded on this page. Existing notebook outputs remain
exploratory and have not been promoted to a reviewed result artifact.

## Interpretation Boundary

The analyses distinguish three questions that should not be collapsed:

- **local imitation:** how closely a substitute reproduces one MLP on held-out
  teacher activations;
- **global replacement sensitivity:** how much the integrated model changes
  before recovery; and
- **repairability:** how much model-level KD reduces that change under a fixed
  recovery protocol.

Pre-recovery KL is therefore a replacement-sensitivity measurement for a
specified fitted operator, not a universal layer-importance or intrinsic
linearity score. Post-recovery KL would additionally depend on the optimizer,
data, update count, trainable scope, and operator capacity.

## Limitations and Controls

- The current calibration curve confounds data quantity with optimizer-update
  count because epochs, not updates, are fixed.
- Operator families are not parameter-matched; capacity and architecture can
  both affect comparisons.
- Recovery checkpoints match update count, but different widths still require
  different computation per update.
- `E_i^{local}` and `K_i^{pre}` must use the same held-out input distribution
  when their relationship or ratio is interpreted.
- The full-depth profile fits every layer independently from dense-teacher
  activations. It does not model error interactions from simultaneous
  replacements.
- One seed cannot quantify initialization variance for learned SwiGLU
  substitutes.
- Layer 0 and the final layer are excluded by project decision; the resulting
  depth profile does not characterize those positions.
- The current full-depth implementation retains many candidate modules and
  activation sets in memory. Recovery-cache streaming addresses only one part
  of total memory use.

## Reproducibility Status

Status: draft and unverified. The notebook exposes its central configuration
and has executed outputs for several stages, but the global-to-local analysis
and consolidated artifact cell still require a clean remote run. A reviewed
artifact should contain exactly the configured operator-layer rows, finite
metrics, resolved model revision, dataset specifications, seed, and all budget
units. No result should move to this page until those conditions are checked.

## Relationships

- [[experiment-initial-block-compression-study]] defines the fixed baseline
  controls from which these sweeps proceed.
- [[experiment-swiglu-operator-design-progression]] uses these capacity and
  evaluation conventions as the first stage before more specialized operator
  designs.
- [[comparison-recovery-objective-trainable-scope]] fixes the precise name and
  boundary of the current replacement-only model-level KD procedure.
- [[concept-replacement-error-propagation]] motivates comparing local error
  with integrated model behavior rather than treating local MSE as sufficient.

## Sources

- `src-grafting-2025` - Sections 3.2-3.3, self-grafting and staged recovery
  context only
- Project implementation - `notebooks/block/baseline-experiments.ipynb`,
  `src/mlp_replacement/data.py`, and `src/mlp_replacement/recovery.py`, inspected
  2026-08-18
