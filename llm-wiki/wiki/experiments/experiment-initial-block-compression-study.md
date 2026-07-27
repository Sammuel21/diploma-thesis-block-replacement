---
id: experiment-initial-block-compression-study
title: Initial Block-Compression Study
summary: Defines the working three-notebook progression from activation geometry through single-block operator comparison to multi-block degradation.
type: experiment
status: draft
created: 2026-07-27
updated: 2026-07-27

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
    - block-importance
    - effective-rank
    - error-propagation
    - mlp-block-replacement
    - operator-comparison
    - pca
  granularities:
    - mlp-block
    - transformer-layer
    - model
    - cross-level
  pipeline_stages:
    - data
    - screening
    - selection
    - replacement
    - integration
    - evaluation
    - analysis

sources: []
related:
  - "[[method-block-importance]]"
  - "[[concept-replacement-error-propagation]]"
  - "[[decision-primary-compression-evaluation-scope]]"
  - "[[implementation-compute-environments]]"
  - "[[decision-working-experiment-code-standards]]"
supersedes: []
superseded_by: []
---

# Initial Block-Compression Study

## Overview and Maturity

**Working initial experiment design.** This page records the current
three-notebook investigation of MLP-block compression. The notebooks are
intended to build experimental understanding in stages; they are not a
finished production pipeline or a final thesis protocol.

The notebooks currently contain no preserved execution outputs. Their designs,
configured metrics, and code paths are project artifacts, but no empirical
finding should be inferred until a run configuration and result artifact are
preserved and reviewed.

## Objective

The study asks three progressively broader questions:

1. Do MLP activations exhibit low-dimensional structure on calibration data?
2. At matched parameter budgets, which smaller operator families approximate
   one frozen MLP block most effectively?
3. With one simple operator fixed, how do layer selection and the number of
   simultaneous replacements affect complete-model degradation?

This progression deliberately separates representation diagnostics, operator
expressivity, and model-level composition.

## Hypotheses

**Researcher hypotheses.** The initial notebooks explore, without yet
establishing, the following propositions:

- post-gating activations may concentrate most variance in substantially fewer
  directions than their nominal intermediate width;
- a compact nonlinear or hybrid operator may outperform a simpler linear
  operator at the same parameter cost;
- BI, local approximation error, and layer depth may describe different parts
  of block replaceability; and
- degradation from several replacements may be non-additive because upstream
  approximation errors change downstream inputs.

## Rationale for the Three-Notebook Progression

| Stage | Experimental unit | Rationale |
| --- | --- | --- |
| Activation analysis | Representations from one untouched block | Establish descriptive geometry before changing the model. |
| Baseline testing | One replacement at one layer | Separate operator family from capacity under matched budgets. |
| Degradation analysis | Several simultaneously replaced layers | Measure selection effects, accumulation, and interaction after a simple operator is understood. |

The degradation analysis remains separate because BI is a layer-selection
baseline, not an operator family. Its dense-linear operator is held fixed so
that selection strategy and replacement count are the intended variables.

The historical MVP motivates the degradation question but is archived
evidence. New notebook results must not be merged with historical logs without
an explicit comparison protocol.

## Configuration

Current defaults are exploratory:

- model: `HuggingFaceTB/SmolLM2-1.7B`;
- main single-block layer: zero-based layer 11;
- sequence length: 128;
- calibration: 48 batches of size 2, or 12,288 tokens;
- operator validation: 24 document-disjoint batches;
- model validation: WikiText-2 validation batches where required;
- seed: 21;
- recovery: disabled; and
- intended execution environment: the shared RTX 4090 environment documented
  in [[implementation-compute-environments]].

The degradation notebook protects layers 0 and 23 from replacement, fits
eligible layers 1 through 22 independently from dense-model activations, and
evaluates prefixes with `k` from 1 through 6.

## Inputs and Artifacts

- [`activation-analysis.ipynb`](../../../notebooks/block/activation-analysis.ipynb)
- [`baseline-testing.ipynb`](../../../notebooks/block/baseline-testing.ipynb)
- [`degradation-analysis.ipynb`](../../../notebooks/block/degradation-analysis.ipynb)
- [`notebooks/block/README.md`](../../../notebooks/block/README.md)
- reusable calculations under
  [`src/mlp_replacement/`](../../../src/mlp_replacement/)
- intended result directory: `data/results/notebook-block-study/`

Only summaries and plot data are intended to be saved. Raw activation tensors
are not experiment artifacts.

## Procedure

### Stage 1: Activation Analysis

Capture the MLP input `x`, post-gating activation `z`, and branch output `y`
for layer 11. Describe distributions and norms, compute centered covariance
spectra, and compare held-out `z` reconstruction using PCA, random coordinate
retention, and top-variance neuron retention. Measure error both in `z` and
after the original down projection.

### Stage 2: Single-Block Baseline Testing

Capture dense-model `(x, y)` pairs for layer 11 and compare zero, mean,
low-rank linear, dense linear, standard MLP, narrow gated MLP, and
linear-plus-nonlinear-residual replacements. Evaluate four approximately
matched parameter tiers. Report local approximation metrics and complete-model
WikiText-2 degradation without recovery.

### Stage 3: Multi-Block Degradation Analysis

Fit one bias-free dense linear replacement for every eligible layer. Compute
canonical whole-layer BI and a separately named MLP-local adaptation. First
scan isolated replacements, then evaluate low/high BI prefixes and seeded
random prefixes for `k` from 1 through 6. Compare observed multi-block loss
degradation with the sum of isolated degradations.

## Metrics

Activation analysis reports:

- covariance eigenvalue spectra;
- participation-ratio effective rank and stable rank;
- dimensions needed for 90%, 95%, and 99% explained variance; and
- held-out reconstruction error before and after the down projection.

Replacement analyses report:

- exact replacement and removed parameter counts;
- local MSE, relative MSE, cosine similarity, output-norm ratio, and
  token-relative error percentiles;
- WikiText-2 loss and perplexity changes; and
- multi-block interaction
  `I(S) = delta_loss(S) - sum(delta_loss({layer}))`.

## PCA and Effective Rank Are Related but Not Equivalent

Both begin with the same covariance eigendecomposition. Effective rank reduces
the eigenvalue distribution to one descriptive scalar. Spectral plots retain
the complete eigenvalue profile. PCA additionally uses the eigenvectors to
construct a particular `k`-dimensional projection and measures actual held-out
reconstruction.

The post-down-projection PCA error also depends on how discarded directions
align with the original down-projection weights; effective rank cannot express
that relationship. Neither measurement directly determines a deployable
replacement width. Replacement rank or hidden width must still be selected
from quality-versus-budget experiments.

## Direct Results

No direct results are recorded. The notebooks have not been promoted to
experiment-backed evidence and currently contain no stored outputs.

## Interpretation

No empirical interpretation is currently permitted. Future results should be
described as exploratory evidence for this model, calibration distribution,
layer set, and seed. They must not support a universal claim that low-rank,
low-BI, or low-local-error blocks are generally replaceable.

## Limitations

- The activation and operator-family stages currently study only layer 11.
- The operator comparison uses one model and one training seed.
- The parameter tiers are aggressive and may all lie below an acceptable
  quality threshold.
- Runtime, latency, and downstream benchmark evaluation are not part of these
  initial notebooks.
- PCA is a reconstruction oracle, not a deployable compressed SwiGLU.
- The adapted MLP-local BI is not canonical whole-layer BI.
- The advanced multi-block comparison is deferred until the single-block stage
  identifies a stronger equal-cost operator.

## Reproducibility Status

Status: working and unverified. The current notebooks expose their main model,
data, seed, budget, and output-path choices, but clean-kernel executions and
preserved result artifacts remain pending. Production hardening and automated
tests are intentionally deferred under
[[decision-working-experiment-code-standards]].

## Relationships

- [[method-block-importance]] defines canonical whole-layer BI and explains
  why the MLP-local variant must remain separately named.
- [[concept-replacement-error-propagation]] motivates isolated and multi-block
  comparisons and the interaction metric.
- [[decision-primary-compression-evaluation-scope]] makes footprint-quality
  trade-offs primary and keeps general systems claims outside this study.
- [[implementation-compute-environments]] records the intended execution
  environment and its interpretation limits.
- [[decision-working-experiment-code-standards]] governs the deliberately
  lightweight maturity and implementation style of these notebooks.

## Sources

No registered literature source is cited directly. The experiment design is a
collaborative project plan. The earlier LLM-generated planning conversation is
context, not primary evidence; related method and concept pages retain their
own registered literature provenance.
