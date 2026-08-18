---
id: experiment-initial-block-compression-study
title: Initial Block-Compression Study
summary: Defines the working activation, practical single-block replacement, and multi-block degradation studies with an optional quantization baseline.
type: experiment
status: draft
created: 2026-07-27
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
    - activation-analysis
    - baseline-evaluation
    - block-importance
    - effective-rank
    - error-propagation
    - mlp-block-replacement
    - operator-comparison
    - pca
    - quantization
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
  - "[[method-hybrid-operator-replacement]]"
  - "[[experiment-swiglu-operator-design-progression]]"
  - "[[experiment-baseline-operator-analysis]]"
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

Working notebook outputs may exist, but no run has been promoted to a reviewed
experiment artifact on this page. Their designs, configured metrics, and code
paths are project artifacts; no empirical finding should be inferred until a
run configuration and result artifact are preserved and reviewed.

## Objective

The study asks four progressively broader questions:

1. Do MLP activations exhibit low-dimensional structure on calibration data?
2. How does one fixed reduced-width SwiGLU compare with complete-removal,
   constant-output, dense-linear, and dense-affine references?
3. With one simple operator fixed, how do layer selection and the number of
   simultaneous replacements affect complete-model degradation?
4. At a comparable storage budget, how competitive is simple MLP quantization
   with learned operator replacement, and does quantizing a replacement add a
   useful final compression step?

This progression deliberately separates representation diagnostics, operator
expressivity, and model-level composition.

## Hypotheses

**Researcher hypotheses.** The initial notebooks explore, without yet
establishing, the following propositions:

- post-gating activations may concentrate most variance in substantially fewer
  directions than their nominal intermediate width;
- a reduced-width SwiGLU with intermediate width
  $r=0.5d_{\mathrm{ff}}$ may preserve more of one MLP's function than zero
  and mean-output controls while retaining 50% of that block's parameters;
- BI, local approximation error, and layer depth may describe different parts
  of block replaceability; and
- degradation from several replacements may be non-additive because upstream
  approximation errors change downstream inputs; and
- uniform MLP quantization may be a strong numerical baseline at matched
  storage cost, while importance-aware mixed precision may improve its
  footprint-quality trade-off.

## Rationale for the Three-Notebook Progression

| Stage | Experimental unit | Rationale |
| --- | --- | --- |
| Activation analysis | Representations from one untouched block | Establish descriptive geometry before changing the model. |
| Baseline testing | Fixed whole-MLP references and controls at one layer | Establish a practical reference before considering architecture or width search. |
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
- single-block replacement: bias-free SwiGLU with intermediate width
  $r=0.5d_{\mathrm{ff}}$;
- single-block conditions: original MLP, zero, mean, dense linear, dense
  affine, and narrow SwiGLU;
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
- [`src/mlp_replacement/baselines.py`](../../../src/mlp_replacement/baselines.py),
  a draft reusable implementation of the fixed single-block baseline
  calculation;
- supporting calculations under
  [`src/mlp_replacement/`](../../../src/mlp_replacement/); and
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

Capture dense-model `(x, y)` pairs for layer 11. Compare the original MLP, a
zero-output removal control, a calibration-output mean control, ridge-fitted
dense linear and affine references, and one bias-free SwiGLU replacement whose
intermediate width is $r=0.5d_{\mathrm{ff}}$. The notebook reads the original
$d_{\mathrm{ff}}$ from the loaded teacher block; for the configured model, the
expected width is 4096 and the expected SwiGLU count is 25,165,824 parameters.
Fit the SwiGLU using activation MSE. Report exact footprints, held-out local
approximation metrics, and complete-model WikiText-2 degradation without
recovery. This stage does not search widths or operator families.

#### Fixed baseline definition

The baseline suite has six deliberately different roles:

| Condition | Construction and fitting | Role in the comparison |
| --- | --- | --- |
| Original MLP | Untouched teacher operator | Uncompressed local and model-quality reference |
| Zero | Returns zero for every input | Complete-removal control |
| Mean | Returns the calibration-target mean for every input | Input-independent constant control |
| Dense linear | Fits $Ax$ by ridge-regularized least squares on calibration pairs | Reference for linear approximability without an offset |
| Dense affine | Fits $Ax+b$ by ridge-regularized least squares on calibration pairs | Tests whether a learned offset explains error beyond the linear map |
| Narrow SwiGLU | Uses $r=\operatorname{round}(\rho d_{\mathrm{ff}})$, with default $\rho=0.5$, and fits activation MSE | Straightforward learned compression baseline in the teacher operator family |

The operator forms are standard explanatory notation and do not require an
external citation. Their selection and assigned baseline roles are project
decisions, not literature-derived claims. The fixed SwiGLU width is not
claimed to be optimal.

#### Comparison contract

All six conditions must share the same loaded checkpoint, target block,
calibration and held-out activation pairs, complete-model validation loader
and batch limit, numerical precision, and seed where randomness applies. The
untouched model is evaluated once. Each substitute is then evaluated on the
held-out activation pairs and temporarily inserted into the same model for
complete-model loss and perplexity evaluation. Recovery is disabled in this
baseline stage.

The suite provides controls and reference levels; it is not a
parameter-matched operator-family comparison. Dense linear, dense affine, and
narrow SwiGLU have different feasible footprints. A later claim that one
architecture is better than another therefore requires matched or
nearest-feasible parameter or byte budgets, or a capacity curve. If recovery
is introduced, every compared candidate must receive the same recovery data,
objective, trainable scope, and optimizer-step budget.

#### Reusable calculation boundary

[`src/mlp_replacement/baselines.py`](../../../src/mlp_replacement/baselines.py)
contains the draft `run_single_block_baselines` runner. It accepts an already
loaded model, a `BlockRef` containing both the layer index and MLP module,
captured training and validation activation pairs, a model-validation loader,
an `OperatorConfig`, and optional SwiGLU-width, ridge, and validation-batch
settings.

Within that prepared context, the runner:

1. evaluates the untouched model once;
2. constructs or fits the five substitute conditions;
3. computes held-out local metrics;
4. temporarily inserts each substitute for complete-model evaluation;
5. records parameter, state-element, and theoretical-weight-byte accounting;
   and
6. returns six standardized result rows, per-operator fit histories, the
   untouched model metrics, and the original and replacement intermediate
   widths in `SingleBlockBaselineResult`.

The runner intentionally does not load models or datasets, build loaders,
capture activations, choose a layer, run model-level recovery, create plots,
or write JSON artifacts. Experimental notebooks own those choices and may
evaluate additional candidates against the returned rows.

**Implementation status.** Repository inspection shows that
`baseline-testing.ipynb` still performs the equivalent baseline calculation
inline. The draft runner has not yet been wired into that notebook or the
operator experiments, and clean-runtime and numerical-parity checks have not
been recorded. It is therefore a proposed reusable boundary, not yet the
authoritative or validated executor of the baseline protocol.

### Working Numerical Baseline Extension

**Researcher baseline idea.** Quantization is retained as an optional numerical
compression comparison around the operator-replacement study, not as another
replacement operator or a required fourth notebook.

The simplest baseline should apply uniform weight quantization to the MLP
weights, initially at 8-bit and 4-bit precision. It should be evaluated before
adding importance estimation so that the contribution of quantization itself
remains visible. A stronger follow-up may use importance-aware mixed precision,
assigning different bit widths under one fixed total storage budget. Because
that variant combines numerical compression with component selection, its
importance metric and allocation rule must be reported separately from the
uniform baseline.

Comparisons with learned replacements must match stored footprint rather than
parameter count: quantization normally changes bytes per parameter while
retaining the number of parameters. Report the quantized MLP footprint and
complete-model footprint separately. After selecting a promising replacement,
quantizing its fitted weights can serve as a small combined structural and
numerical compression experiment.

The exact quantization implementation, calibration convention, importance
estimator, and supported precisions remain open design choices. This extension
should stay small unless its initial results justify deeper investigation.

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
replacement width. The fixed $0.5d_{\mathrm{ff}}$ width is a practical
baseline decision; a later controlled ablation would be required to compare
widths or support an optimality claim.

## Direct Results

No direct results are recorded. Exploratory notebook outputs may exist, but no
run has been promoted to a preserved, reviewed experiment artifact on this
page.

## Interpretation

No empirical interpretation is currently permitted. Future results should be
described as exploratory evidence for this model, calibration distribution,
layer set, and seed. They must not support a universal claim that low-rank,
low-BI, or low-local-error blocks are generally replaceable.

## Limitations

- The activation and practical replacement stages currently study only layer 11.
- The practical replacement baseline uses one model, one layer, and one
  training seed.
- The selected $0.5d_{\mathrm{ff}}$ width is not claimed to be optimal or
  minimal.
- Runtime, latency, and downstream benchmark evaluation are not part of these
  initial notebooks.
- PCA is a reconstruction oracle, not a deployable compressed SwiGLU.
- The current adapted MLP-local score compares the normalized MLP input with
  the raw MLP output. It is not canonical whole-layer BI or the proposed
  residual-aware MLP influence score.
- The degradation notebook's dense-linear replacement is an independent
  diagnostic; comparing its selection curves with the narrow SwiGLU would
  require a separate replication.
- Quantization is currently a baseline idea only; no implementation, matched
  storage protocol, or result artifact has been selected.

## Reproducibility Status

Status: working and unverified. The current notebooks expose their main model,
data, seed, fixed replacement, and output-path choices, but clean-kernel
executions and preserved result artifacts remain pending. The reusable runner
additionally requires a six-row parity check against the inline notebook logic
under one identical prepared context, finite local and model metrics, and
confirmation that temporary replacement restores the teacher after each
condition. Production hardening and automated tests are intentionally deferred
under [[decision-working-experiment-code-standards]].

## Relationships

- [[method-block-importance]] defines canonical whole-layer BI and explains
  why the MLP-local variant must remain separately named.
- [[method-hybrid-operator-replacement]] defines a future candidate family
  that should reuse this stage's comparison protocol without being treated as
  an established improvement.
- [[experiment-swiglu-operator-design-progression]] extends the fixed baseline
  into generic whole-MLP, structure-aware, and teacher-tailored operator
  studies without changing this page into an architecture search.
- [[experiment-baseline-operator-analysis]] varies calibration data, reduced
  SwiGLU width, recovery budget, and layer position without changing the fixed
  six-condition definition recorded here.
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
