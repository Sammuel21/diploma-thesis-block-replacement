---
title: Baseline Experiments Workflow
status: active
created: 2026-09-11
modified: 2026-10-15
sources:
  notebooks:
    - notebooks/block/baseline/baseline-experiments.ipynb
  artifacts:
    - data/results/notebook-block-study/baseline-experiments.json
---

# Baseline Experiments Workflow

Notebook: [baseline-experiments.ipynb](../../../../notebooks/block/baseline/baseline-experiments.ipynb)

Status: an executed historical artifact is present, but it predates the current
notebook artifact schema.

## Purpose

This notebook expands the single-block baseline along several axes. It studies
how local data volume, retained SwiGLU width, and replacement-only recovery
affect layer 11. It then repeats singleton replacement measurements across
model depth to compare operator families and determine whether local fit
predicts model-level damage.

Despite the `Multi-block Experiments` heading, replacements are evaluated one
at a time. The notebook does not integrate several compressed blocks into one
student model.

## Shared configuration

The model is SmolLM2-1.7B. Capture sequences contain 128 tokens and the capture
batch contains two sequences. The first and last Transformer blocks are
excluded from the depth study, leaving layers 1 through 22.

Calibration data, local validation, and recovery use document-disjoint C4
windows; model evaluation uses WikiText-2.

Learned nonlinear operators use AdamW, activation MSE, an operator batch of
2,048 pairs, at most 64 epochs, and validation-based early stopping. Linear
and affine candidates use ridge regression. Nonlinear candidates start from
random weights; teacher-derived initialization is not part of this workflow.

Recovery freezes the dense model and updates only the temporarily inserted
replacement using teacher-to-student KL at learning rate `1e-5`. One streamed
C4 batch is one optimizer update. Recovery streams one teacher-logit batch
across all active width candidates before discarding it; the smaller read-only
validation-logit cache remains in memory.

## 1. Calibration-data scaling

At layer 11, the notebook tests 8, 16, 32, and 48 capture batches, equivalent
to 2,048, 4,096, 8,192, and 12,288 activation pairs. For every budget it fits:

- a bias-free ridge linear operator;
- a ridge affine operator; and
- a randomly initialized SwiGLU retaining 50% of teacher `d_ff`.

All SwiGLU conditions retain the same maximum of 64 epochs. Consequently,
larger calibration sets can produce more optimizer updates per epoch; this is
a fixed-epoch data-scaling study, not a fixed-update comparison. Early stopping
may reduce the actual number of completed epochs.

Each candidate is measured using held-out local error and by temporary
insertion into the language model. This shows whether more local examples help
the operator and whether that improvement survives model integration.

## 2. SwiGLU width and recovery

Using the largest 48-batch calibration set, the notebook fits random SwiGLU
students retaining 25%, 50%, 75%, and 90% of the original intermediate width.
It compares local NMSE and integrated perplexity with the maximum-budget linear
and affine references.

Every width then follows one cumulative recovery trajectory with checkpoints
at 0, 64, 128, 256, 512, and 1,024 optimizer updates. The same streamed teacher
batches are used for all widths. This stage asks how much model-level training
is required to repair each width, rather than refitting a fresh operator for
every checkpoint.

The notebook also constructs a calibration-by-recovery heatmap for the 50%
SwiGLU. It combines the four local calibration budgets with the same six
recovery checkpoints to show whether additional local fitting reduces the
amount of recovery required.

## 3. Full-depth singleton sensitivity

The full-depth study evaluates five candidates at every eligible layer:

```text
linear
affine
SwiGLU at 25%, 50%, and 75% retained width
```

For each operator-layer pair, the workflow:

1. captures calibration and operator-validation activations;
2. fits the operator locally;
3. inserts only that operator and measures teacher KL;
4. performs 64 replacement-only recovery updates; and
5. measures post-recovery KL.

This produces 110 independent cases: 22 layers times five operators. Layers 3,
11, and 19 are highlighted as representative depths, but the artifact covers
all eligible layers.

The workflow also computes

```text
global_to_local_ratio = pre_recovery_KL / (local_NMSE + epsilon)
```

This is a project-defined diagnostic ratio. It asks whether a small local
approximation error causes disproportionately large model-level damage. It is
not a standard theorem or a source-derived importance equation. For this ratio,
the local pairs are captured from the same WikiText model-validation batches
used for the KL comparison, rather than from the C4 operator-validation split.

## Reporting and artifact

The notebook reports:

- calibration tokens against local NMSE and integrated perplexity;
- SwiGLU width against footprint, local fit, and perplexity;
- perplexity trajectories over recovery updates;
- the calibration-by-recovery surface for the 50% SwiGLU;
- pre- and post-recovery KL by layer and operator; and
- local NMSE against KL and the global-to-local ratio across depth.

The available
[baseline-experiments.json](../../../../data/results/notebook-block-study/baseline-experiments.json)
is schema version 2. It stores the dense baseline, calibration scaling, width
scaling, width recovery, and 110-row full-depth KL analyses.

## Results (artifact baseline-experiments.json, schema 2)

These results were moved here from the wiki page
`experiment-baseline-operator-analysis` on 2026-10-15; this section is now
their single source.

### Metric definitions

**Project definition; no external citation required.** For teacher MLP $f_i$,
fitted substitute $g_i$, and held-out teacher input $h_i$:

$$
E_i^{\mathrm{local}}
=
\frac{\mathbb{E}\lVert g_i(h_i)-f_i(h_i)\rVert_2^2}
{\mathbb{E}\lVert f_i(h_i)\rVert_2^2+\varepsilon}.
$$

**Project evaluation definition using standard KL notation; no external
citation required.** If $S_i$ has only MLP $i$ replaced after local fitting:

$$
K_i^{\mathrm{pre}}
=
\mathbb{E}_x\left[D_{\mathrm{KL}}\left(
p_T(\cdot\mid x)\parallel p_{S_i}(\cdot\mid x)
\right)\right].
$$

**Project-proposed diagnostic; no external citation required.**

$$
R_i=\frac{K_i^{\mathrm{pre}}}{E_i^{\mathrm{local}}+\varepsilon}.
$$

$R_i$ is unbounded and has no universal threshold; it expresses global change
per unit of remaining local error only within a shared operator and evaluation
protocol. The two original quantities must remain visible. Additional metrics
include parameter/byte footprint, local MSE and cosine similarity, model loss,
perplexity, and pre/post-recovery KL. `global_to_local_ratio` in the artifact
is $R_i$ with $\varepsilon = 10^{-8}$.

### Findings

The following are **empirical findings from one configured run** in
[baseline-experiments.json](../../../../data/results/notebook-block-study/baseline-experiments.json),
schema version 2:

- Untouched validation loss was 2.66997 and perplexity 14.4396 over 6,096
  predicted WikiText-2 tokens.
- Increasing calibration from 2,048 to 12,288 tokens reduced local relative
  MSE for all three tested families, but also increased learned-operator
  updates.
- At layer 11, increasing SwiGLU ratio from 0.25 to 0.90 reduced local relative
  MSE from 0.6471 to 0.6097; pre-recovery perplexity stayed between 15.1392 and
  15.1687 without a monotonic width ordering.
- After 1,024 replacement-only recovery updates, width-candidate perplexity was
  15.1169-15.1250, still above the untouched reference.
- Across 110 isolated operator-layer cases, mean post-recovery KL was highest
  at layers 1 (2.1012), 22 (1.8350), and 7 (1.4468); layer 20 was next at
  0.1941.
- Local relative MSE did not rank pre-recovery KL reliably across depth in
  this run.

These results separate local imitation, pre-recovery singleton sensitivity,
and fixed-protocol repairability. Pre-recovery KL is operator-conditioned; it
is not a universal block-importance or intrinsic-linearity score.

### Artifact contents

The checked local, ignored schema-2 artifact (created 2026-08-18, checked
2026-08-20) records configuration and result tables but excludes activations,
logits, fitted weights, optimizer state, plots, and hardware telemetry. It
contains four width rows, 24 recovery rows, and 110 rows per full-depth KL
table. Newer notebook additions expecting schema 3 remain pending rerun. The
stage-level run/load path has not received an independent numerical-parity
test.

## Current limitations and Perun handoff

Limits on interpreting the results above:

- One model and seed do not establish stability.
- Calibration quantity is confounded with update count.
- Operator families are not parameter-matched.
- The calibration and width tables measure local error on held-out C4
  operator-validation pairs, while KL uses WikiText-2. The full-depth local
  error and $R_i$ use the same WikiText-2 batches as the KL (notebook at
  commit `5f7daae`, which produced this artifact).
- Every full-depth row replaces one block independently; simultaneous errors
  may interact.
- Recovery matches update count, not computation per update.
- Candidate modules and activation sets still contribute substantial memory;
  streaming addresses only the large recovery-logit cache.

The artifact records the historical 48-batch, random-initialization method. It
is useful for exploratory comparisons but is not the optimized 384-batch,
teacher-initialized operator recipe established later in
[operator distillation](../operator/operator-distillation.md).

There is also a concrete schema mismatch. The current notebook constructs a
schema-3 artifact containing `calibration_recovery`, while the available
schema-2 artifact does not contain that field. Because the notebook is in load
mode and directly requests the missing field, a complete artifact-only run is
not currently reliable. The migration must either regenerate schema 3 or
version the load logic explicitly; the schema-2 artifact should remain
preserved as historical evidence.

For Perun, this mixed notebook maps naturally to two compute jobs: the layer-11
calibration/width/recovery study and the full-depth singleton-sensitivity
study. The latter should process a bounded group of layers at a time instead of
retaining float32 activations and fitted candidates for the entire depth. The
reporting notebook should consume their versioned artifacts and should not own
the fitting or recovery loops.

The `Multi-block interactions` and `Pareto Frontier Baseline` sections are
notes only and have no implemented results in this notebook.

## Configuration appendix

| Setting | Value |
| --- | --- |
| Model | `HuggingFaceTB/SmolLM2-1.7B` |
| Resolved revision | `effd688a12921b4cc83e3312b6feb579f70f9c71` |
| Main single-block target | Layer 11, zero-based |
| Full-depth eligible layers | 1-22; layers 0 and 23 excluded |
| Calibration and recovery corpus | C4 train, shard `en/c4-train.00000-of-01024.json.gz` |
| Model validation corpus | WikiText-2 raw validation |
| Sequence length / capture batch | 128 tokens / 2 sequences |
| Calibration budgets | 8, 16, 32, and 48 batches |
| Operator validation | 24 batches / 6,144 pairs |
| Model validation | 24 batches |
| SwiGLU width sweep | 25%, 50%, 75%, and 90% of teacher `d_ff` |
| Full-depth operators | Linear, affine, and SwiGLU at 25%, 50%, and 75% |
| Initialization | Random for nonlinear operators |
| Local optimizer | AdamW, learning rate `1e-3`, weight decay `0` |
| Local schedule | Constant, operator batch 2,048, maximum 64 epochs |
| Local selection | Early stopping patience 3, minimum delta 0 |
| Ridge coefficient | `1e-4` |
| Recovery checkpoints | 0, 64, 128, 256, 512, and 1,024 updates |
| Full-depth recovery | 64 updates per operator-layer candidate |
| Recovery objective | Replacement-only KL, temperature 1.0, learning rate `1e-5` |
| Activation storage | CPU float32 |
| Seed | 21 |
| Current execution mode | Load |
| Available artifact | `baseline-experiments.json`, schema 2 |
| Current notebook writer | Schema 3 |
