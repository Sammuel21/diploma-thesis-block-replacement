---
metadata_version: 1
title: Operator Study Workflow
type: experiment-workflow
category: experiments/block
status: active
created: 2026-09-11
modified: 2026-09-11
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
sources:
  notebooks:
    - notebooks/block/operator.ipynb
  artifacts:
    - data/results/notebook-block-study/operator-experiments.json
---

# Operator Study Workflow

Notebook: [operator.ipynb](../../../notebooks/block/operator.ipynb)

Status: an executed artifact is present and the notebook currently loads it.
The experiment uses the historical local-fitting method.

## Purpose

This workflow screens several drop-in MLP replacement architectures across
model depth. It asks three related questions:

1. How accurately does each architecture imitate an isolated teacher MLP?
2. How much teacher KL does one replacement cause after model integration?
3. How much of that damage can 64 replacement-only recovery updates remove?

Every candidate accepts and returns the model hidden width, so it can replace
the complete MLP sublayer without changing the surrounding Transformer block.
The experiment evaluates one operator-layer pair at a time; it does not create
a model containing many replacements.

## Operator design space

Ten candidates are implemented:

| Candidate | Construction | Initialization |
| --- | --- | --- |
| Zero | Constant zero output | None |
| Mean | Mean calibration target | None |
| Linear | Ridge-fitted dense linear map | Closed-form fit |
| Affine | Ridge-fitted dense affine map | Closed-form fit |
| SVD low-rank 0.50 | Rank `0.5 * d_model` linear map | Truncated SVD of the fitted dense linear map |
| MLP 0.50 | Two-layer SiLU MLP with width `0.5 * d_model` | Random weights |
| SwiGLU 0.25 | SwiGLU with width `0.25 * d_ff` | Random weights |
| SwiGLU 0.50 | SwiGLU with width `0.50 * d_ff` | Random weights |
| SwiGLU 0.75 | SwiGLU with width `0.75 * d_ff` | Random weights |
| Hybrid 0.25/0.25 | Rank-`0.25 * d_model` linear path plus width-`0.25 * d_model` SiLU path | Random weights |

These are project-selected experimental definitions, not a taxonomy or set of
equations copied from one source. The whole-operator low-rank map and internal
factorization of teacher SwiGLU matrices are distinct ideas; only the former is
implemented here.

The numeric suffixes do not imply matched parameter counts. SwiGLU widths are
fractions of teacher `d_ff`, while low-rank, compact-MLP, and hybrid widths are
fractions of `d_model`. The experiment therefore compares the realized
footprint-quality trade-offs, not pure architecture quality under one equal
parameter budget.

## 1. Local operator fitting

The workflow captures MLP input-output pairs for all eligible layers, 1 through
22. For each layer it fits or constructs every operator and records:

- parameters and fraction of the original MLP footprint;
- local MSE and NMSE;
- local cosine similarity; and
- the selected epoch for gradient-trained operators.

The mean control is computed from the layer's calibration targets. Linear and
affine mappings use ridge regression with coefficient `1e-4`. The low-rank
candidate begins from the truncated SVD of the fitted linear operator. The
remaining learned candidates use AdamW and local activation MSE.

## 2. Singleton integration and recovery

The dense teacher logits are cached on the model-validation batches. For each
layer, every fitted candidate is copied, inserted individually, and evaluated
against that teacher cache. The workflow then freezes the dense model and
performs 64 KL-distillation updates on the replacement parameters only.

```text
locally fitted operator
    -> singleton insertion
    -> pre-recovery teacher KL
    -> 64 replacement-only KL updates
    -> post-recovery teacher KL
    -> restore dense block
```

Candidates at the same layer share the streamed recovery batches. Different
layers remain independent experiments. The KL reduction is
`pre_recovery_kl - post_recovery_kl`; a positive value means recovery repaired
some of the integration damage.

## Reporting and artifact

The notebook reports:

- all operator-layer local and integration measurements;
- operator rankings by median, mean, and worst post-recovery KL;
- the worst layer and median local NMSE for each operator;
- heatmaps and clustered heatmaps of post-recovery KL and KL reduction; and
- a layer-sensitivity ranking for the 50% SwiGLU, including the
  pre-recovery-KL-to-local-NMSE ratio.

Results are stored in
[operator-experiments.json](../../../data/results/notebook-block-study/operator-experiments.json),
schema version 1. It contains 220 local-fitting rows and 220 singleton
sensitivity rows: 22 layers times ten operators.

## Limits and Perun handoff

This is a broad historical screen, not the final operator-selection protocol.
It uses only 48 calibration batches and randomly initializes the nonlinear
students. Teacher-derived initialization is directly defined only for the
same-family SwiGLU candidates and is not used here. The unequal parameter
budgets also prevent attributing every difference solely to architecture.

The notebook captures float32 activations for all 22 layers together and keeps
all fitted candidates until sensitivity evaluation. That layout is convenient
for a notebook but creates avoidable host-memory pressure. A Perun job should
process bounded layer groups, complete all operator fits and singleton
measurements for each group, and append their records to a versioned artifact.
This changes memory residency, not the experiment's data or objective.

The `Local Activation Distillation`, `Operator Architecture Selection`,
`Advanced Personalized Operators`, `Model Integration Loss analysis`, and
secondary-compression sections contain notes or placeholders rather than
separate implemented result sets. They should not be described as completed
experiments.

The artifact does not record wall-clock time, GPU-hours, or peak memory. Those
fields should be added to the future job artifact according to the
[reporting framework](../../methodology/reporting-framework.md).

## Configuration appendix

| Setting | Value |
| --- | --- |
| Model | `HuggingFaceTB/SmolLM2-1.7B` |
| Resolved revision | `effd688a12921b4cc83e3312b6feb579f70f9c71` |
| Eligible / protected layers | 1-22 / 0 and 23 |
| Calibration and recovery corpus | C4 train, shard `en/c4-train.00000-of-01024.json.gz` |
| Model validation corpus | WikiText-2 raw validation |
| Sequence length / capture batch | 128 tokens / 2 sequences |
| Calibration | 48 batches / 12,288 pairs per layer |
| Operator validation | 24 batches / 6,144 pairs per layer |
| Recovery | 64 batches / 64 optimizer updates |
| Model validation | 24 batches |
| Operator count | Ten candidates per layer |
| Nonlinear initialization | Random weights |
| Low-rank initialization | Truncated SVD of fitted dense linear operator |
| Local optimizer | AdamW, learning rate `1e-3`, weight decay `0` |
| Local schedule | Constant, operator batch 2,048, maximum 64 epochs |
| Local selection | Early stopping patience 3, minimum delta 0 |
| Ridge coefficient | `1e-4` |
| Recovery objective | Replacement-only KL, temperature 1.0, learning rate `1e-5` |
| Recovery teacher cache | Float16 logits |
| Activation storage | CPU float32, all eligible layers captured together |
| Seed | 21 |
| Current execution mode | Load |
| Artifact | `operator-experiments.json`, schema 1 |
