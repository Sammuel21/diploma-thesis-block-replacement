---
metadata_version: 1
title: Single-Block Replacement Baseline Workflow
type: experiment-workflow
category: experiments/block
status: draft
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
    - notebooks/block/baseline-testing.ipynb
  artifacts:
    - data/results/notebook-block-study/baselines-layer-11.json
---

# Single-Block Replacement Baseline Workflow

Notebook: [baseline-testing.ipynb](../../../notebooks/block/baseline-testing.ipynb)

Status: the saved artifact is explicitly marked `exploratory-unverified`.

## Purpose

This is the smallest controlled replacement experiment. At zero-based layer
11, it compares the original MLP with five alternatives using the same local
validation pairs and the same language-model validation batches. It asks
whether a learned replacement is better than trivial controls and whether a
nonlinear reduced-width SwiGLU improves on dense linear and affine mappings.

It is not an architecture search. Each candidate is inserted separately, and
there is no model-wide recovery.

## Compared conditions

| Condition | Construction | Experimental role |
| --- | --- | --- |
| Original MLP | Unchanged teacher block | Uncompressed reference |
| Zero | Always returns zero | Complete-removal control |
| Mean | Returns the mean calibration target | Input-independent control |
| Dense linear | Ridge-fitted `Ax` | Tests a bias-free linear approximation |
| Dense affine | Ridge-fitted `Ax+b` | Tests whether a learned bias helps |
| Narrow SwiGLU | Randomly initialized and locally distilled | Preserves the teacher operator family at 50% intermediate width |

The SwiGLU width is correctly defined relative to the teacher intermediate
dimension. For SmolLM2-1.7B, `d_model=2,048` and `d_ff=8,192`, so the 50%
student width is 4,096. Its 25,165,824 parameters are half of the original
MLP's 50,331,648 parameters. The earlier 1,024-wide interpretation based on
`d_model` is not used by this notebook or its current artifact.

## Pipeline

```text
load dense model and fixed data partitions
    -> capture layer-11 calibration and operator-validation pairs
    -> construct or fit the five replacement candidates
    -> evaluate every candidate on held-out local pairs
    -> temporarily insert one candidate into the model
    -> evaluate WikiText-2 loss and perplexity
    -> restore the original MLP and save the artifact
```

The zero and mean controls require no optimizer. The linear and affine
operators use ridge regression with coefficient `1e-4`. The narrow SwiGLU uses
activation MSE, AdamW, a maximum of 64 epochs, and validation-based early
stopping. With 12,288 calibration pairs and an operator batch of 2,048, one
epoch contains six optimizer updates.

The local and model-level evaluations answer different questions. Local NMSE
measures how closely the replacement reproduces the isolated MLP output.
WikiText-2 loss and perplexity measure the effect after that error propagates
through the rest of the model.

## Reporting and artifact

The notebook reports:

- parameter count and retained fraction of the original MLP;
- local MSE, NMSE, R2, cosine similarity, norm ratio, and token-relative error;
- language-model loss, perplexity, and changes from the dense model; and
- the narrow-SwiGLU training and validation history.

Results are stored in
[baselines-layer-11.json](../../../data/results/notebook-block-study/baselines-layer-11.json),
schema version 2. The artifact contains six result rows and records the model
revision, data configuration, environment, candidate definitions, and training
history.

## Limits and Perun handoff

This workflow covers one layer, one SwiGLU width, one random seed, and the
historical 48-batch calibration budget. It does not use teacher-derived
initialization or model recovery, and the artifact has not been promoted beyond
its `exploratory-unverified` status.

The notebook has no run/load switch: executing it loads the model, recomputes
the experiment, and overwrites the artifact. During the Perun migration, that
compute path should become a Python job and the notebook should become an
artifact-only report. The artifact's `operator_training.kind` is currently
named `bottleneck_mlp` even though the fitted candidate is a gated SwiGLU; the
migrated metadata should use one unambiguous operator name.

## Configuration appendix

| Setting | Value |
| --- | --- |
| Model | `HuggingFaceTB/SmolLM2-1.7B` |
| Resolved revision | `effd688a12921b4cc83e3312b6feb579f70f9c71` |
| Target layer | 11, zero-based |
| Calibration corpus | C4 train, shard `en/c4-train.00000-of-01024.json.gz` |
| Model validation corpus | WikiText-2 raw validation |
| Sequence length / capture batch | 128 tokens / 2 sequences |
| Calibration | 48 batches / 12,288 activation pairs |
| Operator validation | 24 batches / 6,144 pairs |
| Model validation | 24 batches |
| Recovery / test | Disabled |
| SwiGLU width | 4,096 / 50% of teacher `d_ff` |
| SwiGLU initialization | Random weights |
| Local optimizer | AdamW, learning rate `1e-3`, weight decay `0` |
| Local schedule | Constant, operator batch 2,048, maximum 64 epochs |
| Local selection | Early stopping patience 3, minimum delta 0 |
| Ridge coefficient | `1e-4` |
| Activation storage | CPU float32 |
| Seed | 21 |
| Current execution behavior | Always recompute and save |
| Artifact | `baselines-layer-11.json`, schema 2, exploratory-unverified |
