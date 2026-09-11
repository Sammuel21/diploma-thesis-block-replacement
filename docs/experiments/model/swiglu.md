---
metadata_version: 1
title: SwiGLU Compression Workflow
type: experiment-workflow
category: experiments/model
status: active
created: 2026-09-10
modified: 2026-09-11
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
sources:
  notebooks:
    - notebooks/model/swiglu.ipynb
  artifacts:
    - data/results/notebook-model-study/swiglu-compression.json
    - data/results/notebook-model-study/swiglu-compression-optimized.json
---

# SwiGLU Compression Workflow

Notebook: [swiglu.ipynb](../../../notebooks/model/swiglu.ipynb)

## Purpose

This workflow distributes one model-wide MLP compression budget across
different Transformer blocks. Instead of forcing every eligible MLP to retain
the same width, it asks whether important blocks should keep more SwiGLU
neurons while less important blocks keep fewer.

The first and last MLP blocks are protected. The remaining blocks share an
exact target of 50% MLP-parameter removal in the main experiment. All policies
therefore spend approximately the same parameter budget; the difference is
where they spend it.

## Allocation policies

Three policies are compared:

- `uniform` assigns the same retained fraction to every eligible block;
- `canonical_bi` uses whole-Transformer-block influence scores;
- `residual_aware_mlp_bi` uses influence measured around the MLP sublayer.

Canonical BI is a prior-work-motivated block-influence signal. The MLP-local
variant and the exact replacement-width allocator are project adaptations and
must not be presented as equations taken directly from MoDeGPT. Their
methodological status is recorded in
[global-to-local operator budget allocation](../../methodology/global-to-local-operator-budget-allocation.md).

Scores are converted to percentile ranks and then to removal propensities:

```text
larger importance rank -> smaller removal propensity -> larger retained width
```

The allocation temperature is 1.0. Whole-neuron rounding is reconciled so all
policies retain the same aggregate MLP budget.

## Historical pipeline

The historical run follows this order:

```text
dense reference
    -> BI scoring
    -> width allocation
    -> random local operator fitting
    -> simultaneous model integration
    -> pre-recovery evaluation
    -> one epoch of replacement-only KL recovery
    -> post-recovery evaluation
```

It uses 48 calibration batches, equivalent to 12,288 activation pairs per
block, and randomly initializes every replacement. Local fitting uses a
2,048-pair operator batch, at most 64 epochs, and early stopping. Local NMSE
and cosine measure approximation quality; model loss, perplexity, and teacher
KL measure the effect after the replacements interact inside the model.

The historical section also sweeps target MLP sparsities of 20%, 30%, 40%, and
50% for all three policies. This produces a model-quality versus compression
curve rather than evaluating only one footprint.

The default historical path loads
[swiglu-compression.json](../../../data/results/notebook-model-study/swiglu-compression.json).

## Optimized replication

The optimized run keeps the model revision, partitions, allocation rules,
policies, and recovery configuration matched to the historical run. It changes
the local fitting method in two controlled stages:

1. `new_data_random` increases calibration to 384 batches, or 98,304 pairs,
   while retaining random initialization.
2. `complete_method` uses the same larger budget and importance-based
   teacher-derived initialization.

This gives a direct decomposition:

```text
historical -> new data only -> new data plus teacher initialization
```

Teacher-derived initialization copies a top-ranked subset of corresponding
gate rows, up rows, and down-projection columns. The neuron ranking uses the
project-defined activation-and-down-projection contribution score described in
the compression baseline workflow. The operator is still distilled afterward,
so initialization is a starting point rather than the final compressed model.

To fit within host RAM, activations are captured for six blocks at a time,
stored on CPU in the model's native dtype, used for every policy in that group,
and then released. No activation cache is written to disk. Grouping changes
memory residency and repeated teacher-forward cost, but not the examples,
operator objective, widths, seeds, or evaluation methodology.

After local fitting, every combination of the two optimized fitting methods
and three allocation policies is integrated and evaluated before and after the
same model-wide recovery procedure.

## Reporting and interpretation

The notebook reports:

- historical versus optimized BI scores and width allocations;
- retained width and local NMSE for every eligible block;
- mean and worst local NMSE, cosine similarity, and best epoch by policy;
- whole-model sparsity, loss, perplexity, and teacher KL; and
- quality before and after model-wide recovery.

The important distinction is that block importance and block approximability
are not the same. BI estimates how much a block affects the model. Local NMSE
estimates how closely a replacement can imitate it at its assigned width. The
model-level evaluation is needed because many individually fitted errors can
interact downstream.

Optimized results and comparisons are stored in
[swiglu-compression-optimized.json](../../../data/results/notebook-model-study/swiglu-compression-optimized.json).

## Limits and handoff

The optimized section fixes the target MLP sparsity, score set, and allocation
temperature. It evaluates the improved fitting methodology, but it does not
search for a better allocation rule. The historical sparsity sweep also uses
the older random-initialization and calibration method. Those open allocation
questions are the purpose of the `swiglu-2` workflow.

## Configuration appendix

The artifacts are the authoritative configuration records. This table
summarizes the settings that define the historical and optimized comparisons.

| Setting | Value |
| --- | --- |
| Model | `HuggingFaceTB/SmolLM2-1.7B` |
| Model and tokenizer revision | `effd688a12921b4cc83e3312b6feb579f70f9c71` |
| Calibration and recovery corpus | C4 train, shard `en/c4-train.00000-of-01024.json.gz` |
| Model validation corpus | WikiText-2 raw validation |
| Sequence length / capture batch | 128 tokens / 2 sequences |
| Historical calibration | 48 batches / 12,288 operator pairs |
| Optimized calibration | 384 batches / 98,304 operator pairs |
| Operator validation | 24 batches / 6,144 pairs |
| Recovery / recovery validation | 64 / 24 batches |
| Model validation / test | 24 / 0 batches |
| Eligible / protected layers | 1-22 / 0 and 23 |
| Main MLP-removal target | 50% of eligible MLP parameters |
| Historical sparsity sweep | 20%, 30%, 40%, and 50% MLP removal |
| Allocation policies | Uniform, canonical BI, residual-aware MLP BI |
| Allocation normalization | Ascending percentile rank |
| Allocation temperature | 1.0 |
| Replacement | Bias-free variable-width SwiGLU |
| Historical initialization | Random weights |
| Optimized initialization comparison | Random weights versus importance-ranked teacher subset |
| Teacher-subset score | RMS intermediate activation times down-projection-column L2 norm; project-defined |
| Local optimizer | AdamW, learning rate `1e-3`, weight decay `0` |
| Local schedule | Constant, operator batch 2,048, maximum 64 epochs |
| Local selection | Early stopping patience 3, minimum delta 0 |
| Optimized activation storage | CPU, model-native dtype, six-block groups, no disk I/O |
| Recovery | Replacement-only KL, AdamW, 1 epoch, learning rate `1e-5`, temperature 1.0 |
| Recovery cache | Float16 |
| Seed | 21 |
| Current execution modes | Historical load; optimized run |
| Artifacts | `swiglu-compression.json` schema 2; `swiglu-compression-optimized.json` schema 3 |
