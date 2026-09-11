---
metadata_version: 1
title: Compression Baseline Workflow
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
    - notebooks/model/compression-baseline.ipynb
  artifacts:
    - data/results/model-compression-baselines/compression-baseline.json
    - data/results/model-compression-baselines/compression-baseline-v2.json
---

# Compression Baseline Workflow

Notebook: [compression-baseline.ipynb](../../../notebooks/model/compression-baseline.ipynb)

## Purpose

This is the controlled model-wide baseline. It replaces every second MLP with
a SwiGLU retaining 50% of the original intermediate width. It does not search
for the best blocks or widths. Its purpose is to ask whether improvements to
local operator fitting translate into better whole-model quality when the
compression architecture is held fixed.

Two layer-selection variants are tested:

- including boundaries permits replacement of the first and last blocks;
- excluding boundaries protects the first and last blocks.

Keeping these variants separate shows whether boundary blocks are unusually
sensitive without mixing that question with importance-based allocation.

## Shared configuration

The model is SmolLM2-1.7B. Text sequences have length 128 and the teacher
capture batch contains two sequences. Every replacement is a bias-free SwiGLU
with 50% retained intermediate width. Local fitting uses batches of 2,048
activation pairs, at most 64 epochs, a constant learning rate of `1e-3`, and
validation-based early stopping with patience three. Model-wide recovery uses
one epoch of replacement-only KL distillation at `1e-5`.

The model revision, validation data, recovery data, seed, layer selections,
operator architecture, and recovery budget are matched across methods.

## Historical run

The historical path uses 48 calibration batches:

```text
48 batches x 2 sequences x 128 tokens = 12,288 activation pairs
```

Each selected MLP is replaced by a randomly initialized operator. The workflow
fits the replacements locally, integrates all selected replacements, evaluates
loss, perplexity, and teacher KL before recovery, performs model-wide recovery,
and evaluates the recovered model.

The two boundary variants are independent model runs. Their results are loaded
from
[compression-baseline.json](../../../data/results/model-compression-baselines/compression-baseline.json)
by default.

## Optimized run

The optimized path raises the calibration budget to 384 batches:

```text
384 batches x 2 sequences x 128 tokens = 98,304 activation pairs
```

It tests two methods for each boundary variant:

1. `new_data_random` uses the larger calibration set but retains random
   initialization.
2. `complete_method` uses the same larger set and initializes each reduced
   SwiGLU from an importance-ranked subset of teacher neurons.

The teacher-neuron score is the RMS intermediate activation multiplied by the
L2 norm of the corresponding down-projection column. This is a project-defined
importance metric, not an equation copied from prior work. The highest-scoring
neurons supply matching gate and up rows and down-projection columns to the
student; local fitting may then adjust all replacement weights.

The 64-epoch limit is fixed rather than holding optimizer updates fixed. At the
larger data size, one epoch contains 48 updates instead of six, so the maximum
budget rises from 384 to 3,072 updates. This is intentional: the optimized run
asks how well the larger dataset can be learned, not how to distribute one
fixed compute budget. See
[operator calibration data and training budget](../../methodology/operator-calibration-data-and-training-budget.md).

The original calibration, operator-validation, recovery, and
recovery-validation partitions are sampled before the additional calibration
partition. Consequently, increasing the training set does not move or replace
the historical evaluation partitions.

## Comparison logic

The three methodology labels form an ablation:

```text
historical random
    -> larger data with random initialization: effect of data and training budget
    -> larger data with teacher initialization: additional effect of initialization
```

Each method is evaluated for both boundary variants before and after the same
recovery procedure. The notebook reports:

- model parameters, total-model reduction, and MLP-parameter reduction;
- validation loss, perplexity, and teacher KL;
- local validation MSE, NMSE, cosine similarity, and best epoch by block; and
- pre-recovery versus post-recovery model quality.

The historical artifact did not store NMSE directly. For the additional NMSE
plot, the notebook reconstructs it from historical MSE and the target scale of
the matched optimized validation partition.

## Python workflow

The headless compute path is
[`workflows.runs.model.compression_baseline`](../../../workflows/runs/model/compression_baseline.py),
driven by
[`compression-baseline.json`](../../../workflows/configs/model/compression-baseline.json).
It exposes historical, optimized, and combined stages and leaves this notebook
unchanged as the explanatory and loading frontend. Each run writes the same
science-artifact schema plus a sibling crash-aware `.run.json` operational
record. Numerical parity still requires a GPU execution comparison; the
migration has not yet been run on Perun.

## Output and limits

Optimized results and the historical comparison are stored in
[compression-baseline-v2.json](../../../data/results/model-compression-baselines/compression-baseline-v2.json).
Both notebook execution switches currently use load mode.

This pipeline is a baseline, not an allocation search. It cannot establish
which layers deserve more capacity, and its short recovery run does not settle
the final model-retraining budget. It establishes whether the local fitting
improvements survive model integration under an intentionally simple control.

## Configuration appendix

The artifacts are the authoritative configuration records. This table is the
human-readable snapshot needed to interpret the experiment.

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
| Replacement | Bias-free SwiGLU, 50% retained intermediate width |
| Initialization methods | Random weights; importance-ranked teacher subset |
| Teacher-subset score | RMS intermediate activation times down-projection-column L2 norm; project-defined |
| Local optimizer | AdamW, learning rate `1e-3`, weight decay `0` |
| Local schedule | Constant, operator batch 2,048, maximum 64 epochs |
| Local selection | Early stopping patience 3, minimum delta 0 |
| Layer selection | Interleaved, stride 2, offset 0 |
| Boundary variants | Protect 0/0 or 1/1 prefix/suffix blocks |
| Integration | One-shot replacement of all selected blocks |
| Recovery | Replacement-only KL, AdamW, 1 epoch, learning rate `1e-5`, temperature 1.0 |
| Recovery cache | Float16 |
| Seed | 21 |
| Current execution modes | Historical load; optimized load |
| Artifacts | `compression-baseline.json` schema 1; `compression-baseline-v2.json` schema 2 |
