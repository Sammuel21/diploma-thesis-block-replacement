---
metadata_version: 1
title: Operator Distillation Workflow
type: experiment-workflow
category: experiments/block/operator
status: active
created: 2026-09-11
modified: 2026-09-19
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
sources:
  notebooks:
    - notebooks/block/operator/operator-distillation.ipynb
  artifacts:
    - data/results/notebook-block-study/operator-distillation-v4.json
---

# Operator Distillation Workflow

Notebook: [operator-distillation.ipynb](../../../../notebooks/block/operator/operator-distillation.ipynb)

Status: the latest executed artifact is `operator-distillation-v4.json`, using
artifact schema version 5. The notebook currently remains in run mode.

## Purpose

This workflow diagnoses and improves local activation distillation for one
fixed replacement problem: a bias-free SwiGLU at zero-based layer 11 retaining
50% of the teacher intermediate width. The frozen teacher supplies pairs of
MLP inputs and outputs, and the student minimizes output MSE.

The experiment separates three causes of fitting quality:

1. the number of unique calibration pairs and the associated training budget;
2. optimizer batch size and update frequency; and
3. random versus teacher-derived initialization.

It then measures how initialization behaves as the retained SwiGLU width
increases. Local NMSE and cosine similarity are the primary diagnostics. This
notebook does not insert the selected operator into the complete model, so it
cannot determine final language-model quality by itself.

## Fixed baseline and data partition

The teacher has `d_model=2,048` and `d_ff=8,192`. The controlled 50% student
therefore has width 4,096 and 25,165,824 parameters. This corrects the earlier
historical interpretation that applied 50% to `d_model` and produced an
incompatible 1,024-wide baseline.

The workflow preserves the original partitions before drawing more training
data:

```text
48 original calibration batches
    -> 24 fixed operator-validation batches
    -> 336 additional calibration batches
```

At two sequences of 128 tokens per capture batch, these are 12,288 original
training pairs, 6,144 validation pairs, and 98,304 total training pairs. Every
later condition uses the same operator-validation set. Additional calibration
pairs are appended after that fixed partition rather than shifting it.

The original setting is recomputed from random weights with the correct
4,096-wide student, 48 calibration batches, an operator batch of 2,048, and a
maximum of 64 epochs. A historical width curve is loaded from
`baseline-experiments.json` only as architecture context; it is not the source
of the corrected baseline measurement.

## 1. Calibration-data study

The first improvement study fits the same randomly initialized 50% SwiGLU on
48, 96, 192, and 384 capture batches:

| Capture batches | Activation pairs | Maximum steps at 64 epochs |
| ---: | ---: | ---: |
| 48 | 12,288 | 384 |
| 96 | 24,576 | 768 |
| 192 | 49,152 | 1,536 |
| 384 | 98,304 | 3,072 |

The operator batch stays at 2,048 and every condition receives the same
maximum of 64 epochs. Larger datasets therefore provide both more unique pairs
and proportionally more possible optimizer updates. This estimates the best
observed fit attainable from each data budget rather than forcing all budgets
into the same 384-update limit. Early stopping can end an individual fit before
the maximum.

See
[operator calibration data and training budget](../../../methodology/operator-calibration-data-and-training-budget.md)
for the full bookkeeping rationale.

## 2. Operator-batch-size screen

Using all 98,304 training pairs, the notebook compares operator batches of 256,
512, 1,024, and 2,048 for exactly eight epochs. Early stopping is disabled in
this short screen. All conditions see the same unique examples and the same
number of pair presentations, while smaller batches perform more frequent
updates.

The screen selects the configuration with the lowest held-out NMSE. In the
current schema-5 artifact this is batch 256. The combined study then repeats
the four calibration sizes with that selected batch, a maximum of 64 epochs,
and early stopping. This separates a short batch-size comparison from the more
expensive data-plus-optimization experiment.

## 3. Teacher-derived initialization

Teacher initialization is possible because the student remains a SwiGLU. One
teacher intermediate neuron corresponds to the same row in `gate_proj`, the
same row in `up_proj`, and the same column in `down_proj`. Selecting `k`
neurons therefore creates a coherent width-`k` student:

```text
student gate rows = selected teacher gate rows
student up rows   = selected teacher up rows
student down cols = selected teacher down columns
```

The notebook first copies all 8,192 neurons into a full-width student and
requires near-zero NMSE on 256 validation pairs. This is a code-correctness
control for projection names, orientations, activation, and copying logic; it
is not a compression result.

It then compares three 4,096-wide starting points on the full 98,304-pair
training set:

- `random_weights` initializes a new student conventionally;
- `random_teacher_subset` copies a seeded random subset of teacher neurons;
- `importance_teacher_subset` copies the highest-scoring teacher neurons.

For teacher neuron `i`, the intermediate coordinate is

```text
z_i(x) = SiLU((W_gate x)_i) * (W_up x)_i
```

and the project-defined selection score is

```text
I_i = RMS_x[z_i(x)] * ||W_down[:, i]||_2
```

The first term measures how strongly the neuron is used on calibration inputs.
The second measures the strength of its path back to the residual-stream
dimension. Their product is a project-defined activation-contribution
heuristic. The equation is not copied one-to-one from WANDA, MiniTron, or
MoDeGPT. The notebook describes it as inspired by activation-aware pruning; a
thesis claim about that inspiration still requires a citation to the relevant
source.

After copying, every student remains fully trainable and is locally distilled.
Teacher selection is therefore an initialization followed by optimization,
not merely the evaluation of a pruned teacher.

## 4. SwiGLU width analysis

The final study tests retained widths of 50%, 60%, 70%, 80%, 90%, and 95% for
all three initialization methods. It uses the full calibration pool, an
operator batch of 2,048, and at most 64 epochs.

The seeded random teacher ordering and importance ordering are computed once.
Larger widths take longer prefixes of the same ordering, so neuron subsets are
nested across widths. This makes the width curves easier to interpret than
independently resampling a different subset at every point.

The reporting compares initial and fitted NMSE, cosine similarity, selected
epoch, optimizer updates, parameter retention, and the advantage of each
teacher-copy method over random initialization.

## Artifact contents

The current
[operator-distillation-v4.json](../../../../data/results/notebook-block-study/operator-distillation-v4.json)
uses schema version 5 and records:

- the corrected original-distillation configuration, metrics, and history;
- calibration-data summaries and histories;
- batch-size screening, the winning configuration, and histories;
- the combined calibration and selected-batch comparison;
- the full-width control and all initialization results;
- selected neuron indices and importance-score summaries; and
- width-sweep results, histories, comparisons, and complete neuron rankings.

Earlier operator-distillation artifacts remain historical and should not be
mixed with schema 5. The artifact filename version and internal schema version
are separate identifiers: `v4` is the filename, while `schema_version` is 5.

## Limits and Perun handoff

This remains a layer-11 local study. It reuses the same operator-validation
partition for early stopping and configuration selection, and it has no
untouched operator-test split. Model loss, perplexity, downstream quality, and
multi-block interactions must be evaluated in the model workflows after the
fitting recipe is frozen.

The notebook is currently in run mode, so executing all cells recomputes the
study and overwrites `operator-distillation-v4.json`. For Perun, the capture,
fitting, importance ranking, and artifact writing should move into one Python
job. The notebook should load the completed artifact and retain only tables and
plots. The job should record stage runtimes, actual epochs and optimizer steps,
GPU-hours, and peak memory following the
[reporting framework](../../../methodology/reporting-framework.md).

The notebook's final `Findings & Solutions` markdown contains statements from
earlier iterations, including a claim that teacher initialization was not yet
implemented. That text is stale. The executable cells and schema-5 artifact are
the authoritative description of the current workflow.

## Configuration appendix

| Setting | Value |
| --- | --- |
| Model | `HuggingFaceTB/SmolLM2-1.7B` |
| Resolved revision | `effd688a12921b4cc83e3312b6feb579f70f9c71` |
| Target layer | 11, zero-based |
| Teacher dimensions | `d_model=2,048`, `d_ff=8,192` |
| Main student | Bias-free SwiGLU, width 4,096 / 50% of teacher `d_ff` |
| Calibration corpus | C4 train, shard `en/c4-train.00000-of-01024.json.gz` |
| Sequence length / capture batch | 128 tokens / 2 sequences |
| Calibration budgets | 48, 96, 192, and 384 batches |
| Operator validation | 24 batches / 6,144 pairs |
| Calibration-study schedule | Batch 2,048, maximum 64 epochs |
| Batch-size screen | 256, 512, 1,024, and 2,048 over eight epochs |
| Screen winner in current artifact | Batch 256 |
| Combined study | Selected batch, maximum 64 epochs |
| Initialization comparison | Random weights, random teacher subset, importance teacher subset |
| Teacher-initialization fit | 98,304 pairs, batch 2,048, maximum 64 epochs |
| Importance scoring batch | 2,048 pairs |
| Importance metric | RMS post-SwiGLU activation times down-column L2 norm; project-defined |
| Full-width control | 8,192 neurons, evaluated on 256 validation pairs, required NMSE at most `1e-5` |
| Width sweep | 50%, 60%, 70%, 80%, 90%, and 95% |
| Local objective | Activation MSE |
| Local optimizer | AdamW, learning rate `1e-3`, weight decay `0` |
| Local schedule | Constant with early-stopping patience 3 except the batch screen |
| Activation storage | CPU float32 |
| Recovery / model integration | Not performed |
| Seed | 21 |
| Current execution mode | Run |
| Artifact | `operator-distillation-v4.json`, schema 5 |
