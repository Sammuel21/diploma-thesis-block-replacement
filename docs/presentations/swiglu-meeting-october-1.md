---
title: Supervisor Meeting Orientation Map, October
status: draft
created: 2026-10-15
modified: 2026-10-15
---

# Orientation map: homogeneous SwiGLU family and quantization baseline

This page is a reading guide to the experiments completed so far. Each card
gives the question an experiment asked, what was done, what came out, and
where the details live. Numbers are copied from the linked reports; the
[results snapshot](../results/2026-10-15-results.md) holds the cross-method
comparison.

## Setup in brief

- **Model:** `HuggingFaceTB/SmolLM2-1.7B`, pinned revision, 24 layers. Only the
  MLPs of layers 1-22 are replaced; layers 0 and 23 are protected.
- **Method:** replace each MLP with a narrower SwiGLU, fitted locally on
  input/output pairs captured from the dense teacher. A global *recovery*
  phase then trains the replacements against the teacher with a KL objective.
- **Removal targets** refer to eligible-MLP parameters. 20% eligible removal is
  12.9% of the whole model, and 50% is 32.4%.
- **Slice-only** means the teacher's top neurons are copied in without a local
  fit.
- **Metrics:** early experiments (SwiGLU-1 to 6) report WikiText prefix
  perplexity (dense: 14.44). The final protocol (SwiGLU-7, SwiGLU-8,
  Quantization-1) reports full test perplexity at 8K context (dense: 6.93),
  KL against the teacher on 8K sequences, and macro accuracy over five
  zero-shot tasks (dense: 67.14%). The two perplexity series use different
  protocols and are never joined into one trend.

## The arc

```text
S1 feasible?  ->  S2 where do parameters go?  ->  S3 how much data and recovery?
-> S4 best recovery recipe?  ->  S5 better structure (discrete widths)
-> S6 does 1B-token recovery help?  ->  S7 train more than the replacements?
-> S8 is the local fit still worth it after recovery?
-> Quantization-1: how does this compare with plain quantization?
```

## Experiment cards

### SwiGLU-1: is replacement feasible, and does importance-based allocation help?

- **Question:** can all 22 eligible MLPs be replaced at once at 50% removal,
  and should block-importance scores decide how much width each layer keeps?
- **Did:** compared uniform widths against two block-influence (BI) allocations
  (whole-block and MLP-local). Fitting was then improved in two controlled
  steps: more calibration pairs (12,288 to 98,304 per block), then
  teacher-neuron initialization.
- **Result:** the fitting improvements dominated. Perplexity after recovery at
  50% went from 46,023 (random init, 12k pairs) to 136 (98k pairs) to **32.6**
  (98k pairs plus teacher init). With the improved fitting, uniform widths
  (32.6) beat both BI allocations (36.1 and 36.3).
- **Lesson:** teacher-derived initialization made model-wide replacement
  viable. BI measures a block's effect in the intact model, not the damage from
  replacing it, which motivated SwiGLU-2.
- **Details:** [report](../experiments/model/swiglu/swiglu.md),
  [notebook](../../notebooks/model/swiglu/swiglu.ipynb)

### SwiGLU-2: where should the parameters go?

- **Question:** which score should set per-layer widths?
- **Did:** fitted every block alone at 25% and 50% width and measured the
  resulting whole-model damage (singleton KL). Screened 19 score and
  temperature policies and minimum-width floors, then fully fitted and
  recovered the finalists.
- **Result:** singleton KL at the 25% probe won. Widths ranged from 21% to 71%
  at the same total budget. Against uniform: perplexity 32.6 to **29.6**
  (-9.3%), KL -12.7%.
- **Lesson:** allocation should be driven by measured replacement damage, not
  by generic importance.
- **Details:** [report](../experiments/model/swiglu/swiglu-2.md),
  [notebook](../../notebooks/model/swiglu/swiglu-2.ipynb)

### SwiGLU-3: calibration size, sparsity, and long recovery

- **Question:** is local fitting saturated, and can long recovery close the gap
  to dense?
- **Did:** calibration of 98k, 196k, and 393k pairs per block. Then four
  targets (20, 30, 40, 50%), each recovered for 100M tokens.
- **Result:** more pairs helped modestly (50% pre-recovery perplexity
  29.7 to 28.9). Recovery helped every target (50%: 28.9 to 24.7 at 10M to
  **22.1** at 100M tokens) but left a clear gap to dense. The full pipeline
  took about 45 hours, too slow to screen new ideas.
- **Lesson:** recovery is valuable but expensive, so the recipe had to be
  optimized before changing the structure again.
- **Details:** [report](../experiments/model/swiglu/swiglu-3.md),
  [notebook](../../notebooks/model/swiglu/swiglu-3.ipynb)

### SwiGLU-4: optimizing the recovery recipe

- **Question:** which optimizer, objective, and trainable scope recover best?
- **Did:** fixed the 50% starting model and ran a 10M-token tournament over
  learning rate and schedule, objective, RMSNorm, and LoRA variants.
- **Result:** constant learning rate 3e-5 with pure KL at temperature 1,
  training only the replacements, won (KL 0.286 to 0.267, perplexity 24.7 to
  24.0). RMSNorm added 0.03 perplexity; LoRA variants were worse or barely
  better at higher cost.
- **Lesson:** a simple replacement-only recipe is the efficient default.
- **Details:** [report](../experiments/model/swiglu/swiglu-4.md),
  [notebook](../../notebooks/model/swiglu/swiglu-4.ipynb)

### SwiGLU-5: discrete per-layer widths

- **Question:** can initialization, allocation, or composition improve the
  compressed structure under the fixed recovery recipe?
- **Did:** compared output-reconstruction initialization, discrete width
  allocation (including the option to keep a layer fully dense), their
  combination, and a composition-aware refit. Search at 2M and 5M tokens, then
  confirmation at 100M.
- **Result:** discrete allocation won. At 50% removal it kept layers 21-22 dense
  and compressed the rest harder. At 100M tokens against SwiGLU-3: perplexity
  22.1 to **20.5**, KL -11.7%. Reconstruction and the refit were neutral.
  Execution was also 3.5-4x faster at the same number of updates.
- **Lesson:** sensitive layers should stay exact while tolerant layers absorb
  more removal. This is the homogeneous baseline allocation.
- **Details:** [report](../experiments/model/swiglu/swiglu-5.md),
  [notebook](../../notebooks/model/swiglu/swiglu-5.ipynb)

### SwiGLU-6: recovery to one billion tokens

- **Question:** does recovery keep helping beyond 100M tokens?
- **Did:** continued the 20% and 50% models to 1B tokens and added complete
  WikiText evaluation and five zero-shot tasks.
- **Result:** KL fell by 16% (20%) and 27% (50%) from 100M to 1B tokens.
  Task macro accuracy: 64.4% (20%) and 59.2% (50%). Neither reached dense
  quality.
- **Lesson:** more recovery repairs damage but does not make the compressed
  topology equivalent to the teacher.
- **Details:** [report](../experiments/model/swiglu/swiglu-6.md),
  [notebook](../../notebooks/model/swiglu/swiglu-6.ipynb)

### SwiGLU-7: should more than the replacements be trained?

- **Question:** does retraining the whole transformer body, or adding attention
  LoRA, beat replacement-only recovery?
- **Did:** rebuilt the models independently and ran 12 one-billion-token runs
  with native 8K sequences: four targets (20-50%) by three trainable scopes.
- **Result:** replacement-only recovery won 3 of 4 final-KL comparisons and
  17 of 24 likelihood comparisons, at the lowest cost (42.6 GPU-hours for four
  runs). The broader scopes gained at most 0.49 points of macro accuracy, mixed
  across targets, with one seed. The independent rebuild reproduced the earlier
  width profiles exactly.
- **Lesson:** replacement-only recovery remains the supported default. This
  closes the homogeneous family apart from SwiGLU-8.
- **Details:** [report](../experiments/model/swiglu/swiglu-7.md),
  [notebook](../../notebooks/model/swiglu/swiglu-7.ipynb)

### SwiGLU-8: is the local fit still worth anything after recovery?

- **Question:** every earlier run fitted locally before recovery. What does
  that fit buy once recovery has run, and can the fitting recipe improve?
- **Did:**
  - *Part A (block level):* 15 fitting settings, 3 seeds, 246 fits on three
    layers, scored on held-out 8K single-block KL. A setting was adopted only
    if it beat the current recipe by more than 5% and by more than seed noise.
  - *Part B (model level):* identical models with and without the local fit,
    eight targets from 20% to 80%, 100M recovery tokens, same seed and same
    token stream.
- **Result:**
  - Part A: no setting met the 5% bar (best: 4.4%), so the existing recipe
    stays. Training on 8K pairs helped at 8K but hurt at 128 tokens.
  - Part B: at token 0 the slice-only models are 2-5 times worse. After 100M
    tokens the fitted arm is still **10-24% better** in 8K KL at every target,
    above the measured run-to-run noise (up to about 3.5%). The gap is still
    shrinking, so the pre-registered verdict is "inconclusive at 100M".
- **Lesson:** local fitting still pays at this budget; whether it persists at
  1B tokens is untested. The 20-80% runs also serve as the homogeneous control
  spectrum for the heterogeneous-operator chapter.
- **Details:** [report](../experiments/model/swiglu/swiglu-8.md),
  [notebook](../../notebooks/model/swiglu/swiglu-8.ipynb)

### Quantization-1: the baseline

- **Question:** how does weight quantization of the same MLPs compare in
  quality per byte?
- **Did:** TorchAO weight-only INT8 and INT4 on the same 66 MLP projections,
  post-training (PTQ), plus INT4 quantization-aware training (QAT) at 100M and
  1B tokens. QAT used byte-identical token streams to SwiGLU-8 (100M) and
  SwiGLU-7 (1B), confirmed by hash.
- **Result:**

  | Variant | Whole-model byte saving | 8K KL | Test PPL 8K | Macro accuracy |
  | --- | ---: | ---: | ---: | ---: |
  | Dense BF16 | 0% | 0.0000 | 6.93 | 67.14% |
  | INT8 PTQ | 32.3% | 0.0030 | 6.94 | 66.99% |
  | INT4 PTQ | 47.5% | 0.0896 | 7.42 | 64.95% |
  | INT4 QAT 1B | 47.5% | 0.0352 | 7.24 | 66.64% |

- **Lesson:** INT8 is lossless within the paired intervals, and QAT removes
  about 60% of the INT4 damage. Quantization-1 is a baseline control and is
  not combined with replacement.
- **Details:** [report](../experiments/model/baseline/quantization-1.md)

## How replacement and quantization compare

- **Size:** at equal bytes, quantization keeps more quality at every budget it
  reaches (up to 47.5% saving). INT8 against SwiGLU 50% (both about 32%):
  8K KL 0.003 against 0.229. Even INT4 without training beats SwiGLU 50% after
  1B recovery tokens (test PPL 7.42 against 8.86). SwiGLU reaches the quality
  frontier only at 75-80% removal, beyond the quantized variants tested.
- **Compute:** SwiGLU removes up to 52% of linear multiply-accumulate
  operations per token. Weight-only quantization keeps the matrix shapes and
  removes none.
- **Speed:** whether the compute reduction becomes wall-clock speed is
  pending ([Speed-1](../experiments/model/baseline/speed-1.md)).

Full tables: [results snapshot 2026-10-15](../results/2026-10-15-results.md).

## What improved quality, in order

1. Teacher-derived initialization over random initialization.
2. Adequate recovery tokens.
3. Width allocation based on measured replacement damage, including keeping
   sensitive layers dense.
4. The constant 3e-5 recovery configuration.
5. Larger local calibration budgets.
6. Output reconstruction, RMSNorm, attention LoRA, and full-body recovery were
   neutral, inconsistent, or inefficient in the tested forms.

The ranking is qualitative: the interventions were not evaluated in a single
factorial experiment. The controlled comparisons inside each report are the
primary evidence.

## Known limits

- One model revision and one experiment seed; no confidence intervals across
  independent training runs.
- Early selection decisions used a small WikiText prefix; later experiments add
  full validation, test, and task evaluation.
- SwiGLU-6 and SwiGLU-7 are not a controlled comparison of 128 versus 8K
  recovery sequence length.
- All replacements are homogeneous SwiGLU; heterogeneous operators are not yet
  tested.

## Reading order

For the chronological narrative with all tables, see the
[SwiGLU progression](../experiments/model/swiglu/swiglu-progression.md). For
findings grouped by topic, see the
[SwiGLU results overview](../experiments/model/swiglu/swiglu-results.md).
