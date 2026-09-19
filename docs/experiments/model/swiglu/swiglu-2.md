---
metadata_version: 1
title: SwiGLU Allocation Study Workflow
type: experiment-workflow
category: experiments/model/swiglu
status: draft
created: 2026-09-10
modified: 2026-09-19
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
sources:
  notebooks:
    - notebooks/model/swiglu/swiglu-2.ipynb
  artifacts:
    - data/results/notebook-model-study/swiglu-2.json
  reference_artifacts:
    - data/results/notebook-model-study/swiglu-compression-optimized.json
---

# SwiGLU Allocation Study Workflow

Notebook: [swiglu-2.ipynb](../../../../notebooks/model/swiglu/swiglu-2.ipynb)

Status: implemented and executed in the notebook. The schema-1 result artifact
is present. A sequential Python migration of the same dependent stages is also
implemented under `workflows/runs/model/`.

## Purpose

The previous SwiGLU workflow showed that local fitting can be improved, but its
nonuniform allocation policies were not clearly better than uniform widths.
This workflow searches for a more informative block score and a less extreme
allocation curve while keeping the global target fixed at 50% eligible
MLP-parameter removal.

It answers four connected questions:

1. Does model-level damage from replacing one block rank blocks better than BI?
2. Does that ranking change between moderate and aggressive compression?
3. How strongly should the allocator separate important and unimportant blocks?
4. Do the promising allocations remain good after full local fitting and
   model-wide recovery?

## Pipeline at a glance

```text
optimized SwiGLU artifact (fixed model, data roles, BI scores, rankings)
|
v
1. DUAL-WIDTH PROBES -- full 98,304-pair fits
   every eligible block at retained widths 25% and 50%
   -> local NMSE/cosine + singleton KL/loss/perplexity
   -> six candidate block-importance scores
|
v
2. WIDTH RESPONSE
   representative best/middle/worst blocks
   -> retained widths 20%, 30%, ..., 80%
   -> inspect where local and singleton damage accelerates
|
v
3. ALLOCATION CONSTRUCTION
   six scores x temperatures {1, 2, 4} + uniform
   -> 19 exact-budget policies at 50% eligible-MLP removal
|
v
4. REDUCED-BUDGET QUALIFICATION -- 24,576 pairs per fit
   fit and evaluate all policies on the allocation-selection split
   -> promote uniform + BI/KL/loss family representatives
|
v
5. MINIMUM-WIDTH ABLATION
   best unbounded policy + retained-width floors {30%, 40%}
   -> promote the best distinct bounded allocation
|
v
6. FULL-BUDGET FINALISTS -- 98,304 pairs per fit
   model evaluation -> one-epoch recovery for every finalist -> re-evaluation
   -> winner = lowest post-recovery teacher KL
|
v
swiglu-2.json (fits, histories, allocations, evaluations, winner, runtime)
```

This is a sequential tournament. Later fits and the final recovery set depend
on promotion decisions made from earlier measurements, so the stages cannot be
interpreted as independent sweeps.

## Inherited controls and data roles

The notebook loads
[swiglu-compression-optimized.json](../../../../data/results/notebook-model-study/swiglu-compression-optimized.json)
to inherit the pinned model revision, data configuration, eligible layers,
teacher-neuron rankings, operator fitting configuration, and recovery
configuration.

The full local-fitting budget is 384 calibration batches, or 98,304 activation
pairs per block. A cheaper screening fit uses 96 batches, or 24,576 pairs. A
separate 12-batch allocation-selection split compares candidate compressed
models. Operator validation, recovery, recovery validation, and final model
validation retain their own roles.

The selection split is important: using it to compare allocation policies
avoids choosing a policy from the same model-validation measurements later
used to report its quality.

Activations are processed in groups inherited from the optimized artifact,
currently six blocks at a time. They are kept in CPU memory only and released
after each group; no activation data is written to disk.

## 1. Dual-width block probes

Every eligible block is fitted independently at two retained widths using
teacher-derived initialization and the full calibration budget:

- 50% is the natural reference because it equals the average retained width
  implied by the global budget;
- 25% is an aggressive stress test that reveals blocks whose behavior fails
  rapidly under stronger compression.

Each fitted operator is evaluated in two ways:

- local NMSE and cosine compare its output with the original MLP output;
- singleton KL, loss delta, and perplexity measure the whole model after only
  that block is replaced.

The 50% operators are retained because together they form the full-budget
uniform control. The 25% weights are released after their measurements are
recorded because later policies usually assign different widths; keeping those
weights would consume memory without avoiding the later fits.

Six candidate allocation scores are assembled:

```text
canonical BI
residual-aware MLP BI
singleton KL at 25% and 50% width
singleton loss delta at 25% and 50% width
```

Perplexity is still reported, but it is not a separate allocation score because
it is the exponential of loss and therefore gives the same ordering as loss.
KL supplies complementary information by directly measuring deviation from
the dense teacher distribution.

## 2. Representative block width study

Two block cohorts are selected automatically:

- one from singleton KL at 50% width;
- one from singleton KL at 25% width.

Within each cohort, `best`, `middle`, and `worst` mean the blocks with minimum,
median, and maximum singleton KL respectively. These labels are therefore tied
to a declared score and width rather than chosen subjectively.

Each unique selected block is swept from 20% through 80% retained width in
10-percentage-point increments. The already fitted 50% point is reused. The
study plots local NMSE and singleton KL for the two cohorts, showing whether
local approximation curves and model-level damage agree and whether the
aggressive and moderate rankings identify the same blocks.

## 3. Score and temperature allocation sweep

For each of the six candidate scores, the workflow tests allocation
temperatures 1.0, 2.0, and 4.0, plus one uniform control. This produces 19
policies.

Scores are rank-normalized before allocation so differently scaled metrics can
use the same mechanism. Higher-scoring blocks receive more retained width. A
temperature of 1.0 creates the strongest contrast in this sweep; larger values
move the allocation closer to uniform. This rank normalization and exponential
allocator are project-defined methodology, motivated by nonuniform global
allocation rather than copied directly from MoDeGPT.

Every policy is reconciled to the same global 50% MLP-removal target using
whole SwiGLU neurons. Thus a policy can win only by distributing a matched
budget better, not by retaining more parameters.

## 4. Reduced-budget qualification

Fully fitting 19 policies across 22 eligible blocks would require 418 expensive
full-budget operator fits. Most candidates are expected to lose, so all 19 are
first fitted with the 25% calibration proxy and evaluated on the dedicated
selection split.

This is a qualifying round, not the final result. It reduces research compute
while allowing several distinct families to advance:

- the uniform control;
- the best BI-family policy by selection KL;
- the best singleton-KL-family policy by selection KL;
- the best singleton-loss-family policy by selection loss delta; and
- the best nonuniform policy overall by selection KL.

Duplicate selections are removed, normally leaving about four or five
finalists. Promoting representatives from each family reduces the risk that a
single noisy proxy decision eliminates an entire approach.

## 5. Minimum-width ablation

The main 19-policy sweep has no minimum retained width. After the best
unbounded policy is known, only that policy is reallocated with 30% and 40%
minimum-width floors. A floor is skipped if it produces the same allocation as
an already tested case.

The floor prevents the allocator from compressing any one block beyond a
declared limit while preserving the same total budget by redistributing removal
elsewhere. Testing it only on the best unbounded policy isolates whether the
winning allocation was too locally aggressive without multiplying the entire
score-temperature search. The best bounded variant joins the finalist set.

## 6. Full-budget confirmation and recovery

Every finalist receives the full 384-batch local fit. The uniform finalist
reuses the full-budget 50% probe operators; other finalists are refitted at
their assigned widths. All finalists are then evaluated on the model-validation
set before recovery.

Model-wide replacement-only KL recovery is applied separately to every
finalist. Recovery can change the ordering because some error patterns are
easier to correct jointly than others. Recovering only the pre-recovery winner
could therefore discard a policy that becomes best after training. The final
winner is selected by the lowest post-recovery teacher KL, with perplexity,
loss, compression, and local fit reported alongside it.

## Python workflow

The headless compute path is
[`workflows.runs.model.swiglu.allocation`](../../../../workflows/runs/model/swiglu/allocation.py),
driven by [`allocation.json`](../../../../workflows/configs/model/swiglu/allocation.json).
It implements the six dependent stages above in one process because later
promotion and finalist choices depend on earlier measurements. The notebook is
retained as the explanatory and loading frontend. The runner writes the
schema-1 science artifact and a sibling crash-aware `.run.json` operational
record. The present artifact was produced by the notebook. Numerical parity of
the Python migration still requires its first successful GPU execution and a
comparison against that notebook artifact.

## Reporting and artifact

The notebook records:

- dual-width probe metrics, training histories, and score rankings;
- representative block membership and width curves;
- every score-temperature width allocation;
- reduced-budget screening and minimum-width-ablation results;
- full-budget finalist fits and pre/post-recovery model evaluations;
- total-model and MLP-parameter reductions; and
- capture, fitting, evaluation, recovery, and total wall-clock times.

The notebook output is
`data/results/notebook-model-study/swiglu-2.json`, schema version 1. It is the
authoritative record for the completed notebook run, including its winning
policy and all promotion decisions.

## Configuration appendix

These values come from the notebook, its completed artifact, and the optimized
reference artifact.

| Setting | Value |
| --- | --- |
| Execution status | Notebook artifact present; Python migration not yet GPU-validated |
| Model | `HuggingFaceTB/SmolLM2-1.7B` |
| Model and tokenizer revision | `effd688a12921b4cc83e3312b6feb579f70f9c71` |
| Reference artifact | `swiglu-compression-optimized.json`, schema 3 |
| Calibration and recovery corpus | C4 train, shard `en/c4-train.00000-of-01024.json.gz` |
| Model validation corpus | WikiText-2 raw validation |
| Sequence length / capture batch | 128 tokens / 2 sequences |
| Full calibration | 384 batches / 98,304 operator pairs |
| Screening calibration | 96 batches / 24,576 operator pairs |
| Operator validation | 24 batches / 6,144 pairs |
| Allocation selection | Separate 12-batch partition |
| Recovery / recovery validation | 64 / 24 batches |
| Model validation / test | 24 / 0 batches |
| Eligible / protected layers | 1-22 / 0 and 23 |
| Replacement | Bias-free variable-width SwiGLU |
| Initialization | Importance-ranked teacher subset for every fit; no random-initialization sweep |
| Teacher-neuron rankings | Reused from the optimized reference artifact |
| Local optimizer | AdamW, learning rate `1e-3`, weight decay `0` |
| Local schedule | Constant, operator batch 2,048, maximum 64 epochs |
| Local selection | Early stopping patience 3, minimum delta 0 |
| Probe retained widths | 25% and 50% |
| Representative width sweep | 20%-80% in 10-percentage-point increments |
| Allocation scores | Canonical BI, residual-aware MLP BI, singleton KL at 25%/50%, singleton loss delta at 25%/50% |
| Allocation temperatures | 1.0, 2.0, and 4.0 |
| Main policy count | 19: six scores times three temperatures, plus uniform |
| Main MLP-removal target | 50% of eligible MLP parameters |
| Minimum-retention ablation | 30% and 40%, applied only to the best unbounded policy |
| Finalist promotion | Uniform; best BI, KL, and loss families; best nonuniform overall; best bounded variant |
| Activation storage | CPU, model-native dtype, six-block groups, no disk I/O |
| Recovery | Every finalist; replacement-only KL, AdamW, 1 epoch, learning rate `1e-5`, temperature 1.0 |
| Winner selection | Lowest post-recovery teacher KL |
| Seed | 21 |
| Planned artifact | `swiglu-2.json`, schema 1 |
