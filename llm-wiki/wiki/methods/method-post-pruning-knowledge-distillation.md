---
id: method-post-pruning-knowledge-distillation
title: Post-Pruning Knowledge Distillation
summary: Recovers a structurally pruned student by matching outputs and optionally intermediate states of the unpruned teacher.
type: method
status: review
created: 2026-07-17
updated: 2026-10-15

epistemic:
  role: mixed
  confidence: high
  verification:
    - source-checked
    - experiment-backed

sources:
  - source_id: src-minitron-2024
    locator: "Section 3; Section 4.3; Appendix A.3-A.5"
    relation: evaluates

related:
  - "[[source-summary-minitron-2024]]"
  - "[[method-retraining-assisted-architecture-search]]"
  - "[[method-two-stage-operator-grafting]]"
  - "[[comparison-recovery-objective-trainable-scope]]"
  - "[[synthesis-homogeneous-swiglu-findings]]"
supersedes: []
superseded_by: []
---

# Post-Pruning Knowledge Distillation

## Overview

Minitron uses the uncompressed model as a teacher and the structurally pruned
model as a student. Recovery trains the student to match teacher logits and,
for selected configurations, intermediate hidden states. The paper calls this
post-pruning process retraining. [src-minitron-2024, Section 3]

## Definition or Description

The paper formulates three possible loss families:

- conventional next-token cross-entropy against ground-truth labels;
- a logit loss between teacher and student token distributions; and
- intermediate-state losses between mapped teacher and student locations.

When hidden dimensions differ, a learned shared linear transform maps student
states to the teacher dimension. Hidden-state comparisons use post-LayerNorm
states. The general formulation combines language-model, logit, and
intermediate losses, with a dynamic coefficient to balance the latter two.
[src-minitron-2024, Section 3; Figure 4]

## Evidence and Rationale

Under the paper's iso-compute 4B comparison, pruning followed by distillation
outperforms both a randomly initialized model and conventional post-pruning
training on HellaSwag and especially MMLU. [src-minitron-2024, Section 4.3;
Table 11]

For the examined base models, forward KLD over logits outperforms reverse KLD,
MSE, cosine logit loss, and a combination with conventional LM loss. A softmax
temperature of 1.0 works best, and restricting the loss to low top-K logits
does not improve results. [src-minitron-2024, Appendix A.3; Tables 15-16]

For models without substantial depth reduction, logit-only distillation
performs as well as or better than adding intermediate losses. Severe depth
reduction can benefit from a carefully selected encoder-block output mapping,
while several other intermediate signals provide no benefit in the paper's
ablations. [src-minitron-2024, Section 4.3; Appendix A.4; Tables 17-18]

## Project Evidence: Adding a Local Fit Before Distillation

The paragraphs above report prior work. This section records project
evidence, labelled separately.

**Empirical finding.** SwiGLU-8's slice-only arm comes closest in this
project to pruning followed by logit distillation. Width slices are chosen by
an activation-based importance score and recovered with temperature-1 forward
KL from the dense teacher. It differs from Minitron in three ways:

- the score aggregation;
- only the replacement MLPs train;
- the budget is 100M tokens.

Adding a local operator fit before the same distillation lowered the 8K KL at
100M tokens at every target from 20% to 80% eligible-MLP removal. The gap was
still shrinking at 100M. The numbers are in the
[SwiGLU-8 report](../../../docs/experiments/model/swiglu/swiglu-8.md) (Part B
results).

**Synthesis.** The evidence suggests that, at short and moderate recovery
budgets, a local reconstruction stage is a cheap complement to post-pruning
distillation. It does not test Minitron's own recipe or its multi-billion-token
budgets.

## Limitations and Open Issues

Calibration and recovery must not be conflated. Minitron uses 1,024 samples for
importance estimation but approximately 1.8B tokens for default lightweight
retraining and up to 94B tokens for final models. Its term `lightweight` is
relative to full pretraining, not to a small local experiment.

The paper evaluates training the pruned student model. It mentions LoRA as a
possible architecture-search optimization but leaves parameter-efficient
recovery to future work. It therefore does not establish that training only
replacement operators, only neighboring blocks, or LoRA adapters provides
equivalent recovery. [src-minitron-2024, Sections 2.3 and 3]

The paper's student is produced by structured pruning, not by substituting a
new learned operator. Transferring the loss recommendations to block
replacement requires direct thesis experiments.

## Relationships

- [[synthesis-homogeneous-swiglu-findings]] records the SwiGLU-8 comparison
  of distillation with and without a preceding local fit.
- [[source-summary-minitron-2024]] provides the complete source context.
- [[method-retraining-assisted-architecture-search]] uses a fixed smaller
  recovery budget to compare architecture candidates before longer training.
- [[method-two-stage-operator-grafting]] also separates local structural change
  from model-level recovery, but initializes replacement operators through
  activation regression rather than producing a student through pruning.
- [[comparison-recovery-objective-trainable-scope]] distinguishes Minitron's
  full pruned-student retraining from the project's replacement-only updates.

## Sources

- `src-minitron-2024` - Section 3, Section 4.3, Appendix A.3-A.5
- Project evidence, checked 2026-10-12 - the SwiGLU-8 records summarized in
  the [SwiGLU-8 report](../../../docs/experiments/model/swiglu/swiglu-8.md).
