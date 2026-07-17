---
id: method-post-pruning-knowledge-distillation
title: Post-Pruning Knowledge Distillation
summary: Recovers a structurally pruned student by matching outputs and optionally intermediate states of the unpruned teacher.
type: method
status: review
created: 2026-07-17
updated: 2026-07-17

authorship:
  created_by: collaborative
  contributors:
    - researcher
    - llm

epistemic:
  role: prior-work
  confidence: high
  verification:
    - source-checked

scope:
  topics:
    - knowledge-distillation
    - compression-recovery
    - recovery-budget
    - loss-design
  granularities:
    - transformer-layer
    - model
    - cross-level
  pipeline_stages:
    - integration
    - recovery
    - evaluation

sources:
  - source_id: src-minitron-2024
    locator: "Section 3; Section 4.3; Appendix A.3-A.5"
    relation: evaluates

related:
  - "[[source-summary-minitron-2024]]"
  - "[[method-retraining-assisted-architecture-search]]"
  - "[[method-two-stage-operator-grafting]]"
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

- [[source-summary-minitron-2024]] provides the complete source context.
- [[method-retraining-assisted-architecture-search]] uses a fixed smaller
  recovery budget to compare architecture candidates before longer training.
- [[method-two-stage-operator-grafting]] also separates local structural change
  from model-level recovery, but initializes replacement operators through
  activation regression rather than producing a student through pruning.

## Sources

- `src-minitron-2024` - Section 3, Section 4.3, Appendix A.3-A.5
