---
id: comparison-recovery-objective-trainable-scope
title: Recovery Objective and Trainable Scope
summary: Distinguishes where a recovery loss is measured from which student parameters are updated in the project, Minitron, and Grafting.
type: comparison
status: review
created: 2026-08-15
updated: 2026-08-18

authorship:
  created_by: collaborative
  contributors:
    - researcher
    - llm

epistemic:
  role: synthesis
  confidence: high
  verification:
    - source-checked

scope:
  topics:
    - compression-recovery
    - knowledge-distillation
    - operator-grafting
    - trainable-parameter-scope
  granularities:
    - mlp-block
    - model
    - cross-level
  pipeline_stages:
    - integration
    - recovery
    - evaluation
    - analysis

sources:
  - source_id: src-minitron-2024
    locator: "Sections 2.3 and 3; Section 4.3; Appendix A.8"
    relation: contextualizes
  - source_id: src-grafting-2025
    locator: "Sections 3.1-3.3 and 5; Appendix B.1"
    relation: contextualizes

related:
  - "[[method-post-pruning-knowledge-distillation]]"
  - "[[method-two-stage-operator-grafting]]"
  - "[[experiment-baseline-operator-analysis]]"
supersedes: []
superseded_by: []
---

# Recovery Objective and Trainable Scope

## Overview

`Model-level` or `global` recovery can describe where a loss is measured
without specifying which parameters are updated. A teacher-student loss on the
final logits is model-level even when optimization is restricted to one
replacement module. This comparison keeps the recovery objective, student
architecture, and trainable-parameter scope separate.

## Comparison

| System | Local or structural stage | Recovery objective | Trainable parameter scope | Precise description |
| --- | --- | --- | --- | --- |
| Current project | Fit replacement MLPs from teacher activation pairs | Forward teacher-to-student KL on the integrated autoregressive model's final logits | Replacement modules only; original model parameters remain frozen | Model-level teacher-logit KD with replacement-only updates |
| Minitron | Structurally prune depth or width dimensions from the teacher | Conventional language-model training or KD; forward KLD is the strongest reported base-model loss | The pruned student is retrained rather than restricting updates to a replacement-only subset | Full-student post-pruning retraining or distillation |
| Grafting, main DiT experiments | Fit inserted operators by local activation regression | Original diffusion training objective after integrating the grafts | End-to-end fine-tuning of the integrated grafted model | Local operator initialization followed by full-model task recovery |
| Grafting, PixArt-Sigma case | Fit the replacement from synthetic activation pairs | Integrated text-to-image recovery | Rank-64 LoRA parameters because full fine-tuning is memory-constrained | Parameter-efficient exception to the main Grafting recovery scope |

Minitron trains the structurally pruned student and leaves parameter-efficient
recovery such as LoRA as future work. It does not evaluate the project's
replacement-only update rule. [src-minitron-2024, Sections 2.3 and 3; Section
4.3; Appendix A.8]

Grafting separates local activation regression from integrated recovery. Its
main DiT experiments use end-to-end fine-tuning, whereas its PixArt-Sigma case
uses rank-64 LoRA. Its integrated objective is the diffusion task objective,
not autoregressive teacher-logit KL. [src-grafting-2025, Sections 3.1-3.3 and
5; Appendix B.1]

## Current Project Implementation

**Observed repository implementation.** The maintained recovery function
caches teacher logits and computes KL from the original model to the integrated
student. Before optimization, it freezes every student parameter and then
enables gradients only for modules identified by the replacement target paths.
The exploratory width-recovery notebook follows the same trainable scope.

Relevant project artifacts:

- [`src/mlp_replacement/compression/recovery.py`](../../../src/mlp_replacement/compression/recovery.py)
  implements teacher-logit caching, the KL loss, and replacement-only recovery;
- [`notebooks/block/baseline-experiments.ipynb`](../../../notebooks/block/baseline-experiments.ipynb)
  applies the same scope in the exploratory width-recovery trajectories.

This implementation observation records behavior, not an empirical claim that
replacement-only recovery is equivalent or superior to full-student training.

## Interpretation Boundary

The project shares the two-stage local-then-integrated structure of Grafting
and the teacher-logit recovery objective evaluated by Minitron, but it does not
duplicate either paper's full recovery protocol. Comparisons of recovery
budgets must therefore report at least the objective, trainable parameter
scope, optimizer steps, batch or token budget, and data distribution.

The phrase `end-to-end retraining` should be reserved for a protocol that
updates the complete integrated student. The current project protocol should
be named `model-level teacher-logit KD with replacement-only updates`.

## Limitations

- The Minitron evidence concerns structured pruning, not learned operator
  replacement.
- The Grafting evidence concerns diffusion transformers, not autoregressive
  LLMs.
- The project implementation can change; this comparison describes the
  repository state inspected on 2026-08-15.

## Relationships

- [[method-post-pruning-knowledge-distillation]] documents Minitron's recovery
  loss design and full pruned-student retraining context.
- [[method-two-stage-operator-grafting]] documents Grafting's local activation
  fitting and integrated fine-tuning stages.
- [[experiment-baseline-operator-analysis]] applies the project's
  replacement-only model-level KD scope to reduced-SwiGLU recovery curves.

## Sources

- `src-minitron-2024` - Sections 2.3 and 3; Section 4.3; Appendix A.8
- `src-grafting-2025` - Sections 3.1-3.3 and 5; Appendix B.1
- Project implementation - `src/mlp_replacement/compression/recovery.py` and
  `notebooks/block/baseline-experiments.ipynb`, inspected 2026-08-15
