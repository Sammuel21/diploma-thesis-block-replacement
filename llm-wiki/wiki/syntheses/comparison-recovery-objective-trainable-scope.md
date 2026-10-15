---
id: comparison-recovery-objective-trainable-scope
title: Recovery Objective and Trainable Scope
summary: Separates model-level loss placement from replacement-only, transformer-body, and LoRA-assisted updates, with source context and completed SwiGLU-7 evidence.
type: comparison
status: review
created: 2026-08-15
updated: 2026-10-15

epistemic:
  role: mixed
  confidence: high
  verification:
    - source-checked
    - experiment-backed

sources:
  - source_id: src-minitron-2024
    locator: "Sections 2.3 and 3; Section 4.3; Appendix A.8"
    relation: contextualizes
  - source_id: src-grafting-2025
    locator: "Sections 3.1-3.3 and 5; Appendices B.1 and D.2; Table D.1"
    relation: contextualizes
  - source_id: src-lora-2021
    locator: "Sections 4.1-4.2"
    relation: defines

related:
  - "[[method-post-pruning-knowledge-distillation]]"
  - "[[method-two-stage-operator-grafting]]"
  - "[[experiment-baseline-operator-analysis]]"
  - "[[synthesis-homogeneous-swiglu-findings]]"
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
| Project short-budget baseline / S7-0 | Fit replacement MLPs from teacher activation pairs | Forward teacher-to-student KL on the integrated autoregressive model's final logits | Replacement modules only; retained dense MLPs and original model parameters remain frozen | Model-level teacher-logit KD with replacement-only updates |
| Project S7-1 | Same fitted start as S7-0 at each target | Same teacher-logit KL | All transformer-body parameters; tied embeddings/head frozen | Transformer-body recovery, not full-model fine-tuning |
| Project S7-2 | Same fitted start plus zero-initialized attention updates | Same teacher-logit KL | All MLPs/RMSNorms plus rank-16 attention q/k/v/o LoRA; attention base weights and tied embeddings/head frozen | MLP/norm recovery with LoRA-assisted attention |
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
5; Appendix B.1 for DiT settings; Appendix D.2, Table D.1 for PixArt LoRA]

## Current Project Implementation

**Observed repository implementation, checked 2026-10-02.** The shared
short-budget recovery function caches teacher logits, freezes the student,
and enables gradients only for replacement target paths. SwiGLU-7 instead
computes the dense teacher online over a finite 1B-token stream and explicitly
selects one of the three scopes above. Loss placement does not determine which
parameters are trainable, nor whether teacher targets are cached.

LoRA's frozen-base, low-rank update is source-derived; S7-2's rank, projection
set, and joint MLP/RMSNorm updates are project experiment choices, not a new
LoRA algorithm. [src-lora-2021, Sections 4.1-4.2]

Relevant project artifacts:

- [`src/mlp_replacement/compression/recovery.py`](../../../src/mlp_replacement/compression/recovery.py)
  implements teacher-logit caching, the KL loss, and replacement-only recovery;
- [`notebooks/block/baseline/baseline-experiments.ipynb`](../../../notebooks/block/baseline/baseline-experiments.ipynb)
  applies the same scope in the exploratory width-recovery trajectories.
- [`workflows/runs/model/swiglu/swiglu_7.py`](../../../workflows/runs/model/swiglu/swiglu_7.py)
  selects production trainable groups and online-teacher recovery;
- [`workflows/configs/model/swiglu/swiglu-7.json`](../../../workflows/configs/model/swiglu/swiglu-7.json)
  fixes scope, native-8K recovery, seed, optimizer, and token budget.

**Empirical finding.** The completed SwiGLU-7 runs, reported in
[`swiglu-7.md`](../../../docs/experiments/model/swiglu/swiglu-7.md), favor S7-0
on likelihood at lower resource use; broader scopes give small, mixed
downstream gains. This is evidence under the fixed 1B-token, single-seed
protocol, not proof that replacement-only updates always outperform broader
adaptation.

## Interpretation Boundary

The project shares the two-stage local-then-integrated structure of Grafting
and the teacher-logit recovery objective evaluated by Minitron, but it does not
duplicate either paper's full recovery protocol. Comparisons of recovery
budgets must therefore report at least the objective, trainable parameter
scope, optimizer steps, batch or token budget, and data distribution.

Use explicit scope names. S7-0 is `model-level teacher-logit KD with
replacement-only updates`; S7-1 is `transformer-body recovery`, not an update
of every model parameter. Unqualified `full-model` or `end-to-end retraining`
would obscure its frozen tied embeddings/head.

## Limitations

- The Minitron evidence concerns structured pruning, not learned operator
  replacement.
- The Grafting evidence concerns diffusion transformers, not autoregressive
  LLMs.
- The project result is specific to one model, seed, optimizer recipe, and
  budget; neither paper predicts the ordering of the S7 scopes.
- Repository observations were refreshed on 2026-10-02. They do not establish
  independent reproduction of the preserved production results.

## Relationships

- [[method-post-pruning-knowledge-distillation]] documents Minitron's recovery
  loss design and full pruned-student retraining context.
- [[method-two-stage-operator-grafting]] documents Grafting's local activation
  fitting and integrated fine-tuning stages.
- [[experiment-baseline-operator-analysis]] applies the project's
  replacement-only model-level KD scope to reduced-SwiGLU recovery curves.
- [[synthesis-homogeneous-swiglu-findings]] records the attribution boundary of
  the completed family and the working replacement-only control; its results
  are in `docs/experiments/model/swiglu/`.

## Sources

- `src-minitron-2024` - Sections 2.3 and 3; Section 4.3; Appendix A.8
- `src-grafting-2025` - Sections 3.1-3.3 and 5; Appendices B.1 and D.2,
  Table D.1
- `src-lora-2021` - Sections 4.1-4.2
- Project implementation - `src/mlp_replacement/compression/recovery.py` and
  the linked SwiGLU-7 runner/configuration, inspected 2026-10-02; preserved
  production results linked from `swiglu-7.md`.
