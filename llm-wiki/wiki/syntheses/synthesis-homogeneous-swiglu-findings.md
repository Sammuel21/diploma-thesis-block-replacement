---
id: synthesis-homogeneous-swiglu-findings
title: Homogeneous SwiGLU Findings and Methodology Boundary
summary: Attribution boundary of the completed homogeneous SwiGLU-1 through SwiGLU-8 family (what is borrowed versus project-defined), with links to the docs that hold its results.
type: synthesis
status: review
created: 2026-10-02
updated: 2026-10-15

epistemic:
  role: mixed
  confidence: medium
  verification:
    - source-checked
    - experiment-backed

sources:
  - source_id: src-minitron-2024
    locator: "Section 2.2; Sections 3 and 4.3; Appendix A.3"
    relation: contextualizes
  - source_id: src-grafting-2025
    locator: "Sections 3.1-3.3"
    relation: contextualizes
  - source_id: src-modegpt-2025
    locator: "Sections 3.1-3.3; Algorithm 1; Equations 10-11"
    relation: contextualizes
  - source_id: src-lora-2021
    locator: "Sections 4.1-4.2"
    relation: contextualizes
related:
  - "[[method-discrete-width-curve-allocation]]"
  - "[[method-global-to-local-operator-budget-allocation]]"
  - "[[comparison-recovery-objective-trainable-scope]]"
  - "[[concept-replacement-error-propagation]]"
  - "[[experiment-initial-block-compression-study]]"
  - "[[experiment-swiglu-operator-design-progression]]"
  - "[[implementation-maintained-mlp-replacement-package]]"
  - "[[method-two-stage-operator-grafting]]"
  - "[[method-post-pruning-knowledge-distillation]]"
supersedes: []
superseded_by: []
---

# Homogeneous SwiGLU Findings and Methodology Boundary

## Overview

**Status as of 2026-10-15.** The homogeneous SwiGLU-1 through SwiGLU-8
family is complete, and so is the separate
[Quantization-1](../../../docs/experiments/model/baseline/quantization-1.md)
baseline. This closes a baseline experiment family, not every possible SwiGLU
research question. The studies use pinned SmolLM2-1.7B, protect layers 0 and
23, and allocate narrower SwiGLUs across eligible layers 1-22. Removal targets
refer to eligible MLP parameters, not whole-model removal or weight
quantization.

The working homogeneous control carried forward is teacher-subset local
initialization, discrete width allocation, and replacement-only T=1 KL
recovery (S7-0). This page keeps only the methodology and citation boundary;
all results live in `docs/`.

## Where the Results Live

- [`swiglu-progression.md`](../../../docs/experiments/model/swiglu/swiglu-progression.md):
  the chronology, what each study resolved, and its evidence records.
- [`swiglu-results.md`](../../../docs/experiments/model/swiglu/swiglu-results.md):
  the family lookup of headline numbers.
- [`swiglu-6.md`](../../../docs/experiments/model/swiglu/swiglu-6.md):
  fixed 20% and 50% topologies recovered to 1B tokens.
- [`swiglu-7.md`](../../../docs/experiments/model/swiglu/swiglu-7.md): the three
  recovery scopes at four targets and the S7-0 working control.
- [`swiglu-8.md`](../../../docs/experiments/model/swiglu/swiglu-8.md): the
  local-fitting configuration selection (Part A), fitted versus slice-only
  recovery at 100M tokens (Part B), the reproducibility check, and the open
  1B follow-up.

Limitations and evidence boundaries for each study are stated in those docs.

## What Is Borrowed and What Is Project-Defined?

| Component | Prior-work boundary | Project contribution / adaptation |
| --- | --- | --- |
| Local fit followed by integrated recovery | Grafting's two-stage pattern, evaluated in diffusion transformers. [src-grafting-2025, Sections 3.1-3.3] | Causal-LLM MLP fitting and teacher-logit recovery protocol, not an exact reproduction of diffusion task fine-tuning. |
| Teacher-neuron initialization | Minitron supplies activation-based width-pruning precedent. [src-minitron-2024, Section 2.2] | Rank by post-gating activation RMS times down-column L2 norm; copy aligned gate/up/down slices, then fit them. This score is not Minitron's exact aggregation rule. |
| Earlier ranked allocation | MoDeGPT supplies the importance-to-sparsity softmax rule. [src-modegpt-2025, Section 3.3, Equations 10-11] | MLP scope, scores, bounds, integer-width realization, and cap bookkeeping are adaptations. |
| Output-aware initialization in SwiGLU-5 | MoDeGPT jointly selects MLP channels and reconstructs the down projection. [src-modegpt-2025, Sections 3.1-3.2, Algorithm 1] | The tested residual-mean bias and gradient-fitted down-only phase are an inspired alternative, not its ridge-leverage/Nyström solver. |
| Teacher-logit KL recovery | Minitron evaluates forward KLD and logit-only width-pruning recovery. [src-minitron-2024, Sections 3 and 4.3, Appendix A.3] | Scope, optimizer, LR, data, and budgets were chosen and tested here; replacement-only updates are not Minitron's full-student recipe. |
| Attention LoRA | Low-rank updates on frozen base weights are LoRA. [src-lora-2021, Sections 4.1-4.2] | Rank 16 on q/k/v/o plus full MLP/RMSNorm updates is the S7-2 design, not a new adaptation algorithm. |
| Local fit versus recovery alone | Grafting tests random initialization, local regression alone, and local regression plus fine-tuning, but not fine-tuning without local regression; its no-local-fit baseline is random weights. [src-grafting-2025, Section 3.3; Table 2] Minitron recovers pruned students by logit distillation without a local operator-fitting stage. [src-minitron-2024, Sections 3 and 4.3] | SwiGLU-8's slice-only arm (teacher slice, then KL recovery) against the fitted arm isolates the local-fit stage for causal-LLM MLP replacement. This is an ablation within this project, not a reproduction of either source. |
| Discrete width allocation | Knapsack and isotonic regression are standard algorithms. | [[method-discrete-width-curve-allocation]] defines the singleton-KL additive surrogate and fixed width grid. Literature novelty has not been established. |

## Relationships

- [[method-discrete-width-curve-allocation]] owns the reusable allocation
  definition; [[method-global-to-local-operator-budget-allocation]] preserves
  the earlier softmax-cap proposal without treating it as the current default.
- [[comparison-recovery-objective-trainable-scope]] separates loss placement
  from parameter scope and records the three S7 variants.
- [[concept-replacement-error-propagation]] explains why integrated evaluation
  remains necessary after singleton-based selection.
- [[experiment-initial-block-compression-study]] and
  [[experiment-swiglu-operator-design-progression]] preserve the initial design
  and generic singleton evidence, not the completed family results.
- [[implementation-maintained-mlp-replacement-package]] points to the READMEs
  that map the current scientific code and runners.
- [[method-two-stage-operator-grafting]] defines the local-fit-then-recovery
  pattern whose first stage SwiGLU-8 ablates.
- [[method-post-pruning-knowledge-distillation]] describes recovery without a
  local operator fit, the closest prior-work analogue of SwiGLU-8's slice-only
  arm.

## Sources

- `src-minitron-2024` - Section 2.2, Sections 3 and 4.3, Appendix A.3.
- `src-grafting-2025` - Sections 3.1-3.3.
- `src-modegpt-2025` - Sections 3.1-3.3, Algorithm 1, Equations 10-11.
- `src-lora-2021` - Sections 4.1-4.2; selectively checked, not fully ingested.
- Project evidence, checked 2026-10-02 - original JSON records listed in the
  linked chronology, including
  [SwiGLU-5 search](../../../data/results/workflows/model/swiglu-5/search/run-001.json),
  [20% confirmation](../../../data/results/workflows/model/swiglu-5/confirmation/run-001-target-0.2.json),
  [50% confirmation](../../../data/results/workflows/model/swiglu-5/confirmation/run-001-target-0.5.json),
  [SwiGLU-6 evaluation](../../../data/results/workflows/model/swiglu-6/evaluation-001.json),
  and the linked SwiGLU-7 preparation/production records. SwiGLU-8 evidence,
  checked 2026-10-12: the
  [merged block check](../../../data/results/workflows/model/swiglu-8/block-check-001/result.json),
  [preparation](../../../data/results/workflows/model/swiglu-8/prepare-001/result.json),
  and the 16 `{fitted,slice}-target-<t>-run-001/result.json` records in
  [the local copy](../../../data/results/workflows/model/swiglu-8/), summarized
  in the [SwiGLU-8 report](../../../docs/experiments/model/swiglu/swiglu-8.md).
  Method implementations
  are in [modules.py](../../../src/mlp_replacement/operators/modules.py),
  [training.py](../../../src/mlp_replacement/operators/training.py), and
  [swiglu_7.py](../../../workflows/runs/model/swiglu/swiglu_7.py), and
  [swiglu_8.py](../../../workflows/runs/model/swiglu/swiglu_8.py).
