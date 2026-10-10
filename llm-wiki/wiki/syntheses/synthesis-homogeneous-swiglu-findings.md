---
id: synthesis-homogeneous-swiglu-findings
title: Homogeneous SwiGLU Findings and Methodology Boundary
summary: Records the completed SwiGLU-1 through SwiGLU-7 progression, its working control, attribution, and limits without duplicating experiment reports.
type: synthesis
status: review
created: 2026-10-02
updated: 2026-10-02

authorship:
  created_by: collaborative
  contributors:
    - researcher
    - llm

epistemic:
  role: mixed
  confidence: medium
  verification:
    - source-checked
    - experiment-backed

scope:
  topics:
    - swiglu
    - mlp-block-replacement
    - global-sparsity-allocation
    - compression-recovery
    - trainable-parameter-scope
    - quality-preservation
  granularities:
    - neuron
    - mlp-block
    - model
  pipeline_stages:
    - selection
    - replacement
    - integration
    - recovery
    - evaluation
    - analysis

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
supersedes: []
superseded_by: []
---

# Homogeneous SwiGLU Findings and Methodology Boundary

## Overview

**Status as of 2026-10-02.** The configured homogeneous SwiGLU-1 through
SwiGLU-7 progression is complete. This closes a baseline experiment family,
not every possible SwiGLU research question. The studies use pinned
SmolLM2-1.7B, protect layers 0 and 23, and allocate narrower SwiGLUs across
eligible layers 1-22. Removal targets refer to eligible MLP parameters, not
whole-model removal or weight quantization.

The [chronology](../../../docs/experiments/model/swiglu/swiglu-progression.md),
[results report](../../../docs/experiments/model/swiglu/swiglu-results.md),
individual guides, and [notebooks](../../../notebooks/model/swiglu/) own the
detailed tables and plots. This page retains only decisions, evidence anchors,
and the methodology/citation boundary needed by future agents.

## Experimental Progression

**Empirical findings, bounded to the recorded protocols.**

| Study | Question resolved / useful handoff |
| --- | --- |
| SwiGLU-1 | Teacher-neuron initialization, not extra random-fit data alone, made assembled replacement viable; BI allocation did not beat uniform. |
| SwiGLU-2 | Operator-conditioned singleton KL gave a better ranked-width policy than the earlier influence scores at the tested 50% budget. |
| SwiGLU-3 | More local pairs and 100M recovery tokens helped; substantial dense-quality gaps remained. |
| SwiGLU-4 | Constant `3e-5`, pure T=1 teacher KL was the efficient recovery recipe; added scope did not justify the small short-budget gains. |
| SwiGLU-5 | Discrete width curves improved the matched-budget frontier; output reconstruction and composition refitting did not establish a useful default advantage. |
| SwiGLU-6 | Fixed 20% and 50% topologies improved through 1B tokens; full held-out evaluation still showed a dense gap. |
| SwiGLU-7 | Independent preparation plus native-8K, 1B-token recovery compared three scopes at four targets; replacement-only was the strongest likelihood/resource control. |

SwiGLU-5's matched-recipe C0/C2 comparison supports the allocation improvement.
Direct SwiGLU-3 versus SwiGLU-5 improvements also include the changed recovery
recipe; they do not isolate allocation alone.

## What Is Borrowed and What Is Project-Defined?

| Component | Prior-work boundary | Project contribution / adaptation |
| --- | --- | --- |
| Local fit followed by integrated recovery | Grafting's two-stage pattern, evaluated in diffusion transformers. [src-grafting-2025, Sections 3.1-3.3] | Causal-LLM MLP fitting and teacher-logit recovery protocol, not an exact reproduction of diffusion task fine-tuning. |
| Teacher-neuron initialization | Minitron supplies activation-based width-pruning precedent. [src-minitron-2024, Section 2.2] | Rank by post-gating activation RMS times down-column L2 norm; copy aligned gate/up/down slices, then fit them. This score is not Minitron's exact aggregation rule. |
| Earlier ranked allocation | MoDeGPT supplies the importance-to-sparsity softmax rule. [src-modegpt-2025, Section 3.3, Equations 10-11] | MLP scope, scores, bounds, integer-width realization, and cap bookkeeping are adaptations. |
| Output-aware initialization in SwiGLU-5 | MoDeGPT jointly selects MLP channels and reconstructs the down projection. [src-modegpt-2025, Sections 3.1-3.2, Algorithm 1] | The tested residual-mean bias and gradient-fitted down-only phase are an inspired alternative, not its ridge-leverage/Nyström solver. |
| Teacher-logit KL recovery | Minitron evaluates forward KLD and logit-only width-pruning recovery. [src-minitron-2024, Sections 3 and 4.3, Appendix A.3] | Scope, optimizer, LR, data, and budgets were chosen and tested here; replacement-only updates are not Minitron's full-student recipe. |
| Attention LoRA | Low-rank updates on frozen base weights are LoRA. [src-lora-2021, Sections 4.1-4.2] | Rank 16 on q/k/v/o plus full MLP/RMSNorm updates is the S7-2 design, not a new adaptation algorithm. |
| Discrete width allocation | Knapsack and isotonic regression are standard algorithms. | [[method-discrete-width-curve-allocation]] defines the singleton-KL additive surrogate and fixed width grid. Literature novelty has not been established. |

## Final Evidence and Working Control

**Empirical finding.** The
[SwiGLU-7 preparation](../../../data/results/workflows/model/swiglu-7/prepare-001/result.json)
and [12 production results](../../../data/results/workflows/model/swiglu-7/)
are `completed`; every trajectory reached exactly 1B tokens and completed
final evaluation. Preparation records no prior-experiment artifact dependency.
Local fitting and monitoring use 128-token sequences; production recovery uses
8,192. The scopes share a fresh token-zero start per target.

S7-0 (replacements only) wins 3/4 final monitoring-KL comparisons and 17/24
full WikiText split/context/target PPL comparisons. S7-1 updates the transformer
body but freezes tied embeddings/head; it is **not full-model fine-tuning**.
S7-2 updates all MLPs/RMSNorms and attention LoRA. Broader scopes show small,
mixed five-task macro gains, at most 0.49 percentage points over S7-0.
At 20%/50% removal, S7-0 full-test 8K PPL is 7.5553/8.8551 versus dense
6.9331: useful compression, not dense-equivalent quality.

**Working project direction recorded in the reports.** Carry forward
teacher-subset local initialization, discrete width allocation, and
replacement-only T=1 KL recovery as the homogeneous control. Heterogeneous
operators, stronger interaction-aware allocation, and
[Quantization-1](../../../docs/experiments/model/baseline/quantization-1.md)
are separate comparisons; quantization is implemented but its scientific GPU
run remains pending. No advantage of those future studies is claimed here.

## Limitations and Reproducibility

- One model and seed do not establish universality. Task example-bootstrap
  intervals are not independent training-seed uncertainty or scope-to-scope
  significance tests.
- SwiGLU-5 mislabeled 2M evaluations as also requested at 5M. Both C2
  confirmations are valid recorded runs, but at the **actual** 5M endpoint C3
  is marginally better at 20%. C2 is not an unqualified 20% search optimum.
- Historical prefix PPL, full-corpus PPL at different contexts, and KL on
  different partitions are distinct metrics. Do not merge them into one curve.
  SwiGLU-6 versus SwiGLU-7 is not a controlled sequence-length ablation:
  allocations, fresh starts, and update geometry also changed.
- Summed singleton KL is a selection surrogate, not a verified composition
  law. Recovery does not remove the remaining structural quality gap.
- Preserved JSON/protocol/hash records support analysis of the original runs,
  not independent reproduction. Local SwiGLU-7 is a reduced export without
  weights or per-example records; complete bundles remain in PERUN PROJECT.
- Workflow timestamps are runtime proxies, not scheduler-billed GPU-hours.
  Resource reporting follows the
  [PERUN accounting contract](../../../docs/infrastructure/perun/perun-resources.md).

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
- [[implementation-maintained-mlp-replacement-package]] routes to the current
  scientific code, runners, and dedicated manual-scratch job.

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
  and the linked SwiGLU-7 preparation/production records. Method implementations
  are in [modules.py](../../../src/mlp_replacement/operators/modules.py),
  [training.py](../../../src/mlp_replacement/operators/training.py), and
  [swiglu_7.py](../../../workflows/runs/model/swiglu/swiglu_7.py).
