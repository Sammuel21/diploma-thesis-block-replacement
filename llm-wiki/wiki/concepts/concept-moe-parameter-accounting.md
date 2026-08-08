---
id: concept-moe-parameter-accounting
title: MoE Parameter Accounting
summary: Separates total stored parameters, model memory, and routing-dependent active parameters in mixture-of-experts models.
type: concept
status: review
created: 2026-07-17
updated: 2026-08-08

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
    - mixture-of-experts
    - total-parameters
    - active-parameters
    - model-memory
    - evaluation
  granularities:
    - moe
    - model
    - cross-level
  pipeline_stages:
    - evaluation
    - analysis

sources:
  - source_id: src-mone-2026
    locator: "Sections 1, 3.1, and 4.1; Section 5.1; Appendix F"
    relation: contextualizes

related:
  - "[[source-summary-mone-2026]]"
  - "[[method-mone-novice-expert-replacement]]"
  - "[[method-global-to-local-operator-budget-allocation]]"
  - "[[concept-model-compression-evaluation-axes]]"
  - "[[decision-primary-compression-evaluation-scope]]"
supersedes: []
superseded_by: []
---

# MoE Parameter Accounting

## Overview

Sparse MoE models store many expert parameters but route each token through
only a subset. Compression evaluation must therefore distinguish the model's
total parameter inventory from the parameters involved in one token's forward
path. [src-mone-2026, Sections 1 and 3.1]

## Definition or Description

- **Total parameters:** all model parameters, including every expert, router,
  attention module, embeddings, and always-active components.
- **Model size or parameter storage:** bytes required to store those parameters
  at a stated numerical precision. Runtime memory measurements may additionally
  include buffers, framework overhead, caches, and temporary tensors.
- **Active parameters per token:** parameters in always-active components plus
  the experts selected for that token. This is routing- and
  architecture-dependent.
- **Novice hit ratio:** the fraction of routed expert selections served by
  constant novices rather than retained expert MLPs. In MoNE this determines
  the realized reduction in active expert computation.

The literature often encodes total and nominal active parameters in names such
as OLMoE 7B-A1B or Qwen3-30B-A3B. Exact accounting still requires the model's
implementation because shared experts, router behavior, and top-k policies
vary. [src-mone-2026, Sections 3.1 and 5.1]

## Evidence and Rationale

MoNE targets the gap between total and active parameters. Although inactive
experts are not evaluated for a token, their weights normally remain resident
in memory. Replacing selected experts with one constant vector each reduces
total stored parameters and measured memory. Active-parameter reduction occurs
only when routing selects a novice. [src-mone-2026, Section 4.1; Appendix F]

The Qwen3-30B-A3B measurements show memory decreasing consistently with nominal
pruning ratio. Runtime speedup, however, depends on batch size and novice hit
ratio rather than pruning ratio alone. [src-mone-2026, Appendix F; Table 16]

## Limitations and Open Issues

Parameter count does not uniquely determine serialized checkpoint size unless
dtype, quantization, tied weights, and metadata are specified. Measured GPU
memory is also not identical to model size on disk.

Active parameters are not a complete latency metric. Operator shape, memory
movement, batching, routing, kernels, and hardware affect runtime. The thesis
currently reports active parameters and memory without attempting a general
latency model.

For reproducibility, each MoE experiment should state the counting convention,
precision, top-k routing, shared-expert treatment, and whether active counts are
nominal or measured from routing traces.

## Relationships

- [[source-summary-mone-2026]] provides the source evidence and model examples.
- [[method-mone-novice-expert-replacement]] changes both stored and
  routing-dependent active parameters.
- [[method-global-to-local-operator-budget-allocation]] is defined first for
  dense per-layer MLPs; an MoE extension must declare whether its global and
  local budgets refer to stored parameters, active parameters, or both.
- [[concept-model-compression-evaluation-axes]] places MoE-specific accounting
  inside the broader model-compression evaluation taxonomy.
- [[decision-primary-compression-evaluation-scope]] makes explicit total and
  active MoE parameter reporting part of the primary measurement contract.

## Sources

- `src-mone-2026` - Sections 1, 3.1, 4.1, and 5.1; Appendix F
