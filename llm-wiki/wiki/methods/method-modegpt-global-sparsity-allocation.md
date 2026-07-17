---
id: method-modegpt-global-sparsity-allocation
title: MoDeGPT Global Sparsity Allocation
summary: Converts layer importance scores into a smoothed nonuniform sparsity distribution under a global compression constraint.
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
    - global-sparsity-allocation
    - block-importance
    - compression-budget
    - nonuniform-compression
  granularities:
    - transformer-layer
    - model
    - cross-level
  pipeline_stages:
    - screening
    - selection
    - integration
    - evaluation
    - analysis

sources:
  - source_id: src-modegpt-2025
    locator: "Section 3.3; Section 4.5; Figure 6; Table 9; Appendices B.10-B.11"
    relation: defines

related:
  - "[[source-summary-modegpt-2025]]"
  - "[[method-modegpt-modular-decomposition]]"
  - "[[method-block-importance]]"
supersedes: []
superseded_by: []
---

# MoDeGPT Global Sparsity Allocation

## Overview

MoDeGPT distributes a fixed global compression target nonuniformly across
Transformer layers. High-importance layers retain more parameters, while
lower-importance layers receive higher sparsity. Entropic regularization
smooths the allocation to limit extreme concentration.
[src-modegpt-2025, Section 3.3]

## Definition or Description

Let `s_i` be the importance of layer `i`, `phi_i` its assigned sparsity, and
`phi_avg` the desired average sparsity. The source maximizes retained weighted
importance plus an entropy term under the global constraint. For the stated
regularization regime, it obtains:

```text
phi = L * phi_avg * softmax(-s / epsilon)
```

The negative sign means a larger importance score maps to lower sparsity. The
paper instantiates `s` with Block Importance, computed from the cosine distance
between each layer's residual input and output. All layer scores require one
calibration forward pass. [src-modegpt-2025, Equations 10-11; Section 3.3]

The allocation selects *how much* to compress each layer. MoDeGPT's modular
decomposition separately determines *how* to realize that target inside MLP,
query-key, and value-output modules.

## Evidence and Rationale

At 30% average compression on LLaMA-2 7B, uniform allocation reports
perplexity 9.06 and average zero-shot accuracy 53.47%. The BI-based allocation
reports 7.51 and 60.78%, respectively, versus the dense baseline at 5.12 and
69.00%. [src-modegpt-2025, Section 4.5; Table 9]

The visualization assigns some layers substantially more sparsity than others;
the source highlights layer 26 at up to 82% in that experiment. This supports
the claim that a global target need not imply equal damage or equal removable
capacity at every depth. [src-modegpt-2025, Section 4.5; Figure 6]

## Limitations and Open Issues

The allocation is only as reliable as the importance score. BI measures
residual directional change, not direct post-compression loss, modular
reconstruction error, or downstream causal importance.

The source reports one principal allocation ablation for LLaMA-2 7B at 30%
compression. The optimal regularization strength and safe layer-level bounds
can vary with model, method, and global compression ratio.

A continuous sparsity distribution and a discrete top-k replacement strategy
are not directly equivalent. A fair comparison must match total parameters
removed and account for different local replacement capacities.

The objective weights layer importance by retained parameters. It does not
directly optimize the final perplexity or task score, so the allocation remains
a proxy even when it outperforms uniform sparsity.

## Relationships

- [[source-summary-modegpt-2025]] provides the full source context.
- [[method-modegpt-modular-decomposition]] realizes each assigned sparsity
  inside the layer's functional modules.
- [[method-block-importance]] defines the layer score used by the source; the
  allocation uses BI continuously rather than only selecting a top-k set.

## Sources

- `src-modegpt-2025` - Section 3.3, Section 4.5, Figure 6, Table 9, and
  Appendices B.10-B.11

