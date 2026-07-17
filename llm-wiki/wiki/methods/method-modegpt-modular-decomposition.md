---
id: method-modegpt-modular-decomposition
title: MoDeGPT Modular Decomposition
summary: Jointly reduces paired matrix dimensions inside MLP and attention modules using calibration-aware closed-form decompositions.
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
    - modular-decomposition
    - structured-compression
    - matrix-decomposition
    - local-reconstruction
    - mlp-width-reduction
  granularities:
    - mlp-block
    - transformer-layer
    - model
    - cross-level
  pipeline_stages:
    - data
    - selection
    - replacement
    - integration
    - recovery
    - evaluation
    - analysis

sources:
  - source_id: src-modegpt-2025
    locator: "Sections 3.1-3.2; Algorithms 1-3; Sections 4.2-4.5; Appendices A and B.6"
    relation: defines

related:
  - "[[source-summary-modegpt-2025]]"
  - "[[method-modegpt-global-sparsity-allocation]]"
  - "[[method-two-stage-operator-grafting]]"
supersedes: []
superseded_by: []
---

# MoDeGPT Modular Decomposition

## Overview

MoDeGPT compresses pairs of matrices that jointly implement one Transformer
function. It minimizes calibration-set output reconstruction error while
reducing the matrices' shared intermediate dimension. The primary procedure is
forward-only and does not require gradient-based recovery.
[src-modegpt-2025, Sections 3.1-3.2]

## Definition or Description

The method partitions a Transformer layer into three module types and assigns a
different decomposition according to the module's nonlinear structure:

| Type | Matrix pair | Procedure |
| --- | --- | --- |
| I | MLP up/gate and down | Nyström approximation of activation correlation |
| II | query and key | shared column selection through CR decomposition |
| III | value and output | input-weighted SVD |

For a gated MLP, the up and gate matrices are treated as a combined projection.
Given target sparsity, the Type-I algorithm computes the correlation of
post-gating intermediate activations, ranks intermediate channels by
deterministic ridge leverage scores, retains the top channels, and recomputes
the down projection in closed form. This reduces the MLP intermediate width
without adding residual adapters. [src-modegpt-2025, Section 2.2; Section 3.2;
Algorithm 1]

Type II uses a shared reduced query-key dimension for each head. Type III
directly solves the linear value-output reconstruction with SVD. The paper
derives reconstruction-error bounds for all three module types.
[src-modegpt-2025, Section 3.2; Algorithms 2-3; Appendix A]

## Evidence and Rationale

Across the evaluated OPT and LLaMA families, MoDeGPT reports substantially
better perplexity than uniform pruning, independent SVD, ShortGPT, and SLEB at
matched nominal compression ratios. At 30% compression, LLaMA-2 7B reports
perplexity 7.51 versus the dense baseline at 5.12; LLaMA-2 13B reports 6.10
versus 4.57. [src-modegpt-2025, Section 4.2; Table 3]

The module ablation reports that the MLP holds 66.84% of the considered module
parameters and produces most absolute perplexity degradation. Normalized by
parameter share, query-key compression is most sensitive. MLP compression also
has the largest temporary memory overhead because its activation-correlation
matrix uses the large intermediate dimension. [src-modegpt-2025, Section 4.5;
Figure 4; Tables 7-8]

Recovery fine-tuning is optional. In the paper's LLaMA-2 7B appendix,
MLP-only LoRA recovery is slightly better on average than tuning all linear
matrices, but improvements remain small and task-dependent despite using 8,000
recovery samples. [src-modegpt-2025, Appendix B.6; Table 20]

## Limitations and Open Issues

This method reduces channels inside an existing operator family. It does not
search arbitrary replacement architectures or remove the nonlinear MLP as a
whole. It therefore answers a different question from replacing an MLP block
with a linear or small-MLP substitute.

The local objective measures module outputs on calibration samples. Its error
bounds do not directly bound full-model language loss or task accuracy after
many modules are changed.

Calibration statistics and numerical linear algebra can be expensive. The
source uses high-precision correlation calculations and reports substantial
temporary memory overhead for MLP compression.

Architecture support is not automatic. The matrix grouping and reconstruction
must match gated MLP, multi-head attention, or the paper's documented
adaptations for variants such as grouped-query attention.

## Relationships

- [[source-summary-modegpt-2025]] provides the full evidence and transfer
  boundaries.
- [[method-modegpt-global-sparsity-allocation]] assigns different target
  sparsities to layers before local decomposition.
- [[method-two-stage-operator-grafting]] learns arbitrary replacement
  operators through local regression and integrated recovery; MoDeGPT instead
  preserves the operator family and solves a constrained decomposition.

## Sources

- `src-modegpt-2025` - Sections 3.1-3.2, Algorithms 1-3, Sections 4.2-4.5,
  Appendices A and B.6

