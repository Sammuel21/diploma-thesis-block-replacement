---
id: method-block-importance
title: Block Importance and MLP Screening Adaptations
summary: Documents canonical Transformer-layer BI and distinguishes raw MLP input-output cosine distance from a proposed residual-aware MLP adaptation.
type: method
status: review
created: 2026-07-17
updated: 2026-08-20

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

scope:
  topics:
    - block-importance
    - importance-screening
    - depth-pruning
    - activation-based-importance
  granularities:
    - mlp-block
    - transformer-layer
    - model
  pipeline_stages:
    - screening
    - selection

sources:
  - source_id: src-minitron-2024
    locator: "Section 2.2; Section 4.2; Table 10"
    relation: evaluates
  - source_id: src-modegpt-2025
    locator: "Section 3.3; Section 4.5; Figure 6; Table 9"
    relation: implements

related:
  - "[[source-summary-minitron-2024]]"
  - "[[source-summary-modegpt-2025]]"
  - "[[method-minitron-activation-based-importance]]"
  - "[[method-modegpt-global-sparsity-allocation]]"
  - "[[method-global-to-local-operator-budget-allocation]]"
  - "[[experiment-initial-block-compression-study]]"
  - "[[experiment-baseline-operator-analysis]]"
  - "[[experiment-swiglu-operator-design-progression]]"
supersedes: []
superseded_by: []
---

# Block Importance and MLP Screening Adaptations

## Overview

Block Importance (BI) is an activation-based sensitivity metric used by
Minitron to rank complete Transformer layers for depth pruning. Minitron adopts
the metric from ShortGPT; the original source has not yet been registered or
ingested in this wiki. [src-minitron-2024, Sections 2.2 and 5]

## Canonical Transformer-Layer BI

**Source-derived equation.** For layer `i`, BI is one minus the expected cosine
similarity between residual representations entering and leaving that complete
Transformer layer, evaluated across calibration samples and token positions:

```text
BI_i = 1 - E[cos(X_i,t, X_i+1,t)]
```

A small score means the layer changes the residual representation relatively
little under this metric; a larger score means a larger directional change.
Minitron interprets this as layer sensitivity and removes layers with the
lowest scores. All layer scores can be collected in one forward pass.
[src-minitron-2024, Section 2.2]

Canonical BI spans the complete decoder layer. It therefore combines changes
from attention, the attention residual connection, the MLP, and the MLP
residual connection. It is the prior-work baseline for importance screening in
this thesis, but it is not an MLP-specific score.

## Residual-Stream Boundaries

**Standard architectural notation.** The following decomposition is
explanatory notation for a pre-normalized residual Transformer, not a new
method claim:

```text
a_i = Attention(Norm_1(h_i))
u_i = h_i + a_i
m_i = MLP(Norm_2(u_i))
h_i+1 = u_i + m_i
```

Here, `h_i` is the complete layer input, `u_i` is the residual representation
after the attention update, `m_i` is the raw MLP update, and `h_i+1` is the
complete layer output. Ordinary use of this decomposition does not require a
method citation, although a thesis description of a specific model should cite
that model's architecture or implementation.

The residual boundary matters because `m_i` is an update added to the ongoing
representation, not the representation passed onward by itself. Comparing an
MLP input directly with `m_i` therefore does not measure the actual
before-versus-after change in the residual stream. Direction also does not
encode update magnitude: a small orthogonal update can change the residual
state little, while a large aligned update can change its magnitude
substantially.

## MLP-Specific Screening Candidates

### Raw MLP Input-Output Cosine Distance

**Project implementation description.** The current optional `mlp_sublayer`
scope compares the normalized MLP input with the raw MLP output:

```text
raw_mlp_score_i = 1 - E[cos(Norm_2(u_i), m_i)]
```

This equation describes the maintained implementation and requires no external
literature citation. The implementation is located in
[`src/mlp_replacement/screening.py`](../../../src/mlp_replacement/screening.py).

This score is not canonical BI. Its values can exceed one when the average
cosine similarity is negative, which means the raw update tends to point partly
against the normalized input. Such a value does not by itself establish that
the MLP is more important. To prevent overinterpretation, results should call
this quantity `raw MLP input-output cosine distance` or `adapted raw MLP score`,
not unqualified MLP BI.

### Residual-Aware MLP Influence

**Synthesis and project-proposed definition.** Applying the before-versus-after
logic of BI specifically around the MLP residual addition gives:

```text
MLP-BI-res_i = 1 - E[cos(u_i, u_i + m_i)]
```

This proposed formalization is not attributed to the registered papers and
does not require a prior-work citation as currently stated. It must remain
labelled as an adapted project metric unless an original source defining the
same quantity is registered and checked.

The residual-aware form isolates the directional change observed immediately
before and after the MLP update. It is better aligned with MLP influence than
the raw input-output score, but it remains a cosine heuristic: it ignores pure
norm changes and does not directly measure downstream loss, replacement
sensitivity, or recovery potential.

## Screening Interpretation

The thesis should keep three questions distinct:

- canonical BI asks how much the complete Transformer layer changes residual
  direction;
- a residual-aware MLP score asks how much the MLP addition changes residual
  direction; and
- held-out operator regression error asks how accurately a chosen replacement
  family approximates the original MLP mapping.

Correlation between these quantities measures agreement between proxies. It
does not establish that either proxy reliably predicts model-level importance.
That requires a model-level reference outcome, such as the validation-loss
change caused by a controlled singleton replacement.

### Current Project Use

**Working project interpretation.** Pre-recovery teacher-to-student KL after a
controlled singleton MLP replacement is now the primary empirical reference
for single-block replacement sensitivity. It directly measures model-wide
output change, but it is conditioned on the replacement family, parameter
budget, local fitting procedure, data, and KL protocol. It is therefore not an
intrinsic architecture-independent layer score.

Canonical BI remains the source-derived, replacement-operator-independent
baseline. Residual-aware MLP BI remains a project-proposed candidate with
closer granularity to the replaced component. Their useful role is now a
falsifiable screening question: do these inexpensive forward-only scores rank
layers similarly to pre-recovery KL for fixed replacement configurations? That
comparison has not yet been executed. The raw MLP input-output cosine distance
is retained only as an optional diagnostic.

An importance score can rank where compression may be safer or riskier; it
does not determine how many blocks should be replaced. Replacement count must
come from a declared footprint target, quality constraint, or separate search
policy.

## Evidence and Rationale

Minitron compares BI with a more expensive leave-one-layer-out perplexity
criterion. After removing 16 layers and retraining with 1.8B tokens, its Table
10 reports LM loss 2.177 for BI ranking and 2.155 for perplexity ranking at the
same 9.39B parameter count. In that specific experiment, BI is computationally
cheaper but does not outperform the perplexity criterion on final LM loss.
[src-minitron-2024, Section 4.2; Table 10]

MoDeGPT uses the same BI form for a different model-level decision. Instead of
removing the lowest-ranked layers, it maps scores to a continuous nonuniform
sparsity allocation. At 30% average compression of LLaMA-2 7B, its reported
allocation outperforms uniform sparsity in both perplexity and average
zero-shot accuracy. [src-modegpt-2025, Sections 3.3 and 4.5; Table 9]

The paper also notes that BI can be extended to contiguous layer groups, but
attributes that extension to other work. That claim has not been independently
checked in this wiki. [src-minitron-2024, Section 2.2]

## Limitations and Open Issues

Minitron's `block` is a complete Transformer layer in the depth-pruning
setting. The thesis studies replacement of an MLP sublayer/operator. Both the
raw MLP score and residual-aware MLP score are project-level adaptations, not
the exact object evaluated by Minitron.

BI measures directional representational change, not direct redundancy,
approximability, downstream causal importance, or expected post-recovery loss.
It can therefore be useful for ranking while remaining an incomplete predictor
of replacement success.

The page remains at `review` with medium confidence until the original ShortGPT
definition and experiments are ingested.

## Relationships

- [[source-summary-minitron-2024]] documents Minitron's use and evaluation of
  BI.
- [[source-summary-modegpt-2025]] documents BI as a global compression-budget
  allocation signal.
- [[method-minitron-activation-based-importance]] covers the different width
  scores used for heads, neurons, and embedding channels.
- [[method-modegpt-global-sparsity-allocation]] converts BI into per-layer
  sparsity rather than a discrete pruning or replacement order.
- [[method-global-to-local-operator-budget-allocation]] accepts BI or another
  direction-corrected importance score as a configurable input for assigning
  per-block replacement caps without using that score to select an operator.
- [[experiment-initial-block-compression-study]] uses canonical BI as a layer
  selection baseline and keeps MLP-local adaptations separately named.
- [[experiment-baseline-operator-analysis]] records the checked singleton KL
  reference for five baseline operators across eligible depth.
- [[experiment-swiglu-operator-design-progression]] extends that reference to
  ten generic operator configurations and retains its operator-conditioned
  interpretation.

## Sources

- `src-minitron-2024` - Sections 2.2 and 5; Section 4.2, Table 10
- `src-modegpt-2025` - Sections 3.3 and 4.5, Figure 6, Table 9
