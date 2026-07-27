---
id: method-block-importance
title: Block Importance
summary: Estimates Transformer-layer sensitivity from cosine distance between the layer input and output residual representations.
type: method
status: review
created: 2026-07-17
updated: 2026-07-27

authorship:
  created_by: collaborative
  contributors:
    - researcher
    - llm

epistemic:
  role: prior-work
  confidence: medium
  verification:
    - source-checked

scope:
  topics:
    - block-importance
    - depth-pruning
    - activation-based-importance
  granularities:
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
  - "[[experiment-initial-block-compression-study]]"
supersedes: []
superseded_by: []
---

# Block Importance

## Overview

Block Importance (BI) is an activation-based sensitivity metric used by
Minitron to rank complete Transformer layers for depth pruning. Minitron adopts
the metric from ShortGPT; the original source has not yet been registered or
ingested in this wiki. [src-minitron-2024, Sections 2.2 and 5]

## Definition or Description

For layer `i`, BI is one minus the expected cosine similarity between residual
representations entering and leaving that layer, evaluated across calibration
samples and token positions:

```text
BI_i = 1 - E[cos(X_i,t, X_i+1,t)]
```

A small score means the layer changes the residual representation relatively
little under this metric; a larger score means a larger directional change.
Minitron interprets this as layer sensitivity and removes layers with the
lowest scores. All layer scores can be collected in one forward pass.
[src-minitron-2024, Section 2.2]

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
setting. The thesis currently studies replacement of an MLP sublayer/operator.
Computing cosine distance around only the MLP sublayer is a related adaptation,
not the exact object evaluated by Minitron.

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
- [[experiment-initial-block-compression-study]] uses canonical BI as a layer
  selection baseline and keeps its MLP-local adaptation separately named.

## Sources

- `src-minitron-2024` - Sections 2.2 and 5; Section 4.2, Table 10
- `src-modegpt-2025` - Sections 3.3 and 4.5, Figure 6, Table 9
