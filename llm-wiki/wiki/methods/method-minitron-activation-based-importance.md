---
id: method-minitron-activation-based-importance
title: Minitron Activation-Based Importance Estimation
summary: Uses forward-pass activations on calibration samples to rank width components without gradients.
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
    - activation-based-importance
    - structured-pruning
    - calibration-data
  granularities:
    - neuron
    - model
    - cross-level
  pipeline_stages:
    - data
    - screening
    - selection

sources:
  - source_id: src-minitron-2024
    locator: "Section 2.2; Section 4.2; Tables 13-14"
    relation: defines

related:
  - "[[source-summary-minitron-2024]]"
  - "[[method-block-importance]]"
  - "[[method-retraining-assisted-architecture-search]]"
  - "[[method-frequency-variance-expert-redundancy]]"
supersedes: []
superseded_by: []
---

# Minitron Activation-Based Importance Estimation

## Overview

Minitron proposes a forward-only strategy for estimating the importance of
attention heads, MLP neurons, and embedding channels. It avoids gradients and
uses activations collected over a calibration dataset. [src-minitron-2024,
Section 2.2]

## Definition or Description

The method observes axis-specific activations:

- attention-head scores use the norm of each head's attention output;
- MLP-neuron scores use the activation produced by the corresponding row of
  the first MLP weight matrix; and
- embedding-channel scores use the corresponding LayerNorm activation.

Scores are aggregated across batch and sequence dimensions and then summed
across layers to obtain network-wide rankings for each width axis. The paper
tests mean absolute value, L2 norm, and variance as aggregation functions. It
selects batch-L2 followed by sequence-mean for its main experiments.
[src-minitron-2024, Section 2.2; Table 13]

The reported calibration dataset consists of 1,024 samples randomly drawn from
the full training blend. Its role is importance estimation, not model recovery.
[src-minitron-2024, Section 4]

## Evidence and Rationale

The method computes scores for several axes in forward passes and avoids the
memory cost of gradient-based saliency. In the paper's 15B-to-8B width-pruning
ablation, aggregation choice causes substantial differences in zero-shot LM
loss. The selected aggregation continues to outperform a deliberately weak
alternative after both receive the same 1.8B-token recovery budget.
[src-minitron-2024, Section 4.2; Table 13; Figure 5]

When embedding width is pruned with one, two, or four rounds of importance
recomputation, pre-recovery losses differ but final validation loss converges
after equal recovery. This is evidence for one-shot importance estimation only
within that tested width-pruning setting. [src-minitron-2024, Table 14]

## Limitations and Open Issues

The MLP score ranks individual hidden neurons and is aggregated network-wide.
It is not a metric for the replaceability of a complete MLP block and does not
evaluate a learned substitute architecture.

The selected aggregation and 1,024-sample budget are empirical choices on the
Nemotron data and architecture. Their transfer to other models, calibration
datasets, and block-replacement objectives must be tested rather than assumed.

**Open question.** Should replacement screening rank blocks using direct
input-output approximation error, full-layer BI, neuron-level activation
statistics, or a combination of these signals?

## Relationships

- [[source-summary-minitron-2024]] is the source-level context.
- [[method-block-importance]] is the separate metric used for depth rather
  than width screening.
- [[method-retraining-assisted-architecture-search]] uses lightweight recovery
  to compare candidates produced after importance-based trimming.
- [[method-frequency-variance-expert-redundancy]] is another forward-only
  calibration method, specialized for routed MoE experts and their suitability
  for constant replacement.

## Sources

- `src-minitron-2024` - Section 2.2, Section 4.2, Tables 13-14
