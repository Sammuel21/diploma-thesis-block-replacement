---
id: method-mone-novice-expert-replacement
title: MoNE Novice Expert Replacement
summary: Replaces a low-variance MoE expert with its constant calibration mean to reduce memory and routed computation.
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
  role: mixed
  confidence: high
  verification:
    - source-checked

scope:
  topics:
    - novice-replacement
    - expert-pruning
    - constant-operator
    - model-memory
  granularities:
    - mlp-block
    - moe
    - model
    - cross-level
  pipeline_stages:
    - selection
    - replacement
    - integration
    - recovery
    - evaluation

sources:
  - source_id: src-mone-2026
    locator: "Sections 4.1 and 4.3; Sections 5.2 and 5.5; Appendix F"
    relation: defines

related:
  - "[[source-summary-mone-2026]]"
  - "[[method-frequency-variance-expert-redundancy]]"
  - "[[concept-moe-parameter-accounting]]"
  - "[[method-two-stage-operator-grafting]]"
supersedes: []
superseded_by: []
---

# MoNE Novice Expert Replacement

## Overview

MoNE replaces a selected MoE expert with a constant vector equal to the
empirical mean of that expert's outputs on routed calibration tokens. The paper
calls this lightweight structure a novice. [src-mone-2026, Sections 4.1 and
4.3]

## Definition or Description

When the router selects a pruned expert, the MoE weighted sum uses the novice
vector instead of evaluating the expert MLP. Under squared output discrepancy,
the mean expert output is the closed-form optimal constant replacement. No
gradient-based local training is required. [src-mone-2026, Equation 8;
Section 4.3]

The novice retains one hidden-size vector rather than the expert's MLP weight
matrices. It therefore reduces stored parameters. Tokens routed to novices also
activate fewer input-dependent parameters, while tokens routed only to retained
experts preserve the original active computation. [src-mone-2026, Section 4.1;
Appendix F]

## Evidence and Rationale

Without weight updates, MoNE reports higher average zero-shot accuracy than the
compared expert-pruning baselines at 25% pruning across five MoE models. The
ablation attributes part of the advantage to retaining novice outputs instead
of deleting selected experts outright. [src-mone-2026, Sections 5.2 and 5.4]

Memory drops predictably with the number of expert matrices replaced. Runtime
active computation depends instead on the proportion of routed expert calls
that hit novices. [src-mone-2026, Appendix F; Table 16]

Continued pretraining is optional rather than part of the closed-form
replacement. The paper evaluates 2B recovery tokens on 25%-pruned OLMoE and
improves average accuracy, but does not fully recover every task.
[src-mone-2026, Section 5.5; Table 3]

## Limitations and Open Issues

A constant novice has no input-dependent capacity. It is suitable only for an
expert whose routed outputs remain concentrated around the calibration mean.
The method can fail under distribution shift or for specialized tasks not
represented during calibration.

The replacement is specific to sparse MoE routing. It cannot directly replace
a dense MLP block that is evaluated for every token unless that block is
exceptionally constant.

**Synthesis.** Constant, linear, and small-MLP substitutes could form an ordered
capacity ladder. Output variance may help decide the minimum capacity worth
evaluating, but this is not tested by MoNE.

## Relationships

- [[source-summary-mone-2026]] provides the complete source context.
- [[method-frequency-variance-expert-redundancy]] selects experts compatible
  with a constant novice.
- [[concept-moe-parameter-accounting]] explains the different effects on total
  and active parameters.
- [[method-two-stage-operator-grafting]] learns input-dependent replacements
  through regression; MoNE instead uses a closed-form constant and optional
  later continued pretraining.

## Sources

- `src-mone-2026` - Sections 4.1 and 4.3, Sections 5.2 and 5.5, Appendix F
