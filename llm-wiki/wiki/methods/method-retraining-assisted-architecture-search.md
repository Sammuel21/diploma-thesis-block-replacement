---
id: method-retraining-assisted-architecture-search
title: Retraining-Assisted Architecture Search
summary: Enumerates architectures under a parameter budget and compares them after equal lightweight recovery to stabilize candidate rankings.
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
    - architecture-search
    - parameter-budget
    - compression-recovery
    - candidate-ranking
  granularities:
    - model
    - cross-level
  pipeline_stages:
    - selection
    - recovery
    - evaluation
    - analysis

sources:
  - source_id: src-minitron-2024
    locator: "Section 2.3; Section 4.3; Table 12; Figure 9; Table 19"
    relation: defines

related:
  - "[[source-summary-minitron-2024]]"
  - "[[method-minitron-activation-based-importance]]"
  - "[[method-post-pruning-knowledge-distillation]]"
supersedes: []
superseded_by: []
---

# Retraining-Assisted Architecture Search

## Overview

Minitron searches for compressed model architectures by enumerating feasible
configurations near a target parameter count, applying structured pruning, and
giving every candidate the same lightweight recovery treatment before final
selection. [src-minitron-2024, Sections 2.3 and 4.3]

## Definition or Description

The search varies layer count, attention-head count, MLP expansion factor, and
embedding dimension. Candidates must fall within five percent of the target 8B
or 4B parameter budget. Restricting dimensions to commonly used values yields
15 feasible 8B candidates and 18 feasible 4B candidates, small enough for
enumeration. Each candidate then receives approximately 1.8B retraining tokens.
[src-minitron-2024, Section 4.3; Table 12; Table 19]

The candidate selected after lightweight retraining can receive a larger final
retraining budget. The short retraining stage is therefore part of the search
objective, not merely a repair performed after selection.

## Evidence and Rationale

The paper reports that relative validation-loss rankings of the 8B candidates
change materially during the first 300 of 400 recovery steps and then become
more stable. This supports evaluating candidate architectures after a shared
recovery budget rather than relying only on immediate post-pruning loss.
[src-minitron-2024, Section 4.3; Figure 9]

The search remains tractable because it constrains both the parameter interval
and the allowed dimensions. The paper notes that more advanced search methods
could be used for larger spaces but does not evaluate them.
[src-minitron-2024, Section 2.3]

## Limitations and Open Issues

Minitron searches global model dimensions, not subsets of MLP blocks and not
the internal architecture of learned substitute operators. A thesis search
over replacement locations, operator types, and expansion ratios has a
different combinatorial structure.

Candidate comparison is meaningful only when recovery data, tokens, optimizer,
and evaluation are controlled. The paper's 1.8B-token search budget is too
large to copy directly into the MVP-scale environment without an explicit
compute analysis.

**Open question.** How small can a shared recovery budget be while still
producing a candidate ranking that predicts longer-recovery performance for
block replacement?

## Relationships

- [[source-summary-minitron-2024]] provides the complete source context.
- [[method-minitron-activation-based-importance]] supplies rankings used before
  structured trimming.
- [[method-post-pruning-knowledge-distillation]] supplies the recovery process
  used to compare feasible candidates.

## Sources

- `src-minitron-2024` - Section 2.3, Section 4.3, Table 12, Figure 9, Table 19
