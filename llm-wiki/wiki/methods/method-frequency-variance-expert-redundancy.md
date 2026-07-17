---
id: method-frequency-variance-expert-redundancy
title: Frequency-Variance Expert Redundancy
summary: Ranks MoE experts by combining router usage with output variance over calibration data.
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
    - expert-redundancy
    - routing-frequency
    - output-variance
    - calibration-robustness
  granularities:
    - moe
    - model
  pipeline_stages:
    - data
    - screening
    - selection
    - analysis

sources:
  - source_id: src-mone-2026
    locator: "Section 4.2; Sections 5.3-5.4; Appendices A, D, and E"
    relation: defines

related:
  - "[[source-summary-mone-2026]]"
  - "[[method-mone-novice-expert-replacement]]"
  - "[[method-minitron-activation-based-importance]]"
supersedes: []
superseded_by: []
---

# Frequency-Variance Expert Redundancy

## Overview

MoNE ranks experts by combining how strongly or frequently the router selects
them with how much their outputs vary on selected calibration tokens. Low use
suggests limited contribution; low output variance suggests that a constant
replacement can approximate the expert. [src-mone-2026, Section 4.2]

## Definition or Description

For each expert in each MoE layer, MoNE records:

- a variance term derived from the L2 norm of the unbiased estimate of expert
  output variance on tokens routed to that expert; and
- a routing term based on average routing scores for calibration tokens where
  the expert is among the selected top-k experts.

The normalized terms are fused into one score. In the paper's convention, a
lower fused score indicates greater redundancy and makes the expert a stronger
candidate for novice replacement. [src-mone-2026, Equations 4-7]

## Evidence and Rationale

The source visualizes experts that are high on both signals and experts that
would be missed by using only one. Its ablation reports that the fused score
combined with novice replacement performs best on average across models,
calibration datasets, and sample sizes. [src-mone-2026, Figures 1 and 4;
Appendices D-E]

Robustness is evaluated using C4 and Zyda2 with 100, 500, and 1,000 samples.
At 25% pruning, MoNE reports the strongest average accuracy/variance trade-off.
At 50%, all methods become less stable. [src-mone-2026, Section 5.3; Figure 3]

## Limitations and Open Issues

The score is replacement-aware: low variance matters because the selected
replacement is a constant mean vector. A different substitute family may need
a different complexity signal. High-variance experts may still be accurately
approximated by a learned linear or nonlinear novice.

Router behavior and output statistics depend on calibration data. Generic
calibration preserves average general-task accuracy more reliably than
specialized Math or GSM8K capability in the paper's appendix.

Frequency and variance should remain separately logged even when fused. The
MMLU ablation shows that the aggregate score is not uniformly optimal for every
task. [src-mone-2026, Section 5.4]

## Relationships

- [[source-summary-mone-2026]] provides the complete source context.
- [[method-mone-novice-expert-replacement]] determines why low output variance
  is useful for the chosen constant replacement.
- [[method-minitron-activation-based-importance]] is another forward-only
  calibration method, but ranks dense-model width components rather than
  routed experts.

## Sources

- `src-mone-2026` - Section 4.2, Sections 5.3-5.4, Appendices A, D, and E
