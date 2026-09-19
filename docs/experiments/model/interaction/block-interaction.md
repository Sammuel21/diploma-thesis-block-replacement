---
metadata_version: 1
title: Block Interaction Analysis
type: experiment-workflow
category: experiments/model/interaction
status: active
created: 2026-09-18
modified: 2026-09-19
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
sources:
  notebooks:
    - notebooks/model/interaction/block-interaction.ipynb
  artifacts:
    - data/results/notebook-model-study/block-interaction.json
---

# Block Interaction Analysis

Notebook: [block-interaction.ipynb](../../../../notebooks/model/interaction/block-interaction.ipynb)

Status: executed as a notebook. Its schema-2 JSON artifact is present. This
study has no headless Python runner.

## Purpose

This experiment measures whether model damage from several locally fitted MLP
replacements is additive. It protects the first and last Transformer blocks
and studies layers 1 through 22 with two replacement classes: a ridge-fitted
linear map and a randomly initialized SwiGLU retaining 50% intermediate width.

The comparison is diagnostic rather than budget matched. The linear and
SwiGLU replacements have different parameter counts, and no global recovery is
performed. Results therefore describe interaction behavior under these two
conditions; they do not establish which operator is better at equal size.

## Pipeline at a glance

```text
SmolLM2-1.7B dense teacher
|
+-- protect layers 0 and 23
`-- capture 12,288 training + 6,144 validation activation pairs per layer
             |
             v
      fit two operators independently at every eligible layer
      +-- linear      : closed-form ridge fit, lambda=1e-4
      `-- swiglu_050  : 50%-width local training, up to 64 epochs
             |
             +---------------------------------------------------+
             |                                                   |
             v                                                   v
  SLIDING-WINDOW STUDY                               PAIRWISE STUDY
  consecutive windows of width 1..4                  profile all singletons
  for each homogeneous operator                      |
  |                                                   +-- homogeneous pairs
  `-- compare joint KL with sum of                    |   linear+linear and
      singleton KL values                             |   swiglu+swiglu
                                                      |
                                                      `-- heterogeneous pairs
                                                          linear at layer i +
                                                          swiglu at layer j
             |                                                   |
             +-------------------------+-------------------------+
                                       v
                         loss / perplexity / teacher KL
                         additive KL residuals and summaries
                                       |
                                       v
                         block-interaction.json + notebook charts
```

## Measurements

For a replaced set `S`, the notebook defines the interaction residual as:

```text
interaction_KL(S) = measured_joint_KL(S)
                    - sum(singleton_KL(layer) for layer in S)
```

This is a project-defined diagnostic. Zero means the measured teacher KL is
additive under the singleton baseline; positive values mean the joint damage
is worse than that baseline; negative values mean it is smaller. The pairwise
tables also retain absolute interaction KL, layer distance, normalized
distance, parameter counts, loss, perplexity, and perplexity delta.

The executed grid contains:

- 44 local fits: 22 eligible layers times two operator classes;
- 164 sliding windows: two operators times all consecutive windows of widths
  one through four;
- 44 singleton model profiles;
- 462 homogeneous pairs: every unordered layer pair for each operator;
- 462 ordered heterogeneous pairs with a linear replacement at one layer and
  a SwiGLU replacement at a different layer.

## Artifact and interpretation boundary

The schema-2 artifact stores the dense reference, local fitting metrics and
SwiGLU histories, every sliding-window row, singleton profiles, homogeneous
and heterogeneous pair rows, aggregate summaries, extremes, and distance
summaries. The notebook can switch from execution mode to artifact-loading
mode to reconstruct its tables and plots without refitting.

The notebook contains a higher-order section heading, but no higher-order
experiment is implemented or stored. The available evidence covers consecutive
windows up to four blocks and exhaustive pairs only. Because operators are fit
with the historical 12,288-pair calibration budget and the operator classes
are not parameter matched, this artifact is best used to study propagation and
non-additivity, not final operator selection or production compression quality.
