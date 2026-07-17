---
id: method-two-stage-operator-grafting
title: Two-Stage Operator Grafting
summary: Initializes replacement operators through local activation regression and repairs their composition through model-level fine-tuning.
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
    - operator-grafting
    - activation-distillation
    - replacement-error-propagation
    - self-grafting
  granularities:
    - mlp-block
    - transformer-layer
    - model
    - cross-level
  pipeline_stages:
    - data
    - replacement
    - integration
    - recovery
    - evaluation

sources:
  - source_id: src-grafting-2025
    locator: "Sections 3.1-3.3 and 4.1; Tables 1-2"
    relation: defines

related:
  - "[[source-summary-grafting-2025]]"
  - "[[concept-replacement-error-propagation]]"
  - "[[method-post-pruning-knowledge-distillation]]"
  - "[[method-mone-novice-expert-replacement]]"
  - "[[method-modegpt-modular-decomposition]]"
supersedes: []
superseded_by: []
---

# Two-Stage Operator Grafting

## Overview

Two-stage grafting edits a pretrained model by first learning each replacement
operator locally and then adapting the integrated model globally. The source
evaluates the method on diffusion transformers, including MLP and attention
replacement. [src-grafting-2025, Section 3.1]

## Definition or Description

### Stage 1: activation distillation

For an original operator `f` and replacement `g`, collect inputs reaching `f`
in the pretrained model and train `g` to minimize a regression loss between
`g(x)` and `f(x)`. This initializes the replacement as a local approximation of
the original computation before insertion. Grafting uses as few as 8,000
samples in its main experiments and trains operator positions independently.
[src-grafting-2025, Sections 3.1 and 4.1]

### Stage 2: integrated recovery

Insert all initialized replacements and fine-tune the edited model end to end
on the original task objective. This stage targets cumulative deviations that
local operator losses do not observe. The primary DiT-XL/2 experiments use 10%
of ImageNet-1K and 50,000 steps. The PixArt-Sigma experiment instead uses
rank-64 LoRA because of memory pressure from long sequences.
[src-grafting-2025, Sections 3.1, 4.1, and 5]

### Self-grafting control

Replace an operator with a randomly initialized operator of the same type and
size, then apply the same two stages. This preserves the architecture and
parameter count, isolating the effectiveness of initialization and recovery
from the capacity change introduced by a smaller substitute.
[src-grafting-2025, Sections 3.2-3.3]

## Evidence and Rationale

Grafting reports catastrophic failure when all operators are inserted with
random initialization. Local regression alone improves results, and integrated
fine-tuning with more data progressively recovers quality. The best local loss
differs by operator: L1 for the studied MHA replacements and L2 for MLPs.
[src-grafting-2025, Tables 1-2]

The paper tests replacement operator, location strategy, and replacement ratio
as independent axes. Partial interleaved MHA replacement is robust, while full
MHA replacement fails for the tested local alternatives. Variable-width MLP
replacement remains effective at full replacement. [src-grafting-2025,
Sections 4.1-4.2; Tables 3-4]

## Limitations and Open Issues

The local objective only matches activations observed on the calibration
distribution. It does not guarantee equivalent behavior on unseen states or
after upstream replacements alter the input distribution.

The global fine-tuning stage can repair composition error, but it also makes
the final result depend on recovery data, optimizer, trainable parameter set,
and compute budget. These must be logged as experimental variables.

The evidence is from diffusion transformers. Applying the method to causal LLM
MLPs is a thesis hypothesis, not a verified consequence of this source.

## Relationships

- [[source-summary-grafting-2025]] provides the complete source context.
- [[concept-replacement-error-propagation]] explains the model-level problem
  addressed by Stage 2.
- [[method-post-pruning-knowledge-distillation]] is a related recovery method,
  but it begins from structured pruning and emphasizes teacher logits rather
  than local operator regression.
- [[method-mone-novice-expert-replacement]] uses a closed-form constant for
  selected low-variance experts instead of training an input-dependent local
  operator.
- [[method-modegpt-modular-decomposition]] uses calibration-aware closed-form
  dimension reduction within the existing operator family instead of learning
  an arbitrary substitute and relying on integrated recovery.

## Sources

- `src-grafting-2025` - Sections 3.1-3.3 and 4.1, Tables 1-2
