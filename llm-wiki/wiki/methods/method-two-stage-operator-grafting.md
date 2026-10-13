---
id: method-two-stage-operator-grafting
title: Two-Stage Operator Grafting
summary: Initializes replacement operators through local activation regression and repairs their composition through model-level fine-tuning.
type: method
status: review
created: 2026-07-17
updated: 2026-10-12

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
    - experiment-backed

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
  - "[[comparison-recovery-objective-trainable-scope]]"
  - "[[synthesis-homogeneous-swiglu-findings]]"
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

## Project Evidence: Is Stage 1 Still Worth It After Stage 2?

The paragraphs above report prior work. This section records project
evidence, labelled separately.

**Empirical finding.** SwiGLU-8 ablated Stage 1 for causal-LLM MLP
replacement on pinned SmolLM2-1.7B. Both arms start from the same
importance-ranked teacher slice at the same widths:

- the fitted arm adds local MSE fitting (Stage 1);
- the slice-only arm skips it.

Both then recover only the replacement MLPs with teacher-logit KL for 100M
tokens. Before recovery, skipping Stage 1 leaves the model 2-5 times worse in
KL. After 100M tokens, the slice-only arm's 8K KL is still 10.4-23.5% higher
at every target from 20% to 80% eligible-MLP removal. That residual gap was
still shrinking between 50M and 100M, so whether it persists is open. See
[[synthesis-homogeneous-swiglu-findings]] and the
[SwiGLU-8 report](../../../docs/experiments/model/swiglu/swiglu-8.md).

**Empirical finding.** SwiGLU-8's block-level study found SwiGLU-7's Stage 1
recipe on a plateau, with no tested change better by 5%. Fitting on 8K-context
activations instead of 128-token ones traded about 4% better 8K KL for about
6% worse 128-token KL. Solving the down projection exactly by least squares
was worse than gradient fitting with early stopping. The source's finding that
L2 suits MLP regression is consistent with the project's MSE objective, but
the two studies are not directly comparable. [src-grafting-2025, Section 3.3;
Table 1]

**Synthesis.** The source's Table 2 compares random initialization, local
regression alone, and local regression plus fine-tuning, but not fine-tuning
without local regression. Its no-local-fit baseline is also random rather than
a teacher slice. [src-grafting-2025, Section 3.3; Table 2] SwiGLU-8 measures
that missing cell for one LLM, one seed, and a 100M-token budget. It does not
show that Stage 1 is necessary at longer budgets.

## Limitations and Open Issues

The local objective only matches activations observed on the calibration
distribution. It does not guarantee equivalent behavior on unseen states or
after upstream replacements alter the input distribution.

The global fine-tuning stage can repair composition error, but it also makes
the final result depend on recovery data, optimizer, trainable parameter set,
and compute budget. These must be logged as experimental variables.

The source's evidence is from diffusion transformers. Applying the method to
causal LLM MLPs is a thesis hypothesis, not a verified consequence of this
source. The project evidence above supports Stage 1's value for one LLM at
100M recovery tokens only.

**Open question.** Does Stage 1's advantage survive to 1B recovery tokens, and
does it depend on sparsity? A follow-up is proposed but not decided.

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
- [[comparison-recovery-objective-trainable-scope]] separates Grafting's main
  end-to-end recovery scope, its PixArt LoRA exception, and the project's
  replacement-only updates.
- [[synthesis-homogeneous-swiglu-findings]] records the SwiGLU-8 ablation of
  Stage 1 and its limits.

## Sources

- `src-grafting-2025` - Sections 3.1-3.3 and 4.1, Tables 1-2
- Project evidence, checked 2026-10-12 - the SwiGLU-8 records summarized in
  the [SwiGLU-8 report](../../../docs/experiments/model/swiglu/swiglu-8.md),
  with the local copy in
  [data/results/workflows/model/swiglu-8](../../../data/results/workflows/model/swiglu-8/).
