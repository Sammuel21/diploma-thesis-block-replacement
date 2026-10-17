---
title: Stacking-1 - SwiGLU Replacement Plus Weight Quantization
status: draft
created: 2026-10-17
modified: 2026-10-17
---

# Stacking-1 - SwiGLU Replacement Plus Weight Quantization

**Status:** design settled on 2026-10-17 in a grilling session after the
[supervisor meeting of 2026-10-16](../../../meetings/2026-10-16-supervisor-feedback.md).
Not implemented and not run. Implementation starts after
[SwiGLU-9](../swiglu/swiglu-9.md) is submitted. The binding decisions are the
2026-10-16 and 2026-10-17 rows in the
[shared decision log](../../../agents/decisions.md); if this page and the log
disagree, the log wins until the conflict is raised.

## Questions

1. Does SwiGLU replacement followed by INT8, INT4, or INT3 weight quantization
   reach quality-per-byte points that neither method reaches alone?
2. Does the order matter: quantizing after recovery, or recovering while
   quantization is simulated?
3. Does local fitting still matter once the model is quantized?

## Why this experiment

In the [results snapshot](../../../results/2026-10-15-results.md),
quantization beats SwiGLU replacement at every size it reaches (up to 47.5%
whole-model byte saving), and SwiGLU reaches the frontier only beyond that.
The supervisor expected this (INT8 is close to free, INT4 often too) and
suggested stacking the two methods instead of only competing them.

## Win criterion

A model wins if it is on the frontier of a metric's plot: nothing else saves
at least as many bytes with better quality. There is one plot per metric (8K
KL, test perplexity, task macro accuracy), and each plot includes SwiGLU
alone, quantization alone, and the stacked models. Quantization alone,
including INT3, must be on the same plot; otherwise a stacked model could look
like a winner only because its strongest equal-size competitor is missing.

Approximate whole-model byte savings (offline arithmetic: eligible MLPs are
about 64.8% of the BF16 bytes; INT8 costs about 8 bits, INT4 4.25, INT3 3.25
bits per weight):

| Variant | Saving |
| --- | ---: |
| INT4 alone (measured) | 47.5% |
| SwiGLU 50% + INT8 | about 48.6% |
| SwiGLU 20% + INT4 | about 51.0% |
| INT3 alone | about 51.6% |
| SwiGLU 80% alone (measured) | 51.8% |
| SwiGLU 50% + INT4 | about 56% |
| SwiGLU 80% + INT4 | about 61% |

The three variants near 51% are the cleanest equal-size comparison.

## Pipeline

Each variant is an ordered stage list set in the configuration. Only stages
that are used are implemented; low-rank factorization is a possible future
stage (structural, numerical, and algebraic redundancy as three stage kinds).

| Variant | Stages |
| --- | --- |
| Order A: quantize after recovery | replace → fit → recover (BF16) → PTQ → evaluate |
| Order B: train while quantized | replace → fit → recover with simulated quantization (QAT) → convert to real integers → evaluate |
| Order C (deferred) | replace → fit → recover (BF16) → QAT → convert → evaluate |
| Quantization alone (0% row) | dense model → PTQ or QAT → evaluate |

**Quantized scope:** every MLP linear layer in layers 1-22, as in
Quantization-1, including layers the allocator kept dense. Attention,
embeddings, norms, and layers 0 and 23 stay BF16.

## Run grid

**Order A**, conversion and evaluation only:

| SwiGLU models | INT8 | INT4 | INT3 |
| --- | :---: | :---: | :---: |
| SwiGLU-8 100M fitted, 20-80% (8 targets) | ✓ | ✓ | ✓ |
| SwiGLU-8 100M slice-only, 20/50/80% | ✓ | ✓ | ✓ |
| 1B fitted and slice-only, 20/50/80% (SwiGLU-7 and SwiGLU-9) | ✓ | ✓ | ✓ |

That is 17 models × 3 = 51 evaluations. The slice-only rows answer question 3
at no training cost. The 1B rows run after SwiGLU-9 finishes.

**Order B**, QAT recovery from SwiGLU-8's token-0 starts, 100M tokens:

| Fitted models | INT4 | INT3 |
| --- | :---: | :---: |
| 20/50/80% | ✓ | ✓ |

**Quantization alone (0% row):** INT3 PTQ and INT3 QAT at 100M. INT8 and INT4
alone come from Quantization-1.

**Order comparison:** A and B both use 100M training tokens in total. Order C
runs only if A and B differ clearly, and is then compared with B at equal
total tokens (100M + 100M against 200M). Whether any stacking variant goes to
1B is decided after the 100M results. INT8 is PTQ only, as in Quantization-1.

## Quantization method

- **Same family as Quantization-1** (TorchAO 0.17.0): weight-only; INT8
  symmetric per output channel; INT4 and INT3 asymmetric with groups of 128
  input weights; round-to-nearest. Each group stores its own scale and zero
  point in 16 bits each, which adds 0.25 bits per weight.
- **INT3 bytes are calculated** from 3.25 bits per weight and labelled as
  calculated. TorchAO may keep 3-bit weights in 8-bit containers on GPU, in
  which case measured memory would hide the saving. TorchAO's INT3 support is
  checked before implementation; this was not verified when the design was
  settled.
- **INT3 PTQ uses the weakest standard method.** 3-bit results are usually
  reported with error-compensating methods such as GPTQ or AWQ. INT3 QAT is
  the fair INT3 point. GPTQ is added only if INT3 PTQ is much worse than INT3
  QAT; it would need its own environment decision and a registered source.
- **Every external method or library is registered and cited** before the
  thesis uses it.

## Width alignment

Groups of 128 need every quantized row length to be a multiple of 128. In a
SwiGLU block, the down projection's row length is the layer's width, and the
allocator gives one layer per model a boundary width that is not a multiple
(4,915 at 20%; 5,529, 3,379, and 2,252 at 60-80%). When a model is quantized,
those down projections are padded with zero neurons to the next multiple
(4,915 → 4,992). Zero neurons always output zero, so the function is
unchanged and the architectural edit is not affected. The real width is
reported and the padding bytes are counted. BF16 SwiGLU models stay
untouched.

## Evaluation

The full SwiGLU-8 final evaluation and Quantization-1's 8K KL, so every point
is comparable with the existing frontier.

## Cost and storage

| Item | Estimate |
| --- | --- |
| Order A, 51 evaluations | 15-25 GPU-h (low confidence; about 0.3-0.5 h each) |
| Order B, 6 QAT runs | 9-10 GPU-h (Quantization-1's QAT 100M took 1.51 GPU-h) |
| INT3 alone, PTQ and QAT | about 2 GPU-h |
| Storage, keeping everything | about 90-120 GB |

All models are kept by default. Deleting the order-A models after evaluation
is a later option if PROJECT gets crowded, since PTQ rebuilds them from the
kept BF16 models.

## Limits

- One run per variant.
- INT3 PTQ uses round-to-nearest.
- Bytes for INT3 are calculated, not measured.
- Speed is not evaluated beyond optional raw measurements; the quantized
  kernels are not optimized (decision log, 2026-10-16).
