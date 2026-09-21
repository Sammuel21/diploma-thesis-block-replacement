---
metadata_version: 1
title: Homogeneous SwiGLU Results Overview
type: experiment-synthesis
category: experiments/model/swiglu
status: active
created: 2026-09-21
modified: 2026-09-21
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
sources:
  artifacts:
    - data/results/notebook-model-study/swiglu-compression.json
    - data/results/notebook-model-study/swiglu-compression-optimized.json
    - data/results/notebook-model-study/swiglu-2.json
    - data/results/workflows/model/swiglu-3/run-001.json
    - data/results/workflows/model/swiglu-4/run-001.json
    - data/results/workflows/model/swiglu-5/search/run-001.json
    - data/results/workflows/model/swiglu-5/confirmation/run-001-target-0.2.json
    - data/results/workflows/model/swiglu-5/confirmation/run-001-target-0.5.json
---

# Homogeneous SwiGLU Results Overview

This page answers **what the homogeneous SwiGLU experiments achieved** and what
the evidence does, and does not, support. It groups results by finding rather
than retelling each workflow. For the sequence of decisions from SwiGLU-1
through SwiGLU-5, see the [experimental progression](swiglu-progression.md);
for implementation details, follow the [individual experiment pages](README.md).

All results concern the pinned SmolLM2-1.7B model. Layers 0 and 23 are
protected; layers 1–22 are eligible. A 20% or 50% removal target refers to
**parameters in those eligible MLPs**, not the whole model. Every replacement
is a narrower SwiGLU, although some eligible MLPs may be retained unchanged.

## The result in one view

The first random-initialized, model-wide 50% replacement was unusable: even
after its short recovery, WikiText validation perplexity was about 46,023.
Teacher-derived neuron initialization and stronger local fitting made a viable
uniform model at PPL 32.629. The best completed 100M-token SwiGLU-5 models
are substantially better again, but neither reaches the dense teacher's PPL
14.4396 or zero teacher KL. The early random result and the final runs are
**not** equal-budget ablations; they mark the practical starting point and
achieved frontier. [Initial artifact](../../../../data/results/notebook-model-study/swiglu-compression.json),
[optimized artifact](../../../../data/results/notebook-model-study/swiglu-compression-optimized.json).

| Model | Eligible-MLP removal | Whole-model parameter removal | WikiText PPL ↓ | Fixed teacher KL ↓ |
| --- | ---: | ---: | ---: | ---: |
| Dense teacher | 0% | 0% | **14.4396** | 0 by definition |
| SwiGLU-3, 100M tokens | 20% | 12.94% | 16.6543 | 0.075510 |
| **SwiGLU-5, 100M tokens** | **20%** | **12.94%** | **15.9491** | **0.074664** |
| SwiGLU-3, 100M tokens | 50% | 32.35% | 22.0908 | 0.220696 |
| **SwiGLU-5, 100M tokens** | **50%** | **32.35%** | **20.5286** | **0.194910** |

At the matched 100M-token endpoint, SwiGLU-5 reduces PPL relative to SwiGLU-3
by about **0.71** at 20% removal and **1.56** at 50%. The remaining PPL gaps
to dense are still about **1.51** and **6.09**, respectively. The 20% KL gain
is small; the 50% KL gain is more substantial. These are end-to-end method
comparisons: SwiGLU-5 changes both allocation and the recovery configuration,
so the full 100M difference is not attributable to allocation alone.
[SwiGLU-3 artifact](../../../../data/results/workflows/model/swiglu-3/run-001.json),
[SwiGLU-5 confirmations](swiglu-5.md#confirmation).

## What made the difference

### Better local construction made replacement viable

The earliest result established a threshold: simultaneous MLP replacement
could not be repaired adequately when the smaller operators began from random
weights. Selecting teacher neurons and fitting on more activation pairs cut
the damage dramatically. This was an enabling improvement, not evidence that
uniform width was the best final allocation.

### Replacement damage was a better allocation signal than block influence

The matched SwiGLU-2 comparison holds the 50% eligible-MLP parameter budget
and its short recovery protocol fixed. Values below are **PPL / teacher KL**;
the 25% label is the *probe width* used to score each layer, not the global
removal target.

| Allocation | Before recovery | After the same short recovery |
| --- | ---: | ---: |
| Uniform | 32.735 / 0.7469 | 32.629 / 0.7436 |
| Best canonical BI finalist | 32.949 / 0.7500 | 32.885 / 0.7476 |
| **Singleton KL at 25% width** | **29.707 / 0.6522** | **29.597 / 0.6493** |

Canonical block influence describes the original model; singleton KL directly
measures the damage from *this replacement at this width*. In this experiment,
the latter was a more useful score. It was still only one point on each
layer's width-response curve. This project-specific score-to-width adaptation
is not a head-to-head test of the full MoDeGPT method.
[SwiGLU-2 artifact](../../../../data/results/notebook-model-study/swiglu-2.json).

### Recovery helped, but allocation still mattered

With 393,216 local calibration pairs, SwiGLU-3's 50% model started at PPL
28.915. Its 100M-token replacement-only recovery reached 22.091: a large
improvement, but still far from dense. On the exact SwiGLU-3 starting model,
SwiGLU-4 found that constant `3e-5`, pure temperature-1 KL and
replacement-only training improved the matched 10M PPL from 24.686 to 24.037.
Extra RMSNorm or LoRA scope offered too little quality gain for its additional
cost, or performed worse. [SwiGLU-3 artifact](../../../../data/results/workflows/model/swiglu-3/run-001.json),
[SwiGLU-4 artifact](../../../../data/results/workflows/model/swiglu-4/run-001.json).

SwiGLU-5 then kept the improved recovery recipe equal across its search
candidates. The old ranked-width control (`S5-C0`) and discrete-width candidate
(`S5-C2`) provide the most informative construction comparison. Each cell is
**PPL / fixed recovery-validation KL**:

| Eligible-MLP removal | Construction | Before recovery | At the true 5M endpoint |
| ---: | --- | ---: | ---: |
| 20% | Earlier ranked widths | 17.688 / 0.1230 | 17.206 / 0.0964 |
| 20% | **Discrete width curves** | **16.906 / 0.1104** | **16.404 / 0.0919** |
| 50% | Earlier ranked widths | 28.915 / 0.4075 | 24.459 / 0.2846 |
| 50% | **Discrete width curves** | **28.181 / 0.3975** | **23.305 / 0.2814** |

Both constructions have the same target budget and SwiGLU-5 recovery schedule.
The comparison supports a real advantage from the discrete construction,
including its necessary refits at newly selected widths; it is not a pure
mathematical-solver-only ablation. At 20% removal the discrete method retained
11 of 22 eligible MLPs dense; at 50% it retained layers 21 and 22 dense and
compressed more tolerant layers harder. Output-aware reconstruction and the
composition-aware refit did not establish a consistent additional gain.
[SwiGLU-5 search artifact](../../../../data/results/workflows/model/swiglu-5/search/run-001.json).

## Cost and interpretation

The gain from width curves was not free. SwiGLU-2's complete 19-policy notebook
recorded **1.27 hours**; SwiGLU-5's complete two-target search recorded **6.97
hours**, including 4.84 hours building both legacy-subset and output-aware
width curves. Those totals cover different experimental scopes and are not
per-allocator GPU-hour estimates. The discrete solver itself was not the
observed bottleneck; fitting and evaluating many widths was. After selection,
SwiGLU-5's 100M-token trajectories ran about **3.5–3.9 times faster** than
SwiGLU-3's at the same token and optimizer-update budgets. That recovery
speedup does **not** make the whole SwiGLU-5 search cheaper than SwiGLU-3.

There are four important reading boundaries:

- WikiText PPL uses a stable family reference, but early experiments' teacher
  KL and later fixed recovery-validation KL use different splits. Compare KL
  within the matched tables above, not as one continuous SwiGLU-1-to-5 series.
- Singleton-KL sums are an allocation *surrogate*. Earlier
  [block-interaction evidence](../interaction/block-interaction.md) found
  non-additive joint damage, so assembled-model evaluation remains decisive.
- The original SwiGLU-5 search selected `S5-C2` at both targets because a 2M
  row was mislabeled as 5M for winner selection. At the genuine 5M endpoint,
  the combined output-reconstruction-plus-discrete candidate was marginally
  better at 20% (0.000128 KL and 0.029 PPL). The completed 20% confirmation
  therefore follows a valid near-runner-up, not the exact 5M winner. The 50%
  winner is unaffected. Historical artifacts remain unchanged.
- A matched four-way **uniform versus BI versus singleton-25% versus discrete**
  comparison under one SwiGLU-5 fitting and recovery protocol has not been
  run. The SwiGLU-2 table isolates the earlier allocation signals; the
  SwiGLU-5 table isolates the later construction under its own recovery recipe.

The defensible conclusion is that homogeneous SwiGLU replacement reached a
useful, substantially improved compression–quality frontier, especially at
20% eligible-MLP removal. The dense gap persists, most visibly at 50%, and
heterogeneous operator allocation remains untested in this experiment family.
