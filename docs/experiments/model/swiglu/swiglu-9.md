---
title: SwiGLU-9 - Local Fitting at 1B Tokens and Calibration Scaling
status: draft
created: 2026-10-17
modified: 2026-10-17
---

# SwiGLU-9 - Local Fitting at 1B Tokens and Calibration Scaling

**Status:** design settled on 2026-10-17 in a grilling session after the
[supervisor meeting of 2026-10-16](../../../meetings/2026-10-16-supervisor-feedback.md).
Not implemented and not run. The binding decisions are the 2026-10-16 and
2026-10-17 rows in the [shared decision log](../../../agents/decisions.md);
this page explains them. If this page and the log disagree, the log wins until
the conflict is raised with the researcher.

## Questions

1. **Part B (main):** does local fitting still improve the compressed model
   after 1B recovery tokens, or does the slice-only model catch up? The
   supervisor called this the most important open question.
2. **Part A (side):** how far does the local fit keep improving with more
   calibration pairs beyond 393,216, and does a keep-near-teacher penalty
   reduce the remaining overfitting?

## Why this experiment

[SwiGLU-8](swiglu-8.md) Part B found that local fitting lowers the 8K KL at
100M by 10-24% at every target, but the gap was still shrinking, so every
target was "inconclusive at 100M". A rough power-law extrapolation suggested
a gap of 3-8% at 1B; that is an extrapolation, not a result, and only a 1B
measurement settles it.

SwiGLU-8 Part A found the local-fitting recipe on a plateau: no setting moved
the mean 8K single-block KL by more than 4.4%. Its stage-2 indicator fired
(mild overfitting at width 0.5 in all three test layers), and SwiGLU-3 showed
that each doubling of calibration pairs up to 393,216 still helped by about
3-4%. Stage 2 was deferred until Part B showed whether fitting matters.

## Part A - calibration scaling (block level)

Part A is a grid of loss × calibration size. All settings use the
`c1e-3-128` recipe; only the loss and the number of pairs change.

| Loss | 393,216 | 786,432 | 1,572,864 | 3,145,728 |
| --- | :---: | :---: | :---: | :---: |
| MSE (current) | reused from SwiGLU-8 Part A (seeds 21-23) | ✓ | ✓ | ✓ |
| MSE + L2-SP, weak | ✓ | ✓ | ✓ | ✓ |
| MSE + L2-SP, strong | ✓ | ✓ | ✓ | ✓ |

New settings use seed 21: 11 settings, 66 fits. **L2-SP** adds a penalty that
pulls the weights toward their starting values, here the copied teacher
slice, instead of toward zero as weight decay does. The original method is
`src-l2-sp-2018`; its penalty equation is source-derived and is cited at a
checked locator when reported. The two strengths are chosen and justified in
the implementation plan, relative to the scale of the MSE loss. All local
fits so far (SwiGLU-7 and SwiGLU-8) are pure MSE; L2-SP was never run.

**Why a full grid:** more data and L2-SP are two fixes for the same mild
overfitting at width 0.5. The likely pattern is that L2-SP helps at 393,216
pairs and its benefit shrinks as the data grows. Only the full grid shows
whether the two fixes add up. SwiGLU-3 (about 3-4% per data doubling) and
SwiGLU-8 Part A (a plateau, at most 4.4%) suggest the gains will be small and
may not pass the 5% adoption bar. That is still a useful result: it shows
where the local fit saturates and why the recipe stays.

- **Scope:** SwiGLU-8 Part A's six combinations (layers 3, 11, 19 at widths
  0.25 and 0.5).
- **Scoring and adoption:** as SwiGLU-8 Part A: single-block KL on the 8K
  selection stream decides, 128-token KL guards, NMSE is reported. The
  seed-noise term of the adoption rule comes from SwiGLU-8 Part A's three
  `c1e-3-128` seeds, since the new settings run one seed (SwiGLU-8 seed noise
  was 0.2-0.8%).
- **Use of the result:** an adopted setting applies only to later experiments
  (heterogeneous, Stacking). It never changes Part B, so the 1B answer refers
  to the recipe measured at 100M.
- **Data and memory:** a pair is one token position (MLP input and output).
  393,216 pairs are 3,072 windows of 128 tokens, so C4 shard 0 is not a limit.
  Memory is: FP32 inputs plus outputs for one layer take about 6.4 GB at
  393,216 pairs, 26 GB at 1.57M, and 52 GB at 3.1M (offline arithmetic; the
  storage format in code was not checked). Jobs have 128 GB RAM, so about 3M
  is the ceiling without code changes. The new calibration windows must stay
  disjoint from the 8K streams, as in SwiGLU-8.
- **Cost:** roughly 5-12 GPU-h (estimate, low confidence; the 3,145,728-pair
  fits dominate).
- **Attribution:** L2-SP is `src-l2-sp-2018` (Li et al., ICML 2018),
  registered on 2026-10-17; check the locator before reporting.

## Part B - fitted versus slice-only at 1B tokens

**Targets:** 20, 50, and 80% eligible-MLP removal (supervisor rule for 1B
runs). Full 20-80% trends stay at 100M in SwiGLU-8.

**Models:**

| Model | Source |
| --- | --- |
| Fitted 20%, fitted 50% | Reused: SwiGLU-7 S7-0 1B runs (same models and recipe; SwiGLU-8 B1 and P2 matched them exactly at token 0). Only their 8K KL is evaluated |
| Fitted 80%, slice-only 20%, 50%, 80% | New 1B runs from SwiGLU-8's token-0 starts (`prepare-001/prepared/starts/` on PERUN) |
| Fitted 50%, second copy (the twin) | New 1B run, to measure run-to-run noise |

**Recipe:** S7-0 replacement-only recovery to exactly 1B tokens, identical to
SwiGLU-8 Part B except the token budget. Each run restarts from token 0,
because the 100M optimizer checkpoints were deleted. Monitoring KL every 25M
tokens, matching SwiGLU-7, so the gap is a full curve from 0 to 1B. Final
evaluation at 1B is the full SwiGLU-8 evaluation, including Quantization-1's
8K KL.

**Why the twin.** Runs are not bitwise repeatable: GPUs add numbers in an
order that can change between runs, rounding makes the order matter in the
last digits, and training magnifies those differences over many steps. At
100M two runs of the same model differed by up to 3.5% at a single
checkpoint. The twin is the same model as SwiGLU-7's fitted 50% (same start,
data, order, and recipe), so any difference between the two is noise. Only
fitted 20% and 50% already exist at 1B, so a twin there costs one run; 50% is
the middle of the range.

**Verdict per target (fixed before running):**

- Gap g = (slice-only − fitted) / fitted on the 8K KL at 1B.
- Noise = the largest relative difference between the two fitted-50% runs,
  over the 25M monitoring checkpoints from 500M to 1B and their final 8K KL.

| Result | Verdict |
| --- | --- |
| g < 5%, noise below 5% | Closed: local fitting no longer matters after 1B tokens |
| g ≥ 5%, larger than the noise | Local fitting still matters at 1B |
| Noise 5% or more, and g within the noise | Can't tell |

The 5% bar replaces the 100M rule's 2% and is a judgment call made after the
100M results: at 100M a 10% KL gap came with a 1-2.4% perplexity gap, so 5%
KL is roughly 1% perplexity. The noise is measured at 50% only and assumed
similar at 20% and 80%.

**Reported per target, descriptive:**

- the gap curve from 0 to 1B on the monitoring KL;
- the closing point: the first checkpoint after which the monitoring gap stays
  below 5% until 1B, or "never";
- the token equivalent: how many recovery tokens the slice-only model needs
  to reach the fitted model's quality at a given point.

**Reuse:** the six 1B models at 20/50/80% are reused wherever applicable, for
example by Stacking-1 order A and as the 1B homogeneous reference for the
heterogeneous chapter (decision log, 2026-10-17).

## Cost

| Item | Estimate |
| --- | --- |
| Part A | 5-12 GPU-h (low confidence) |
| Part B, 5 new 1B runs | 55-60 GPU-h; each run about 11-12 h on one H200 (S7-0 measured 9.8-10.5 h of training) |
| 8K KL of the two reused SwiGLU-7 models | about 0.6 GPU-h |
| Storage | about 15-20 GB on PROJECT; check `perunfsusage` before submission |

With the runs in parallel, Part B takes about half a day of elapsed time plus
queueing. A resource plan under
[PERUN resource accounting](../../../infrastructure/perun/perun-resources.md)
is written before submission.

## Implementation needed

- The SwiGLU-8 runner fixes recovery at 100M through its configuration. Part
  B needs a 1B target, monitoring every 25M, separate output names, and a
  small job evaluating the 8K KL of existing bundles.
- The twin's reproduction check against SwiGLU-7 covers the full 1B history
  with SwiGLU-8's 1% review rule (record and review, never stop), plus the
  exact token-0 check.
- Part A extends the block check with larger calibration sets and an L2-SP
  loss option in the shared fitting code; pure MSE stays the default.
- No smoke or reduced-budget runs.

## Limits

- One twin, at 50% only.
- One run per model; the 8K KL is measured once per final model.
- The 1B answer is about the current fitting recipe; a better Part A recipe is
  not tested at 1B.
- The 5% bar is a judgment call.
