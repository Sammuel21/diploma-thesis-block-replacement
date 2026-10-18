---
title: SwiGLU-9 - Local Fitting at 1B Tokens and Calibration Scaling
status: draft
created: 2026-10-17
modified: 2026-10-18
reference_artifacts:
  - data/results/workflows/model/swiglu-9/part-b-summary-001/result.json
  - data/results/workflows/model/swiglu-9/block-check-001/result.json
  - data/results/workflows/model/swiglu-9/reference-kl-001/result.json
  - data/results/workflows/model/swiglu-9/fitted-target-0.5-1b-run-001/result.json
  - data/results/workflows/model/swiglu-9/fitted-target-0.8-1b-run-001/result.json
  - data/results/workflows/model/swiglu-9/slice-target-0.2-1b-run-001/result.json
  - data/results/workflows/model/swiglu-9/slice-target-0.5-1b-run-001/result.json
  - data/results/workflows/model/swiglu-9/slice-target-0.8-1b-run-001/result.json
---

# SwiGLU-9 - Local Fitting at 1B Tokens and Calibration Scaling

**Status:** design settled on 2026-10-17 in a grilling session after the
[supervisor meeting of 2026-10-16](../../../meetings/2026-10-16-supervisor-feedback.md)
and implemented the same day (see [Implementation](#implementation)). All nine
tasks ran on PERUN on 2026-10-17/18 (jobs 117283-117285, commit `ae67eab`) and
completed on the first attempt with every stopping check passed. The results
were audited on 2026-10-18 by recomputing every reported value from the run
records.

- **Part B outcome:** under the logged rule, local fitting no longer matters
  after 1B recovery tokens, at all three targets. The 8K KL gap g is 2.77% at
  20%, 1.83% at 50%, and 0.33% at 80%, all below the 5% bar, and the twin
  noise is 0.31%. A small fitted advantage remains at 20% and 50%. At 100M the
  gaps were 10-18%. See [Part B results](#part-b-results-2026-10-18).
- **Part A outcome:** no setting qualifies, so the `c1e-3-128` MSE recipe at
  393,216 pairs stays. Neither more calibration data nor L2-SP improves the
  mean 8K single-block KL by more than 2.2%. See
  [Part A results](#part-a-results-2026-10-18).

**Next step:** the researcher reviews these results and the proposed decision
rows. Whether later experiments still need local fitting is a question for
the supervisor. The load-only analysis notebook is still to be written.

The binding decisions are the 2026-10-16 and 2026-10-17 rows in the
[shared decision log](../../../agents/decisions.md); this page explains them.
If this page and the log disagree, the log wins until the conflict is raised
with the researcher.

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

New settings use seed 21: 11 settings, 66 fits. Each layer also refits MSE at
393,216 pairs once more as a reproduction check (6 fits, 72 in total); it
confirms that the reused SwiGLU-8 values still apply and it calibrates L2-SP.
All local fits so far (SwiGLU-7 and SwiGLU-8) are pure MSE; L2-SP was never
run.

**L2-SP** adds a penalty that pulls the weights toward their starting values,
here the copied teacher slice, instead of toward zero as weight decay does:
Ω(w) = (α/2)‖w − w⁰‖² [src-l2-sp-2018, Section 3, paragraph "L2-SP",
Eq. (2); locator checked on arXiv 1802.01483v2 on 2026-10-17]. The paper's
Eq. (3) adds a second term for parameters without a pre-trained counterpart.
Every replacement weight here is copied from the teacher, so that term does
not apply (project interpretation). The paper minimizes the negative
log-likelihood with SGD and momentum (Section 4.2). Here the loss is the
local-fit MSE and the optimizer AdamW without weight decay, with the penalty
added to the loss as in the paper. Early stopping still uses the validation
MSE alone.

**The two strengths (project-proposed, chosen in the implementation plan).**
The MSE scale differs about 100-fold between layers 3 and 19 (SwiGLU-8 anchor
fits), so one α for all six combinations would be strong in one layer and
weak in another. Each layer-width combination therefore gets its own α from
its MSE fit at 393,216 pairs, the reproduction fit:

> α = 2ρ · ΔMSE / d²

ΔMSE is the training-MSE gain of that fit over the unfitted slice on the
393,216-pair prefix, and d² = ‖w_fit − w⁰‖². At the fit's own displacement,
the penalty then costs ρ times the fit's gain. In a one-dimensional quadratic
model of the training MSE along the fit direction, the L2-SP minimum lies at
1/(1+ρ) of that displacement. **Weak ρ = 0.1** keeps about 91% and mainly
damps directions of low curvature, where overfitting happens. **Strong
ρ = 1** keeps 50%, which brackets the effect from the other side. The tenfold
spacing follows the paper's decade grid (Figure 1 scans α from 10⁻³ to 1).
The paper's Section 4.4.2 states the general form of this picture: under
simplifying assumptions, the L2-SP optimum is an affine combination of the
unregularized optimum and w⁰ along the Hessian's eigenvectors. Its derivation
is in the paper's supplement, which was not checked.

α stays fixed across the four sizes. The MSE is a mean, so its weight does not
grow with the number of pairs, as with the paper's fixed α. More data can then
only reduce the variance that L2-SP counters, which is exactly the question
below. Adam's per-coordinate scaling changes the optimization path, but not
this objective. Each fit therefore records its realized ‖w − w⁰‖² relative to
the MSE fit's, so the pull is measured rather than assumed.

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
  reference is SwiGLU-8's `c1e-3-128` seed mean over seeds 21-23. The new
  settings run one seed, so their missing spread is taken to equal the
  reference's: the seed-noise term is 2·SD/mean of SwiGLU-8's three seeds
  (SwiGLU-8 seed noise was 0.2-0.8%).
- **Use of the result:** an adopted setting applies only to later experiments
  (heterogeneous, Stacking). It never changes Part B, so the 1B answer refers
  to the recipe measured at 100M.
- **Data:** a pair is one token position (MLP input and output). 393,216
  pairs are 3,072 windows of 128 tokens, which uses up SwiGLU-8's
  calibration partitions. The larger sizes add 21,504 windows from C4 shard 0
  using the same sampler: seed 21, one window per document. The new windows
  skip every record that supplied a 128-token window and every record up to
  the last 8K selection-stream document (record 1601), so they stay disjoint
  from all 8K streams. The sizes are nested prefixes of one list. The neuron
  ranking is the one from the 393,216 pairs at every size, so only fitting
  changes.
- **Memory:** pairs are stored in BF16 on CPU (checked in code on 2026-10-17;
  the earlier FP32 figures were twice too high). That is 3.2 GB per layer at
  393,216 pairs and 25.8 GB at 3,145,728. Each task captures one layer once,
  at the largest size, and serves every size as a prefix view. The estimated
  peak is 45-55 GB of the 128 GB requested (arithmetic, not measured).
- **Attribution:** L2-SP is `src-l2-sp-2018` (Li, Grandvalet, and Davoine,
  ICML 2018), at the locator above.

### Part A results (2026-10-18)

Source: `swiglu-9/block-check-001/result.json`. The adoption table was
recomputed from its 72 fit rows and SwiGLU-8's three `c1e-3-128` reference
seeds; it matched the merge exactly.

**Checks.**
- D1, D2, P5, A1, and A2-A3 passed in all three layers.
- The six reproduction fits (MSE at 393,216 pairs, seed 21) reproduced
  SwiGLU-8's seed-21 fits exactly: the relative difference is 0 for both the
  128-token and the 8K KL. Reusing SwiGLU-8's cell is therefore exact.
- The 21,504 extension windows are document-disjoint from the 8K streams, as
  designed.

**No setting qualifies; `c1e-3-128` at 393,216 pairs stays.** The values are
reductions in single-block KL against the SwiGLU-8 reference mean, so
positive is better. The seed-noise term is 0.42%. The adoption bar is a mean
8K reduction above 5% and above the noise, with no combination more than 5%
worse.

| Setting | Mean 8K | Mean 128-token | Worst combination | Fails |
| --- | ---: | ---: | --- | --- |
| MSE, 786,432 | +0.35% | +3.16% | -8.52% (layer 3, width 0.25, 8K) | mean, noise, worst |
| MSE, 1,572,864 | +0.92% | +4.08% | -8.63% (layer 3, width 0.25, 8K) | mean, worst |
| MSE, 3,145,728 | +1.41% | +4.38% | -7.27% (layer 3, width 0.25, 8K) | mean, worst |
| L2-SP weak, 393,216 | -0.22% | +0.63% | -4.18% (layer 3, width 0.25, 8K) | mean, noise |
| L2-SP weak, 786,432 | +1.61% | +2.68% | -1.40% (layer 3, width 0.25, 8K) | mean |
| L2-SP weak, 1,572,864 | +1.48% | +3.03% | -4.88% (layer 3, width 0.25, 8K) | mean |
| L2-SP weak, 3,145,728 | **+2.15%** | +3.09% | -2.96% (layer 3, width 0.25, 8K) | mean |
| L2-SP strong, 393,216 | -2.99% | -5.48% | -12.16% (layer 19, width 0.25, 128) | mean, noise, worst |
| L2-SP strong, 786,432 | -2.55% | -4.79% | -10.93% (layer 19, width 0.5, 128) | mean, noise, worst |
| L2-SP strong, 1,572,864 | -2.45% | -4.91% | -11.11% (layer 19, width 0.25, 128) | mean, noise, worst |
| L2-SP strong, 3,145,728 | -2.28% | -4.72% | -10.20% (layer 19, width 0.5, 128) | mean, noise, worst |

**Reading.**

- **More data helps the 128-token KL but barely the 8K KL.** Eight times the
  pairs lowers the mean 128-token KL by 4.4%, which is less than SwiGLU-3's
  3-4% per doubling would predict. The mean 8K KL improves by at most 1.4%.
  At layer 3, width 0.25, the 8K KL gets 7-9% *worse* at every larger size,
  while the other five combinations improve by 1-5%. One possible
  explanation, not tested: the calibration windows are 128 tokens long, so a
  closer fit to short-context activations does not transfer to 8K contexts
  at that combination.
- **Weak L2-SP is the best family but still far from the bar.** It reaches
  +2.15% at 3,145,728 pairs and has the smallest worst cases. Layer 3,
  width 0.25 is again its weakest combination.
- **Strong L2-SP is worse everywhere.** It is mainly worse at layer 19 and at
  layer 11 width 0.5. Pulling the fit back toward the copied teacher slice
  costs more than the overfitting it removes.
- **The overfitting indicator disappears without a KL gain.** The stage-2
  indicator (best epoch at most 3 while the training MSE still falls, at
  width 0.5) fires for MSE at 393,216 pairs in all three layers, and at
  786,432 in layers 3 and 19. It is gone for MSE from 1,572,864 pairs, and for
  weak L2-SP up to 1,572,864 pairs. Removing the overfitting symptom buys at
  most about 2% of 8K KL. The overfitting is real, but it costs little.
- **The realized L2-SP pull is stronger than the one-dimensional model
  predicted.** The displacement ratio ‖w − w⁰‖² relative to the MSE fit's is
  0.49-0.95 for weak L2-SP (about 0.91 predicted) and 0.13-0.19 for strong
  (0.5 predicted). The strong setting therefore brackets the effect from
  further away than designed. MSE with more data moves the weights
  1.05-2.31 times as far as at 393,216 pairs.
- **The α values span about 100×** (weak: 3.0e-7 at layer 3 to 3.1e-5 at
  layer 19), as the MSE scale predicted. Each value was recomputed from the
  rule α = 2ρ·ΔMSE/d² and matches.

**Use:** nothing is adopted, so the heterogeneous chapter and Stacking-1 keep
`c1e-3-128` at 393,216 pairs. Part B was never affected (design rule).

## Part B - fitted versus slice-only at 1B tokens

**Targets:** 20, 50, and 80% eligible-MLP removal (supervisor rule for 1B
runs). Full 20-80% trends stay at 100M in SwiGLU-8.

**Models:**

| Model | Source |
| --- | --- |
| Fitted 20%, fitted 50% | Reused: SwiGLU-7 S7-0 1B runs (same models and recipe; SwiGLU-8 B1 and P2 matched them exactly at token 0). Only their 8K KL is evaluated |
| Fitted 80%, slice-only 20%, 50%, 80% | New 1B runs from SwiGLU-8's token-0 starts (`prepare-001/prepared/starts/` on PERUN) |
| Fitted 50%, second copy (the twin) | New 1B run, to measure run-to-run noise |

**Recipe:** S7-0 replacement-only recovery to exactly 1B tokens. The
configuration's `recovery` block equals SwiGLU-7's: segment endpoints at 100M
and 1B, monitoring KL every 25M tokens, and WikiText perplexity at 10, 25,
50, 100, 250, 500M, and 1B. It differs from SwiGLU-8 Part B only in the token
budget. Each run restarts from token 0, because the 100M optimizer
checkpoints were deleted. SwiGLU-8's prepared stream holds only 100M tokens,
so the runs read SwiGLU-7's 1B stream, whose first 100M tokens are SwiGLU-8's
stream (check P4). Final evaluation at 1B is the full SwiGLU-8 evaluation,
including Quantization-1's 8K KL.

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
  That is 21 checkpoints plus the final value, with SwiGLU-7's run as the
  denominator, as in SwiGLU-8's B2 review.

| Result | Verdict |
| --- | --- |
| g < 5%, noise below 5% | Closed: local fitting no longer matters after 1B tokens |
| g ≥ 5%, larger than the noise | Local fitting still matters at 1B |
| Noise 5% or more, and g within the noise | Can't tell |

The logged rule does not cover a slice-only advantage larger than a noise of
5% or more (g < −noise). The code reports that case as "outside the logged
rule" rather than guessing.

The 5% bar replaces the 100M rule's 2% and is a judgment call made after the
100M results: at 100M a 10% KL gap came with a 1-2.4% perplexity gap, so 5%
KL is roughly 1% perplexity. The noise is measured at 50% only and assumed
similar at 20% and 80%.

**Reported per target, descriptive:**

- the gap curve from 0 to 1B on the monitoring KL;
- the closing point: the first checkpoint after which the monitoring gap stays
  below 5% until 1B, or "never";
- the token equivalent: how many recovery tokens the slice-only model needs
  to reach the fitted model's quality at a given point. It is computed as the
  first slice-only crossing of the fitted value, interpolated linearly
  between neighbouring checkpoints, or "> 1B";
- g at 50% recomputed with the twin, and the twin's final-evaluation and
  environment differences from SwiGLU-7.

**Reuse:** the six 1B models at 20/50/80% are reused wherever applicable, for
example by Stacking-1 order A and as the 1B homogeneous reference for the
heterogeneous chapter (decision log, 2026-10-17).

### Part B results (2026-10-18)

Source: `swiglu-9/part-b-summary-001/result.json`. Every value below was
recomputed independently from the five run records, `reference-kl-001`, and
SwiGLU-7's S7-0 run records, and matched the summary. The 100M values are
SwiGLU-8's.

**Verdict: closed at every target.** Under the logged rule, local fitting no
longer matters after 1B recovery tokens.

| Target | Whole-model removal | 8K KL fitted | 8K KL slice-only | **g at 1B** | g at 100M | Absolute gap 1B / 100M | Verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 20% | 12.9% | 0.058213 (SwiGLU-7) | 0.059827 | **2.77%** | 12.6% | 0.0016 / 0.0099 | closed |
| 50% | 32.4% | 0.157788 (SwiGLU-7) | 0.160673 | **1.83%** | 10.4% | 0.0029 / 0.0238 | closed |
| 80% | 51.8% | 0.375228 | 0.376480 | **0.33%** | 18.4% | 0.0013 / 0.0999 | closed |

**Noise.** The twin noise is 0.31%: the largest relative difference between
the two fitted-50% runs over the 21 monitoring checkpoints from 500M to 1B
occurred at 900M, and their final 8K KL differed by only 0.02%. The noise is
far below 5%, so the verdict is valid.
- With the twin as the fitted model, g at 50% is 1.81%, against 1.83% with
  SwiGLU-7's run.
- The twin's final evaluation matches SwiGLU-7's: perplexities within 0.06%,
  task accuracies within 0.6 percentage points. The recorded environment is
  identical.

**The gap over training** (monitoring KL, the recovery-validation KL behind
the curves):

| Target | 0 | 10M | 25M | 50M | 100M | 500M | 1B | Range 600M-1B | Gap stays < 5% from |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 20% | 111% | 34% | 24% | 21% | 14% | 5.5% | 5.0% | +0.0% to +6.4% | 1B |
| 50% | 201% | 26% | 20% | 65%* | 11% | 4.3% | 3.5% | +1.8% to +4.1% | 450M |
| 80% | 427% | 71% | 52% | 35% | 18% | 1.7% | -0.8% | -1.4% to +1.1% | 325M |

\* The flagged transient in the slice-only 50% run; see the review below.

**Token equivalent.** The slice-only model reaches the fitted model's
monitoring KL at 100M after 296M tokens at 20% (3.0×), 174M at 50% (1.7×), and
177M at 80% (1.8×). It reaches the fitted model's 500M value after 537M, 677M,
and 537M tokens. It reaches the fitted 1B value within 1B only at 80% (after
929M). Local fitting is worth a head start of roughly 75-200M recovery tokens
early on. Long recovery erases most of it.

**Reading.**

- **Closed does not mean zero.** At 20% and 50% the fitted model is still
  better, by 2.8% and 1.8% on the 8K KL. That is well above the 0.31% twin
  noise but below the 5% practical bar. The bar is a judgment call made after
  the 100M results (decision log, 2026-10-17). The accurate statement is "after
  1B tokens, local fitting leaves under 3% of 8K KL".
- **At 20%, the monitoring metric sits near the bar.** The monitoring gap ends
  at 4.97% and moves between 0.0% and 6.4% from 600M onward. Both arms move
  together at several checkpoints (at 925M both rise to 0.072), so these
  checkpoint values depend on the data just seen. The deciding 8K KL
  (196,608 positions) gives 2.77%. The verdict stands under the logged rule,
  but the 20% result is the least clear-cut of the three.
- **At 80%, the arms are effectively equal.** From 800M onward the slice-only
  model is slightly better on the monitoring KL (by up to 1.4%), and its test
  perplexity at 8K context is 3.5% lower (11.97 against 12.41). The fitted
  model is better on the 8K KL (by 0.33%) and on macro task accuracy (by 0.34
  points). The perplexity difference is larger than the twin's perplexity
  noise at 50% (0.06%), but noise at 80% was not measured.
- **Secondary metrics at 20% and 50%:** test perplexity at 8K context differs
  by 0.12% and 0.11%, and macro accuracy by -0.26 and +0.18 points
  (slice-only minus fitted).

**Observation: the gap closes faster at higher removal.** At 100M the 80%
target had the largest gap (18.4%) and the steepest decline (SwiGLU-8:
−16.9 points between 50M and 100M, against −5.6 at 20%). At 1B it has the
smallest gap, both relatively and in absolute KL (0.0013, against 0.0099 at
100M). Possible explanations, none tested:
1. **The relative measure.** g divides by the fitted KL, which is about six
   times larger at 80% than at 20%. This explains the order of 20% and 50%:
   50% has the larger absolute gap (0.0029 against 0.0016). It does not
   explain 80%, whose absolute gap is also the smallest.
2. **More for recovery to repair.** A heavily compressed model starts far from
   the teacher, so recovery training corrects large errors quickly. The local
   fit's head start is then a smaller share of what recovery repairs anyway.
3. **Allocation structure.** Which layers are cut, and how far, differs
   between targets. SwiGLU-8 already found 75% and 80% behaving differently
   for this reason.

Each target has one run per arm, and the noise was measured at 50% only.

**Flag review (B2/C2).** Two of the 45 C2 values exceed 1%. Both are from the
slice-only 50% run at the 50M checkpoint:

| Tokens | Monitoring KL, SwiGLU-9 / SwiGLU-8 | Difference | WikiText perplexity, SwiGLU-9 / SwiGLU-8 | Mean training KL over the preceding 25M |
| ---: | --- | ---: | --- | --- |
| 10M | 0.35386 / 0.35379 | +0.02% | 25.56 / 25.55 | 0.4449 / 0.4449 |
| 25M | 0.30705 / 0.30678 | +0.09% | 24.09 / 24.11 | 0.3913 / 0.3912 |
| **50M** | **0.39129 / 0.27224** | **+43.7%** | **25.79 / 22.98** | 0.3560 / 0.3523 |
| 75M | 0.25188 / 0.25005 | +0.73% | - | 0.3342 / 0.3314 |
| 100M | 0.23897 / 0.23729 | +0.71% | 21.73 / 21.73 | 0.3187 / 0.3164 |

- **The jump is a transient in the model state at the 50M checkpoint.** Both
  validation metrics jump, while the mean training KL over the preceding 25M
  tokens differs by only 1%. Whatever happened was short.
- **The run recovered within 25M tokens** and stays within 1% of SwiGLU-8
  afterward. The 1B values, about 950M tokens later, are unaffected.
- **The cause is not identified.** Both runs see the same data in the same
  order. GPU nondeterminism amplified by training, as in SwiGLU-8's B2 review,
  is consistent with this, but this jump is far larger than any earlier bump
  (at most 3.5%), and unlike those it also moved perplexity.
- **Everything else is well within 1%.** The other 43 C2 values are within 1%
  (the largest is 0.99%, slice-only 20% at 100M). All 48 B2 values of the twin
  are within 0.31%. All C1, B1, and P4 checks passed.
- **Conclusion:** the flagged values do not affect the Part B verdict.

## Implementation

Implemented on 2026-10-17 without any run. Runner:
[`swiglu_9.py`](../../../../workflows/runs/model/swiglu/swiglu_9.py).
Configuration:
[`swiglu-9.json`](../../../../workflows/configs/model/swiglu/swiglu-9.json).
PERUN job: [`swiglu_9.sbatch`](../../../../workflows/jobs/perun/swiglu_9.sbatch),
described in the [PERUN job README](../../../../workflows/jobs/perun/README.md#swiglu-9-workflow).
The copy-paste commands are in the
[PERUN runbook](../../../infrastructure/perun/perun-runbook.md#9-swiglu-9). The
analysis notebook is written in the analysis phase.

The runner is new; `swiglu_7.py` and `swiglu_8.py` are unchanged. It reuses
their helpers and mirrors SwiGLU-8's recovery loop, so finished experiments
keep their code. The configuration is checked against SwiGLU-7's (`model`,
`data`, `calibration`, `local_fitting`, `evaluation`, `recovery`, S7-0) and
SwiGLU-8's (`long_context`, `allocation`, acceptance tolerances, the
`c1e-3-128` recipe) versioned configurations. It pins every SwiGLU-7 and
SwiGLU-8 input by sha256.

| Command | Does | Output (`perun-results/swiglu-9/`) |
| --- | --- | --- |
| `block-check --task-index i` | Part A for layer 3, 11, or 19 (tasks 0-2): data checks, slices, the reproduction fit with L2-SP calibration, then the 11 new settings at both widths (24 fits). Resumes per completed fit | `block-check-001-task-NN/` |
| `merge-block-check` | Login node, no GPU. Checks that the three tasks used identical configuration, code, software, 8K streams, and extension windows. Joins the 72 fits and applies the adoption rule against SwiGLU-8's three seeds. Reports the stage-2 indicator per setting and the α table. Safe to re-run | `block-check-001/` |
| `train --arm --target` | Part B (tasks 0-4: fitted 0.5 twin, fitted 0.8, slice-only 0.2, 0.5, 0.8). S7-0 recovery to 1B from SwiGLU-8's start on SwiGLU-7's stream, the full final evaluation, and the 8K KL of the reloaded BF16 bundle | `{arm}-target-{t}-1b-run-001/` |
| `reference-kl` | 8K KL of SwiGLU-7's S7-0 1B bundles at 20% and 50%, after checking each bundle manifest against its pinned run | `reference-kl-001/` |
| `summarize-part-b` | Login node, no GPU. The verdict per target, the twin noise, gap curves, closing points, token equivalents, and every B2/C2 value flagged for review. Safe to re-run | `part-b-summary-001/` |

**Shared code**, each extension with a default that keeps earlier experiments
unchanged:

- `fit_operator_fp32_detailed(..., l2_sp_alpha=0.0)` adds the L2-SP penalty
  toward the parameters on entry. At zero, no copy is made and the loss is the
  same expression, so MSE fits are unchanged.
- `swiglu5.fitting.fit_operator` forwards `l2_sp_alpha` for teacher-slice
  starts only, so MSE and L2-SP fits share one code path.
- `sample_partitioned_windows(..., excluded_records=None)` skips listed
  records before drawing a window start; `None` keeps the sampling sequence.

### Acceptance checks

These are part of the real runs, not separate smoke runs. Tolerances are
SwiGLU-8's (relative difference r). A failed stopping check ends the stage
after recording its values.

| ID | Check | Reference | Tolerance | On failure |
| --- | --- | --- | --- | --- |
| D1 | Base 128-token windows, sha256 of the four 8K streams, extension window count, and its document disjointness | SwiGLU-8 block check | exact | stop |
| D2 | The layer's neuron ranking | SwiGLU-8 block check | exact | stop |
| P5 | Teacher-against-itself 8K KL (block check and reference KL) | 0 | ≤ 1e-6 absolute | stop |
| A1 | Slice NMSE at both widths | SwiGLU-8 slices | r ≤ 1e-4 | stop |
| A2-A3 | Reproduction fit: local NMSE and MSE, single-block KL at 128 and 8K | SwiGLU-8 `c1e-3-128`, seed 21 | r ≤ 1e-3 | stop |
| P4 | First 100M tokens of SwiGLU-7's 1B stream | SwiGLU-8's 100M stream | identical sha256 | stop |
| C1 | Token-0 monitoring KL and PPL, all five runs | SwiGLU-8's 100M run of the same arm and target | r ≤ 1e-4 | stop |
| C2 | Same runs at 10-100M | same | recorded; reviewed against 1% | never stops |
| B1 | Twin, token-0 monitoring KL and PPL | SwiGLU-7 S7-0 at 50% | r ≤ 1e-4 | stop |
| B2 | Twin at 10M-1B (41 KL and 7 PPL values) | same | recorded; reviewed against 1% | never stops |

C1-C2 and the reproduction fit go beyond the logged design. They are cheap
guards against a wrong start or wrong data before 10 h of GPU time, and the
reproduction fit backs the reuse of SwiGLU-8's 393,216-pair cell. As in
SwiGLU-8, flagged B2/C2 values are investigated before any Part B result is
used.

**Not yet verified.** Contract tests in `tests/test_swiglu9_contracts.py` and
`tests/test_local_fitting_contracts.py` cover:

- the configuration and its rejection of changed contracts;
- the L2-SP option (zero is the plain fit, the pull grows with α, and the
  recorded MSE is the data term);
- the excluded-record sampling and the α formula, including its 1/(1+ρ)
  property;
- the prefix views, the data and reproduction checks, and single-seed
  adoption;
- the merge, the verdict table, the twin noise window, gap curves, closing
  points, and token equivalents;
- the C1/C2 checks, the Part B summary, and the job mapping.

The full suite passed (105 tests, one skipped, against 83 before). GPU
execution, real data, memory use, and the acceptance checks themselves have
not run; the first real stages are also the first execution test. They ran on
PERUN on 2026-10-17/18, and every stopping check passed on the first attempt
(see the results sections and [Measured resource use](#measured-resource-use-2026-10-18)).

## Pre-run resource plan

Recorded 2026-10-17 under
[PERUN resource accounting](../../../infrastructure/perun/perun-resources.md),
before any submission. It is to be compared with measured use afterward.

| Stage | Tasks | Request per task | Expected elapsed | Expected GPU-h | Ceiling GPU-h | CPU core-h | Peak RAM / VRAM | Storage |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| A: `block-check` | 3 (array 0-2) | gpu_short, 1 H200, 8 CPU, 128 GB, 6 h | 0.8-2.7 h | 2.5-7 | 18 | 20-56 | ≤ 55 GB (estimate) / ≤ 40 GB | PROJECT < 0.1 GB; SCRATCH ≤ 5 GB per task |
| B: `train` | 5 (array 0-4) | gpu_short, 1 H200, 8 CPU, 64 GB, 14 h | 9.5-11 h | 48-55 | 70 | 380-440 | 9 GiB / 34 GiB (measured S7-0) | PROJECT ~12 GB; SCRATCH ≤ 25 GB per task |
| B: `reference-kl` | 1 | gpu_short, 1 H200, 8 CPU, 64 GB, 1 h | 0.1-0.3 h | 0.1-0.3 | 1 | ≤ 2.4 | ~9 GiB / ≤ 30 GiB | PROJECT < 1 MB; SCRATCH ~6 GB |
| Merge, summary | login node | - | minutes | 0 | 0 | - | - | < 10 MB |
| **Total** | **9** | | critical path about 11 h, plus queue | **about 51-62** | **89** | **about 400-500** | | PROJECT ~12 GB new |

**Evidence:**

- SwiGLU-7's S7-0 1B runs used the same recipe and budget. They took
  10.25-10.91 h of workflow time: 9.83-10.48 h of training and 805-846 s of
  final evaluation. Peaks were ≤ 8.9 GiB RAM and ≤ 33.5 GiB VRAM.
- In SwiGLU-8, the 80% runs trained about 12% faster than the 20% runs. Its
  8K KL took 4-5 s per model, so the reference job is mostly model loading.
- For Part A, SwiGLU-8's anchor fits ran at about 4 s per epoch at 393,216
  pairs, 351 s for six fits. Its anchor task took 23 minutes in total, setup
  included.
- Fits scale with the number of pairs. The bounds are the same number of
  updates (lower) and the same number of epochs (upper) as at 393,216.

**Confidence:** medium-high for Part B. Low for Part A, because the epoch
counts at larger sizes and under L2-SP are unknown, and its peak RAM is
arithmetic.

**Submission:** all nine tasks fit under `MaxJobs=12`, and nothing waits on
another stage. Submit the three `sbatch` calls back to back while almost
nothing else is running, because of the submit-time "4 nodes" check in
[perun-status.md](../../../infrastructure/perun/perun-status.md). Before
submitting, check `perunfsusage` against the 1 TB PROJECT allocation. After
each job, add a [perun-log.md](../../../infrastructure/perun/perun-log.md)
entry and keep workflow timing separate from scheduler allocation.

### Measured resource use (2026-10-18)

**Timing source.** Workflow time runs from each `result.json`'s
`created_at_utc` to its `updated_at_utc`. That is Python's run time, including
the final evaluation and stage-out preparation but not the job script's
staging. Scheduler allocation is unknown: `sacct` is disabled, and the
`scontrol` records had expired when they were requested. GPU-hours below are
therefore a workflow-timing proxy, not scheduler allocation.

| Stage | Tasks | Planned per task | Measured per task | GPU-h planned | GPU-h measured (proxy) | Peak RAM / VRAM |
| --- | ---: | --- | --- | ---: | ---: | --- |
| `train` | 5 | 9.5-11 h | 9.37-10.62 h | 48-55 | 50.0 | 5.9-8.9 GiB / 25.1-31.3 GiB |
| `block-check` | 3 | 0.8-2.7 h | 1.26 h (layer 3), 2.69 h (layer 11), 1.48 h (layer 19) | 2.5-7 | 5.4 | not recorded |
| `reference-kl` | 1 | 0.1-0.3 h | 96 s | 0.1-0.3 | 0.03 | not recorded |
| **Total** | **9** | | critical path 10.62 h | **51-62** | **about 55.5** | |

**Training breakdown:**
- 9.03-10.23 h of training, 820-939 s of final evaluation, and 262-544 s of
  checkpointing per run;
- 27,150-30,750 tokens per second;
- the 80% runs trained about 11% faster than the 50% runs (9.03 and 9.08 h
  against 9.79 and 9.94 h), as in SwiGLU-8;
- peak VRAM stayed below SwiGLU-7's 33.5 GiB.

**Against the plan:**
- **Training:** within the expected band. The 14 h limit left at least 3.4 h
  of margin.
- **Block check:** inside the 2.5-7 GPU-h band, at the upper middle. The
  per-task 6 h limit left at least 3.3 h of margin.
- **Reference KL:** finished in 96 s against the 6-18 minutes expected. The
  plan overestimated the model loading.
- **CPU core-hours:** about 8 × 55.5 ≈ 444, within the planned 400-500.
- **Storage:** 11 GB on PROJECT, against about 12 GB planned. The local
  analysis copy (JSON and logs) is 4.1 MB.
- **Failures and retries:** none.

**Queue.** The jobs were submitted at about 17:30 UTC with
`--begin=now+10minutes` and started Python at 18:57 UTC. Both times were read
from the PERUN clock (assumed UTC+2) and the result records. The wait of
about 1.3 h after the begin time was for free GPUs. The `--begin` delay
avoided PERUN's refusal of new submissions while 4 jobs run, which would
otherwise have hit the block-check and reference-KL submissions.

**Next comparable request:** the training request (14 h, 64 GB) can stay. The
block check would fit in 4 h, and the reference KL in 15 minutes.

## Limits

- One twin, at 50% only.
- One run per model; the 8K KL is measured once per final model.
- The 1B answer is about the current fitting recipe; a better Part A recipe is
  not tested at 1B.
- The 5% bar is a judgment call.
- Part A runs one seed per new setting; its noise term borrows SwiGLU-8's
  seed spread.
- The L2-SP strengths come from a one-dimensional model of each combination's
  own fit; they are a reasoned bracket, not a tuned optimum.

## Sources and attribution

- `src-l2-sp-2018` (Li, Grandvalet, and Davoine, ICML 2018; arXiv
  1802.01483v2):
  - Section 3, paragraph "L2-SP", Eq. (2): the penalty;
  - Eq. (3): the term for new parameters, not used here;
  - Section 4.2: SGD, cross-validated α, β = 0.01;
  - Figure 1: the α grid;
  - Section 4.4.2: the optimum as an affine combination along the Hessian's
    eigenvectors.

  All were checked on 2026-10-17. The use on a copied teacher slice and the
  strength rule α = 2ρ·ΔMSE/d² are project choices.
- The quadratic line model behind 1/(1+ρ) is standard mathematics, used as an
  explanation; it needs no citation.
