---
title: SwiGLU-8 - Local-Fitting Attribution and Homogeneous Control Spectrum
status: draft
modified: 2026-10-12
reference_artifacts:
  - data/results/workflows/model/swiglu-7/prepare-001/result.json
  - data/results/workflows/model/swiglu-7/S7-0-target-0.2-run-001/result.json
  - data/results/workflows/model/swiglu-7/S7-0-target-0.3-run-001/result.json
  - data/results/workflows/model/swiglu-7/S7-0-target-0.4-run-001/result.json
  - data/results/workflows/model/swiglu-7/S7-0-target-0.5-run-001/result.json
---

# SwiGLU-8 - Local-Fitting Attribution and Homogeneous Control Spectrum

**Status:** design settled on 2026-10-10; implemented on 2026-10-11, with
Part A expanded the same day into a configuration-selection study (see
[Implementation](#implementation)). Part A ran on PERUN on 2026-10-11 and kept
SwiGLU-7's recipe (see [Part A results](#part-a-results-2026-10-11)). Part B's
preparation and all 16 runs completed on 2026-10-12. The four fitted 20-50%
runs needed a rerun after an over-strict reproduction check stopped them. The
review found run-to-run noise of up to about 3.5% at single checkpoints, a
caveat for Part B's 2% rule (see [Why B2 changed](#implementation)). **Part B
outcome:** local fitting lowers the 8K KL at 100M by 10-24% at every target,
but the gap is still shrinking, so every target is "inconclusive at 100M"
under the decided rule (see [Part B results](#part-b-results-2026-10-12)).
The binding decisions are
recorded in the [shared decision log](../../../agents/decisions.md); this page
explains them. If this page and the log disagree, the log wins until the
conflict is raised with the researcher.

## Questions

1. Does local fitting still improve the compressed model after recovery, or
   does recovery make it unnecessary?
2. Can the local fit itself be improved, including by capturing calibration
   pairs from 8,192-token context instead of 128?
3. What does homogeneous SwiGLU achieve across 20-80% eligible-MLP removal
   at a 100M-token recovery budget? These runs are the homogeneous control
   spectrum for the heterogeneous-operator chapter.

## Why this experiment

**Local fitting is the step that makes the method "replacement".** The
pipeline scores teacher neurons, copies the selected ones into a narrower
SwiGLU (the *slice*), and then locally fits that slice to the teacher MLP's
input/output pairs before model-level recovery. Without the local fit, the
method is structured pruning followed by distillation, which is close to
Minitron. [src-minitron-2024, Section 2.2; Sections 3 and 4.3]

**Its contribution after recovery has never been measured.** SwiGLU-7's
records show that local fitting reduces block-level NMSE by a median of about
30% relative to the slice. Recovery, however, trains the same weights again.
Every SwiGLU experiment recorded the slice's NMSE before fitting, but none
evaluated an unfitted slice in the assembled model, before or after recovery.
Grafting's full self-grafting experiment compares random initialization,
local fitting alone, and local fitting plus fine-tuning, but not fine-tuning
without local fitting. Its "no local fitting" baseline is also random weights,
not a teacher slice. [src-grafting-2025, Section 3.3, Table 2]

**The current fit may not be at its practical ceiling.** SwiGLU-7's 135 local
fits show two patterns:

| Width | Behavior | Diagnosis |
| --- | --- | --- |
| 0.25 | Validation error improves slowly; median best epoch about 22, latest 47 | Optimization-limited, close to a plateau |
| 0.5 and wider | Validation error stops improving after 1-3 epochs while training error keeps falling | Overfitting |

The local-fitting learning rate (constant `1e-3`) was carried over from
random-initialization fitting and never tuned for teacher-initialized fits.
The earlier learning-rate study, SwiGLU-4, tuned *recovery*, not local
fitting. Calibration pairs come from 128-token sequences, while recovery and
use run at 8,192 tokens. If, as is common in language models, the first
token of each sequence carries very large activations, 128-token capture
over-represents those tokens in the squared-error loss. This is a hypothesis
that the 8K recipe tests; it has not been checked for SmolLM2.

## Design overview

| Part | What it does | Decides |
| --- | --- | --- |
| A. Block check | Compares local-fitting settings on three layers and two widths, with three seeds | Which fitting setting the model runs use, and why |
| B. Model runs | Slice-only and fitted models at eight targets, recovered to 100M tokens | Whether local fitting matters after recovery; the homogeneous control spectrum |

Part B's token-0 checkpoint is the before-recovery comparison; there is no
separate stage for it.

## Part A - local-fitting configuration selection

Part A is a configuration-selection study for the local-fitting recipe
(decision log, 2026-10-11). Besides choosing the setting Part B uses, it gives
the thesis's operator chapter the evidence for *why* each local-fitting
setting was chosen. The notebook's section 2, "Learning configuration", reports
it, including the table "Why each setting was kept".

**Scope:** layers 3, 11, and 19 (early, tolerant middle, sensitive late) at
widths 0.25 and 0.5. Layers 1, 7, and 22 are excluded because their very large
activations make them atypical.

**Stage 1 settings**, each fitted at every layer-width combination:

| Settings | Training context | Schedule | Learning rate | Question |
| --- | --- | --- | --- | --- |
| `c1e-3-128`, `c3e-4-128`, `c1e-4-128` | 128 | Constant, at most 64 epochs, patience 3 | `1e-3`, `3e-4`, `1e-4` | Is SwiGLU-7's rate (`c1e-3-128`, the reference) too high? |
| `cos1e-3-128`, `cos3e-4-128`, `cos1e-4-128` | 128 | Cosine over a fixed 32 epochs, best epoch kept | same | Does a decaying rate help the fit settle? |
| The six settings above with `-8k` | 8,192 | as above | same | Does 8K training context help, and at which rate? |
| `c1e-3-128-b256` | 128 | Constant | `1e-3`, batch 256 | Does the batch-2,048 choice hold for teacher-slice fits? |
| `ols-128`, `ols-8k` | 128, 8,192 | None | None | Does the exact least-squares down projection beat gradient training? |

Every gradient setting runs with seeds 21, 22, and 23; least squares has no
randomness and runs once. That is 41 setting-seed runs and 246 fits. Seeds
change only the order of training batches. Shared settings follow SwiGLU-7:
AdamW, zero weight decay, 393,216 calibration pairs, importance-ranked
teacher-subset initialization, and an MSE objective. Batch is 2,048 except in
the batch check.

Cosine runs use a fixed 32-epoch decay because SwiGLU-7's fits stop early: a
64-epoch cosine would barely decay before early stopping. 32 covers SwiGLU-7's
longest fit at these combinations (best epoch 32). Their patience equals the
epoch budget, so they never stop early.

**Earlier evidence**, reported alongside and not retested:

- **Batch 2,048:** in the
  [operator-distillation study](../../block/operator/operator-distillation.md#2-operator-batch-size-screen),
  batch 256 reaches the same fit when trained to early stopping.
- **393,216 calibration pairs:** chosen in SwiGLU-3, where
  98,304 → 196,608 → 393,216 pairs each improved the allocation-selection KL.
- **Importance-ranked teacher slice:** the operator-distillation
  initialization study.

**Stage 2** (deferred on 2026-10-11; see the results below), only if stage 1
leaves the wider widths overfitting:

- a penalty keeping weights near the copied teacher slice, at a weak and a
  strong setting (the L2-SP method; Li et al., ICML 2018, to be registered
  before the thesis cites it); and
- 786,432 calibration pairs, twice the SwiGLU-7 count.

**Evaluation:** single-block KL, meaning the teacher KL when only that block
is replaced, at 128 and at 8,192 tokens. NMSE is reported alongside, but it
is not the selection criterion, because a few large-magnitude tokens dominate
it.

**Selection data is separate from final-evaluation data** (decision log,
2026-10-11).

| Data | Used for | Final evaluation? |
| --- | --- | --- |
| 128-token calibration pairs (393,216) and operator validation (6,144) | Training and early stopping at 128 | No |
| 128-token allocation selection (3,072 tokens) | Single-block KL at 128; width allocation | No |
| 8K calibration (48 × 8,192) and validation (1 × 8,192) streams | Training and early stopping at 8K | No |
| 8K selection stream (24 × 8,192) | Single-block KL at 8K, the deciding score | No |
| Quantization-1's 8K KL stream (24 × 8,192) | Final 8K KL at 100M | Yes |

All 8K streams use Quantization-1's protocol and packing. They come from
consecutive, non-overlapping C4 shard-0 documents, and they skip every
document that supplied a 128-token window. Part A never scores on the KL
stream, so the final 8K comparison is not used to choose the recipe.

**Adoption rule:** each setting is summarized by its seed mean and sample
standard deviation per combination. A setting is adopted only if all of the
following hold:

- averaged over the six combinations, its 8K single-block KL is more than 5%
  below the current setting's seed mean;
- that reduction is larger than the mean seed noise, defined per combination
  as the two settings' standard deviations added together and divided by the
  current mean;
- no combination is more than 5% worse at 128 or at 8K.

The largest qualifying 8K reduction wins; otherwise Part B keeps SwiGLU-7's
recipe. Part B fits with seed 21.

### Part A results (2026-10-11)

The block check ran as PERUN array 105196: 15 tasks, all completed. The
login-node merge verified that every task used the same configuration, code,
software, data streams, and neuron rankings. It wrote
`perun-results/swiglu-8/block-check-001/result.json` (sha256
`a268f1b0537579580eac5fe908ac6fccb06cf6912688a57a4db972fa28c32724`). The
anchor task's checks passed: P5 (the teacher's 8K KL against itself is zero),
A1 (slices match SwiGLU-7), and A2-A3 (`c1e-3-128` reproduces SwiGLU-7's fits
and single-block KL). The values below were read on PERUN from that file; the
notebook analysis follows once the results are copied home.

**Outcome: no setting qualified, so Part B uses `c1e-3-128`** (decision log,
2026-10-11).

Reduction of the mean single-block KL against `c1e-3-128` (positive is
better), averaged over the six layer-width combinations:

| Setting | At 8K | At 128 | Mean seed noise | Worst combination | Why not adopted |
| --- | ---: | ---: | ---: | ---: | --- |
| `c3e-4-128` | 1.18% | 0.77% | 0.32% | 0.39% | Below 5% |
| `c1e-4-128` | 1.62% | 0.76% | 0.25% | -0.77% | Below 5% |
| `cos1e-3-128` | -1.02% | 0.40% | 0.29% | -7.37% | Below 5%; below noise; a combination more than 5% worse |
| `cos3e-4-128` | 1.96% | 0.70% | 0.26% | -1.65% | Below 5% |
| `cos1e-4-128` | 2.01% | -0.57% | 0.25% | -7.13% | Below 5%; a combination more than 5% worse |
| `c1e-3-8k` | 3.43% | -6.93% | 0.42% | -9.33% | Below 5%; a combination more than 5% worse |
| `c3e-4-8k` | 4.17% | -6.18% | 0.28% | -8.32% | Below 5%; a combination more than 5% worse |
| `c1e-4-8k` | 4.39% | -5.95% | 0.27% | -8.51% | Below 5%; a combination more than 5% worse |
| `cos1e-3-8k` | 3.61% | -6.66% | 0.30% | -9.85% | Below 5%; a combination more than 5% worse |
| `cos3e-4-8k` | 4.36% | -5.94% | 0.29% | -8.26% | Below 5%; a combination more than 5% worse |
| `cos1e-4-8k` | 4.20% | -6.22% | 0.26% | -10.50% | Below 5%; a combination more than 5% worse |
| `c1e-3-128-b256` | -3.82% | -4.23% | 0.78% | -6.35% | Below 5%; below noise; a combination more than 5% worse |
| `ols-128` | -9.25% | -12.74% | 0.21% | -26.09% | Below 5%; below noise; a combination more than 5% worse |
| `ols-8k` | -6.94% | -18.79% | 0.21% | -27.77% | Below 5%; below noise; a combination more than 5% worse |

`c1e-3-128` seed means range from 0.0094 (layer 11, width 0.5, 128 tokens) to
0.0554 (layer 19, width 0.25, 8K), with seed standard deviations of at most
0.00027.

What the numbers show:

- **Seeds barely matter.** The mean seed noise is 0.2-0.8% of the KL, so
  differences of 1-2% are real rather than chance.
- **The recipe is on a plateau.** No setting moves the mean 8K KL by more than
  4.4%.
- **Lower learning rates help slightly.** `3e-4` and `1e-4` lower the 8K KL by
  1.2-2.0% and the 128-token KL by up to 0.8%. Cosine decay adds nothing
  beyond its learning rate, and `cos1e-3-128` is slightly worse.
- **The fit specializes to its training context.** Fitting on 8K pairs lowers
  the 8K KL by 3.4-4.4% but raises the 128-token KL by 5.9-6.9%; single
  combinations are up to 10.5% worse. This is a trade-off, not an improvement.
  It is consistent with the hypothesis that 128-token capture over-weights
  early-position tokens, but Part A does not test that mechanism.
- **Batch 2,048 stays.** Batch 256 is 3.8% worse at 8K and 4.2% worse at 128.
  In the operator-distillation study, both reached the same NMSE when trained
  to early stopping; with teacher-slice initialization and KL scoring, batch
  256 is worse.
- **Gradient training beats least squares.** The exact least-squares down
  projection is 6.9-9.3% worse at 8K and 12.7-18.8% worse at 128, with single
  combinations up to 28% worse. Minimizing the training MSE exactly transfers
  worse to KL than gradient training with early stopping.

**Stage-2 indicator: overfitting remains at width 0.5 in all three layers.**
All nine `c1e-3-128` fits at width 0.5 reach their best validation epoch at
epoch 3 (one at epoch 4) and stop three epochs later. Meanwhile their
training MSE falls a further 1.4-2.3% (for example, layer 3 from 0.06786 to
0.06636). The overfitting is mild, and early stopping already keeps the best
epoch. **Stage 2 is deferred** until Part B shows whether local fitting
matters after recovery (decision log, 2026-10-11). SwiGLU-3's calibration
curve, where each doubling of calibration pairs lowered allocation KL by about
3-4%, suggests stage 2 would also stay below the 5% bar. That is an
indication from a different metric, not a test.

**Limits.** Three layers and two widths. Single-block KL, not full-model
quality. Seeds vary only the batch order. The 5% threshold is a judgment call
fixed before the run.

## Part B - model runs at 100M tokens

**Arms:** slice-only (teacher slice, no local fitting) and fitted (the recipe
selected by Part A). Within a target, both arms use the same neurons and the
same widths; only local fitting differs.

**Targets:** 20, 30, 40, 50, 60, 70, 75, and 80% eligible-MLP removal. The 75%
target matches Quantization-1's nominal INT4 storage reduction; 70% and 80%
bracket it.

**Widths and allocation:**

- Widths 0.125 and 0.0625 are added to SwiGLU-7's options (0.25 to 1.0 in
  steps of 0.125). SwiGLU-only allocation cannot exceed 75% removal without
  them.
- At 20-50%, the fitted arm keeps SwiGLU-7's exact widths, so the comparison
  with SwiGLU-7 isolates fitting. The implementation checks whether
  re-solving the allocation with the extended widths would change these
  targets and reports the result.
- At 60-80%, widths come from the discrete allocator over width curves
  extended to the new widths. If Part A adopts a new recipe, all width curves
  are refitted with it so that allocation uses one recipe; otherwise
  SwiGLU-7's curves are extended with the two new widths.

**Recovery:** identical to SwiGLU-7's replacement-only recipe (S7-0). Only
replacement SwiGLUs train; layers kept at full width, attention, norms, and
embeddings stay frozen. AdamW, constant `3e-5`, zero weight decay, temperature
1 KL with no cross-entropy, 8,192-token sequences with an effective batch of
8,192 tokens, seed 21, and the same ordered C4 recovery stream starting at
shard 1. Because the learning rate is constant, a 100M-token run is
identical to the first 100M tokens of a 1B-token run.

**Checkpoints:** 0, 10, 25, 50, 75, and 100M tokens. Monitoring KL is measured
at all six and WikiText prefix perplexity at all but 75M, matching SwiGLU-7.

**Final evaluation at 100M:** the full SwiGLU-7 final evaluation (full
WikiText-2 validation and test perplexity at contexts 128, 2,048, and 8,192,
plus the five zero-shot tasks) and Quantization-1's 8K KL. The quantization
baseline and the heterogeneous comparison use these metrics.

**Interpretation, fixed in advance:**

| Slice-only versus fitted at 100M | Conclusion |
| --- | --- |
| Gap shrinks to within about 2% | Local fitting does not matter after recovery |
| Gap stays clearly larger and is not shrinking | Local fitting matters |
| Anything else | Inconclusive at 100M |

The 2% threshold is a judgment call and is reported as such. There is no
repeat run: both arms use seed 21 and the identical ordered recovery stream,
so the comparison is paired. The 2026-10-12 rerun showed that runs are not
bitwise repeatable; see [Part B results](#part-b-results-2026-10-12).

**Promotion:** only clear 100M results that serve as thesis headline
comparisons are promoted to 1B-token recovery.

### Part B results (2026-10-12)

All values come from the local copy `data/results/workflows/model/swiglu-8/`:
preparation `prepare-001`, and the 16 runs `{fitted,slice}-target-<t>-run-001`
from arrays 105292 and 106606. The notebook has not yet been run on them. The
dense reference is the pinned SmolLM2-1.7B: test perplexity 11.982 / 7.267 /
6.933 at contexts 128 / 2,048 / 8,192, and task macro accuracy 67.14%.

**Checks.**

- Preparation passed P1-P5 with zero difference from SwiGLU-7. P1 covers 135
  slice NMSE values, P2 92 fitted values, and P3 12 dense-evaluation values.
  The 100M stream equals SwiGLU-7's first 100M tokens, and the teacher's 8K KL
  against itself is 0.
- The fitted 20-50% runs matched SwiGLU-7 exactly at token 0 (B1). Their B2
  review is in [Why B2 changed](#implementation).

**Allocation.** At 20-50%, the fitted and slice-only arms reuse SwiGLU-7's
widths. Re-solving with the extended curves would change none of 20-40%. At
50% it would change only layers 7 and 11, for a predicted KL 0.1% lower, so
SwiGLU-7's widths stand. At 60-80%, the allocator used the curves extended to
512 and 1,024. Layers 1 and 22 stay wide (22 is kept dense up to 70%). The
narrowest widths go to the middle layers 5-15:

| Target | Widths of layers 1-22 |
| ---: | --- |
| 60% | 4096 4096 3072 3072 2048 2048 2048 3072 3072 2048 1024 1024 1024 1024 2048 3072 4096 6144 5120 5120 5529 8192 |
| 70% | 4096 3072 2048 2048 2048 1024 1024 2048 2048 1024 1024 512 1024 1024 1024 2048 3072 3379 4096 4096 4096 8192 |
| 75% | 4096 2048 2048 2048 2048 512 1024 1024 1024 1024 512 512 512 1024 1024 2048 2048 3072 4096 3072 4096 6144 |
| 80% | 4096 2048 2048 2048 512 512 512 1024 512 512 512 512 512 1024 1024 1024 2048 3072 3072 3072 2252 4096 |

**Main result: local fitting still helps after 100M tokens, at every target,
but its advantage is still shrinking.** Here g = (slice - fitted) / fitted on
the 8K KL at 100M, the deciding metric. The monitoring gap is the same ratio
on the recovery-validation KL.

| Target | Whole-model removal | 8K KL fitted | 8K KL slice-only | **g** | Monitoring gap at 0 / 50M / 100M | Change 50M-100M | Rule verdict |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 20% | 12.9% | 0.0787 | 0.0887 | **12.6%** | 111% / 22.1% / 16.5% | -5.6 pp | Inconclusive at 100M |
| 30% | 19.4% | 0.1161 | 0.1288 | **10.9%** | 110% / 18.9% / 8.3%* | -10.6 pp* | Inconclusive at 100M |
| 40% | 25.9% | 0.1701 | 0.1883 | **10.7%** | 143% / 14.9% / 11.6% | -3.3 pp | Inconclusive at 100M |
| 50% | 32.4% | 0.2288 | 0.2527 | **10.4%** | 201% / 14.9% / 10.4% | -4.5 pp | Inconclusive at 100M |
| 60% | 38.8% | 0.3040 | 0.3357 | **10.4%** | 301% / 16.3% / 11.7% | -4.6 pp | Inconclusive at 100M |
| 70% | 45.3% | 0.4094 | 0.4549 | **11.1%** | 364% / 17.9% / 11.7% | -6.2 pp | Inconclusive at 100M |
| 75% | 48.5% | 0.4707 | 0.5815 | **23.5%** | 425% / 45.6% / 24.1% | -21.5 pp | Inconclusive at 100M |
| 80% | 51.8% | 0.5433 | 0.6432 | **18.4%** | 427% / 35.2% / 18.2% | -16.9 pp | Inconclusive at 100M |

\* The fitted 30% run ends on the noise bump described in the B2 review, so its
100M monitoring gap, and probably its g, are a few points too low.

What the pre-registered rule says:

- g exceeds the 2% "does not matter" bound at every target, by 5-10 times.
  It also exceeds the run-to-run noise measured in the B2 review (up to about
  3.5% at a single checkpoint).
- The monitoring gap still fell by 3-22 percentage points between 50M and
  100M, so it is not stable. **Every target is therefore "inconclusive at
  100M".** Local fitting gives a clear advantage at this budget, but these runs
  cannot show whether the advantage would survive longer recovery.

**How the gap closes.** Recovery removes most of the slice-only arm's initial
handicap within the first 10M tokens. At token 0 the slice-only models are
2-5 times worse: 110-427% higher KL, and the ratio rises with removal. Later
the gap shrinks steadily:

| Target | 10M | 25M | 50M | 75M | 100M |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20% | 33.2% | 24.1% | 22.1% | 17.3% | 16.5% |
| 30% | 26.4% | 20.1% | 18.9% | 17.0% | 8.3%* |
| 40% | 25.3% | 19.2% | 14.9% | 12.5% | 11.6% |
| 50% | 26.2% | 19.6% | 14.9% | 12.4% | 10.4% |
| 60% | 31.9% | 22.1% | 16.3% | 13.9% | 11.7% |
| 70% | 38.7% | 26.6% | 17.9% | 14.0% | 11.7% |
| 75% | 90.2% | 65.6% | 45.6% | 32.7% | 24.1% |
| 80% | 70.7% | 52.5% | 35.2% | 24.4% | 18.2% |

A power law fitted to 25-100M and extended to 1B suggests a gap of about
3-8% at 1B. That is an extrapolation ten times beyond the data, not a result.
It suggests the advantage shrinks to a few percent but may stay above 2%. Only
a 1B measurement can settle it.

**Final evaluation.** The fitted arm is better on perplexity at every target.
On tasks the two arms are mostly within evaluation uncertainty.

| Target | Test PPL 8,192, fitted / slice-only | Slice-only higher by | Test PPL 128, fitted / slice-only | Task macro, fitted / slice-only | Fitted macro minus dense |
| ---: | --- | ---: | --- | --- | ---: |
| 20% | 7.726 / 7.793 | 0.9% | 13.577 / 13.719 | 62.80% / 61.86% | -4.34 pp |
| 30% | 8.170 / 8.265 | 1.2% | 14.767 / 14.960 | 60.70% / 60.00% | -6.44 pp |
| 40% | 8.843 / 8.994 | 1.7% | 15.944 / 16.328 | 58.13% / 58.14% | -9.01 pp |
| 50% | 9.645 / 9.873 | 2.4% | 17.895 / 18.259 | 55.65% / 55.47% | -11.49 pp |
| 60% | 10.839 / 11.085 | 2.3% | 20.828 / 20.966 | 53.60% / 52.10% | -13.55 pp |
| 70% | 12.474 / 12.732 | 2.1% | 24.666 / 24.693 | 50.76% / 49.54% | -16.38 pp |
| 75% | 13.654 / 14.781 | 8.3% | 27.112 / 29.614 | 48.70% / 47.15% | -18.44 pp |
| 80% | 14.970 / 15.952 | 6.6% | 30.061 / 32.402 | 46.99% / 45.61% | -20.15 pp |

Per task, the arms differ by -1.8 to +3.3 points. Most differences are smaller
than the 95% half-widths of each model's paired comparison with dense
(0.5-3.4 points). HellaSwag is the exception: the fitted arm is ahead at every
target, by 0.7-3.3 points, more than its half-width of 0.5-0.9. No paired
between-arm interval was computed; that needs the raw task records, which
remain on PERUN.

**The homogeneous control spectrum (fitted arm, 100M tokens).** Removing
13-52% of all parameters raises 8K test perplexity from 6.93 (dense) to
7.73-14.97 and lowers task macro accuracy from 67.1% to 62.8-47.0%. The 8K KL
grows smoothly with removal, from 0.079 to 0.543. BF16 bundles shrink from
3.19 GiB to 2.78 GiB at 20% and 1.54 GiB at 80%. These are the 100M points
the heterogeneous chapter is compared against.

**Context from SwiGLU-7 (fitted arm only).** For 20-50%, SwiGLU-7 continued
the same fitted models to 1B. From 100M to 1B its monitoring KL fell a further
20-29%, its 8K test perplexity dropped to 7.56 / 7.87 / 8.31 / 8.86, and task
macro rose by 1.2-2.8 points. Whether the slice-only arm improves faster over
those 900M tokens is exactly what 1B slice-only runs would measure. SwiGLU-7's
1B bundles remain on PERUN, so the fitted side of such a comparison only needs
its 8K KL evaluated.

**Observations without an explanation yet.**

- The gap at 75% (23.5%) is larger than at 80% (18.4%). At 75% the gap also
  closes more slowly: 45.6% at 50M, against 35.2% at 80%. The two allocations
  differ in structure. At 75%, layer 22 stays at 6144 and layers 6 and 11-13
  drop to 512. At 80%, more middle layers drop to 512 but layer 22 is at 4096.
  So the gap need not be monotone in removal. No cause has been tested.
- The 8K KL and the monitoring KL at 100M differ by -0.4% to +6.5% in both
  arms, and they rank the two arms the same way at every target.

**Limits.**

- One seed per arm. Runs are not bitwise repeatable: 0.01-0.41% at 10M, and
  single checkpoints up to about 3.5%.
- The 8K KL is measured once per final model.
- Task differences between arms have no paired interval.
- Everything stops at 100M tokens, and the 1B estimate above is an
  extrapolation.

### Possible next work: the fitted/slice-only divergence at 1B

**Status:** proposed by the researcher on 2026-10-12. Not decided,
implemented, or scheduled.

**Question.** Does local fitting's advantage survive to 1B recovery tokens, or
does it fall to the noise level? How does the answer depend on sparsity? Part
B left every target "inconclusive at 100M", and only a longer run can resolve
that.

**Proposed design.**

- **Targets:** 20%, 50%, and 80% eligible-MLP removal (low, mid, and high
  sparsity), both arms.
- **Recipe and starts:** the S7-0 recipe to exactly 1B tokens, from the
  existing SwiGLU-8 token-0 starts (`prepare-001/prepared/starts/` on PERUN),
  so no new preparation is needed. Each run restarts from token 0, because the
  100M optimizer checkpoints were deleted.
- **Monitoring:** every 25M tokens, matching SwiGLU-7, so the gap can be
  plotted as a full divergence curve from 0 to 1B.
- **Final evaluation at 1B:** the full SwiGLU-8 evaluation, including the 8K
  KL, the deciding metric.
- **The fitted 20% and 50% arms already exist.** SwiGLU-7's S7-0 runs are the
  same models with the same recipe to 1B (B1 and P2 matched them exactly at
  token 0). They have monitoring every 25M, and their 1B bundles are on PERUN.
  The minimal plan therefore needs only those two bundles' 8K KL evaluated,
  plus four new 1B runs: slice-only 20%, 50%, and 80%, and fitted 80%.
  Rerunning the fitted 20% and 50% arms too would give a same-pipeline pair at
  extra cost; the measured run-to-run noise means both runs would still be
  single samples.

**Cost estimate.** Each 1B run takes about 11-12 h on one H200 (S7-0 measured
9.8-10.5 h of training, plus monitoring and evaluation). With all runs in
parallel, that is about half a day of elapsed time. The minimal plan needs
about 44-48 GPU-h, and the plan with fitted reruns about 66-72 GPU-h. The
estimate is based on SwiGLU-7's measured 1B timings; confidence is medium.

**Implementation needed.**

- The SwiGLU-8 runner fixes recovery at 100M through its configuration. A 1B
  variant needs a recovery target of 1B, monitoring every 25M, separate output
  names (for example `-1b-run-001`), and a small job to evaluate the 8K KL on
  existing bundles.
- If the fitted 20% and 50% arms are rerun, their reproduction check against
  SwiGLU-7 would cover the full 1B history and use the 1% review rule.

**Interpretation to fix before running.** Keep g on the 8K KL, now at 1B. The
2% "does not matter" bound should be read against the measured noise: about
0.4% run to run at 10M, and up to about 3.5% at a single checkpoint. Fix the
rule for that before any run.

## Implementation requirements from the design

1. **Fitted arm at 20-50% (decided 2026-10-10).** SwiGLU-7 deleted its
   intermediate checkpoints after the 1B runs, so it has no 100M weights on
   which to run the full final evaluation. The fitted arm is therefore run to
   100M at all eight targets. At 20-50% these runs must reproduce SwiGLU-7's
   first 100M of monitoring values when the current recipe is kept, which
   serves as a reproduction check. If Part A adopts a new recipe, they are new
   runs anyway.
2. **8K data overlap.** Quantization-1's 8K KL uses C4 train shard 0, which is
   also SwiGLU-7's calibration source. The 8K calibration sequences in Part A
   must be drawn disjointly from the 8K KL evaluation sequences. Since
   2026-10-11, Part A's 8K score also uses its own selection stream, so no
   choice is made on final-evaluation data.
3. **Shared neuron selection (decided 2026-10-10).** SwiGLU-7's artifacts
   record widths but not which neurons were selected. SwiGLU-8 therefore
   selects neurons once per target with SwiGLU-7's importance score, saves the
   selection in its own artifacts, and builds both arms from it. Equality with
   SwiGLU-7's selection is checked indirectly by the reproduction check below.
4. **Shared 8K KL code (decided 2026-10-10).** Quantization-1's 8K KL
   evaluation (`teacher-kl-t1-all-valid-positions-c4-shard0-24x8192-v1`) moves
   from `workflows/runs/model/baseline/quantization.py` into shared code in
   `src/`, used by both experiments, with identical behavior. The move happens
   before either experiment runs.

## Acceptance checks

These are part of the real runs, not separate smoke runs. Tolerances are
stated in the implementation plan.

1. **Part A:** the current recipe at layers 3, 11, and 19 reproduces the
   local-fit error SwiGLU-7 recorded for the same layer and width.
2. **Part B:** while the current recipe is kept, the fitted runs at 20-50%
   reproduce SwiGLU-7's monitoring values over the first 100M tokens.

If either check fails, work stops for investigation before continuing.

## Implementation boundaries

- All parts run on PERUN.
- Run SwiGLU-8 before the planned refactor changes `src/`, so its source
  fingerprints and SwiGLU-7 reproduction are not affected by unrelated
  changes.
- Compute uses a headless runner in `workflows/runs/model/swiglu/` with a
  configuration in `workflows/configs/model/swiglu/` and a PERUN job script.
  Analysis, including Part A plots, uses a load-only
  `notebooks/model/swiglu/swiglu-8.ipynb`.
- No smoke or reduced-budget runs.

## Resources

Rough estimates from SwiGLU-7's workflow timings; not a resource plan and
not billing:

| Item | Estimated H200-hours |
| --- | ---: |
| Part A, stage 1 (246 fits plus evaluation, as 15 parallel tasks that each repeat the setup) | 16-31 |
| Part A, stage 2, if needed | 1-2 |
| Width curves extended to 0.125 and 0.0625 (full refit if a new recipe is adopted) | 2-6 |
| Part B, 16 runs to 100M (two arms at eight targets) at about 1.2-1.5 hours each, plus about 15 minutes of final evaluation per run | 23-28 |
| **Total** | **about 42-67** |

### Pre-run resource plan

Recorded 2026-10-11 under
[PERUN resource accounting](../../../infrastructure/perun/perun-resources.md),
before any submission, and to be compared with measured use afterward. The
evidence is SwiGLU-7's measured workflow timing: S7-0 trained 100M tokens in
3,541-3,775 s, final evaluation took 805-846 s, and peak usage was 8.9 GiB RAM
and 33.5 GiB VRAM. SwiGLU-7's six fits at the Part A combinations took 380 s
together, about 4.4 s per epoch. Confidence is medium. The main uncertainties
are how many epochs the 512- and 1024-wide fits and the low-learning-rate
settings need, and the time to score each fit (single-block KL at 128 and 8K),
which SwiGLU-7 did not record; it is estimated at 1-2 minutes per fit.
The block check runs as 15 parallel tasks, one per setting. Each task repeats
about 20-40 minutes of setup (model loading, data, pair capture) and keeps its
fit states on SCRATCH, about 1.5 GB. The slowest tasks are the low-learning-rate
settings, whose fits can run all 64 epochs.

| Stage | Tasks | Request per task | Expected elapsed | Expected GPU-h | Ceiling GPU-h | CPU core-h | Peak RAM / VRAM | Storage |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| A: `block-check` | 15 (one array, tasks 0-14) | gpu_short, 1 H200, 8 CPU, 128 GB, 6 h | 1-3 h each; about 2-3 h in total | 16-31 | 90 | 128-248 | ≤ 50 GB / ≤ 40 GB per task | PROJECT < 0.1 GB; SCRATCH ~2 GB per task |
| A2: stage 2 (only if triggered) | 1 | same, 6 h | 1-2 h | 1-2 | 6 | 8-16 | same | same |
| B0: `prepare` | 1 | gpu_short, 1 H200, 8 CPU, 128 GB, 24 h | 3-6 h (recipe unchanged); 4-8 h (refit) | 3-8 | 24 | 24-64 | ≤ 60 GB / ≤ 40 GB | PROJECT ~25 GB (16 FP32 starts and 0.5 GB data); SCRATCH ~40 GB |
| B1: `train` | 16 (one array, tasks 0-15) | gpu_short, 1 H200, 8 CPU, 128 GB, 6 h | 1.2-1.5 h each | 20-24 | 96 | 160-190 | 8.9 GiB / 33.5 GiB (S7-0) | PROJECT ~38 GB (bundles and benchmarks); SCRATCH ≤ 20 GB per task |
| **Total** | 32-33 | | Compute critical path about 7-14 h, plus queue and the review gate | **about 39-63** | 210-216 | 312-502 | | PROJECT ~63 GB new |

Before each submission, check `perunfsusage` against the 1 TB PROJECT
allocation and the totals in
[perun-status.md](../../../infrastructure/perun/perun-status.md). After each
job, add a [perun-log.md](../../../infrastructure/perun/perun-log.md) entry
with `scontrol` facts while they are available, update the status totals, and
keep workflow timing separate from scheduler allocation.

### Measured resource use (2026-10-12)

Workflow timing comes from each artifact's `created_at_utc` to
`updated_at_utc`, on one H200 per task. It is a proxy, not scheduler
allocation; allocation GPU-hours were not recorded, since `sacct` is
disabled.

| Stage | Plan | Measured |
| --- | --- | --- |
| A: block check, 15 tasks | 16-31 GPU-h; about 2-3 h in total | **7.9 GPU-h**. The longest task was 1.04 h (`c1e-4-128`, task 2). About 1.1 h in total, with at most 12 tasks at once |
| B0: prepare | 3-8 GPU-h | **3.4 GPU-h** |
| B1: 16 runs | 20-24 GPU-h; 1.2-1.5 h each | **20.2 GPU-h**. Each run took 1.18-1.33 h: 0.91-1.05 h of training plus 0.23-0.24 h of final evaluation |
| First attempts of fitted 20-50%, stopped by B2 | - | **0.7 GPU-h** |
| **Total** | **about 39-63** | **about 32.1 GPU-h** |

Peak usage per training run was 8.8 GiB RAM and 25.1-33.5 GiB VRAM, matching
S7-0. Elapsed time from the block-check submission to the last rerun's end
was about one day. It included two review gates, the overnight chain, and the
B2 rerun.

## Implementation

Implemented on 2026-10-11 without any run. Runner:
[`swiglu_8.py`](../../../../workflows/runs/model/swiglu/swiglu_8.py).
Configuration:
[`swiglu-8.json`](../../../../workflows/configs/model/swiglu/swiglu-8.json).
PERUN job: [`swiglu_8.sbatch`](../../../../workflows/jobs/perun/swiglu_8.sbatch),
described in the [PERUN job README](../../../../workflows/jobs/perun/README.md#swiglu-8-workflow).
The copy-paste commands for every stage are in the
[PERUN runbook](../../../infrastructure/perun/perun-runbook.md#3-swiglu-8).
Analysis: the load-only
[`swiglu-8.ipynb`](../../../../notebooks/model/swiglu/swiglu-8.ipynb), which
applies the decided Part B interpretation in code. It reads the complete local
copy under `data/results/workflows/model/swiglu-8/`, including the stopped
first attempts and SwiGLU-7's S7-0 runs. Its sections are design, learning
configuration, local fit versus no local fit, context length, the 20-80%
control spectrum, and conclusions with the 1B outlook, followed by an appendix
on acceptance, data, and compute.
The configuration validates every block it shares with SwiGLU-7 and
Quantization-1 against their versioned configurations. It also pins the
SwiGLU-7 source artifacts by sha256.

| Command | Does | Output |
| --- | --- | --- |
| `block-check --task-index i` | Part A as 15 array tasks, one per setting (three seeds for each gradient setting): 246 fits in total, single-block KL at 128 and on the 8K selection stream, NMSE on both validation sets. Task 0 (SwiGLU-7's setting) also runs the unfitted slices and P5, A1, and A2-A3. Resumes per completed fit | `block-check-001-task-NN/` |
| `merge-block-check` | Login node, no GPU. Checks that every task used identical configuration, code, software, data streams, and neuron rankings; joins the fits; computes the multi-seed adoption decision and the stage-2 indicator. Safe to re-run | `block-check-001/` |
| `prepare` | Part B: neuron rankings, curve extension or refit, allocation, start fits with the adopted setting at seed 21, and the 16 token-zero starts; it refuses unless `part_b` names the block check's adopted setting and its sha256 | `prepare-001/` |
| `train --arm --target` | S7-0 recovery to 100M, monitoring at 0/10/25/50/75/100M, the full SwiGLU-7 final evaluation, and the 8K KL of the reloaded BF16 bundle | `{fitted,slice}-target-{t}-run-001/` |

**Shared code.**
- The 8K KL lives in `src/mlp_replacement/evaluation/teacher_kl.py`, shared with
  Quantization-1.
- `fit_operator_fp32_detailed` accepts a cosine schedule, decayed once per
  epoch over the fit's epoch budget (32 epochs in Part A).
- `refit_down_projection_least_squares` solves the FP64 normal equations
  without regularization. The runner records a least-squares fit like a
  gradient fit (saved state and metrics), so Part B can use it if it is
  adopted.
- `sample_partitioned_windows` can report each window's source record.
- SwiGLU-7's `persist` and `evaluate_final_model` take the workflow name from
  the artifact, so SwiGLU-7's own outputs are unchanged.

**Data.**
- The 8K calibration, validation, and selection streams follow the last
  8K-KL document in that order and skip every record that supplied a
  128-token window. A hard check stops the job if any two share a document.
  A byte check confirms that the indexed shard reproduces the KL stream.
- The block check also reports how many 128-token windows came from 8K-KL
  documents. That overlap already existed in SwiGLU-7.
- The selection never depends on the recipe; the 8K recipe reuses the
  128-token ranking.

**Acceptance checks.** These are the decided tolerances (relative difference
r). A failed check stops the stage, which records the values first. The
exception is B2: since 2026-10-12 it records every comparison and its exact
difference, marks it against a 1% review limit, and never stops the run
(decision log).

| ID | Check | Reference | Tolerance |
| --- | --- | --- | --- |
| A1 | Slice NMSE before fitting at the six combinations | SwiGLU-7 `initial_local_nmse` | r ≤ 1e-4 |
| A2 | Current-recipe fitted NMSE and MSE | SwiGLU-7 `local_nmse`, `local_mse` | r ≤ 1e-3 |
| A3 | Current-recipe single-block KL at 128 tokens | SwiGLU-7 curve `raw_teacher_kl` | r ≤ 1e-3 |
| B1 | Fitted arm at 20-50%, token-0 monitoring KL and PPL | S7-0 history | r ≤ 1e-4 |
| B2 | Same runs, KL at 10-100M and PPL at 10/25/50/100M | S7-0 history | Recorded; reviewed against r ≤ 1e-2 (was a 1e-3 stop until 2026-10-12) |
| P1 | Slice NMSE at all 135 SwiGLU-7 fit points | SwiGLU-7 `initial_local_nmse` | r ≤ 1e-4 |
| P2 | Fitted start operators at 20-50% (recipe unchanged) | SwiGLU-7 `local_nmse`, `local_mse` | r ≤ 1e-3 |
| P3 | Dense teacher PPL and task accuracy | SwiGLU-7 dense evaluation | PPL r ≤ 1e-4; accuracy identical |
| P4 | 100M recovery stream | SwiGLU-7's first 100M tokens | identical sha256 |
| P5 | Teacher-against-itself 8K KL; KL stream equal in both stages | 0; block-check stream | ≤ 1e-6 absolute; identical sha256 |

A1-A3 and B1-B2 are required by the decision log. P1-P5 are cheap additions
that catch a wrong selection, wrong data, or a changed environment before the
expensive stages run. B1, B2, and P2 apply only while the current recipe is
kept.

**Why B2 changed (2026-10-12).** The first Part B attempt (array 105292)
completed 12 of 16 runs. The four fitted 20-50% runs matched SwiGLU-7 exactly
at token 0. Afterwards they drifted slightly, and the 1e-3 B2 stop ended them:

| Run | Stopped at | Recovery KL, relative difference | WikiText PPL, relative difference |
| --- | --- | ---: | ---: |
| Fitted 20% | 10M | 1.19e-3 (stop) | 2.56e-4 |
| Fitted 30% | 10M | 2.05e-3 (stop) | 4.97e-4 |
| Fitted 40% | 10M | 2.17e-4 | 1.02e-3 (stop) |
| Fitted 50% | 25M | 5.92e-4 (3.58e-4 at 10M) | 1.04e-3 (stop; 2.98e-4 at 10M) |

The differences are small and point in both directions: lower than SwiGLU-7
at 10M, higher at 25M. That fits run-to-run GPU noise in 8K-sequence training
on different nodes, but this is an inference, not yet a measurement. The
earlier exact reproductions involved only local fits, token-0 evaluation, or
128-token training resumed on one machine, so the 1e-3 limit did not carry
over. The stopped runs are kept on PERUN as `fitted-target-{t}-run-001-b2-stopped`.
The rerun's values at 10M (and at 25M for 50%) against these measure
SwiGLU-8's own run-to-run noise. If the rerun reproduced the stopped runs
exactly, the difference from SwiGLU-7 would be systematic and would be
investigated before any Part B interpretation.

**Rerun and B2 review (2026-10-12).** The rerun (array 106606) completed all
four runs, so Part B now has all 16. The rerun did not reproduce its stopped
first attempt: at 10M the two differ by 0.01-0.41% in recovery KL and
0.01-0.16% in WikiText perplexity. **SwiGLU-8's 8K-sequence training is
therefore not repeatable run to run**, and the first attempt's gaps to
SwiGLU-7 (at most 0.21%) lie within that noise.

Against SwiGLU-7, 34 of the 36 B2 values are within 1%. The two flagged values
are single-checkpoint bumps:

| Run | Point | SwiGLU-8 KL | SwiGLU-7 KL | Difference | Neighbouring points |
| --- | --- | ---: | ---: | ---: | --- |
| Fitted 20% | 75M | 0.079554 | 0.080681 | 1.40% | SwiGLU-7 rose from 0.080062 at 50M, while SwiGLU-8 stayed flat; both agree within 0.8% at 100M |
| Fitted 30% | 100M | 0.116592 | 0.112604 | 3.54% | SwiGLU-8 rose from 0.114848 at 75M, while SwiGLU-7 fell; they agree within 0.1% at 75M |

WikiText perplexity at the same two points agrees within 0.05%. Each pipeline
has its own bump at a different point, although both see the same data in the
same order. The flags go in opposite directions. **Conclusion: GPU noise
amplified by training, not a pipeline difference.** All four reruns are
usable.

**Caveat for Part B.** A single checkpoint's recovery KL can move by up to
about 3.5% from noise alone, and the fitted 30% run ends on such a bump. Part
B's per-target rule calls a gap meaningful at 2% (decision log, 2026-10-11),
which assumed repeatable runs. Gaps far above that are unaffected. A target
whose gap is near the noise level needs replicate runs or a smoothed measure
before any conclusion. That choice is made once the gaps are known.

**Not yet verified.** Contract tests cover:

- the configuration and the Part A grid;
- the multi-seed adoption rule and the stage-2 indicator;
- the acceptance comparisons and the selection fingerprint;
- the document disjointness of the three 8K streams;
- the settings reaching the fitting code;
- the cosine schedule and the least-squares refit;
- the job mapping.

Offline, the allocation code rebuilds SwiGLU-7's 20-50% widths exactly from
its recorded curves. The notebook's Part A cells were also run once against a
synthetic block-check result, which checks field names and table code only,
not science. GPU execution, real data streams, and the acceptance checks
themselves have not run; the first real stages are also the first execution
test.

## Sources and attribution

- `src-grafting-2025`, Section 3.3, Table 2: local fitting followed by
  fine-tuning; no fine-tuning-only condition.
- `src-minitron-2024`, Section 2.2; Sections 3 and 4.3: width pruning followed
  by distillation, without local fitting.
- L2-SP (Li et al., ICML 2018) and cosine learning-rate decay (Loshchilov and
  Hutter, 2017) are standard published methods. They are not yet registered;
  register them before the thesis cites them.
- The exact down-projection refit is ordinary least squares and needs no
  citation. Its relation to MoDeGPT's down-projection reconstruction
  [src-modegpt-2025, Sections 3.1-3.2] is noted when it is reported.
- The fitting observations come from
  `data/results/workflows/model/swiglu-7/prepare-001/result.json`, checked
  2026-10-10.
