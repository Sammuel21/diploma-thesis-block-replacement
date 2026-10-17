---
title: SwiGLU-9 - Local Fitting at 1B Tokens and Calibration Scaling
status: draft
created: 2026-10-17
modified: 2026-10-17
---

# SwiGLU-9 - Local Fitting at 1B Tokens and Calibration Scaling

**Status:** design settled on 2026-10-17 in a grilling session after the
[supervisor meeting of 2026-10-16](../../../meetings/2026-10-16-supervisor-feedback.md).
Implemented on 2026-10-17 without any run (see [Implementation](#implementation));
the [pre-run resource plan](#pre-run-resource-plan) is recorded. **Next step:**
the pre-submission checks and the three submissions in the
[PERUN runbook](../../../infrastructure/perun/perun-runbook.md#9-swiglu-9), after
the researcher's go. The binding decisions are the 2026-10-16 and 2026-10-17
rows in the [shared decision log](../../../agents/decisions.md); this page
explains them. If this page and the log disagree, the log wins until the
conflict is raised with the researcher.

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
not run; the first real stages are also the first execution test.

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
