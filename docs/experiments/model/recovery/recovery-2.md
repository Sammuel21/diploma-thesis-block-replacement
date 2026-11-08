---
title: Recovery-2 - Decay Length at 1B
status: active
created: 2026-11-08
modified: 2026-11-08
---

# Recovery-2 - decay length at 1B

**Status (2026-11-08):** designed in Recovery-1 session 2b. The plan session then
settled the open reference-row point (median of four), the update layout, the rows, and
the time limit. It implemented the runner, configuration, job, CPU tests, and the runbook
section. Nothing is committed and nothing has run. **Next step:** the researcher commits
and pushes, then submits the array from a fresh PERUN login once the queue is empty
([runbook section 15](../../../infrastructure/perun/perun-runbook.md#15-recovery-2)).
3B runs (Stacking-2, Quantization-3) wait for this experiment's results. The binding
decisions are the rows in the [shared decision log](../../../agents/decisions.md); if this
page and the log disagree, the log wins.

## Question

For each Recovery-1 family, does a 1-sqrt learning-rate decay at the end of a
1B-token recovery beat the constant 3e-5 recipe, and does a long decay (50%)
beat a short one (20%)? Each family's answer sets the schedule of its 3B run.

## Why

[Recovery-1](recovery-1.md) found at 100M that a decay helps dense INT3 QAT
(+5.8% at 20%, +7.6% at 50%) and not B 20% INT4 or SwiGLU 20% (best +0.6% and
0.0%). Its constant trunk had stalled for dense and was still falling fast for
the other two. Three observations argue for a second screen at 1B before any 3B
run:

1. **B and SwiGLU slow down sharply by 1B.** At the end of each window, the
   recorded 1B constant runs gain far less per token than the Recovery-1 trunks
   did at 100M. By 1B, B and SwiGLU are flatter than dense INT3 was when the
   decay helped it. The two metrics differ (the trunks use the selection stream,
   the 1B runs monitoring KL or the final stream), so this comparison is
   indicative only:

   | Run | Window | Change per 25M |
   | --- | --- | ---: |
   | Dense INT3 trunk (Recovery-1) | 75-100M | −1.85% (80-100M: −0.4%) |
   | B 20% INT4 trunk | 75-100M | −3.0% |
   | SwiGLU 20% trunk | 75-100M | −2.4% |
   | B 20% INT4, 1B constant run | 750M-1B | about −0.3% |
   | SwiGLU 20%, 1B constant run | 750M-1B | about −0.36% |

   A screen at 100M therefore cannot say whether a decay would help them at 1B
   or 3B. Without that test the 3B comparison of B against dense INT3 could
   favour dense only because only dense was given a decay.
2. **A single 1B run cannot choose a decay length.** At 100M the dense gain was
   still rising with decay length (2.4 → 5.8 → 7.6%), with no plateau at 20%.
   One 1B run at 20% would test whether 20% transfers, not whether a longer
   decay is better at 3B.
3. **The recorded 1B constant endpoints are noisy references.** Dense INT3 at 1B
   swings ±11% between 25M rows (row-to-row standard deviation 10.6% after 500M).
   Its last row is its lowest. A paired comparison against a new constant trunk
   avoids judging a decay against one lucky or unlucky endpoint.

**Literature context.** These sources are not yet registered in the
[source registry](../../../../llm-wiki/raw/sources.yml), and their locators come
from automated reads of the arXiv HTML, not checked in the PDFs. Every
comparable retraining recipe found ends with a decay:

- Dremov et al. 2025, arXiv 2509.22935 (*Compute-Optimal Quantization-Aware
  Training*): the QAT phase uses 5% re-warmup and cosine decay to zero
  (App. A.3). Running the full-precision cooldown inside QAT improved every
  4- and 6-bit run (Section 5, Tables 2 and 10).
- Sreenivas et al. 2024, arXiv 2408.11796 (*LLM Pruning and Distillation in
  Practice: The Minitron Approach*): KL-only distillation after pruning with
  peak 1e-4 and cosine decay (Table 4).
- Xu et al. 2026, arXiv 2606.14150 (*Small LLMs: Pruning vs. Training from
  Scratch*): cosine with 5% warmup to 10% of a swept peak (App. D.1). Pruned
  models keep improving to 500B tokens with diminishing returns (Tables 12-15).

None of them tests constant against decay for repairing a compressed model, or
schedules for stacked compression.

## Design

**Families and starts.** The three Recovery-1 families. Each starts where its
recorded 1B constant run started, trains on that run's data, and copies that run's
update layout:

| Family | Recorded 1B constant run (fidelity reference) | Data | Update layout | Decay starts (50%, 20%) |
| --- | --- | --- | --- | --- |
| `dense-int3` | Quantization-2 `qat-int3-1b-002` | 1B stream `f78bf805…` | One segment from token 0 | 500,006,912; 800,006,144 |
| `b-0.2-int4` | Stacking-1 `b-fitted-0.2-1b-int4-001` | 1B stream `f78bf805…` | Two segments: 0-100M, 100M-1B | 500,007,168; 800,006,400 |
| `swiglu-0.2` | SwiGLU-7 `S7-0-target-0.2-run-001` | 1B stream `f78bf805…` | Two segments: 0-100M, 100M-1B | 500,007,168; 800,006,400 |

Checked on 2026-11-08 from the local artifacts:

- **One stream.** All three read one file, SwiGLU-7's `prepare-001` stream, which is
  `f78bf805…`. Quantization-2's 1B QAT built a stream with that hash, and Stacking-1's
  B 1B run read the SwiGLU-7 file.
- **Token 0.** Each recorded 1B run's token-0 row equals its 100M run's exactly, which
  Recovery-1 already reproduced.
- **SwiGLU-7 and SwiGLU-8.** SwiGLU-7's 20% run and SwiGLU-8's are the same model on
  the same recipe ([SwiGLU-8](../swiglu/swiglu-8.md), "The fitted 20% and 50% arms
  already exist"), with identical token-0 monitoring KL (0.110631).

**Update layout** (decision log, 2026-11-08). SwiGLU-7's 1B recipe, which Stacking-1
reused, trains in two segments. The loop restarts at 100M from the optimizer state saved
there. Because 100,000,000 is not a multiple of 8,192, the first segment ends with a
256-token update (update 12,208), and every later 8K window starts 256 tokens away from
where a single run from token 0 would put it. B and SwiGLU repeat that layout, so their
trunks see the recorded runs' windows and updates; dense trained in one segment and keeps
it. Each decay start is its request rounded up to an update boundary of the family's own
layout, which is the recorded run's 500M or 800M row.

**Branching.** This is the Recovery-1 method at 1B. One constant 3e-5 trunk per
family trains to 1B and keeps the trainable and optimizer state at each decay
start. Each decay restores its state and trains to 1B with the rate falling
1-sqrt to 0. Peak 3e-5, no warmup.

```text
constant trunk  0 ─────────── 500M ────────── 800M ─────────── 1B   (rate stays 3e-5)
50% decay                      └──── 1-sqrt to 0 ──────────────── 1B   (dense from 500,006,912; B, SwiGLU from 500,007,168)
20% decay                                     └── 1-sqrt to 0 ─── 1B   (dense from 800,006,144; B, SwiGLU from 800,006,400)
```

Each family trains 1.7B tokens (1B + 500M + 200M), against 3B for three
separate runs. The trunk also repeats the recorded constant 1B run, so it is
both the constant reference and a 1B noise sample. The decay lengths are those
of Recovery-1 (20%, 50%), so the ordering can be compared across budgets; 10%
was the weakest at 100M and is dropped.

**Score.** The 8K KL on SwiGLU-8's selection stream (`85165ff8…`), as in
Recovery-1. Each family's old history metric (dense: the final 8K stream; B and
SwiGLU: the 128-token monitoring KL, with WikiText perplexity at 10, 25, 50, 100, 250,
500M and 1B) is also recorded for the fidelity check.

**Rows** (decision log, 2026-11-08). Both scores at every row:

- the trunk at 10M and every 25M to 1B, which are the recorded runs' 41 rows,
  including the four reference rows and both decay starts;
- each decay every 25M from its start to 1B, with its midpoint (750M or 900M) and
  its end: 20 rows for the 50% decay and 8 for the 20% decay.

**Reliability.** No resume (researcher decision, 2026-11-08). Each task runs in
one job with its decay-start states in CPU memory. A crash means rerunning that
task from token 0.

## Winner rule

The Recovery-1 rule at 1B, with a median reference, fixed before any result
(decision log, 2026-11-08):

1. The reference is the **median** of the trunk's selection-stream values at
   750M, 800M, 900M, and 1B (the same 75/80/90/100% points as Recovery-1).
2. A decay qualifies if its 1B value is more than 3.5% below the reference.
3. With no qualifier, the family keeps constant. With both qualifying, the
   shorter decay wins if it is within 3.5% of the better one; otherwise the 50%
   decay wins.
4. Each family decides on its own.

**Why the median, not Recovery-1's lowest row.** At 100M the lowest-of-four reference
guarded against a constant spike. At 1B, the recorded constant runs swing more than the
3.5% bar between rows. On their old metrics:

- dense INT3's four reference rows are 0.0738, 0.0794, 0.0804, and 0.0692, so its median
  is 10.8% above its lowest;
- B's median is 2.0% above its lowest, and SwiGLU's 3.5%.

With the lowest row, a decay would have to land about 13% below the median constant level
for dense, and 5.4% for B and 6.7% for SwiGLU. Recovery-1's dense gains at 100M were
5.8-7.6%. A dense false rejection is the one error that favours B in the 3B comparison.
The median is robust to one spike or one dip. Its cost is the opposite error: 4 of dense's
11 recorded constant rows from 750M to 1B would themselves pass it, so a decay that is no
better than a good constant row can be adopted. That error is cheap, because Recovery-1's
worst decay cost 2.5%, within noise. The summary also reports the decision the
lowest-of-four reference would give; it never decides. These numbers come from the
recorded runs' old metrics, because no 1B selection-stream curve existed before this
experiment. The rule is project-proposed and needs no citation.

## From Recovery-2 to 3B

| Family's Recovery-2 result | Its 3B schedule |
| --- | --- |
| No decay qualifies | Constant 3e-5 to 3B |
| 20% wins | Constant to 2.4B, then 1-sqrt to 0 over the last 600M |
| 50% wins | Constant to 1.5B, then 1-sqrt to 0 over the last 1.5B |

Transfer from 1B to 3B is assumed and reported as a limit; a 3B screen would
cost about three times as much per family.

## Gates

A task **stops** on any of these failures:

| Check | Compares | Tolerance |
| --- | --- | --- |
| `inputs` | sha256 of the 1B stream (SwiGLU-7's record against the pin; the file reread), selection stream, final stream, and monitoring batches; every pinned record | Exact |
| `K3` (B, SwiGLU) | The start before padding and fake quantization against SwiGLU-8's token-0 row | Relative 1e-4 |
| `K6-start` (dense, B) | Every projection's fake-quantized weight against the deployed weight | Bitwise |
| `teacher-self-kl` | The teacher against itself on the selection stream | Absolute 1e-6 |
| `token-0` | The trunk's token-0 row against the recorded 1B run's: dense final-stream KL; B and SwiGLU monitoring KL and WikiText perplexity | Relative 1e-4 |
| `selection-token-0` | The token-0 selection KL against the family's Recovery-1 trunk (same start, stream, and code) | Relative 1e-4 |

**Recorded, never stopping:** `trunk-curve`, the trunk's old-metric rows against the
recorded 1B run at every row from 10M to 1B, at 1%, read with the 2026-11-01 sign-change
rule. Flags are expected, at least for dense INT3, whose recorded 1B run swings ±11%. As in
Recovery-1, flags hold the decision until the researcher reviews them. The other flag is an
old metric that disagrees with the selection stream on whether a decay qualifies.

Recovery-1's token-equivalent is not reported: the constant curve it reads ends at 1B, so
every value would be "> 1B".

## Pre-run resource plan

Recorded 2026-11-08 under the
[PERUN resource contract](../../../infrastructure/perun/perun-resources.md).
Estimates, not measurements.

| Task | Family | Training tokens | Expected | Time limit | Ceiling GPU-h | Peak RAM / VRAM |
| --- | --- | ---: | --- | --- | ---: | --- |
| 0 | Dense INT3 | 1.7B | 23.3-24.0 h | 30 h | 30 | ~27 GiB (two snapshots of ~13 GB) / ~50 GiB |
| 1 | B 20% INT4 | 1.7B | 20.0-20.6 h | 30 h | 30 | ~14 GiB / ~37 GiB |
| 2 | SwiGLU 20% | 1.7B | 17.6-18.2 h | 30 h | 30 | ~14 GiB / ~31 GiB |
| **Total** | | 5.1B | **61-63 GPU-h** | | **90** | |

- **Request per task:** one H200, 8 CPUs, 128 GB RAM, `gpu_short`; one array of
  three independent tasks. Expected CPU use about 490 core-hours, ceiling 720.
- **Basis:** measured training speed per 100M tokens, times 17:
  - Recovery-1's trunks: 1.356, 1.169, and 1.024 h for dense, B, and SwiGLU;
  - the recorded 1B runs: 1.349, 1.176, and 1.041 h.

  The two agree within 2%. Each task also has about 70 evaluation rows at a measured
  6-13 s each (0.15-0.25 h), plus model loading, hashing the 4 GB stream, and snapshot
  copies (about 0.1 h, estimate). Peak RAM and VRAM scale Recovery-1's measurements to two
  snapshots.
- **Storage:** PROJECT under 1 MB per family (JSON only); inputs are read in place,
  including SwiGLU-7's 4 GB stream; SCRATCH about 1 GB.
- **Confidence:** high on speed. The main risk is a crash, since there is no resume:
  up to about 24 GPU-hours lost for dense.
- **Headroom:** under 1% of the 8,424-GPU-hour allocation (about 495 GPU-hours
  logged so far).

## Limits

- One seed; peak 3e-5, no warmup, and the 1-sqrt shape only. Peak rate,
  warmup, and shape stay untested (Recovery-1 limits).
- 1B only; transfer to 3B is assumed.
- The last update of each decay runs at rate 0 (end-of-update convention, as in
  Recovery-1).
- The median reference can adopt a decay that is no better than a good constant row
  (see the winner rule); the lowest-of-four outcome is reported next to it.

## Implementation

Implemented on 2026-11-08; not committed, not run.

| File | Role |
| --- | --- |
| [recovery_2.py](../../../../workflows/runs/model/recovery/recovery_2.py) | `train --task-index {0,1,2}` and `summarize` |
| [recovery-2.json](../../../../workflows/configs/model/recovery/recovery-2.json) | Families with their update layouts, decays, rows, the rule, gates, and every sha256 pin |
| [recovery_2.sbatch](../../../../workflows/jobs/perun/recovery/recovery_2.sbatch) | PERUN array `0-2`, 30 h, 128 GB; the selection stream is given by `RECOVERY2_SELECTION_STREAM` |
| [test_recovery2_contracts.py](../../../../tests/test_recovery2_contracts.py) | CPU contract tests |

**Reuse.** No shared code changed:

- **Recovery-1, imported unchanged:** the family starts (`open_dense`, `open_order_b`,
  `open_swiglu`), scoring, the fidelity curve, and the flags. Recovery-1's runner and
  `src/` are unchanged.
- **Recipe:** Stacking-1's `recovery_1b` block, which is SwiGLU-7's 1B recipe. The
  loader requires it to agree with Quantization-2's QAT block and Stacking-1's 100M
  block key by key.
- **Loop:** each segment is one direct call of the shared loop, with token counts shifted
  by the segment's start as `recover_exact_segment` shifts them. That function stays
  constant-only. The 1-sqrt rate depends only on t − D and T − D, so the shift leaves it
  unchanged.
- **Segment boundary:** the trunk carries the optimizer state across 100M, as SwiGLU-7
  and Stacking-1 did.

**Commands** (PERUN form in the
[runbook](../../../infrastructure/perun/perun-runbook.md#15-recovery-2)):

```bash
python -m workflows.runs.model.recovery.recovery_2 train --task-index 0 \
  --work-dir WORK --output-dir OUT --prepared .../swiglu-8/prepare-001/result.json \
  --selection-stream .../long-context-selection.int32 --source-root .../perun-results
python -m workflows.runs.model.recovery.recovery_2 summarize --results-root .../perun-results/recovery-2 \
  [--accept-flags FAMILY ...]
```

**What was tested** (CPU, 2026-11-08, project Python; 14 new tests, and Recovery-1's 17
still pass):

- On a tiny Llama model with a segment end that leaves a partial update, as 100M does,
  the trunk equals the recorded path bitwise in both layouts. The recorded path is one
  `recover_exact_segment` call per segment with the optimizer state carried. The two
  layouts give different weights, so the test can tell them apart.
- A decay restored from the trunk's snapshot ends bitwise equal to the same stable-decay
  run without the snapshot. Every branch row's rate equals the 1-sqrt formula in absolute
  tokens within 1e-12.
- From the local artifacts, the planned rows reproduce every recorded 1B row's tokens, and
  for B and SwiGLU its update count (12,208 at 100,000,000). Each decay start is a recorded
  row.
- The configuration loads through the Stacking-1 and Quantization-2 validators. It refuses
  a new peak, warmup, other decays, reordered families, a wrong layout, a wrong Recovery-1
  run, a non-median reference, a missing reference row, a new row interval, and a bad pin.
  Every pin matches the local copies.
- The token-0 gates pass on a match and stop on a 0.1% difference from Recovery-1.
- The rule: none, one, and both decays qualify; the tie rule; a spike and a dip; a missing
  row; the lowest-of-four line. The summary holds flagged families and writes only after
  `--accept-flags`. The job's array order, module, time limit, inputs, and line endings.

Not tested: GPU execution, TorchAO and MSLK paths (TorchAO is not installed
locally), the real token-0 gates, snapshot memory, and run time. The in-run
gates of the real screen are the authority there. No smoke run (AGENTS.md).
