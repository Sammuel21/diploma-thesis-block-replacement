---
title: Recovery-2 - Decay Length at 1B
status: planned
created: 2026-11-08
modified: 2026-11-08
---

# Recovery-2 - decay length at 1B

**Status (2026-11-08):** designed with the researcher in Recovery-1 session 2b.
The decision rows were written on 2026-11-08; nothing is
implemented. **Next step:** a fresh plan-and-implementation session builds the
runner, configuration, job, and tests, writes the pre-run resource plan, and
gives the researcher the PERUN commands. 3B runs (Stacking-2, Quantization-3)
wait for this experiment's results. The binding decisions are the rows in the
[shared decision log](../../../agents/decisions.md); if this page and the log
disagree, the log wins.

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
recorded 1B constant run started and trains on that run's data:

| Family | Recorded 1B constant run (fidelity reference) | Data |
| --- | --- | --- |
| `dense-int3` | Quantization-2 `qat-int3-1b-002` | 1B stream `f78bf805…` |
| `b-0.2-int4` | Stacking-1 `b-fitted-0.2-1b-int4-001` | 1B stream `f78bf805…` |
| `swiglu-0.2` | SwiGLU-7 `S7-0-target-0.2-run-001` | SwiGLU-7's 1B stream (to confirm equal to `f78bf805…` in the plan session) |

SwiGLU-7's 20% run and SwiGLU-8's are the same model on the same recipe
([SwiGLU-8](../swiglu/swiglu-8.md), "The fitted 20% and 50% arms already
exist"). Their token-0 monitoring KL is identical (0.110631; checked on
2026-11-08 from the local artifacts).

**Branching.** This is the Recovery-1 method at 1B. One constant 3e-5 trunk per
family trains to 1B and keeps the trainable and optimizer state at each decay
start. Each decay restores its state and trains to 1B with the rate falling
1-sqrt to 0. Peak 3e-5, no warmup, and each start rounded up to the next
8,192-token update boundary.

```text
constant trunk  0 ─────────── 500M ────────── 800M ─────────── 1B   (rate stays 3e-5)
50% decay                      └──── 1-sqrt to 0 ──────────────── 1B   (from 500,006,912)
20% decay                                     └── 1-sqrt to 0 ─── 1B   (from 800,006,144)
```

Each family trains 1.7B tokens (1B + 500M + 200M), against 3B for three
separate runs. The trunk also repeats the recorded constant 1B run, so it is
both the constant reference and a 1B noise sample. The decay lengths are those
of Recovery-1 (20%, 50%), so the ordering can be compared across budgets; 10%
was the weakest at 100M and is dropped.

**Score.** The 8K KL on SwiGLU-8's selection stream (`85165ff8…`), as in
Recovery-1. Each family's old history metric is also recorded for the fidelity
check.

**Reliability.** No resume (researcher decision, 2026-11-08). Each task runs in
one job with its decay-start states in CPU memory. A crash means rerunning that
task from token 0.

## Winner rule

The Recovery-1 rule at 1B, fixed before any result:

1. The reference is the trunk's lowest selection-stream value at 750M, 800M,
   900M, and 1B (the same 75/80/90/100% points as Recovery-1).
2. A decay qualifies if its 1B value is more than 3.5% below the reference.
3. With no qualifier, the family keeps constant. With both qualifying, the
   shorter decay wins if it is within 3.5% of the better one; otherwise the 50%
   decay wins.
4. Each family decides on its own.

**Open point for the plan session, before any result.** With dense INT3's ±11%
swings at 1B, the lowest-of-four reference is strict: it can reject a real gain.
At 100M it guarded against a constant spike. The plan session either keeps it
or replaces it before submission (for example with the median of the four
rows), and records the choice as a decision row.

## From Recovery-2 to 3B

| Family's Recovery-2 result | Its 3B schedule |
| --- | --- |
| No decay qualifies | Constant 3e-5 to 3B |
| 20% wins | Constant to 2.4B, then 1-sqrt to 0 over the last 600M |
| 50% wins | Constant to 1.5B, then 1-sqrt to 0 over the last 1.5B |

Transfer from 1B to 3B is assumed and reported as a limit; a 3B screen would
cost about three times as much per family.

## Gates

As in Recovery-1, against each family's recorded 1B constant run:

- **Stops:**
  - sha256 of every stream and pinned record;
  - the token-0 row within relative 1e-4;
  - K3 (B, SwiGLU) and the bitwise fake-quantization check (dense, B);
  - the selection-stream teacher self-KL within 1e-6.
- **Recorded, never stopping:** the trunk's old-metric rows against the recorded
  run every 25M at 1%, read with the 2026-11-01 sign-change rule. Flags are
  expected, at least for dense INT3, whose recorded 1B run swings ±11%. As in
  Recovery-1, flags hold the decision until the researcher reviews them.

## Rough cost (not the pre-run plan)

Scaled from Recovery-1's measured trunk speed (1.36, 1.17, 1.02 h per 100M) to
1.7B tokens, plus scoring:

| Task | Family | Training tokens | Rough GPU-hours |
| --- | --- | ---: | ---: |
| 0 | Dense INT3 | 1.7B | ~24 |
| 1 | B 20% INT4 | 1.7B | ~21 |
| 2 | SwiGLU 20% | 1.7B | ~18 |
| | **Total** | | **~60-65** |

- The time limit is about 30 h per task, within `gpu_short`'s 2 days.
- RAM: dense holds two snapshots (Recovery-1's three peaked at 39.3 GiB), so
  128 GB is ample.
- PROJECT: JSON only.
- The pre-run resource plan follows the resource contract in the
  implementation session.

## Limits

- One seed; peak 3e-5, no warmup, and the 1-sqrt shape only. Peak rate,
  warmup, and shape stay untested (Recovery-1 limits).
- 1B only; transfer to 3B is assumed.
- The last update of each decay runs at rate 0 (end-of-update convention, as in
  Recovery-1).

## Implementation (to be built)

These are the expected pieces. The plan session confirms them and the reuse:

- `workflows/runs/model/recovery/recovery_2.py`: reuses Recovery-1's family
  starts, scoring, gates, trunk, and branch code where they generalize.
  Recovery-1's runner itself stays fixed at 100M.
- `workflows/configs/model/recovery/recovery-2.json`: families, the 1B recipes
  and streams, decay lengths, rows, the rule, and every sha256 pin.
- `workflows/jobs/perun/recovery/recovery_2.sbatch`: array `0-2`.
- CPU contract tests, and a runbook section.

No smoke run (AGENTS.md); the in-run gates are the authority.
