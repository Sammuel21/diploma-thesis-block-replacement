---
title: Recovery-1 - Recovery Schedule Screen
status: active
created: 2026-11-07
modified: 2026-11-07
---

# Recovery-1 - recovery schedule screen

**Status (2026-11-07):** designed, grilled, and implemented; not run. The
implementation is uncommitted and awaits the researcher's review. **Next step**
(session 2b): the researcher commits, pulls on PERUN, and submits the three
tasks. After the results are home, `summarize` applies the winner rule and the
session settles each family's schedule and the rerun set. The binding decisions
are the 2026-11-07 rows in the [shared decision log](../../../agents/decisions.md);
if this page and the log disagree, the log wins.

## Question

Does a learning-rate decay at the end of recovery beat the constant recipe, for
dense INT3 QAT, for B 20% INT4 (SwiGLU replacement with INT4 QAT, order B), and
for SwiGLU alone at 20% in BF16? Recovery-1 does not pick a schedule by
judgment: it fixes the rule that picks it, before any result exists.

## Why

Since SwiGLU-4, every recovery and QAT run has used constant 3e-5: no warmup,
no decay, no weight decay, fused AdamW, KL at T=1, an 8,192-token effective
batch (about 122k updates per 1B tokens), and seed 21. SwiGLU-4 chose it on a
5M-token tournament against 1% warmup plus cosine
([SwiGLU-4](../swiglu/swiglu-4.md#stage-1-configuration-tournament)).

Under that recipe the families behave differently:

| Family | 8K KL at 100M | 8K KL at 1B | Shape of the 1B curve |
| --- | ---: | ---: | --- |
| Dense INT3 QAT | 0.0804 | 0.0709 | Flat from about 125M; spikes at 100M and 825M; the 1B row is its lowest |
| B 20% INT4 | 0.1023 | 0.0770 | Still falling at 1B |
| SwiGLU 20% BF16 | 0.0787 | - | Still falling at 1B (monitoring KL) |

Values from [Quantization-2](../baseline/quantization-2.md#results-of-the-fixed-reruns-2026-11-07),
[Stacking-1](../stacking/stacking-1.md#order-b-at-1b-results-2026-11-02), and
[SwiGLU-8](../swiglu/swiglu-8.md#part-b-results-2026-10-12).

A 3B comparison under constant could favour stacking only because INT3 stalls
(decision log, 2026-11-07). The supervisor suggested warmup-stable-decay (WSD),
the schedule named by MiniCPM [`src-minicpm-2024`, Section 4.2].

**Two hypotheses for the INT3 stall** (researcher hypotheses, not tested by
this design):

- A learning-rate noise floor. In the river-valley picture, a high constant rate
  makes progress along the valley while oscillating across it, and the decay
  removes that oscillation [`src-wen-2024`, abstract and Section 1]. A decay
  should help.
- QAT weight oscillation around rounding boundaries. Nagel et al. report that
  such oscillations "happen irrespective of the learning rate", while a lower
  rate reduces their amplitude [`src-nagel-2022`, Section 2.2]. A decay may help
  less, or not at all.

Recovery-1 measures whether a decay helps; it cannot tell these mechanisms
apart. The papers train from scratch with large batches (Nagel et al.: CNNs with
quantized weights and activations), so their transfer to repairing a pretrained
model with small batches is part of the hypothesis.

## Design

Settled in the Recovery-1 grilling on 2026-11-07 (decision log, 2026-11-07).

**Families and starts.** Each family starts exactly where its recorded constant
run started, on SwiGLU-8's 100M stream (sha256 `570409c1…`):

| Family | Start | Trainable | Recorded constant run |
| --- | --- | --- | --- |
| `dense-int3` | Pinned SmolLM2-1.7B with INT3 fake quantization of the 66 MLP projections | All 66 projections | Quantization-2 `qat-int3-100m-002` |
| `b-0.2-int4` | SwiGLU-8's fitted 20% token-0 start, padded, all 66 projections INT4 fake-quantized | Replacement modules (S7-0) | Stacking-1 `b-fitted-0.2-100m-int4-002` |
| `swiglu-0.2` | The same start in BF16 | Replacement modules (S7-0) | SwiGLU-8 `fitted-target-0.2-run-001` |

B and SwiGLU differ only by INT4.

**Candidates.** Peak 3e-5, no warmup, one shape: the 1-sqrt cooldown to zero
over the last 10, 20, or 50% of the 100M budget. For decay start $D$ and budget
$T$, the rate after an update that ends at token $t$ is

$$\eta(t) = \eta_{\text{peak}} \cdot \left(1 - \sqrt{\frac{t - D}{T - D}}\right) \quad (t > D), \qquad \eta(t) = \eta_{\text{peak}} \quad (t \le D).$$

This is source-derived: the 1-sqrt function of Hägele et al.
[`src-hagele-2024`, Section 3.2], in this page's notation. Two choices are
the project's: each decay start is rounded up to the next update boundary
(90,005,504, 80,003,072, and 50,003,968 tokens), and the final rate is zero.
The fractions bracket the literature: about 10% in MiniCPM
[`src-minicpm-2024`, Section 4.3], and a plateau of benefit at about 20% in
Hägele et al. [`src-hagele-2024`, Section 3.2]. 50% tests whether a longer
decay keeps helping.

**Branching.** Without warmup and at the same peak, a stable-decay run equals
the constant run until its decay starts. So one GPU task per family trains a
constant **trunk** to 100M and keeps the trainable tensors and the optimizer
state at each decay start, in CPU memory. Then each decay restores its state and
trains to 100M with the decay. Each decay is compared with constant from an
identical state, and a family costs 180M tokens instead of 400M. No trained
model is kept.

```text
constant trunk   0 ────────── 50.0M ───────── 80.0M ─── 90.0M ─── 100M   (constant reference)
50% decay                       └──────── 1-sqrt to 0 ───────────── 100M
20% decay                                         └──── 1-sqrt ──── 100M
10% decay                                                   └ 1-sqrt 100M
```

**Score.** The 8K KL to the dense teacher on SwiGLU-8's Part A selection stream
(24 × 8,192 tokens, sha256 `85165ff8…`), which the final evaluation never uses
(decision log, 2026-10-11). It is measured on the training-time model: INT4
training weights equal the deployed ones bit for bit, and INT3 carries the
known 0.18% autocast difference (decision log, 2026-11-07), equally in every
candidate. Each family's **old history metric** is also recorded: the final 8K
stream for dense INT3 (Quantization-2's rows), the 128-token monitoring KL for
B and SwiGLU. It serves only the fidelity check and the secondary
token-equivalent, never the rule.

**Rows.** Trunk: 0, 10, 25, 50, 75, 80, 90, and 100M. Each decay: its midpoint
(95, 90, or 75M) and 100M. Every row records both scores; B and SwiGLU rows also
record WikiText perplexity where their recorded runs have it.

## Winner rule

Fixed before any result (decision log, 2026-11-07). Per family, on the
selection-stream 8K KL:

1. The reference $R$ is the trunk's lowest value at 75, 80, 90, and 100M, so a
   single constant spike cannot hand a decay the win. The dense INT3 1B run
   spiked +87% at exactly its 100M row.
2. A decay qualifies if its 100M value is below $R \cdot (1 - 0.035)$.
3. If none qualifies, the family keeps constant. If several do, the shortest
   decay within 3.5% of the best qualifier wins, because it stays closest to
   constant.
4. Each family decides on its own.

The bar, 3.5%, is the largest measured single-checkpoint noise: SwiGLU-8's BF16
runs at 100M; QAT twins reached 3% (decision log, 2026-11-02). It is also
Heterogeneous-2's bar. The rule is project-proposed and needs no citation.

**Reported, outside the rule:**

- The **token-equivalent**: the first token count at which the family's 1B
  constant run (Quantization-2 `qat-int3-1b-002`, Stacking-1
  `b-fitted-0.2-1b-int4-001`, SwiGLU-7 `S7-0-target-0.2-run-001`) reaches a
  candidate's 100M old-metric value, linear between 25M rows, or "> 1B". It is
  a project-defined descriptive measure; for dense INT3 it rests on the final
  stream.
- The **noise sample**: the largest relative difference between the trunk and
  its recorded run over 10-100M.
- **Flags**, which hold a family's decision until the researcher has reviewed
  them. While any family is held, `summarize` prints but writes nothing; after
  the review, `--accept-flags FAMILY` writes `summary-001` with the acceptance
  recorded. The flags are:
  - a trunk row outside 1% of the recorded run;
  - SwiGLU's token-0 selection KL against Heterogeneous-1's;
  - an old metric that disagrees with the selection stream on whether a decay
    qualifies.

## Gates

A task **stops** on any of these failures:

| Check | Compares | Tolerance |
| --- | --- | --- |
| `inputs` | sha256 of the recovery stream, selection stream, final stream, and monitoring batches against the configuration pins; the selection-stream file against its SwiGLU-8 record | Exact |
| `K3` (B, SwiGLU) | The start before padding and fake quantization against SwiGLU-8's token-0 row (Stacking-1's check) | Relative 1e-4 |
| `K6-start` (dense, B) | Every projection's fake-quantized weight against the deployed weight | Bitwise |
| `teacher-self-kl` | The teacher against itself on the selection stream | Absolute 1e-6 |
| `token-0` | The trunk's token-0 row against the recorded run's: dense final-stream KL; B and SwiGLU monitoring KL and WikiText perplexity | Relative 1e-4 |

**Recorded, never stopping:**

- `trunk-curve`: the trunk's 10-100M old-metric rows against the recorded run,
  at 1%. Flags are read with the 2026-11-01 sign-change rule: a gap that grows
  steadily needs a cause before the family's result is used.
- `selection-token-0` (SwiGLU): the token-0 selection KL against
  Heterogeneous-1's `assemble-001-target-0.2` (0.13732). That value came from a
  refit of the same model, not from the start file.

## After the screen

- **Transfer.** A family that adopts a decay gets one 1B run with the same decay
  fraction before any 3B run. It doubles as its matched rerun: dense INT3 1B and
  B 20% INT4 1B are the pair near 51% saving, and SwiGLU 20% 1B is its 20% rerun.
  It is scored against the kept 1B constant model on the selection stream
  (evaluation only), with the same bar; if it fails, the family uses constant at
  3B (decision log, 2026-11-07). No mid-run checkpoint survives the 1B runs:
  Quantization-2 and Stacking-1 delete them at completion.
- **Peak rate.** If Heterogeneous-2 Step 1 finds that SwiGLU gains more than
  3.5% at 1e-4, Recovery-1 still runs at 3e-5 and reports it as a limit; a
  higher-peak screen, which would need warmup and full runs, is a separate
  decision (decision log, 2026-11-07).
- The final rerun set is decided from the results (decision log, 2026-11-07).
  Stacking-2 and Quantization-3 (3B) follow, outside Recovery-1.

## Limits

- Untested: warmup, peak rate, decay shape, and final rate.
- One seed. The decays are paired with the trunk, but each comparison is still
  one trajectory.
- 100M only; transfer to 1B and 3B is checked only for adopting families.
- The score is on the training-time model; INT3 deployed KL runs 1.0-2.5% above
  it (Quantization-2, Limits).
- The dense token-0 gate compares the final-stream KL only. The recorded dense
  run's 8K perplexity needs Quantization-1's WikiText tokens, which Recovery-1
  does not load.

## Implementation

Implemented on 2026-11-07; uncommitted; nothing run on PERUN.

| File | Role |
| --- | --- |
| [recovery.py](../../../../src/mlp_replacement/compression/recovery.py) | `token_learning_rate_multiplier` gains `wsd_one_minus_sqrt` with `decay_start_tokens` (refuses warmup); `recover_trainable_by_tokens` passes it through. Existing schedules and callers are unchanged |
| [recovery_1.py](../../../../workflows/runs/model/recovery/recovery_1.py) | `train --task-index {0,1,2}` and `summarize` |
| [recovery-1.json](../../../../workflows/configs/model/recovery/recovery-1.json) | Families, decays, rows, rule, gates, and every sha256 pin |
| [recovery_1.sbatch](../../../../workflows/jobs/perun/recovery/recovery_1.sbatch) | PERUN array `0-2`, 4 h, 128 GB; the selection stream is given by `RECOVERY1_SELECTION_STREAM` |
| [test_recovery1_contracts.py](../../../../tests/test_recovery1_contracts.py) | CPU contract tests |

**Reuse.** The runner builds each start with the code its recorded run used:

- dense INT3: Quantization-1/2's `load_teacher`, `prepare_qat`, and the K6 check;
- B: Stacking-1's `load_start`, `pad_stage`, and the order-B preparation;
- SwiGLU: SwiGLU-7's `configure_strategy` (S7-0).

It calls the shared loop directly from token 0, with the 100M horizon, so the
decay sees absolute token counts. `recover_exact_segment` stays constant-only.
The configuration loader requires the dense QAT and SwiGLU-8 recipes to agree
key by key and refuses any new peak, warmup, or family order.

**Commands** (PERUN form in the [runbook](../../../infrastructure/perun/perun-runbook.md#14-recovery-1)):

```bash
python -m workflows.runs.model.recovery.recovery_1 train --task-index 0 \
  --work-dir WORK --output-dir OUT --prepared .../swiglu-8/prepare-001/result.json \
  --selection-stream .../long-context-selection.int32 --source-root .../perun-results
python -m workflows.runs.model.recovery.recovery_1 summarize --results-root .../perun-results/recovery-1 \
  --source-root .../perun-results [--accept-flags FAMILY ...]
```

**What was tested** (CPU, 2026-11-07, project Python; 16 new tests):

- On a tiny Llama model, a decay branched from the trunk's snapshot ends
  bitwise equal to one uninterrupted stable-decay run on the same data.
- The direct constant call equals the `recover_exact_segment` path the recorded
  runs used.
- Multiplier values and refusals; existing schedules unchanged.
- The decay starts sit on update boundaries and on trunk rows.
- The configuration loads through the Stacking-1 and Quantization-2 validators
  and refuses a new peak, warmup, reordered families, a missing reference row,
  or a bad pin.
- Every pin matches the local result copies.
- The winner rule: none, one, and several qualifiers, the spike guard, a
  missing row. The token-equivalent interpolation, the flags, and the job's
  array order.

Not tested: GPU execution, TorchAO and MSLK paths (TorchAO is not installed
locally), the real token-0 gates, snapshot memory, and run time. The in-run
gates of the real screen are the authority there. No smoke run (AGENTS.md).

## Pre-run resource plan

Recorded 2026-11-07 under the
[PERUN resource contract](../../../infrastructure/perun/perun-resources.md).
Estimates, not measurements.

| Task | Family | Tokens | Expected | Time limit | Ceiling GPU-h | Peak RAM / VRAM |
| --- | --- | ---: | ---: | --- | ---: | --- |
| 0 | Dense INT3 | 180M | 2.5-2.8 h | 4 h | 4 | ~55 GB (three snapshots of ~13 GB) / ~50 GiB |
| 1 | B 20% INT4 | 180M | 2.2-2.5 h | 4 h | 4 | ~40 GB / ~37 GiB |
| 2 | SwiGLU 20% | 180M | 1.9-2.2 h | 4 h | 4 | ~40 GB / ~31 GiB |
| **Total** | | | **6.6-7.5 GPU-h** | | **12** | |

- **Request per task:** one H200, 8 CPUs, 128 GB RAM, `gpu_short`; one array
  of three independent tasks. Expected CPU use 53-60 core-hours, ceiling 96.
- **Basis:** measured training time per 100M tokens in the recorded runs:
  4,845 s (dense INT3), 4,200 s (B), 3,717 s (SwiGLU), times 1.8. Evaluation
  rows take 5-11 s each (also measured), about 3 minutes per task. Model
  loading, start checks, and snapshot copies add about 0.2 h (estimate).
- **Storage:** PROJECT under 0.1 GB (JSON only); inputs are read in place;
  SCRATCH about 1 GB.
- **Confidence:** medium. The main uncertainties are node speed and the time to
  copy a 13 GB snapshot to CPU memory.
- **Headroom:** under 0.2% of the 8,424-GPU-hour allocation. The 20 GPU-hour
  figure from the planning brief was for independent runs.

## Sources

Registered in [the source registry](../../../../llm-wiki/raw/sources.yml) on
2026-11-07; locators from automated fetches of the arXiv HTML renderings, not
yet confirmed in the PDFs; no full ingestion.

- `src-minicpm-2024`: Hu et al., arXiv 2404.06395v3. Section 4.2, Eq. (1),
  names WSD; Section 4.3, decay of about 10% of steps.
- `src-hagele-2024`: Hägele et al., NeurIPS 2024, arXiv 2405.18392v3.
  Section 3.2: the 1-sqrt cooldown and a benefit plateau at about 20%.
- `src-wen-2024`: Wen et al., arXiv 2410.05192v3. Abstract and Section 1:
  the river-valley explanation of the decay.
- `src-nagel-2022`: Nagel et al., ICML 2022, arXiv 2203.11086v2. Section 2.2:
  QAT weight oscillation and its learning-rate dependence.
