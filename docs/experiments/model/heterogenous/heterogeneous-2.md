---
title: Heterogeneous-2 - Forced Models After Recovery
status: draft
created: 2026-11-04
modified: 2026-11-06
---

# Heterogeneous-2 - Forced Models After Recovery

**Status:** Step 1 is implemented and not yet run. Steps 2 and 3 are conditional and not
built. The binding decisions are the 2026-11-06 entries in the
[shared decision log](../../../agents/decisions.md), which win if the two disagree; this
page explains them.
**Next step:** the researcher commits, deploys, and submits the six runs of Step 1
([commands](../../../infrastructure/perun/perun-runbook.md#13-heterogeneous-2)).

## Why this experiment

[Heterogeneous-1](heterogeneous-1.md) asked whether new blocks (plain MLP, linear + SwiGLU,
and others) earn a place in the model. Its knapsack never picked one. But the knapsack judged
blocks by their damage *before* recovery training, and recovery repairs a lot: in SwiGLU-8 a
110-427% gap at token 0 fell to 10-24% by 100M tokens. So "the allocator refused it" does not
yet mean "it cannot help". The open question is whether new blocks catch up after real
training. With no natural hybrids to train, this experiment forces them in.

**Four terms used on this page.**

- **Forced model:** the eight layers predicted to suffer least must use a non-SwiGLU block, and
  the knapsack solves the rest at the same parameter budget
  ([forcing rule](heterogeneous-1.md#stage-5---assembled-hybrids-and-the-degradation-curve)).
  Heterogeneous-1 built these at token 0 and never trained them.
- **Token 0:** the model before any recovery training.
- **Monitoring KL:** KL to the teacher on a fixed set of 128-token sequences, measured during
  training at 0, 10, 25, 50, 75, and 100M tokens. SwiGLU-8 recorded these curves.
- **8K KL:** KL on a separate stream of 24 sequences of 8,192 tokens, measured once at the end.
  It is SwiGLU-8's headline score, and it is not the selection stream Heterogeneous-1 scored.

## Plan in three steps

| Step | What | Cost | Runs when |
| --- | --- | ---: | --- |
| 1 | Forced models to 100M, scored by KL, plus a learning-rate check | about 10 GPU-hours | Now |
| 2 | Forced models to 1B, with perplexity and tasks | about 10-12 GPU-hours per target | Per target, if Step 1 passes it |
| 3 | Retraining scopes, the no-fit test, SwiGLU's S7-1 and S7-2 at 80% | up to about 170 GPU-hours | Only if 1B still looks competitive |

If every target stops at Step 1, the thesis says the allocator selected no hybrid on
SmolLM2-1.7B, so hybrids were enforced in the eight layers predicted to be least harmful, and
they failed to beat pure SwiGLU after 100M tokens. It must not claim new operators cannot work.

## Step 1 - what runs

| Task | Name | Model | Rate | Why |
| ---: | --- | --- | --- | --- |
| 0 | `swiglu-0.8-control` | SwiGLU-8's 80% | 3e-5 | Proves the new training code reproduces SwiGLU-8's curve |
| 1 | `forced-0.2` | Forced 20% (8 linear + SwiGLU) | 3e-5 | Main comparison |
| 2 | `forced-0.5` | Forced 50% (7 linear + SwiGLU, 1 plain MLP) | 3e-5 | Main comparison |
| 3 | `forced-0.8` | Forced 80% (8 plain MLP) | 3e-5 | Main comparison |
| 4 | `forced-0.8-lr1e-4` | Forced 80% | 1e-4 | Learning-rate check |
| 5 | `swiglu-0.8-lr1e-4` | SwiGLU-8's 80% | 1e-4 | Learning-rate check |

Everything else is SwiGLU-8's recipe: the same 100M-token stream in the same order, seed 21,
training only the replaced blocks, a constant rate, and the same checkpoints.

**Why a sixth run.** None of the five main runs can test the new training code against a
recorded curve: the only SwiGLU model among them trains at 1e-4, and SwiGLU-8's record exists
only at 3e-5. So task 0 retrains SwiGLU-8's own 80% model at 3e-5. It goes first, and the
other five start once it passes.

## Reading the main comparison

For each target, gap = (forced KL − SwiGLU-8 KL) / SwiGLU-8 KL, on monitoring KL at the same
token count. Positive means forced is worse. This is a project-defined rule on standard
arithmetic and needs no citation. For scale, the token-0 gaps are +20.5%, +8.9%, and +7.6%
([degradation curve](heterogeneous-1.md#stage-5-results-2026-11-06)).

| Gap at 100M | Change from 50M to 100M | Outcome for that target |
| --- | --- | --- |
| Below −3.5% | any | Go to 1B: forced is better |
| −3.5% to +3.5% | any | Go to 1B: level |
| Above +3.5% | shrank by more than 3.5 points | Go to 1B: still catching up |
| Above +3.5% | otherwise | Stop |

**Why the 8K KL is recorded too.** The rule stays on monitoring KL, but each run also records
the 8K KL at 100M, and the summary flags a target whose 8K gap falls in a different band
(better, level, or worse) for the researcher to decide. The reason is noise. A single
checkpoint can bump by up to about 3.5% from noise alone
([SwiGLU-8](../swiglu/swiglu-8.md)), which is the size of the bar, and SwiGLU-9 once saw a
43.7% transient at 50M ([review](../swiglu/swiglu-9.md)). On SwiGLU-8's fitted-versus-slice
gaps at 100M, monitoring KL and the 8K KL differed by up to 3.9 points at 20-30% and at most
1.3 points at 40% and above (offline calculation, 2026-11-06).

## Reading the 1e-4 check

**Why it exists.** 3e-5 was tuned on SwiGLU, and SwiGLU-4 tried nothing above it for
replacements, so it may hold back new blocks. Four 80% curves result: forced and SwiGLU, each
at 3e-5 and 1e-4. SwiGLU's 3e-5 curve is its recorded one.

**The reading,** all at 100M on monitoring KL:

1. Take each family's better rate: the lower of its two KLs.
2. Calculate the gap between those two best values.
3. Compare it with the plain 3e-5 gap. The difference is the shift.

Comparing each family at its own better rate means a rate that merely hurts SwiGLU cannot
flatter the forced model. Invented example: forced 0.60 at 3e-5 and 0.50 at 1e-4; SwiGLU 0.52
at both. The 3e-5 gap is +15.4%, the better-rate gap is −3.8%, so the shift is 19.2 points
toward new blocks.

| Shift toward new blocks | Meaning | What happens |
| --- | --- | --- |
| 3.5 points or less | The rate is not what separates them | The 3e-5 results stand at all three targets |
| More than 3.5 points | The check favours new blocks | See below |

**If it favours new blocks:**

1. The 80% decision uses the better-rate curves, which the summary does by itself.
2. Work pauses for the researcher's go on four more runs: forced and SwiGLU at 1e-4, at 20%
   and 50%. That is about 6 GPU-hours and no new code, only new rows in the configuration and
   the job. It is needed because 80% tested only plain MLP, while 20% and 50% are mostly
   linear + SwiGLU, so 80% says little about them.
3. For Step 2's planning: if SwiGLU's better rate was also 1e-4, a forced 1B run at 1e-4 cannot
   fairly face the existing 3e-5 1B models and needs a SwiGLU partner, about 10-12 GPU-hours
   per target. A constant rate that wins at 100M may also not win at 1B.

**Side finding.** If SwiGLU itself improves by more than 3.5% at 1e-4, that is reported as a
limit of its recipe. It does not reopen the homogeneous results.

**Limit either way.** This is one target, one seed, and 100M tokens. A neutral result clears
plain MLP only.

## Fidelity checks

These are part of the real runs. r is the relative difference.

| Check | Compared with | Limit | On failure | Where |
| --- | --- | --- | --- | --- |
| Data, streams, neuron ranking | SwiGLU-8's records | exact | Stops | Reused from Heterogeneous-1 |
| Teacher against itself | zero | 1e-6 absolute | Stops | Reused |
| Each refitted block | its recorded price | r ≤ 1e-3 | Stops | Reused |
| Token-0 KL: monitoring, 128-token and 8K selection | the recorded `assemble` values | r ≤ 1e-4 | Stops before any training | New, every run |
| Token-0 monitoring KL | SwiGLU-8's recorded row | r ≤ 1e-4 | Stops | New, SwiGLU models |
| Checkpoint schedule | SwiGLU-8's recorded checkpoints | exact | Stops | New, every run |
| Trainable parameter count | the sum of the plan's block sizes | exact | Stops | New, every run |
| Non-finite KL | - | - | Stops | Every checkpoint |
| Curve: monitoring KL at 10-100M and 8K KL | SwiGLU-8's recorded 80% run | r ≤ 1% | Run finishes, then the job fails, so the other five do not start | New, control only |

**Token 0.** Each model must reproduce its recorded monitoring KL: forced 20% 0.13334, forced
50% 0.43234, forced 80% 1.13169, SwiGLU-8 80% 1.05148. Refits were bit-identical in
Heterogeneous-1, so a difference of exactly 0 is expected. Each model has one boundary width
with no recorded price (widths 6965, 3074, 716, and 2252 at layers 16, 9, 7, and 21); the
token-0 check covers those blocks.

**The curve.** SwiGLU-9's repeat of this exact model stayed within 0.1% of SwiGLU-8's at every
checkpoint up to 100M (offline calculation, 2026-11-06), so 1%, the project's existing review
limit, is ten times that. Training is not stopped mid-run, because a single transient can
happen and a complete curve is needed to judge it. A flag leaves `result.json` complete with the
check marked failed; only the job's exit code fails, which holds back the other five. A flag is
read under the 2026-11-01 rule: a gap that changes sign or falls back is noise, and a gap that
grows steadily needs a cause before anything is used.

**Why a match is expected** (from reading the code, not shown by a run). Both paths use the same
model loader, optimizer settings, stream, schedule, and training function. Blocks are inserted
after refitting instead of loaded from a saved start, there is no WikiText monitoring, and no
checkpoints are written. Training has no dropout or random sampling, so random state should not
matter.

## Implementation

Implemented on 2026-11-06 without any run. Runner:
[`heterogeneous_2.py`](../../../../workflows/runs/model/heterogenous/heterogeneous_2.py).
Configuration:
[`heterogeneous-2.json`](../../../../workflows/configs/model/heterogenous/heterogeneous-2.json).
PERUN job:
[`heterogeneous_2.sbatch`](../../../../workflows/jobs/perun/heterogenous/heterogeneous_2.sbatch).
`heterogeneous_1.py`, `swiglu_8.py`, and `src/` are unchanged. Results copied home go to
`data/results/workflows/model/heterogeneous-2/`.

| Command | Does | Output (`perun-results/heterogeneous-2/`) |
| --- | --- | --- |
| `train --task-index i` | One run. Rebuilds the model recorded by `assemble`, refits its blocks, checks token 0, trains to 100M, and scores the 8K KL | `<name>-run-001/` |
| `summarize` | Login node. Applies both reading rules and refuses runs on different code. Safe to re-run | `summary-001/` |

**What is reused.** `open_task`, `fit_needed`, `menu_prices`, `completed_table_rows`, and
`score_point` from Heterogeneous-1; `configure_strategy`, `desired_requests`,
`segment_schedule`, and `recover_exact_segment` from SwiGLU-8's training path. The configuration
points at Heterogeneous-1's, whose loader refuses a `recovery` block that differs from
SwiGLU-8's, so the recipe is inherited and not copied. The learning rate goes into a copy of the
settings, because `completed_table_rows` compares the original with what the tables recorded.

**No save, load, resume, or checkpoint code.** A run takes under 2 hours, so a failed task is
rerun, and KL needs no saved model. Trained models are not kept; Step 2 restarts from token 0.
The 8K KL is scored in memory. Recovery runs under the same mixed-precision mode as SwiGLU-8's
saved model, the October audit found its BF16 conversion changed perplexity by exactly 0, and the
control's 8K KL checks this against the recorded 0.5433.

**Verification.** `tests/test_heterogeneous2_contracts.py` covers both reading rules and their
boundaries, the stops before training, the control's outcome, the summary, the job's run order,
and the real training loop on a tiny random model with two different blocks inserted, on the CPU.
Before any run, the pinned files, layer plans, checkpoint schedules, and Heterogeneous-1's table
check were confirmed against the local result files (offline check, 2026-11-06). The new path has
not run on a GPU; the control run is what tests it.

## Pre-run resource plan

Recorded 2026-11-06 under
[PERUN resource accounting](../../../infrastructure/perun/perun-resources.md), before any
submission, to be compared with measured use afterward. This is workflow-timing arithmetic, not
scheduler allocation. All tasks use `gpu_short`, one H200, 8 CPUs, 128 GB, and a 4-hour limit.

| Task | Refit | Training | Expected total |
| --- | ---: | ---: | ---: |
| Control, SwiGLU 80% | 42 min | about 55 min | about 1.75 h |
| Forced 20% | 6 min | about 62 min | about 1.2 h |
| Forced 50% | 27 min | about 60 min | about 1.55 h |
| Forced 80%, each of two | 48 min | about 55 min | about 1.85 h |
| SwiGLU 80% at 1e-4 | 42 min | about 55 min | about 1.75 h |

| Field | Value |
| --- | --- |
| Work decomposition | Task 0, then array 1-5, which waits for task 0 with `--dependency=afterok` |
| Parallelism | One task, then up to five at once; tasks do not communicate. PERUN allows 20 running |
| Expected use | About 10 GPU-hours (range 9-12), about 80 CPU core-hours |
| Requested ceiling | 24 GPU-hours |
| Critical path | About 3.6 h: the control, then the longest task, plus queue |
| Peak VRAM | At most about 40 GiB expected; SwiGLU-8 recorded 25-31 GiB. Now recorded per run |
| Peak RAM | At most about 60 GB, an arithmetic estimate as for `assemble`, whose real peak is Unknown. Now recorded per run |
| Storage | PROJECT under 50 MB, SCRATCH at most 5 GB per task, local export a few MB |
| Inputs | Read in place from `perun-results/swiglu-8/` and `perun-results/heterogeneous-1/` |
| Project use so far | Roughly 500 of 8,424 GPU-hours |

**The earlier estimate of about 6.5 GPU-hours** was five runs at 1.3 hours and left out refitting
the blocks. Five runs are about 8.2, and the control adds about 1.75.

**Evidence.** SwiGLU-8 trained 100M tokens in 3,275-3,717 s. Refit times are the recorded times of
each model's exact blocks, with boundary widths estimated from their nearest neighbour.
`assemble` took 0.25, 0.82, and 1.40 h at 20%, 50%, and 80%.

**Confidence:** medium. Training speed varied by about 40% between nodes in Heterogeneous-1
(257-360 s per 10M tokens), so the 4-hour limit leaves margin. `--dependency` has not been used
on PERUN by this project; the fallback is to submit tasks 1-5 by hand after the control.

## Limits

- **20% is not perfectly paired.** Forced 20% replaces layer 4, which SwiGLU-8 keeps dense, so it
  trains 382.5M parameters against 332.2M at equal total size. This may help the forced model.
- **Forced models differ beyond the eight forced layers,** because the knapsack re-solves the rest.
  Total sizes match SwiGLU-8's to within about 4,100 parameters.
- **At 50%,** the SwiGLU-8 baseline differs from the solver's widths at layers 7 and 11, because
  that model kept SwiGLU-7's widths.
- **One seed per model** (decision log, 2026-11-02), and 100M is early. Differences within the
  measured noise are reported as unresolved.
- **The forcing rule picks layers by predicted damage,** so these are the most favourable layers
  by that measure, not the only possible hybrids.

## Steps 2 and 3

Both are conditional, and nothing is built for them.

**Step 2** runs per target at 1B on forced models, with perplexity and tasks, under the rule above.
It needs saving and loading of non-SwiGLU blocks, because `evaluate_final_model` exports and
reloads a bundle and that path is SwiGLU-only.

**Step 3** tests the three retraining scopes, the no-fit arm, and SwiGLU's S7-1 and S7-2 at 80%,
only if 1B still looks competitive, and chosen from those results. See
[SwiGLU-7](../swiglu/swiglu-7.md) for the scopes.

## Sources and attribution

The forced models, the Step 2 rule, and the learning-rate reading are project-defined. The
allocator's prior work is in the
[width-allocation note](../../../methodology/width-allocation-integer-program.md), Section 8.
Numbers belong to their linked experiment pages.
