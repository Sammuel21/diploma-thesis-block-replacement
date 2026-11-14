---
title: Deep Analysis Part 2 - The 2026-11-13 Meeting
status: draft
created: 2026-11-13
modified: 2026-11-14
---

# 1. The 2026-11-13 meeting: record, readings, and proposed decisions

Part of the [deep analysis, part 2](README.md). It covers your topic 1.
Evidence labels and flags are defined in the [README](README.md#labels-and-flags).

I was not at the meeting. Everything about what was said comes from your
digest. A meeting page in the style of the earlier ones is drafted at
[drafts/2026-11-13-supervisor-feedback.md](drafts/2026-11-13-supervisor-feedback.md),
ready to move to `docs/meetings/` once you have corrected it.

## Short answer

- **Your three decision ideas are sound, and I agree with all three.**
- **Your reading of the local-fitting suggestion is correct.** Stored data
  already answers most of it: local error can choose the operator inside a
  layer and cannot choose where to compress ([file 2](02-local-fit-as-selector.md)).
- **The knapsack will not explode on a mixture-of-experts model. The price
  table will.** The solve stays small with thousands of experts; measuring a
  KL for every expert and every menu item does not ([file 3](03-moe.md)).
- **One point from the meeting conflicts with the decision log** (speed
  comparisons between methods), and **four of the snapshot's five questions
  are not answered in your digest.** Both are listed below for you to settle.

## 1.1 Point by point

| # | Topic | My reading | Flags |
| --- | --- | --- | --- |
| 1.1 | No 3B run | Agreed, and it matches Part 1's estimate. One loose end: the two decay results are selection-stream scores from runs that kept no model, so they have no perplexity or task accuracy and cannot be drawn on the frontier plots (1.3 below) | DECISIVE (closes a branch); MODEL-CONCRETE numbers |
| 1.2 | Heterogeneous collapses into SwiGLU | Agreed. It is a property of this setting: a dense model, where every block sees every token, and a substitute that starts from the teacher's own neurons. A menu priced from local error would have picked plain MLPs at 70-90% removal only, at a 1-5% predicted cost, which fits the level result at 80% | Closed negative result; DENSE-ONLY |
| 1.3 | Local fit as a first filter | You understood it correctly. It is three questions in one (filter, allocator, hypothesis), and the answers differ: yes as a filter within a layer, no as an allocator between layers, yes again once each layer gets one measured KL | DECISIVE for the annotation; TRANSFERABLE HYPOTHESIS |
| 1.4 | Mixture-of-experts | The right place for the mixed menu, as Part 1 argued. Since then a literature check found a crowded field; what still looks open is narrow and specific | New result possible; MOE-ONLY |
| 1.5 | vLLM and a speed frontier | Feasible for part of the comparison, not all of it: [file 6](06-reporting-model.md) | PROTECTIVE for the fairness of the comparison |
| 1.6 | Other model families | Last, as you say. [File 5](05-agnosticism-and-factors.md) lists which claims are worth replicating and which are settings | Generality |
| 1.7 | More operator families, picked by local fit with several losses (added 2026-11-14) | Not for a better dense point: nothing in Heterogeneous-1 suggests one. Worth it as test items for the local-fit selector and as the MoE menu. On MoE a dense linear map costs 48-100% of one expert, so linear items must be low-rank. [File 8](08-operators-losses-moe-model.md) | NEUTRAL for the dense frontier; PROTECTIVE for the local-fit claim; enabling for MoE |

### On 1.7: the point the first digest left out

You added this on 2026-11-14: hybrid operators were discussed, and only zero,
mean, linear, plain MLP, and the two linear hybrids have been tested. Three
short answers; the detail is in [file 8](08-operators-losses-moe-model.md).

- **Which operators:** low-rank linear, a slice with a low-rank correction, a
  diagonal map, a conditional constant, and on MoE redirecting or sharing
  experts. KAN, RBF, and structured matrices have no way in (8.2-8.3).
- **Which losses:** inside a layer MSE already picks like KL in 163 of 176
  cells, so new scores have little to gain there. The room is between layers
  and between experts. Two scores could work there without a probe: error
  relative to the residual stream, and a gradient-weighted error (8.5).
- **Which MoE model:** OLMoE-1B-7B. Per token it runs 8 × 1,024 = 8,192
  neurons at hidden size 2,048, the shape of SmolLM2's MLP (8.7).

### On 1.3: your question, answered directly

*"He wants to test how comparable local-fit selection is to the KL knapsack.
Am I understanding it correctly?"* Yes. And your objection in the meeting was
right too: the stored data contains exactly that case, 13 times in 176 (a
plain MLP wins on local error and loses on KL), each time at a small cost.

The part worth bringing back to him is that "instead of the knapsack" has two
meanings with opposite answers **[ran: file 2]**:

- *Instead of pricing every operator by KL* (a filter): works.
- *Instead of pricing layers by KL* (the allocation): does not. An allocation
  priced by local error alone is predicted to be worse than uniform widths.

### On 1.4: where the cost really is

| Quantity | Dense SmolLM2 | OLMoE (16 layers × 64 experts) | Qwen3-30B (48 × 128) |
| --- | ---: | ---: | ---: |
| Blocks to allocate | 22 | 1,024 | 6,144 |
| Solver work, 5 menu items, budget in eighths of an expert | trivial | about 42 million table updates | about 1.5 billion |
| KL measurements for a full price table, 5 items | 110 | 5,120 | 30,720 |
| With one measured KL per expert plus local error | 22 | 1,024 | 6,144 |

Expert counts are from the model configurations **[lit: agent-read]**; the
rest is arithmetic **[calc]**.

- **The solver is not the problem.** A multiple-choice knapsack grows linearly
  in the number of blocks. One implementation note: your current solver
  stores the full list of choices in every table state
  ([allocation.py](../../../src/mlp_replacement/compression/allocation.py),
  lines 202-230), which is fine for 22 layers and too slow for a thousand
  experts. Back-pointers fix that; it is a small change **[read]**.
- **MoNE does not shrink the problem.** A novice is one menu item ("replace by
  a constant"), not a smaller set of experts. Every expert still chooses.
- **Your instinct about the residual error is the right one.** Local error
  inside a layer, plus a small number of model-level measurements to set each
  layer's scale, is exactly what the dense data supports. Whether the scale
  can be shared by all experts of a layer is the open question that decides
  the cost ([file 3](03-moe.md), section 3.3).

## 1.2 Conflicts with the decision log

AGENTS.md asks me to point these out and not choose silently.

| Meeting point | Existing row | Conflict | What settles it |
| --- | --- | --- | --- |
| A speed frontier comparing experiment classes in vLLM | 2026-10-25: "no speed frontier, and no speed comparisons between methods. No kernel or engine work"; 2026-11-01: speed supplement is linear FLOPs only | Direct. The 2026-10-25 row was itself a supervisor recommendation | A new row that supersedes or narrows it; options in [file 6](06-reporting-model.md), section 6.4 |
| No 3B run | 2026-11-07 (past-1B hypothesis), 2026-11-08 (3B schedules), the 3B sentences of the three 2026-11-11 rows | Superseded by the meeting | Proposed row 1 |
| MoE as planned work | 2026-10-16: "a bonus if time allows" | Only if you now treat it as planned | Proposed row 3, if you agree |
| Heterogeneous-2 Step 2 | 2026-11-06: a level target goes to 1B | Only if Step 2 is dropped | Your answer to question 1 below |
| Return to local fitting | 2026-11-07: "Local fitting is not reopened for a new schedule" | None. That row is about schedules; this is a different question | Proposed row 2 makes the boundary explicit |
| New operator families on dense SmolLM2 (added 2026-11-14) | 2026-11-13: heterogeneous replacement on dense SmolLM2 is closed | Only if they are meant as frontier candidates | Treat them as test items of the local-fit study; [file 8](08-operators-losses-moe-model.md), question 1 |
| Low-rank linear items (added 2026-11-14) | 2026-11-04: "Low-rank linear is excluded" | Direct. New evidence: on MoE the dense map is not affordable | A row, if you agree; file 8, question 2 |

## 1.3 Proposed decision rows

**Update, 2026-11-13, later the same day.** You approved these, and six rows
dated 2026-11-13 are now in the [decision log](../../agents/decisions.md):
rows 1-3 below, row 4 as option 2 of [file 6](06-reporting-model.md) (speed
for BF16 models only), a row closing Heterogeneous-2 Steps 2 and 3 (question 1
of section 1.4), and a row for the decay reruns with kept models (option B
below). The rows they supersede or narrow are marked in the log. Rows 3, 4,
and the heterogeneous row rest on your reading of the meeting; strike them
before committing if the supervisor meant otherwise.

The text below is the proposal as first written.

| # | Proposed row (2026-11-13) | Reason / source |
| --- | --- | --- |
| 1 | The 3B comparison is not run; Stacking-2 and Quantization-3 are cancelled. Near 51% saving the thesis reports the 1B results with each family's best schedule (Recovery-2): dense INT3 QAT 0.0542 at 51.6% against B 20% INT4 0.0778 at 50.9%, on the selection stream. | Supervisor meeting 2026-11-13. Supersedes the 2026-11-07 past-1B row and the 2026-11-08 3B-schedule row |
| 2 | The annotation's locality hypothesis is tested directly: how well the local fit on calibration data predicts damage at the model output, and whether it can select operators or allocate widths in place of measured KL. This does not reopen the closed question of whether fitting still matters after retraining. The design is settled in its own session. | Supervisor suggestion, 2026-11-13 |
| 3 | Mixture-of-experts work is planned, not only a bonus: the SwiGLU workflow and the mixed operator menu are tested on at least one MoE model. Model, scope, and baselines are settled in a design session. | Supervisor meeting 2026-11-13. Would supersede the 2026-10-16 MoE bonus row |
| 4 | (One of the options in file 6, section 6.4, once chosen.) | Resolves the speed conflict above |

**A loose end under row 1.** By the 2026-11-07 rerun row, a family that
adopts a schedule gets a matched rerun "where the frontier is decided",
including the pair near 51%. Recovery-2 kept no model. So today the thesis has
the decay result as a selection-stream KL only; the frontier tables still show
the constant-rate models (0.0709 and 0.0770 on the final stream). Two honest
ways to close this:

| Option | Cost (rough) | Result |
| --- | --- | --- |
| A. Report as is | none | The size frontier keeps the constant-rate models; the decay result is a separate table with its own metric |
| B. Rerun the two 1B decays and keep the models | about 25-30 GPU-h, from Recovery-2's measured speeds | The decay models get perplexity and task accuracy and replace the constant-rate points near 51% |

My recommendation is B, because the sentence "dense INT3 wins near 51%" will
be a headline and should rest on the same metrics as the rest of the frontier
**[view]**. It is your call; the rerun row already allows it.

## 1.4 Questions only you can answer

These are the snapshot's own questions that your digest does not settle.

| # | Question | My recommended answer |
| --- | --- | --- |
| 1 | Is Heterogeneous-2 Step 2 (80% at 1B) dropped? | Yes. A tie cannot give a better SwiGLU-alone point, and the block-level evidence is decisive. Needs a row that supersedes the 2026-11-06 Step 2 rule |
| 2 | Is the peak learning-rate screen (1e-4) wanted? | Yes, small: SwiGLU alone at 20/50/80%, 100M tokens. It is the one known lever on the "blue dot" (+9.8% at 80%), and dense INT3 has already received a schedule worth 35% |
| 3 | Was the evaluation-thesis framing discussed? | If not, raise it next time. The return to the annotation's hypothesis fits it well |
| 4 | Weekly meetings, and the next date? | - |
| 5 | Is the MLP-first retry of the stalled hybrid wanted? | Optional. It answers a narrow question at block level; I would not let it delay the local-fit study |

Added 2026-11-14: six further questions on operators, losses, and the MoE
model are in [file 8](08-operators-losses-moe-model.md), section 8.8.

## 1.5 What changes in the order of work

A suggestion for discussion, replacing Part 1's section 3.6 where the meeting
overtook it **[view]**:

1. **No GPU:** correct and file the meeting page; decide the rows above; post
   the TorchAO issue ([file 7](07-torchao-bug.md)).
2. **The local-fit study, steps 1-2** ([file 2](02-local-fit-as-selector.md),
   section 2.8). It answers the supervisor's question, the annotation's
   hypothesis, and Part 1's two allocator checks with one small set of runs.
3. **The decay reruns with kept models,** if you choose option B above.
4. **MoE step M0-M1** ([file 3](03-moe.md)), after a design session.
5. **Speed, as a contained bonus** ([file 6](06-reporting-model.md)).
6. **Other models last.**

Part 1's remaining protective checks (sibling-model anchors, the dense 2-bit
baseline) are unchanged by the meeting and still cheap.
