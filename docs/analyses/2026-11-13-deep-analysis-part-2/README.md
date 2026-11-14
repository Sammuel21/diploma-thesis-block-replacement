---
title: Deep Analysis Part 2, 2026-11-13 - Index and Summary
status: draft
created: 2026-11-13
modified: 2026-11-14
---

# Deep analysis of the thesis project, part 2 (2026-11-13)

A continuation of the [deep analysis of 2026-11-12](../2026-11-12-deep-analysis/README.md),
written after the supervisor meeting of 2026-11-13. It extends Part 1 and
does not redo it; the [duplication table](#what-part-1-already-covers) says
what is where. It is an analysis, not a plan. Nothing in it is a decision by
itself.

**Update, later the same day.** After you approved them, six decision rows
dated 2026-11-13 were added to the [decision log](../../agents/decisions.md);
[file 1](01-meeting-3.md), section 1.3, lists them. Apart from that and one
index row in `docs/analyses/README.md`, nothing outside this folder was
changed.

**Update, 2026-11-14.** [File 8](08-operators-losses-moe-model.md) was added
for a meeting point the first digest left out: more operator families, more
local losses, and an MoE model that matches SmolLM2. It is point 11 below
and point 1.7 of [file 1](01-meeting-3.md). Nothing was run for it and no
decision row was added.

## The answer in one page

1. **The supervisor's question about local fitting can largely be answered
   from data you already have.** Inside a layer, local error ranks options
   almost exactly like measured KL (rank correlation 0.995) and picks the same
   operator in 163 of 176 cases. Between layers it is useless: a width
   allocation priced by local error alone is predicted to be worse than
   uniform widths. [File 2](02-local-fit-as-selector.md)
2. **One measured KL per layer repairs it.** Damage is proportional to local
   error within a layer, with a different constant per layer. With 22
   measurements instead of 176, local error reproduces the allocator's
   choices to within 1-4% of predicted damage. This is what makes a
   thousand-expert model affordable. [File 2](02-local-fit-as-selector.md),
   [file 3](03-moe.md)
3. **Three layers (1, 7, and 22) write outputs hundreds of times larger than
   their neighbours.** This is why normalised local error misleads there. It
   may also explain two older puzzles: the monitoring perplexity that mostly
   measures position 1, and the layer pairs that collapse the model. That
   link is a hypothesis with a five-minute check. [File 2](02-local-fit-as-selector.md),
   section 2.4
4. **Your three decision ideas are sound.** One loose end under "no 3B run":
   the decay results near 51% come from runs that kept no model, so they have
   no perplexity or task accuracy yet. One conflict: a speed frontier across
   methods contradicts the 2026-10-25 decision row. [File 1](01-meeting-3.md)
5. **The knapsack does not explode on a mixture-of-experts model; the price
   table does.** The literature there is crowded and fast. Constant
   replacement, a global one-choice-per-expert program, and narrower experts
   are all published. A measured cost per option inside an exact allocation
   over a mixed menu was not found, and neither was any measurement of
   whether expert damage adds up. [File 3](03-moe.md)
6. **A design made from the annotation alone converges on yours.** It differs
   in two useful places: it would fit each block on the inputs it really
   sees after earlier replacements (the direct test of "isolated function"),
   and it would evaluate on text outside the calibration domain.
   [File 4](04-isolated-design.md)
7. **With one model nothing is shown to be model-independent, but claims can
   be sorted by why they should transfer.** Four levels, applied to fourteen
   of your claims, and a scorecard form for the "other models" bonus.
   [File 5](05-agnosticism-and-factors.md)
8. **Your own results rank what matters.** What is removed and the order of
   the stages change results by 1.3 to 10 times and persist. How things are
   chosen and initialised matters a lot without retraining and a few percent
   after 1B tokens. The retraining budget is the hidden axis.
   [File 5](05-agnosticism-and-factors.md)
9. **Speed is two numbers that rank the families in opposite orders.** In
   vLLM, dense against narrower BF16 models is low risk; INT4 is moderate
   work; INT3 depends on a kernel library three weeks old.
   [File 6](06-reporting-model.md)
10. **The TorchAO defect is real, still live on yesterday's upstream main,
    and apparently unreported.** Worth an issue now; a pull request is worth
    offering. [File 7](07-torchao-bug.md)
11. **More operators will not move the dense frontier, and are still worth
    adding** (added 2026-11-14): as test items for the local-fit selector and
    as the MoE menu. On MoE a dense linear map costs 48-100% of one expert,
    so linear items must be low-rank. New local scores have little to gain
    inside a layer and much between layers and experts. OLMoE-1B-7B runs,
    per token, an MLP of exactly SmolLM2's shape.
    [File 8](08-operators-losses-moe-model.md)

## Files

| File | Covers | Your topics |
| --- | --- | --- |
| [01-meeting-3.md](01-meeting-3.md) | Readings of each meeting point, conflicts with the decision log, proposed rows, questions for you | 1 |
| [02-local-fit-as-selector.md](02-local-fit-as-selector.md) | The offline study of local error against measured KL; a design to finish the question | 1.3, 2 |
| [03-moe.md](03-moe.md) | Prior work, what is open, a price ladder, steps, model choice | 1.4, 2 |
| [04-isolated-design.md](04-isolated-design.md) | A second design from the annotation alone, compared with yours | 3 |
| [05-agnosticism-and-factors.md](05-agnosticism-and-factors.md) | What transfers; what separates a method from tuning | 4, 6, 1.6 |
| [06-reporting-model.md](06-reporting-model.md) | What to track, three axes, measured speed, vLLM | 5, 1.5 |
| [07-torchao-bug.md](07-torchao-bug.md) | Independent verification and a recommendation | 7 |
| [08-operators-losses-moe-model.md](08-operators-losses-moe-model.md) | Added 2026-11-14: which further operators have a way in, local losses as scores and as objectives, which MoE model matches SmolLM2 | 1.7, 2 |
| [drafts/2026-11-13-supervisor-feedback.md](drafts/2026-11-13-supervisor-feedback.md) | A meeting page in the style of `docs/meetings/`, for you to correct and move | 1 |
| `figures/` | Two figures for file 2 | - |
| `private/` | The scripts that produce every number of file 2 and file 7, the 2026-11-01 TorchAO issue draft, and three raw research reports from sub-agents with their links. The folder ignores itself in git, as in Part 1 | - |

**Suggested reading order.** This page, file 1, file 2. Then file 3 before
any MoE design, file 6 before answering the supervisor on speed, file 7
before posting the issue. Files 4 and 5 when planning and when writing.

## Labels and flags

**Evidence labels**, as in Part 1, with one addition:

| Label | Meaning |
| --- | --- |
| **[read]** | I read the cited file in this session |
| **[ran]** | A script or command executed in this session |
| **[calc]** | Arithmetic on numbers from the docs or from agent-read sources |
| **[lit]** | Literature; verify before citing |
| **[agent]** | Read on the web by a research sub-agent on 2026-11-13, through a tool that summarises pages. Not read by me. A discovery aid only |
| **[view]** | My judgment |

**Priority flags (return on effort):**

| Flag | Meaning |
| --- | --- |
| DECISIVE | Answers a thesis question or changes a headline sentence |
| NEW RESULT / NEW AXIS | Adds something the thesis cannot say today |
| PROTECTIVE | Defends a claim you already make |
| INCREMENTAL | A small, measurable gain |
| NEUTRAL | No meaningful change expected or observed |
| COSTLY | Poor return for the effort |

**Agnosticism flags** (defined in [file 5](05-agnosticism-and-factors.md)):
STRUCTURAL, TRANSFERABLE HYPOTHESIS, FAMILY-SPECIFIC (DENSE-ONLY or
MOE-ONLY), MODEL-CONCRETE.

## Priority queue

Everything this analysis proposes, in the order I would take it **[view]**.
Costs are rough scoping numbers from throughput in your docs, not PERUN
resource plans.

| # | Item | Priority | Transfers? | Rough cost | Where |
| --- | --- | --- | --- | --- | --- |
| 1 | Settle the proposed rows and open questions | DECISIVE (unblocks the rest) | - | A conversation | File 1 |
| 2 | Post the TorchAO issue | INCREMENTAL | - | 1 hour | File 7 |
| 3 | Local-fit study: joint KL of about 15 allocations at token 0, then 100M retraining of four | DECISIVE | TRANSFERABLE HYPOTHESIS | 8-15 GPU-h | File 2, 2.8 |
| 4 | Fit blocks on their real inputs against today's parallel fit, no retraining | DECISIVE without retraining; probably NEUTRAL after 1B | TRANSFERABLE HYPOTHESIS | A few GPU-h | File 4, 4.4 |
| 5 | Per-position size of the MLP output at layers 1, 7, 22 | Explanation | MODEL-CONCRETE | Minutes | File 2, 2.4 |
| 6 | Top-1 agreement and an out-of-domain KL on kept models | PROTECTIVE | - | Under 1 GPU-h | File 4, 4.4 |
| 7 | Rerun the two 1B decays near 51% and keep the models | PROTECTIVE | MODEL-CONCRETE | 25-30 GPU-h | File 1, 1.3 |
| 8 | MoE steps M0-M2 on one model | NEW RESULT, with novelty risk | MOE-ONLY | Unknown until one KL pass is timed | File 3 |
| 9 | A gradient-weighted local error | INCREMENTAL on the dense model; enabling for MoE | TRANSFERABLE HYPOTHESIS | New code, a few GPU-h | File 2, 2.8 |
| 10 | Measured speed, tier A (dense against narrower BF16 models) | NEW AXIS | Engine-specific numbers | About a week of work, few GPU-h | File 6 |
| 11 | Peak learning-rate screen for SwiGLU alone | INCREMENTAL (fairness) | MODEL-CONCRETE | 4-14 GPU-h | Part 1, idea F |
| 12 | Transfer scorecard on a second model | Generality | That is its purpose | Last | File 5 |

Added 2026-11-14, belonging with item 3 and not a separate experiment: more
local scores and a small set of closed-form operators recorded in the
local-fit study (PROTECTIVE on the dense model, enabling for MoE; about 8
GPU-hours for six layers, rough; [file 8](08-operators-losses-moe-model.md),
section 8.6).

Unchanged from Part 1 and still cheap: sibling-model anchors (PROTECTIVE),
the dense 2-bit baseline (PROTECTIVE).

What I would not spend on **[view]**: Heterogeneous-2 Step 2 at 1B
(NEUTRAL), further operator families on the dense model as candidates for
the frontier (NEUTRAL; file 8 keeps a few as test items only), KAN or RBF
blocks (COSTLY), INT3 in a serving engine (COSTLY), published pruning
methods on your model (COSTLY).

## What Part 1 already covers

| Topic | Part 1 | What Part 2 adds |
| --- | --- | --- |
| The 3B question | File 1 section 1.6, Q5: recommended against | Decided at the meeting; only the kept-model loose end is new |
| Heterogeneous on a dense model | File 3 section 3.1, Q7 | One number: what a menu priced by local error would have picked |
| Allocator checks (knapsack against uniform; ranking test) | File 3 idea A, Q6 | Folded into the local-fit study, which runs the same models for a second reason |
| Local fitting | File 1: "closed"; file 3: do not reopen | **A different question.** Part 1 closed "does fitting still matter after retraining". The supervisor asked "does local fit predict global damage". Part 1 did not propose that study for the dense model; it proposed measured-against-proxy only for MoE |
| Mixture-of-experts | File 3 section 3.2, Q8: steps M1-M3 | A literature check, the pricing cost and a way round it, a model table |
| Reporting and speed | File 2: three axes, three budgets, anchors, two regimes, options A-C | A savings ledger, how to draw three axes, current engine facts, options for the log conflict |
| The TorchAO report | File 3 section 3.4, Q12: "do it" | An independent reproduction on current upstream, and three updates for the draft |
| The layer pairs that collapse the model | File 3 idea A | A candidate cause (hypothesis) |
| Sibling anchors, dense 2-bit, what retraining repairs, nested widths, code, documentation, session history | Files 2-6 | Nothing; not revisited |
| Model agnosticism, differentiating factors, a design from the annotation alone | Not covered | Files 4 and 5 |

## What was done, and what was not

| Done | Detail |
| --- | --- |
| Read in full | `annotation.md`, `decisions.md`, Part 1 (README and files 1, 2, 3, 7), the results snapshots of 2026-11-01 and 2026-11-13, the 2026-11-06 meeting page, the reporting framework, the backlog, `allocation.py`, the 2026-11-01 issue draft |
| Ran, offline | `private/tools/local_fit_vs_kl.py` and `private/tools/make_figures.py` on stored artifacts of SwiGLU-7, SwiGLU-8, and Heterogeneous-1 (no GPU; the repository's own solver was loaded, not re-implemented) |
| Ran, TorchAO | Two scripts on CPU against torchao 0.18.0 and the upstream main source of 2026-11-12, in a temporary environment outside the project; five searches of the upstream tracker |
| Delegated | Four sub-agents on the same model: one design from the annotation alone, three web research tasks. The three research reports are saved in `private/agent-reports/` |
| Added 2026-11-14 (file 8) | Read `heterogeneous-1.md`, `operator.md`, `operators/modules.py`, the backlog, and the hybrid wiki page; eleven web searches whose summaries were read, not the pages; arithmetic on parameter counts. No script and no GPU run |

| Not done | Consequence |
| --- | --- |
| No GPU run, no PERUN access | Every result here is a single-block, before-retraining quantity or arithmetic on your docs |
| Web sources were read by sub-agents through a summarising tool, not by me | Every **[agent]** fact needs a direct read before it is cited. Many of those papers are from 2026 |
| The design "from the annotation alone" was not blind | The sub-agent saw one-line memory notes about the project ([file 4](04-isolated-design.md), 4.1). Its report was not saved to a file |
| Heterogeneous-1 and Heterogeneous-2 pages, the width-allocation note, and the wiki were not re-read | Their numbers are taken from Part 1 and the snapshots |
| I was not at the meeting | File 1 and the meeting draft rest on your digest |
| No file outside `docs/analyses/` was changed; nothing was committed | The proposed rows and the meeting page are drafts |

The session was interrupted once by the usage limit; three research agents
were resumed afterwards and finished.

## Equations and figures

- One equation, in [file 2](02-local-fit-as-selector.md), section 2.6: the
  second-order approximation of KL. It is standard and needs a citation; its
  form for an activation is my composition and is labelled an explanatory
  formalization.
- The "differentiating" test in [file 5](05-agnosticism-and-factors.md) and
  the three MoE hypotheses in [file 3](03-moe.md) are project-proposed.
- Two formulas in [file 8](08-operators-losses-moe-model.md), section 8.5: a
  score relative to the residual stream (project-proposed) and a
  gradient-weighted estimate of KL (the standard sampling estimate of the
  form in file 2; derived in that session, not run).
- Two figures, drawn from stored artifacts by `private/tools/make_figures.py`.

## Where praise is due

- **Your artifacts made the main finding possible.** Every local fit stored
  its held-out error next to the measured KL, for every layer, width, and
  operator. A question the supervisor raised on Friday could be answered from
  files written weeks earlier, at no cost.
- **Your objection in the meeting was correct,** and you raised it before
  seeing any data: an operator can win locally and lose globally.
- **Your remark on unstructured pruning is the right fairness rule,** and it
  generalises into the ledger of file 6.
- **Asking whether anything here is model-independent,** while expecting the
  answer to be uncomfortable.

## Suggested next step

This is a phase boundary. I suggest a fresh session that starts from
[file 1](01-meeting-3.md), sections 1.3 and 1.4: settle the rows and the open
questions, correct and move the meeting page, and post the TorchAO issue.
Then one design session for the local-fit study, which answers the
supervisor's question, the annotation's hypothesis, and Part 1's two
allocator checks with one small set of runs.
