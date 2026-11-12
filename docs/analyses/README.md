---
title: Deep Analysis 2026-11-12 - Index and Summary
status: draft
created: 2026-11-12
modified: 2026-11-12
---

# Deep analysis of the thesis project, 2026-11-12

An exploratory review of the whole project: direction, experiment design,
result reporting, future options, code, documentation, the agent setup, and
your own thinking as recorded in the session logs. It is an analysis, not a
plan. Nothing in it is a decision, and nothing outside this folder was
changed.

## The answer in one page

1. **The work is rigorous and the thesis is in better shape than the "quantization
   wins" plots suggest.** It has three real results: replacement converges
   to width pruning; the calibration step is a head start whose value falls
   as retraining grows; precision saves bytes, structure saves compute, and
   the two combine. Framed that way, the negative results are the
   contribution. [File 1](01-direction-and-design.md)
2. **On bytes alone, the structural track is finished.** No planned
   experiment changes that. What still has room: compute, a joint choice of
   width and bits, and mixture-of-experts models.
   [File 1](01-direction-and-design.md), Section 1.6
3. **Recovery-2 has probably already answered the 3B question.** At 1B with
   each family's best schedule, dense INT3 is ahead of the stacked model on
   both bytes and KL (0.054 against 0.078). Worth showing the supervisor
   before spending two more sessions on it.
4. **Four cheap checks protect the main claims.** Knapsack against uniform;
   the ranking test of the allocator's additivity; a dense 2-bit baseline,
   which is the missing control for "stacking extends the frontier"; and
   smaller sibling models as anchors. Together about 25-30 GPU-hours.
   [File 3](03-future-work.md)
   Two findings came out of this session's own offline checks on stored
   artifacts. First, the allocator's predicted edge over uniform widths is
   modest: 27% at 20% removal, only 9-13% from 50% upward, before any
   retraining. Second, in your interaction study damage compounds between
   *neighbouring* layers and fades with distance, and three specific pairs
   (layer 22 with layers 1, 5, or 7) collapse the model; the "distance 17"
   reading in your sessions does not hold, and the allocator narrows layer
   22 only at 75% and 80%, the two targets with the unexplained
   fit-versus-slice jump. [File 3](03-future-work.md), idea A
5. **The sibling anchor may be uncomfortable and is worth it.** By its model
   card, SmolLM2-360M would match or beat every compressed model beyond
   about 56% saving on task accuracy. That sets the range in which
   compressing the 1.7B model makes sense.
   [File 2](02-reporting-and-speed.md), Section 2.4
6. **On speed, you and your supervisor are both right.** Quantization helps
   when memory traffic limits speed; narrower layers help when arithmetic
   does. You already report the two quantities that bound each. A measured
   comparison is safe only between dense and SwiGLU, both BF16.
   [File 2](02-reporting-and-speed.md), Section 2.7
7. **Mixed operator types failed on a dense model for a structural reason,
   and the same reason says where they should work:** a mixture-of-experts
   model, where blocks are used unequally. A small version of that chapter
   needs no retraining and is directly comparable with the MoNE paper.
   [File 3](03-future-work.md), Section 3.2
8. **The code is correct as far as I read it, and lean at its core.** The
   bloat is concentrated: copied launcher blocks, 4,293 lines of ports that
   never ran, and shared helpers trapped inside finished experiment files.
   [File 4](04-implementation.md)
9. **The documentation is thorough and has started to cost you.** One fact
   lives in five places that disagree; the decision log is 31,000 tokens
   read at every session start; the closest prior work is still
   unregistered; nothing yet links results to thesis text.
   [File 5](05-documentation-and-agents.md)
10. **Your session logs hold the history the repository lacks:** how the
    method developed, ideas that were never run, several open items that
    dropped out of handoffs, and a few claims to re-check before they are
    written into the thesis. The reasons behind the settings are mostly in
    the docs already; one reason and two attributions are missing.
    [File 6](06-session-findings.md)

## Files

| File | Covers | Your topics |
| --- | --- | --- |
| [01-direction-and-design.md](01-direction-and-design.md) | What has been established; what the thesis is; strengths and gaps of the design; the experiment classes; whether it is a dead end | 1.1, 1.3; summary items 1-3 |
| [02-reporting-and-speed.md](02-reporting-and-speed.md) | A fair reporting frame: three axes, three retraining budgets, same-class baselines, sibling anchors; the speed question | 7, 8 |
| [03-future-work.md](03-future-work.md) | Heterogeneous and mixture-of-experts plans; nine new ideas ranked; old ideas from your sessions; what to skip; a possible order | 3, 4, 5 |
| [04-implementation.md](04-implementation.md) | Correctness, leanness, structure, tests, efficiency | 1.2, 2.1-2.3 |
| [05-documentation-and-agents.md](05-documentation-and-agents.md) | The decision log, experiment pages, the wiki, a claims ledger, `AGENTS.md`, grilling, memory | 2.4, 2.5; summary item 7 |
| [06-session-findings.md](06-session-findings.md) | What 91 sessions hold that the repository does not | 6; summary item 5 |
| [07-questions.md](07-questions.md) | Twelve questions with recommended answers, in three rounds | The grilling you left to me |
| `private/` | The six session digests with dated quotes, and the scripts used. The folder ignores itself in git | Raw material for file 6 |

**Suggested reading order.** This page, then file 1, then file 7. Files 2
and 3 when planning; files 4 and 5 before the next refactor or cleanup
session; file 6 when writing.

## What was done, and what was not

| Done | Detail |
| --- | --- |
| Read in full | `annotation.md`, `decisions.md`, the wiki index, both results snapshots, the four meeting notes, the width-allocation note, `dependencies.md`, `core-eval.md`, `grilling.md`, the results sections of Heterogeneous-1, Recovery-2, and the SwiGLU progression, the MoNE source summary, four memory notes |
| Read in full (code) | About 3,900 lines: listed in [file 4](04-implementation.md), Section 4.1 |
| Ran | A structural scan of all Python and launcher files; the unit-test suite locally (237 pass, 10 skipped; not the pinned environment); extraction and filtering of 91 session logs; two offline calculations on stored artifacts (block interaction; SwiGLU-8 preparation) |
| Delegated | Six sub-agents read the filtered session text and wrote digests. I read all six and spot-checked nine quotes |
| Checked on the web | Five facts at abstract or documentation level; each is marked where it is used |

| Not done | Consequence |
| --- | --- |
| No GPU run and no PERUN access; two result artifacts opened (block interaction; SwiGLU-8 preparation) | Apart from those two offline calculations, every number comes from your docs or is arithmetic on them. Nothing here is a new measurement |
| The large runners, the notebooks, and several `src/` modules were not read | No correctness claim about them |
| Most literature is recalled, not re-read | Each such item is marked "recalled"; verify before citing |
| No file outside `docs/analyses/` was changed; nothing was committed | The pending corrections named in files 5 and 7 are still pending |

## Notes on how this was produced

- **Who wrote it.** You preferred that report writing go to a simpler model.
  I wrote the seven analysis files myself. The judgment in them is in the
  wording, and a writer starting cold would have needed the whole context
  sent again. The six session digests in `private/` were written by Opus
  sub-agents, which is the part that suited delegation.
- **Cost.** The first attempt at session mining (nine sub-agents on full
  transcripts) ran into the weekly limit and produced nothing. The second
  (six sub-agents on text filtered by a script) finished at about one
  million tokens. The lesson is in [file 5](05-documentation-and-agents.md),
  Section 5.6.
- **Equations.** The one formula in this analysis (byte saving at a given
  bit-width, [file 1](01-direction-and-design.md), Section 1.6) is standard
  bookkeeping and needs no citation. The "repair factor" discount in
  [file 3](03-future-work.md), idea D is a project-proposed heuristic.

## Where praise is due

- The reproduction gates. Bit-exact agreement with an earlier run before
  every new run is why this project's numbers can be trusted.
- Finding and fixing two mismatches in a widely used quantization library,
  and re-running everything they touched.
- Writing down a rule before the result, and keeping it when the result was
  inconvenient.
- Predicting the heterogeneous outcome, accepting it in advance, and
  designing the degradation curve so that a "no" would still say something.
- Asking, repeatedly and for months, whether the comparison is fair to the
  method that is losing.

## Suggested next step

This is a phase boundary. I suggest a fresh session that starts from
[file 7](07-questions.md), takes Q1 and Q5 first, and turns whatever you
decide into decision rows. The corrections that have been waiting since
2026-11-06 (Q9) and the Heterogeneous-2 write-up do not depend on any of
those answers and can be done first.
