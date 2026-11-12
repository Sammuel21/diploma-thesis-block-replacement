---
title: Deep Analysis 2026-11-12 - What the Session Logs Hold
status: draft
created: 2026-11-12
modified: 2026-11-12
---

# 6. What your session logs hold that the repository does not

Part of the [deep analysis of 2026-11-12](README.md). It covers your topic 6
and summary item 5: your own thinking across the Codex and Claude sessions.
Evidence labels are defined in [file 1](01-direction-and-design.md).

## Short answer

- **It was worth doing, though the gap is smaller than I first wrote.** The
  sessions hold how the method developed, what was considered and dropped,
  and where ideas came from. The reasons behind the settings themselves are
  mostly recorded already: of ten I checked, six are in the docs, three
  partly, and one is missing (Section 6.4, corrected on 2026-11-12).
- **Three things matter most for the thesis:**
  1. One missing reason (why the first and last layers are protected) and
     two missing attributions (Section 6.4).
  2. Items that were deferred and then silently dropped from every handoff
     (Section 6.5). The structured-pruning baseline is the main one.
  3. Claims you relied on that deserve a second look before they are
     written into the thesis (Section 6.6).
- **Your early instincts were mostly right, and several came true.** You
  proposed per-layer operator choice in March, budget allocation in August,
  stacking in July, and you predicted the heterogeneous result before it
  ran.

## 6.1 What was done

| Step | Detail | Label |
| --- | --- | --- |
| Sources | 53 earlier Claude Code sessions (2026-10-06 to 2026-11-11) and 38 Codex sessions for this project (2026-03-20 to 2026-10-13): 91 in all, of which 82 held substantive typed prompts. Codex sessions of other projects were identified by their working folder and not read | **[ran]** |
| Extraction | A script kept your typed prompts, cut pasted logs, and dropped tool output. Automated review sessions and exact duplicates were removed | **[ran]** |
| Filtering | A second pass kept only prompts with at least 250 characters of your own text: 1.3 MB from 3.9 MB | **[ran]** |
| Reading | Six sub-agents (Opus) each read one time slice in full and wrote a structured digest with dated quotes | Delegated |
| Checking | I read all six digests. I checked nine key quotes against the extracted text by search; all nine were found as quoted | **[read]**, **[ran]** |

**Limits.** The digests are summaries made by a model. They are a discovery
aid, not evidence. Many first prompts in the Claude sessions are handoff
texts drafted by an earlier agent session and pasted by you; they carry your
approval, not necessarily your wording. Pasted material was cut, so some
answers you reacted to are not visible. "Not found in docs" means a search
for one or two key terms found nothing, not that the idea is certainly
absent.

## 6.2 The arc, in eight steps

| When | What happened | The thought that moved it |
| --- | --- | --- |
| March-April | A one-block linear proof of concept grew into the MVP: one-shot against iterative, importance-guided against random selection, linear against small MLP | "A linear layer should be bad for important blocks and good for unimportant ones." The MVP then contradicted its own importance score |
| June | First framing: block level, model level, deployment scenarios. The supervisor cut deployment and latency and suggested the mixture-of-experts addition | Scope set to parameters, size, and memory |
| July | Consolidation: MVP frozen, wiki started, taxonomy of compression approaches | "Compression itself could be formulated as an operator": replacement, then factorization, then quantization |
| August | Bottom-up block studies. "Replace k blocks" became "one global target, split into per-block budgets". The baseline was found to have cut width on the wrong dimension | Budget allocation separated from the choice of operator |
| Early September | Local fitting stalled at NMSE 0.48; fixing epochs instead of updates and starting from teacher neurons brought it to 0.20 | "Operator capacity is not the problem; the training process is" |
| Mid to late September | SwiGLU-2 to 7: measured damage for allocation, calibration size, retraining recipe, discrete widths, 1B tokens, trainable scope. Notebooks became views over Python jobs; PERUN | Local fitting declared closed; homogeneous phase declared finished on 2026-09-21 |
| October | Quantization nearly matched the dense model at INT4. The switch to Claude Code. SwiGLU-8/9, stacking, the INT3 incident | A pivot to several axes and to "prune, then quantize" |
| November | Fair reporting, heterogeneous operators, the allocator's prior work, learning-rate schedules | "If near-homogeneous is simply better, that stands as the result" |

## 6.3 Threads that run through the whole project

**1. From importance scores to measured damage.** You began with a
forward-only block-influence score, defined importance yourself on
2026-08-03 as "how much the original operator is important for entire model
performance", tested linear approximability as a proxy, found that neither
score predicted it, and ended with damage measured at the model's output.
This is the method's real intellectual path, and the thesis can tell it in
that order. It is also the claim the mixture-of-experts chapter could test
again ([file 3](03-future-work.md), Section 3.2).

**2. Layering methods.** On 2026-09-12 you described your approach as
"layering one method on top of each other". The same idea appears in July
(replacement, factorization, quantization), in October (stacking), and in
your definition of "stacked" on 2026-10-20. It is your organizing principle
and the reason the experiment classes are shaped as they are.

**3. "Why not just structured pruning?"** You asked it on 2026-09-04 and
left it open. It returned as the slice-only arm (SwiGLU-8), as the pruning
baseline (2026-11-01), and as the discovery of Puzzle. It is the question
the thesis is actually about ([file 1](01-direction-and-design.md), Section
1.2).

**4. Fairness of the comparison.** From the day quantization won
(2026-10-13) you looked for the axis on which structure pays: speed, then
time to first token, then FLOPs. You also raised it as a question of scope
on 2026-10-15: whether "optimizing purely for footprint" is what the
assignment asks. [File 2](02-reporting-and-speed.md) is my answer.

**5. Understanding before scaling.** In August you wrote that you wanted to
"go from simple to complex because I really want to understand what is
happening at each stage". That produced the activation analysis, the
global-to-local ratio, and the interaction study. It stopped when
model-level runs took over, and it is what I suggest picking up again
([file 3](03-future-work.md), idea E).

**6. Against over-engineering.** The same request appears in March, July,
August, September, and November, in nearly the same words. It is now a rule,
and the newest runners show it ([file 4](04-implementation.md), Section 4.3).

## 6.4 Reasons behind settings: what is recorded and what is not

**Correction, 2026-11-12.** The first version of this section said that
none of the ten reasons below is in `docs/`. That was wrong. I had relied on
the sub-agents' one-term searches without checking each item myself. A
second, direct search shows that six of the ten are recorded, three are
partly recorded, and one is missing. The gap is smaller than first stated.

| Setting or choice | The reason, from your sessions | Recorded? |
| --- | --- | --- |
| First and last layers are never replaced | Replacing the first block with a linear layer spiked the loss in the MVP; the supervisor called it known behaviour and you agreed not to touch them (2026-07-18, 2026-08-03) | **No.** The protection is stated in several pages; the reason is in none |
| The fitting recipe was re-examined before testing whether fitting matters | So that a null result could not be blamed on a poor fit (2026-10-10) | Partly: the same argument is written for the new operators (`heterogeneous-1.md`, "could lose only because it was trained badly"), not for SwiGLU-8 itself |
| 25% and 75% were considered as 1B targets | Meant to match the sizes of INT8 and INT4; corrected to 50% and 75%; the supervisor's 20/50/80 rule was used in the end (2026-10-16) | Partly: in the 2026-10-16 meeting note, which is git-ignored |
| Notebooks load results; jobs compute them | Notebooks are "a nice frontend"; memory and run length forced computation into jobs (2026-09-09) | Partly: the practice is recorded, the reason is not |
| Local-fit batch size 2,048 | Batch 256 won only through more updates; both reach the same fit; 2,048 is faster | Yes: decision log, 2026-10-11 |
| Screening at 100M, winners to 1B | Stay granular within 100M; promote the best | Yes: decision log, 2026-10-10 |
| Local fitting stays in the pipeline after SwiGLU-9 | Other operator families need a fitting recipe even if SwiGLU does not | Yes: decision log, 2026-11-04; `heterogeneous-1.md` |
| Low-rank linear is excluded from the heterogeneous menu | It is replacement plus a second kind of reduction, so it is stacking | Yes: decision log, 2026-11-04; `heterogeneous-1.md` |
| Retraining at 8K tokens, fitting at 128 | The mismatch led to the 8K fitting arm of SwiGLU-8 Part A | Yes: `swiglu-8.md` |
| Embeddings and the output head are never trained | Accepted on the condition that it is documented | Yes: `swiglu-4.md` |

What is genuinely only in the sessions, after this re-check:

- the reason for the protected layers (first row);
- where two ideas came from: the hybrid "linear + small MLP" (the
  supervisor's) and the data loaders (a published code base, suggested by
  the supervisor); see the attribution table below;
- ideas you raised and never ran (Section 6.5 and [file 3](03-future-work.md),
  Section 3.4);
- items that were deferred and dropped (Section 6.5).

**Attribution, which the thesis needs to get right.** Your sessions record
where several ideas came from:

| Idea | Origin, as you relayed it |
| --- | --- |
| Data loaders and the C4 / WikiText-2 split | The supervisor's suggestion, from a published code base |
| Linear layers as ablations, then hybrids; "linear + small MLP" as an adapter that restores non-linearity | The supervisor (2026-07-21, 2026-10-01) |
| The mixture-of-experts chapter | Agreed with the supervisor on 2026-06-10 |
| 3-bit quantization | The supervisor; you had not planned it (2026-10-20) |
| Stacking, for example SwiGLU 20% with INT4 | The supervisor (2026-10-16); you had proposed "prune, then quantize" on 2026-10-14 |
| A three-phase learning-rate schedule; going to 3B tokens | The supervisor (2026-11-06) |
| Per-layer operator choice, budget allocation from importance, the fit-versus-slice question, the degradation curve, the token-saving view of fitting | Yours |

## 6.5 Items that fell through

Each was raised, deferred, and then left out of later handoffs.

| Item | Last seen | State |
| --- | --- | --- |
| A structured-pruning baseline of the same class | 2026-11-03: "need to discuss this more in depth before deciding" | Later handoffs list it as out of scope; no session took it up. [File 2](02-reporting-and-speed.md), Section 2.3 proposes a resolution |
| Third-party reproducibility | 2026-11-07: you asked how an outsider can replicate up to GPU noise | The decision row was left undecided |
| The quantizer bug as an upstream report | 2026-11-01: you wanted it as a side result | Draft and patch exist; not posted |
| Corrections after finding Puzzle | 2026-11-06 | The source is unregistered; the note's "no paper combining ..." sentence and the 2026-10-17 novelty row still stand uncorrected |
| The stage-2 overfitting flag from SwiGLU-8 Part A | 2026-10-11 | Deferred; later absorbed into SwiGLU-9 Part A. Worth one sentence saying so |
| Heterogeneous-2 Step 1 results in the docs | 2026-11-08 | Audited and in memory; the page says "not yet run" |
| The fit-or-no-fit test and the scope study for non-SwiGLU operators | 2026-11-04 | Lost when Heterogeneous-2 was redesigned around forced models |
| A backlog and changelog | 2026-09-29 | `docs/backlog/` holds one four-line item from before Quantization-1 |
| Weekly progress meetings | 2026-11-07 | Not confirmed with the supervisor |

## 6.6 Things to re-check before they reach the thesis

Statements that you, or an assistant you relied on, made in sessions and
that later decisions rest on.

| # | Statement | Why to check |
| --- | --- | --- |
| 1 | Minitron "explicitly mentioned no benefit to iterative replacements" (2026-07-19; your reason for dropping iterative replacement) | As I recall the paper, the finding is that *iterative importance estimation* brings no benefit over single-shot. That is a different thing from replacing blocks one at a time with retraining in between **[lit: recalled]**. The decision can stand on compute cost alone; the citation should match the paper |
| 2 | Non-additivity starts at a layer distance of about 17 (2026-10-01, 2026-10-25; you asked whether your understanding was correct) | Checked in this session on the stored artifact: it is not. Interaction is strongest between neighbours and falls with distance; the jump at large distances is three to five specific pairs involving layers 22 and 1. See [file 3](03-future-work.md), idea A |
| 3 | "You can save up to 500M tokens by fitting" (your reading of SwiGLU-9's closing points, 2026-10-18) | A closing point is when the gap falls under 5%, not the number of tokens saved. The snapshot's own figure is a head start of 74-196M tokens |
| 4 | The teacher-initialization result that made neuron selection the main operator (NMSE 0.47 / 0.27 / 0.20) | One layer, one seed. Later experiments support it indirectly; say which |
| 5 | The 3e-5 retraining rate "was picked once, in SwiGLU-4, from two rates on a short test" | If accurate, it is a stated limit of every comparison; Heterogeneous-2's +9.8% at 1e-4 supports the concern |
| 6 | Recovery-1's "token equivalent" of 979M and 989M for a decay at 100M | It extrapolates along a nearly flat curve; a small gain becomes a huge token figure. Recovery-2 dropped the measure; do not quote the Recovery-1 values as acceleration |
| 7 | "QAT oscillation is independent of the learning rate" (attributed to Nagel et al. 2022 in one handoff), against the 2026-11-07 row's "a lower learning rate reduces oscillation amplitude" (same source) | The two readings disagree as written. As I recall the paper, the *frequency* of oscillation does not depend on the rate while the *amplitude* does, which would reconcile them **[lit: recalled]**. Check the locator before using either to explain the INT3 curve |
| 8 | Grafting uses about 2.05M token pairs per layer (2026-09-12), used to judge your calibration size | Check the locator before any comparison in the thesis |
| 9 | "Order B wins because trained blocks compensate for the rounding of frozen layers" | An untested explanation; idea E would test it |

## 6.7 What the thesis must explain

These are the points where you needed an explanation while doing the work.
A reader will need the same ones. They make a good checklist for the
background and method chapters.

| Topic | The confusion, as it appeared |
| --- | --- |
| Hidden size against MLP width | Halving the width on the wrong dimension kept 12.5% of parameters, not 50% |
| Calibration pairs, batches, epochs, optimizer updates | "I thought if I just increase the calibration data that's all that's needed" |
| Tokens against updates | Equal token counts are not equal amounts of training when batch geometry differs |
| Local fit against global quality | "If block activations get closer to the original, model impact should be minimal, no?" |
| Importance of a block | Canonical block influence, the MLP-only variant, and measured damage are three different things |
| What quantization is | Not "BF16 to BF8": groups, scales, zero points, and why widths are padded to multiples of 128 |
| Training precision | FP32 master weights under BF16 computation, and why training-time and deployed weights can differ |
| Order A and order B | You did not recognize the labels on first meeting them |
| Where the allocator stops | It chooses a menu item per layer; recipes and test sizes are chosen before it |
| Pareto frontier | "You can't have a configuration that is better in every way, or can you?" |

## 6.8 What the sessions say about working with agents

Recorded here because it supports [file 5](05-documentation-and-agents.md).

- **Plain language is the most repeated request** of the whole record, from
  August ("walls of text") to November ("short labels or part numbers
  without background do not work for me").
- **Question lists without context cost round trips.** Many replies to
  numbered design questions are "explain this better".
- **You lose track across parallel sessions**: which session owns a plan,
  what is running on PERUN. This is the practical argument for one status
  page.
- **Unrequested scope changes are the recurring breach**: edited markdown
  cells in August, an over-expanded "small section" in October, one
  experiment folded into another's manual.
- **Claims of readiness were sometimes wrong** in the Codex period (an
  environment "ready" without a module; a launcher "verified" by syntax
  check only). Your rule that "tested" means executed comes from these.
- **Cost shaped behaviour.** Usage limits decided when you worked and
  whether you delegated.

## Where the raw material is

| What | Where | In git? |
| --- | --- | --- |
| The six digests, with dated quotes | `docs/analyses/private/sessions/` | No. The folder ignores itself, because the digests quote private sessions |
| The extraction and structure scripts | `docs/analyses/private/tools/` | No (they contain local paths) |
| The filtered prompt text (1.3 MB) | Not copied into the repository. It can be rebuilt from your local session logs with the two extraction scripts and `lean.py` | No |

Delete `docs/analyses/private/.gitignore` if you want the digests tracked.
Read them first: they contain your informal remarks as well as research
content.
