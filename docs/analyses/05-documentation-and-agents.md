---
title: Deep Analysis 2026-11-12 - Documentation, Knowledge Base, and Agent Setup
status: draft
created: 2026-11-12
modified: 2026-11-12
---

# 5. Documentation, knowledge base, and the agent setup

Part of the [deep analysis of 2026-11-12](README.md). It covers your topics
2.4 (documentation of implementations), 2.5 (documentation of results), and
the optional item 7 (agent policies: `AGENTS.md`, logging, grilling, the
knowledge base). Evidence labels are defined in
[file 1](01-direction-and-design.md).

## Short answer

- **The record-keeping is the strongest I have seen in a student project,
  and it is now working against you in two ways.** It costs tokens every
  session, and the same fact lives in up to five places that drift apart.
- **The decision log has become a changelog.** 153 rows, about 31,000
  tokens, read by every agent before any work. 111 of the rows are about one
  experiment's parameters. Splitting standing decisions from experiment
  decisions would cut the per-session cost by roughly two thirds and lose
  nothing.
- **The wiki is not doing the job the thesis needs from it.** It has 40
  pages and 29 registered sources, its front page is a month stale, and the
  closest prior work to your whole pipeline (Puzzle) is still not in it.
  What the thesis needs is a related-work ledger. That is a table, not a
  wiki.
- **Nothing connects results to thesis text yet.** The piece that is missing
  is a claims ledger: each sentence the thesis will assert, its evidence,
  its caveat. Writing has to start this semester, and this is the bridge.
- **The agent rules are good. The grilling format is the weak part:** your
  logs show you repeatedly could not follow its questions.

## 5.1 Where a fact lives today

Take one fact: "Heterogeneous-2 Step 1 ran, and the 80% forced model is
level with SwiGLU." On 2026-11-12 **[read]**:

| Place | What it says |
| --- | --- |
| Agent memory (2026-11-08) | Ran, audited; 20/50% stop, 80% level; docs pending approval |
| `heterogeneous-2.md` | "Step 1 is implemented and not yet run" |
| `dependencies.md`, table | "Step 1 implemented 2026-11-06, not yet run" |
| `dependencies.md`, diagram | Heterogeneous-1 is labelled "done" in its box and coloured "planned" by its class line |
| Decision log | The Step 1 design rows; no outcome row |
| Results snapshot | Section 4: "Pending. The block-level study comes first" |

Five places, four different states. The correct one is in the least visible
place: a memory file only one tool can read. This is the pattern to fix. It
comes from a good instinct (write everything down) meeting a rule that is
right but slow (agents change docs only with approval), so results sit in
memory "pending approval" while the pages go stale.

More examples of drift **[read]**:

- The wiki's front page still says SwiGLU-9 "is implemented but not yet
  run". It ran on 2026-10-17/18. The page does not mention Stacking,
  Heterogeneous, or Recovery at all.
- The decision log links to `docs/meetings/`, which is git-ignored. Five
  tracked documents point into it. In the public appendix those links are
  dead, and the 2026-10-16 and 2026-11-06 rows cite them as their source.
- The 2026-10-10 row says the three never-run notebook ports are each
  marked as such. I could not find the marking ([file 4](04-implementation.md),
  Section 4.3).
- Memory holds 39 files. Eight are your working preferences, which is what
  memory is for. About thirty are status snapshots ("uncommitted", "pending
  approval", "next: ...") that were true on the day they were written.

**The principle you already wrote down** (2026-10-10 row): "an agent can
determine the current state from one or two entry files, each fact has one
source of truth, nothing looks current that is not". The setup has drifted
from it. The recommendations below are ways back.

## 5.2 The decision log

### What the numbers say [ran]

| Measure | Value |
| --- | ---: |
| Rows | 153 |
| Written in October / November | 104 / 47 |
| Rows carrying a superseded, refined, or amended note | 22 |
| Rows that name a specific experiment | 111 |
| Rows that name none (project-level) | 42 |
| Median row length | 60 words; 13 rows exceed 150 words |
| Size | about 12,000 words, about 31,000 tokens |

Root `AGENTS.md` requires every agent to read the log before planning,
steering, or implementing. With about 50 sessions since October, that is up
to about 1.5 million input tokens (less in the early sessions, when the log
was shorter) spent on re-reading rows such as the time limit of a finished
job array.

### Assessment [view]

The log does three jobs at once:

1. **Standing rules** ("one seed per variant", "bytes are the primary axis",
   "thesis in English"). Every session needs these.
2. **An experiment's design record** ("Recovery-2's reference is the median
   of four rows"). Only that experiment's sessions need these, and the
   experiment page repeats them.
3. **A history of reversals** (rows that supersede rows). Nobody needs these
   at session start; they are for audit.

Mixing them is why it is long, and why the binding-until-changed rule now
binds agents to hundreds of parameters of finished runs.

### Recommendation

Keep the mechanism; split the file.

| File | Holds | Read when |
| --- | --- | --- |
| `decisions.md` | Standing decisions only, about 40 rows, current wording, no superseded rows | Every session |
| A decisions section in each experiment page (most already have one in prose) | That experiment's design rows | Sessions on that experiment |
| `decisions-archive.md` | The full dated history, as it is today | On demand, for audit |

Rules that keep it honest: a row goes to the standing file only if it would
still matter after the experiment that prompted it is finished; a standing
row that changes is edited in place and its old wording moves to the
archive.

This is a documentation change with no scientific content, so it is safe.
It needs your approval because the log is binding.

## 5.3 Experiment pages and result snapshots

**Experiment pages** (600-800 lines each) hold the plan, the implementation
notes, the checks, the results, and the measured resource use in one file.
As lab notebooks they are excellent: I could reconstruct what happened and
why from them alone, and every number names its artifact. Two costs
**[view]**:

- They are long to load. An agent that needs one result reads the whole
  history of how the experiment was planned.
- The results are near the bottom, under the plan. A reader who wants "what
  did we learn" has to scroll past "what did we intend".

A light fix that needs no restructuring: keep a "Results summary" block at
the top of every page, as Heterogeneous-1 already does. That page is the
model to copy.

**Result snapshots** are the best documents in the repository: figure
first, one small table, three bullets, sources linked. The "open gaps"
table at the end is honest and useful. Two additions would complete them:
the reporting items in [file 2](02-reporting-and-speed.md), Section 2.8, and
a rule that a snapshot is not "final" while any experiment page it cites is
stale.

**`dependencies.md`** is a good idea and the right single place for status.
It only works if it is the *first* thing updated at a phase boundary, not
the last. I would make it the one status source and have the experiment
page, the snapshot, and memory point to it instead of repeating it.

**The PERUN runbook** is about 14,000 words. Your sessions show why it
exists (earlier cheatsheets did not work; you had to ask for commands
repeatedly). It is the cost of the launcher design in
[file 4](04-implementation.md): every experiment has its own launcher and
therefore its own chapter. Sharing launcher blocks for new experiments
shrinks future runbook chapters as well.

## 5.4 The wiki

### State [read, ran]

- 40 tracked pages: 14 methods, 6 source summaries, 4 concepts, 4
  experiments, 3 each of implementations, research, and syntheses.
- 29 registered sources; 4 ingested in full.
- A 490-line schema that governs page types, metadata, and operations.
- Its front page was last brought up to date around 2026-10-17.
- By your own audit of 2026-10-15, it had been read 12 times across all
  sessions. You decided then to review it after the first thesis chapter,
  and to fold it into `docs/` if the method pages go unused.

### Assessment [view]

Your original aim in July was sound: keep prior work, your own ideas, and
generated summaries strictly apart. That discipline is the valuable part and
it shows everywhere (equations carry labels; claims carry locators).

The wiki as a *structure* has not paid off. The evidence is the Puzzle
episode. The paper that most closely matches your pipeline (per-layer block
library, local distillation, replace-one-block KL scores, an integer program
with one choice per layer, then global distillation) was found by accident
on 2026-11-06, because the supervisor asked whether the allocator was new.
A system built to track prior work missed the most relevant prior work for
four months, and five days later it is still not registered, because
registering it waits on approval.

The lesson is not that the wiki was badly made. It is that page types and
schemas do not find papers. A search does, and a short list is enough to
keep what the search finds.

### Recommendation: a related-work ledger

One table, which is also the skeleton of the related-work chapter:

| Column | Example |
| --- | --- |
| Source | Puzzle (Bercovich et al., ICML 2025) |
| What it does, in one line | Per-layer block library, replace-one-block KL scores, mixed-integer program, global distillation |
| What we take from it | The formulation of allocation as one choice per layer |
| How we differ | Parameter or byte budget; exact dynamic program; measured super-additivity; quantization on a byte frontier |
| Locator checked | Sections read, version |
| Status | Checked in the PDF / abstract only / recalled |

About thirty rows would cover the thesis. The existing source summaries and
method pages stay as they are for the four ingested papers.

**Sources this analysis relies on or recommends, with how far each was
checked.** None is a claim that it must be cited; each is a lead.

| Source | Why it matters | Checked how |
| --- | --- | --- |
| Puzzle (arXiv 2411.19146) | Closest prior pipeline | Your 2026-11-06 session read the v5 HTML; not registered |
| Kuzmin et al., "Pruning vs Quantization: Which is Better?" (NeurIPS 2023) | Your headline result, found independently | Abstract checked this session |
| ParetoQ (Meta, NeurIPS 2025) | Low-bit QAT; a break between 2 and 3 bits | Abstract checked this session |
| SmolLM2 model cards and paper (arXiv 2502.02737) | Sibling anchors | 360M card checked; 1.7B partly |
| vLLM "Registering a Model" documentation | Feasibility of a same-kernel speed test | Checked this session |
| EvoPress, DarwinLM | Search against additive allocation | In your note as unregistered leads |
| Hubara et al. 2020; HAWQ-V3 | Integer programs for per-layer choices | In your memory note; not verified by me |
| MatFormer; Flextron | Nested-width training | Recalled; verify |
| LLM-Streamline; ReplaceMe | "Replacement" at the layer level | Recalled; verify |
| Transcoders (Dunefsky et al. 2024) | An MLP replaced by a function trained on its input/output pairs, in the interpretability literature; they report the same local-versus-global error gap you measured | Recalled; verify |
| MC-SMoE; Lu et al. (expert pruning); STUN; MoE-Pruner | Prior work for the mixture-of-experts chapter | Recalled; verify |
| Minitron | Its "no benefit from iterating" finding concerns importance estimation, not iterative replacement; your 2026-07-19 reason for dropping iterative replacement cites it | Recalled; check the locator ([file 6](06-session-findings.md)) |

## 5.5 From results to thesis text: a claims ledger

The decision log says writing starts this semester and the Diploma Seminar
needs one finished chapter. I found private notes under `docs/vault/` but no
thesis text and no outline in the tracked tree. The documentation records
*what was done* very well. Nothing yet records *what will be asserted*.

A claims ledger is one page:

| Claim, as the thesis will state it | Evidence | Budget and scope | Caveat | Status |
| --- | --- | --- | --- | --- |
| Local fitting no longer matters after 1B tokens | SwiGLU-9 Part B: g = 2.8 / 1.8 / 0.3% | 20/50/80%, one model, one seed, constant 3e-5 | Not zero at 20%; twin noise 0.31% | Ready |
| Stacking extends the frontier beyond 51.6% | Stacking-1 | Against INT8/4/3 only | No dense 2-bit control | Needs a control |
| ... | | | | |

The twelve findings in [file 1](01-direction-and-design.md), Section 1.1 are
a first draft of its rows. Its value: it shows at a glance which claims are
ready to write, which need one more run, and which experiments support no
claim at all. It also gives every future experiment a test before it
starts: which row does it fill?

## 5.6 The agent setup

### `AGENTS.md`

It is clear, short enough to follow (about 1,500 words), and its rules have
visible origins in real incidents: only you touch PERUN (an agent tried
SSH), no cross-session messages (one burned the limit), no smoke tests,
equations carry provenance. The "objective judgment" section is the best
part; I have tried to follow it in this analysis. Two observations **[view]**:

- **The approval rule has a cost that is now visible.** "Do not modify
  unless I approve" is right for code and for science. Applied to status
  lines, it produces the drift in Section 5.1. A narrow standing permission
  would remove most of it: *agents may update status fields and "next step"
  lines in the experiment page and `dependencies.md` at a phase boundary
  without asking, and must list what they changed.* Results sections,
  decision rows, and anything interpretive stay behind approval.
- **"Read the decision log before any work"** should become "read the
  standing decisions" once the log is split (5.2).

### Grilling

The protocol is sound in structure: decisions as a tree, a recommended
answer with every question, facts looked up by the agent, a summary that
becomes decision rows. Its outputs (the design sections of Heterogeneous-1,
Recovery-1, and Recovery-2) are good.

Its weak point is the question format. Your logs show the same reaction many
times, across sessions and months **[read: L4, L6]**: "I don't understand
this question", "need this explained better", "please elaborate, I really
don't understand this". On 2026-11-06 you wrote it as an instruction:
"Explain things plainly with context first; ... short labels or part numbers
without background do not work for me."

The current format gives a title, a question, and a one-line reason. For a
research decision that is too little to answer with a reason of your own,
which is what the protocol asks of you. Suggested changes to `grilling.md`:

| Now | Suggested |
| --- | --- |
| Up to seven questions per round | Up to four |
| Title, question, recommended answer, one-line reason | Add two lines before the question: what this is about in plain words, and what changes depending on the answer |
| Options "when useful" | Always name the realistic alternatives and what each costs |
| Jargon and run labels allowed | Every label (S7-0, order B, K7) is spelled out on first use in each round |

[File 7](07-questions.md) uses this format, as a trial.

### Memory

Memory is doing the job the documents should do: it holds the current state
of experiments. That is fragile. It is specific to one tool, which conflicts
with your decision to stay tool-agnostic, and it is invisible in the
repository. I would keep memory for what only memory can hold (your working
preferences, pointers) and move status to `dependencies.md`, with the narrow
permission above to keep it current.

### Sessions and cost

Your 2026-10-15 audit found that token cost follows session length. Three
structural causes are in your control and are covered above: the log read at
every start (5.2), long experiment pages (5.3), and runners that require six
other files to understand ([file 4](04-implementation.md), Section 4.4).

One more, from this session's own experience. Nine sub-agents on full
transcripts hit the weekly limit before finishing. Six sub-agents on
transcripts filtered down to your substantive prompts finished, at about one
million tokens in total. Sub-agents are worth their cost only on input that
has been cut down by a script first. That agrees with your 2026-11-01 row on
read-only search sub-agents.

## 5.7 Summary of recommendations

| # | Change | Effort | Effect |
| --- | --- | --- | --- |
| 1 | Split the decision log into standing, per-experiment, and archive | An hour, documentation only | About two thirds fewer tokens at every session start |
| 2 | Make `dependencies.md` the one status source; allow agents to update status lines unasked | A rule change | Ends the drift in 5.1 |
| 3 | Start a claims ledger | An hour | Connects results to writing; tests new experiments |
| 4 | Start a related-work ledger; register Puzzle first | A few hours | The related-work chapter's skeleton |
| 5 | Give grilling questions context and alternatives; four per round | Edit one file | Fewer round trips |
| 6 | Put a results summary at the top of each experiment page | Minutes per page | Cheaper reads |
| 7 | Decide what happens to `docs/meetings/` links in the public appendix | A decision | No dead links for the opponent |
| 8 | Prune memory to preferences and pointers | Half an hour | One place for state |
