---
title: Deep Analysis 2026-11-12 - Questions for You
status: draft
created: 2026-11-12
modified: 2026-11-12
---

# 7. Questions for you

Part of the [deep analysis of 2026-11-12](README.md). You left it to me
whether to use the grilling protocol. A live grilling was not possible in a
long unattended run, so this file holds the questions I would have asked,
each with my recommended answer.

Nothing here is decided. None of it becomes a decision-log row unless you
say so. The format is the one proposed in
[file 5](05-documentation-and-agents.md), Section 5.6: what the question is
about, the realistic options, a recommendation, and what changes with the
answer. Treat it as a trial of that format as well.

The rounds are ordered by dependency: the answers to round 1 shape round 2.

## Round 1: what the thesis is

### Q1. What is the thesis's main claim?

**What this is about.** The annotation describes a method: replace MLP
blocks with small trained substitutes. Your results say the best substitute
is a pruned piece of the original block, that the training step stops
mattering after long retraining, and that quantization keeps more quality
per byte. The thesis can present this in two ways.

| Option | The thesis says | Consequence |
| --- | --- | --- |
| A. Evaluation thesis | "We tested MLP replacement systematically. It converges to width pruning; its calibration step is a head start; structure and precision save different things and combine." | Every result you have is a positive contribution, including the negative ones. Matches the annotation's second goal |
| B. Method thesis | "We present a replacement pipeline and show where it wins." | Needs a place where it wins alone. Today that is only compute, which you do not measure as speed |

**Recommended: A.** It is what the evidence supports, and it removes the
opponent's strongest question ("why not just pruning?") by making it the
thesis's own.

**What changes.** With A, the next experiments should protect and explain
the three results. With B, they should hunt for a win, which is how the
3B comparison came about.

### Q2. Should compute be a second reported axis?

**What this is about.** Bytes are the primary axis by the 2026-07-18
decision. Linear FLOPs are currently a supplement labelled "compute work,
not speed". It is the one axis on which replacement wins without help.

| Option | Consequence |
| --- | --- |
| A. Keep FLOPs as a supplement | Nothing changes; the thesis reads as "quantization wins" |
| B. Report bytes and compute as two axes, plus a joint view | One more figure per snapshot; no new runs; no speed claim |
| C. Add measured speed as an axis | Opens the kernel question for quantized models |

**Recommended: B**, with the bounds sentence from
[file 2](02-reporting-and-speed.md), Section 2.7. It stays inside the
supervisor's warning: no kernels, no latency estimate.

**What changes.** With B, stacked models become the natural "both savings"
family, and SwiGLU alone has a stated purpose.

### Q3. Should smaller models of the same family appear on the plots?

**What this is about.** SmolLM2-360M exists, saves about 79% of the bytes
of the 1.7B model, and by its model card scores about 58% on four of your
tasks. Your stacked models reach 58% at 56% saving and fall to 48-50% at
61-62%. If your own evaluation agrees, the smaller model beats every
compressed model beyond about 56% saving on task accuracy.

| Option | Consequence |
| --- | --- |
| A. Do not include it | The opponent may raise it; the extreme-compression results have no reference |
| B. Evaluate 360M and 135M with your evaluation and plot them | Under one GPU-hour. May show the far end of your frontier is not competitive |
| C. B, plus distil the teacher into the 360M model with your recipe | Adds the cleanest control for "inherit weights or just distil"; needs new code |

**Recommended: B now, C as an option.** An unflattering reference that you
show yourself is worth more than a flattering plot that omits it.

**What changes.** With B, the question "how far can compression go?" gets
an answer, and plans such as a 90% target can be judged against it first.

### Q4. When do new experiment classes stop, and when does writing start?

**What this is about.** The thesis is due 2027-05-07. The decision log says
writing starts this semester and the seminar needs one finished chapter. I
found no thesis text in the tracked tree. The planned work (3B runs,
Heterogeneous-2 steps 2 and 3, mixture-of-experts, other models) has no end
date.

| Option | Consequence |
| --- | --- |
| A. No fixed date | Experiments expand to fill the time; writing is squeezed |
| B. Freeze new classes at a date (for example the end of January), and start the related-work and method chapters now | Experiments after the date only fill gaps in the claims ledger |

**Recommended: B.** The related-work and method chapters do not depend on
any pending result. A claims ledger ([file 5](05-documentation-and-agents.md),
Section 5.5) shows what is ready to write.

**What changes.** With B, every proposed run needs an answer to "which
sentence of the thesis does this support?"

## Round 2: the next experiments

### Q5. What happens to the 3B comparison?

**What this is about.** After the 2026-11-06 meeting, the plan is to train
B 20% INT4 (stacked) and dense INT3 to 3B tokens to see whether stacking can
win near 51% saving. Recovery-2 finished afterwards. With each family's
best schedule at 1B, dense INT3 is at KL 0.054 with 51.6% saving and the
stacked model at 0.078 with 50.9%. Dense is ahead on both.

| Option | Cost (rough) | What you learn |
| --- | --- | --- |
| A. Run both as planned | About 76 GPU-hours and two sessions | Almost certainly that dense stays ahead |
| B. Show the supervisor the Recovery-2 table and ask whether 3B is still wanted | A conversation | Whether the question is already considered answered |
| C. Replace it with the dense 2-bit baseline | About 15 GPU-hours | Whether stacking's region above 52% survives; I cannot predict the outcome |

**Recommended: B, proposing C.** The supervisor asked for 3B before
Recovery-2 existed. This is his suggestion, so it is his call and yours.

**What changes.** If C is chosen, the thesis's stacking claim gets its
missing control. If A, it gets a confirmation of what the 1B table shows.

### Q6. Run the cheap checks on the allocator?

**What this is about.** The knapsack is the centre of the method. It has
never been compared with a uniform allocation under the final recipe, and
its additivity assumption is known to underestimate damage by 23-55%
without anyone knowing whether that changes its choices. The same runs
would also give you the structured-pruning baseline you left open on
2026-11-03. By the allocator's own cost table, its predicted edge over
uniform before retraining is 27% at 20% removal and only 9-13% from 50%
upward, so the outcome is not a foregone conclusion. This session's offline
check of your interaction study adds a
concrete prediction for the ranking test: damage compounds between
neighbours, and narrowing layer 22 together with layers 5-7 may be far
worse than the sum predicts ([file 3](03-future-work.md), idea A).

| Option | Cost (rough) | Consequence |
| --- | --- | --- |
| A. Leave as a stated limit | None | "We did not test whether the allocator helps" in a thesis built on it |
| B. Uniform against knapsack at 20/50/80%, fitted and slice-only, 100M tokens | 7-9 GPU-hours | Fills the 2 × 2 table: pruning baseline, allocator ablation, fitting ablation |
| C. B plus the ranking test (about twenty allocations, no training) | 3-4 GPU-hours more | An original result on when the additive solve can be trusted |

**Recommended: C.** It is the cheapest work on this list and protects the
most.

**What changes.** If the knapsack clearly beats uniform, the method chapter
stands as written. If it does not, you learn it now and the chapter says so.

### Q7. How far does Heterogeneous-2 go?

**What this is about.** No new block type was picked. Forced models were
17% and 8% worse at 20% and 50%, and about 2% worse at 80%. By your rule
the 80% model continues to 1B (Step 2). Step 3 (three training scopes and a
no-fit test, up to about 170 GPU-hours) was designed for hybrids that the
solver would pick.

| Option | Consequence |
| --- | --- |
| A. Stop after Step 1 and write it up | The chapter says: no hybrid was selected; forced hybrids did not beat SwiGLU at 100M; here is why |
| B. Run Step 2 at 80% only | Adds "level at 1B as well", about 10-12 GPU-hours |
| C. Run Steps 2 and 3 | A large budget for a result that is very likely "no difference" |

**Recommended: A or B, not C.** B follows your logged rule; A is defensible
because the block-level evidence is decisive. Either way, write Step 1 into
the page now: it still says "not yet run".

**What changes.** Nothing in the thesis's conclusions, in my estimate. That
is the argument for not spending more here.

### Q8. What should the mixture-of-experts chapter be?

**What this is about.** It is a bonus by the 2026-10-16 decision. MoNE
replaces rarely used, low-variance experts with constants, chosen by a
local score, without retraining. Your tools generalize both its menu and
its score.

| Option | Scope | Consequence |
| --- | --- | --- |
| A. Drop it | None | The thesis stays on one dense model |
| B. Step M1 only: reproduce MoNE on one small model, then choose experts by measured single-expert KL instead of its score | No retraining; new capture code | A complete small chapter, directly comparable with a paper's table |
| C. M1 plus the mixed menu and a 100M retraining | Larger | Tests whether a mixed menu wins where blocks are used unequally |

**Recommended: B, with C only if B works and time remains.** It is the one
place where the idea that failed on the dense model has a structural reason
to succeed, and where your method's central claim can be tested against a
published baseline.

**What changes.** With B, the thesis gains a second model family and a
literature comparison in one step, which also covers part of "other
models".

## Round 3: closing open items

### Q9. Record the Puzzle finding?

**What this is about.** On 2026-11-06 you found that Puzzle (ICML 2025)
already combines a per-layer block library, replace-one-block KL scores, an
integer program with one choice per layer, and global distillation. The
source is still unregistered. The width-allocation note still says "no paper
combining ..." and calls the heterogeneous extension "the most clearly new
part". The 2026-10-17 row rests the claim on three points that all overlap
Puzzle.

**Recommended:** register the source, correct the note's Section 8, and add
a row that supersedes the 2026-10-17 novelty row. State what remains
distinct: a byte or parameter budget, an exact dynamic program, the measured
super-additivity, quantization compared on a byte frontier, and the
fit-versus-slice evidence.

**What changes.** The related-work chapter is built on the right
foundation. The longer this waits, the more text gets written on the old
claim.

### Q10. Split the decision log and allow status updates without asking?

**What this is about.** The log is 153 rows and about 31,000 tokens, read
at every session start; 111 rows concern single experiments. Separately,
results that agents have audited sit in memory while pages say "not yet
run", because agents may not edit pages without approval.

**Recommended:** (1) a standing-decisions file of about 40 rows, design rows
in each experiment page, and the full history in an archive; (2) a narrow
permission: agents may update status and "next step" lines in the
experiment page and `dependencies.md` at a phase boundary and must list
what they changed.

**What changes.** Lower cost per session, and one place that is always
current.

### Q11. What happens to the three never-run notebook ports?

**What this is about.** `swiglu_initial.py`, `swiglu_2_allocation.py`, and
`baseline/compression.py` are 4,293 lines, a sixth of the runner code. The
2026-10-10 row keeps them and says each is marked "never run; the notebook
is authoritative". I did not find that marking. The 2026-10-15 row deletes
code that never produced a result and has no caller.

| Option | Consequence |
| --- | --- |
| A. Add the marking the row requires | The appendix reader knows what they are |
| B. Delete them; git keeps them | The appendix contains only code that ran |

**Recommended: B.** The notebooks are the record; the ports add untested
code to a repository an opponent will read.

**What changes.** A sixth less runner code to keep consistent.

### Q12. Three small closures

**What this is about.** Three items your sessions left open, each cheap.

| Item | Recommended |
| --- | --- |
| The quantizer mismatch as an upstream report | Post the issue you drafted. It is an outward action, so only you can do it. One sentence in the thesis then cites it |
| Third-party reproducibility (the undecided 2026-11-07 row) | One paragraph: pinned inputs, seeds, and code hashes reproduce results up to the measured GPU noise (0.3% for BF16, up to 3% for QAT) |
| Macro-accuracy frontier membership decided by differences of a few tenths of a point | Treat differences under one point as ties, or run the existing paired bootstrap on stored samples. No GPU needed |

**What changes.** Three fewer loose ends at the defence.

## If you answer only one question

Q1. It decides how every other answer reads.
