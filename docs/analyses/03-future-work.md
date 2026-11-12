---
title: Deep Analysis 2026-11-12 - Future Work, Defined and New
status: draft
created: 2026-11-12
modified: 2026-11-12
---

# 3. Future work: the planned topics, new ideas, and what to skip

Part of the [deep analysis of 2026-11-12](README.md). It covers your topics 3
(planned future work: heterogeneous and MoNE), 4 (new ideas), and 5 (only
changes that matter). Evidence labels are defined in
[file 1](01-direction-and-design.md).

All GPU-hour figures here are **rough scoping numbers** taken from throughput
already measured in your docs (1.0-1.4 hours per 100M tokens on one H200,
0.2-0.3 hours per final evaluation). They are not PERUN resource plans; a
real plan needs `perun-resources.md` and its own session.

## Short answer

Ranked by what they would add to the thesis per unit of your time:

| Rank | Idea | Kind | What it changes | Rough cost |
| --- | --- | --- | --- | --- |
| 1 | A. Allocator checks (knapsack against uniform; ranking test) | Protects a claim | Makes the method chapter defensible | 10-13 GPU-h |
| 2 | B. Sibling-model anchors | Protects and reframes | Sets the range where compression is worth it | under 1 GPU-h |
| 3 | C. Dense 2-bit baseline | Protects a claim | Decides whether "stacking extends the frontier" survives | about 15 GPU-h |
| 4 | Mixture-of-experts chapter, redesigned around a measured cost | New result | The one setting where a mixed menu should win; comparable with a paper | 10-30 GPU-h for the no-retraining version (uncertain) |
| 5 | E. What retraining repairs | Explanation | Turns four observations into an account | 2-4 GPU-h, evaluation only |
| 6 | D. Width and bits in one allocator | New result | Unifies stacking with the allocator | 30-60 GPU-h |
| 7 | G. One replication on a second model | Generality | Answers "one model" | 30-60 GPU-h (uncertain) |
| 8 | F. Peak learning rate for the BF16 families | Fairness | Moves the "blue dot" about 10% | 4-14 GPU-h |
| 9 | H. One retraining run that yields every budget | New idea | Higher risk, real novelty | 10-15 GPU-h |

Items 1-3 and 5 together cost less than the planned 3B pair and each removes
a question you would otherwise meet at the defence. Item 4 is where I would
put the remaining ambition.

## 3.1 The planned topic: heterogeneous replacement

### Where it stands

Heterogeneous-1 answered the block-level question thoroughly: no new block
type is picked at any target, and at equal size SwiGLU does 1.05 to 13 times
less damage. Heterogeneous-2 Step 1 forced eight layers to other types: 17%
and 8% worse at 20% and 50% (stop), about 2% worse at 80% ("level", so your
rule sends it to 1B) **[read: heterogeneous-1.md; memory note of 2026-11-08,
results not yet on the experiment page]**.

### My reading

This is a completed negative result with a clear cause ([file 1](01-direction-and-design.md),
Section 1.6): on a dense model, every block processes every token, and a
slice of the teacher starts from a trained function. The supervisor's goal,
"a better blue dot", was not reached by changing the block type. Two
observations from your own data point at what would move it instead
**[read]**:

- SwiGLU 80% gained 9.8% from a higher peak learning rate alone
  (Heterogeneous-2's check), which is more than any hybrid delivered.
- The heterogeneity that pays is in *how* each layer is compressed (fewer
  neurons or fewer bits), not in what kind of function it is. That is
  idea D.

### What I would do with the remaining heterogeneous plan

| Item | My view | Reason |
| --- | --- | --- |
| Step 2 at 80%, 1B | Optional. Run it only if you want the sentence "level at 1B as well" | Expected outcome is within noise; the rule allows it and it is cheap |
| Step 3 (scopes, no-fit, about 170 GPU-h) | Skip unless Step 2 shows a clear win | It was designed for hybrids that the solver would pick; none exists |
| Write-up | Do now | Step 1 results are home but the page still says "not yet run" |

The thesis sentence you drafted on 2026-11-06 ("the allocator selected no
hybrid ... so hybrids were enforced ... and failed to beat pure SwiGLU") is
the right framing. I would add the cause and the pointer forward: *mixed
menus need blocks that differ in how much they matter per token; a dense
model's blocks do not, a mixture-of-experts model's do.*

## 3.2 The planned topic: mixture-of-experts, extending MoNE

This is where I think the project's remaining ambition is best spent
**[view]**. Here is why, and a design that fits what you have built.

### Why the same tools should behave differently there

MoNE **[read: your wiki source summary]** removes experts that are rarely
routed to and whose output varies little, and replaces each with a constant
"novice" (its mean output). It scores experts by routing frequency and
output variance, and its main comparisons use **no retraining**.

Three things line up with your work:

1. **MoNE's menu is a special case of yours.** Per expert it offers two
   items: keep, or mean. Your allocator already handles keep, several
   SwiGLU widths, mean, and zero, with one item per block and a budget.
2. **MoNE's score is a local proxy; you have the measured alternative.**
   Frequency times variance is roughly the squared error a mean replacement
   makes at that expert's output. Your central methodological result (F2 in
   file 1) is that damage measured at the model's output is a better guide
   than local proxies. That claim can be tested directly: choose experts by
   measured single-expert KL instead of MoNE's score, at the same number of
   removed experts.
3. **The no-retraining setting is where your calibration step matters most.**
   F5 says local fitting is worth 2-5 times in KL without retraining. On a
   dense model that advantage was erased by long retraining. In MoNE's
   setting there is no retraining to erase it.

And one thing is different in the right way: experts are used very
unequally. On a dense model the mean and zero items cost 1.5-1.6 times more
damage than the smallest SwiGLU. For an expert that almost no token reaches,
the mean item is nearly free. The mixed menu therefore has a real reason to
produce a mixed model.

### A design in three steps

| Step | What | Retraining | Answers |
| --- | --- | --- | --- |
| M1 | Reproduce MoNE at 25% and 50% of experts on one small model (OLMoE 7B-A1B is the smallest in their table). Then select the same number of experts by measured single-expert KL instead | None | Does a measured cost beat the frequency-variance proxy? |
| M2 | Give each expert the full menu (keep, narrower SwiGLU at a few widths with and without local fitting, mean, zero) and solve at the byte budgets of M1 | None | Does a mixed menu beat all-or-nothing replacement at equal bytes? Does local fitting matter here? |
| M3 | Retrain the best M1/M2 models for 100M tokens with your recipe | 100M | Does the ordering survive retraining, as it did not for fitting on the dense model? |

M1 alone is a complete, modest chapter: one model, one published baseline,
one clear question, and a result that is directly comparable with a paper's
own table, which nothing else in the thesis is.

### What to expect, honestly

- **Bytes:** quantization will very likely still win on bytes in a
  mixture-of-experts model, for the same reason as on SmolLM2. The chapter
  should not promise otherwise. Its claim is about *how to choose what to
  remove*, and about the menu.
- **Compute:** removing a rarely used expert saves almost no compute,
  because it was rarely run. Narrowing a heavily used expert does. So the
  byte and compute axes come apart even more clearly than on a dense model,
  which fits the three-axis reporting in [file 2](02-reporting-and-speed.md).
- **Cost:** a single-expert table for 16 layers of 64 experts is about a
  thousand forward evaluations per menu item. I guess 4-6 GPU-hours per item
  on one H200, but nothing here is measured; treat it as unknown until a
  plan session estimates it. Capturing per-expert input/output pairs is new
  code: pairs exist only for tokens routed to that expert, and rarely used
  experts will have few.
- **Risks:** a new model family and its routing code; MoNE's exact
  evaluation suite (nine tasks) differs from yours; the paper's reported
  sensitivity to calibration data (its Appendix A) applies to you too.

### Prior work to check before designing

Expert pruning and merging is an active area. From memory, to be verified
and registered before any claim **[lit: recalled]**: expert merging by
routing statistics (MC-SMoE), expert pruning and skipping (Lu et al.),
structured-then-unstructured pruning of experts (STUN), and
MoE-Pruner. The gap I believe is open is *a measured model-level cost inside
an exact per-expert allocation with a mixed menu*; that belief needs the
same kind of check that found Puzzle.

## 3.3 New ideas

### A. Check the allocator's assumptions (rank 1)

**What.** Two small experiments.

1. *Knapsack against uniform.* At 20/50/80%, build the uniform-width model
   (fitted and slice-only) and retrain for 100M tokens with the standard
   recipe. Together with SwiGLU-8's existing arms this fills the 2 × 2 table
   in [file 2](02-reporting-and-speed.md), Section 2.3.
2. *The ranking test* you proposed in the width-allocation note. At one or
   two budgets, assemble a spread of allocations (the solver's, near-optimal
   runners-up, uniform, a few random feasible ones), measure each one's real
   joint KL before retraining, and compare the order with the predicted
   order. Every grid width is already fitted, so this is assembly and
   evaluation only.

**Why it matters.** The allocator is the centre of the method and its
novelty is gone (Puzzle). What remains yours is *evidence about when the
additive solve can be trusted*. Puzzle, as far as your check read it, does
not discuss additivity. A plot of predicted against real damage for twenty
allocations is a small original result, and it answers the first question an
opponent will ask.

**What could happen.** If knapsack beats uniform clearly: the method chapter
stands. If it does not: that is a finding too, and far better found now.

#### How large is the allocator's predicted benefit?

An offline calculation on `swiglu-8/prepare-001/result.json` **[ran]**. For
each target I summed the smoothed single-layer KL of the recorded knapsack
allocation and of a uniform allocation with the same removal. The knapsack
sums reproduce the predicted values in your width-allocation note exactly,
which confirms the artifact was read correctly.

| Target | Uniform, predicted sum | Knapsack, predicted sum | Knapsack lower by |
| ---: | ---: | ---: | ---: |
| 20% | 0.124 (interpolated) | 0.090 | 27.5% |
| 30% | 0.187 (interpolated) | 0.148 | 20.9% |
| 40% | 0.259 (interpolated) | 0.214 | 17.1% |
| 50% | 0.338 | 0.295 | 12.7% |
| 60% | 0.446 (interpolated) | 0.390 | 12.5% |
| 70% | 0.578 (interpolated) | 0.513 | 11.4% |
| 75% | 0.648 | 0.590 | 9.0% |
| 80% | 0.754 (interpolated) | 0.677 | 10.2% |

"Interpolated" means the uniform width is not on the measured grid and was
priced by a straight line between its neighbours. These curves flatten as
width grows, so a straight line lies above them and overstates the uniform
cost a little; those rows flatter the knapsack. The 50% and 75% rows use
measured widths only.

**What this says.** By its own objective, and before any retraining, the
allocator expects to be 9-13% better than uniform from 50% upward and
17-28% better at low removal. That is a modest edge. Compare local fitting:
its advantage before retraining was 110-427%, and retraining reduced it to
10-24% at 100M tokens and to 0.3-2.8% at 1B. Where a width is placed cannot
be undone by retraining the way a poor fit can, so the allocator's edge
should last better than the fit's did. But it starts an order of magnitude
smaller. It is entirely possible that, after 100M tokens, knapsack and
uniform differ by a few percent **[view]**. That is why this ablation is
not a formality, and why its outcome belongs in the thesis either way.

#### What the stored interaction study already shows

While checking your "distance 17" reading (Section 3.4), I ran an offline
calculation on `data/results/notebook-model-study/block-interaction.json`,
the artifact of the pairwise study: all 231 pairs of the 22 eligible layers,
for two operators. No GPU; the script only reads the stored rows **[ran]**.
"Relative interaction" is the pair's real KL minus the sum of the two
single-layer KLs, divided by that sum (the residual your note defines as
Eq. 8, made relative).

| | SwiGLU at 50% width | Ridge-fitted linear |
| --- | ---: | ---: |
| Pairs whose damage exceeds the sum | 192 of 231 (83%) | 166 of 231 (72%) |
| Median relative interaction, all pairs | +4.2% | +3.5% |
| Median, neighbours (distance 1-2), layers 1 and 22 excluded | +10.5% | +12.6% |
| Median, far apart (distance 10 or more), layers 1 and 22 excluded | +2.7% | +1.3% |
| Rank correlation of relative interaction with distance | −0.47 | −0.50 |
| The same, layers 1 and 22 excluded | −0.58 | −0.66 |

Two things follow.

1. **Interaction falls with distance; it does not rise.** Neighbouring
   layers compound each other's damage by about 10%; layers ten or more
   apart are close to additive. This fits the growth of the allocator's
   underestimate with removal (1.23 at 20% to 1.55 at 80%): the more is
   removed, the more compressed layers sit next to each other.
2. **A handful of specific pairs collapse the model.** For SwiGLU, the
   three largest interactions are:

   | Pair | Distance | KL of the pair | Sum of the two alone |
   | --- | ---: | ---: | ---: |
   | 5 and 22 | 17 | 12.75 | 0.73 |
   | 7 and 22 | 15 | 11.11 | 0.80 |
   | 1 and 22 | 21 | 12.17 | 1.94 |

   For linear the list is 7 and 22, 1 and 7, 1 and 22, 6 and 22, 5 and 22.
   Each of these layers is tolerable alone and catastrophic with its
   partner. They sit at distances 15-21, where each distance has only one
   to seven pairs, which is why a plot of the mean against distance appears
   to "explode" there. With layers 1 and 22 left out, the six pairs at
   distance 17 or more have a median interaction near zero (+0.1% for
   SwiGLU, −2.9% for linear).

3. **The allocator does reach these pairs, at exactly two targets.** The
   widths it chose, read from `swiglu-8/prepare-001/result.json` **[ran:
   offline read]**, as a percentage of the original width:

   | Target | Layer 5 | Layer 6 | Layer 7 | Layer 22 |
   | ---: | ---: | ---: | ---: | ---: |
   | 20-60% | 62 to 25 | 100 to 25 | 100 to 25 | 100 (dense) |
   | 70% | 25 | 12 | 12 | 100 (dense) |
   | 75% | 25 | 6 | 12 | 75 |
   | 80% | 6 | 6 | 6 | 50 |

   Layer 22 stays dense up to 70% and is narrowed only at 75% and 80%,
   together with layers 5-7 at their narrowest. Those are the two targets
   where SwiGLU-8 recorded an unexplained jump in the fit-versus-slice gap
   (23.5% and 18.4%, against 10-13% at every other target; your note lists
   it as "the 75%/80% gap anomaly ... untested"). **Hypothesis, not a
   result:** the anomaly is this interaction, which the unfitted arm
   suffers more. The fitted models show no collapse (real over predicted
   damage rises smoothly, 1.49, 1.53, 1.55), so good local fits appear to
   defuse it. A direct test is cheap: at 80%, keep layer 22 dense, re-solve
   the rest, and compare both arms before retraining.

**Limits.** The study's operators are not the allocator's: 12,288
calibration pairs, a randomly initialized SwiGLU at one width, and a linear
map (the limits your note already lists). Whether teacher-initialized,
well-fitted SwiGLUs at realistic widths show the same fragile pairs is
untested, and point 3 suggests the effect is much weaker for them.

**Why it matters.** This is a mechanism-level result that is sitting unused
in an artifact: a few early layers (1, 5-7) and the last eligible layer
depend on each other in a way no single-layer price can see. It is the
right form of your idea of adding rules to the allocator: the rules would
be about specific pairs and about neighbours, not about distance. It also
gives the ranking test a concrete prediction to check: allocations that
narrow layer 22 together with layers 5-7 should be much worse than the sum
predicts. If only a few layers interact strongly, the exact solve can be
kept by enumerating the choices for those layers and running the dynamic
program on the rest. **That last step is a project-proposed idea, not a
source-derived method.**

### B. Sibling-model anchors (rank 2)

Covered in [file 2](02-reporting-and-speed.md), Section 2.4. Evaluation
only. It is listed here because it also disciplines future work: any plan to
"push compression further" should first say where it would sit relative to
the next smaller model.

### C. A dense 2-bit baseline (rank 3)

Covered in [file 1](01-direction-and-design.md), Section 1.6. One
post-training conversion, one 100M quantization-aware run, and, if it is
competitive, one 1B run with the decay Recovery-2 selected for dense INT3.
Check library support for 2-bit weights first.

### D. Width and bits in one allocator (rank 6)

**What.** Today stacking applies one bit-width to every layer and lets the
knapsack choose only widths. Section 9 of your width-allocation note already
writes down the alternative: each layer's option is a pair (width, bits),
the budget is bytes, and the same exact solve applies. Dense quantization
(full width, low bits everywhere) and plain SwiGLU (16 bits everywhere)
become two corners of one menu.

**Why it matters.** It is the principled form of the stacking chapter. It
answers "is it better to drop a bit or to drop neurons, layer by layer?"
with the project's own machinery, and it would show *where* in the network
each kind of saving is cheap.

**The trap, and a fix that uses data you already have.** Prices measured
before retraining will mislead this solve, because retraining repairs the
two kinds of damage very differently **[read/calc from the snapshots]**:

| Damage | Before retraining | After 100M-1B tokens | Repaired by about |
| --- | ---: | ---: | ---: |
| Rounding to 4 bits, dense | KL 0.090 | 0.029 | 3.1 times |
| Removing 20% of eligible width | KL 0.111 | 0.061-0.079 | 1.4-1.8 times |

The width row is indicative only: its "before" value is the 128-token
monitoring KL and its "after" values are 8K KL, because no 8K value at
token 0 is in the pages I read. The direction is not in doubt.

A solver that sees only the left column will under-use quantization. A
simple correction is to discount each option's price by its family's
measured repair factor at the intended budget. **This is a project-proposed
heuristic, not a source-derived method**, and it would need its own check.
It is also the general answer to the question Heterogeneous-1 raised: can
prices measured before retraining be trusted?

**Cost and risk.** New price tables (each width at each bit-width), then
three or four retraining runs. Mixed bit-widths per layer are prior work
(for example HAWQ-V3, unregistered in your wiki), and they complicate any
later speed claim. I would do this only after A-C, and only if the dense
2-bit result leaves a region worth contesting.

### E. What does retraining repair? (rank 5)

**What.** An evaluation-only study on models you already keep. For a fixed
set of inputs, record how far each layer's hidden state is from the
teacher's, layer by layer, for: fitted and slice-only models at 0, 100M, and
1B tokens; orders A and B; a dense quantized model.

**Why it matters.** Four of your findings are observations without an
account: damage adds up faster than the sum of its parts (F10); the
fit-versus-slice gap closes (F5); order B beats order A (F8); and the fit
gap nearly vanishes under INT3. A depth profile of the error would show
whether retraining removes error where it is created or compensates
downstream, whether order B's trained blocks absorb the rounding error of
the frozen layers (the explanation your notes give and label untested), and
where super-additivity comes from.

**Why it fits you.** It is the thread you started in August (activation
geometry, the global-to-local ratio, interaction windows) and dropped when
model-level runs took over. It costs a few GPU-hours and no new training.
For a thesis, an explanation is worth more than another row.

### F. Peak learning rate for the BF16 families (rank 8)

Heterogeneous-2 measured +9.8% for SwiGLU 80% at 1e-4. The 2026-11-07 row
logged this as "a limit of the recovery recipe" and left a higher-peak
screen as a separate decision. It is tuning, which you said you do not want.
I list it for one reason: dense INT3 now has a schedule worth 35% and SwiGLU
has an untuned rate worth at least 10%, so the comparison between them is no
longer between equally tuned recipes. One short screen (two rates, 100M
tokens, 20/50/80%) is enough to either fix that or state it precisely.

### G. One replication on a second model (rank 7)

You logged this as a low-priority bonus on 2026-11-07. I would raise its
priority above further SmolLM2 experiments **[view]**: one model is the
standard objection to a compression study, and you now have a pipeline that
runs end to end.

- **Scope:** the final pipeline only (allocate, slice, retrain 100M, the
  three quantization formats, order B at one target), not the nine-step
  history.
- **Choice of model:** prefer one where MLPs are a larger share of the
  model, because that is where an MLP-only method has the most room. SmolLM2
  has an MLP four times as wide as its hidden size. Some families of similar
  size use a larger ratio **[lit: recalled, verify the exact configurations
  before choosing]**.
- **What it buys:** the sentence "the ordering of methods held on a second
  model family", or the discovery that it did not.

### H. One retraining run that yields every budget (rank 9)

**What.** Neuron selection ranks each layer's neurons once, so widths are
nested: the 50% model's neurons are a subset of the 80%-retained model's,
layer by layer, whenever the allocations are nested. Retraining could sample
a different target at each step and train all of them inside one set of
weights. One 1B-token run would then produce a family of models from 20% to
80%, instead of one run per target.

**Prior work.** Nested-width training of feed-forward blocks exists
(MatFormer; Flextron) **[lit: recalled, verify and register]**. What would be
yours is applying it to the *retraining of a pruned pretrained model* with
allocator-chosen, non-uniform nested widths.

**Why it is interesting.** It attacks the cost that shaped your whole
design (the "20/50/80 only at 1B" rule exists because each target needs its
own run), and it gives a deployable property: one checkpoint, several sizes.

**Risks.** Quality per target is usually a little below dedicated training.
The allocations at different targets must be made nested, which constrains
the solver. It does not fit the QAT runs. I would treat it as a stretch
goal; it is the most novel item on this list and the least certain.

## 3.4 Ideas from your own sessions that never reached the docs

The session mining ([file 6](06-session-findings.md)) recovered ideas you
raised and never ran. My verdict on each **[view]**:

| Idea (when) | Verdict | Reason |
| --- | --- | --- |
| Block-distance rules in the knapsack, from your reading of the pairwise plot in `block-interaction.ipynb` that damage stops adding up beyond a distance of about 17 layers (2026-10-01, 2026-10-25; you asked "is my understanding correct?") | The reading does not hold; a better idea replaces it | Checked offline on the stored artifact in this session (idea A, "What the stored interaction study already shows"). Interaction *falls* with distance. The apparent explosion at distance 17 and beyond comes from three to five specific pairs that include layer 22 or layer 1, in bins of one to five pairs. Rules about specific pairs and about neighbours are the form worth testing |
| Shapley values per block; higher-order interaction terms (2026-08-30) | Fold into E | Exact Shapley needs far too many evaluations for 22 layers. The useful core, "which blocks matter jointly", is what the ranking test and the depth profile measure |
| Giving each block its own calibration budget (2026-09-06, 2026-09-12) | Drop | Fitting stopped mattering after retraining (F5), so its budget cannot matter either, except with no retraining |
| Merging several low-importance blocks or sharing one MLP across them (2026-04-11) | Drop for this thesis | It is depth reduction with weight sharing, a different method class; the "zero" item already priced the cheapest version and lost |
| An "ideal smallest operator" per block (2026-07-21) | Reframe, do not run | Your width curves *are* the answer in the form "damage against size"; there is no threshold, only a trade-off. Say that in the thesis |
| Output variance by depth as a signal for width or bit-width (2026-08-11) | Subsumed by D | The allocator measures the thing the proxy guesses |
| Low-rank factorization as a third stage whose savings are given back to the allocator (2026-09-21, 2026-10-10) | Keep as "future work" text only | On bytes it will lose to quantization for the same reason width does; on compute it competes with width, not with bits. It adds a class without changing a headline |
| The TorchAO quantizer bug as an upstream report (2026-11-01) | Do it | A filed issue with a minimal reproduction is a concrete, citable side contribution and costs an hour. The draft exists and is unposted (your memory note). Posting is an outward action, so it is yours to do |
| A third-party reproducibility statement (2026-11-07) | Decide | The row was left undecided. One paragraph is enough: pinned inputs and seeds give results reproducible up to the measured GPU noise (0.3-3%) |

## 3.5 What I would not do

| Not this | Because |
| --- | --- |
| Order C (BF16 retraining, then a QAT stage) | Already deferred; it answers no thesis question |
| More operator families on the dense model | Heterogeneous-1 is decisive, and the cause is structural |
| More seeds across the board | The large effects do not need them. Spend repeats only where a conclusion rests on a few percent |
| 90% removal | The sibling anchor will very likely make this region moot |
| Kernel or engine work for quantized models | The can of worms; see [file 2](02-reporting-and-speed.md) |
| Re-opening local fitting | Closed with strong evidence |

## 3.6 A possible order of work

This is a suggestion for discussion, not a plan; each item needs its own
session and your go.

1. **This week, no GPU:** write up Heterogeneous-2 Step 1; decide the
   Puzzle corrections that have been pending since 2026-11-06; take the
   Recovery-2 table to the supervisor with the 3B question.
2. **Cheap protection (about 25-30 GPU-hours in total):** B, A, C, in that
   order.
3. **Explanation:** E, on the kept models.
4. **The one ambitious addition:** the mixture-of-experts step M1.
5. **Then freeze.** Pick a date after which no new experiment class starts.
   The decision log says writing starts this semester and the seminar needs
   one finished chapter; the related-work and method chapters can be written
   now, because none of the items above changes them.
