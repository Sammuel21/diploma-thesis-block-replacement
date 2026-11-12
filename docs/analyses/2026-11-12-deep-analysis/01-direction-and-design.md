---
title: Deep Analysis 2026-11-12 - Direction and Experiment Design
status: draft
created: 2026-11-12
modified: 2026-11-12
---

# 1. Direction and experiment design

Part of the [deep analysis of 2026-11-12](README.md). It covers your topics 1.1
(design of the current work), 1.3 (is it a dead end), and the summary items
"directional quality", "creative quality", and "experiment classes".

Evidence labels used in every file of this analysis:

| Label | Meaning |
| --- | --- |
| **[read]** | I read the cited file in this session |
| **[ran]** | A script or command executed in this session |
| **[calc]** | Arithmetic on numbers taken from the docs; nothing was run on a GPU |
| **[lit]** | Literature. "checked" means an abstract-level web check in this session; "recalled" means from my memory and to be verified before citing |
| **[view]** | My judgment |

## Short answer

- **The work is rigorous, and the direction is sound, but the thesis is not
  the thesis the annotation describes.** The annotation asks whether a small
  trained substitute can stand in for an MLP block. Your experiments answer:
  the best substitute you found is a piece of the original block (its most
  active neurons), the local training gives a head start that disappears with
  enough retraining, and no other kind of block beats it. That is a real
  finding. It should be stated as the main result, not hidden behind frontier
  plots.
- **On bytes alone, the structural track is finished.** Narrower MLPs cannot
  catch quantization on storage, and no experiment in the current plan can
  change that. Further gains there are tuning.
- **The design is not a dead end, because bytes are only one of the things
  the method buys.** Three directions still have room for results that would
  change what the thesis can say: compute (the only axis where replacement
  wins alone), a joint choice of width and bit-width in one allocator, and
  mixture-of-experts models, where blocks really do differ in how much they
  are used.
- **The biggest risk is not scientific.** It is that experiments keep being
  added at the frontier's most contested point while the method's own
  assumptions (the allocator, the protected layers, one model, one seed) stay
  unchecked, and while writing has not started.

## 1.1 What the work has established

Each row is my reading of the experiment pages and the 2026-11-01 results
snapshot **[read]**. "Strength" is my view of how well the evidence would hold
up against an opponent.

| # | Finding | Where | Strength |
| --- | --- | --- | --- |
| F1 | Starting a narrower SwiGLU from the teacher's most active neurons beats a random start by a wide margin | Operator distillation, SwiGLU-1 | Strong inside the project; one layer and one seed in the original test |
| F2 | Measured single-block damage is a better guide for distributing width than block-influence scores | SwiGLU-2, SwiGLU-5 | Moderate: the comparisons changed more than one thing |
| F3 | Whole-model retraining repairs a lot and keeps repairing up to 1B tokens, but never reaches the dense model | SwiGLU-3, 6, 7 | Strong |
| F4 | Training only the replaced blocks is as good as training the whole body | SwiGLU-7 | Moderate: within noise at 100M, one seed |
| F5 | Local fitting matters without retraining (2-5 times lower KL), matters at 100M tokens (10-24%), and no longer matters at 1B (0.3-2.8%) | SwiGLU-8, SwiGLU-9 | Strong: paired runs, twin noise 0.31% |
| F6 | Quantization keeps more quality per byte than replacement at every size both reach | Quantization-1/2 against SwiGLU-8/9 | Very strong: the gaps are 2-10 times |
| F7 | Replacement followed by quantization reaches sizes that the tested quantization-only models do not | Stacking-1 | Strong for what was tested; see the missing control in Section 1.6 |
| F8 | Quantizing during retraining (order B) beats quantizing after it (order A) by 24-52% | Stacking-1 | Strong |
| F9 | No other block type (zero, mean, linear, plain MLP, two hybrids) beats a SwiGLU of the same size at any layer | Heterogeneous-1 | Strong at block level; 505 of 506 comparisons |
| F10 | Summed single-block damage underestimates the real joint damage by 23-55%, more as more is removed | Width-allocation note, Heterogeneous-1 | Strong as a measurement; its effect on the allocator's choices is untested |
| F11 | A learning-rate decay helps quantization-aware training a lot (35% for dense INT3 at 1B) and helps replacement little | Recovery-1, Recovery-2 | Strong for dense INT3; marginal for SwiGLU |
| F12 | The library's training-time quantizers did not match its deployed ones (INT3 badly, INT4 slightly); fixed and re-run | Quantization-1/2 | Strong, and a contribution in its own right |

Twelve findings from eight months is a lot. F5, F6, F9, and F12 are the ones I
would build the thesis around.

## 1.2 What the thesis actually is

The annotation's first goal is to "replace individual MLP blocks with smaller
substitutes ... (e.g., shallower MLPs, linear layers, or hybrids) ... using
local calibration data". Read against that sentence, your results say three
things **[view]**:

1. **Replacement converges to pruning.** Every substitute that was built
   from scratch lost to the one built by keeping teacher neurons (F1, F9).
   Keeping neurons and then training on calibration pairs is what the
   literature calls structured width pruning with reconstruction. Your own
   width-allocation note already says so: "the word 'replacement' alone does
   not separate it from ZipLM".
2. **The calibration step is a head start, not a lasting gain.** F5 gives a
   clean, monotone picture: the value of local fitting falls as the
   retraining budget rises. Without retraining it is large. With 1B tokens
   it is gone.
3. **Structure and precision are different kinds of saving.** Fewer bits
   save bytes cheaply and save no computation. Fewer neurons save
   computation and cost a lot of quality per byte (F6). Together they reach
   further than either (F7).

These three are a coherent thesis, and they answer the question you left open
on 2026-09-04, "why not just structured pruning?" **[read: session mining,
L2]**. The answer is: *that is what was tested, and within these operator
families and budgets, a trained substitute does not beat a pruned one once the
model is retrained.* An opponent cannot use that question against you if the
thesis asks it first.

What the thesis is **not** is a new compression method that beats existing
ones. The allocator is prior work (SPDY, ZipLM, and above all Puzzle, per your
own 2026-11-06 check). The operator is pruning. The recovery is standard
distillation. The contribution is the controlled evaluation, which is exactly
the annotation's second goal: "test and evaluate the improvements and
trade-offs of this replacement strategy".

**Recommendation.** Decide now that the thesis is an evaluation thesis with
three headline results (replacement converges to pruning; fitting's value
decays with retraining; structure and precision trade off on different axes).
Then every further experiment is judged by one test: does it change, sharpen,
or generalize one of those three sentences? Section 1.6 and
[file 3](03-future-work.md) apply that test.

## 1.3 Directional quality

### What is strong

- **Paired comparisons.** Both arms of a comparison use the same seed and
  the same ordered token stream, so differences come from the thing that was
  changed. This is why the one-seed design is defensible for large effects.
- **Selection data is separate from evaluation data** since 2026-10-11. Most
  student work, and a good share of published work, does not do this.
- **Rules are fixed before results.** The adoption rule, the verdict bars,
  and the winner rules were all written down first. The SwiGLU 20% decay
  result (wins by 0.16 points, reported as marginal instead of overridden)
  shows the rule was respected when it was inconvenient.
- **Fidelity gates.** Bit-exact reproduction of earlier runs before each new
  run is a stronger correctness check than code review. It caught the INT3
  quantizer bug's cousins and it is the reason I trust numbers I did not
  re-derive.
- **Negative results are kept.** "Nothing adopted", "no hybrid picked", and
  "closed at 1B" are recorded as results. That is the right culture for an
  evaluation thesis.
- **Noise is measured.** Twin runs give a real noise floor (0.31% for BF16,
  up to 3% for QAT), and results inside it are called unresolved.

### What is weak

| # | Weakness | Why it matters | Cheap fix? |
| --- | --- | --- | --- |
| W1 | **The allocator was never compared with a uniform allocation under the final recipe.** SwiGLU-2 beat uniform with a ranked rule and one epoch of retraining; SwiGLU-5 beat the ranked rule with the knapsack. No run compares knapsack and uniform directly. | The method chapter is built on the allocator. "Does it help?" is the first question an opponent asks. An offline calculation in this session shows its *predicted* edge over uniform is only 9-13% from 50% removal upward, before retraining ([file 3](03-future-work.md), idea A) | Yes: three to six 100M runs, about 4-9 GPU-hours (rough) |
| W2 | **The allocator's additivity assumption is known to be wrong in size, and unknown in effect.** You measured 1.23-1.55 times super-additivity (F10). Whether it changes which allocation wins was proposed as a "ranking test" on 2026-10-17 and never run. Heterogeneous-1's forced models are the only partial evidence. | Published work (EvoPress, **[lit: recalled, in your note as an unregistered lead]**) claims the additive solve can pick clearly worse allocations. | Yes: assembly and evaluation only, no training |
| W3 | **The frontier above 51.6% is uncontested by construction.** No quantization-only model below 3 bits was tested, so "only stacked models exist there" is true because nothing else was tried. | F7 is the thesis's main positive claim about stacking. See Section 1.6. | Yes: a dense INT2 baseline, about 15 GPU-hours (rough) |
| W4 | **One model, one seed.** Logged as a limit. For 2-10 times effects this is fine. For the claims that rest on 3-6% differences (decay winners, the 80% forced model "level") it is thin. | External validity is the standard criticism of single-model compression studies. | Partly: one replication of the final pipeline on a second model |
| W5 | **The retraining recipe was tuned on one family and one budget.** The 3e-5 rate came from SwiGLU-4 (two rates, short runs). Heterogeneous-2 then found SwiGLU 80% gains 9.8% at 1e-4, more than local fitting is worth at 1B. Dense INT3 meanwhile received a schedule worth 35%. | The comparison between families is no longer on equally tuned recipes. The ordering does not change (gaps are far larger), but the "blue dot" is understated. | Yes: the peak-rate screen that the 2026-11-07 row left as "a separate decision" |
| W6 | **Protected layers are assumed, not measured.** Layers 0 and 23 are excluded because a *linear* replacement of the first block spiked the loss in the MVP **[read: L1, L2]**. No SwiGLU width was ever priced there. | The allocator has a "keep dense" option, so it could decide this itself. | Yes: two more width curves |
| W7 | **No external anchor.** Every model on the frontier is a derivative of one teacher. A natively trained smaller model of the same family is never shown. | Without it, the reader cannot tell whether a 60% saving is worth having. See [file 2](02-reporting-and-speed.md). | Yes: evaluation only |
| W8 | **Little mechanism.** The work says which option is better far more often than why. Order B's advantage, the vanishing fit gap under INT3, and the growth of super-additivity all have untested explanations. | Mechanism is what turns a table of results into a thesis discussion. | Partly: evaluation-only analyses on kept models |

None of these is a flaw in what was run. They are gaps around it. W1, W2, W3,
and W7 are cheap and each one removes a question you would otherwise have to
answer at the defence with "we did not test that".

## 1.4 Creative quality

**What is inventive here [view]:**

- The **fit-versus-slice design** with identical streams, a noise twin, a
  closing point, and a token-equivalent that subtracts the fit's own cost.
  The reframing from "does fitting matter?" to "how many tokens does it
  save?" was yours (2026-10-18) and it is the better question.
- The **trunk-and-branch design** of Recovery-1 and Recovery-2: one constant
  run, with decays branching from saved states. It compares schedules from
  an identical state at about half the cost.
- The **degradation curve** in Heterogeneous-1: when the solver refuses to
  pick new blocks, force 1, 2, 4, and 8 of them and measure the price. You
  proposed it before the run, and you also predicted the outcome.
- The **bitwise deployed-path check** for quantization-aware training. It is
  a small idea with a large effect: it found two real library mismatches.
- **Linear FLOPs as "compute work, not speed"**: an honest way to credit
  structure without opening the kernel question.

**What is missing creatively [view]:**

- The experiments are almost all *horse races*: A against B on a metric. The
  early work (activation geometry, the global-to-local ratio, interaction
  windows) asked *why*. That thread stopped in September. Your session logs
  show ideas in this direction that were never run: Shapley-style block
  values, higher-order interaction terms, and distance-dependent
  non-additivity **[read: L1, L3, L5]**.
- One such result is already sitting in a stored artifact. An offline check
  in this session on the pairwise interaction study found that damage
  compounds most between neighbouring layers and fades with distance, and
  that a few specific pairs (layer 22 with layers 1, 5, or 7) collapse the
  model although each layer is tolerable alone. Details and limits are in
  [file 3](03-future-work.md), idea A. This is the kind of finding the
  horse races cannot produce.
- Each experiment is designed carefully in isolation, but the set was not
  designed as a whole. The result is depth at a few points (20/50/80% at 1B,
  the 51% crossing) and no ablation of the pipeline's own parts (W1, W2, W6).
- The operator search stayed inside "smaller versions of an MLP". The one
  setting where different block types have a natural reason to exist, a
  model whose blocks are used unequally, is still a bonus item.

## 1.5 The experiment classes

| Class | What it holds | Opinion |
| --- | --- | --- |
| `block/` baseline, operator, analysis | Single-block studies in notebooks | Good material for a "block level" chapter. Frozen. The operator screen was superseded by Heterogeneous-1's better-trained version; say so in the thesis so the two are not read as contradicting |
| `model/swiglu` | SwiGLU-1 to 9 | The numbering records history, not questions. For the thesis, regroup by question: initialization (1), allocation (2, 5), calibration and fitting recipe (3, 8A, 9A), retraining recipe (4, 6, 7), fit-versus-slice (8B, 9B). The progression page already half-does this |
| `model/baseline` | Quantization-1/2, Speed-1, an old compression baseline | Quantization is no longer a "baseline" in the story; it is the competing axis. The old Grafting-style baseline is the first version of the method, not a pruning baseline (your 2026-11-03 session reached the same conclusion) |
| `model/stacking` | Orders A and B at 100M and 1B | Well defined ("two or more kinds of redundancy reduction"). Its natural next step is not order C or more tokens; it is letting the allocator choose bits as well as width ([file 3](03-future-work.md), idea D) |
| `model/recovery` | Schedule screens | Right to be its own class, because it cuts across all families. Its results change the baselines, so it should be read as "recipe", not as a new result family |
| `model/heterogenous` | Heterogeneous-1 and 2 | The block-level study is thorough and its answer is clear. See Section 1.6 |
| `model/interaction` | One notebook skeleton | Never finished. It is where W2 belongs |

**Two classes are missing [view]:**

1. **Validation of the method's own assumptions**: allocator against uniform,
   the ranking test, the protected layers, the interaction study. These are
   scattered or absent. They are cheap and they are what a methods chapter
   needs.
2. **External anchors**: sibling models and, where possible, a published
   training-free method run on your model. See [file 2](02-reporting-and-speed.md).

## 1.6 Is the current design a dead end?

Track by track **[view, with the cited numbers read or calculated]**.

### SwiGLU alone, on bytes: yes

SwiGLU 20% at 1B tokens saves 12.9% of bytes at 8K KL about 0.06. INT8
without any training saves 32.3% at KL 0.003. No recipe change closes a gap
of twenty times in KL at two and a half times the saving. The one remaining
lever with a measured effect above 5% is the peak learning rate (+9.8% at
80%, W5). It would move the blue dot. It would not move it onto the size
frontier.

### Local fitting: closed, and that is fine

SwiGLU-9 answered it. The useful move now is presentational: report it by
retraining budget (none, 100M, 1B), because without retraining it is the
largest effect in the whole project.

### Other block types on a dense model: closed at block level

Heterogeneous-1 is decisive: at equal size, SwiGLU wins 505 of 506
comparisons, and the linear part "never starts to pay". Heterogeneous-2
Step 1 then showed forced models 17% and 8% worse at 20% and 50%, and about
2% worse at 80%.

The reason is structural, not a tuning accident **[view]**. In a dense model
every MLP sees every token, and a slice of the teacher inherits a trained
function that a fresh operator has to relearn. A plain MLP also spends its
parameters less well than a gated one at equal count. I would not expect
more retraining to reverse a 1.1-1.5 times disadvantage that recovery did not
close at block level.

By your logged rule the 80% forced model goes to 1B (Step 2). My view is
that its expected outcome is "level within noise", which adds a sentence but
not a finding. It is your rule and a cheap run (about 10-12 GPU-hours), so
this is a judgment call, not an error. Step 3 (three scopes, no-fit test,
about 170 GPU-hours at most) I would not run unless Step 2 shows a clear win.

### Stacking near 51%: Recovery-2 has already answered the 3B question

The plan after the 2026-11-06 meeting is to train B 20% INT4 and dense INT3
to 3B tokens and see whether stacking can beat quantization alone near 51%
saving. Recovery-2's own results, which arrived after that plan was made,
bear directly on it **[read: recovery-2.md, results table]**:

| At 1B tokens, best schedule, selection-stream 8K KL | Byte saving | KL |
| --- | ---: | ---: |
| Dense INT3 QAT, 50% decay | 51.6% | 0.0542 |
| B 20% INT4, 20% decay | 50.9% | 0.0778 |

Dense INT3 now saves more bytes *and* is 30% closer to the teacher. For
stacking to win at 3B, B would have to improve about 30% more than dense
does over the extra 2B tokens. B's own trunk was falling about 0.3% per 25M
tokens at the end of 1B **[read]**, and a decay bought it 5.6% where it
bought dense 35%. I put the chance that 3B reverses the order below one in
ten **[view]**.

The two 3B runs cost about 76 GPU-hours of training (41 for dense, 36 for B,
from the measured 1.36 and 1.19 hours per 100M tokens **[calc]**). GPU time is
not your scarce resource; your attention and the calendar are. The supervisor
asked for this, so it is his call and yours. What I would bring to the
2026-11-13 meeting is the table above and this question: *given that the
schedule study moved dense INT3 from 0.071 to 0.054 and B from 0.081 to
0.078, is the 3B comparison still the best use of the next two weeks?*

### Stacking above 52%: the interesting part, with a missing control

Above 51.6% the frontier is all stacked models. That is the thesis's best
positive result about combining methods. It has one gap (W3): the cheapest
way to save more bytes than INT3 is not to remove neurons, it is to use
2 bits.

Byte saving for weight-only quantization of the eligible MLPs, with the
project's convention of 0.25 extra bits per weight for group scales
**[calc; standard bookkeeping, no citation needed]**:

$$
\text{saving}(b) = f\cdot\Big(1-\frac{b+0.25}{16}\Big), \qquad f = 0.647
$$

where $f$ is the eligible MLPs' share of model bytes. It reproduces your
numbers (47.5% at 4 bits, 51.6% at 3 bits) and gives:

| Format | Calculated saving | What sits near it today |
| --- | ---: | --- |
| 2 bits | 55.6% | B 20% INT3 1B (54.2%, KL 0.128), B 50% INT4 1B (56.1%, KL 0.173) |
| Ternary (1.58 bits) | 57.3% | B 50% INT3 1B (58.1%, KL 0.215) |
| 1 bit | 59.6% | A 60% INT3 (59.4%) |
| Nothing left (ceiling) | 64.7% | B 80% INT3 1B (62.1%, KL 0.435) |

So a dense 2-bit model would land exactly in the region where stacking is
currently unopposed. Whether it wins there is open. Dense KL went from 0.030
at 4 bits to 0.071 at 3 bits under the constant recipe. Published work finds
a qualitative break between 3 and 2 bits, where the model has to change its
representations and needs far more training **[lit: ParetoQ, NeurIPS 2025,
abstract checked]**. A dense 2-bit model could come out anywhere between
about 0.15 and 0.4. That uncertainty is the point: unlike the 3B run, this
experiment has an outcome I cannot predict, and either outcome changes a
headline sentence.

- If dense 2-bit loses to the stacked models: F7 becomes a strong claim
  ("removing neurons is a better way to pass 52% than removing the third
  bit").
- If it wins: stacking's range shrinks to above about 56-58%, and the thesis
  says so.

Either way the claim becomes defensible. Whether the pinned library supports
2-bit weights the way it supports 3-bit has to be checked first, as the
2026-10-17 row required for INT3.

### Where the real headroom is

| Direction | Why it is not a dead end | Detail |
| --- | --- | --- |
| Compute | It is the only axis where replacement wins without help, and it has only been counted, not exploited | [File 2](02-reporting-and-speed.md), Sections 2.4-2.5 |
| Width and bits in one allocator | It turns stacking from "apply one bit-width to everything" into an allocation problem your tooling already solves, and it includes dense quantization as a special case | [File 3](03-future-work.md), idea D |
| Mixture-of-experts models | Experts are used unequally, so a mixed menu has a reason to exist; the published baseline needs no retraining, so your results would be directly comparable with a paper for the first time | [File 3](03-future-work.md), Section 3.2 |
| Explaining what retraining repairs | Uses only the models you already keep; turns F3, F5, F8, and F10 from observations into an account | [File 3](03-future-work.md), idea E |

## 1.7 One page for the supervisor

If I had to put this file on one slide:

1. The thesis has three results: replacement converges to pruning; the
   calibration step is a head start whose value falls with retraining;
   precision saves bytes and structure saves compute, and they combine.
2. Four cheap checks protect those results: knapsack against uniform, the
   ranking test, a dense 2-bit baseline, and sibling models as anchors.
3. Recovery-2 already suggests how the 3B comparison ends.
4. The mixed-operator idea failed on a dense model for a structural reason,
   and that same reason says where it should work: mixture-of-experts.
