---
title: Deep Analysis Part 2 - A Design from the Annotation Alone
status: draft
created: 2026-11-13
modified: 2026-11-13
---

# 4. Starting from the annotation alone: a second design, compared with yours

Part of the [deep analysis, part 2](README.md). It covers your topic 3.
Evidence labels and flags are defined in the [README](README.md#labels-and-flags).

## Short answer

- **A design made from the annotation alone lands very close to what you
  built.** Same substitute families, same teacher-neuron start, same
  single-swap KL prices in a knapsack, same four retraining regimes, the same
  rivals. Your design is the natural one, not an accident of its history.
- **It differs in two places that are worth acting on.** It would test
  isolation directly, by fitting each block on the inputs it will really see
  after the blocks before it were replaced. And it would evaluate on text
  from outside the calibration domain.
- **Its blind predictions mostly came true,** including the one this
  session's offline check confirmed: local error predicts damage within a
  layer and not between layers.
- **It is not a clean experiment.** The second designer saw one-line hints
  about your project. Read it as a second opinion, not as proof.

## 4.1 How it was made, and its limit

I could not do this myself: by the time the question came up I had read the
whole project. So a fresh sub-agent received only the annotation and a
practical frame (one student, one large GPU per job, an informed opponent),
with file access and web search forbidden **[ran: one sub-agent, no tools]**.

**The limit.** The session attached your memory index to that agent
automatically. It did not open any file, but it saw the index's one-line
notes, and it said so itself. The hints it listed: SwiGLU substitutes, a
knapsack allocator, INT3 and INT4 stacking, the 100M and 1B budgets, Puzzle, a
super-additivity factor of 1.23-1.55, an allocator edge of 9-13% over uniform,
and a stalled linear + MLP hybrid. Its predictions (d) and (e) below are
therefore not blind. A truly clean run needs a session started outside this
project folder.

Its report was not saved to a file; the tables below are my summary of it.

## 4.2 Design choices, side by side

| Choice | From the annotation alone | Yours | Reading |
| --- | --- | --- | --- |
| Model | A 1.5B dense SwiGLU model, a 0.5B scout, one 7-8B check, a second family | SmolLM2-1.7B only | Same class. It plans the scale and family checks from the start |
| Substitutes | Zero or mean; linear and low-rank in closed form; narrow SwiGLU from teacher neurons; plain MLP at equal parameters; linear + SwiGLU on the residual; low-rank factorisation as a comparator | The same, without low-rank (by your stacking definition) | Converged |
| **Inputs for fitting** | Three regimes: (i) teacher input, teacher output; (ii) the *student's* input, after earlier blocks were replaced; (iii) a target that also corrects earlier error | (i) only: every block is fitted against the dense model, then all are inserted together **[read: fit keys `...:dense:...`; `run_one_shot_replacement`]** | **Different. See 4.4, item 1** |
| Local loss | Normalised MSE; alternatives: gradient-weighted MSE, a KL through the frozen rest of the model with only the block trainable | MSE (and L2-SP). The KL-through-the-model variant exists as Heterogeneous-1's single-block recovery, a diagnostic at three layers | Mostly converged |
| Allocation | Single-swap KL per option, exact knapsack; test additivity; if it fails, search with real evaluations; controls: uniform, random, protected ends | The same core; additivity measured, not acted on; no uniform or random control yet | Converged; your controls are missing (Part 1, W1 and W6) |
| Retraining | None, local only, 10-100M, 1B; KL to the teacher; only the substitutes trainable; cosine decay | The same regimes; constant rate until Recovery-1 and Recovery-2 | Converged |
| Metrics | KL and top-1 agreement; perplexity on two corpora **plus one outside the calibration domain**; a task suite; whole-model bytes; measured speed at batch 1 and 32 | KL; perplexity; five tasks; bytes; linear FLOPs | **Different: 4.4, item 2** |
| Rivals | Other structured pruning, layer dropping, low-rank, 3- and 4-bit with and without training at equal budget, **the natively trained smaller sibling**, replacement plus 4-bit | Quantization with and without training; stacking | Part 1 proposed the sibling too |
| Fairness | Equal bytes and equal latency; equal data and retraining budget; equal tuning per arm; three seeds on headline points | Equal bytes; equal tokens; equal tuning per operator family; one seed with a measured noise twin | Close |

## 4.3 Its predictions against your results

| Question | Its prediction (confidence) | What you measured |
| --- | --- | --- |
| Which substitute wins? | Narrow SwiGLU from teacher neurons above about 15% width; a hybrid may win on local MSE at very small sizes, not after training (medium) | SwiGLU wins 505 of 506 on KL. At 6.25% a plain MLP wins on local error in 13 cells and never on KL ([file 2](02-local-fit-as-selector.md)) |
| Does local MSE predict end-to-end quality? | Within a layer and family yes (rank correlation above 0.9); across layers poorly without a sensitivity weight (medium-high) | 0.995 within a layer; −0.14 to −0.31 across layers; fixed by one measured KL per layer |
| Does fitting matter after long distillation? | Under 5% at 1B tokens (medium) | 0.3-2.8% |
| Does starting from the teacher's weights keep mattering? | Yes: a random start stays 10-30% worse (medium) | **Not the case at high removal:** the forced 80% model, with eight plain MLPs that were randomly initialised and then locally fitted, was within 2.1% of SwiGLU after 100M tokens. Untested at 20-50% in this form |
| Replacement against 4-bit at equal memory? | Loses clearly; with 3-bit QAT "a coin flip" (low-medium; hinted) | Loses clearly; dense INT3 with a decay leads by 30% near 51% |
| Allocation against uniform? | 10-30% without retraining, under 10% after (medium; hinted) | 11-38% predicted without retraining; unmeasured after |
| Where is the real advantage? | Latency on ordinary kernels; combining with quantization; a forward-only path for models too large to fine-tune; cheap per-layer search (medium-high) | Compute and stacking, as Part 1 said. The "too large to fine-tune" use is a framing you have not used |

- **The local-versus-global prediction was not among the hints and was
  right.** The literature predicts it too ([file 2](02-local-fit-as-selector.md),
  section 2.6). So the result is expected, which makes it easier to defend
  and less of a surprise to claim.
- **The one miss is informative.** Inheriting the teacher's weights is worth
  less than an expert would guess once most of a block is gone. That agrees
  with a known result for structured pruning, that the pruned *architecture*
  matters more than the inherited weights after enough training **[lit: Liu
  et al., ICLR 2019, agent-read]**.

## 4.4 What it would do that you have not

Ranked by what I think they are worth **[view]**.

| # | Idea | Why it matters | Cost (rough) | Flags |
| --- | --- | --- | --- | --- |
| 1 | **Fit each block on its real inputs** (sequentially, after the earlier blocks were replaced), and compare with today's parallel fit, before any retraining | It is the annotation's "isolated function" premise as an experiment. Your joint damage is 23-55% above the sum of single-block damages; sequential fitting is the standard remedy for exactly that compounding, and published work reports that the choice of inputs changes the outcome **[lit: Shin et al. 2406.15524, agent-read]** | Fits only: about 22 fits per target plus token-0 evaluations, a few GPU-h | DECISIVE without retraining, probably NEUTRAL after 1B (by your own fit-versus-slice result). TRANSFERABLE HYPOTHESIS |
| 2 | **Evaluate on text outside the calibration domain** (code, or another language) | Every stream you score on is C4, WikiText, or English tasks. A substitute fitted on C4 may be worse off-domain than a quantized model that was never fitted to anything. Nobody has looked | Evaluation only on kept models, under 1 GPU-h | PROTECTIVE; the *direction* of the result is unknown |
| 3 | The natively trained sibling as a rival | Proposed independently of Part 1 | Under 1 GPU-h | PROTECTIVE (Part 1, idea B) |
| 4 | Top-1 agreement beside KL | A fidelity number a reader can picture ("the model picks the same next token 9x% of the time"); it tracks KL closely in published work **[lit: Dutta et al. 2407.09141, agent-read]** | Free with the KL evaluation | INCREMENTAL, reporting only |
| 5 | Random allocation as a control | Shows how much of the knapsack's edge is "any non-uniform choice" | Evaluation only at token 0 | PROTECTIVE; folds into file 2's step 1 |
| 6 | Published structured-pruning methods on your model | The same-class baseline | Open-ended (foreign code) | COSTLY; Part 1 already called it a bonus |

Item 1 is the one I would add to the local-fit study. It needs no new
concept: the same fits, with inputs captured from the partly replaced model.
It also matters most where you are heading next, because the
mixture-of-experts comparisons use no retraining.

## 4.5 What you did that it did not think of

These are the parts of your work that a competent fresh design lacks
**[view]**:

- The **bitwise check between training-time and deployed weights**, which
  found two real library mismatches.
- The **paired fit-versus-slice design** with a noise twin and a token
  equivalent.
- **Selection data kept apart from evaluation data.**
- The **trunk-and-branch schedule screens.**
- **Order A against order B** as a designed comparison.
- The **forced-model degradation curve** when the solver declined to pick.

## 4.6 My own view from the annotation

If I had only the annotation, I would build the thesis around its one testable
sentence, that a block can be treated "as an isolated function" **[view]**.
That gives three measurements in order: does local error predict global
damage (file 2), does fitting in isolation cost anything against fitting in
context (item 1 above), and does either still matter after retraining (your
SwiGLU-8 and SwiGLU-9). You built the machinery first and are returning to the
hypothesis now. Nothing was lost by that order: the data for the first
measurement already existed, and the third is done.
