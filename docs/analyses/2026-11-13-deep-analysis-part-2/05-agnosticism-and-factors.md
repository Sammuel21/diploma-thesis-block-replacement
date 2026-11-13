---
title: Deep Analysis Part 2 - What Transfers and What Makes a Real Difference
status: draft
created: 2026-11-13
modified: 2026-11-13
---

# 5. What transfers to other models, and what makes a real difference

Part of the [deep analysis, part 2](README.md). It covers your topics 4
(model agnosticism) and 6 (differentiating factors). Evidence labels and flags
are defined in the [README](README.md#labels-and-flags).

## Short answer

- **Neither question is a dumb one, and they are the same question from two
  sides.** "Does it transfer?" asks whether a finding survives a change of
  model. "Is it differentiating?" asks whether it survives a change of
  budget and of tuning effort.
- **With one model, nothing is shown to be model-independent.** What you can
  do is sort every claim by *why* it should transfer. Four levels are enough,
  and they are the agnosticism flags used in this analysis.
- **Your own results already rank the factors.** What kind of redundancy is
  removed, and in which order the stages run, changes results by 1.3 to 10
  times and persists with training. How cleverly things are chosen and
  initialised changes them a lot without retraining and by a few percent
  after 1B tokens.
- **The label "hyperparameter" does not predict the size.** The largest
  recipe effect you measured (a learning-rate decay for dense INT3, 35%) is
  bigger than most method effects, and it decided the contest near 51%.

## 5.1 Four levels of transfer

| Level (flag) | Meaning | How sure can you be |
| --- | --- | --- |
| STRUCTURAL | True by construction for any transformer with residual MLP blocks | Certain; it is arithmetic or a definition |
| TRANSFERABLE HYPOTHESIS | A mechanism says it should hold, published work reports it on other models, and you measured it on one | Likely in direction, unknown in size |
| FAMILY-SPECIFIC | Depends on a property that differs between architectures (dense or MoE, gated MLP, the MLPs' share of parameters) | Holds within the family, must be re-derived outside |
| MODEL-CONCRETE | A number or setting found on SmolLM2-1.7B | Assume it does not transfer |

### Your claims, sorted

| Claim or component | Level | Why | Cheapest test elsewhere |
| --- | --- | --- | --- |
| Weight-only quantization saves bytes and no arithmetic; narrower MLPs save both | STRUCTURAL | Counting | None needed |
| The byte formula and the ceiling (64.7% here) | STRUCTURAL form, MODEL-CONCRETE value | The eligible share differs: over 90% in MoE experts **[agent]** | Recompute the share |
| The exact knapsack over a per-block menu | STRUCTURAL | An algorithm | None needed |
| Within a block, damage is proportional to local error | TRANSFERABLE HYPOTHESIS | Second-order approximation ([file 2](02-local-fit-as-selector.md), 2.6) | Price one layer at three widths |
| Local error cannot rank layers | TRANSFERABLE HYPOTHESIS | Published allocators all add a per-layer weight **[agent]** | The same three-width probe on a few layers |
| Quantization keeps more quality per byte than width removal | TRANSFERABLE HYPOTHESIS | A published head-to-head study agrees **[lit: Kuzmin et al. 2023]** | One PTQ model against one sliced model |
| Fitting and selection signals stop mattering after long retraining | TRANSFERABLE HYPOTHESIS | Reported repeatedly, including in MoNE's own fine-tuning table **[agent]** | Hard to test cheaply; cite |
| Training with quantization simulated beats quantizing afterwards (order B over A) | TRANSFERABLE HYPOTHESIS | Standard QAT reasoning | One pair of runs |
| A decay helps QAT much more than BF16 retraining | TRANSFERABLE HYPOTHESIS | Low-bit weights oscillate at a constant rate (your registered Nagel source) | One branched run |
| Teacher neurons beat a fresh block of equal size | FAMILY-SPECIFIC (dense) | Every block sees every token; a slice inherits a trained function | In MoE: per expert, on its routed tokens |
| No other operator type is picked | FAMILY-SPECIFIC (dense), by your own logged wording | Blocks are used equally | The MoE chapter is this test |
| Damage adds up 1.23-1.55 times faster than the sum | MODEL-CONCRETE size, TRANSFERABLE direction | Published work finds additivity fails under aggressive compression **[agent]** | The token-0 comparison at one target |
| Layers 1, 7, 22 behave differently; layers 0 and 23 are protected | MODEL-CONCRETE | Tied to where this model writes its very large activations (hypothesis) | The per-position check of file 2 |
| 3e-5, the 20% and 50% decays, the crossing at 51.6%, the 3.5% and 5% bars | MODEL-CONCRETE | Tuned or observed here | Re-tune per model |

### What a few more models can and cannot give you

- **One model:** confidence comes only from the mechanism and the literature.
- **Two or three models:** you can replicate *orderings*, not numbers. Three
  agreeing models are weak statistics and still the field's normal standard.
- **A practical form for the "meta-analysis" of topic 1.6.** Before running
  anything on a second model, write down six to eight ordinal claims from the
  table above (for example "order B beats order A", "local error ranks
  options within a layer"). Then score each one per model: replicated, not
  replicated, not testable. That scorecard is the transferability result, and
  it needs only the final pipeline, not the nine-step history.
- **Which properties I would expect to move the numbers [view]:** model size;
  the MLPs' share of parameters and their width ratio; dense or MoE; whether
  the model has very large fixed activations and where; how long it was
  trained. I have no measurement for any of these here.

## 5.2 What separates a method from tuning

### A working definition

This is a project-proposed test, not a source-derived one. A design choice is
*differentiating* if it passes these, in order:

| # | Test | What it rules out |
| --- | --- | --- |
| 1 | **Size:** larger than run-to-run noise *and* larger than what tuning the other arm gains | Wins that are really one arm's better recipe |
| 2 | **Persistence:** still there as the retraining budget grows (none, 100M, 1B) | Head starts |
| 3 | **Reach:** makes a region or an axis reachable that was not (bytes beyond a limit, compute saving) | Movement along an existing curve |
| 4 | **Order:** changes which method wins somewhere | Uniform shifts |

Test 1 is the one the literature warns about most: across 81 pruning papers,
different training and fine-tuning settings produced nearly as much
variation as different methods **[lit: Blalock et al. 2020, section 4.5,
agent-read]**. Your rule of equal tuning per operator family is the right
defence. It was not yet applied *between* compression families when dense
INT3 received a schedule and SwiGLU kept an untuned peak rate.

### Your measured factors, ranked

Effects are on 8K KL unless stated, taken from your pages **[read; calc for
the ratios]**. "Persists" asks whether the effect survives more retraining.

| Factor | Kind | Size | Persists? | Verdict |
| --- | --- | --- | --- | --- |
| Bits against width at equal bytes | What is removed | 2-10 times | Yes | Differentiating |
| Structure also saves compute (13-52% of linear FLOPs) | What is removed | A second axis | Yes | Differentiating (reach) |
| Stacking beyond 51.6% saving | What is removed | Reach to 62.1% | Yes | Differentiating, pending the 2-bit control of Part 1 |
| Training with quantization (dense INT4: 0.090 → 0.029) | Stage order | About 3 times | Yes | Differentiating |
| Order B against order A | Stage order | 20-44% at 100M, 24-52% at 1B | Grows | Differentiating |
| Training-time quantizer equal to the deployed one | Correctness | 19% for INT4; valid or invalid for INT3 | Yes | Not a method, and larger than most recipe effects |
| Decay for dense INT3 QAT | Recipe | 6% at 100M, 35% at 1B | Grows | **Differentiating for QAT;** it decided the 51% contest |
| Local fitting | Start | 110-427% at none, 10-24% at 100M, 0.3-2.8% at 1B | No | Differentiating only without retraining |
| Measured KL against local error as the allocator's price | Signal | 19-215% predicted, at none | Unmeasured | Open: file 2, steps 1-2 |
| Knapsack against uniform widths | Signal | 11-38% predicted, at none | Unmeasured | Open: the same runs |
| Teacher start against a fresh block | Start | 17% and 8% at 20% and 50%; 2% at 80% (forced models, 100M) | Shrinks at high removal | Differentiating at low removal only; thin evidence |
| Another operator type at equal size | Function class | Never better (505 of 506) | - | Not differentiating upward |
| Peak rate 1e-4 for SwiGLU 80% | Recipe | 9.8% at 100M | Untested | Tuning, until shown otherwise |
| Decay for SwiGLU alone; for B 20% INT4 | Recipe | 3.7% (noise); 5.6% | - | Tuning |
| Which blocks are trainable | Stage detail | Within noise at 100M | Small at 1B | Tuning |
| Fitting recipe (rate, schedule, context, data size, penalty) | Recipe | At most 4.4% on single-block KL | - | Tuning |

### What the ranking says

- **What you remove and the order of the stages persist. How cleverly you
  choose and initialise is repaid by retraining.** Every "start" and
  "signal" factor is large at zero budget and small or unmeasured after 1B.
  Published work reports the same wash-out for selection signals after
  fine-tuning **[lit: Liu et al. 2019; CLADO section 5.2; MoNE Table 3;
  agent-read]**.
- **So the retraining budget is the hidden axis of the whole comparison.** A
  factor has no single verdict; it has one per budget. That is the strongest
  argument for Part 1's proposal to report every family at three budgets.
- **The decay is the instructive exception.** It is a recipe setting with a
  stage-sized effect, because it fixes something only one family has (weights
  that keep jumping across rounding boundaries). The useful question is
  therefore not "method or hyperparameter?" but "does it address a mechanism
  the other arm does not have?".
- **Known and unknown.** The direction of the top rows is known from
  published work on other models (TRANSFERABLE HYPOTHESIS). Every size in the
  table is MODEL-CONCRETE. The two open rows are the cheapest unknowns you
  have left.

### For future designs

Before calling a new idea a method contribution, ask four things: which
mechanism does it address that the baseline lacks; at which budget is it
judged; was the baseline given the same tuning effort; and does it change an
ordering or only a position. An idea that fails all four can still be worth
running. It should then be reported as recipe sensitivity, not as a result.
