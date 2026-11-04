---
title: Heterogeneous-1 - Operator Recipes, Cost Tables, and Hybrid Allocation
status: draft
created: 2026-11-04
modified: 2026-11-04
---

# Heterogeneous-1 - Operator Recipes, Cost Tables, and Hybrid Allocation

**Status:** design settled on 2026-11-04 in a grilling session; not implemented.
The binding decisions are in the
[shared decision log](../../../agents/decisions.md) (2026-11-04 entries); this
page explains them. If the two disagree, the log wins until the conflict is
raised with the researcher. Model-level recovery belongs to
[Heterogeneous-2](heterogeneous-2.md).

## Questions

1. How should each new operator be fitted, and does tuning matter?
2. Do pre-recovery prices rank operators correctly?
3. Which layers can use simpler operators, and at which removal targets are new
   operators picked?
4. At what width does the linear part of a hybrid stop paying?
5. Does summed single-block damage still predict real damage when operators are
   mixed?
6. How much does damage grow as more layers are forced to non-SwiGLU operators?

## Why this experiment

**A heterogeneous model is a hybrid.** It does not replace every MLP with a new
operator. The allocator already chooses, per layer, between keeping the dense
MLP and several SwiGLU widths. This experiment adds more items to that per-layer
menu. SwiGLU stays the default, and a new operator enters only where it does
less damage for its size.

**Most of SwiGLU-1 to 9 transfers.** Only three of those studies' topics were
about the operator itself: teacher-neuron initialization (SwiGLU-1), calibration
size (SwiGLU-3, SwiGLU-9 Part A), and the local-fitting recipe (SwiGLU-8 Part A).
The rest tuned allocation (SwiGLU-2, 5) or recovery (SwiGLU-4, 6, 7), which apply
to any operator. So a new operator needs a fitting recipe, not nine iterations.

**Why a recipe study before model runs.** The 2026-10-11 decision requires it.
Without it, an operator could lose only because it was trained badly, and
"the allocator did not pick it" would mean nothing.

**Why two experiments.** Model-level design depends on which operators get
picked and at which targets. If new operators are rarely picked, Heterogeneous-2
shrinks a lot, so fixing it now would be guessing.

## Operators

An MLP has $3dD = 50{,}331{,}648$ parameters ($d=2048$, $D=8192$). The formulas
below are standard parameter bookkeeping and need no citation.

| Operator | What it is | Parameters | Start |
| --- | --- | --- | --- |
| SwiGLU (default) | Today's narrower gated MLP | $3dk$ | Top teacher neurons |
| Zero | Drops the layer's MLP; attention stays | 0 | None |
| Mean | Outputs one constant vector | $d$ | Mean teacher output |
| Linear | One $d\times d$ matrix | $d^2$ (8.33%) | Closed form or gradient |
| Plain MLP | Two matrices, one activation, no gate | $2dh$ | Random |
| Linear + SwiGLU | Dense linear part plus a SwiGLU part | $d^2+3dk$ | Linear fit plus top teacher neurons |
| Linear + plain MLP | Dense linear part plus a plain MLP | $d^2+2dh$ | Linear fit plus random |

**Linear has one size.** Both dimensions of the matrix are fixed by the hidden
size, so there is no width to change. Smaller means low-rank. Stacking linear
maps collapses into one map.

**Low-rank linear is excluded.** Built by truncating the SVD of a fitted map, it
is replacement plus algebraic reduction, which is stacking under the 2026-10-20
definition. The same function could be trained directly as two thin matrices, so
the label depends on construction; leaving it out avoids that argument and costs
little. At equal size it scored about the same as dense linear in the old screen.

**Why the hybrid has a linear part (project-proposed hypothesis).** A plain MLP
can represent linear behaviour by itself, since
$\mathrm{SiLU}(x)-\mathrm{SiLU}(-x)=x$ (standard algebra, no citation). That
costs two neurons per direction, so a full $d\times d$ map built from neurons
would take 33.3% of an MLP, four times the dedicated 8.33%. A linear part is
therefore only a cheaper way to buy linear behaviour, and it pays off only if the
teacher MLP has enough of it. A plain MLP can beat or tie a hybrid, as it did in
the old screen.

Linear + SwiGLU is the more interesting form: it keeps the teacher start, and the
linear part tries to capture the combined linear trend of the dropped neurons.
For example, 16.7% SwiGLU plus 8.33% linear is 25% of an MLP and competes with a
plain 25% SwiGLU. The dense linear part costs as much as 683 SwiGLU neurons, so
the question is which of the two buys more. One weak piece of evidence against
it: SwiGLU-5's output reconstruction, a constant correction with a refit, changed
almost nothing (KL 0.2822 against 0.2814 at 50%, 5M tokens). That is not a linear
branch. The definition follows the wiki's
[hybrid operator page](../../../../llm-wiki/wiki/methods/method-hybrid-operator-replacement.md),
which is a researcher proposal, not prior work.

## Stage 1 - recipe selection

Selects how each operator is trained. It repeats
[SwiGLU-8 Part A](../swiglu/swiglu-8.md#part-a---local-fitting-configuration-selection).
The adoption rule picks the recipe, not the knapsack; the knapsack never sees
recipes.

**Fixed from Part A:** layers 3, 11, and 19; the 128-token and 8K calibration
data; single-block KL at 128 and 8K, with 8K on the selection stream deciding;
the multi-seed adoption rule; `c1e-3-128` as the reference cell. Final-evaluation
streams are never used for selection (decision log, 2026-10-11).

**Test sizes** are chosen to pick a recipe; they are not model widths.

| Size | SwiGLU | Plain MLP | Linear + SwiGLU | Linear + plain MLP |
| ---: | ---: | ---: | ---: | ---: |
| 12.5% | 1024 | 1536 | 341 | 512 |
| 25% | 2048 | 3072 | 1365 | 2048 |

Linear is tested at 8.33% only, against a SwiGLU of width 683. Linear + SwiGLU
lands 2,048 parameters below the size, because neurons are whole numbers.

These sizes replace Part A's 25% and 50% because linear operators cannot reach
50%, a hybrid cannot go below 8.33%, and the narrow end is where the allocator
works: layers at width 2048 or below number 0-1 at 20-40% removal, 4 at 50%, 9 at
60%, 14 at 70%, and 16 at 75-80% (SwiGLU-8 allocations). SwiGLU-8 already measured
the 25% SwiGLU cell.

**Settings.** Part A's 13 gradient settings: rate (`1e-3`, `3e-4`, `1e-4`) ×
schedule (constant, cosine) × context (128, 8K), plus batch 256. Three seeds each
(21, 22, 23). For SwiGLU, seeds changed only the batch order and the noise term
was 0.2-0.8% of KL; for random-start operators they also change the starting
weights, so the noise must be measured. Closed-form fits have no randomness and
run once.

- **Linear** adds closed-form fits at both contexts. In Part A the exact
  least-squares down projection transferred 7-9% worse to 8K KL than gradient
  training, so the closed form should not be assumed best.
- **Hybrids** add one comparison at the reference setting: joint training against
  fitting the linear part first.
- **Calibration size and L2-SP** are added only if the overfitting indicator
  fires, and L2-SP only for teacher-start operators, because it pulls weights
  toward their start.

**SwiGLU check.** SwiGLU runs the same grid at width 1024. Each competitor tries
about 14 settings at 12.5%, while SwiGLU's recipe was chosen at 25% and 50%. If it
is not the best at 1024, SwiGLU would look worse than it is. Width 1024 also has
no 8K score yet. The likely outcome is confirmation, since the recipe was on a
plateau. A better recipe is reported as a limit, and hybrid SwiGLU layers keep
`c1e-3-128` to stay paired with SwiGLU-8. Width 683 gets one reference fit.

**What it gives.** A recipe table per operator, and the removal of "it was badly
tuned" as an explanation. A partly null result is likely: for SwiGLU nothing beat
the reference by more than 4.4%. A null result still shows that SwiGLU's recipe
transfers.

## Stage 2 - single-block recovery

Checks whether menu prices can be trusted. It uses the Stage 1 winners and picks
nothing.

**Reasoning.** A menu price is the damage right after local fitting, before any
recovery. The knapsack treats that as the operator's real cost, but recovery
repairs operators by different amounts. In SwiGLU-8 a 110-427% gap at token 0
fell to 10-24% by 100M and 0.3-2.8% by 1B, so pre-recovery damage can badly
overstate what remains. A random-start operator is the most likely case: it could
look terrible on the menu, recover well, and never be picked. "The allocator
refused it" must not be mistaken for "it cannot help".

**Procedure.** Insert one operator alone into the dense model at layer 3, 11, or
19, at both test sizes. Freeze everything else, train only that operator against
the teacher with KL for 10M tokens, and measure single-block KL at 0, 1, 5, and
10M. 10M was chosen because most of the slice-only handicap in SwiGLU-8 closed
within it. About 27 short runs.

**Reading.** At each layer and size, rank the equal-size operators by damage
before and after. A swap counts when the new leader is ahead by more than 5%, the
project's usual bar. Invented example at layer 11, 25%: SwiGLU 0.017, then 0.015;
plain MLP 0.030, then 0.014. The plain MLP looks worst before recovery but ends
best, so the knapsack would wrongly refuse it. A swap, or no new operator being
picked at any target, makes the forced runs in Heterogeneous-2 required.

This is single-block recovery, so it does not show joint recovery, and early
rankings can flip later ([SwiGLU-7](../swiglu/swiglu-7.md#scope-ranking-against-recovery-budget-read-only-check-2026-11-04)).

## Stage 3 - 22-layer cost tables

Measures each item's price at every eligible layer with the winning recipes.
Every SwiGLU menu size gets an equal-sized competitor, so low removal is measured
and not excluded by construction: at 20-40% many layers sit above 50% width.

| Size | SwiGLU | Plain MLP | Linear + SwiGLU | Linear + plain MLP |
| ---: | ---: | ---: | ---: | ---: |
| 3.125% | 256 (new) | - | - | - |
| 6.25% | 512 | 768 | - | - |
| 12.5% | 1024 | 1536 | 341 | 512 |
| 25% | 2048 | 3072 | 1365 | 2048 |
| 37.5% | 3072 | 4608 | 2389 | 3584 |
| 50% | 4096 | 6144 | 3413 | 5120 |
| 62.5% | 5120 | 7680 | 4437 | 6656 |
| 75% | 6144 | 9216 | 5461 | 8192 |
| 87.5% | 7168 | 10752 | 6485 | 9728 |

Zero, mean, and linear add one item each. That is about 530 fits. The SwiGLU
columns other than width 256 come from SwiGLU-8. Prices are single-block KL on the
allocation-selection partition, as for the existing width curves, so they drop
straight into the allocator.

**Width 256.** On today's menu the 90% solve puts 13 of 22 layers at the 512
floor. If zero were picked there, that might only reflect SwiGLU's menu ending at
512. Width 256 changes the control, so SwiGLU-only is re-solved with it first and
any change at 20-80% is reported, keeping its gain separate from the new
operators' gain.

## Stage 4 - knapsack solves

Each layer has a menu; every item has a price (parameters) and a damage
(single-block KL). There are two conditions: exactly one item per layer, and the
total price within budget. The goal is the smallest total damage. See the
[width-allocation note](../../../methodology/width-allocation-integer-program.md).

New operators only add items, so **there are no new conditions**, and the solve
stays exact because every price remains a multiple of $d$. In practice, a new
item is picked only if it does roughly less damage than a SwiGLU of the same size
at that layer. Two side rules are adjusted: isotonic smoothing applies within each
operator family, and leftover budget becomes extra neurons in a SwiGLU layer only.

Solves run at 20, 30, 40, 50, 60, 70, 75, and 80%, and at 90% as an option. They
give which operators, layers, and targets are picked, and the size at which the
linear part stops paying.

## Stage 5 - assembled hybrids and the degradation curve

Build the chosen hybrids and measure their real token-0 KL against the predicted
sum, compared with SwiGLU's 1.23-1.55×.

**Degradation curve.** Force 1, 2, 4, and 8 layers to non-SwiGLU operators at the
same budget, at the layers with the smallest predicted extra damage, and let the
knapsack solve the rest. If near-homogeneous is best, that stands as the result,
and this curve shows what each added non-SwiGLU layer costs and whether damage
grows as predicted. Token-0 points cost minutes each.

**Output of the experiment:** the list of targets where the hybrid differs from
SwiGLU-8's model, which decides the size of Heterogeneous-2.

## Order of work

```text
Start, in parallel:
  R  recipe selection (operators and the SwiGLU check)
  Z  zero and mean, all 22 layers
  S  SwiGLU 256, all 22 layers, then the SwiGLU-only re-solve

After R (the researcher reviews the adoption table), in parallel:
  C  22-layer cost tables
  D  single-block recovery

After C, Z, and S:
  K  knapsack solves (CPU, seconds)

After K, one task per target, in parallel:
  T  assembled hybrids and degradation curve at token 0

After T and D: report, then Heterogeneous-2 is designed
```

R is the only stage that meets a PERUN limit: at one task per setting it is about
70 tasks, and the queue allows 50, with each array task counted. It needs grouping
or two batches.

## What changes in the workflow

| Stage | Settings change? | Code change? |
| --- | --- | --- |
| Data, calibration pairs | No | No |
| SwiGLU neuron choice, fits, cost tables | No, reused | No |
| Fitting new operators | Yes | Small: modules and fitters exist; the hybrid forms need one module |
| Single-block KL | No | No, it accepts any module |
| Knapsack | No new conditions | Yes: menu items are hard-wired to width and SwiGLU's price formula |
| Building the model | No | Yes: only SwiGLU is built today |
| Recovery (S7-0) | No | No, it trains what the replaced modules contain |
| Saving and loading models | No | Yes, for Heterogeneous-2 only |
| Final evaluation | No | No |

## Evidence before the run

**Old operator screen.** Random starts, 12,288 pairs, one seed, WikiText-2. See
[reading the means](../../block/operator/operator.md#reading-the-means-read-only-check-2026-11-04).
Away from layers 1, 7, and 22 the hybrid and plain MLP tie. The linear fit used
about 6 pairs per input dimension and was probably data-starved; today's 393,216
pairs give 32 times more.

**Entry bars (offline calculation, 2026-11-02 to 2026-11-04).** The highest
single-block KL an item can have and still be picked, from SwiGLU-8's width curves:
the best total on today's menu minus the best total of the other 21 layers when
that layer spends the item's price. The boundary step is ignored. This is a
project-defined diagnostic on a standard dynamic program.

| Item | Bar at 80%, layers 11-13 | Old screen value | Gap |
| --- | ---: | ---: | ---: |
| Zero | 0.031-0.032 | 0.052-0.056 | about 1.7× |
| Linear | 0.025-0.026 | 0.050-0.055 | about 2× |

SwiGLU 512 already costs 0.025-0.026 at those layers with fewer parameters than
linear. At 90% the zero bar rises only to about 0.037. This is an expectation, not
a result: the old values are on WikiText-2 with a weak linear fit, the bars are on
C4, and both are before recovery.

## Limits and risks

- **The result may be near-homogeneous.** That is a valid finding. The equal-size
  comparison, the entry bars, the degradation curve, and the forced runs are what
  support it.
- **Summed damage is an approximation.** It underestimates real damage by 23-55%
  for SwiGLU and could be worse when several MLPs are zeroed. Stage 5 measures it.
- **Recovery was tuned on SwiGLU.** One shared recipe is a stated limit.
- **Three layers and two sizes** select each recipe, as in Part A.
- Layers 1, 7, and 22 are not in the recipe study; the old layer-22 anomaly can
  only appear in the 22-layer tables.

## Resources

Rough estimates from SwiGLU-8's measured timing; not a PERUN resource plan, which
the implementation plan must add.

| Stage | GPU-hours |
| --- | ---: |
| Recipe selection, 4 operators | 30-50 |
| SwiGLU check at 1024 and 683 | 3-4 |
| Single-block recovery | about 3 |
| 22-layer tables | 18-30 |
| **Total** | **55-90** |

## Sources and attribution

The hybrid operator, the entry bar, and the degradation curve are project-defined.
The allocator's prior work is in the
[width-allocation note](../../../methodology/width-allocation-integer-program.md),
Section 8. Numbers belong to their linked experiment pages.
