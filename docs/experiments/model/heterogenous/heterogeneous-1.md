---
title: Heterogeneous-1 - Operator Recipes, Cost Tables, and Hybrid Allocation
status: draft
created: 2026-11-04
modified: 2026-11-04
---

# Heterogeneous-1 - Operator Recipes, Cost Tables, and Hybrid Allocation

**Status:** design settled on 2026-11-04 in a grilling session and implemented
the same day (see [Implementation](#implementation)). Recipe selection and the
baseline table have run ([Stage 1 results](#stage-1-results-2026-11-05)), and the
recipes are pinned.
**Next step:** sitting 2, the cost tables and single-block recovery
([runbook](../../../infrastructure/perun/perun-runbook.md#125-sitting-2-cost-tables-and-recovery)).
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
  fitting the linear part first. Both start from the same point: the nonlinear
  part at its start, and the linear part as the ridge fit of what that start
  leaves unexplained. A fit of the whole output would count twice what the teacher
  slice already explains (31-78% of the output at widths 1024-4096 on ordinary
  layers, 94-100% at layers 1, 7, and 22). Joint training trains both parts; linear
  first freezes the linear part, so only the freezing differs.
- **Calibration size and L2-SP** are added only if the overfitting indicator
  fires, and L2-SP only for teacher-start operators, because it pulls weights
  toward their start. The indicator is SwiGLU-8's vote at each operator's largest
  tested size: best epoch at most 3 while training MSE still falls, in most seeds
  of a layer, in at least two of the three layers. If it fires, work stops and the
  follow-up is planned then; neither option is pre-built.

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

### Stage 1 results (2026-11-05)

Source: `heterogeneous-1/select-001/result.json` (sha256
`92e11c47128996b6afa952c79ac8b47844e0a3ff2e2742cec0b97e57be429520`), from array 143434
and job 143435 at commit `fbc9142`.

**Checks.** All 24 tasks completed. Data identity and checks 2, 3, 5, and 6 passed; the
relative differences in checks 2, 3, and 5 are exactly 0, and the self-KL is -2.8e-9 to
-1.2e-9. The manual sanity gate passed: `kl128` 0.0618 at layer 3 and 0.0318 at layer 11.

**Adopted recipes.** Reductions are against `c1e-3-128`; positive is better.

| Operator | Adopted | Strongest other setting | Why it was not adopted |
| --- | --- | --- | --- |
| SwiGLU, width 1024 | `c1e-3-128` | `c1e-4-8k`: 8K +3.96%, 128 -6.91% | Below 5%; a combination more than 5% worse |
| Linear | `c1e-3-128` | `c3e-4-8k`: 8K +6.13%, worst -14.49% | A combination more than 5% worse |
| Plain MLP | `c1e-3-128-b256`: 8K +8.08%, 128 +10.01%, noise 1.39%, worst +4.15% | - | - |
| Linear + SwiGLU | `c1e-3-128` | `c3e-4-8k`: 8K +7.74%, worst -9.99% | A combination more than 5% worse |
| Linear + plain MLP | `c1e-3-128`, see below | `c1e-3-8k`: 8K +13.01%, worst -7.90% | A combination more than 5% worse; the table is not reliable |

SwiGLU's recipe holds at width 1024, so no limit is reported. As in SwiGLU-8, 8K
settings trade 128-token KL for 8K KL. Every plain MLP fit reached the 64-epoch cap; with
the adopted recipe, validation MSE still fell 1.1-2.3% over the last 16 epochs, so plain
MLP prices are for this budget, not a converged fit.

**Linear + plain MLP mostly failed to train at layers 3 and 11.** The overfitting
indicator fired there, but it is a false alarm: 88 of this operator's 252 gradient fits
returned their starting weights (best epoch 0), against 0 of 723 for the other operators.
With the reference recipe every seed at both sizes stopped after 3 epochs, so widths 512
and 2048 score the same. The ridge start is already good and validation MSE rises
slightly at first. Patience 3 ended those runs while it was recovering (layer 3:
0.21990 at the start, then 0.22009, 0.22007, 0.22006). That is the immediate cause, not
a full explanation: at layer 3 a 32-epoch cosine run at `1e-3`, which cannot stop early,
also never beat its start. Layer 19 trained normally. The adoption table for this
operator compares mostly stalled fits, and more calibration data would treat a problem
that is not present.

**No follow-up was run** (researcher decision, 2026-11-05). Where this hybrid trained, it
still lost clearly at 25% of an MLP: 8K KL 0.0355 at layer 11 (`linear-first`) and 0.0784
at layer 19, against SwiGLU's 0.0236 and 0.0555, linear + SwiGLU's 0.0284 and 0.0628, and
plain MLP's 0.0303 and 0.0659. A fix is therefore unlikely to change which items are
picked, although wide sizes were not tested. `c1e-3-128` is pinned as the rule's outcome.
Its cost-table rows at stalled layers are not a fair price for the operator, so read
`best_epoch` before quoting them; zero means the row is the start. The linear part's
question is answered with linear + SwiGLU, which trained normally.

**Measured use.** Workflow timing, not scheduler allocation; the scheduler records had
expired. `select` took 45.9 GPU-hours against 30-50 planned, 0.81-2.53 h per task, and
the baseline table 1.44 h. Fits ran at 3.1-5.1 s per epoch, and batch 256 was no slower.
PERUN held the array at 15 running tasks (`%15`, not requested), so it ran in two waves.

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
within it. About 30 short runs: the 27 of the decision, plus SwiGLU 683 at each
layer so that linear has an equal-size partner.

**Reading.** At each layer and size, rank the equal-size operators by damage
before and after. The 8K selection KL decides, because it is Part A's deciding
score, recovery trains at 8K, and it has 196,608 positions against 3,072; the
128-token KL is reported beside it. A swap counts when the new leader's KL is more
than 5% below the old leader's at 10M, the project's usual bar. Invented example at
layer 11, 25%: SwiGLU 0.017, then 0.015; plain MLP 0.030, then 0.014. The plain MLP
looks worst before recovery but ends best, so the knapsack would wrongly refuse
it. A swap, or no new operator being picked at any target, makes the forced runs
in Heterogeneous-2 required.

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

**Width 256.** On today's menu the 90% solve puts 12 of 22 layers at the 512
floor and one just above it. If zero were picked there, that might only reflect
SwiGLU's menu ending at 512. Width 256 changes the control, so SwiGLU-only is
re-solved with it first and any change at 20-80% is reported, keeping its gain
separate from the new operators' gain.

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
sum, compared with SwiGLU's 1.23-1.55×. The real KL is the same monitoring KL as
that comparison, with the 128-token and 8K selection KL beside it.

**Degradation curve.** Force 1, 2, 4, and 8 layers to non-SwiGLU operators at the
same budget, at the layers with the smallest predicted extra damage, and let the
knapsack solve the rest. If near-homogeneous is best, that stands as the result,
and this curve shows what each added non-SwiGLU layer costs and whether damage
grows as predicted. It runs at 20, 50, and 80%. A layer's extra damage is the
predicted total with that layer limited to non-SwiGLU items, minus the unrestricted
total; dense counts as SwiGLU. Layers are ranked by that value, ties by layer.
Layers already non-SwiGLU score zero, so they normally come first. The boundary step
sits outside the knapsack, so another layer's value can be negative and rank ahead of
them; this appeared on small random menus, not on SwiGLU-8's curves with placeholder
prices (offline calculation, 2026-11-04). A point equal to the unrestricted solve is
recorded and not rebuilt, so a target that already has eight non-SwiGLU layers gives
no forced point.

**Output of the experiment:** the list of targets where the hybrid differs from
SwiGLU-8's model, which decides the size of Heterogeneous-2. A target counts only
if its solve has at least one non-SwiGLU layer. SwiGLU-only width changes are
reported separately: at 50% the solver already differs from SwiGLU-8's model at
layers 7 and 11, because that model kept SwiGLU-7's widths.

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
  K  knapsack solves (CPU job, about 30 minutes)

After K, one task per target, in parallel:
  T  assembled hybrids and degradation curve at token 0

After T and D: report, then Heterogeneous-2 is designed
```

R runs as 24 tasks, one per operator, layer, and size, which fits PERUN's limits of
20 running jobs and 50 queued. K is a job because one full-menu solve takes 8.5-21 s
and the forcing rule needs about 26 solves per target (offline calculation with
placeholder costs, 2026-11-04).

## What changes in the workflow

| Stage | Settings change? | Code change? |
| --- | --- | --- |
| Data, calibration pairs | No | No |
| SwiGLU neuron choice, fits, cost tables | No, reused | No |
| Fitting new operators | Yes | Small: modules and fitters exist; the hybrid forms need one module |
| Single-block KL | No | No, it accepts any module |
| Knapsack | No new conditions | Yes: menu items are hard-wired to width and SwiGLU's price formula |
| Building the model | No | Not here: token-0 assembly swaps modules in temporarily. Heterogeneous-2 needs it, because only SwiGLU is built today |
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
- **Trained linear is under-trained at layers 1 and 7.** At the real update budget it
  ties ridge at layers 3 and 11 (KL 0.0757 and 0.0755, 0.0452 and 0.0451), but at
  layers 1 and 7 it reaches the epoch cap with KL 9.6 and 16.5 against ridge's 2.7
  and 1.5. Targets there reach about 27,000. If the gradient recipe is kept, those
  cost-table rows overstate linear's damage and must not be quoted as the operator's
  damage. No pick changes: ridge itself is 9-42 times above the highest entry bar at
  layers 1, 7, and 22. Hybrids start from ridge and are not affected. Plain MLP at
  width 768 also reaches the cap at layer 7, at 1.8 times its bar, so read
  `epochs_completed` there. Offline calculation with a separate training loop on
  about 8% of the pairs, 2026-11-04; layer 22's trained fits were not measured.
- **The 64-epoch cap may bind.** In SwiGLU-8, 9 of 18 `c1e-4-128` fits used all 64
  epochs even from a teacher start, and random starts will do so more often. The
  merge reports the share per setting, so a low-rate setting that was cut off can
  be read as such.

## Implementation

Implemented on 2026-11-04 without any run. Runner:
[`heterogeneous_1.py`](../../../../workflows/runs/model/heterogenous/heterogeneous_1.py).
Configuration:
[`heterogeneous-1.json`](../../../../workflows/configs/model/heterogenous/heterogeneous-1.json).
PERUN job:
[`heterogeneous_1.sbatch`](../../../../workflows/jobs/perun/heterogenous/heterogeneous_1.sbatch).
Copy-paste commands: the
[PERUN runbook](../../../infrastructure/perun/perun-runbook.md#12-heterogeneous-1).
The load-only analysis notebook comes after the results. Results copied home go to
`data/results/workflows/model/heterogeneous-1/`.

| Command | Stage | Does | Output (`perun-results/heterogeneous-1/`) |
| --- | --- | --- | --- |
| `select --task-index i` | R | One operator's grid at one layer and size, 24 tasks. Tasks 0-2 SwiGLU 1024 (with one fit at 683), 3-5 linear, 6-11 plain MLP, 12-17 linear + SwiGLU, 18-23 linear + plain MLP | `select-001-task-NN/` |
| `baseline-table` | Z, S | Zero, mean, and SwiGLU 256 at all 22 layers | `baseline-table-001/` |
| `merge-select` | Adoption | Login node. Checks that every task used identical configuration, code, software, and data; applies the adoption rule per operator; reports the overfitting indicator and the share of fits at the epoch cap. Safe to re-run | `select-001/` |
| `cost-table --task-index i` | C | All 23 non-SwiGLU items at one layer, 22 tasks | `cost-table-001-layer-LL/` |
| `recover --task-index i` | D | One layer's 10 single-block recoveries, 3 tasks | `recovery-001-layer-LL/` |
| `solve` | K | `cpu_short`. Check 1, SwiGLU with width 256, the full menu at 20-80% and 90%, and the forcing rule. Refuses a table made under another configuration or, for cost tables, other recipes | `solve-001/` |
| `assemble --task-index i` | T | One target's models at token 0, 8 tasks | `assemble-001-target-T/` |
| `summarize` | Report | Login node. Swap tables, picked operators, hybrid targets, real against predicted damage, and whether Heterogeneous-2's forced runs are required. Safe to re-run | `summary-001/` |

Between `merge-select` and the next stages the researcher reviews the adoption
table and pins the recipes and the merge's sha256 under `recipes` in the
configuration. `cost-table`, `recover`, `solve`, and `assemble` refuse to start
otherwise. SwiGLU menu items always keep `c1e-3-128`.

**Fit details**, settled in the implementation plan on 2026-11-04:

| Item | Choice |
| --- | --- |
| Plain MLP | SiLU, as in the old study and the algebra above; no bias; PyTorch's default initialization |
| Linear | No bias, because its price is $d^2$; gradient settings start from the default initialization |
| Closed form | `ridge-128`, `ridge-8k`: ridge `1e-4` scaled by the mean Gram diagonal, the old study's value, solved in FP64 |
| Random starts | Built on the CPU in FP32 right after seeding with the setting's seed, so seeds change the start reproducibly |
| SwiGLU part | Top neurons from SwiGLU-8's ranking, recomputed from the 393,216 pairs and checked by digest; 8K settings reuse it |
| Mean | FP64 mean of the 393,216 128-token targets |
| Single-block recovery | S7-0's recipe unchanged: AdamW, constant `3e-5`, zero weight decay, temperature-1 KL, no cross-entropy, 8,192-token sequences, effective batch 8,192, seed 21, on the first 10M tokens of SwiGLU-8's stream. It trains the whole operator, so a `linear-first` linear part is trained here too |
| Allocator rules | Mean costs one budget unit (2,048 values); smoothing runs within each family; leftover budget widens SwiGLU layers only |

**Data.** Every GPU task rebuilds the 128-token partitions and 8K streams with
SwiGLU-8's code and checks them against its block check, as SwiGLU-9 did. SwiGLU-8's
`prepare-001` holds only the KL stream, and rebuilding takes about a minute.
Calibration pairs are captured per layer. The final-evaluation KL stream is never
scored.

**Shared code**, each extension leaving SwiGLU-only behaviour unchanged:

- `SwiGLUWidthCurvePoint` gains `operator`, and `build_operator_curve` builds a
  non-SwiGLU family. `build_swiglu_width_curve` cannot: it rejects widths above
  8,192, prices at $3dk$, and would read linear + plain MLP width 8,192 as dense.
- `allocate_discrete_swiglu_widths` widens SwiGLU items only, and names an operator
  in a row only when it is not SwiGLU. Its output on SwiGLU-8's curves is identical
  to the previous code at all eight targets and at 90% (offline comparison,
  2026-11-04).
- `DenseHybridReplacement` is the one new module. `HybridReplacement` stays low-rank.
- `fit_ridge_linear(..., dtype=torch.float32)` gains an optional precision.

**Stored.** JSON only. No fitted state is kept: `assemble` refits what it needs,
which avoids about 47 GB and makes each refit a reproduction check. There is no
resume: a `select` task takes at most about 3 h, `assemble` can take longer at high
removal, setup takes about a minute, and a failed task's partial JSON is still staged
out.

### Acceptance checks

These are part of the real runs. A failed check stops the stage after recording
its values; r is the relative difference.

| # | Check | Reference | Tolerance | Where |
| --- | --- | --- | --- | --- |
| 1 | SwiGLU-only allocator at eight targets | 60-80%: recorded allocations. 20-50%: the recorded re-solve, because SwiGLU-8 kept SwiGLU-7's widths | exact widths; predicted KL r ≤ 1e-12 | start of `solve` |
| 2 | SwiGLU `c1e-3-128`, seed 21, width 2048 | SwiGLU-8's block check at that layer | r ≤ 1e-3 | first fit of all 24 `select` tasks |
| 3 | SwiGLU 1024, same recipe | SwiGLU-8's curve KL and fit | r ≤ 1e-3 | `select` tasks 0-2 |
| 4 | Item parameter count, a multiple of the hidden size | Size tables above | exact | every item; again in `solve` |
| 5 | Hybrid with a zero linear part | Its nonlinear part alone | r ≤ 1e-4 | start of hybrid `select` tasks |
| 6 | Teacher against itself, 128-token and 8K selection KL | 0 | ≤ 1e-6 absolute | every GPU task |
| 7 | SwiGLU-8's model reassembled | Its recorded token-0 monitoring KL | r ≤ 1e-4 | first point of every `assemble` task |
| 8 | Refit item | Its `select` row in `recover`, its price in `assemble` | r ≤ 1e-3 | `recover`, `assemble` |
| D | Window digest, stream sha256, neuron ranking digest | SwiGLU-8 | exact | every GPU task |

Checks 7 and 8 go beyond the design. They are cheap, and check 7 tests the whole
assembly path against a recorded value before anything new is measured. Check 6's
128-token path compares an autocast forward pass with a non-autocast cached result;
zero is expected but unproven, and a failure appears in the first two minutes. On an
RTX 3060 with transformers 5.3.0 the two passes were bit-identical and the self-KL was
-4.6e-9 (offline calculation, 2026-11-04); PERUN has another GPU and version.

**Manual sanity gate.** The checks above prove the SwiGLU path, and check 8 only shows
that a new operator matches itself, so no recorded value tests a new operator. The
first linear fit of `select` tasks 3 and 4 gives that reference: `kl128` should be near
0.076 at layer 3 and 0.045 at layer 11 (offline calculation, 2026-11-04). More than a
factor of 2 away stops the work for investigation. The factor allows for the smaller
sample behind those values. The commands are in the
[runbook](../../../infrastructure/perun/perun-runbook.md#123-watch-sitting-1).

**Not yet verified.** `tests/test_heterogeneous1_contracts.py` covers the allocator
rules, check 1 on SwiGLU-8's recorded curves, prices, the residual start, linear
first, the forcing and swap rules, the merge, the job mapping, and stage D's real
recovery loop on a tiny random model on the CPU. GPU execution, real data, memory
use, fit lengths, and checks 2, 3, and 5-8 have not run; the first real stages are
also the first execution test.

## Pre-run resource plan

Recorded 2026-11-04 under
[PERUN resource accounting](../../../infrastructure/perun/perun-resources.md),
before any submission, to be compared with measured use afterward. This is
workflow-timing arithmetic, not scheduler allocation. All GPU stages use
`gpu_short`, one H200, and 8 CPUs.

| Stage | Tasks | Time limit, RAM | Expected per task | Expected GPU-h | Ceiling GPU-h | CPU core-h | Peak RAM / VRAM |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| R `select` | 24 (array 0-23) | 4 h, 64 GB | 1.2-2.4 h | 30-50 | 96 | 240-400 | ≤ 30 GB / ≤ 25 GB |
| Z, S `baseline-table` | 1 | 3 h, 64 GB | 1.2-1.7 h | 1.2-1.7 | 3 | 10-14 | ≤ 30 GB / ≤ 20 GB |
| C `cost-table` | 22 (array 0-21) | 3 h, 64 GB | 0.7-1.4 h | 15-30 | 66 | 120-240 | ≤ 30 GB / ≤ 25 GB |
| D `recover` | 3 (array 0-2) | 3 h, 64 GB | 1-1.8 h | 3-5.5 | 9 | 24-44 | ≤ 30 GB / ≤ 34 GiB |
| K `solve` | 1 | `cpu_short`, 4 CPUs, 16 GB, 2 h | 0.3-0.7 h | 0 | 0 | ≤ 8 | ≤ 4 GB / none |
| T `assemble` | 8 (array 0-7) | 6 h, 128 GB | 0.2-1.5 h | 4-8 | 48 | 32-64 | ≤ 60 GB / ≤ 25 GB |
| **Total** | **59** | | critical path about 9-12 h, plus queue and review | **about 55-95** | **222** | **about 430-770** | |

Storage: PROJECT under 1 GB, SCRATCH at most 5 GB per task, local export tens of MB.
Project use so far is about 410 of 8,424 GPU-hours. No sitting queues more than 25
tasks or two arrays.

**Evidence** (offline calculation on SwiGLU-8's records, 2026-11-04):

- Local fits take 3.2-5.6 s per epoch, rising with size, on SwiGLU-7's and SwiGLU-8's
  records. Batch 256 is no slower per epoch at widths 2048 and 4096. Teacher-start
  width 512-1024 runs about 50-55 epochs, and width 4096 and above about 5.
- A block-check task's setup takes about 1.2 minutes, not the 20-40 minutes in
  SwiGLU-8's plan, and scoring a fit at both contexts about 4 s.
- A `select` task is bounded by its grid: 3 seeds × (6 constant × 64 epochs + 6
  cosine × 32 epochs + batch 256 × 64 epochs) is 1,920 epochs, at most about 2.6 h.
  Hybrid tasks add linear first, 2,112 epochs and one start per fit, at most about
  3 h. A hybrid start of 15 s is assumed, not measured.
- `cost-table` is at most about 2.1 h if all 23 items reach 64 epochs. A flat 7 s per
  epoch would pass the limit, against a recorded maximum of 5.64 s at the same size.
- `assemble` has the least certain load. At 80%, rebuilding SwiGLU-8's own model takes
  42 minutes of recorded fits, and about 21 further items at the epoch cap would pass
  3 h. There is no resume, so its limit is 6 h. A limit is not consumption.
- S7-0 trained at 27,150-30,750 tokens per second, so 10M tokens take about 6 minutes.
- The page's earlier rough estimate was 55-90 GPU-hours.

**Confidence:** medium for teacher-start fits. Low for random-start epoch counts,
which are the main uncertainty; peak RAM is arithmetic, not measured.

## Sources and attribution

The hybrid operator, the entry bar, and the degradation curve are project-defined.
The allocator's prior work is in the
[width-allocation note](../../../methodology/width-allocation-integer-program.md),
Section 8. Numbers belong to their linked experiment pages.
