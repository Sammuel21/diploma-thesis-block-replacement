---
title: Width Allocation as an Integer Program
status: draft
created: 2026-10-17
modified: 2026-10-17
sources:
  code:
    - src/mlp_replacement/compression/allocation.py
  artifacts:
    - data/results/workflows/model/swiglu-5/search/run-001.json
    - data/results/workflows/model/swiglu-7/prepare-001/result.json
    - data/results/workflows/model/swiglu-7/S7-0-target-<t>-run-001/result.json
    - data/results/workflows/model/swiglu-8/prepare-001/result.json
    - data/results/workflows/model/swiglu-8/fitted-target-<t>-run-001/result.json
    - data/results/notebook-model-study/block-interaction.json
---

# Width Allocation as an Integer Program

Status: working note for the supervisor meeting after 2026-10-16 (decision
log, 2026-10-16, point 4). It is not thesis text. It describes the allocator
that every homogeneous SwiGLU experiment since SwiGLU-5 uses, the evidence
behind it, its prior work, and one proposed extension.

**In short.** The allocator picks one width for each of the 22 eligible MLPs
so that the summed predicted damage is as small as possible within a
parameter budget. This is a multiple-choice knapsack, a standard integer
linear program. The code solves it exactly with a dynamic program, so no ILP
solver is needed. "Exactly" refers to the predicted damage. That prediction
adds up per-layer damage measured one layer at a time, and our data show that
the real joint damage is 23-55% larger than the sum. Whether the sum still
picks the right allocation has not been tested. The formulation follows
SPDY, which the researcher records as an inspiration together with ZipLM.

Every equation below carries one of four labels: **standard** (textbook
mathematics or bookkeeping), **source-derived** (cited at a locator),
**project-proposed** (a modelling choice of this project), or **project
heuristic** (an engineering rule without an optimality claim).

## 1. What the allocator decides

Input: for every eligible layer, a short curve of measured damage against
retained width. Output: one width per layer, under a budget on retained
eligible-MLP parameters. The allocator does not choose neurons, fit
operators, or recover the model. Those steps come before and after it:

```text
fit each layer at every grid width  ->  measure singleton KL per (layer, width)
        ->  isotonic smoothing  ->  knapsack (this note)  ->  boundary width
        ->  fresh local fit of the chosen widths  ->  recovery training
```

## 2. Setting and notation

| Symbol | Meaning | SmolLM2-1.7B value |
| --- | --- | --- |
| $E$ | eligible layers; $n=\lvert E\rvert$ | layers 1-22, $n=22$ (0 and 23 protected) |
| $d$ | hidden size | 2,048 |
| $D_i$ | original MLP width of layer $i$ | 8,192 |
| $K_i$ | measured width options of layer $i$ | SwiGLU-5/7: 2,048 to 7,168 in steps of 1,024, plus dense 8,192; SwiGLU-8 adds 512 and 1,024 |
| $c_i(k)$ | cost of width $k$: smoothed singleton KL (Section 5) | measured |
| $p_i(k)$ | retained parameters at width $k$ | Eq. (1) |
| $s$ | target: fraction of eligible-MLP parameters **removed** | 0.2-0.8 |
| $B$ | retained-parameter budget | Eq. (2) |

**Eq. (1), standard bookkeeping, no citation.** A bias-free SwiGLU of width
$k$ has gate, up, and down projections:

$$
p_i(k) = 3dk, \qquad p_i(D_i) = 3dD_i \ \text{(dense module kept)}.
$$

A candidate with an output bias adds $d$. The current experiments use none.

**Eq. (2), standard bookkeeping, no citation.** With
$P_E=\sum_{i\in E}3dD_i = 1{,}107{,}296{,}256$,

$$
B = P_E - \operatorname{round}(s\,P_E).
$$

The target counts only eligible-MLP parameters. Removing 20% of them removes
12.9% of the whole model ([SwiGLU-8 Part B](../experiments/model/swiglu/swiglu-8.md#part-b-results-2026-10-12)).

## 3. The multiple-choice knapsack as an integer linear program

**Eq. (3), standard formulation.** Let $x_{ik}\in\{0,1\}$ mean "layer $i$
gets width $k$":

$$
\begin{aligned}
\min_{x}\quad & \sum_{i\in E}\sum_{k\in K_i} c_i(k)\,x_{ik} \\
\text{s.t.}\quad & \sum_{k\in K_i} x_{ik} = 1 \qquad \forall i\in E, \\
& \sum_{i\in E}\sum_{k\in K_i} p_i(k)\,x_{ik} \le B, \\
& x_{ik}\in\{0,1\}.
\end{aligned}
$$

The first constraint says every layer gets exactly one width; the second is
the budget. This is the multiple-choice knapsack problem: classes of items,
exactly one item per class, one capacity. It is NP-hard in general, which is
why the structure in Section 4 matters. Citation needed: a knapsack textbook
(not yet registered; see Section 8.3).

As a model-compression allocation, the same constrained problem appears in
SPDY with a runtime budget instead of parameters
[`src-spdy-2022`, Section 3.1, Eq. (1)].

## 4. Why the dynamic program over hidden-size units is exact

### 4.1 The recurrence

Every $p_i(k)$ is a multiple of $d$, which the code asserts. Measure weights
in units of $d$: $w_i(k)=p_i(k)/d=3k$. Because all weights are integers,

$$
\sum_{i}\sum_{k} p_i(k)\,x_{ik}\le B
\iff
\sum_{i}\sum_{k} w_i(k)\,x_{ik}\le U, \qquad U=\lfloor B/d\rfloor .
$$

So the change of units loses nothing: the problem in units is the same
problem.

**Eq. (4), standard dynamic program.** Order the layers $1,\dots,n$. Let
$F_j(u)$ be the smallest total cost of the first $j$ layers using exactly
$u$ units:

$$
F_0(0)=0,\quad F_0(u)=+\infty\ (u>0),\qquad
F_j(u)=\min_{k\in K_j,\ w_j(k)\le u}\Big[F_{j-1}\big(u-w_j(k)\big)+c_j(k)\Big].
$$

The answer is $\min_{u\le U}F_n(u)$, and the chosen widths come from keeping
the choice behind each table entry. Citation needed: the knapsack textbook
(Section 8.3). SPDY uses the same recurrence over discretized time
[`src-spdy-2022`, Section 3.2, Eqs. (2)-(3), Algorithm 1].

### 4.2 Why it is exact

1. **Every allocation is in the table.** Any feasible choice of widths uses
   some total $u\le U$ after layer $j$, so it is one candidate for $F_j(u)$.
2. **Discarding is safe.** Take two partial allocations of layers $1..j$
   that use the same $u$ units. Every completion that fits one fits the
   other, and it adds the same cost to both, because the cost is a sum over
   layers. So the cheaper one is never worse, and keeping only it per $u$
   loses no optimum. This is the principle of optimality, and it holds only
   because the objective is a sum (additive) and the budget is a sum.
3. **No rounding.** Weights are exact integers in units of $d$, so the
   table works with exact budgets, unlike SPDY's time buckets.

Together, the dynamic program returns an optimal solution of Eq. (3).

### 4.3 Size and determinism

The worst case is $O(n\cdot\max_i\lvert K_i\rvert\cdot U)$ time. In practice
the grid widths are multiples of 512 neurons, so reachable totals are
multiples of 1,536 units. With $U\approx 432{,}537$ at 20% removal, each layer
has at most about 282 reachable states. The code stores only reachable totals.
Ties break deterministically: lower cost first, then less unused budget, then
the lexicographically smaller width tuple
([`allocation.py`](../../src/mlp_replacement/compression/allocation.py),
`allocate_discrete_swiglu_widths`).

### 4.4 What "exact" does not cover

- **The objective is a surrogate.** The optimum is exact for summed
  singleton KL, not for the real model (Section 5).
- **The grid.** Only measured widths are options.
- **The boundary width.** After the solve, the leftover budget
  $R=B-\sum_i p_i(k_i)$ buys $e=\lfloor R/3d\rfloor$ extra neurons for one
  layer.

**Eq. (5), project heuristic, no citation.** Among the layers whose next
grid width $k_i^{+}$ is at least $e$ neurons wider, pick the steepest
interpolated gain,

$$
i^{*}=\arg\max_{i}\ \frac{c_i(k_i)-c_i(k_i^{+})}{k_i^{+}-k_i},
$$

and give it width $k_{i^*}+e$ with linearly interpolated cost. This spends at
most one layer's leftover, and its cost is not measured when the choice is
made. It is outside the exactness argument. The chosen width is then fitted
like any other. Examples are the widths 5,529, 3,379 and 2,252 in
[SwiGLU-8's 60-80% allocations](../experiments/model/swiglu/swiglu-8.md#part-b-results-2026-10-12).

## 5. The additive singleton-KL objective

### 5.1 The modelling choice

**Eq. (6), project-proposed modelling choice, no external citation.** For
each layer and grid width, fit the replacement, insert it **alone** into the
dense model, and measure teacher-to-student KL at temperature 1 on the
allocation-selection partition. The allocator then assumes

$$
\mathrm{KL}\big(\text{all chosen replacements together}\big)
\;\approx\; \sum_{i\in E} c_i(k_i).
$$

This is not a theorem and is not taken from MoDeGPT, whose allocation maps
importance scores through a softmax [`src-modegpt-2025`, Section 3.3,
Eqs. 10-11]. SPDY makes the same kind of assumption with a different
per-layer error and calls it strong [`src-spdy-2022`, Section 3.1].

**What the allocator actually needs.** The sum need not be accurate. It only
needs to put allocations **at the same budget** in the same order as the true
joint damage. A constant factor, for example joint = 1.3 × sum for every
allocation, would not change which allocation wins. Damage that grows faster
for some layers or widths than others could.

### 5.2 Isotonic smoothing

**Eq. (7), standard algorithm.** For one layer with raw costs
$y_1,\dots,y_m$ ordered by increasing width, solve

$$
\min_{z}\ \sum_{j=1}^{m}(y_j-z_j)^2
\quad\text{s.t.}\quad z_1\ge z_2\ge\dots\ge z_m ,
$$

with pool-adjacent-violators: merge neighbouring points that break the order
into their mean until none do (`nonincreasing_isotonic` in
[`allocation.py`](../../src/mlp_replacement/compression/allocation.py)).
Citation needed: an isotonic-regression reference (Section 8.3).

*Why.* Each width is fitted and measured independently, so noise could make a
narrower width look less harmful than a wider one. The knapsack would exploit
that inversion and take the parameters for free. Monotone costs remove the
temptation. Raw values are stored beside smoothed ones. Monotonicity is
imposed by the project, not observed as a guarantee.

*Observed.* A read-only check of the artifacts found that smoothing changed
**no** point: 0 of 154 curve points in SwiGLU-5's search, 0 of 154 in
SwiGLU-7's preparation, and 0 of 198 in SwiGLU-8's (offline calculation,
2026-10-17). So far it has been an inactive safeguard. It would matter for
noisier costs, such as the extension in Section 9.

### 5.3 What has been measured about additivity

Additivity was tested at the block level, where damage was super-additive.
It was **never tested at the allocator level**, where only the ranking
matters. Two pieces of evidence exist, both offline calculations on stored
artifacts.

**(a) Block-interaction study, consecutive windows.** The
[block-interaction study](../experiments/model/interaction/block-interaction.md)
defines the residual below.

**Eq. (8), project-defined diagnostic.**

$$
r(S)=\mathrm{KL}_{\text{joint}}(S)-\sum_{i\in S}\mathrm{KL}_{\text{singleton}}(i).
$$

Relative residuals $r(S)/\sum_{i\in S}\mathrm{KL}_{\text{singleton}}(i)$,
computed from the study's artifact on 2026-10-17:

| Operator | Window of 2 | Window of 3 | Window of 4 |
| --- | ---: | ---: | ---: |
| Random-init SwiGLU, 50% width | median +11% (range -11% to +58%) | +20% (+3% to +45%) | +34% (+15% to +54%) |
| Ridge-fitted linear | +9% (-24% to +86%) | +22% (-31% to +88%) | +30% (-35% to +82%) |

Limits: 12,288 calibration pairs, operators that are not the allocator's
fitted SwiGLUs, at most four neighbouring layers, one width.

**(b) Allocator prediction against the joint model at token 0.** The
allocator's predicted sum (`predicted_teacher_kl` in the preparation
artifacts) against the fitted model's monitoring KL before any recovery:

| Target | Predicted sum | Joint KL at token 0 | Joint / predicted |
| ---: | ---: | ---: | ---: |
| 20% | 0.0901 | 0.1106 | 1.23 |
| 30% | 0.1481 | 0.1834 | 1.24 |
| 40% | 0.2144 | 0.2809 | 1.31 |
| 50% | 0.2949 | 0.3969 | 1.35 |
| 60% | 0.3903 | 0.5481 | 1.40 |
| 70% | 0.5125 | 0.7646 | 1.49 |
| 75% | 0.5896 | 0.9012 | 1.53 |
| 80% | 0.6770 | 1.0515 | 1.55 |

20-50% are SwiGLU-7's S7-0 runs (SwiGLU-8's fitted runs match them exactly
at token 0). 60-80% are SwiGLU-8's fitted runs.

Limits:
- The two KLs use different held-out C4 partitions: allocation selection
  versus recovery validation.
- At 60-80% the predicted sum contains one interpolated boundary cost.
- Each row is a different budget. Rising ratios show that damage compounds
  as more is removed. They say nothing about the ordering of allocations
  **within** one budget, which is what the allocator relies on.

**Interpretation.** Both sources agree that the sum underestimates joint
damage, and that the underestimate grows with the amount replaced. This
matches SPDY's caution [`src-spdy-2022`, Section 3.1]. Unregistered later
work claims that the additive dynamic program can choose clearly worse
allocations (Section 8.3).

**A possible ranking test (proposal, not decided).** At one or two budgets,
assemble several allocations: the solver's choice, close runners-up from the
table, a uniform allocation, and SwiGLU-8's re-solved 50% variant. Measure
each one's joint token-0 KL and compare the two orderings. All grid widths
are already fitted, so this costs only assembly and evaluation passes, not
new fits or recovery. It answers the ranking question at token 0, not after
recovery.

## 6. The evidence so far

| Study | What was compared | Result | Limits |
| --- | --- | --- | --- |
| [SwiGLU-2](../experiments/model/swiglu/swiglu-2.md) | Ranked allocation from singleton KL at 25% width against uniform widths, 50% target | Lower post-recovery PPL and KL than uniform ([progression](../experiments/model/swiglu/swiglu-progression.md)) | Precursor ranking rule, not the knapsack; one-epoch recovery |
| [SwiGLU-5](../experiments/model/swiglu/swiglu-5.md#observed-result) | Knapsack allocation (S5-C2) against the ranked control (S5-C0) with the same recovery recipe, 5M tokens | Better KL and PPL at 20% and 50% | The 20% confirmation is the near-runner-up because of the label defect; the 100M comparison with SwiGLU-3 also changes the recovery learning rate |
| [SwiGLU-7](../experiments/model/swiglu/swiglu-7.md) | Allocator rebuilt independently from the pinned model and data | Fixed control for 20-50% | No alternative allocator in the run |
| [SwiGLU-8](../experiments/model/swiglu/swiglu-8.md#part-b-results-2026-10-12) | Re-solve at 20-50% with the extended grid; new allocations at 60-80% | Re-solving changes nothing at 20-40%, two layers at 50% (0.1% lower predicted KL). At 60-80% the middle layers 5-15 get the narrowest widths | The 75%/80% gap anomaly may relate to allocation structure; untested |
| Section 5.3 | Predicted sum against joint KL | Super-additive, growing with removal | Ranking untested |

**Gaps.**
- The knapsack was never compared with uniform widths directly. It beat the
  ranked rule (SwiGLU-5), which beat uniform (SwiGLU-2), but under different
  recipes.
- The allocator-level ranking has never been checked.
- No alternative cost (for example local NMSE) has been tried inside the
  knapsack.
- One model, one seed per allocation.

## 7. Equation summary

| Eq. | Content | Label | Citation |
| --- | --- | --- | --- |
| (1) | $p_i(k)=3dk$ | Standard bookkeeping | No |
| (2) | Retained budget $B$ | Standard bookkeeping | No |
| (3) | Multiple-choice knapsack ILP | Standard; SPDY's form as compression allocation | Textbook (unregistered); `src-spdy-2022` §3.1 Eq. (1) |
| (4) | Dynamic-program recurrence | Standard; same as SPDY | Textbook (unregistered); `src-spdy-2022` §3.2 Eqs. (2)-(3) |
| (5) | Boundary-width rule | Project heuristic | No |
| (6) | Joint KL ≈ summed singleton KL | Project-proposed modelling choice | No; contrast `src-modegpt-2025` §3.3 |
| (7) | Isotonic regression | Standard algorithm | Reference (unregistered) |
| (8) | Interaction residual | Project-defined diagnostic | No |
| (9) | Byte weight per option | Standard bookkeeping, following Stacking-1 | No |

## 8. Prior work and positioning

### 8.1 Registered inspirations

- **SPDY** [`src-spdy-2022`]. Chooses one sparsity level per layer from a
  discrete set to minimize summed per-layer error under a summed runtime
  budget. Solves it exactly with a dynamic program over discretized time
  (Sections 3.1-3.2). Its per-layer error is a parametric curve whose
  coefficients an outer random search tunes against the loss of the
  stitched model (Section 3.3).
- **ZipLM** [`src-ziplm-2023`]. Prunes attention heads and FFN intermediate
  neurons with a layer-wise reconstruction update and stores several pruned
  versions per layer (Section 3.1). Chooses per-layer levels with an
  SPDY-based search under a runtime budget (Section 3.2), then distills
  (Section 3.3). It is the closest prior application to structured FFN
  width.

The researcher records both as inspirations for the allocator. Their
locators come from automated fetches of the HTML renderings and still need
confirming in the PDFs.

### 8.2 What differs here

| | SPDY / ZipLM | This project |
| --- | --- | --- |
| Budget | Runtime | Retained parameters (bytes in Section 9) |
| Per-option cost | Parametric curve tuned by search (SPDY); search with a layer-wise error prior (ZipLM) | Measured whole-model singleton KL per option, isotonic-smoothed |
| Solve | Dynamic program inside an outer search | One dynamic-program solve, no search |
| Local step | Weight reconstruction of the pruned layer | Gradient-trained local fit of a narrower SwiGLU |
| After allocation | Fine-tuning or distillation | Recovery training to 100M or 1B tokens |
| Model | CNNs, BERT, GPT-2 | 1.7B causal LLM |

**Positioning.** The allocation algorithm is not a contribution. The
project's homogeneous use (teacher neuron selection plus local fitting) is
close to structured width pruning with layer-wise reconstruction, so the word
"replacement" alone does not separate it from ZipLM. What is specific to this
project:
- a measured model-level cost per option inside an exact solve;
- evidence after long recovery;
- the measured super-additivity of Section 5.3;
- the heterogeneous extension of Section 9, the most clearly new part.

A literature search on 2026-10-17 found no paper combining fitted narrower
replacements, a measured singleton-KL cost, and an exact multiple-choice
knapsack. That search was short and is not proof of absence.

### 8.3 Unregistered references (leads, not evidence)

Seen through web searches and automated fetches on 2026-10-17. None is
registered, and no claim in this note rests on them beyond pointing to them.

- Kellerer, Pferschy, Pisinger, *Knapsack Problems* (Springer, 2004): the
  multiple-choice knapsack chapter, for Eqs. (3)-(4).
- Ayer et al. (1955) or Barlow et al. (1972): isotonic regression and
  pool-adjacent-violators, for Eq. (7).
- EvoPress (Sieberling et al., arXiv 2410.14649): argues that per-layer
  errors are neither additive nor monotone, and that whole-model
  evolutionary search beats an SPDY-like dynamic program.
- DarwinLM (Tang et al., arXiv 2502.07780): evolutionary search with
  whole-model KL fitness for non-uniform structured MLP-width and
  attention-head pruning of LLMs, then post-training.
- SNIPER (Goel et al., arXiv 2608.12953) and GAC (Xin et al., arXiv
  2604.09595): recent knapsack allocations solved by dynamic programming.
  GAC is a multiple-choice knapsack under a parameter budget.
- HAWQ-V3 (Yao et al., 2021): per-layer bit-width allocation as an integer
  linear program, prior work for Section 9.

## 9. Extension: operator family × width × bit-width per layer

**Status: project proposal, not decided.** The heterogeneous chapter is
agreed (decision log, 2026-10-16), but it is undesigned beyond the per-family
recipe study (2026-10-11). Stacking-1 decided uniform bit-widths only.

### 9.1 Formulation

Each layer's option becomes $o=(f,k,b)$: operator family $f$, capacity $k$,
and storage format $b\in\{\text{BF16},\text{INT8},\text{INT4},\text{INT3}\}$.
The problem keeps the shape of Eq. (3): still one option per layer and still
a multiple-choice knapsack. Two things change.

**Eq. (9), standard bookkeeping following Stacking-1.** The weight becomes
bytes, because quantization changes bytes per parameter:

$$
w_i(o)=\Big\lceil \tfrac{1}{8}\,\beta_b\,p_i(f,k)\Big\rceil ,
\qquad \beta_b\approx 16,\ 8,\ 4.25,\ 3.25 \text{ bits per weight},
$$

including group-scale overhead and the zero padding of down projections to
multiples of 128
([Stacking-1](../experiments/model/stacking/stacking-1.md); decision log,
2026-10-17). INT3 bytes are calculated, not measured.

**Budget units.** For an exact dynamic program, weights must be integers in a
common unit. Use their greatest common divisor, or round weights **up** to a
unit; rounding up keeps every solution feasible and costs at most one unit
per layer of unused budget.

Costs stay singleton KL: fit $(f,k)$, quantize it to $b$, insert it alone,
and measure.

### 9.2 What it costs to build

- With, for example, 3 families × 9 capacities × 4 formats there are 108
  options per layer and 2,376 singleton evaluations. The dynamic program
  stays small.
- Local fits are needed for every family and capacity: 3 × 9 × 22.
- Post-training quantization needs no new fit; quantization-aware variants
  would.
- GPU cost is not estimated here; it needs the PERUN resource documents.

### 9.3 Risks

1. **Costs before recovery may rank families wrongly.** In SwiGLU-8 the
   slice-only arm started 110-427% worse than the fitted arm in monitoring
   KL. At 100M tokens the gap was 10-24% in 8K KL
   ([Part B results](../experiments/model/swiglu/swiglu-8.md#part-b-results-2026-10-12)).
   Damage before recovery overstated what recovery could not fix. Families
   may differ in how much recovery repairs, so singleton costs measured
   before recovery could favour the wrong family.
2. **Additivity likely gets worse.** Quantization noise in every layer adds
   to replacement error. Section 5.3 already shows super-additivity for
   replacement alone.
3. **Smoothing becomes two-dimensional.** Cost should fall with $k$ within
   $(f,b)$ and with $b$ within $(f,k)$. One-dimensional pool-adjacent-violators
   no longer applies directly. Options: bivariate isotonic regression, or
   only removing dominated options. An option is dominated if it is no
   cheaper and no lighter than another in the same layer. Removing dominated
   options is standard knapsack preprocessing and does not change the
   optimum.
4. **Mixed bit-widths per layer are prior work** (HAWQ-V3, unregistered) and
   go beyond Stacking-1's uniform formats.
5. **Speed.** Mixed formats per layer complicate kernels. Speed stays
   optional and descriptive (decision log, 2026-10-16).

## 10. Questions for the supervisor

1. **Keep the additive dynamic program as a stated simplification, or add a
   search-based comparison (EvoPress-style)?** The additive solve is cheap
   and exact for its surrogate. Section 5.3 shows the surrogate is biased.
   Whether the bias changes decisions is untested.
2. **Run the ranking test of Section 5.3 before the heterogeneous chapter?**
   It is cheap and directly tests the assumption the allocator depends on.
3. **Scope of the extension.** Allocate family × width with uniform bits
   (smaller, closer to Stacking-1), or family × width × bits per layer
   (larger; overlaps mixed-precision prior work)?
4. **Budget unit for the heterogeneous chapter.** Parameters, as in the
   homogeneous control, or bytes, so that quantization can be compared
   fairly?

## Sources

Registered: `src-spdy-2022` (Section 3.1, Eq. (1); Section 3.2,
Eqs. (2)-(3), Algorithm 1; Section 3.3), `src-ziplm-2023` (Sections
3.1-3.3), and `src-modegpt-2025` (Section 3.3, Eqs. 10-11); see
[`llm-wiki/raw/sources.yml`](../../llm-wiki/raw/sources.yml). Unregistered
leads are listed in Section 8.3.

Project evidence: the artifacts in the front matter. The offline calculations
of Sections 5.2-5.3 were run read-only on 2026-10-17. Experiment numbers
belong to their linked experiment pages.

Wiki: [Discrete width-curve allocation](../../llm-wiki/wiki/methods/method-discrete-width-curve-allocation.md)
and [replacement error propagation](../../llm-wiki/wiki/concepts/concept-replacement-error-propagation.md).
The older softmax-cap proposal is
[global-to-local operator budget allocation](global-to-local-operator-budget-allocation.md).
