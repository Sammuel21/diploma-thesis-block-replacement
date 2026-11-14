---
title: Deep Analysis Part 2 - More Operators, Local Losses, and a Matched MoE Model
status: draft
created: 2026-11-14
modified: 2026-11-14
---

# 8. More operators, more local losses, and an MoE model that matches SmolLM2

Part of the [deep analysis, part 2](README.md). Added on 2026-11-14 for a
point of the 2026-11-13 meeting that the first digest left out (your topic
1): the discussion of hybrid operators. It extends
[file 1](01-meeting-3.md) (new point 1.7),
[file 2](02-local-fit-as-selector.md), and [file 3](03-moe.md). Evidence
labels and flags are defined in the [README](README.md#labels-and-flags).

**About the sources.** Nothing was run for this file. Two sub-labels are used
for literature. **[lit: search]** means a web search on 2026-11-14 returned a
summary of the page and I did not open the page. **[lit: recalled]** means
from memory, unchecked. Both are discovery aids, and none of these sources is
registered in the wiki. Links are at the [end](#sources).

## Short answer

- **More operators will very likely not move the dense frontier. They are
  still worth adding, for two other reasons.** They are test items for the
  local-fit selector, whose evidence so far rests on three families and the
  linear map. And they are the menu of the mixture-of-experts chapter.
- **Heterogeneous-1 already says where a new operator can get in:** a start
  from the teacher's weights, a price the slices cannot offer, or blocks that
  are used unequally. That test keeps low-rank linear maps, a slice with a
  low-rank correction, and a conditional constant. It drops KAN, RBF, and
  structured matrices.
- **On an MoE model a dense linear map is not a cheap item.** It costs $d^2$
  whatever the expert's width: 67% of one OLMoE expert and 100% of a
  Granite-3b expert. Linear items there have to be low-rank, and that
  conflicts with a 2026-11-04 decision row.
- **"Loss" means two things: what a fit minimises, and what ranks the fitted
  candidates.** For ranking inside one layer, MSE has almost nothing left to
  gain on the dense model. The room is between layers and between experts,
  where the scale of the error is the problem. Two scores could remove the
  scale: error relative to the residual stream, and a gradient-weighted
  error.
- **OLMoE-1B-7B is the closest thing to "SmolLM2 as an MoE".** It has the
  same hidden size, and per token it runs 8 experts of 1,024 neurons, which
  is 8,192 neurons: exactly the width of SmolLM2's MLP. No model gives you
  the same control. The shared protocol does.

**Flags.** Operators on the dense model: NEUTRAL for the frontier, PROTECTIVE
for the local-fit claim. Scores that are comparable between blocks: enabling
for MoE, a TRANSFERABLE HYPOTHESIS. Model choice: MOE-ONLY.

## 8.1 What was said, and how it sits with the decision log

I was not at the meeting. This rests on your message of 2026-11-14 and on
[backlog item 4](../../backlog/backlog-ideas.md) (2026-11-13): hybrid
operators were discussed, only zero, mean, linear, plain MLP, and the two
linear hybrids were tested, and the supervisor encouraged more creative
operators. Your idea since then: widen the set, pick by local-fit loss, try
several losses, and compare dense with MoE.

AGENTS.md asks me to point out conflicts and not choose silently.

| Your idea | Existing row or text | Conflict | What settles it |
| --- | --- | --- | --- |
| New operator families on dense SmolLM2 | 2026-11-13: "heterogeneous replacement on dense SmolLM2 is closed"; this analysis and Part 1: further families on the dense model are NEUTRAL | Only if they are meant as candidates for a better frontier point | Treat them as test items of the local-fit study (the 2026-11-13 locality row), with no model-level run. Question 1 in 8.8 |
| Low-rank linear maps | 2026-11-04: "Low-rank linear is excluded" (built by truncating a fitted map, so stacking) | Direct | New evidence: on MoE the dense map is not affordable (8.4). Needs a row. Question 2 in 8.8 |
| Selection by local fit "instead of the knapsack" | File 2: local error can choose within a layer and cannot choose between layers | None, if "selection" means the operator inside a block | The between-block price still needs one of the rungs of 8.5 |
| Equal tuning for every new trained operator | 2026-11-04: each operator repeats the recipe study, about 8 GPU-hours | Not a conflict, a cost: it applies to every family trained by gradient descent | Prefer families with a closed-form fit (8.3) |

## 8.2 Where can a new operator get in?

What Heterogeneous-1 measured **[read: heterogeneous-1.md]**:

- SwiGLU slices won 505 of 506 comparisons at equal price.
- Operators that start from random weights fell further behind as they grew
  (plain MLP: 1.05 to 6.67 times SwiGLU's damage from 6.25% to 87.5% of an
  MLP). The one competitor with a teacher start stayed closest at wide sizes
  (linear + SwiGLU: 1.14 to 1.52).
- The free items, zero and mean, do 1.5 to 1.6 times the damage of the
  smallest slice (width 256, 3.1% of an MLP).
- 10M tokens of single-block training changed no ranking.

From that, a new operator has three ways in **[view]**:

| Way in | Why it helps | On dense SmolLM2 |
| --- | --- | --- |
| A start from the teacher | A slice inherits a trained function; a fresh block has to learn it from the calibration pairs | Only SwiGLU slices and linear + SwiGLU have one |
| A price the slices cannot offer | Between mean (2,048 parameters) and the narrowest slice (1.57M) the menu is empty | Untested. It can matter only at 70-90% removal, where width 256 changed the solve. Narrower slices (128, 64) are the obvious rival |
| Blocks that are used unequally | A rarely used block can take a crude substitute at almost no cost | Does not exist on a dense model. This is MoE |

A candidate with none of the three has no mechanism for winning. I use that
as the filter below.

## 8.3 Candidates

Parameter counts are standard bookkeeping and need no citation; $d = 2048$,
and shares are of one SmolLM2 MLP (50.3M) **[calc]**.

**Worth a place in a screen:**

| # | Operator | What it is | Parameters | Fit | Way in | Prior work to check |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Low-rank linear, rank $r$ | Two thin matrices, $d \to r \to d$ | $2dr$: rank 64 is 0.52%, rank 256 is 2.1% | Closed form (reduced-rank regression, a standard method) | Fills the empty price range; the only affordable linear item on MoE | Excluded by the 2026-11-04 row |
| 2 | Slice + low-rank correction | A SwiGLU slice, plus a rank-$r$ linear map fitted to what the slice misses | $3dk + 2dr$ | Slice, then closed form | Teacher start. It is linear + SwiGLU with a cheap linear part: rank 64 costs as much as 43 neurons, the dense map cost 683 | The factorised form on your [hybrid page](../../../llm-wiki/wiki/methods/method-hybrid-operator-replacement.md) |
| 3 | Diagonal map | One scale and one shift per dimension | $2d$ | Closed form | A nearly free step between mean and rank $r$ | - |
| 4 | Conditional constant | $K$ stored output vectors, chosen by a small detector on the input | about $2dK$ | Clustering or a short fit | Only if layers 1, 7, and 22 really write a few fixed, very large vectors (file 2's hypothesis) | Run file 2's per-position check first |
| 5 | Redirect to a kept expert (MoE only) | Tokens routed to expert $i$ are served by expert $j$ | 0 | None | Unequal use; experts of one layer share their input and output space | Expert merging: MC-SMoE, HC-SMoE **[agent]** |
| 6 | Shared expert + low-rank difference (MoE only) | One expert per group plus a rank-$r$ correction per member | $2dr$ per member | Closed form on what the shared expert misses | Same | LorExperts, MoBE **[agent: abstract or snippet only]** |

- **The control for items 1-3 is a narrower slice.** Add SwiGLU 128 and 64
  to the menu first and re-solve, as Heterogeneous-1 did with width 256.
  Otherwise a gain from a cheaper price would be credited to the new family.
- **Item 2 is the one honest retest of "does a linear part pay".** In
  Heterogeneous-1 it never did, at a price of 683 neurons. At 43 neurons the
  question is open **[view]**.
- **Items 1, 3, 5, and 6 need no training run.** Their fit and their local
  error come from one pass over the calibration pairs. That matters more than
  it seems: see 8.4.

**Considered and not recommended:**

| Operator | Why not |
| --- | --- |
| KAN, FastKAN, RBF units | Every connection stores several numbers instead of one (about 8-10 for the usual spline grid **[lit: recalled]**), so at equal bytes there are about a tenth of the connections. No teacher start. A comparison at equal parameters found plain MLPs generally better outside symbolic-formula tasks, and spline activations added to an MLP did not help elsewhere **[lit: search, arXiv 2407.16674]**. Your backlog's doubt about distances in 2,048 dimensions stands |
| Other activations in a plain MLP (GELU, ReLU, softmax units, one learned per neuron) | Same bytes and the same random start. It changes the shape of the nonlinearity, not what the block can read from its input. Cheap to try ([`create_activation`](../../../src/mlp_replacement/operators/modules.py) already takes three), low expectation |
| Deeper narrow MLP ($d \to h \to h \to d$) | Adds only $h^2$ parameters, but still reads the input through $h$ features, which is the limit at small $h$. Random start, harder to train |
| Structured matrices inside the block (Kronecker, tensor-train, Monarch, block-diagonal) | Fewer bytes per connection, but the teacher's matrices do not have that structure, so there is no teacher start, and they need their own learning rates and initialisation **[lit: search, Qiu et al., ICML 2024]**. Projecting the teacher onto a structure is algebraic reduction, which the 2026-10-20 definition calls stacking |
| Low-rank factors of the teacher's own three matrices | Stacking by the same definition. Part 1's verdict: future-work text only |
| Bilinear block (SwiGLU without its activation), other gates | The same size as SwiGLU, so no price advantage, and no reason to expect less damage |
| Per-token or per-position tables | Not a function of the block's input alone, so it leaves the annotation's "isolated function" |
| One operator shared by several layers | A different method class (Part 1, section 3.4) |
| Splitting the dense MLP into experts with a router | Saves arithmetic per token and no bytes. Published as MoEfication **[lit: search, Findings of ACL 2022]**. See 8.7 |

**One attribution point.** Your wiki page calls the linear + MLP hybrid a
researcher proposal with no external source. A published block has the same
shape: the "skip transcoder" adds an affine skip connection to an MLP that is
trained to imitate a transformer MLP block, and reports a lower
reconstruction loss with it **[lit: search, arXiv 2501.18823]**. Its purpose
differs (wide and sparse, for interpretability, not small). Two things
follow. If the hybrid is described in the thesis, that paper probably has to
be read and cited as related work. And its start, the nonlinear output
weights and the skip at zero and the bias at the mean output, is close to the
"linear first" start that Heterogeneous-1's follow-up proposed against the
stalled fits.

## 8.4 What changes on a mixture-of-experts model

One expert has $3dm$ parameters for expert width $m$. A dense linear map has
$d^2$, so its share of one expert is $d/(3m)$ **[calc; configurations from
file 3, agent-read; OLMoE confirmed by search]**.

| Model | $d$ | $m$ | One expert | Dense linear map | Rank-64 linear |
| --- | ---: | ---: | ---: | ---: | ---: |
| SmolLM2-1.7B (one MLP) | 2048 | 8192 | 50.3M | 8.3% | 0.5% |
| OLMoE-1B-7B | 2048 | 1024 | 6.3M | 67% | 4.2% |
| Qwen3-30B-A3B | 2048 | 768 | 4.7M | 89% | 5.6% |
| DeepSeek-V2-Lite | 2048 | 1408 | 8.7M | 48% | 3.0% |
| Granite-3.1-3b-a800m | 1536 | 512 | 2.4M | 100% | 8.3% |
| Granite-3.1-1b-a400m | 1024 | 512 | 1.6M | 67% | 8.3% |

- **Linear and both linear hybrids fall off the menu** unless the linear part
  is low-rank. This is the new evidence against the 2026-11-04 exclusion.
- **Fits multiply by the number of experts** (1,024 in OLMoE). An item with a
  closed-form fit or no fit gets its local error for every expert from one
  capture pass per layer. An item trained by gradient descent needs one run
  per expert and size.
- **Each expert sees few tokens.** With 8 of 64 experts active, an average
  expert sees one token in eight, and rarely used experts far fewer. Small
  items can be fitted from few tokens; a wide fitted expert cannot. Your old
  dense linear fit was already starved at 6 pairs per input dimension
  **[read: operator.md]**.
- **The error must carry the routing weight.** An expert adds its output
  times its gate, so the error it causes is the gate times its own error,
  summed over the tokens routed to it (file 3, hypothesis 1).
- **Items 5 and 6 couple choices.** If expert $i$ points at expert $j$, then
  $j$ must be kept, and the knapsack assumes independent choices. Either fix
  the kept targets first (for example the most used experts) or keep these
  items out of the exact solve **[view]**.
- **Every item already has a published single-item method:** constant (MoNE),
  zero (REAP and others), narrower (SlimMoE), merge, shared basis **[agent]**.
  That is useful. Each menu item has an outside baseline, and the open
  question stays the one in file 3: choosing among them per expert with a
  price measured or calibrated at the model output.

## 8.5 Local losses: two roles, and where there is room

### Role 1: the score that ranks fitted candidates

Notation: $\delta_t$ is the substitute's output minus the teacher MLP's
output on token $t$; $h_t$ is the teacher's residual stream after the block;
$g_{\ell,t}$ is defined under rung 3.

| Rung | Score | Needs | Comparable between layers? | Status |
| ---: | --- | --- | --- | --- |
| 0 | MSE, or NMSE, at the block output | The fit's own validation pairs | No: rank correlation with KL across layers −0.31 to +0.35 (file 2) | Stored today. Inside a layer the two rank identically, because NMSE divides by one constant per layer |
| 1 | Token-weighted variants: 1 − cosine, relative error per token, MSE without the first position, Huber | The same pairs | 1 − cosine is not (−0.20 to −0.42); the others unknown | Cosine is stored, the rest are new. They change which tokens count: plain MSE is dominated by the largest outputs |
| 2 | Error relative to the stream: $\mathbb{E}_t\,\lVert\delta_t\rVert^2 / \lVert h_t\rVert^2$ | The stream's size per token, one more captured number | Hypothesis: closer | Untested |
| 3 | Gradient-weighted error (formula below) | One backward pass over the calibration text, stored per layer | Should be, by construction | Untested here |
| 4 | One measured KL per layer × a rung 0-1 score | 22 KL passes | Yes: within 1-4% of the measured-KL allocation (file 2) | Shown offline |
| 5 | Measured single-block KL | One KL pass per option | It is the target | Today's price |

**Rung 2** is a project-proposed definition and needs no citation. Its
reasoning is mine and is a hypothesis: the next layer reads the stream
through a normalisation, so what it sees is the error relative to the stream,
not relative to the MLP's own output. File 2 supports the negative half only.
Dividing by the MLP's own output size is the wrong scale, and so is not
dividing: per unit of plain squared error the three outlier layers still
differ by about 200 times (0.77, 0.060, and 0.0034 for layers 1, 7, and 22)
**[calc on file 2, section 2.4]**.

**Rung 3.** With labels sampled from the teacher's own output and one
backward pass per sequence $s$, let $g_{\ell,t}$ be the gradient of that
sequence's log-probability with respect to layer $\ell$'s MLP output at
position $t$. Then

$$
\widehat{\mathrm{KL}}_\ell \;\approx\; \tfrac{1}{2}\,
\mathbb{E}_{s}\Big[\big(\textstyle\sum_{t\in s} g_{\ell,t}^{\top}\delta_t\big)^{2}\Big].
$$

This is the standard sampling estimate of the quadratic form in file 2,
section 2.6, so it inherits that section's status: an explanatory
formalization built on a standard approximation, to be cited to the same
source when used. I derived it in this session and did not run it; check it
before relying on it. The sum sits inside the square because an error at one
position changes the predictions at later positions. It is the step 3 of file
2 and the rung 3 of file 3, written out. Weighting block errors by squared
gradients is published for quantization (BRECQ **[lit: search, arXiv
2102.05426]**; a 2025 preprint argues its diagonal form is inaccurate), and
for expert bit-widths (GEMQ **[agent]**).

**On experts** every rung keeps its form, with $\delta_t$ replaced by the
gate times the expert's error and the mean replaced by a sum over routed
tokens. MoNE's and REAP's published scores are then rung-0 scores of one menu
item each (the constant and zero), as file 3 notes. So "several local losses"
on an MoE model is a family that contains the published baselines.

**Where there is room [calc on file 2].**

| Choice | What MSE achieves today | Room left |
| --- | --- | --- |
| The operator inside one layer | Same pick as KL in 163 of 176 cells; mean loss 0.31%, largest 8.8% | Almost none. A new score can only repair those 13 cells |
| Pricing another family from one probe | Plain MLP within 7%, linear + SwiGLU within 5%, the linear map 23% off | Some, for families that err in a different direction. This is what more operators test |
| Which layer to compress | Worse than uniform widths | Large. Only rung 4 is shown to work; rungs 2 and 3 would work without any probe |
| Which expert to compress | Unmeasured | Unknown; the reason to do this at all |

### Role 2: the objective a fit minimises

- **Today every fit minimises MSE.** Another objective changes which errors
  remain at the same size.
- **Only the gradient-weighted objective has a mechanism behind it:** if
  damage is a quadratic form in the error, the fit that minimises damage
  minimises that form, not the plain squared error.
- **A rough reference from your data, not a ceiling:** training one block on
  the real objective (KL at the output) for 10M tokens cut its damage by
  17-18% (Heterogeneous-1, stage 2) **[read]**. And what fitting buys shrinks
  to 0.3-2.8% after 1B tokens of retraining (SwiGLU-9).
- **So:** INCREMENTAL without retraining, NEUTRAL after it **[view]**. One
  alternative objective on a few layers is enough to say so with a
  measurement.

## 8.6 One table to fill

The study you describe is file 2's analysis with more rows and one more
column. Cells marked "known" are from file 2; allocation numbers are extra
predicted damage against the measured-KL allocation at 20-80%.

| Score | Within a layer, dense | Between layers, dense | On experts |
| --- | --- | --- | --- |
| MSE or NMSE | Known: rank correlation 0.995, picks 163 of 176 | Known, fails: +22% to +215% (NMSE), +17% to +160% (MSE) | Open. MoNE's and REAP's scores are of this kind |
| 1 − cosine | Not reported | Known, fails: +14% to +64% | Open |
| Relative per token; without the first position; Huber | Open | Open | Open |
| Relative to the stream | Open | Open: the hypothesis | Open |
| Gradient-weighted | Open | Open: expected to work | Open; GEMQ uses one for bits |
| One KL per layer × NMSE | As NMSE | Known: +1.4% to +4.2% | Open: one per layer (16) or one per expert (1,024)? File 3, section 3.4 |

- **One fit, many scores.** Every score of rungs 0-3 can be recorded on the
  same fitted block, so adding scores costs a capture, not more fits.
- **But the dense fits have to be repeated.** Heterogeneous-1 kept no fitted
  weights, only JSON, so new scores cannot be computed on the stored rows
  **[read: heterogeneous-1.md]**. Rough scoping from its measured cost
  tables: about 8 GPU-hours for six layers (3, 11, 19 and 1, 7, 22) with the
  existing 25 options, about 30 for all 22 layers **[calc]**. Not a PERUN
  resource plan.
- **The closed-form operators of 8.3 add little** to that: a fit takes
  seconds and a KL score about 4 seconds.
- **On MoE this is step M1 of file 3,** with the menu of 8.3 and the scores of
  8.5. It needs no separate experiment.

## 8.7 Which MoE model matches SmolLM2-1.7B?

"Comparable" has four meanings, and they pick different models.

| Meaning | Model | For | Against |
| --- | --- | --- | --- |
| The same MLP shape per token | **OLMoE-1B-7B** | Hidden size 2,048, as SmolLM2. 8 active experts × 1,024 neurons = 8,192 neurons per token, SmolLM2's MLP width **[calc; configuration: lit: search, and file 3]**. 1.3B active against 1.7B. A published MoNE row. Apache-2.0 **[agent]** | 6.9B stored, four times SmolLM2. 16 layers against 24. Another tokenizer and other training data |
| An MoE built from a dense model of SmolLM2's class | Qwen1.5-MoE-A2.7B with Qwen-1.8B | Its model card says it was upcycled from Qwen-1.8B. Dense Qwen1.5-1.8B has hidden size 2,048 and 24 layers, like SmolLM2 **[lit: search; the dense configuration from mirrors, not the official repository]** | 14.3B stored. 60 routed experts plus shared ones that are always on. No MoNE row. Research licence **[agent]**. The dense control would be Qwen, not SmolLM2. Which 1.8B checkpoint it started from, and how the experts were derived, I could not confirm |
| The same stored size | Granite-3.1-1b-a400m | 1.3B in total | Hidden size 1,024, experts of width 512, no published baseline; experts per token not confirmed |
| Literally the same control | SmolLM2 with its MLPs split into experts | Same teacher, tokenizer, and data | Synthetic. The experts would be groups of one MLP's neurons with a router fitted afterwards, so removing one is width removal under another name. It says nothing about trained MoE models and has no baseline |

One search found no official SmolLM MoE, only community builds **[lit:
search]**. That does not rule one out.

**Recommendation [view].**

- **OLMoE-1B-7B as the MoE reference model.** File 3 chose it because it is
  the smallest model with a MoNE row. The shape match is a second,
  independent reason: per token it applies an MLP of SmolLM2's size, made of
  eight pieces.
- **The Qwen pair belongs to the transfer scorecard** of
  [file 5](05-agnosticism-and-factors.md), if compute remains. It is the only
  place found where a dense model and an MoE built from it can both be run.
- **Granite as a development model only,** as in file 3.
- **No model gives "the same control". The protocol does:** the same
  calibration source and sizes, KL against the unmodified model, the same
  tasks and token budgets. The token streams must be rebuilt for another
  tokenizer, so absolute KL is not comparable between the two models. Only
  orderings are, which is what the scorecard records. File 3's step M0
  (reproducing MoNE's row) needs MoNE's own data and nine tasks and stays
  separate.

**To check before choosing:** which OLMoE checkpoint MoNE used (0924 or
0125); that its expert block has SwiGLU's three matrices, so that
`GatedMLPReplacement` maps onto it **[lit: recalled]**; and the time of one KL
pass.

## 8.8 Questions for you

Nothing here is decided. A row is needed for 2 at least, because the
exclusion is logged.

| # | Question | My recommended answer |
| --- | --- | --- |
| 1 | On the dense model, are new operators test items for the local-fit study, or candidates for the frontier? | Test items only. The 2026-11-13 heterogeneous-closed row stays as it is, and no model-level run is added |
| 2 | Lift the 2026-11-04 exclusion of low-rank linear? | Yes, as its own family fitted directly, not truncated from a dense fit; for MoE menus and for the dense test |
| 3 | Which families enter the first screen? | Closed-form only: low-rank linear at five or six ranks, slice + low-rank correction, diagonal, with SwiGLU 128 and 64 as the control. The conditional constant waits for the per-position check. No KAN or RBF |
| 4 | Which scores? | Record rungs 0-3 on every fit. At most one alternative training objective, the gradient-weighted one |
| 5 | The MoE reference model? | OLMoE-1B-7B, settled in the MoE design session, as the 2026-11-13 row says |
| 6 | What did the supervisor say about operators, in his words? | The meeting draft marks it "to confirm" |

**Wiki advisory.** If items of this file enter the thesis, these sources
need registering and a direct read: the skip transcoder paper (it bears on
the hybrid page's "no external source"), BRECQ, and, if they are discussed,
KAN with the equal-parameter comparison, MoEfication, and the structured
matrices paper. None was ingested.

## Sources

Search results of 2026-11-14, summaries only; open the page before citing.

- OLMoE configuration: <https://huggingface.co/docs/transformers/model_doc/olmoe>,
  <https://huggingface.co/allenai/OLMoE-1B-7B-0125/blob/main/config.json>
- Qwen1.5-MoE: <https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B>,
  <https://qwenlm.github.io/blog/qwen-moe>
- Qwen1.5-1.8B configuration (a mirror):
  <https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-AWQ/blob/0decca9d02c2e30a19bab7cb7eedb033a4831126/config.json>
- Granite-3.1-1b-a400m:
  <https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-base/blob/main/config.json>
- Skip transcoders: <https://arxiv.org/abs/2501.18823>
- BRECQ: <https://arxiv.org/abs/2102.05426>; the critique:
  <https://arxiv.org/abs/2506.11543>
- KAN or MLP: <https://arxiv.org/abs/2407.16674>
- MoEfication: <https://aclanthology.org/2022.findings-acl.71>
- Structured matrices: <https://proceedings.mlr.press/v235/qiu24f.html>
