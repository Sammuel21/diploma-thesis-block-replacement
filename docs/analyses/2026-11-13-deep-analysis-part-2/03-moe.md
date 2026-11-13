---
title: Deep Analysis Part 2 - Mixture-of-Experts
status: draft
created: 2026-11-13
modified: 2026-11-13
---

# 3. Mixture-of-experts: what is taken, what is open, and a design that scales

Part of the [deep analysis, part 2](README.md). It covers your topic 1.4 and
extends [Part 1, file 3, section 3.2](../2026-11-12-deep-analysis/03-future-work.md),
which proposed the steps M1-M3. That section is not repeated here. New in this
file: a literature check, the cost problem you raised, and a way around it.

**About the sources.** Every paper and model fact in this file was read by a
research sub-agent on 2026-11-13 through a tool that summarises pages, not by
me. Many of the papers are from 2026 and newer than my own knowledge. They are
marked **[agent]**. The full report with links is in
`private/agent-reports/moe-prior-work.md`. Read a source before citing it.

## Short answer

- **The field is crowded and fast.** The check found about twenty papers on
  removing, merging, or quantizing experts, several from the last three
  months.
- **Three ideas are already published:** replacing an expert by a constant
  (MoNE), one global program with one choice per expert (GEMQ, for
  bit-widths), and narrower experts (SlimMoE, with heavy distillation).
- **What the check did not find** is the combination your tooling already
  does on a dense model: a *measured* cost at the model output, for *each
  option* of each expert, inside an *exact* allocation over a *mixed menu*.
  Also not found: any measurement of whether expert damage adds up.
- **The cost problem has a way out, and it is the result of file 2.** Local
  error ranks options inside a layer; a few model-level measurements set each
  layer's scale. If that holds for experts, a price table for a thousand
  experts needs tens of measurements, not thousands.
- **Expect quantization to win on bytes again.** The chapter's claim would be
  about how to choose and what to offer, not about beating low-bit experts.

**Flags.** ROI: a possible NEW RESULT, with real risk from the pace of the
field. Scope: MOE-ONLY; the pricing method is the TRANSFERABLE HYPOTHESIS of
file 2.

## 3.1 What is already published

The papers that constrain your claim most **[agent]**:

| Paper | What it does | How it scores | Budget across layers | Retraining |
| --- | --- | --- | --- | --- |
| MoNE (arXiv 2507.00390) | Expert → constant vector (its mean output) | Local: spread of the expert's output × mean routing score | Same ratio per layer | None in the method |
| REAP (2510.13999) | Remove experts | Local: routing gate × output size | Uniform | None |
| HOPE (2609.18916) | Remove experts | Local pairwise terms within a layer, solved jointly | Uniform; its cross-layer variant did worse | None |
| EvoESAP (2603.06003) | Remove experts | Whole allocations scored at the model output | Non-uniform, by evolutionary search | None |
| MoE-I² (2411.01016) | Remove, then low-rank inside survivors | Loss with the expert removed | Non-uniform, heuristic | LoRA |
| GEMQ (2605.23078) | Bits per expert, 1 to 3 | Estimated loss increase (gradient-weighted), not measured | One global program, one choice per expert | Router only |
| SlimMoE (2506.18349) | Narrower experts | Not in the abstract | Not read | Distillation, 400B tokens |

Three details from MoNE matter for your design **[agent: MoNE sections 4-5,
Tables 1, 3, 10]**:

- **Its score is a local error in disguise.** The spread of an expert's
  output on the tokens routed to it is the error a constant makes there. So
  MoNE prices one menu item (the constant) by its local error, and REAP
  prices another (zero) the same way. That reading is mine, not the papers'.
- **Its own tables show where a local score runs out.** At 50% removal on
  Qwen3-30B a plain routing-score baseline beats it (68.06 against 66.32).
- **After fine-tuning, its advantage is gone:** 64.18 against 64.09 for the
  routing-score baseline on OLMoE. This is the same pattern as your
  fit-versus-slice result: a selection signal that matters without retraining
  and not with it. The chapter therefore belongs in the no-retraining regime.

## 3.2 What still looks open

| Open item | Closest prior work | Why yours would differ |
| --- | --- | --- |
| A measured model-output cost per expert and per menu item, in an exact allocation over a mixed menu (keep, narrower, constant, remove) | GEMQ (exact, but bits only, estimated cost); MoE-I² (measured, but heuristic allocation); EvoESAP (measured, but search over keep or remove) | It is the dense pipeline, unchanged in kind |
| Whether damage from several replaced experts adds up, within a layer and across layers | HOPE measures a local pairwise term (about a third of the single term on one model) | Your pairwise interaction study, repeated on experts |
| How well cheap local scores agree with measured importance | One workshop audit found that no routing statistic predicted measured loss change (rank correlation −0.08 to +0.20) | File 2's analysis, repeated on experts |

"Not found" means about 25 searches found nothing. The agent did not search
full texts, and the field moves monthly. This needs the same kind of check
that found Puzzle, repeated just before any claim is written.

## 3.3 Three hypotheses that carry file 2 over

These are project hypotheses, to be tested, not results **[view]**.

1. **Within a layer, an expert's damage is proportional to its total local
   error.** All experts of one layer write into the same place in the model,
   so they share whatever comes after. "Total" matters: the error is summed
   over the tokens routed to that expert, weighted by the gate, so a rarely
   used expert is cheap. Per-expert normalised error would repeat the mistake
   NMSE makes on a dense model.
2. **Between layers, the scale differs,** as it does on the dense model and
   as GEMQ notes for its own scores ("not comparable across layers")
   **[agent]**. One measured KL per layer would set it.
3. **Damage adds up better within a layer than across layers,** because two
   experts of one layer interact only on tokens that use both. HOPE's number
   says this is not negligible, so it has to be measured.

If hypothesis 1 holds, the price of *every* option of *every* expert comes
from local errors, which one capture pass per layer provides, plus one
measurement per layer.

## 3.4 A ladder of prices, and how to choose a rung

| Rung | Price of an option | Model-level measurements, OLMoE (1,024 experts, 5 options) |
| --- | --- | ---: |
| 0 | A published local score (MoNE, REAP) | 0 |
| 1 | Total local error of that option | 0 |
| 2 | Rung 1 × one measured KL per layer | 16 |
| 3 | Rung 1 weighted by a gradient estimate (the GEMQ idea; file 2, step 3) | 0, plus one backward pass |
| 4 | One measured KL per expert × local error | 1,024 |
| 5 | Measured KL for every option | 5,120 |

**How to choose without paying for rung 5.** Measure the full menu for a
sample of experts, for example 96 spread over layers and usage levels (about
480 measurements). Compare each rung's ranking with that truth. This is file
2's analysis on a sample, and it is the mixture-of-experts version of the
supervisor's question.

## 3.5 Steps

Revised from Part 1. Costs are unknown until a plan session measures one KL
evaluation on the chosen model.

| Step | What | Retraining | Answers |
| --- | --- | --- | --- |
| M0 | Reproduce MoNE's row for one model at 25% | None | Is the new model path trustworthy? Your reproduction-gate habit, applied to a paper |
| M1 | The price ladder on a sample of experts | None | Is the local fit predictive in an MoE model? Which rung is enough? |
| M2 | Same number of removed experts, chosen by MoNE, by REAP, and by the chosen rung with a free budget across layers | None | Does a measured, non-uniform choice beat local scores? |
| M3 | The mixed menu at M2's byte budgets, with and without local fitting | None | Does a mixed menu beat all-or-nothing? Does fitting matter here? |
| M4 | The pairwise interaction study on a sample of expert pairs | None | Does damage add up, within and across layers? |
| M5 | 100M tokens of retraining for the best two | 100M | Expected to erase most differences (MoNE's own Table 3); optional |

M0-M2 are a complete small chapter. M3 is where your substitutes come in.

## 3.6 Which model

Configurations as read from the model cards and configuration files
**[agent]**; the expert share is the agent's arithmetic.

| Model | Layers × experts (active per token) | Hidden / expert width | Parameters, total / active | Share in experts | In MoNE's table |
| --- | --- | --- | --- | ---: | --- |
| OLMoE-1B-7B | 16 × 64 (8) | 2048 / 1024 | 7B / 1.3B | about 93% | Yes: 66.63 → 61.04 at 25% |
| Granite-3.1-1b-a400m | 24 × 32 (not checked) | 1024 / 512 | 1.3B / 0.4B | about 93% | No |
| Granite-3.1-3b-a800m | 32 × 40 (8) | 1536 / 512 | 3.3B / 0.8B | about 91% | No |
| DeepSeek-V2-Lite | 27 × 64 + 2 shared (6) | 2048 / 1408 | 15.7B / 2.4B | about 92% | Yes |
| Qwen3-30B-A3B | 48 × 128 (8) | 2048 / 768 | 30.5B / 3.3B | about 95% | Yes |

- **Recommended: OLMoE-1B-7B as the main model.** It is the smallest model
  with a published MoNE row, so M0 has a target and M2 a comparison. Its
  hidden size equals SmolLM2's.
- **A Granite model as a cheap development model,** if iteration speed
  becomes the limit. It has no published baseline, so it cannot replace
  OLMoE.
- **Experts hold over 90% of the parameters,** against 64.7% for the eligible
  MLPs of SmolLM2. Structural methods have more room here than on the dense
  model **[calc, on agent-read configurations]**.

## 3.7 What to expect, honestly

| Topic | Expectation |
| --- | --- |
| Bytes | Low-bit experts will very likely win. GEMQ reports Mixtral going from 87 to 16 GB at 1-3 bits per expert **[agent]**. Say so up front |
| Compute | Removing a rarely used expert saves almost nothing; narrowing a heavily used one does. Bytes and compute separate even more than on a dense model |
| Calibration data | MoNE reports a large effect of the calibration source on OLMoE (56.97 with C4 against 61.04 with its own data) **[agent]**. M0 must use the paper's data; your own data choice is then a stated variable |
| Evaluation | MoNE uses nine tasks; yours are five. M0 needs theirs |
| New code | Capture per expert (pairs exist only for routed tokens), substitutes that keep the router's gate, a solver with back-pointers ([file 1](01-meeting-3.md), section 1.1) |
| Serving | An engine that supports the model may assume the same number of experts in every layer **[agent: vLLM's Qwen3 class]**; a mixed menu would need a custom class |
| Novelty | Re-check the literature the week you write the claim |
