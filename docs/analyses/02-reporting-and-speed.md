---
title: Deep Analysis 2026-11-12 - Result Reporting and Speed
status: draft
created: 2026-11-12
modified: 2026-11-12
---

# 2. Result reporting (the "360 view") and the speed question

Part of the [deep analysis of 2026-11-12](README.md). It covers your topic 8
(how to report results fairly) and topic 7 (research objectives: bytes only,
or speed as well). Evidence labels are defined in
[file 1](01-direction-and-design.md).

## Short answer

- **Your instinct is right, and the literature agrees with your data.** A
  NeurIPS 2023 study that compares the two families head-on concludes that
  quantization beats pruning in most settings, and that pruning can win only
  at very high compression **[lit: Kuzmin et al., "Pruning vs Quantization:
  Which is Better?", abstract checked]**. Pruning papers mostly compare with
  other pruning methods, which is why their tables look better than yours.
  Your result is the expected one, reported honestly.
- **Do not hide that result. Frame it.** Report three things side by side:
  bytes, compute, and both together. Report each at three retraining
  budgets. Add two kinds of anchor that are missing today: comparisons
  inside the pruning class, and natively trained smaller models.
- **The single most valuable addition costs under one GPU-hour:** evaluate
  SmolLM2-360M and SmolLM2-135M with your own evaluation and put them on
  every frontier plot. I expect the 360M model to beat every compressed
  model past about 56% saving on task accuracy. If so, that sets the range
  in which compressing the 1.7B model makes sense, and the thesis should say
  it before an opponent does.
- **On speed: you and your supervisor are both right, about different
  regimes.** You already report the two numbers that bound speed in the two
  regimes (bytes and linear FLOPs). A real speed measurement is defensible
  only for the comparison that involves no quantized kernels: dense against
  SwiGLU, both BF16, in a serving engine. That is a contained bonus, not a
  can of worms.

## 2.1 Why the current reporting feels unfair, stated precisely

Three separate things are mixed in the feeling that "quantization wins
everything" **[view]**:

1. **One axis.** The headline frontier is quality against bytes. On that
   axis, lower precision is cheaper than fewer neurons until precision runs
   out. This is not a weakness of your method; it is how the two kinds of
   redundancy behave.
2. **A cross-class comparison only.** SwiGLU is compared with quantization,
   never with another way of removing structure. A reader cannot tell
   whether your pruning is good pruning.
3. **Fidelity is the primary metric.** 8K KL measures closeness to the
   teacher. It is the least noisy metric and the right one for comparing
   derivatives of one teacher. It cannot compare a compressed model with an
   independently trained one, and it is not the same thing as quality.

Each has its own fix, below.

## 2.2 Proposed frame: three axes, three budgets, four classes

### Axes

| Axis | Metric | Who wins today |
| --- | --- | --- |
| Storage | Whole-model byte saving against quality | Quantization to 51.6%, stacking beyond |
| Compute | Linear-FLOP saving against quality | SwiGLU alone; quantization saves none |
| Both | The set of models not beaten on bytes *and* compute at once | Stacked models; nothing else saves both |

The third row is the fair summary and it is not in the snapshot yet. A model
such as B 50% INT4 saves 56% of bytes and 32% of linear FLOPs. No
quantization-only model can match the second number and no SwiGLU-only model
the first.

**A concrete figure to add (proposal).** A "dominance map": byte saving on
one axis, linear-FLOP saving on the other. Each model is a point, coloured by
quality. Mark the models that no other model beats on both savings at equal
or better quality. Quantization-only models lie on the left edge (zero FLOP
saving), SwiGLU-only models along a diagonal, and stacked models fill the
interior. One picture then shows what each family is for. Everything it needs
is already in the result files.

### Retraining budgets

Report every family at three budgets, because finding F5 shows that
conclusions change with budget:

| Budget | What it means | State of the data |
| --- | --- | --- |
| None (calibration only) | The setting most pruning papers and the annotation itself describe | Token-0 rows exist for SwiGLU fitted and slice-only. Missing: token-0 SwiGLU followed by post-training quantization (evaluation only) |
| 100M tokens | Cheap retraining, about 1-1.6 GPU-hours | Complete at eight targets |
| 1B tokens | Long retraining, about 9-15 GPU-hours | Complete at 20/50/80% |

This is also the natural home for your June idea of "deployment scenarios".
The supervisor cut scenarios defined by latency. Scenarios defined by *how
much retraining the user can afford* need no speed measurement and fall out
of data you already have: with no budget, fit locally; with a small budget,
fit and retrain; with a large budget, skip the fit.

### Method classes

| Class | Members you have | Members to add |
| --- | --- | --- |
| Precision | INT8, INT4, INT3; post-training and quantization-aware | Dense 2-bit ([file 1](01-direction-and-design.md), Section 1.6) |
| Structure, pruned | Slice-only with allocated widths (SwiGLU-8/9) | Slice-only with uniform widths |
| Structure, fitted ("replacement") | Fitted with allocated widths | Fitted with uniform widths |
| Combined | Orders A and B | (later) allocator-chosen bits |

## 2.3 The pruning baseline you left open

You raised a same-class baseline on 2026-11-01 and 2026-11-03, deferred it,
and it dropped out of every later handoff **[read: session mining, L6]**. My
recommendation:

- **Do not re-run the Grafting-style "every second MLP halved" model.** As
  the 2026-11-03 session concluded, it is the first version of your own
  method under old settings, not a pruning baseline.
- **Define the baseline as a 2 × 2 that you mostly have:**

| | Uniform widths | Allocated widths (knapsack) |
| --- | --- | --- |
| **Slice-only** (pruning) | *missing* | SwiGLU-8/9 slice-only arm |
| **Fitted** (replacement) | *missing* | SwiGLU-8/9 fitted arm |

  The top-left cell is plain structured pruning in the Minitron style:
  activation-based neuron importance, the same width everywhere, then
  distillation. The two missing cells are the same runs that weakness W1 in
  file 1 asks for. One small set of runs therefore gives you the pruning
  baseline, the allocator ablation, and the fitting ablation in one table.
  At 20/50/80% and 100M tokens that is six runs, roughly 7-9 GPU-hours
  **[calc: rough, from 1.1-1.6 hours per 100M-token run in the snapshot; not
  a PERUN resource plan]**.

- **Optionally, one published training-free method at the no-retraining
  budget only.** SmolLM2 uses the Llama architecture, so public
  implementations of methods such as FLAP or SliceGPT may run with small
  changes **[lit: recalled; untested]**. This is the only way to put a
  number from someone else's method on your model. Treat it as a bonus: an
  external codebase is an open-ended time cost.

## 2.4 Anchors: smaller models of the same family

**The idea.** SmolLM2 exists at 135M, 360M, and 1.7B parameters with one
tokenizer. A compressed 1.7B model is only worth having if it is better than
simply using the smaller sibling. This is the standard sanity check in
pruning-and-distillation work (Minitron compares with similarly sized
trained models **[lit: recalled]**) and it is absent from the snapshots.

**What the published numbers suggest [lit: model cards, checked for 360M;
1.7B checked except Winogrande].** These use a different harness (lighteval)
and four reported tasks, so they are *indicative only*:

| Model | Bytes at BF16 | Saving against 1.7B | HellaSwag | ARC (avg) | PIQA | Winogrande | Mean of four |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SmolLM2-1.7B | 3.19 GiB | 0% | 68.7 | 60.5 | 77.6 | 59.4 (recalled) | 66.6 |
| SmolLM2-360M | about 0.67 GiB | about 79% | 54.5 | 53.0 | 71.7 | 52.5 | 57.9 |

Your own five-task macro for the dense 1.7B model is 67.14%, close to the
card's 66.6, so the scales are at least compatible.

**What that would mean, if your evaluation confirms it [calc on the table
above and the 2026-11-01 snapshot]:**

| Model | Byte saving | Macro accuracy |
| --- | ---: | ---: |
| INT3 QAT 1B | 51.6% | 64.78% |
| B 20% INT3 1B | 54.2% | 61.19% |
| B 50% INT4 1B | 56.1% | 58.16% |
| B 80% INT4 1B | 61.3% | 49.85% |
| SmolLM2-360M, untouched (indicative) | about 79% | about 58% |

- The 360M sibling would match B 50% INT4 on accuracy while saving far more,
  and would beat everything beyond 56% outright.
- Draw a straight line from INT3 QAT 1B to the sibling. At 54.2% saving it
  passes near 64.1%; at 56.1%, near 63.7%. The stacked models sit 3 and 5.5
  points *below* that line. A line is not a model, but it shows how much
  quality the stacked region gives up per byte compared with the two models
  at its ends.

**Why this helps the thesis rather than hurting it [view].** It turns "how
far can we push compression?" into a question with an answer: *the useful
range for compressing this model on bytes ends where the next smaller
sibling takes over.* That is a finding, it is honest, and it gives the
extreme-compression experiments (80%, 90%) a clear interpretation instead of
an open-ended chase. It also reshapes what "pushing the frontier" should
mean: the open territory is between INT3 and the sibling, and the interesting
question is whether *anything* derived from the 1.7B model can live there.

**What to run.** Your core evaluation on SmolLM2-360M and 135M: WikiText
perplexity at 8K and the five tasks. Evaluation only; well under one
GPU-hour. Do **not** compare them on teacher KL (see 2.6).

**An optional stronger control.** Distil the 1.7B teacher into SmolLM2-360M
with your own recovery recipe and budget (100M or 1B tokens). That isolates
what your compressed models gain from *inheriting the teacher's weights*
from what they gain from *distillation alone*. It is the cleanest version of
the question the thesis is about. It needs a small change (all parameters
trainable, a different student) and is cheaper per token than your current
runs because the student is small.

## 2.5 Literature context, without pretending it is comparable

You cannot put other papers' numbers on your frontier: different models,
data, and budgets. What you can do is one clearly labelled context table:
"what structured methods report at similar removal", with the model and the
retraining budget in each row, and your own slice-only and fitted rows
beside it. Its purpose is narrow: to show that large quality loss at 20-50%
structural removal is normal for the class. Your own note of 2026-10-13
("even MoDeGPT reports doubling in PPL at 50% sparsity") is exactly this
argument **[read: L4]**; it belongs in the thesis with its locator.

Sources worth registering for this and for the related-work chapter are
listed in [file 5](05-documentation-and-agents.md), Section 5.4.

## 2.6 Fidelity and quality are different metrics

A small point with large consequences for how results read **[view]**:

- **8K teacher KL** asks "how close is this model to its teacher?". Use it
  to compare derivatives of one teacher, to select recipes, and to study
  retraining. Keep it primary there.
- **Perplexity and task accuracy** ask "how good is this model?". Use them
  whenever a model that is not derived from the teacher appears (siblings),
  and for any statement about usefulness.

The snapshots already report all three. What is missing is one sentence
saying which question each answers, and the consequence: a frontier in KL
flatters whatever stays near the teacher's weights.

## 2.7 Speed: what is true, and what can be claimed

### Two regimes

How fast a language model generates depends on which resource limits it.
This is standard computer-architecture reasoning, stated plainly **[view; no
equation, no citation needed for the reasoning itself]**:

| Regime | When it applies | What limits speed | Who benefits |
| --- | --- | --- | --- |
| Memory-bound | One request at a time, token-by-token generation | Reading the weights from memory for each token | Quantization: a quarter of the bytes to read |
| Compute-bound | Large batches, or processing a long prompt | Arithmetic in the matrix products | Width reduction: smaller matrices. Weight-only quantization still multiplies at full precision |

Both statements you have heard are therefore correct:

- **Your supervisor:** with optimized kernels, quantization would look much
  faster than it does in your plain-PyTorch measurement. True, in the
  memory-bound regime.
- **You:** the main advantage of structural compression is less computation.
  True, in the compute-bound regime. Your own Speed-1 shows the trace of it:
  time to first token on a 7,936-token prompt fell 16-25% for SwiGLU, while
  token-by-token generation did not move, because plain PyTorch generation
  is limited by per-token overhead, not by either resource.

### What you can already say without measuring anything

Byte saving bounds the gain in the memory-bound regime. Linear-FLOP saving
bounds the gain in the compute-bound regime. You report both. So the thesis
can state, with no kernel work at all:

> Each method's potential speed benefit is bounded by a quantity we report:
> bytes for memory-bound serving, linear FLOPs for compute-bound serving.
> Realizing either bound depends on the serving software and is outside this
> thesis.

That sentence is accurate, it credits structure fairly, and it does not open
the can of worms. I recommend it as the default position. It is consistent
with the 2026-10-25 and 2026-11-01 decision rows, which rejected a *computed
latency estimate*; this is a statement about bounds, not an estimate.

### A contained measurement, if you want one

The kernel problem exists because quantized weights need special code that
differs between libraries. One comparison avoids it entirely: **dense BF16
against SwiGLU BF16**. Both use the same ordinary matrix kernels, so a
serving engine treats them identically and the difference is the structure.

| Option | What is measured | Risk | My view |
| --- | --- | --- | --- |
| A | Nothing new; the bounds argument above | None | Default |
| B | Dense against SwiGLU 20/50/80%, BF16 only, in a serving engine, at low and high batch | Contained: one engine, no quantized kernels | A good bonus if a week is free |
| C | Quantized models in the same engine | Open-ended: kernel support, formats, fairness | Do not do it; cite published kernel results if a number is needed |

**Feasibility note for option B.** Your SwiGLU models have a different MLP
width in every layer. The standard model configuration has one
`intermediate_size` for all layers, so a stock engine cannot load them.
vLLM documents a supported way to register an out-of-tree model class
through a plugin, without changing its source **[lit: vLLM "Registering a
Model" documentation, checked]**. A subclass of its Llama model that reads
per-layer widths is a small amount of code. I have not tested this; treat it
as "documented as possible", not "known to work".

**What option B could show.** At high batch, throughput should approach the
linear-FLOP saving (up to about 2 times at 80% removal, since eligible MLPs
are 64.8% of linear FLOPs **[calc]**), reduced by attention's cost. That
would be a measured, kernel-neutral speed result for structural compression,
which the thesis currently lacks.

### Should speed become a research objective?

My recommendation **[view]**: keep bytes as the primary axis, as decided on
2026-07-18. Promote **compute** (linear FLOPs) from "supplement" to a second
reported axis, because it is exact, hardware-independent, and the only axis
where the thesis's own method wins. Keep measured speed as option B, a
bonus. That respects the supervisor's warning and still lets the thesis say
what structural compression is for.

## 2.8 A reporting checklist for the next snapshot

1. Add the dominance map (bytes against linear FLOPs).
2. Present each family at three retraining budgets, with the fit-versus-slice
   result as the worked example of why budget matters.
3. Add the 2 × 2 structure table (uniform or allocated, sliced or fitted)
   once the missing cells exist.
4. Add sibling models on the perplexity and accuracy plots.
5. State once what KL measures and what it does not.
6. State the two-regime bounds sentence in place of any speed frontier.
7. Keep the lean style you asked for: figure first, one small table, three
   bullets.
