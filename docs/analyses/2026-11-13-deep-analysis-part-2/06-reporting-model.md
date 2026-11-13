---
title: Deep Analysis Part 2 - The Reporting Model and Measured Speed
status: draft
created: 2026-11-13
modified: 2026-11-13
---

# 6. The reporting model: what to track, how to draw three axes, and what speed can be measured

Part of the [deep analysis, part 2](README.md). It covers your topics 5 and
1.5. [Part 1, file 2](../2026-11-12-deep-analysis/02-reporting-and-speed.md)
already proposed three axes, three retraining budgets, sibling anchors, and a
"bounds" position on speed. That is not repeated. New here: a ledger of what
each method class really saves, how to show three axes, and what a serving
engine can measure today.

**About the sources.** Engine and benchmark facts were read by a research
sub-agent on 2026-11-13 from documentation and source files, through a tool
that summarises pages. They are marked **[agent]**; the full report with
links is in `private/agent-reports/serving-engines.md`. Nothing was run.
Check the cited lines before building on any of them.

## Short answer

- **Your point about unstructured pruning is right.** Without sparse kernels
  it saves bytes at best and no time, so it belongs beside quantization, not
  beside your method. The general rule: describe each class by what it saves
  in four quantities, and compare classes only on quantities both can change.
- **Three axes are enough: bytes, speed, quality.** The retraining budget is
  not a fourth axis; it is the condition under which the other three are
  read (one panel per budget).
- **Do not draw three-dimensional plots.** Compute which models are unbeaten
  on all three axes at once, then mark them on two ordinary frontiers.
- **Speed is two numbers, and they rank the families in opposite orders.**
  One request at a time favours fewer bits. A saturated server favours
  narrower layers. Published measurements show both.
- **In vLLM: dense against narrower BF16 models is low risk. Adding INT4 is
  moderate work. Adding INT3 depends on a kernel library that is three weeks
  old.** My recommendation is the first, with INT4 as an optional second
  step.

**Flags.** ROI: PROTECTIVE for the fairness of the comparison; a NEW AXIS if
measured. Scope: speed numbers are the least transferable in the thesis
(specific to one engine, one GPU, one version); the regime argument is
STRUCTURAL.

## 6.1 What each class saves

| Class | Bytes stored | Arithmetic per token | Bytes read per token | Needs special software? |
| --- | --- | --- | --- | --- |
| Weight-only quantization | Yes | No: products still run at full precision | Yes, if the kernel keeps weights packed | Yes, a kernel per format |
| Unstructured pruning | Only in a sparse format, which stores positions too | No, without sparse kernels | No, without sparse kernels | Yes, and rarely available on GPUs |
| Narrower MLPs (yours) | Yes | Yes | Yes | No: ordinary dense kernels |
| Removing or replacing experts (MoE) | Yes | Little, for rarely used experts | Little | No, unless counts differ per layer |
| Stacked | Yes, both | Yes, the structural part | Yes | Yes, for the quantized part |

- **This is the fairness rule in one table.** A storage frontier can include
  every class. A speed frontier can include a class only where its saving is
  realised by software you can name.
- **A caution for your own bytes.** TorchAO's generic low-bit tensor stores
  one byte per weight and expands to float inside each call **[agent]**. Your
  INT3 size is already labelled "calculated"; the same applies to any memory
  or speed claim for it.

## 6.2 What to track for every model

| Group | Quantity | Status |
| --- | --- | --- |
| Quality | 8K KL to the teacher; perplexity; task macro accuracy | Have |
| Quality | Top-1 agreement with the teacher | New, free with the KL pass ([file 4](04-isolated-design.md)) |
| Quality | KL on text outside the calibration domain | New, evaluation only ([file 4](04-isolated-design.md)) |
| Size | Whole-model byte saving, calculated and as stored | Have |
| Compute | Linear FLOPs per token | Have |
| Speed | Time per output token at one request; output tokens per second at saturation; time to first token on a long prompt; peak GPU memory | Speed-1 has eager-PyTorch values only |
| Condition | Retraining tokens (none, 100M, 1B) and GPU-hours to produce | Have |
| Condition | Software each model needs to realise its saving | New, one column |
| Trust | Noise band from a twin run | Have |

Use byte saving, not "sparsity" or removal percentage, on the size axis: 20%
of the eligible MLPs and 20% of the bytes are different things, and only
bytes are comparable between classes.

## 6.3 How to show three axes

1. **Two ordinary frontiers per budget:** quality against byte saving, and
   quality against speed (or against linear-FLOP saving until speed is
   measured).
2. **One computed set:** the models that no other model beats on bytes,
   speed, and quality at once. Draw those with filled markers on both
   frontiers and list them in one small table. A model can be off both 2D
   frontiers and still be in this set; those are the stacked models, and
   showing that is the point.
3. **One "what do I get" table:** at a few quality levels (for example KL
   under 0.05, 0.1, 0.2, 0.4), the largest byte saving, the largest speed
   gain, and the model that offers the most of both.
4. **Part 1's map** (byte saving against compute saving, colour for quality)
   is the third projection of the same data. Use it or item 2, not both.

A perspective 3D scatter hides points behind each other and cannot be read
off paper **[view]**. What I expect the computed set to show: each family
owns one edge (quantization the byte edge, narrower MLPs the compute edge)
and stacked models fill the space between. That expectation is not computed
here.

## 6.4 Measured speed: what is true and what is feasible

### The two regimes, with published numbers

| Source **[agent]** | One request at a time | Saturated or batched |
| --- | --- | --- |
| llama.cpp, an 8B model: 16-bit → 4-bit → 2-bit | 29 → 72 → 80 tokens/s | Prompt processing 923 → 822 → 784 tokens/s (slower) |
| vLLM study, 8B model, one A6000: BF16 → 4-bit weights | 24.5 s → 9.7 s per request | 1.08 times BF16 at maximum throughput |
| FLAP, 7B model, structured removal of 20% and 50% | - | 25.8 → 33.9 → 42.9 tokens/s |

- **Fewer bits help when reading weights is the limit, and the gain flattens
  below about 4 bits.** They do not help, or cost a little, when arithmetic
  is the limit.
- **Narrower layers help most exactly there.**
- So a single "speed" axis would favour whichever regime was measured. Report
  both, as two columns or two panels.

### Three tiers in vLLM

| Tier | What is compared | What it needs **[agent]** | Risk |
| --- | --- | --- | --- |
| A | Dense against SwiGLU 20/50/80%, all BF16 | The stock Llama class reads one MLP width. Either a small out-of-tree model class registered through the documented plugin hook, or the in-tree class for NVIDIA's searched models, which reads a width per block (whether a plain checkpoint can use it is not documented) | Low |
| B | A + INT4 (dense and stacked) | Export of your INT4 weights into the format vLLM's kernels read; no converter is documented. The fast 4-bit kernel supports a zero point and groups of 128, which is your scheme, and wants layer sizes in multiples of 64 and 128, which your padding already gives. Asymmetric INT8 is not in that kernel. Quantizing only the MLPs is supported | Moderate |
| C | B + INT3 | The two established kernels reject 3 bits. A new kernel library (announced 2026-10-21, self-reported numbers) lists 1 to 8 bits with a zero point; your exact scheme is not a documented case and its group sizes are not documented | High |

- **Do not use the official TorchAO route in vLLM for speed.** It calls
  TorchAO's own tensor inside every layer, with no vLLM kernel **[agent]**.
  That would repeat Speed-1's problem: measuring kernel maturity, not the
  method.
- **llama.cpp has real 3-bit kernels and per-layer widths, but it would
  re-quantize your weights with its own algorithm** and is a one-request
  engine **[agent]**. Its numbers would describe a different model.
- **No published paper the agent read compares quantization and width
  reduction on one measured speed axis.** Doing tier B carefully would be a
  small result in itself.

### A minimal protocol

From the engine's own benchmark documentation and the MLPerf rules
**[agent]**, combined by me:

1. All models in one job on one GPU, one engine version. (Your 2026-10-11 row
   already notes that array jobs on different GPUs are not comparable.)
2. Fixed synthetic prompts and a fixed output length, so every model emits
   the same number of tokens.
3. One request at a time, then a sweep of concurrency up to saturation.
4. Warm-up excluded, prompt caching reset between runs, three repeats.
5. Report time to first token, time per output token, output tokens per
   second, median and 99th percentile, and peak memory.
6. For tier B: before any timing, a bitwise check that the exported weights
   equal the deployed ones, in the spirit of your K4 gate.

### Options for the conflict with the decision log

The 2026-10-25 row forbids speed comparisons between methods. The supervisor
has now asked for one. Four ways to settle it:

| Option | Content | My view |
| --- | --- | --- |
| 1 | Keep the rows. State the bounds: bytes bound the gain at one request, linear FLOPs at saturation (Part 1, section 2.7) | The safe default; costs nothing |
| 2 | Tier A as a new speed experiment. Narrow the 2026-10-25 row to: measured comparisons are allowed between models that run on the same kernels in one engine | **Recommended.** A measured, kernel-neutral speed result for the thesis's own method |
| 3 | Option 2 plus tier B, with INT4 as the one quantized representative; INT8 and INT3 are placed by the bound and labelled | Only if the supervisor wants quantization on the measured axis. Starts with the export and its bitwise check |
| 4 | All families including INT3 | Not recommended: the comparison would rest on a kernel nobody has benchmarked independently |

Whichever is chosen, one sentence should go to the supervisor with it: at
saturation the published gain of 4-bit weights is about 1.1 times, and at one
request about 2.5 times, so "how fast is the quantized model" has no single
answer.
