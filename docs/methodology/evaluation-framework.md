# Model Compression Evaluation Framework

Status: review draft. The primary scope records the direction discussed with
the supervisor. The exact benchmark suite remains subject to researcher
approval before implementation.

## Objective

The thesis should answer a concrete question: **how much model footprint was
removed, what quality was preserved, and what calibration or recovery budget
was required to obtain that result?**

The primary objective is therefore a footprint-quality trade-off. Inference
latency, throughput, energy, and deployment scenarios are not primary research
targets. They may be reported only as controlled descriptive measurements when
useful.

## Evaluation Axes

| Axis | What it answers | Thesis role |
| --- | --- | --- |
| Total parameters | How much model structure remains? | Primary |
| Compression ratio | What fraction of parameters was removed? | Primary |
| Theoretical weight bytes | How many bytes do unique parameter tensors require at the stated precision? | Primary |
| Serialized model bytes | How large is the actual deployable checkpoint? | Primary |
| Resident model memory | How much RAM or VRAM does the loaded model occupy under a controlled protocol? | Primary |
| Active parameters | How many parameters participate in a token path in an MoE model? | Primary for MoE |
| Quality | How much predictive and downstream behavior was preserved? | Primary |
| Calibration and recovery cost | What data and compute were required to produce the model? | Supporting |
| FLOPs or operation count | How much arithmetic does a declared workload imply? | Secondary proxy |
| Latency and throughput | How does one fixed system execute the model? | Optional descriptive |
| Energy, cost, and hardware coverage | What are the broader deployment consequences? | Outside current scope |

Parameter count is the most general structural measure, but it does not by
itself determine disk size or runtime memory. Disk bytes also depend on dtype,
quantization, file format, sharding, metadata, and weight tying. Runtime memory
can exceed a weight-only checkpoint because it includes activations, temporary
tensors, framework state, and KV cache. It can also be lower under offloading,
memory mapping, or a different runtime representation.

For MoE models, always report total and active parameters separately. Active
parameters are routing-dependent and do not directly guarantee latency.

## Quality Protocol

Quality requires both intrinsic language-model evaluation and downstream task
evaluation. Local operator MSE and model-level distillation loss are training or
diagnostic signals; neither replaces evaluation on held-out data.

### Routine screening

Run on every model-level candidate:

- full fixed WikiText-2 loss and perplexity evaluation;
- zero-shot PIQA;
- zero-shot ARC-Easy; and
- zero-shot WinoGrande.

PIQA is a good fast anchor because it is a two-choice physical-commonsense task.
ARC-Easy adds science reasoning, and WinoGrande adds commonsense coreference.

### Confirmation suite

Run on shortlisted and final models:

- all routine evaluations;
- zero-shot ARC-Challenge; and
- zero-shot HellaSwag.

The five downstream tasks are therefore PIQA, ARC-Easy, ARC-Challenge,
WinoGrande, and HellaSwag. This is the compact main suite used by MoDeGPT for
its LLaMA-2/3 compression results, so it gives the thesis a literature-aligned
comparison point.

### Optional final breadth

Optional tasks remain outside the canonical five-task average. This preserves a
stable primary comparison even when additional final analyses are added.

| Option | What it adds | When to use it | Why it is not routine |
| --- | --- | --- | --- |
| MMLU | broad multi-domain knowledge | final or strongly shortlisted models under a separately pinned conventional protocol | many subtasks and a substantially larger evaluation budget |
| BoolQ | passage-based yes/no reading comprehension | when the core suite needs a reading-comprehension check | narrower and prompt-sensitive |
| OpenBookQA | another multiple-choice science-reasoning view | when additional science robustness justifies overlap with ARC | partially redundant with ARC-Easy and ARC-Challenge |
| GSM8K | multi-step arithmetic reasoning | only when mathematical or generative reasoning becomes an explicit research question | decoding, prompting, and answer normalization add cost and confounding choices |

Do not add tasks merely to increase the benchmark count. A smaller, fixed,
fully specified suite is stronger than a larger ad hoc collection.

### Named profiles

- `smoke`: limited examples for integration testing; never thesis evidence.
- `routine`: WikiText-2, PIQA, ARC-Easy, and WinoGrande.
- `confirmation`: `routine` plus ARC-Challenge and HellaSwag.
- `extended-knowledge`: `confirmation` plus a pinned MMLU protocol.
- `conditional-math`: `confirmation` plus a pinned GSM8K protocol.

BoolQ and OpenBookQA are declared individual additions. Any optional benchmark
must be chosen before inspecting comparative task results, run on the exact
dense baseline and every model in the comparison, and reported separately from
the canonical five-task macro average.

## Comparison Rules

1. Evaluate the exact dense baseline and compressed model with the same code,
   tokenizer, prompts, task splits, harness revision, dtype, and context rules.
2. Keep evaluation data separate from calibration, replacement training,
   recovery, hyperparameter selection, and early stopping.
3. Use full benchmark splits for reportable results. Limited-example runs are
   smoke tests only.
4. Report every task separately, then an unweighted macro average as a summary.
5. Report dense value, compressed value, absolute delta, and optional retained
   percentage for every primary metric.
6. Preserve sample-level correctness where possible so dense-compressed
   uncertainty can be estimated with a paired procedure.
7. Treat perplexity from different tokenizers as non-comparable unless the
   comparison is explicitly qualified.

For multiple-choice tasks, preserve all metrics emitted by the pinned harness.
The proposed primary values are length-normalized accuracy for PIQA, ARC-Easy,
ARC-Challenge, and HellaSwag, and accuracy for WinoGrande.

## Experiment Report Contract

Every reportable run should identify:

- base model, revision, tokenizer, dtype, and numerical format;
- replacement architecture, replaced locations, strategy, and random seed;
- total parameters, removed parameters, compression ratio, weight bytes,
  checkpoint bytes, and controlled resident memory;
- calibration and recovery datasets, examples, tokens, sequence length, and
  split policy;
- trainable parameter count, optimizer, schedule, steps, epochs, and recovery
  objective;
- WikiText-2 loss and perplexity;
- all applicable benchmark accuracies and dense-compressed deltas;
- software versions and evaluation task configuration; and
- wall-clock, GPU time, and peak training memory when available.

The final analysis should use Pareto plots or tables rather than one arbitrary
combined score. A method is stronger when it preserves more quality at the same
footprint or achieves a smaller footprint at a comparable quality level.

## Reproducibility Decisions Still Needed

Before implementation, register the canonical PIQA, ARC, WinoGrande, and
HellaSwag publications and the selected LM Evaluation Harness release or commit
in the wiki source registry. Register the canonical source for each optional
benchmark before activating its profile. Then pin task YAML and dataset
revisions in the evaluation configuration.

The implementation phase must also define a fresh-process memory protocol.
Notebook allocator state is not a reliable final measurement of resident model
memory.

## Knowledge-Base Traceability

- [Evaluation axes](../../llm-wiki/wiki/concepts/concept-model-compression-evaluation-axes.md)
- [Primary scope decision](../../llm-wiki/wiki/research/decision-primary-compression-evaluation-scope.md)
- [Quality preservation method](../../llm-wiki/wiki/methods/method-quality-preservation-evaluation.md)
- [MoE parameter accounting](../../llm-wiki/wiki/concepts/concept-moe-parameter-accounting.md)

Registered evidence used for this draft:

- `src-minitron-2024`, especially Section 4
- `src-modegpt-2025`, especially Sections 4.1-4.5 and Appendix B
- `src-mone-2026`, especially Section 5.1 and Appendix F
