---
id: concept-model-compression-evaluation-axes
title: Model Compression Evaluation Axes
summary: Distinguishes structural footprint, storage, memory, computation, compatibility, cost, and quality when evaluating a compressed language model.
type: concept
status: review
created: 2026-07-18
updated: 2026-07-18

authorship:
  created_by: collaborative
  contributors:
    - researcher
    - llm

epistemic:
  role: synthesis
  confidence: high
  verification:
    - source-checked

scope:
  topics:
    - model-compression
    - parameter-count
    - model-size
    - memory
    - compute-complexity
    - quality-preservation
  granularities:
    - model
    - moe
    - cross-level
  pipeline_stages:
    - evaluation
    - analysis

sources:
  - source_id: src-minitron-2024
    locator: "Section 4; Tables 2-9"
    relation: contextualizes
  - source_id: src-modegpt-2025
    locator: "Sections 4.1-4.5; Tables 2-9; Appendices B.9 and B.16"
    relation: supports
  - source_id: src-mone-2026
    locator: "Sections 4.1 and 5.1; Appendix F"
    relation: supports

related:
  - "[[concept-moe-parameter-accounting]]"
  - "[[decision-primary-compression-evaluation-scope]]"
  - "[[method-quality-preservation-evaluation]]"
supersedes: []
superseded_by: []
---

# Model Compression Evaluation Axes

## Overview

"Smaller model" is not one measurement. Compression can change the number of
stored parameters, serialized bytes, resident memory, active computation, and
model quality by different amounts. A valid report therefore names the axis it
measured and avoids treating one proxy as all the others.

For this thesis, the central result is a trade-off between structural footprint
and quality preservation. Runtime and systems measurements can supplement that
result, but they answer different questions.

## Structural Footprint

### Parameter count

Parameter count is the most architecture-level measure of model size. Report:

- total parameters in the dense baseline and compressed model;
- parameters removed and the resulting compression ratio;
- unique stored parameters when weight tying is present; and
- trainable parameters separately, because trainable count describes the
  recovery procedure rather than the deployed model's size.

Parameter count predicts neither exact checkpoint bytes nor runtime memory
without additional information about precision, quantization, storage format,
and execution.

### Active parameters in MoE models

Sparse MoE models require a second count. Total parameters describe everything
stored, while active parameters describe the parameters selected for a token's
forward path. Shared components, routing top-k, shared experts, and the counting
convention must be stated. Active parameters remain a computation proxy rather
than a direct latency measurement. [src-mone-2026, Sections 4.1 and 5.1;
Appendix F]

## Storage Footprint

Two storage quantities are useful and should not be conflated:

- **Theoretical weight bytes** are the sum of `numel * element_size` over
  unique stored parameter tensors.
- **Serialized checkpoint bytes** are the actual bytes of the deployable model
  artifact, including storage-format and metadata overhead but excluding
  optimizer state unless the report explicitly concerns a training checkpoint.

A checkpoint must record dtype or quantization, format, sharding, and whether
weights are tied or duplicated. Parameter count alone gives only an estimate.

## Runtime Memory

Runtime memory is not inherently smaller than storage size. Under the same
precision, weight-only checkpoint bytes and resident weight bytes may be close.
Runtime memory can be larger because it also includes activations, temporary
tensors, framework state, and, during autoregressive generation, a KV cache. It
can be smaller when weights are quantized, memory-mapped, offloaded, or only
partially resident, or when the compared disk artifact contains training state.

At minimum, distinguish:

- resident model or weight memory after loading;
- peak memory for a fixed evaluation workload; and
- peak training or recovery memory.

These measurements require a controlled process boundary and fixed dtype,
device, batch size, sequence length, and software environment.

## Computation and Runtime Behavior

### FLOPs or operation count

FLOPs estimate arithmetic work. They can differ per token because Transformer
work depends on layer dimensions, context length, attention implementation,
batch shape, and, for MoE models, routing. A generated token with an existing KV
cache also follows a different computation pattern from prompt prefill. FLOPs
therefore require an explicit workload definition.

### Latency and throughput

Latency is not a single "generation speed" value. Relevant measurements include
time to first token, inter-token latency, and end-to-end request latency.
Throughput measures completed work per unit time, commonly tokens per second,
under a stated load and batching policy. Low single-request latency and high
batched throughput are related but not equivalent optimization goals.

MoDeGPT reports that throughput changes with structured dimensions, but its
measurements are tied to a specific model, batch, sequence length, GPU, and
implementation. [src-modegpt-2025, Section 4.4; Appendix B.16]

### Energy and economic cost

Energy per token and monetary cost per token are derived system-level outcomes.
They depend on hardware utilization, runtime, pricing, and measurement scope.
They are useful deployment measures but are outside the primary thesis target.

## Hardware Compatibility

Hardware compatibility is a constraint category, not a duplicate of disk size.
A checkpoint may fit on storage but still fail to execute because of insufficient
RAM or VRAM, unsupported numerical formats, unavailable operators, kernel
requirements, or software-stack constraints. The thesis does not attempt a
general compatibility matrix.

## Quality Preservation

Quality is a family of measurements rather than one metric:

- language-model negative log-likelihood and perplexity measure predictive
  fit on a fixed corpus;
- downstream benchmark accuracy samples specific capabilities;
- robustness, calibration, safety, and instruction following are further
  dimensions that require separate protocols.

Perplexity is most defensible for paired comparisons that use the same model
family, tokenizer, corpus split, context handling, and preprocessing. Downstream
tasks should be reported individually; a suite average is a summary, not a
substitute for the task-level results. Minitron and MoDeGPT both combine
language-model evaluation with downstream tasks rather than relying on local
reconstruction error alone. [src-minitron-2024, Section 4; src-modegpt-2025,
Sections 4.2-4.3]

## Reporting Principle

**Synthesis.** Do not collapse all axes into an undocumented scalar. Compare
methods through paired dense-versus-compressed measurements and identify the
Pareto frontier:

- at the same footprint, prefer higher preserved quality;
- at the same quality level, prefer a smaller footprint; and
- report the calibration and recovery budget separately as the cost of
  obtaining the compressed model.

## Limitations and Open Issues

The axes above do not imply that the thesis will optimize all of them. In
particular, active parameters and FLOPs do not guarantee proportional latency,
and checkpoint bytes do not guarantee hardware compatibility.

The exact memory measurement procedure must be finalized with the maintained
implementation. Quality coverage also remains limited by the selected datasets
and benchmark protocol.

## Relationships

- [[concept-moe-parameter-accounting]] defines routing-aware parameter and
  memory accounting for sparse MoE models.
- [[decision-primary-compression-evaluation-scope]] selects which axes are
  primary, secondary, optional, or outside the thesis scope.
- [[method-quality-preservation-evaluation]] defines the proposed quality
  benchmark protocol.

## Sources

- `src-minitron-2024` - Section 4 and Tables 2-9
- `src-modegpt-2025` - Sections 4.1-4.5 and Appendices B.9 and B.16
- `src-mone-2026` - Sections 4.1 and 5.1 and Appendix F
