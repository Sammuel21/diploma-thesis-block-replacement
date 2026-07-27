---
id: decision-primary-compression-evaluation-scope
title: Primary Compression Evaluation Scope
summary: Defines the thesis objective as a footprint-quality trade-off while treating inference systems metrics as optional controlled observations.
type: decision
status: review
created: 2026-07-18
updated: 2026-07-27

authorship:
  created_by: collaborative
  contributors:
    - researcher
    - llm

epistemic:
  role: decision
  confidence: high
  verification:
    - supervisor-reviewed

scope:
  topics:
    - evaluation-scope
    - model-compression
    - parameter-count
    - model-size
    - memory
    - quality-preservation
  granularities:
    - model
    - moe
    - cross-level
  pipeline_stages:
    - evaluation
    - analysis

sources:
  - source_id: src-modegpt-2025
    locator: "Sections 4.2-4.5; Appendices B.9 and B.16"
    relation: contextualizes
  - source_id: src-mone-2026
    locator: "Section 5.1; Appendix F"
    relation: contextualizes
  - source_id: src-minitron-2024
    locator: "Section 4"
    relation: contextualizes

related:
  - "[[concept-model-compression-evaluation-axes]]"
  - "[[concept-moe-parameter-accounting]]"
  - "[[method-quality-preservation-evaluation]]"
  - "[[implementation-compute-environments]]"
  - "[[experiment-initial-block-compression-study]]"
supersedes: []
superseded_by: []
---

# Primary Compression Evaluation Scope

## Statement

**Project decision.** The thesis will primarily evaluate the trade-off between
model footprint and quality preservation. It will not optimize or generalize
inference latency, throughput, energy, or deployment scenarios.

The researcher reports that this scope was agreed with the supervisor. The
direction is marked `supervisor-reviewed`; the wording of this page remains at
`review` until the researcher validates it.

## Primary Measurements

Every final compressed model should report:

- total parameter count, parameters removed, and compression ratio;
- theoretical parameter bytes and actual serialized model checkpoint bytes;
- model-resident memory under a controlled loading protocol;
- total and active parameter counts for MoE models, with the routing and
  counting convention stated; and
- language-model loss/perplexity plus downstream benchmark accuracy under the
  fixed quality protocol.

The thesis question is not "which model has one best score?" It is which
replacement and recovery choices produce better footprint-quality trade-offs.

## Supporting Process Measurements

These values explain what it cost to obtain the result but are not properties
of the deployed model:

- calibration examples and tokens;
- replacement-training and recovery examples and tokens;
- trainable parameter count;
- optimizer steps, epochs, and learning-rate schedule;
- wall-clock or GPU time when available; and
- peak memory during compression or recovery.

They are necessary for fair comparisons between methods with different
training requirements.

## Secondary and Optional Measurements

Static FLOP or operation estimates may be reported as secondary structural
proxies when the workload and counting method are fixed.

Latency and throughput may be included as optional descriptive measurements
for one explicitly controlled hardware and software setup. Such values must not
be presented as general deployment performance or as a primary optimization
claim. No custom CUDA-kernel work is required by the thesis scope.

## Outside the Current Scope

- deployment-scenario recommendation;
- cross-hardware latency or throughput comparisons;
- energy and monetary cost optimization;
- general hardware-compatibility certification; and
- kernel-level inference optimization.

Excluding these topics does not mean they are scientifically unimportant. It
keeps the thesis centered on block replacement, integration, recovery, and the
resulting model-size versus quality trade-off.

## Decision Criteria

A compression approach is preferable when it is Pareto-superior under a common
protocol: it preserves more quality at the same footprint or reaches a smaller
footprint at the same quality level. Final claims require:

1. the exact dense baseline evaluated by the same pipeline;
2. the same model family, tokenizer, data split, and quality protocol;
3. explicit calibration and recovery budgets;
4. per-task results rather than only a suite average; and
5. no inference generalization beyond the measured system.

## Rationale

Minitron, MoDeGPT, and MoNE all separate some form of model-size reduction from
quality measurement. MoDeGPT and MoNE additionally show that memory and runtime
behavior depend on implementation and workload, not parameter count alone.
[src-minitron-2024, Section 4; src-modegpt-2025, Sections 4.2-4.5;
src-mone-2026, Section 5.1 and Appendix F]

The chosen boundary is also practical. A rigorous inference study would add
hardware, kernels, batching, cache behavior, serving runtime, and measurement
method as major experimental axes. Those systems questions could obscure the
thesis's central research contribution.

## Revisit Conditions

Revisit this decision only if the supervisor changes the thesis scope, a final
claim explicitly depends on runtime acceleration, or a compression method
changes parameter count without changing actual storage or resident memory as
expected.

## Limitations and Open Issues

The exact fresh-process protocol for model-resident and peak memory remains to
be designed with the maintained implementation. The benchmark suite is proposed
in [[method-quality-preservation-evaluation]] and awaits researcher approval.

No durable meeting record is currently registered as a wiki source. The
supervisor provenance in this page is based on the researcher's direct report
and should be linked to a meeting note if one is later created.

## Relationships

- [[concept-model-compression-evaluation-axes]] defines all candidate axes
  before this decision narrows the thesis scope.
- [[concept-moe-parameter-accounting]] defines total, stored, and active MoE
  parameter accounting.
- [[method-quality-preservation-evaluation]] specifies the proposed intrinsic
  and downstream quality protocol.
- [[implementation-compute-environments]] records the available project
  environments and the limits on interpreting hardware-specific observations.
- [[experiment-initial-block-compression-study]] applies the footprint-quality
  boundary to working block-level and model-level experiments.

## Sources

- `src-minitron-2024` - Section 4
- `src-modegpt-2025` - Sections 4.2-4.5 and Appendices B.9 and B.16
- `src-mone-2026` - Section 5.1 and Appendix F
