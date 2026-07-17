---
id: source-summary-mone-2026
title: MoNE - Replacing Redundant Experts with Lightweight Novices for Structured Pruning of MoE
summary: MoNE replaces low-usage, low-variance MoE experts with constant mean-output novices to reduce stored and active parameters.
type: source-summary
status: review
created: 2026-07-17
updated: 2026-07-17

authorship:
  created_by: collaborative
  contributors:
    - researcher
    - llm

epistemic:
  role: prior-work
  confidence: high
  verification:
    - source-checked

scope:
  topics:
    - mixture-of-experts
    - expert-redundancy
    - novice-replacement
    - structured-pruning
    - parameter-accounting
    - calibration-robustness
  granularities:
    - mlp-block
    - moe
    - model
    - cross-level
  pipeline_stages:
    - data
    - screening
    - selection
    - replacement
    - recovery
    - evaluation
    - analysis

sources:
  - source_id: src-mone-2026
    locator: "Entire paper; especially Sections 3-5 and Appendices A, C, E, and F"
    relation: defines

related:
  - "[[method-frequency-variance-expert-redundancy]]"
  - "[[method-mone-novice-expert-replacement]]"
  - "[[concept-moe-parameter-accounting]]"
supersedes: []
superseded_by: []
---

# MoNE - Replacing Redundant Experts with Lightweight Novices for Structured Pruning of MoE

## Bibliographic Identity

- Registered source: `src-mone-2026`
- Authors: Geng Zhang, Yuxuan Han, Yuxuan Lou, Yiqi Zhang, Wangbo Zhao, and
  Yang You
- Year: 2026
- Venue: arXiv preprint 2507.00390v2; reports acceptance at ICLR 2026
- Organization: National University of Singapore
- Source snapshot: arXiv version 2

## Research Question

The paper asks how to reduce the storage cost of sparse MoE language models
without relying only on expert access frequency or extensive recovery. It seeks
an expert-redundancy metric and a low-cost replacement that remain effective
across model architectures, calibration datasets, and calibration sizes.
[src-mone-2026, Sections 1 and 3.2]

## Method

MoNE operates layer by layer:

1. Run calibration samples through the MoE model and collect each expert's
   router access behavior and output activations.
2. Combine routing frequency/score with output variance. Experts that are both
   infrequently used and stable in output receive lower scores and are treated
   as more redundant.
3. Replace each selected expert with a lightweight `novice`: the empirical mean
   output vector of that expert on calibration tokens routed to it.

The novice is constant and does not perform an input-dependent MLP computation.
The mean is the closed-form minimizer of squared output discrepancy for a
constant replacement. Main pruning comparisons apply no weight updates;
continued pretraining is evaluated separately. [src-mone-2026, Sections
4.1-4.3 and 5.1]

## Evidence

The study evaluates five MoE base models: OLMoE 7B-A1B, Moonlight 16B-A3B,
DeepSeek-V2-Lite 16B-A3B, Qwen2-57B-A14B, and Qwen3-30B-A3B. It tests 25% and
50% expert-pruning ratios. Robustness experiments vary C4 versus Zyda2 and
100, 500, or 1,000 calibration samples. [src-mone-2026, Sections 5.1 and 5.3]

The primary evaluation averages zero-shot accuracy across nine tasks in
lm-evaluation-harness. Additional appendices report specialized Math and GSM8K
results, detailed robustness tables, ablations, and Qwen3-30B-A3B memory and
latency measurements. Optional recovery continues pretraining a 25%-compressed
OLMoE model for 2B tokens over 512 steps. [src-mone-2026, Sections 5.1 and 5.5;
Appendices A and F]

## Findings

The following are findings reported by the paper and have not been reproduced
in this thesis:

- With 100 Zyda2 calibration samples and 25% pruning, MoNE obtains the highest
  average zero-shot accuracy among the compared pruning methods for all five
  models. Qwen2-57B-A14B drops from 71.89 to 71.75 average accuracy.
  [src-mone-2026, Section 5.2; Table 1]
- Pruning ratios that keep average loss below one point vary by architecture:
  16% for OLMoE and Moonlight, 20% for DeepSeek-V2-Lite, 25% for
  Qwen2-57B-A14B, and 24% for Qwen3-30B-A3B. [src-mone-2026, Table 2]
- The combined frequency-variance score plus novice replacement performs best
  on average in the ablation, especially under aggressive pruning. Frequency
  alone is slightly better for MMLU in the reported aggregate.
  [src-mone-2026, Section 5.4; Figure 4]
- At 25% pruning, the method reports a better accuracy/variance frontier across
  model, data-source, and sample-size changes. At 50%, degradation and variance
  increase for every method. [src-mone-2026, Section 5.3; Figure 3]
- Continued pretraining with 2B tokens moves the MoNE-compressed OLMoE model to
  the closest average accuracy to the original among the compared compressed
  methods, but still does not fully restore the baseline. [src-mone-2026,
  Section 5.5; Table 3]
- For Qwen3-30B-A3B, Table 16 reports memory decreasing from 62.72 GB to 47.72
  GB and 32.73 GB at 25% and 50% pruning for batch size 1. Memory reduction is
  consistent across tested batch sizes. [src-mone-2026, Appendix F; Table 16]

## Limitations

The novice is an input-independent mean and is therefore appropriate only when
the selected expert's output variance is small on a representative calibration
distribution. Distribution shift or specialized tasks can invalidate this
assumption.

Appendix A demonstrates this limitation clearly: generic Zyda2 calibration
causes severe Math and GSM8K degradation after pruning, whereas task-specific
calibration recovers much of the specialized capability. Average accuracy over
general benchmarks should not be interpreted as universal preservation.
[src-mone-2026, Appendix A; Tables 4-8]

At 50% pruning, all approaches suffer larger quality loss. The fused score is
not uniformly best for every task, and MMLU sometimes favors frequency-only
selection. [src-mone-2026, Sections 5.3-5.4]

Active computation is routing-dependent. A nominal expert-pruning ratio does
not directly determine active parameter reduction because only tokens routed
to novices skip expert MLP computation. The paper calls this the novice hit
ratio. [src-mone-2026, Appendix F]

## Thesis Relevance

MoNE extends the thesis replacement perspective to MoE expert granularity. It
shows that replaceability can depend on two different properties: how often a
component is used and how much its output varies when used. Neither signal
alone fully characterizes redundancy.

The novice is a useful minimum-capacity baseline for replacement architecture
search. It is a constant output vector rather than a linear or nonlinear
operator. If a learned substitute cannot outperform this baseline for a
low-variance component, its additional parameters may not be justified.

The paper also motivates separate reporting of total parameters, model storage
or measured memory, and active parameters per token. For MoE models these axes
can move differently, and active parameters depend on runtime routing.

Calibration robustness must be treated as an evaluation axis. MoNE's general
results across C4 and Zyda2 do not eliminate task-specific failures, so the
thesis should avoid declaring a component redundant from one small generic
calibration sample alone.

## Claims Requiring Verification

- The novice baseline should be tested independently in the selected thesis
  MoE model before claiming transfer.
- Active-parameter accounting must match the target model's routing and shared
  expert implementation.
- The relationship between output variance and learned-substitute capacity
  remains a thesis hypothesis.
- Inference latency results are hardware- and framework-specific and remain
  outside the current thesis evaluation scope.

## Relationships

- [[method-frequency-variance-expert-redundancy]] records the screening metric.
- [[method-mone-novice-expert-replacement]] records the closed-form constant
  replacement.
- [[concept-moe-parameter-accounting]] separates total, stored, and active
  parameter measures.

## Sources

- `src-mone-2026` - entire paper, especially Sections 3-5 and Appendices A, C,
  E, and F
