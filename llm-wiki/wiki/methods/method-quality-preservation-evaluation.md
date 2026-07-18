---
id: method-quality-preservation-evaluation
title: Quality Preservation Evaluation
summary: Defines a tiered intrinsic and zero-shot benchmark protocol for paired evaluation of dense and compressed language models.
type: method
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
  confidence: medium
  verification:
    - source-checked

scope:
  topics:
    - quality-preservation
    - perplexity
    - zero-shot-evaluation
    - benchmark-suite
    - reproducibility
  granularities:
    - model
    - cross-level
  pipeline_stages:
    - evaluation
    - analysis

sources:
  - source_id: src-modegpt-2025
    locator: "Section 4.1; Tables 2-4; Appendix B.2"
    relation: motivates
  - source_id: src-minitron-2024
    locator: "Section 4; Tables 2-9"
    relation: contextualizes
  - source_id: src-mone-2026
    locator: "Section 5.1 and evaluation appendices"
    relation: contextualizes

related:
  - "[[concept-model-compression-evaluation-axes]]"
  - "[[decision-primary-compression-evaluation-scope]]"
  - "[[source-summary-modegpt-2025]]"
  - "[[source-summary-minitron-2024]]"
supersedes: []
superseded_by: []
---

# Quality Preservation Evaluation

## Overview

This method evaluates whether structural compression preserves language-model
behavior. It combines one intrinsic corpus metric with a tiered set of
zero-shot multiple-choice tasks. The tiers reduce repeated experiment cost
without allowing smoke-test subsets to become thesis evidence.

The protocol is a proposal at `review` status. The exact harness revision and
dataset revisions must be pinned when it is implemented.

## Evaluation Tiers

### Tier 0: smoke test

Use a small explicit example limit only to validate model loading, tokenization,
task integration, output logging, and memory behavior. Label every result as a
smoke test. Do not compare methods scientifically and do not report these
numbers as thesis findings.

### Tier 1: routine screening

Run on every candidate that reaches model-level evaluation:

- full fixed WikiText-2 evaluation for loss and perplexity;
- PIQA;
- ARC-Easy; and
- WinoGrande.

PIQA and WinoGrande are two-choice tasks, while ARC-Easy adds a different
science-question domain. This tier is intended to catch broad quality collapse
without paying for the complete confirmation suite after every candidate.

### Tier 2: confirmation

Run on shortlisted or final compressed models:

- every Tier 1 evaluation;
- ARC-Challenge; and
- HellaSwag.

The resulting five downstream tasks are `arc_easy`, `arc_challenge`, `piqa`,
`winogrande`, and `hellaswag`. This exactly matches the compact zero-shot suite
reported for the main LLaMA-2/3 experiments in MoDeGPT, which improves
comparability with registered prior work. [src-modegpt-2025, Section 4.3;
Table 3]

### Tier 3: optional breadth

Optional tasks can be added to final models only when they answer a stated
capability question and the compute budget allows it. They remain outside the
canonical five-task macro average so that the primary comparison does not change
between experiment rounds.

## Proposed Task Contract

| Evaluation | Capability sampled | Setting | Primary reported metric | Tier |
| --- | --- | --- | --- | --- |
| WikiText-2 | next-token predictive fit | fixed full evaluation split | mean NLL/loss and perplexity | 1 |
| PIQA | physical commonsense | zero-shot multiple choice | length-normalized accuracy | 1 |
| ARC-Easy | grade-school science questions | zero-shot multiple choice | length-normalized accuracy | 1 |
| WinoGrande | commonsense coreference | zero-shot multiple choice | accuracy | 1 |
| ARC-Challenge | harder grade-school science questions | zero-shot multiple choice | length-normalized accuracy | 2 |
| HellaSwag | plausible event continuation | zero-shot multiple choice | length-normalized accuracy | 2 |

## Optional Benchmark Catalogue

| Harness task or group | Status | Capability added | Main trade-off | Inclusion rule |
| --- | --- | --- | --- | --- |
| `mmlu` | optional final breadth | broad multi-domain knowledge | many subtasks and a substantially larger evaluation budget | Run only on final or strongly shortlisted models under a separately pinned conventional protocol; report separately from the zero-shot macro average. |
| `boolq` | optional diversity | passage-based yes/no reading comprehension | prompt-sensitive and narrower than a broad knowledge suite | Add when reading comprehension is a relevant missing capability and run it for the dense baseline and every compared final model. |
| `openbookqa` | optional diversity | multiple-choice science reasoning with external-knowledge framing | partially overlaps the ARC science domain | Add only when an additional science-reasoning check justifies the overlap. |
| `gsm8k` | conditional research task | multi-step arithmetic reasoning | generative decoding, prompting, and answer normalization add cost and confounding choices | Add only if mathematical reasoning or recovery of generative reasoning becomes an explicit research question. |

Minitron includes MMLU and GSM8K in its broader quality evaluation, while the
large-model MoDeGPT evaluation extends its compact suite with tasks including
BoolQ and OpenBookQA. These examples motivate the options; they do not require
the thesis to run every available benchmark. [src-minitron-2024, Section 4;
Tables 2-9; src-modegpt-2025, Section 4.3; Table 4]

An optional task must be selected before inspecting the compared models' task
results. Record its harness task identifier, prompt and shot setting, split,
metric, dataset revision, and inclusion rationale. Never add only the task on
which a preferred compression method happens to perform well.

## Named Evaluation Profiles

- `smoke`: explicit example limits for integration checks; not thesis evidence.
- `routine`: WikiText-2, PIQA, ARC-Easy, and WinoGrande.
- `confirmation`: `routine` plus ARC-Challenge and HellaSwag.
- `extended-knowledge`: `confirmation` plus a pinned MMLU protocol.
- `conditional-math`: `confirmation` plus a pinned GSM8K protocol.

BoolQ and OpenBookQA are individual declared additions rather than automatic
members of every extended profile. Profile names describe methodology; the
maintained implementation may map them to version-pinned harness task IDs.

When the pinned evaluation harness emits both raw and normalized accuracy,
preserve both in the result artifact and designate the table's metric as the
primary comparison. Do not silently change prompts, splits, or normalization
between models.

## Paired Evaluation Procedure

1. Evaluate the exact dense baseline locally with the same code path used for
   every compressed model. Do not substitute leaderboard or paper values.
2. Keep benchmark examples out of calibration, replacement training, recovery,
   candidate selection, and hyperparameter tuning.
3. Use zero-shot evaluation for the canonical suite unless a separate
   few-shot research question is declared.
4. Use the complete harness-selected evaluation split for reported results.
   A limited split remains a smoke test even if it is reproducible.
5. Pin the model revision, tokenizer revision, evaluation-harness version and
   commit, task configuration, dataset revision, number of shots, dtype,
   maximum context, batch behavior, and random seed.
6. Preserve per-sample outputs or correctness where licensing permits. This
   enables paired uncertainty estimates between the dense and compressed
   models.

The same tokenizer and preprocessing are essential for interpreting perplexity
changes. Perplexity values from models with different tokenizers should not be
treated as directly comparable without further qualification.

## Reported Quality Values

For language modeling, report dense and compressed loss, loss delta,
perplexity, and perplexity ratio. For each downstream task, report:

- dense accuracy;
- compressed accuracy;
- absolute change in percentage points; and
- optional retained accuracy, `compressed / dense`, expressed as a percentage.

Also report the unweighted macro average across the fixed five-task suite, but
keep every task-level value visible. If sample-level results are retained, use
a paired bootstrap confidence interval for the dense-compressed difference or
another declared paired procedure. A small average gain must not hide a severe
regression on one task.

## Evidence and Rationale

MoDeGPT combines WikiText-2 perplexity with the same five zero-shot tasks and
shows that perplexity and suite accuracy can rank compression choices
differently. [src-modegpt-2025, Sections 4.2-4.5; Tables 2, 3, and 9]

Minitron likewise reports validation loss and WikiText-2 perplexity together
with downstream evaluations. Its broader suite demonstrates that one intrinsic
metric is not a complete quality account, while its varying few-shot settings
also show why the prompt protocol must be recorded. [src-minitron-2024,
Section 4; Tables 2-9]

## Limitations and Open Issues

The five tasks sample only a subset of model capability. They do not establish
instruction following, safety, factuality, long-context behavior, code
generation, or mathematical reasoning.

Static benchmarks can contain contamination or model-specific prompt effects.
Paired comparison against the exact dense baseline reduces some confounding for
compression damage, but it does not validate absolute capability claims.

Before this page can become `verified`, register and check the original PIQA,
ARC, WinoGrande, and HellaSwag benchmark publications and a pinned
LM Evaluation Harness software revision. The currently registered compression
papers support the suite choice but are not the canonical definitions of all
tasks or metrics.

## Relationships

- [[concept-model-compression-evaluation-axes]] places this method within the
  broader footprint-and-quality evaluation taxonomy.
- [[decision-primary-compression-evaluation-scope]] makes quality preservation
  a primary thesis axis and limits systems benchmarking.
- [[source-summary-modegpt-2025]] supplies the closest literature-aligned
  five-task suite.
- [[source-summary-minitron-2024]] supplies broader evidence for combining
  intrinsic and downstream evaluation.

## Sources

- `src-modegpt-2025` - Section 4.1, Tables 2-4, and Appendix B.2
- `src-minitron-2024` - Section 4 and Tables 2-9
- `src-mone-2026` - Section 5.1 and evaluation appendices
