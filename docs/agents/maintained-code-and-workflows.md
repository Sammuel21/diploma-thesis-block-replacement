---
metadata_version: 1
title: Maintained Code and Workflow Implementation Standard
type: architecture
category: agents/code-and-workflows
status: active
created: 2026-09-18
modified: 2026-09-18
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---

# Maintained Code and Workflow Implementation Standard

## Purpose and evidence base

This document governs implementation under `src/` and `workflows/`. It was
derived from the maintained `mlp_replacement` package, the model workflow
runners, the Perun launcher boundary, and the current SwiGLU-3 workflow.

The package favors small domain functions, immutable dataclasses for stable
configuration and results, explicit model-mutation boundaries, short purpose
docstrings, grouped imports, and natural line wrapping near the existing
88-to-100-character convention. The workflow runners are intentionally more
procedural: they expose configuration and stage order while calling reusable
scientific operations from `src/`.

## Responsibility boundaries

Keep reusable scientific behavior in `src/mlp_replacement/`:

- `analysis/` owns diagnostic and screening computations;
- `compression/` owns allocation, selection, surgery, recovery, and reusable
  compression workflows;
- `evaluation/` owns operator, language-model, benchmark, and footprint
  evaluation;
- `operators/` owns replacement modules, initialization, and fitting;
- root modules own configuration, data, capture, model loading, and run logs.

Keep experiment-specific orchestration in `workflows/runs/`, explicit choices
and budgets in `workflows/configs/`, and executor-specific resource requests
and process launching in `workflows/jobs/`. Scheduler files must not implement
selection, fitting, recovery, or evaluation logic.

Within maintained experiments, group by scientific class. The homogeneous
SwiGLU progression belongs under `model/swiglu/`; model baselines and
interaction studies belong under `model/baseline/` and `model/interaction/`;
block studies use `block/baseline/`, `block/operator/`, and `block/analysis/`.
Reusable allocation and recovery behavior remains package code even when only
SwiGLU-5 currently consumes it.

Reuse an existing abstraction when it expresses the required domain operation.
Add a shared helper when multiple callers need it or when the operation deserves
a stable scientific name. Do not add a generic framework, registry, result
hierarchy, or duplicate configuration type for one current workflow.

## Scientific contract

Treat the following as observable scientific behavior:

- model and tokenizer identity and revisions;
- dataset source, revision, partition order, and sample construction;
- seeds, dtypes, batch sizes, optimizer settings, and token or update budgets;
- eligible and protected layers, trainable scope, and selection rules;
- loss definitions, temperature, averaging, checkpoint selection, and parameter
  accounting; and
- artifact schema, field names, units, and provenance.

Do not change these values or semantics as cleanup. Make a requested semantic
change explicit in configuration and documentation, and preserve compatibility
when a small extension is sufficient.

Model mutation must be visible and exception-safe. Temporary replacement
helpers restore the original module in `finally` blocks. Large activation,
teacher-cache, replacement, optimizer, and CUDA resources should have a clear
lifetime and be released between memory-heavy stages when practical.

## Portable execution

Maintained Python workflows must run from a repository checkout on
darthmachinus and from a staged repository on Perun. Therefore:

- resolve repository-relative paths independently of the caller's working
  directory;
- keep Python scientific logic independent of Slurm;
- do not hardcode host paths, usernames, credentials, accounts, QoS values,
  environment locations, or cache locations;
- keep executor resource values in job files or documented launch parameters;
  and
- treat resource requests as measured configuration, not permanent cluster
  facts.

One experiment per process remains the default. Process exit is an intentional
resource boundary for model state, activations, teacher logits, and CUDA
allocator state. Parallel or distributed execution requires an explicit design.

## Artifacts, interruption, and continuation

Every workflow must expose an explicit configuration and write a unique,
machine-readable science artifact. Refuse accidental overwrites. Write JSON and
checkpoint files atomically so interruption cannot leave a partially committed
file at the final path.

Initialize crash-aware status before expensive resource loading when practical.
Record the active stage, completed-stage summaries, relevant environment and
configuration, final artifact path, and failure information. Console and Slurm
logs support diagnosis; they do not replace structured state.

Add resumable continuation when a stage is long enough that replay is costly.
The SwiGLU-3 workflow is the current reference for this boundary:

- resume requires the original explicit output;
- configuration and prerequisite artifacts are fingerprinted;
- current, best, and scientific milestone states have distinct meanings;
- checkpoint metadata records tokens and optimizer updates;
- required RNG and optimizer state is restored; and
- persisted progress is checked against available checkpoint files.

Do not copy this checkpoint machinery into a short workflow. Its complexity is
justified by recovery runs whose loss would be material.

Do not add or run reduced-budget smoke workflows unless the researcher asks for
one in the current task. When explicitly requested, keep them separate from
scientific defaults and never interpret their metrics as thesis results.

## Verification

Verification follows the change's failure cost and scientific reach:

- low-risk configuration or reporting plumbing: parse configuration, inspect
  imports and paths, and exercise the focused transformation when cheap;
- reusable scientific behavior: add or run a focused test when it checks a
  stable or risky invariant rather than mirroring implementation;
- workflow integration: inspect stage order, artifact creation, and failure
  handling, then use the intended scientific run unless a smoke path was
  explicitly requested;
- checkpoint or resume behavior: check fingerprints, state restoration,
  atomic commits, and continuation from a small interrupted run; and
- full scientific equivalence: compare inputs, operations, metrics, and
  artifact semantics, then run the required controlled experiment.

Do not create a broad test framework, synthetic fixture hierarchy, or unrelated
lint cleanup for a local change. A successful smoke run establishes
executability, not notebook parity or scientific validity.

## Implementation-agent suitability

Before delegating a change, report one of these ratings:

| Rating | Typical change | Suggested model |
| --- | --- | --- |
| Low | Isolated CLI/config plumbing, serialization field, or direct adaptation with exact references | GPT-5.6 Luna Max or GPT-5.6 Terra medium |
| Medium | One maintained module or workflow stage with settled semantics and bounded interactions | GPT-5.6 Terra high, followed by focused review |
| High | Scientific method, artifact schema, checkpoint/resume, failure recovery, memory lifetime, or Perun/darthmachinus compatibility | GPT-5.6 Sol, high or above |

Short code can be high risk when failure would invalidate a long experiment or
silently change its meaning. If the user has not selected a model, explicitly
ask whether they want the cheaper eligible implementation model or the stronger
recommended model. Do not spawn an implementation agent automatically.

The handoff must identify affected files, required behavior, invariants,
reference implementation, allowed scope, and verification. A stronger model
should review high-risk changes against the diff and invariants without
repeating the complete repository analysis.

Astra is reserved for manual selection by the researcher as the primary
session model. An agent must not recommend, select, or spawn Astra for delegated
implementation or review.
