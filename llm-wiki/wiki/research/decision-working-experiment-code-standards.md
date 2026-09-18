---
id: decision-working-experiment-code-standards
title: Working Experiment Code Standards
summary: Separates direct notebook implementation from reliable maintained code and workflow implementation, with scoped agent routing for each.
type: decision
status: review
created: 2026-07-27
updated: 2026-09-18

authorship:
  created_by: collaborative
  contributors:
    - researcher
    - llm

epistemic:
  role: decision
  confidence: high
  verification:
    - unverified

scope:
  topics:
    - coding-standards
    - exploratory-experiments
    - implementation-scope
    - notebook-workflow
    - workflow-reliability
    - agent-routing
  granularities:
    - cross-level
  pipeline_stages:
    - analysis
    - infrastructure

sources: []
related:
  - "[[experiment-initial-block-compression-study]]"
  - "[[implementation-compute-environments]]"
  - "[[implementation-maintained-mlp-replacement-package]]"
supersedes: []
superseded_by: []
---

# Working Experiment Code Standards

## Statement

**Project decision.** Maintained notebooks and maintained execution code have
different implementation boundaries. Notebooks optimize for directness,
readability, and researcher-controlled narrative. Reusable source code and
long-running workflows add reliability in proportion to their scientific and
compute risk, including explicit configuration, portable execution, structured
artifacts, failure records, and resumable continuation when replay is costly.

Operational instructions are maintained in
[`notebooks/AGENTS.md`](../../../notebooks/AGENTS.md),
[`src/AGENTS.md`](../../../src/AGENTS.md), and
[`workflows/AGENTS.md`](../../../workflows/AGENTS.md). The detailed human-facing
standards are the
[notebook implementation standard](../../../docs/agents/notebook-implementation.md)
and the
[maintained code and workflow standard](../../../docs/agents/maintained-code-and-workflows.md).
This wiki page records the rationale and decision rather than duplicating every
operational rule.

## Motivation and Prior Overengineering

**Researcher assessment.** An initial GPT Sol X-high implementation pass
overengineered the block-compression notebooks. It introduced automated tests,
assertion-based verification, and additional record, replacement-bank, and
configuration abstractions before the experimental interfaces had stabilized.
Those additions increased the amount of code without answering the immediate
research questions and were removed.

This is not a claim that testing or abstraction is generally undesirable. It
sets their timing: first make the experiment scientifically understandable and
useful, then add hardening on demand around behavior that has become stable.

## Observed Repository Style

**Repository observation.** The maintained notebooks under `notebooks/model/`
and `notebooks/block/`, excluding `activation-analysis.ipynb` as directed by the
researcher, use sequential experiment cells, visible configuration, a small
number of local helpers, no notebook-defined framework classes, DataFrame-based
reporting, and compact JSON artifacts. `swiglu-2.ipynb` and
`operator-distillation.ipynb` are the primary current style references.

**Researcher decision.** Existing notebook Markdown is a researcher-authored
boundary. Implementation agents preserve it unless prose changes are requested.
An agent may add a short plain-text transition cell when necessary, without a
new Markdown heading.

The maintained `src/mlp_replacement/` package provides the current style
reference:

- four-space Python indentation and no tab-based layout;
- small functions organized by responsibility;
- immutable dataclasses for stable configuration and result records;
- short descriptive names with explicit domain terms such as `layer_index`,
  `validation_mse`, and `removed_parameters`;
- brief one-sentence docstrings on public functions and classes;
- explicit model mutation isolated in context managers or workflow functions;
  and
- standard-library, third-party, and local imports separated into groups.

The maintained `workflows/` layer is intentionally more procedural. It keeps
experiment-specific stage order in Python runners, explicit scientific choices
under `workflows/configs/`, and executor-only launch behavior under
`workflows/jobs/`. SwiGLU-3 is the current reference for a long workflow whose
cost justifies atomic artifacts, fingerprints, failure state, checkpoints, and
resume behavior.

## Formatting Rules

1. Use four spaces per indentation level. Do not align blocks with manual
   columns or use tabs.
2. Keep a short definition or call on one line when it remains naturally
   readable.
3. For definitions or calls with five or fewer parameters or arguments, prefer
   a compact row-like layout. If the expression must wrap, group the arguments
   naturally on one or a small number of continuation lines instead of placing
   every argument on its own line.
4. One-parameter-or-argument-per-line formatting is appropriate by default
   only for longer definitions or calls with more than five items, or when the
   individual expressions are themselves too complex to scan as a group.
5. Format dictionaries, lists, and configuration blocks according to their
   structure. Keep short related values together and expand genuinely long or
   nested content; do not apply a mechanical one-item-per-line rule.
6. Prefer readable lines around the existing 88-to-100-character convention,
   but do not contort simple expressions solely to satisfy a number.

Preferred short-call form:

```python
result = evaluate_candidate(model, replacement, validation_loader, max_batches=24)
```

If the surrounding names make that line too long, retain the row-like grouping:

```python
result = evaluate_candidate(
    model, replacement, validation_loader, max_batches=24
)
```

Reserve the expanded form for calls that are genuinely long:

```python
result = evaluate_candidate(
    model,
    replacement,
    calibration_loader,
    validation_loader,
    metrics,
    output_path,
    max_batches=24,
)
```

The threshold is a readability default rather than a formatter rule. Prefer
the layout a researcher can scan and edit naturally, without arbitrary manual
alignment.

## Docstrings and Comments

- Give a public module, class, or function a brief description docstring when
  its purpose is not already obvious from local notebook context.
- Prefer one sentence. Use a longer docstring only when mathematical meaning,
  side effects, data conventions, or restoration behavior need explanation.
- Do not add docstrings to every local closure, plotting fragment, or obvious
  notebook helper merely for coverage.
- Comments should explain a scientific convention or a non-obvious reason, not
  narrate the next line of code.

## Notebook Rules

- Preserve existing Markdown cells, their order, and their heading structure
  unless the researcher requests prose changes.
- Keep the experimental story in notebook order: setup, configuration and data,
  analysis or intervention, nearby reports, and saved summaries.
- Put important constants and budget choices near the beginning.
- Use only a short plain-text transition cell when implementation requires a new
  separator; do not add an unsolicited heading or long explanation.
- Keep visualization and interpretation sequencing in the notebook.
- Move logic into `src/mlp_replacement/` only when it is reused, expresses a
  clear domain operation, or would otherwise obscure the experiment.
- Save compact metrics and plot data, not raw activation tensors.
- Do not add notebook assertion cells, periodic self-check evaluations, or
  other verification-only sections unless the researcher requests them.
- Preserve unrelated outputs, metadata, and cell IDs; do not reserialize the
  complete notebook merely to change selected cells.

## Abstraction Rules

- Implement the current experiment, not hypothetical future experiments.
- Do not add a wrapper class, registry, result hierarchy, duplicate
  configuration type, or generic orchestration layer for a single current use.
- Use plain dictionaries, tuples, or small dataclasses when they make the
  immediate data flow clearer.
- Extract a shared helper when at least two callers need the same domain logic
  or when a single scientifically meaningful operation deserves a stable name.
- Preserve backward compatibility of the maintained package when a small
  extension is sufficient.
- Do not refactor historical MVP paths while building maintained experiments.

## Testing Policy

Do not add automated tests, synthetic fixtures, mocks, assertion-based notebook
verification, or test-only abstractions to exploratory notebooks unless the
researcher explicitly asks for them.

For maintained source and workflows, verification is proportional to risk. Add
or run focused coverage when it protects a stable scientific invariant,
artifact contract, failure boundary, or continuation behavior. Use a reduced
smoke path for integration checks when one exists. Do not create a broad test
framework or unrelated lint cleanup for a local change.

Basic runtime errors may still be used at genuine external boundaries when
continuing would silently produce an invalid experiment. Such checks should be
short and local rather than generalized into verification infrastructure.

## Workflow Reliability

**Project decision.** A maintained workflow exposes explicit versioned
configuration, portable repository-relative paths, unique outputs, structured
provenance, and atomic persistence. Long or expensive stages record progress
and failures before their cost becomes material. Resume support is added only
when replay cost justifies the extra state contract, and it validates that the
configuration and prerequisite artifacts match the interrupted run.

Python workflow code must remain usable both in a direct checkout on the shared
RTX 4090 machine and in a staged Perun checkout. Machine identities, user paths,
credentials, accounts, QoS values, and environment locations are not embedded
in scientific code. Slurm files request resources and launch a process; they do
not own scientific logic.

## Implementation-Agent Routing

**Project decision.** Model suitability is based on unresolved judgment and
failure cost rather than diff size. Bounded reporting work with exact reference
cells defaults to Luna Max; settled multi-file work with moderate dependency
tracing is eligible for Terra; scientific-method, artifact-schema,
checkpoint/resume, memory-lifetime, and cross-environment changes default to Sol
when delegated. Astra is reserved for manual selection by the researcher as the
primary session model and is not a delegation target.

Before delegating maintained `src/` or `workflows/` implementation, the agent
reports a low, medium, or high risk rating and recommends a model tier. If the
researcher has not already selected a model, the agent asks whether to use the
cheaper eligible implementation model or the stronger recommendation. This
rating does not authorize automatic delegation.

## Change-Scope Rules

- Preserve unrelated researcher changes and historical artifacts.
- Keep a requested notebook change local unless a shared helper is genuinely
  necessary.
- Describe known limitations rather than solving every future concern in the
  first implementation.
- Label preliminary notebooks and outputs as working or exploratory so their
  simplicity is not confused with production readiness.

## Current Application

These rules govern maintained notebooks, the reusable source package, and the
workflow layer, including [[experiment-initial-block-compression-study]]. Their
different verification requirements reflect different failure costs and do not
establish that scientific results are correct.

[[implementation-maintained-mlp-replacement-package]] records the package-level
responsibility boundaries used when notebook logic is promoted into reusable
source modules.

## Revisit Conditions

Revisit this decision when the researcher asks for production hardening, a
shared interface stabilizes across several experiments, failures become costly
or difficult to detect, or code is promoted from exploratory notebooks into a
maintained experiment pipeline.

## Relationships

- [[experiment-initial-block-compression-study]] is the first notebook suite
  governed by this decision.
- [[implementation-compute-environments]] records where the working notebooks
  are intended to run but does not change their implementation maturity.
- [[implementation-maintained-mlp-replacement-package]] applies this decision
  to the maintained source-code organization.
- [[implementation-compute-environments]] provides the execution environments
  whose portability requirements apply to workflow implementation.

## Sources

No registered literature source is cited. This page records a researcher
instruction and a project-level inspection of the maintained repository.
