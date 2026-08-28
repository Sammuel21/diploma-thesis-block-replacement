---
id: decision-working-experiment-code-standards
title: Working Experiment Code Standards
summary: Keeps initial experiment code direct, naturally formatted, lightly documented, and free of unsolicited testing or speculative abstractions.
type: decision
status: review
created: 2026-07-27
updated: 2026-08-28

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

**Project decision.** Initial thesis experiments should optimize for directness,
readability, and ease of modification. They are working research code, not a
production framework. Production hardening, broad abstraction, and automated
tests are added only when the researcher explicitly requests them.

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

The audit also found mechanically inconsistent wrapping: 19 Python lines under
`src/`, `pipelines/`, and `configs/` exceed 99 characters, including several
long signatures and calls. These are formatting issues to clean incrementally,
not a reason to introduce a formatter, lint framework, or broad refactor during
an experiment change.

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

- Keep the experimental story in notebook order: rationale, configuration,
  data capture, analysis or intervention, plots, and saved summaries.
- Put important constants and budget choices near the beginning.
- Use Markdown immediately before a non-obvious or expensive section to state
  what question it answers.
- Keep visualization and interpretation sequencing in the notebook.
- Move logic into `src/mlp_replacement/` only when it is reused, expresses a
  clear domain operation, or would otherwise obscure the experiment.
- Save compact metrics and plot data, not raw activation tensors.
- Do not add notebook assertion cells, periodic self-check evaluations, or
  other verification-only sections unless the researcher requests them.

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

Do not add automated tests, a test directory, synthetic fixtures, mocks,
assertion-based notebook verification, or test-only abstractions unless the
researcher explicitly asks for them.

When tests are requested later, add only focused coverage for the behavior then
considered stable or risky. Experiment execution, plotting, and inspection of
results are research activities; they should not be expanded into a testing
framework by default.

Basic runtime errors may still be used at genuine external boundaries when
continuing would silently produce an invalid experiment. Such checks should be
short and local rather than generalized into verification infrastructure.

## Change-Scope Rules

- Preserve unrelated researcher changes and historical artifacts.
- Keep a requested notebook change local unless a shared helper is genuinely
  necessary.
- Describe known limitations rather than solving every future concern in the
  first implementation.
- Label preliminary notebooks and outputs as working or exploratory so their
  simplicity is not confused with production readiness.

## Current Application

These rules govern [[experiment-initial-block-compression-study]]. The current
three notebooks are intentionally working initial experiments. Their lack of
automated tests is a chosen maturity boundary, not evidence that their
scientific results have been verified.

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

## Sources

No registered literature source is cited. This page records a researcher
instruction and a project-level inspection of the maintained repository.
