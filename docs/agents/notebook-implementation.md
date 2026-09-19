---
metadata_version: 1
title: Notebook Implementation Standard
type: architecture
category: agents/notebooks
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

# Notebook Implementation Standard

## Purpose and evidence base

This document defines how implementation agents change maintained research
notebooks. It was derived from the current notebooks under `notebooks/model/`
and `notebooks/block/`, excluding `block/activation-analysis.ipynb` at the
researcher's request because that notebook is not a style reference.

The strongest current references are:

- `model/swiglu/swiglu-2.ipynb` for model-level experiment sequencing, reporting, and
  artifact construction;
- `block/operator/operator-distillation.ipynb` for local-fitting studies and compact
  transition cells;
- `model/swiglu/swiglu-3.ipynb` for a load-only reporting notebook; and
- `model/interaction/block-interaction.ipynb` for model-level interaction analysis.

Across the eligible corpus, notebooks use direct sequential cells, a small
number of local functions, no notebook-defined framework classes, visible
configuration, DataFrame-based reporting, and compact JSON artifacts. Existing
Markdown heading depth varies across notebooks, so those historical headings
are not a formatting template for new agent-authored prose.

## Researcher-authored Markdown

Existing Markdown cells define the researcher's intended narrative and section
structure. Preserve their text, order, and heading levels unless the user
explicitly requests documentation changes.

When code needs a new transition cell, use a short plain-text label or one
plain sentence. Examples from maintained notebooks include `reporting`,
`configuration`, and `saving artifact`. Do not introduce a new Markdown heading
for such a transition, and do not expand it into an unsolicited explanation of
implementation details.

Preserve unrelated outputs, notebook metadata, and cell IDs. Edit the selected
cells rather than serializing the complete notebook through a formatter. Do not
clear all outputs unless the user asks for a clean notebook artifact.

## Cell and code organization

Keep the notebook readable from top to bottom:

1. imports and repository-root discovery;
2. visible paths, switches, budgets, and experiment constants;
3. model, data, or saved-artifact preparation;
4. the experiment or analysis in the researcher's section order;
5. nearby tables and plots; and
6. compact artifact or provenance handling.

Use the nearest relevant notebook as a concrete reference. Name the exact
reference cells in an implementation handoff. “Follow the repository style” is
too vague for a cheaper implementation model.

Keep code direct and locally understandable:

- use ordinary dictionaries, lists, DataFrames, and plotting calls;
- introduce a local helper only when it removes repeated logic or gives a
  meaningful operation a clear name;
- do not introduce notebook-local classes, registries, configuration
  frameworks, or speculative reusable layers;
- import stable scientific operations from `src/mlp_replacement/`; and
- leave notebook-specific table shaping, plotting, and interpretation in the
  notebook.

Important scientific choices must remain visible. Do not hide calibration
budgets, recovery budgets, selected layers, dtypes, evaluation batches, or
artifact paths behind an unnecessary wrapper.

## Reporting and artifacts

Construct tables from named records or DataFrames and keep metric names
consistent with the artifact that produced them. A reporting cell must not
silently change checkpoint selection, averaging, denominators, or missing-value
handling.

A load-only notebook reads completed artifacts and reconstructs tables and
plots without loading the model or dataset. Do not duplicate expensive workflow
logic in such a notebook.

Persist compact configuration, metrics, training histories needed for analysis,
runtime information, and provenance. Do not save raw activation tensors merely
to make notebook reporting convenient. Preserve existing schema keys when a
downstream notebook or document consumes them.

## Scope and verification

Implement only the requested experiment or report. Avoid opportunistic
refactoring, added dashboards, extra plots, generalized styling layers, or
cleanup of unrelated cells.

Do not add automated-test cells, assertion-only cells, mocks, fixtures, broad
linting, or formatter-driven notebook rewrites unless requested. Proportional
verification normally consists of:

- parsing the notebook JSON;
- inspecting the changed cells and their immediate dependencies;
- checking referenced imports and artifact fields; and
- confirming that existing Markdown, cell IDs, and unrelated outputs were
  preserved.

Run a cheap reporting cell or existing smoke path when it materially checks the
change. Do not load a model or start fitting solely to verify a presentation
change.

When reading notebooks for implementation, extract the relevant cell sources.
Avoid returning the complete notebook JSON or embedded outputs to the model.

## Implementation-agent suitability

Rate the proposed implementation before delegation:

| Rating | Typical notebook change | Suggested model |
| --- | --- | --- |
| Low | One or more reporting cells with exact artifact fields and reference cells | GPT-5.6 Luna Max |
| Medium | Multi-cell computation with settled methodology but some dependency tracing | GPT-5.6 Terra, high |
| High | New methodology, model mutation, fitting/recovery behavior, or ambiguous artifact semantics | GPT-5.6 Sol, high or above |

The rating measures how much unresolved judgment remains, not how many lines
will change. A small cell that chooses the wrong denominator or checkpoint can
still be high risk.

Delegation is a user choice. Supply the selected agent with the scoped
instructions, exact target cells, allowed Markdown changes, reference cells,
artifact fields, and completion checks. Review the resulting diff once; avoid
continuous high-cost supervision of routine implementation.

Astra is reserved for manual selection by the researcher as the primary
session model. An agent must not recommend, select, or spawn Astra for delegated
implementation or review.
