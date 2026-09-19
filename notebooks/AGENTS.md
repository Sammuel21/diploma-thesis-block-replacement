# AGENTS.md

## Scope

- These instructions apply to maintained work under `notebooks/` and supplement
  the repository-root `AGENTS.md`.
- Before modifying a notebook, read
  [`../docs/agents/notebook-implementation.md`](../docs/agents/notebook-implementation.md).
- Treat `notebooks/mvp/` as a frozen historical path. Do not modernize it while
  changing a maintained notebook.

## Researcher-authored structure

- Preserve existing Markdown cells, their order, and their heading structure
  unless the user explicitly asks for prose changes.
- A new transition cell may contain a short plain-text label or description,
  such as `reporting` or `saving artifact`. Do not add a Markdown heading for
  such a transition.
- Preserve unrelated cells, outputs, metadata, and cell IDs. Do not clear all
  outputs, normalize the complete notebook JSON, or rewrite unrelated cells.

## Implementation style

- Implement the minimum code needed for the stated experiment or report.
- Use the nearest relevant maintained notebook as the primary style and
  structure reference. For model-level reporting and allocation work, prefer
  `model/swiglu/swiglu-2.ipynb`; for local-fitting methodology, prefer
  `block/operator/operator-distillation.ipynb`. Do not use
  `block/analysis/activation-analysis.ipynb` as a style source.
- Route homogeneous SwiGLU reports to `model/swiglu/`, model baselines to
  `model/baseline/`, model interaction studies to `model/interaction/`, and
  block work to `block/{baseline,operator,analysis}/`.
- Keep the experiment readable in execution order: setup, configuration and
  data, experiment or artifact loading, reporting, then artifact handling.
- Prefer direct cells, plain dictionaries and DataFrames, and a small local
  helper only when it removes real repetition. Do not introduce notebook-local
  framework classes, registries, generic orchestration, or speculative
  abstractions.
- Reuse maintained operations from `src/mlp_replacement/`; keep notebook-only
  plotting, table construction, and interpretation in the notebook.
- Keep important budgets, switches, target layers, and artifact paths visible
  near the beginning of the applicable section.

## Verification boundary

- Do not add test cells, assertion-only cells, mocks, fixtures, broad linting,
  or formatter-driven rewrites unless the user explicitly requests them.
- Verify the notebook parses as JSON, inspect only the changed cells and their
  immediate dependencies, and check referenced artifact fields or imports.
- Do not execute expensive model loading, fitting, or recovery merely to test a
  reporting change. Do not suggest, implement, or execute a smoke path unless
  the user explicitly requests one in the current task.

## Model suitability

- Reporting-only work with exact source cells and artifact fields is a
  low-risk candidate for GPT-5.6 Luna Max.
- Multi-cell experiment logic or changes that require scientific judgment are
  a medium-risk candidate for GPT-5.6 Terra.
- New methodology, model mutation, recovery, or ambiguous artifact semantics
  require GPT-5.6 Sol for delegated work unless the user explicitly chooses
  another delegation model.
- Astra is reserved for manual selection by the user as the primary session
  model. Do not recommend or spawn it as a delegated agent.
- A model rating is a recommendation, not authority to spawn an agent. Follow
  the root delegation rule.
