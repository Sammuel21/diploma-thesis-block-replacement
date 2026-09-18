# AGENTS.md

## Scope

- These instructions apply to maintained code under `src/` and supplement the
  repository-root `AGENTS.md`.
- Before modifying maintained code, read
  [`../docs/agents/maintained-code-and-workflows.md`](../docs/agents/maintained-code-and-workflows.md).

## Package boundaries

- Keep reusable scientific operations in `src/mlp_replacement/` and keep
  notebook-specific plots, tables, and narration in `notebooks/`.
- Preserve the current responsibility split among `analysis/`, `compression/`,
  `evaluation/`, `operators/`, and the cross-cutting configuration, data,
  capture, model, and run-log modules.
- Extend an existing domain abstraction when it fits. Do not add a framework,
  registry, wrapper hierarchy, or duplicate configuration object for one use.

## Scientific and runtime behavior

- Treat data partitions, model and tokenizer revisions, dtypes, trainable
  scope, parameter accounting, losses, seeds, and artifact fields as part of
  the scientific contract. Do not change them incidentally.
- Keep model mutation explicit and exception-safe. Restore temporary
  replacements and release large resources on failure where the surrounding
  code already provides that boundary.
- Preserve compatibility with both direct execution on darthmachinus and
  scheduler-launched execution on Perun. Do not hardcode machine paths,
  usernames, credentials, account names, QoS values, or cache locations.

## Verification and model routing

- Match verification to risk: inspect imports and focused behavior for a local
  change; use an existing smoke path for integration behavior; reserve an
  expensive scientific run for changes that cannot be checked otherwise.
- Before delegating an implementation, classify it as low, medium, or high
  risk using the linked standard and recommend a model tier. If the user has
  not chosen a model, explicitly ask whether they want the cheaper eligible
  model or the stronger recommended model.
- Do not delegate automatically, and do not describe a scientific or
  infrastructure-critical change as mechanical merely because its diff is
  short.
