# AGENTS.md

## Scope

- These instructions apply to `workflows/` and supplement the repository-root
  `AGENTS.md`.
- Before proposing or modifying a workflow intended to execute on TUKE Perun,
  read [`../docs/infrastructure/perun.md`](../docs/infrastructure/perun.md).
- Before modifying a Perun Slurm file, also read
  [`jobs/perun/README.md`](jobs/perun/README.md).

## Boundaries

- Keep reusable scientific operations in `src/mlp_replacement/`.
- Keep explicit migrated-workflow choices and budgets under `configs/`, split
  by `model/` and `block/` granularity.
- Keep concrete block and model experiment orchestration under `runs/`.
- Keep executor-specific resource requests and process launching under `jobs/`.
- Do not put scientific selection, fitting, recovery, or evaluation logic in a
  scheduler file.
- Do not claim parity with a notebook until inputs, operations, metrics, and
  artifact semantics have been compared explicitly.

## Perun requirements

- Never hardcode credentials, tokens, usernames, SSH material, account names,
  or QoS values.
- Submit-directory staging requires repository-relative paths and submission
  from the repository root.
- Keep Python environments and reusable caches outside hidden directories in
  the staged repository.
- Default to one experiment per process and one GPU unless the Python workflow
  explicitly implements distributed execution.
- Preserve unique structured artifacts and failure information for unattended
  runs.
- Treat resource values as measured configuration, not permanent cluster facts.
- Preserve unresolved infrastructure uncertainty and validate it with a smoke
  job or current official documentation before relying on it.
