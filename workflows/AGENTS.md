# AGENTS.md

## Scope

- These instructions apply to `workflows/` and supplement the repository-root
  `AGENTS.md`.
- Before modifying a maintained workflow, read
  [`../docs/agents/maintained-code-and-workflows.md`](../docs/agents/maintained-code-and-workflows.md).
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

## Workflow reliability

- Keep scientific defaults in a versioned configuration. Smoke overrides must
  be explicit and must not silently replace those defaults.
- Use repository-relative paths for portable inputs and outputs. Resolve paths
  independently of the launcher's working directory.
- Give every scientific run a unique output. Refuse accidental overwrites and
  persist structured artifacts atomically.
- For long or expensive stages, record status before expensive work, preserve
  completed-stage summaries and failure information, and add resumable
  checkpoints when replay cost justifies their complexity.
- Validate resume inputs against configuration or artifact fingerprints and
  restore the state needed for scientifically equivalent continuation.
- Keep scheduler logs operational; they do not replace the structured science
  artifact or crash-aware run state.

## Verification and model routing

- Check configuration ownership and schema, artifact contracts, path handling,
  and failure behavior. Use a reduced smoke configuration for integration
  checks when available; do not substitute smoke results for scientific runs.
- Before delegating an implementation, classify it as low, medium, or high
  risk using the linked standard and recommend a model tier. If the user has
  not chosen a model, explicitly ask whether they want the cheaper eligible
  model or the stronger recommended model.
- Treat recovery, checkpoint/resume, artifact-schema, memory-lifetime, and
  Perun-launch changes as high risk by default.

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
