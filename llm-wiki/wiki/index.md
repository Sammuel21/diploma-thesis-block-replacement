# Wiki Index

This is the content-oriented entry point for maintained thesis knowledge.
Read it before searching individual pages and update it after every ingest,
substantive update, archive, or distillation operation.

## Sources

- [[source-summary-minitron-2024|Minitron (2024)]] - Combines structured
  pruning, activation-based importance, architecture search, and
  distillation-based recovery. Status: `review`.
- [[source-summary-grafting-2025|Grafting (2025)]] - Edits pretrained
  diffusion transformers through local activation distillation and integrated
  model recovery. Status: `review`.
- [[source-summary-mone-2026|MoNE (2026)]] - Replaces low-use, low-variance MoE
  experts with constant mean-output novices. Status: `review`.
- [[source-summary-modegpt-2025|MoDeGPT (2025)]] - Jointly decomposes matrix
  pairs inside Transformer modules and allocates nonuniform sparsity from BI.
  Status: `review`.

All currently registered supervisor-provided papers have been ingested.

## Concepts

- [[concept-replacement-error-propagation|Replacement error propagation]]:
  Explains why isolated operator accuracy does not guarantee accuracy after
  multiple replacements are composed.
- [[concept-moe-parameter-accounting|MoE parameter accounting]]: Separates
  total stored parameters, measured memory, and routing-dependent active
  parameters.
- [[concept-model-compression-evaluation-axes|Model compression evaluation axes]]:
  Distinguishes structural footprint, storage, memory, computation,
  compatibility, cost, and quality.

## Entities

No model, dataset, software, or other entity pages have been created yet.

## Methods

- [[method-minitron-activation-based-importance|Minitron activation-based importance]]:
  Forward-only width-component ranking from calibration activations.
- [[method-block-importance|Block Importance and MLP screening adaptations]] -
  Canonical Transformer-layer BI, raw MLP input-output cosine distance, and a
  proposed residual-aware MLP influence score with their boundaries separated.
- [[method-post-pruning-knowledge-distillation|Post-pruning knowledge distillation]]:
  Recovery of a pruned student from uncompressed-teacher outputs and optional
  intermediate states.
- [[method-retraining-assisted-architecture-search|Retraining-assisted architecture search]]:
  Candidate comparison under a shared lightweight recovery budget.
- [[method-two-stage-operator-grafting|Two-stage operator grafting]]: Local
  activation regression followed by integrated model-level fine-tuning.
- [[method-frequency-variance-expert-redundancy|Frequency-variance expert redundancy]]:
  Ranks MoE experts using routing behavior and output variance.
- [[method-mone-novice-expert-replacement|MoNE novice expert replacement]]:
  Replaces selected experts with closed-form constant mean outputs.
- [[method-modegpt-modular-decomposition|MoDeGPT modular decomposition]]:
  Forward-only structured dimension reduction for MLP and attention matrix
  pairs.
- [[method-modegpt-global-sparsity-allocation|MoDeGPT global sparsity allocation]]:
  Maps BI scores to a smoothed per-layer sparsity distribution under a global
  compression target.
- [[method-global-to-local-operator-budget-allocation|Global-to-local operator budget allocation]]:
  Converts one whole-model parameter-sparsity target into configurable
  importance-aware MLP replacement caps without selecting local operators.
- [[method-quality-preservation-evaluation|Quality preservation evaluation]]:
  Defines routine, confirmation, and optional profiles for intrinsic and
  downstream model evaluation.

## Implementations

- [[implementation-compute-environments|Research compute environments]]:
  Records the local MVP and shared remote experiment capacity, setup, and
  constraints without identifying the remote host or its owner.

## Research

- [[decision-primary-compression-evaluation-scope|Primary compression evaluation scope]]:
  Makes footprint-quality trade-offs primary and limits systems metrics to
  optional controlled observations.
- [[decision-working-experiment-code-standards|Working experiment code standards]]:
  Keeps initial notebook code direct and readable while deferring tests and
  production abstractions until explicitly requested.

## Experiments and Findings

- [[experiment-initial-block-compression-study|Initial block-compression study]]:
  Working three-notebook design covering activation geometry, one practical
  narrow-SwiGLU baseline, independent dense-linear multi-block degradation, and
  an optional quantization baseline. Status: `draft`; no direct results are
  recorded.

No MVP evidence has been ingested into canonical experiment pages. The
historical archive remains available at `docs/prototype/mvp/`.

## Syntheses and Comparisons

No synthesis pages have been created yet.

## Maintenance

- Schema version: 1.1
- Last structural lint: 2026-08-08 (global-to-local operator budget methodology)
- Orphan pages: none detected
- Registered source collections: 3
- Registered individual sources: 4
- Ingested individual sources: 4
