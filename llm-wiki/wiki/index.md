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
- [[method-block-importance|Block Importance]] - Transformer-layer sensitivity
  from input-output cosine distance, currently documented through Minitron's
  use of the metric.
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
- [[method-quality-preservation-evaluation|Quality preservation evaluation]]:
  Defines routine, confirmation, and optional profiles for intrinsic and
  downstream model evaluation.

## Implementations

No implementation pages have been created yet. The page type and template are
available for stable components and pipelines during codebase consolidation.

## Research

- [[decision-primary-compression-evaluation-scope|Primary compression evaluation scope]]:
  Makes footprint-quality trade-offs primary and limits systems metrics to
  optional controlled observations.

## Experiments and Findings

No MVP evidence has been ingested into canonical experiment pages yet. The
historical archive is available at `docs/prototype/mvp/`.

## Syntheses and Comparisons

No synthesis pages have been created yet.

## Maintenance

- Schema version: 1.1
- Last structural lint: 2026-07-18 (evaluation framework update)
- Orphan pages: none detected
- Registered source collections: 3
- Registered individual sources: 4
- Ingested individual sources: 4
