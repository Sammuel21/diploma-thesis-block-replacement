# Wiki Index

This is the content-oriented entry point for maintained thesis knowledge.
Read it before searching individual pages and update it after every ingest,
substantive update, archive, or distillation operation.

## Current Research Focus

The homogeneous SwiGLU-1 through SwiGLU-7 progression is complete. Start from
[[synthesis-homogeneous-swiglu-findings]] for the working discrete-width,
replacement-only recovery control, its attribution, and its limits. Detailed
tables and plots remain in the linked experiment reports and notebooks.

Baseline and generic operator-family singleton profiles are also preserved.
Heterogeneous replacement and generic multi-block interactions remain separate
research directions; the interaction skeleton is
[`block-interaction.ipynb`](../../notebooks/model/interaction/block-interaction.ipynb).
Its subset construction, operator controls, interaction metric, ordering, and
recovery protocol remain to be designed. Singleton KL profiles must not be
assumed to compose additively. The separate
[Quantization-1](../../docs/experiments/model/baseline/quantization-1.md)
baseline is implemented, with its scientific GPU execution still pending.

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
- [[source-summary-phase-transitions-compression-2026|Phase Transitions in LLM Compression (2026)]] -
  Supporting Perspective on structural, numerical, and algebraic redundancy
  axes and fitted compression-collapse boundaries. Status: `draft`;
  Supplementary Information pending review.

All currently registered supervisor-provided papers have been ingested.
The researcher-found Phase Transitions Perspective remains `in-review` because
its main article was checked but its central supplementary material was not
accessible during ingestion.

Canonical WikiText, PIQA, ARC, WinoGrande, HellaSwag and LM Evaluation Harness
v0.4.13 references are now individually registered in
[`raw/sources.yml`](../raw/sources.yml) for SwiGLU-6 evaluation. They remain
`registered`; no paper ingestion or source-summary pages were added.

TorchAO 0.17.0, its compatibility table, MSLK 1.1.0+cu128, TorchAO serialization,
and PyTorch benchmarking references are registered in the same registry for
[Quantization-1](../../docs/experiments/model/baseline/quantization-1.md).
Selected APIs were checked; all five entries remain `registered`. Scientific
backend validation and experiment execution are pending.

The original LoRA paper (2021 preprint v2) is registered for recovery-scope
attribution. Sections 4.1-4.2 were selectively checked; the source remains
`registered`, with no full ingestion or source-summary page implied.

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
  Adapts MoDeGPT's importance-weighted softmax allocation into bounded,
  feasible-size MLP replacement caps; the full heterogeneous proposal remains
  unverified, separate from the working homogeneous control.
- [[method-discrete-width-curve-allocation|Discrete width-curve allocation]]:
  Selects homogeneous SwiGLU widths by minimizing a summed singleton-KL
  surrogate under one shared parameter budget; not a composition theorem.
- [[method-hybrid-operator-replacement|Hybrid linear-nonlinear MLP replacement]]:
  Defines a project-proposed linear-plus-nonlinear drop-in operator while
  leaving its internal capacity allocation as unresolved future work.
- [[method-quality-preservation-evaluation|Quality preservation evaluation]]:
  Defines routine, confirmation, and optional profiles for intrinsic and
  downstream model evaluation.

## Implementations

- [[implementation-maintained-mlp-replacement-package|Maintained MLP replacement package]]:
  Separates reusable scientific code, top-level experiment runners, and
  executor-specific jobs; records completed dedicated SwiGLU-7 execution
  without certifying generic launchers or pending quantization runs.
- [[implementation-compute-environments|Research compute environments]]:
  Records the local MVP and shared remote experiment capacity, setup, and
  constraints without identifying the remote host or its owner.

## Research

- [[decision-primary-compression-evaluation-scope|Primary compression evaluation scope]]:
  Makes footprint-quality trade-offs primary and limits systems metrics to
  optional controlled observations.
- [[decision-working-experiment-code-standards|Working experiment code standards]]:
  Separates researcher-structured notebook work from reliable maintained code
  and workflows, and routes implementation agents by unresolved judgment and
  failure cost.

## Experiments and Findings

- [[experiment-initial-block-compression-study|Initial block-compression study]]:
  Working design covering activation geometry, a fixed six-condition
  single-block baseline protocol with a draft reusable runner, independent
  dense-linear degradation, and links to completed homogeneous work and the
  implemented-but-unexecuted quantization baseline. Status: `draft`; numerical
  evidence remains on downstream pages and reports.
- [[experiment-baseline-operator-analysis|Baseline operator analysis]]:
  Artifact-backed single-run evidence for calibration scaling, reduced-SwiGLU
  capacity, replacement-only recovery, and full-depth isolated replacement
  sensitivity. Status: `draft`; schema-2 evidence is checked, while newer
  notebook additions remain pending rerun.
- [[experiment-swiglu-operator-design-progression|SwiGLU operator-design progression]]:
  Artifact-backed full-depth comparison of ten generic operator
  configurations. Status: `draft`; preserves the checked singleton results,
  links the completed homogeneous family, and leaves branch/factorization/
  nested-topology proposals unverified.

No MVP evidence has been ingested into canonical experiment pages. The
historical archive remains available at `docs/prototype/mvp/`.

## Syntheses and Comparisons

- [[comparison-recovery-objective-trainable-scope|Recovery objective and trainable scope]]:
  Separates loss placement from parameter scope, including all three completed
  S7 variants and their Minitron, Grafting, and LoRA attribution.
- [[synthesis-homogeneous-swiglu-findings|Homogeneous SwiGLU findings and methodology boundary]]:
  Compact SwiGLU-1 through SwiGLU-7 decisions, evidence anchors, borrowed-method
  boundaries, and limitations. Status: `review`; experiment-backed, not
  independently reproduced or a claim of literature novelty.

## Maintenance

- Schema version: 1.1
- Last structural lint: 2026-10-02 (all maintained pages and source registries)
- Last focused scientific/provenance review: 2026-10-02 (homogeneous SwiGLU)
- Orphan pages: none detected
- Registered source collections: 6
- Registered individual sources: 17
- Ingested individual sources: 4
- Sources in review: 1
