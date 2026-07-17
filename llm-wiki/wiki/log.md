# Wiki Operation Log

This file is append-only. Entries use the format:

`## [YYYY-MM-DD] operation | subject`

## [2026-07-17] bootstrap | LLM-wiki architecture

- Defined the schema, metadata vocabulary, source registries, templates, and
  scoped agent instructions.
- Registered source collections without ingesting individual sources.
- Preserved legacy notes as raw migration inputs.
- Created an empty content index for controlled pilot ingestion.

## [2026-07-17] lint | Bootstrap structure

- Checked registered collection paths and required collection metadata.
- Checked template frontmatter fields, Markdown links, and fenced blocks.
- Confirmed that no maintained knowledge pages or individual sources exist yet.

## [2026-07-17] register | Core source-paper registry

- Simplified the collection catalogue to primary papers, secondary papers, and
  the researcher idea inbox.
- Registered four supervisor-provided papers with bibliographic identity,
  canonical URLs, local paths, and SHA-256 hashes.
- No paper was summarized or marked as ingested.

## [2026-07-17] schema | Implementation page type

- Extended the canonical schema from version 1 to version 1.1 with an additive
  `implementation` page type.
- Added the `wiki/implementations/` location and an implementation template
  covering code paths, architecture, version, validation, limitations, and
  experiment relationships.
- Updated the human architecture and metadata references.
- No page migration was required because no implementation pages existed.

## [2026-07-17] ingest | Minitron structured pruning and distillation

- Read the registered arXiv v2 snapshot of `src-minitron-2024` and marked that
  source as ingested.
- Created a source summary and connected method pages for activation-based
  importance, Block Importance, post-pruning knowledge distillation, and
  retraining-assisted architecture search.
- Recorded the distinction between calibration and recovery budgets, the
  context-dependent meanings of one-shot and iterative compression, and the
  transfer boundary between structured pruning and learned MLP-block
  replacement.
- Left pages at `review` pending researcher review. The original ShortGPT source
  for BI, independent reproduction, and transfer to block replacement remain
  unresolved.

## [2026-07-17] ingest | Diffusion Transformer grafting

- Read the registered arXiv v2 snapshot of `src-grafting-2025` and marked that
  source as ingested.
- Created a source summary, a two-stage operator-grafting method page, and a
  replacement-error-propagation concept page.
- Integrated grafting with the existing recovery graph while preserving the
  distinction between activation-regressed replacement and post-pruning logit
  distillation.
- Recorded self-grafting as a control and marked transfer from diffusion
  transformers to autoregressive LLMs as unresolved.

## [2026-07-17] ingest | MoNE expert replacement

- Read the registered arXiv v2 snapshot of `src-mone-2026` and marked that
  source as ingested.
- Created pages for frequency-variance expert screening, constant novice
  replacement, and MoE parameter accounting.
- Connected MoNE to the activation-importance and operator-grafting methods
  while preserving its closed-form, routing-specific replacement design.
- Recorded calibration sensitivity on specialized tasks and separated total
  parameters, measured memory, active parameters, and novice hit ratio.

## [2026-07-17] ingest | MoDeGPT modular decomposition

- Read the registered ICLR 2025 conference paper `src-modegpt-2025` and marked
  that source as ingested.
- Created a source summary and separate method pages for local modular
  decomposition and model-level global sparsity allocation.
- Updated Block Importance to record its continuous budget-allocation use and
  connected MoDeGPT's closed-form compression to learned operator grafting.
- Recorded the separation between calibration and optional LoRA recovery,
  module-normalized sensitivity, calibration dependence, and temporary
  compression memory cost.
