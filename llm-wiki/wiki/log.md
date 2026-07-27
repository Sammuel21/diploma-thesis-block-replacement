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

## [2026-07-18] distill | Model compression evaluation framework

- Created a taxonomy separating parameters, storage, runtime memory, compute,
  systems behavior, compatibility, cost, and quality.
- Recorded the supervisor-reviewed direction to prioritize footprint-quality
  trade-offs and keep inference systems metrics outside the primary scope.
- Proposed a three-task routine suite and a five-task confirmation suite aligned
  with MoDeGPT, with WikiText-2 loss and perplexity evaluated separately.
- Distilled the resulting measurement and reporting contract to
  `docs/methodology/evaluation-framework.md`.
- Left the benchmark protocol at `review` pending researcher approval and
  registration of canonical benchmark and evaluation-harness sources.

## [2026-07-18] lint | Evaluation framework update

- Checked required frontmatter fields and filename-to-ID agreement for all 18
  maintained wiki pages.
- Checked internal wikilinks and references to the four registered source IDs.
- Checked changed files for whitespace errors.
- No broken wiki links, unregistered source references, or structural errors
  were found.

## [2026-07-18] update | Benchmark option catalogue

- Preserved PIQA, ARC-Easy, and WinoGrande as the routine downstream suite and
  ARC-Challenge and HellaSwag as confirmation additions.
- Defined MMLU, BoolQ, OpenBookQA, and GSM8K as optional or conditional choices
  with explicit inclusion rules and trade-offs.
- Added named smoke, routine, confirmation, extended-knowledge, and
  conditional-math profiles to the wiki and human methodology document.
- Kept optional tasks outside the canonical five-task macro average to prevent
  changing the primary comparison after results are observed.

## [2026-07-18] lint | Benchmark option catalogue

- Rechecked required metadata and filename-to-ID agreement for all 18
  maintained wiki pages after the benchmark update.
- Rechecked internal wikilinks and references to the four registered source
  IDs.
- Rechecked changed files for whitespace errors.
- No broken wiki links, unregistered source references, or structural errors
  were found.

## [2026-07-21] update | Research compute environments

- Added a sanitized implementation record for the local MVP and shared remote
  experiment environments.
- Recorded reproducible user-level setup, capacity constraints, and the
  successful remote smoke run without host identifiers, network data, account
  names, supervisor information, or other-user details.
- Linked the record to the primary footprint-quality evaluation decision and
  explicitly excluded latency and deployment-performance interpretation.

## [2026-07-27] update | Working initial block-compression experiments

- Added one draft experiment page for the activation-analysis,
  baseline-testing, and degradation-analysis notebook progression.
- Recorded the scientific rationale, current configuration, PCA/effective-rank
  distinction, intended artifacts, and interpretation limits without claiming
  unexecuted results.
- Added a project decision for direct working-experiment code, natural
  multiline formatting, brief docstrings, restrained abstraction, and tests
  only on explicit researcher request.
- Recorded the GPT Sol X-high overengineering lesson as a researcher assessment
  and linked the new pages to BI, error propagation, evaluation scope, and the
  compute environment.

## [2026-07-27] lint | Initial experiments and code standards

- Checked all 21 maintained pages for frontmatter structure, controlled values,
  dates, filename-to-ID agreement, required body sections, and registered
  source references.
- Checked internal wikilinks, repository artifact links, orphan pages, trailing
  whitespace, and reciprocal `related` edges.
- Repaired four pre-existing one-way relationship edges involving evaluation
  axes, source summaries, compute environments, and MoE parameter accounting.
- Found no remaining structural, graph, or provenance errors or warnings. This
  mechanical lint does not verify scientific claims or unexecuted experiments.
