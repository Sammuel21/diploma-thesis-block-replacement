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

## [2026-07-27] update | Quantization baseline idea

- Added uniform MLP weight quantization as an optional numerical baseline for
  the initial block-compression study, beginning with 8-bit and 4-bit variants.
- Recorded importance-aware mixed precision as a stronger follow-up and kept
  its selection contribution distinct from the uniform baseline.
- Required storage-footprint matching rather than parameter-count matching and
  retained quantization of the best fitted replacement as a small combined
  compression experiment.
- Left implementation, calibration, importance estimation, and precise budget
  choices unresolved; no empirical result is claimed.

## [2026-07-27] lint | Quantization baseline update

- Checked the changed experiment page, index entry, and operation log for
  conflict markers and trailing whitespace.
- Confirmed that the update introduces no new wiki links or source claims and
  preserves the experiment page's draft, unverified status.

## [2026-07-27] governance | Scoped wiki instructions

- Condensed `llm-wiki/AGENTS.md` by routing detailed metadata and operation
  procedures to the authoritative `SCHEMA.md` instead of repeating them.
- Preserved the source boundary, evidence discipline, historical-MVP boundary,
  explicit-operation requirement, and wiki completion checklist.
- Limited source-registry consultation to operations involving source identity
  or provenance.

## [2026-07-27] update | Human-readable call formatting

- Revised the working experiment code standard so definitions and calls with
  five or fewer items remain compact or use row-like continuation lines.
- Reserved one-parameter-or-argument-per-line formatting for calls with more
  than five items or individually complex expressions.
- Removed the mechanical one-item-per-line expectation for containers in favor
  of natural grouping based on structure and readability.

## [2026-08-02] update | Focused block-notebook baseline

- Reframed activation analysis as a concise descriptive report while retaining
  its covariance, reconstruction, and local JSON artifact calculations.
- Replaced the matched-budget operator-family grid with one fixed bias-free
  narrow SwiGLU at `r = d_model / 2`, plus original, zero, and mean controls.
- Kept dense-linear multi-block degradation as an independent diagnostic and
  removed B4 and stronger-equal-cost dependency language.
- Updated the experiment page and index without recording empirical findings;
  clean remote execution and result review remain pending.

## [2026-08-02] lint | Focused block-notebook baseline

- Parsed all code cells in the three block-study notebooks and validated their
  notebook structure, unique cell IDs, cleared execution state, and outputs.
- Checked all 21 maintained wiki pages for required metadata, filename-to-ID
  agreement, verification-state validity, required sections, and internal links.
- Confirmed that the experiment remains `draft` and `unverified`, with no direct
  numerical result recorded before the planned remote execution.

## [2026-08-03] governance | Equation provenance and notebook delivery choice

- Extended claim-level provenance rules to require explicit classification of
  introduced equations and a clear statement of their citation requirements.
- Added a repository-wide requirement to confirm whether Jupyter notebook work
  should be applied to the `.ipynb` file or returned as chat code blocks when
  the user has not already specified the delivery form.
- Kept schema version 1.1 because the change adds non-breaking evidence guidance
  and requires no page migration.

## [2026-08-03] update | BI and MLP screening boundaries

- Extended the existing Block Importance method page instead of creating a
  duplicate importance-screening page.
- Kept canonical complete-layer BI as the source-derived baseline and separated
  it from the project's raw MLP input-output cosine-distance implementation.
- Added a residual-stream decomposition and a residual-aware MLP influence
  score explicitly labelled as synthesis and a project-proposed definition.
- Clarified that proxy agreement does not establish model-level importance
  without a controlled model-loss outcome.
- Aligned the initial experiment page with the more precise raw-MLP-score
  terminology without recording a new empirical result.

## [2026-08-08] update | Global-to-local operator budget methodology

- Added a draft project method that converts one whole-model parameter-
  sparsity target into bounded per-block replacement-budget caps without
  selecting local operators.
- Separated fixed accounting and conservation rules from configurable
  importance, normalization, allocation-temperature, local-construction, and
  recovery policies.
- Defined a rank-normalized, size-weighted softmax reference allocator and
  distinguished normalized global removal shares from local retention caps
  and realized sparsities.
- Added optional leftover-budget reconciliation based on feasible marginal
  utility, followed by recovery only after the architecture is finalized.
- Connected the method to BI, MoDeGPT allocation, replacement-error
  propagation, footprint-quality evaluation, and MoE parameter accounting.
- Did not promote exploratory importance-notebook output to an empirical
  finding and made no notebook, source-registry, or production-code change.

## [2026-08-08] lint | Global-to-local operator budget methodology

- Checked all 22 maintained pages for required metadata, controlled values,
  valid dates, filename-to-ID agreement, verification states, required body
  sections, and registered source locators and relations.
- Checked internal wikilinks, repository artifact links, index coverage,
  duplicate IDs and relationships, orphan pages, reciprocal `related` edges,
  exact conflict markers, trailing whitespace, and changed-file whitespace.
- Reviewed the changed methodology surfaces for consistent use of global
  target, initial cap, hard ceiling, actual use, retention, and realized
  sparsity, and checked that every new equation states its provenance class.
- Found no remaining structural, graph, or provenance errors or warnings. The
  new method remains `draft` and `unverified`; lint does not establish its
  empirical effectiveness.

## [2026-08-08] update | Portable Markdown math rendering

- Replaced all 11 plain-text equation blocks in the global-to-local operator
  budget method with `$$`-delimited MathJax supported by both GitHub Markdown
  and Obsidian.
- Used `aligned` environments for multi-line systems and ordinary `$...$`
  delimiters for mathematical notation in prose and tables.
- Preserved the method's equations, epistemic labels, source attribution, and
  unverified status; no methodological or empirical claim was added.

## [2026-08-08] lint | Portable Markdown math rendering

- Rechecked all 22 maintained pages for required metadata, registered sources,
  internal links, index coverage, and reciprocal `related` edges.
- Confirmed 11 balanced display-math blocks, balanced `aligned` environments,
  no unmatched inline-math delimiters, and no remaining plain-text equation
  fences in the changed method page.
- Found no structural, graph, math-delimiter, or changed-file whitespace
  errors or warnings. Rendering was expressed in the shared GitHub-and-Obsidian
  MathJax syntax without introducing platform-specific HTML or equation images.

## [2026-08-08] distill | Global-to-local operator budget methodology

- Created a compact human-facing methodology document under `docs/methodology/` with a
  variable table and six numbered steps from global accounting through
  evaluation.
- Preserved the distinction between fixed accounting, configurable importance,
  downstream operator construction, and optional unused-budget reconciliation.
- Kept the method explicitly project-proposed and unverified, retained the
  MoDeGPT motivation boundary, and added traceability to the maintained wiki.
- Added the document to `docs/README.md` and linked it back from the detailed method
  page.

## [2026-08-08] lint | Distilled operator budget methodology

- Checked the distilled document's numbered structure, balanced inline and
  display math, relative links, documentation navigation, and wiki backlink.
- Rechecked the 22-page maintained wiki graph after the backlink addition,
  including metadata structure, registered sources, internal links, index
  coverage, and reciprocal `related` relationships.
- Found no documentation-link, math-delimiter, structural, graph, or
  changed-file whitespace errors or warnings. This lint does not establish the
  proposed allocator's empirical effectiveness.
