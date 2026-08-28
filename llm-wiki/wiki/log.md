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

## [2026-08-09] update | Budget scope and discrete operator sizing

- Distinguished local, eligible-MLP, and whole-model parameter-reduction
  fractions and defined the exact accounting relationships between them.
- Allowed the allocator target to be declared over either the eligible MLP
  scope or the whole model, with both forms converted to one removal quota and
  replacement budget.
- Defined parameter reduction as the project term for smaller dense
  replacements while retaining sparsity only as explicitly scoped shorthand.
- Added integer-dimension, optional alignment, and generic feasible-candidate
  projection rules explaining why actual operator use can remain below a
  continuous cap.
- Kept operator choice downstream of allocation and added no empirical claim,
  source-registry change, notebook change, or documentation distillation.

## [2026-08-09] lint | Budget scope and discrete operator sizing

- Checked all 22 maintained pages for required metadata, controlled values,
  dates, filename-to-ID agreement, verification states, required body
  sections, registered source IDs, locators, and relations.
- Checked wikilinks, repository links, index coverage, duplicate IDs and
  relationships, reciprocal `related` edges, orphan coverage, conflict
  markers, trailing whitespace, and changed-file whitespace.
- Confirmed balanced display-math delimiters and `aligned` environments in the
  changed method and reviewed every added equation as standard accounting,
  standard parameter counting, or an explicit project-proposed definition.
- Reviewed terminology for local, eligible-MLP, and whole-model reduction,
  retention, continuous caps, discrete actual use, and optional alignment.
- Found no wiki structural, graph, provenance, math, or semantic errors or
  warnings. The earlier compact `docs/` distillation remains unchanged and is
  explicitly identified on the method page; lint does not verify empirical
  effectiveness.

## [2026-08-09] update | Hybrid replacement method and baseline width

- Corrected the practical single-block baseline so its 50% width is defined
  against the teacher MLP intermediate dimension $d_{\mathrm{ff}}$, not
  $d_{\mathrm{model}}$.
- Updated the unexecuted notebook design and maintained experiment page to
  expect width 4096, 25,165,824 replacement parameters, and 50% block-parameter
  retention for the configured SmolLM2-1.7B target.
- Created `method-hybrid-operator-replacement` as a draft, unverified project
  method covering dense or factorized linear branches, compact nonlinear
  corrections, staged local fitting, controls, and model-level evaluation.
- Kept internal linear/nonlinear capacity allocation explicitly unresolved;
  the block-budget source, branch families, feasible integer projection, split
  selection rule, and unused-capacity policy remain future work.
- Updated reciprocal method and experiment links, notebook navigation, and the
  exploratory hybrid note in `budget.ipynb`. No literature source, empirical
  result, or optimality claim was added.

## [2026-08-09] lint | Hybrid replacement method and baseline width

- Checked all 23 maintained pages for frontmatter, required fields, controlled
  type/status/epistemic values, dates, filename-to-ID agreement, verification
  states, and type-required body sections.
- Checked registered source references, all wikilinks and repository links,
  index coverage, reciprocal `related` edges, duplicate IDs, conflict markers,
  trailing whitespace, display-math delimiters, and `aligned` environments.
- Confirmed that active maintained wiki pages contain no stale
  $d_{\mathrm{model}}/2$, 12.5%, or 87.5% baseline semantics and that the
  hybrid page explicitly leaves internal capacity allocation unresolved.
- Found zero structural, graph, provenance, math, or scoped semantic errors or
  warnings. The new method and corrected baseline remain `draft` and
  `unverified`; lint does not establish empirical effectiveness.

## [2026-08-11] update | SwiGLU operator-design progression

- Created `experiment-swiglu-operator-design-progression` as a draft,
  unverified path from fixed controls and generic whole-MLP substitutes to
  structure-aware internal surgery and teacher-tailored nested operators.
- Formalized the configured teacher SwiGLU as gate, value, interaction, and
  down-projection components and recorded their interface constraints without
  claiming that the decomposition determines an optimal replacement.
- Defined aligned intermediate channels across gate rows, value rows, and down
  columns, and kept proposed component scores explicitly conditional on
  held-out ablation or model-quality validation.
- Separated Minitron neuron importance, MoDeGPT gated-MLP decomposition, and
  Grafting replacement/recovery as prior-work context from the researcher's
  proposed block-specific composition direction.
- Corrected operator notation by distinguishing linear rank $r_L$ from
  nonlinear width $r_N$ and teacher width $d_{\mathrm{ff}}$; nonlinear width
  is not constrained by the model-width rank bound.
- Updated the fixed baseline experiment, hybrid method, reciprocal links, and
  wiki index. No notebook, maintained implementation, or empirical finding was
  changed.

## [2026-08-11] lint | SwiGLU operator-design progression

- Checked all 24 maintained pages for required frontmatter, controlled type
  and status values, dates, filename-to-ID agreement, duplicate IDs, required
  body sections, and the new experiment-specific sections.
- Checked registered source references, wikilinks, reciprocal `related` edges,
  index coverage, repository-relative links, conflict markers, trailing
  whitespace, changed-file whitespace, and display-math delimiters.
- Reviewed the new equations as explanatory teacher notation, standard
  parameter counting, or explicit project-proposed definitions with their
  citation requirements stated in the page.
- Found no structural, graph, provenance, link, math-delimiter, or scoped
  semantic errors. The experiment remains `draft` and `unverified`; lint does
  not establish effectiveness or literature novelty.

## [2026-08-11] ingest | Phase Transitions Perspective and Tier I axes

- Registered `src-phase-transitions-compression-2026` in the researcher-found
  secondary collection as a supporting scholarly-secondary source.
- Reviewed the complete main article and created a draft source summary that
  separates its peer-reviewed Perspective status, comparative evidence,
  synthesis, and stronger speculative or universal claims.
- Recorded structural, numerical, and algebraic redundancy as the source's
  taxonomy without adopting its error-additivity, universal-threshold, 90%
  redundancy, or deployment claims as project facts.
- Kept the source `in-review` because the publisher's supplementary download
  returned a client challenge and the stated orthogonality proof could not be
  independently inspected.
- Extended the SwiGLU operator-design progression with compact Tier I axes for
  family, capacity, calibration data, layer, recovery, randomness, and later
  composition studies.
- Added a coarse-to-boundary protocol and labelled its operational boundary as
  a project-proposed definition rather than a source-derived phase-transition
  result. No notebook, implementation, or empirical finding was changed.

## [2026-08-11] lint | Phase Transitions source and Tier I axes

- Checked all 25 maintained pages for required frontmatter shape, controlled
  metadata values, dates, filename-to-ID agreement, and duplicate IDs.
- Checked all five registered source IDs, source locators and relations,
  index coverage, wikilinks, reciprocal `related` edges, required sections on
  the new source and updated experiment pages, conflict markers, trailing
  whitespace, and display-math balance.
- Reviewed the new material for separation of source claims, synthesis,
  project-proposed definitions, and unresolved supplementary evidence.
- Found no structural, graph, provenance, formatting, or scoped semantic
  errors. The new source remains `in-review`, and the operator experiment
  remains `draft` and `unverified`; lint does not validate either source claims
  or operator effectiveness.

## [2026-08-13] update | Single-block baseline protocol and runner boundary

- Consolidated the six baseline conditions and their distinct control,
  approximability, and learned-compression roles in the existing initial
  block-compression experiment page rather than creating a duplicate method or
  file-level implementation page.
- Recorded the shared-context comparison contract and clarified that the fixed
  suite is not a parameter-matched operator-family comparison.
- Documented the inputs, responsibilities, outputs, and exclusions of the
  draft `src/mlp_replacement/baselines.py` runner and linked it to the planned
  SwiGLU operator-design progression.
- Kept the runner explicitly unintegrated and unverified: the current baseline
  notebook still executes inline logic, and no runtime-parity result or
  empirical finding was promoted.

## [2026-08-13] lint | Single-block baseline documentation

- Checked all 25 maintained pages for required frontmatter fields, controlled
  type, status, and epistemic-role values, dates, unique IDs, filename-to-ID
  agreement, required relationship and source sections, and balanced display
  math.
- Checked registered source IDs, index coverage, wikilinks, reciprocal
  `related` edges, repository-relative links, conflict markers, trailing
  whitespace, and changed-file whitespace.
- Reviewed the new operator equations as standard explanatory notation and the
  baseline selection as an explicit project decision requiring no external
  citation.
- Found no structural, graph, provenance, link, formatting, or scoped semantic
  errors. The baseline experiment and reusable runner remain `draft` and
  `unverified`; lint does not establish runtime parity or empirical quality.

## [2026-08-15] update | Recovery objective and trainable scope

- Created `comparison-recovery-objective-trainable-scope` to distinguish the
  location of a recovery objective from the parameters updated by it.
- Recorded the current project as model-level teacher-logit KD with
  replacement-only updates, based on the maintained recovery implementation
  and exploratory width-recovery notebook.
- Compared that scope with Minitron's full pruned-student retraining,
  Grafting's main end-to-end DiT recovery, and Grafting's PixArt-Sigma LoRA
  exception using the registered source snapshots.
- Added reciprocal links from the Minitron and Grafting method pages and added
  the comparison to the wiki index. No implementation or empirical finding
  was changed.

## [2026-08-15] lint | Recovery objective and trainable scope

- Checked all 26 maintained pages for required frontmatter, controlled type,
  status, and epistemic-role values, dates, unique IDs, filename-to-ID
  agreement, and required body sections.
- Checked all registered source references, index coverage, wikilinks,
  reciprocal `related` edges, repository-relative links, conflict markers,
  trailing whitespace, and balanced display-math delimiters.
- Reviewed the new comparison for explicit separation of source-derived
  claims, repository-observed behavior, and synthesis terminology.
- Found no structural, graph, provenance, link, formatting, or scoped semantic
  errors. Lint does not establish that replacement-only recovery is
  empirically equivalent to the source-paper recovery protocols.

## [2026-08-18] update | Baseline operator-analysis methodology

- Created `experiment-baseline-operator-analysis` as the methodology page for
  `baseline-experiments.ipynb`, keeping it distinct from the fixed
  six-condition `baseline-testing.ipynb` protocol.
- Recorded the calibration-data, reduced-SwiGLU width, replacement-only
  recovery, full-depth pre-recovery KL, and global-to-local analysis designs.
- Distinguished calibration data, local optimizer exposure, validation, KL
  evaluation, and recovery-update budgets; no notebook result was promoted.
- Documented the current large CPU teacher-logit cache and a planned shared
  streaming or chunked recovery design without changing implementation code.
- Marked unexecuted cells, placeholder analyses, workflow controls, and memory
  optimizations as pending or planned, and added reciprocal experiment,
  recovery-scope, and error-propagation links.

## [2026-08-18] lint | Baseline operator-analysis methodology

- Checked all 27 maintained pages for required frontmatter fields, controlled
  type, status, epistemic-role, verification, and source-relation values,
  dates, unique IDs, filename-to-ID agreement, required body sections,
  conflict markers, trailing whitespace, and balanced display math.
- Checked the five registered source IDs, source locators, index coverage,
  wikilinks, reciprocal `related` edges, and repository-relative links.
- Reviewed the new experiment page for separation of prior-work context,
  project definitions, researcher questions and hypotheses, observed
  implementation state, planned work, and non-promoted notebook output.
- Found no structural, graph, provenance, link, formatting, or scoped semantic
  errors. The experiment remains `draft` and `unverified`; lint does not
  establish the adequacy of its budgets, the memory benefit of an unimplemented
  streaming loop, or any empirical result.

## [2026-08-20] update | Controlled operator-family comparisons

- Extended `experiment-swiglu-operator-design-progression` with three distinct
  comparison views: matched-footprint architecture controls, within-family
  capacity curves, and cross-family Pareto analysis.
- Recorded that equal stored parameters do not imply equal expressivity, using
  dense $d^2$ and rank-$d/2$ factorized linear maps as the project control.
- Kept the rationale explicitly project-defined and unverified; no empirical
  operator ranking or result was promoted.

## [2026-08-20] update | Artifact-backed singleton profiling and active focus

- Promoted bounded results from the local schema-2 baseline artifact and
  schema-1 operator artifact to experiment-backed evidence while retaining
  both experiment pages at `draft` status.
- Recorded the calibration, width, recovery, full-depth KL, matched-footprint
  operator, local-to-global, and layer-sensitivity observations with their
  single-model, single-seed, isolated-replacement limitations.
- Distinguished operator-conditioned pre-recovery KL as the current empirical
  reference for singleton replacement sensitivity from canonical BI and the
  project-proposed residual-aware MLP BI screening candidates.
- Updated the umbrella experiment and wiki index to make simultaneous
  multi-block replacement interaction the active next focus, linked to the
  new model-level notebook skeleton without inventing a frozen protocol or
  result.
- Left activation tensors, logits, fitted weights, and local ignored artifacts
  outside the maintained wiki; only artifact paths, configurations, checked
  summaries, and bounded interpretations were recorded.

## [2026-08-20] lint | Singleton profiling evidence and focus transition

- Checked all 27 maintained pages for required frontmatter, controlled type,
  status, epistemic, verification, granularity, pipeline-stage, source-
  relation, ID, filename, and date values.
- Checked registered source IDs, index coverage, wikilinks, reciprocal
  `related` edges, repository-relative links including both local artifacts and
  the new model-level notebook, conflict markers, trailing whitespace, display
  math balance, and changed-file whitespace.
- Cross-checked the baseline and operator artifacts for row counts, unique
  operator-layer keys, shared-row consistency, KL-reduction arithmetic,
  configuration provenance, and the numerical summaries promoted to the wiki.
- Reviewed the changed pages for separation of source-derived BI, project-
  proposed residual-aware MLP BI, artifact-backed singleton KL findings,
  anomalous observations, pending notebook additions, and unimplemented
  multi-block methodology.
- Found no structural, graph, provenance, link, formatting, or scoped semantic
  errors or warnings. Lint does not establish cross-seed, cross-model, or
  simultaneous-replacement generality.

## [2026-08-20] update | Compact block-experiment documentation

- Compacted `experiment-initial-block-compression-study`,
  `experiment-baseline-operator-analysis`, and
  `experiment-swiglu-operator-design-progression` to reduce duplicated
  methodology, configuration, and implementation detail.
- Kept the umbrella page focused on stage ownership, the baseline page focused
  on calibration/capacity/recovery evidence, and the operator page focused on
  generic-family results plus the bounded teacher-tailored progression.
- Preserved stable IDs, metadata, equations, source locators, artifact links,
  direct numerical findings, epistemic labels, and unresolved limitations.
- Reduced the three pages from 8,039 to 3,510 words in total without changing
  their experiment or verification status.

## [2026-08-20] lint | Compact block-experiment documentation

- Checked all 27 maintained pages and five registered sources for required
  metadata, controlled type/status/epistemic values, dates, unique IDs,
  filename-to-ID agreement, index coverage, source registration, required
  relationship and source sections, reciprocal wikilinks, repository links,
  conflict markers, trailing whitespace, and balanced display math.
- Reviewed the compacted pages for retained separation of prior work,
  researcher hypotheses, project definitions, experiment-backed findings, and
  pending work.
- Found no structural, graph, provenance, link, formatting, or scoped semantic
  errors. Lint does not expand the evidence beyond the existing single-model,
  single-seed, singleton-replacement artifacts.

## [2026-08-22] update | Correct MoDeGPT allocation attribution

- Reclassified the reference negative-softmax allocation in
  `method-global-to-local-operator-budget-allocation` as a source-derived
  adaptation of MoDeGPT Section 3.3, Equations 10-11, rather than an original
  project allocation rule.
- Added the equal-block-size equivalence between MoDeGPT's assigned sparsity
  and the project's assigned parameter-removal fraction.
- Preserved target-scope conversion, parameter-size weighting, bounded cap
  semantics, discrete operator realization, and unused-budget reconciliation
  as explicitly project-proposed extensions whose effectiveness remains
  unverified.
- Updated the method's source relation and the wiki index summary without
  changing any experiment, implementation, or distilled documentation.

## [2026-08-22] lint | MoDeGPT allocation attribution

- Checked all 27 maintained pages and five registered sources for required
  metadata, IDs, dates, controlled source relations, source registration,
  index coverage, wikilinks, required sections, display-math balance, conflict
  markers, and trailing whitespace.
- Reviewed the changed method against the registered MoDeGPT source at Section
  3.3, Equations 10-11, and verified that adopted and project-proposed parts
  are now distinguished explicitly.
- Found no structural, graph, provenance, formatting, or scoped semantic
  errors. This lint does not empirically validate transfer of the allocation
  rule from continuous sparsity to MLP replacement budgets.

## [2026-08-28] update | Restructure maintained source-package responsibilities

- Grouped reusable source modules under `analysis/`, `compression/`,
  `evaluation/`, and `operators/`, while retaining cross-cutting configuration,
  data, model, capture, and run-log modules at the package root.
- Moved operator baselines into `operators/`; model-level selection, surgery,
  recovery, and workflows into `compression/`; and activation, screening,
  sensitivity, and interaction diagnostics into `analysis/`.
- Updated required relative imports, the maintained pipeline entry point, and
  seven working notebooks without intentionally changing implementation logic.
- Added `implementation-maintained-mlp-replacement-package` to document module
  responsibilities, data flow, entry points, maturity, and limitations.
- Updated affected repository links and removed the stale link to the deleted
  `degradation-analysis.ipynb`; the unused interaction helper remains preserved
  as provisional code.

## [2026-08-28] lint | Maintained source-package restructuring

- Checked 28 maintained wiki pages for required frontmatter, unique IDs,
  filename-to-ID agreement, index coverage, required relationship and source
  sections, wikilinks, repository links, conflict markers, and trailing
  whitespace.
- Parsed the seven notebooks whose imports changed, resolved 63 maintained
  absolute imports and 28 relative source imports statically, and found no
  remaining imports of the superseded root module paths.
- Runtime import execution was unavailable on the local machine because no
  Python interpreter is installed; the moved package must therefore receive a
  runtime import check in the configured remote environment before experiment
  execution.
- Found no remaining structural, graph, link, formatting, or scoped semantic
  errors. This lint verifies organization and import targets, not experimental
  behavior or result reproducibility.
