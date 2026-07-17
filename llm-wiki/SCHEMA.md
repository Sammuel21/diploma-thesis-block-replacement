# Thesis LLM Wiki Schema

- Schema version: 1.1
- Effective date: 2026-07-17
- Status: active

This file is the canonical specification for the thesis knowledge base. It
defines structure, metadata, provenance, and maintenance workflows. Human
guides in `docs/knowledge-base/` explain the same system, but this file governs
wiki operations when the two disagree.

## 1. Design principles

1. **Provenance before fluency.** A polished claim without traceable origin is
   less useful than an explicitly uncertain claim.
2. **Separate authorship from evidence.** Who wrote a page is different from
   where its claims came from.
3. **Separate source authority from research priority.** A paper supplied by
   the supervisor is `core` because of curation, while its academic evidence
   kind is recorded independently.
4. **Raw sources are immutable.** Corrections create a new version or a wiki
   annotation; they do not silently rewrite evidence.
5. **The wiki compounds.** Ingestion updates existing pages and relationships,
   not only a disconnected source summary.
6. **Human documentation is distilled.** `docs/` is not a working-note inbox.
7. **Ambiguity is visible.** Contradictions, uncertainty, and missing evidence
   are recorded rather than resolved by invention.

## 2. Directory model

```text
llm-wiki/
|-- AGENTS.md               scoped agent behavior
|-- SCHEMA.md               canonical specification
|-- README.md               orientation
|-- raw/
|   |-- collections.yml     registered source collections
|   |-- sources.yml         individually registered sources
|   `-- legacy/             immutable migration inputs
|-- templates/              page templates, excluded from page lint
`-- wiki/
    |-- index.md             content-oriented navigation
    |-- log.md               append-only operation history
    |-- sources/             source-summary pages
    |-- concepts/            stable concepts and definitions
    |-- entities/            models, datasets, software, and named artifacts
    |-- methods/             algorithms, metrics, and procedures
    |-- implementations/     maintained software components and pipelines
    |-- research/            questions, hypotheses, and decisions
    |-- experiments/         experiments and evidence-bound findings
    `-- syntheses/           comparisons and cross-source synthesis
```

The physical hierarchy stays deliberately small. Page metadata expresses
granularity, pipeline stage, verification, and topic without multiplying
folders for every possible classification axis.

## 3. Page types and placement

| Type | Purpose | Default directory |
| --- | --- | --- |
| `source-summary` | Faithful summary and critique of one registered source | `wiki/sources/` |
| `concept` | Stable terminology, definitions, or theoretical object | `wiki/concepts/` |
| `entity` | Model, dataset, software package, organization, or named artifact | `wiki/entities/` |
| `method` | Algorithm, metric, loss, architecture, or workflow | `wiki/methods/` |
| `implementation` | Maintained project software component or pipeline linked to methods and evidence | `wiki/implementations/` |
| `research-question` | Explicit question the thesis may answer | `wiki/research/` |
| `hypothesis` | Testable researcher proposition | `wiki/research/` |
| `decision` | Project or methodology decision with rationale | `wiki/research/` |
| `experiment` | Design, configuration, execution, and direct results | `wiki/experiments/` |
| `finding` | Evidence-backed interpretation tied to experiments | `wiki/experiments/` |
| `comparison` | Structured comparison across methods or evidence | `wiki/syntheses/` |
| `synthesis` | Integrated understanding across several sources or results | `wiki/syntheses/` |

Create a new page when the object has a stable identity that other pages should
link to. Update an existing page when adding evidence, qualification, or an
attribute of the same object. Do not create a second page merely because a new
source discusses an existing concept.

Implementation pages use architectural granularity. Create them for stable
components, pipelines, or complete workflows that connect research methods to
code and experiments. Do not create a page for every file, class, or function.
An implementation page documents what exists and where; it does not by itself
verify that the implementation is scientifically correct or empirically
effective.

## 4. Canonical page frontmatter

Every maintained page below `wiki/`, except `index.md` and `log.md`, must begin
with this structure:

```yaml
---
id: method-block-importance
title: Block Importance
summary: One-sentence description used by the wiki index.
type: method
status: draft
created: 2026-07-17
updated: 2026-07-17

authorship:
  created_by: collaborative
  contributors:
    - researcher
    - llm

epistemic:
  role: synthesis
  confidence: medium
  verification:
    - source-checked

scope:
  topics:
    - block-importance
  granularities:
    - mlp-block
    - model
  pipeline_stages:
    - screening
    - selection

sources:
  - source_id: src-example-2026
    locator: "Section 3.2"
    relation: defines

related:
  - "[[concept-block-replaceability]]"
supersedes: []
superseded_by: []
---
```

Empty lists must be written as `[]`. Dates use ISO 8601 `YYYY-MM-DD`. Quoted
strings are recommended when punctuation could change YAML parsing.

## 5. Metadata blocks

### 5.1 Identity and lifecycle

Required fields:

- `id`: stable, repository-unique, lowercase kebab-case identifier.
- `title`: human-readable page title.
- `summary`: one sentence suitable for `index.md`.
- `type`: one value from the page-type table.
- `status`: lifecycle state.
- `created`: original creation date; never rewritten during ordinary updates.
- `updated`: date of the latest substantive content or metadata update.

Allowed `status` values:

- `inbox`: captured but not classified or checked;
- `draft`: structured but incomplete;
- `review`: ready for researcher review;
- `verified`: reviewed and adequately supported for its stated role;
- `superseded`: replaced by a newer page or decision;
- `archived`: retained only for historical context.

`verified` does not mean universally true. It means the page has met the
verification requirements appropriate to its role and clearly states its
limits.

Optional identity fields are `aliases`, `reviewed`, and `reviewers`.

### 5.2 Authorship

`authorship` records who produced the wiki text, not who originated the
underlying scientific idea.

Allowed `created_by` values:

- `researcher`: primarily written by the thesis author;
- `llm`: primarily drafted by an LLM and not yet collaboratively rewritten;
- `collaborative`: materially shaped by both researcher and LLM.

`contributors` is a list using the roles `researcher`, `llm`, `supervisor`, or
an explicit human/tool identifier when useful. Never infer that the researcher
authored LLM-generated prose merely because it is stored in this repository.

### 5.3 Epistemic status

`epistemic` records what kind of knowledge the page claims to contain.

Allowed `role` values:

- `prior-work`: primarily reports existing literature;
- `synthesis`: combines or interprets several inputs;
- `hypothesis`: proposes a testable claim not yet established;
- `empirical-finding`: interprets recorded project evidence;
- `decision`: records a chosen project or methodology direction;
- `mixed`: contains several roles that are distinguished within the body.

Allowed `confidence` values are `low`, `medium`, `high`, and `not-assessed`.
Confidence is an explicit judgment, not a substitute for citations.

Allowed `verification` values:

- `unverified`: not checked against an authoritative source or evidence;
- `source-checked`: claims checked against cited source locations;
- `experiment-backed`: claims tied to preserved experiment evidence;
- `reproduced`: result reproduced in a separate controlled execution;
- `supervisor-reviewed`: content or direction reviewed with the supervisor.

`verification` is a list because several checks may apply. `unverified` must
not appear with another verification value.

An experiment draft remains `unverified` until its referenced artifacts exist
and have been checked. Use `experiment-backed` only after the page identifies
the preserved configuration and result artifacts on which its claims depend.

### 5.4 Research scope

`scope` supports discovery without imposing a deep folder hierarchy.

- `topics`: open-ended lowercase kebab-case terms. Reuse existing terms before
  introducing synonyms.
- `granularities`: one or more controlled research levels.
- `pipeline_stages`: one or more controlled workflow stages.

Allowed `granularities`:

- `weight`
- `neuron`
- `mlp-block`
- `transformer-layer`
- `model`
- `moe`
- `cross-level`

Allowed `pipeline_stages`:

- `data`
- `screening`
- `selection`
- `replacement`
- `integration`
- `recovery`
- `evaluation`
- `analysis`
- `infrastructure`

### 5.5 Evidence links

`sources` links page claims to entries in `raw/sources.yml`.

Each source link contains:

- `source_id`: required registered source identifier;
- `locator`: required for a specific claim when the source supports stable
  locators such as page, section, theorem, figure, table, or code line;
- `relation`: how the source relates to the page;
- `note`: optional concise qualification.

Allowed `relation` values:

- `defines`
- `supports`
- `contradicts`
- `motivates`
- `extends`
- `implements`
- `evaluates`
- `contextualizes`

A bare URL in a wiki page is not a substitute for source registration.

### 5.6 Wiki relationships

- `related`: Obsidian wikilinks to conceptually related pages.
- `supersedes`: pages whose role this page explicitly replaces.
- `superseded_by`: page that replaced this page.

Relationships are not evidence citations. They provide graph navigation.
Whenever practical, important relationships should be reciprocal.

For implementation pages, `related` should connect the implemented methods,
governing decisions, and experiments that use the artifact. Explain the edge
semantics in the page body because ordinary Obsidian wikilinks are untyped.

## 6. Source registry

`raw/collections.yml` describes collections. It does not assert that individual
items were read. `raw/sources.yml` is the item-level registry used by citations.

A collection entry contains:

- `id`: unique lowercase kebab-case collection ID;
- `title`: human-readable collection name;
- `locations`: repository-root-relative paths or stable remote locations;
- `storage`: where the files are physically maintained;
- `origin`: default provenance for items in the collection;
- `evidence_kinds`: one or more default material kinds;
- `priority`: default thesis curation priority;
- `mutability`: whether underlying items can change;
- `ingestion_policy`: required review behavior; and
- `notes`: collection boundary or qualification.

Optional `include` and `recursive` fields constrain a directory collection.
`recursive` defaults to `false` when omitted. Item-level registration overrides
a collection default when the specific item differs.

Allowed collection `storage` values:

- `repository-tracked`
- `repository-local-ignored`
- `external-local`
- `remote`

Allowed collection `mutability` values:

- `immutable`
- `immutable-snapshot`
- `mutable`
- `append-only`
- `mixed`

Allowed `ingestion_policy` values:

- `one-at-a-time-reviewed`
- `selective`
- `selective-with-source-verification`
- `experiment-by-experiment`

Collection `origin`, `evidence_kinds`, and `priority` use the item-level
vocabularies below. Collection registration is an inventory operation, not a
claim that all contained items share equal relevance or quality.

An individual source entry follows this structure:

```yaml
- id: src-example-2026
  title: Full Source Title
  collection_id: primary-papers
  origin: supervisor-provided
  evidence_kind: scholarly-primary
  priority: core
  authors:
    - Author Name
  year: 2026
  canonical_url: https://example.org/source
  local_path: ../resources/direct/example.pdf
  sha256: lowercase-sha256-or-null
  added: 2026-07-17
  ingestion_status: registered
  notes: null
```

At least one of `canonical_url` or `local_path` is required. A local source
should receive a SHA-256 hash at registration. Mutable sources must be hashed
again when a new version is ingested. `local_path` is interpreted relative to
the repository root; do not store machine-specific absolute paths.

Allowed source `origin` values:

- `supervisor-provided`
- `researcher-found`
- `researcher-authored`
- `llm-generated`
- `collaborative`
- `project-generated`
- `external-other`
- `mixed`

Allowed `evidence_kind` values:

- `scholarly-primary`
- `scholarly-secondary`
- `book`
- `dataset`
- `software`
- `official-documentation`
- `web-reference`
- `source-list`
- `researcher-note`
- `llm-summary`
- `meeting-record`
- `project-decision`
- `experiment-log`

Allowed `priority` values:

- `core`: directly shapes the thesis scope or methodology;
- `supporting`: relevant evidence or comparison material;
- `background`: educational context rather than a central thesis source;
- `exploratory`: unvalidated lead, idea, or possible extension.

Allowed `ingestion_status` values are `registered`, `in-review`, `ingested`,
`rejected`, and `superseded`.

## 7. Claim-level provenance

Page-level metadata is insufficient for a mixed page. Apply these body rules:

1. Cite source-derived factual claims using a registered source ID and a
   locator where possible.
2. Label original propositions as **Researcher hypothesis.** until evidence
   changes their status.
3. Label an interpretation produced during synthesis as **Synthesis.** when a
   reader could otherwise mistake it for a source claim.
4. Label project results as **Empirical finding.** and link the experiment page
   or preserved log.
5. Label unresolved issues as **Open question.**
6. Use quotation marks and exact locators for direct quotations. Prefer
   paraphrase and never reproduce copyrighted material excessively.
7. A legacy LLM summary can guide discovery but cannot verify a scientific
   claim. Check the original source before setting `source-checked`.
8. Preserve meaningful disagreement. State which sources or experiments
   conflict and do not silently choose a winner.

## 8. Naming and linking

- Page filenames equal their stable `id` plus `.md`.
- IDs and topic terms use lowercase kebab-case ASCII.
- Use descriptive type prefixes, for example `concept-`, `entity-`, `method-`,
  `implementation-`, `rq-`, `hypothesis-`, `decision-`, `experiment-`,
  `finding-`, `comparison-`, and `synthesis-`.
- Individual source IDs use the `src-` prefix.
- Internal page links use `[[page-id]]` or `[[page-id|Readable label]]`.
- Repository paths use ordinary Markdown links when the target is not a wiki
  page.
- Do not rename an ID casually. A rename requires link updates and a log entry.

## 9. Required page body

After frontmatter, every page contains:

1. a concise overview;
2. the type-appropriate main content;
3. explicit evidence, limitations, or uncertainty;
4. a `## Relationships` section explaining important links; and
5. a `## Sources` section listing the source IDs used.

Source-summary pages additionally contain bibliographic identity, research
question, method, evidence, findings, limitations, relevance to this thesis,
and claims that require independent verification.

Experiment pages additionally contain objective, hypotheses, configuration,
inputs, procedure, metrics, direct results, interpretation, limitations,
artifact paths, and reproducibility status.

Implementation pages additionally contain purpose, implemented methods and
decisions, architecture and responsibilities, interfaces and data flow,
repository locations and entry points, version and maturity, validation,
limitations, and experiments that use the implementation. Repository links
identify artifacts; they are not evidence citations.

## 10. Operations

### Register

1. Identify the collection and source origin.
2. Add one item to `raw/sources.yml` with stable ID, path or URL, and hash.
3. Set `ingestion_status: registered`.
4. Do not create synthetic claims or mark the source as read.

### Ingest

1. Register the source if necessary and set it `in-review`.
2. Read the original source, including relevant figures or appendices.
3. Discuss key takeaways, uncertainties, and thesis relevance with the user.
4. Create or update one source-summary page.
5. Update existing concept, method, research, or synthesis pages affected by
   the evidence. Create new pages only for stable new objects.
6. Add source locators and explicit contradictions.
7. Set the source `ingested`, update `wiki/index.md`, and append `wiki/log.md`.
8. Report changed pages and unresolved questions.

### Query

1. Read `wiki/index.md` first.
2. Read relevant pages and follow their registered sources when precision is
   required.
3. Distinguish prior work, synthesis, hypotheses, and empirical findings.
4. Cite source IDs and state uncertainty in the answer.
5. File a useful new synthesis only when the user requests or approves it.

### Update

1. Preserve stable IDs and creation dates.
2. Add new evidence without erasing prior disagreement or limitations.
3. Update `updated`, index summaries, reciprocal links, and verification.
4. Use supersession metadata when a page or decision is genuinely replaced.
5. Append the operation to `wiki/log.md`.

### Distill

1. Select reviewed wiki pages appropriate for a human document.
2. Resolve or explicitly disclose remaining contradictions.
3. Write coherent prose in `docs/` with direct source traceability.
4. Link the output from the relevant wiki pages and operation log.
5. Do not replace the wiki pages with the distilled document.

### Lint

Check four levels:

- **Structural:** parseable YAML, required fields, allowed values, unique IDs,
  filename/ID agreement, and valid dates.
- **Graph:** broken links, missing reciprocal relationships, duplicates, and
  orphan pages.
- **Provenance:** unregistered sources, missing locators, unsupported claims,
  mixed authorship presented incorrectly, and invalid verification states.
- **Semantic:** contradictions, stale summaries, ambiguous terminology,
  superseded decisions, and gaps requiring user or supervisor judgment.

Mechanical errors can be proposed for direct repair. Ambiguous provenance or
meaning must be reported and clarified with the user before correction.

## 11. Special-file exceptions

The canonical page frontmatter does not apply to:

- `README.md`, `AGENTS.md`, and `SCHEMA.md`;
- `raw/collections.yml` and `raw/sources.yml`;
- `wiki/index.md` and `wiki/log.md`;
- files under `templates/`; and
- immutable files under `raw/legacy/`.

## 12. Schema evolution

Schema changes require:

1. a version increment when compatibility changes;
2. coordinated updates to templates and human metadata documentation;
3. a migration plan for existing pages;
4. a `schema` entry in `wiki/log.md`; and
5. user approval before bulk migration.
