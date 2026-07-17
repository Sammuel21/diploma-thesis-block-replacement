# AGENTS.md

## Scope

These instructions apply to `llm-wiki/` and supplement the repository-root
`AGENTS.md`. Root approval, quality, and thesis-scope rules remain in force.

## Required context

- Read `SCHEMA.md` before creating or modifying maintained wiki content.
- Read `wiki/index.md` before querying or deciding whether a page exists.
- Consult `raw/collections.yml` and `raw/sources.yml` for source identity.
- Treat `docs/annotation.md` as the baseline thesis scope.

## Ownership boundaries

- `raw/` is a source boundary. Never rewrite source content or legacy inputs.
- `wiki/` is collaboratively maintained by the researcher and LLM.
- `docs/` is human-oriented distilled documentation and is outside this
  scoped tree; update it only during an explicitly requested distillation.
- `SCHEMA.md` is authoritative for page structure and workflow.

## Provenance behavior

- Never present researcher-authored ideas as prior work.
- Never present LLM summaries or inferences as primary evidence.
- Never promote a collection entry into a citation without individually
  registering the source in `raw/sources.yml`.
- Check source-derived claims against the original source before marking them
  `source-checked`.
- Tie empirical findings to experiment pages and preserved artifacts.
- Preserve contradictions, uncertainty, and negative results.
- Ask the user when authorship, source identity, intended meaning, or claim
  status cannot be established from available material.

## Operations

### Register

Register source identity, origin, evidence kind, priority, path or URL, and hash
without implying that the source has been read.

### Ingest

Process sources one at a time unless the user explicitly approves a batch.
Discuss takeaways and ambiguous interpretations before finalizing integration.
Update the source summary, affected knowledge pages, `wiki/index.md`, source
status, and `wiki/log.md` as one coherent operation.

### Query

Start from `wiki/index.md`, follow relevant pages and sources, and distinguish
prior work, synthesis, hypotheses, decisions, and empirical findings in the
answer. Do not file chat output automatically.

### Update

Prefer updating an existing page over creating a near-duplicate. Preserve page
IDs, disagreement, and history. Use explicit supersession instead of silently
replacing an older conclusion.

### Lint

Separate mechanical findings from epistemic findings. Mechanical fixes may be
proposed together. Provenance conflicts and ambiguous meaning require user
review. Do not claim that lint proves scientific correctness.

### Distill

Promote only reviewed material into `docs/`. Retain source traceability and
link the human document back to its wiki inputs.

## Completion requirements

For every wiki-changing operation:

1. validate required metadata and controlled values;
2. update `updated` dates on substantively changed pages;
3. maintain index entries and important reciprocal links;
4. append a concise operation-log entry;
5. report changed files, evidence used, and unresolved issues; and
6. do not modify unrelated code or historical prototype artifacts.
