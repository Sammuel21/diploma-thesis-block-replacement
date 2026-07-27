# AGENTS.md

## Scope and Authority

- These instructions apply to `llm-wiki/` and supplement the repository-root
  `AGENTS.md`; root approval and quality rules remain in force.
- `SCHEMA.md` is authoritative for page types, metadata, provenance, naming,
  required sections, and operation workflows. Do not duplicate those rules
  here.

## Required Context

- Read `SCHEMA.md` before modifying maintained wiki content.
- Start from `wiki/index.md` when locating or querying wiki knowledge.
- Read `docs/annotation.md` for thesis-facing methodology or scope decisions.
- Consult `raw/collections.yml` and `raw/sources.yml` when registering,
  ingesting, citing, or validating source provenance.

## Boundaries

- `raw/` is an immutable source boundary; corrections belong in a new source
  version or wiki annotation.
- `wiki/` is the maintained, collaborative research knowledge base.
- Update `docs/` only during an explicitly requested distillation.
- Treat `docs/prototype/mvp/` and `notebooks/mvp/` as historical evidence, not
  the current methodology specification.
- Do not modify code as part of a wiki operation unless explicitly approved.

## Evidence Discipline

- Separate prior work, researcher hypotheses, synthesis, project decisions,
  and empirical findings using the schema's epistemic labels.
- Support source-derived claims with individually registered original sources;
  collection entries and LLM summaries are discovery aids, not evidence.
- Link empirical claims to preserved experiment artifacts. Do not infer
  findings from planned or unexecuted experiments.
- Preserve uncertainty, contradictions, negative results, and historical
  context. Ask the researcher when provenance or intended meaning is unclear.
- Do not ingest sources, promote wiki material, or distill documentation merely
  because relevant files are available; perform the requested operation only.

## Working Behavior

- Follow the applicable workflow in `SCHEMA.md` for register, ingest, query,
  update, lint, or distill operations.
- Prefer updating an existing page over creating a near-duplicate, and preserve
  stable page IDs and unrelated content.
- Keep mechanical lint distinct from scientific or provenance review; lint
  cannot establish that a claim is correct.

## Completion Checklist

For every wiki-changing operation:

1. validate metadata and controlled values;
2. update substantive page dates, index entries, and important reciprocal
   links where applicable;
3. append a concise entry to `wiki/log.md`;
4. report changed files, evidence used, and unresolved issues; and
5. leave unrelated code and historical artifacts untouched.
