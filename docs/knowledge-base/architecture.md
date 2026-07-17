# LLM-Wiki Architecture

## Purpose

The thesis repository uses a persistent LLM wiki to accumulate research
knowledge without confusing raw evidence, working synthesis, historical
prototype material, and finished human documentation.

The architecture adapts the three-layer LLM-wiki pattern to academic work:

1. raw sources provide traceable inputs;
2. the wiki integrates knowledge incrementally; and
3. the schema governs how that integration happens.

This repository adds a separate `docs/` publication layer and a frozen MVP
archive because research continuity and implementation history have different
lifecycles from the knowledge graph.

## System map

```text
External diploma workspace
|-- resources/direct/       supervisor-provided core papers
|-- resources/              supporting papers
`-- ideas/                  researcher-authored idea inbox
            |
            | registered by path, origin, priority, and evidence kind
            v
development/llm-wiki/raw/
|-- collections.yml         collection-level catalog
|-- sources.yml             item-level citation registry
`-- legacy/                 immutable snapshots of pre-wiki notes
            |
            | controlled one-source ingestion
            v
development/llm-wiki/wiki/
|-- sources/                source summaries
|-- concepts/               stable terminology and objects
|-- entities/               models, datasets, software, and named artifacts
|-- methods/                metrics, architectures, and procedures
|-- implementations/        project software components and pipelines
|-- research/               questions, hypotheses, and decisions
|-- experiments/            designs, results, and findings
`-- syntheses/              comparisons and integrated analyses
            |
            | explicit review and distillation
            v
development/docs/           coherent documentation for human readers
```

## Responsibilities

| Layer | Primary content | Mutation policy | Evidentiary role |
| --- | --- | --- | --- |
| External source library | PDFs, books, raw ideas | LLM read-only | Original evidence or input |
| `llm-wiki/raw/` | Registries and legacy snapshots | Append/register; do not rewrite sources | Provenance boundary |
| `llm-wiki/wiki/` | Maintained Markdown knowledge | Collaborative, schema-governed | Synthesis and research memory |
| `docs/` | Distilled explanations | Deliberate human-facing edits | Communication, not replacement evidence |
| `docs/prototype/mvp/` | Frozen prototype record | Historical corrections only | Implementation and empirical history |

## Why sources can remain outside Git

The source library contains large PDFs and books and already exists alongside
the development repository. Copying it into Git would duplicate storage and
make repository history expensive. Instead:

- `raw/collections.yml` records the collection location and default treatment;
- `raw/sources.yml` records each source used by the wiki;
- local files receive SHA-256 hashes during individual registration; and
- public sources retain canonical URLs when available.

This means a Git clone contains the knowledge base and source identities, but
not necessarily every source binary. Full offline reconstruction also requires
the external source library or retrieval through recorded canonical URLs.

## Collection registration versus ingestion

Registration answers: *What material exists, where is it, who supplied or
created it, and how should it be treated?*

Ingestion answers: *What does one source say, how does it affect current
knowledge, and which exact claims does it support or contradict?*

The distinction prevents a folder of PDFs from being treated as read merely
because the agent can see it. Initial ingestion remains one source at a time so
the researcher can verify emphasis, interpretation, and provenance.

## Navigation and history

`wiki/index.md` is content-oriented. It maps stable knowledge pages and their
one-line summaries. Agents read it before searching the graph.

`wiki/log.md` is chronological and append-only. It records registration,
ingestion, substantive updates, lint passes, distillation, and schema changes.
Git records file history; the operation log records research intent.

## Governance files

The root `AGENTS.md` defines repository-wide behavior, approval rules, thesis
context, and source hierarchy. `llm-wiki/AGENTS.md` adds wiki-specific
workflows within its directory. More specific instructions do not duplicate
the global policy.

`llm-wiki/SCHEMA.md` is the canonical data and workflow specification. It is a
normal Markdown file, so the scoped `AGENTS.md` explicitly requires agents to
read it.

## MVP boundary

The completed MVP remains executable at `notebooks/mvp/`, `scripts/intro/`,
and `configs/intro_config.py`. Its evidence is preserved under
`data/mvp/results/logs/`, with a manifest in `docs/prototype/mvp/`.

Future maintained implementation belongs under a separately reviewed `src/`
and `pipelines/` architecture. The documentation restructuring does not imply
that the MVP code has already been productionized.

Stable implementations also receive wiki graph nodes. These nodes explain
which methods and decisions a component implements, where its code and
configuration live, and which experiments used it. They operate at component
or pipeline granularity; individual files and functions remain ordinary code
references unless they have an independently meaningful architectural role.

## Wiki lifecycle

1. **Register:** add a source identity without making knowledge claims.
2. **Ingest:** integrate one source into summaries and existing pages.
3. **Query:** navigate from the index and answer with provenance.
4. **Update:** incorporate evidence while preserving disagreement and history.
5. **Lint:** check structure, graph health, provenance, and semantic drift.
6. **Distill:** publish reviewed knowledge into `docs/` for human readers.

The first pilot should ingest one core paper, one supporting paper, and one
researcher idea. The schema should be revised from that experience before any
bulk migration.

## Design references

- Andrej Karpathy, [LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f),
  provides the raw/wiki/schema pattern and compounding-artifact model.
- Starmorph, [How to Build Karpathy's LLM Wiki](https://blog.starmorph.com/blog/karpathy-llm-wiki-knowledge-base-guide),
  provides a useful starting point for page types, frontmatter, index, log, and
  maintenance workflows.
- OpenAI, [Introducing Codex](https://openai.com/index/introducing-codex/),
  documents directory-scoped and nested `AGENTS.md` behavior.

The thesis implementation extends these patterns with academic provenance,
claim-level attribution, experiment evidence, researcher hypotheses, and a
separate human-documentation layer.
