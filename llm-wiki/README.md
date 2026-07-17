# Thesis LLM Wiki

This directory is the compounding research knowledge base for the diploma
thesis. It is optimized for disciplined collaboration between the researcher
and an LLM while remaining readable in ordinary Markdown and Obsidian.

## Layers

1. `raw/` registers immutable source material and preserved input notes. Raw
   material is evidence or migration input; it is never silently rewritten.
2. `wiki/` contains maintained source summaries, concepts, methods,
   implementations, research objects, experiments, and syntheses.
3. `SCHEMA.md` defines the canonical structure, metadata, provenance rules, and
   workflows. `AGENTS.md` applies those rules to agent behavior in this tree.

Human-oriented, reviewed explanations live separately in `docs/`. Promotion
from the wiki into `docs/` is an explicit distillation operation.

## Start here

- Read [SCHEMA.md](SCHEMA.md) before creating or changing wiki pages.
- Use [raw/collections.yml](raw/collections.yml) to discover source libraries.
- Use [raw/sources.yml](raw/sources.yml) for individually registered sources.
- Use [wiki/index.md](wiki/index.md) to navigate maintained knowledge.
- Use [wiki/log.md](wiki/log.md) to inspect wiki operations chronologically.
- Use [templates/](templates/) when creating pages.

## Operations

- **Register:** describe a source collection or individual source without yet
  synthesizing it.
- **Ingest:** process one source, discuss it, and integrate it into the wiki.
- **Query:** answer from maintained wiki knowledge while retaining source
  traceability.
- **Update:** incorporate new evidence without silently erasing disagreement.
- **Lint:** check structure, links, provenance, contradictions, and staleness.
- **Distill:** turn reviewed wiki material into coherent human documentation.

The first controlled pilot begins with individually reviewed source ingestion.
Implementation nodes are added only for stable architectural components, not
for every source file or function.
