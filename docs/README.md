# Documentation

This directory contains documentation intended primarily for human readers.
It is deliberately separate from `llm-wiki/`, which is the incremental,
LLM-maintained research knowledge base.

## Core scope

- [Thesis annotation](annotation.md) defines the baseline problem and goals.

## Knowledge-base documentation

- [Architecture](knowledge-base/architecture.md) explains the repository's
  raw, wiki, and human-documentation layers.
- [Metadata reference](knowledge-base/metadata.md) explains the YAML
  frontmatter, provenance model, and allowed values.

## Methodology

- [Model compression evaluation framework](methodology/evaluation-framework.md)
  defines the current footprint, memory, quality, benchmark, and reporting
  contract.

## Historical prototype

- [MVP archive](prototype/mvp/README.md) records the completed prototype,
  executable artifact locations, experiment evidence, and limitations.

## Publication rule

Material enters `docs/` only when it has been deliberately distilled for a
human audience. Working ideas, source summaries, hypotheses, and evolving
syntheses belong in `llm-wiki/`. Historical material belongs under
`docs/prototype/` and must be clearly identified as historical rather than
current methodology.
