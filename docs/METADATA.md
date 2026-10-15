---
title: Documentation Metadata Profile
status: active
modified: 2026-10-15
---

# Documentation Metadata Profile

## Purpose

Every repository-maintained Markdown document under `docs/` begins with a
small YAML front matter block. It names the document, states whether it is
current, and, for experiment documents, records the evidence behind it.
Obsidian shows the block as "Properties"; no repository code reads it.

Ignored personal material under `docs/vault/` and `docs/meetings/` is outside
this profile. Frozen non-Markdown prototype files also retain their original
formats. Git history records who changed a document and when it was created.

## Required metadata

```yaml
---
title: Operator Distillation Workflow
status: active
modified: 2026-09-11
---
```

| Field | Meaning |
| --- | --- |
| `title` | Human-readable document name |
| `status` | Current document lifecycle state |
| `modified` | Date of the latest intentional content change, as `YYYY-MM-DD` |

Lifecycle status is one of:

- `draft`: incomplete or still changing materially;
- `active`: current maintained documentation;
- `superseded`: replaced by a newer document; or
- `archived`: preserved historical documentation.

## Experiment sources

Experiment documents additionally record their executable notebook and
existing result artifacts:

```yaml
sources:
  notebooks:
    - notebooks/block/operator/operator-distillation.ipynb
  artifacts:
    - data/results/notebook-block-study/operator-distillation-v4.json
```

These paths establish traceability but do not prove that the prose is correct.
A planned artifact that does not exist is described in the document body rather
than listed as an existing source. An existing artifact consumed as an input,
rather than produced by the documented workflow, may be listed separately
under `reference_artifacts`.

## Relationship to the LLM wiki

This profile applies only to `docs/`. Maintained pages under `llm-wiki/wiki/`
follow `llm-wiki/SCHEMA.md` and the provenance model described in
[the LLM-wiki metadata reference](knowledge-base/metadata.md).
