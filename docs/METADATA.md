---
metadata_version: 1
title: Documentation Metadata Profile
type: metadata-reference
category: documentation
status: active
created: 2026-09-11
modified: 2026-09-11
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---

# Documentation Metadata Profile

## Purpose

Every repository-maintained Markdown document under `docs/` begins with a
small YAML front matter block. It identifies the document, its lifecycle, its
authorship, and whether the current text has been reviewed by a human.

Ignored personal material under `docs/vault/` and `docs/meetings/` is outside
this profile. Frozen non-Markdown prototype files also retain their original
formats.

The field meanings are inspired by
[Dublin Core Metadata Terms](https://www.dublincore.org/specifications/dublin-core/dcmi-terms/)
such as title, type, created, and modified. This is a project application
profile, not a claim of formal Dublin Core conformance. The profile is
intentionally simpler than a
[Frictionless Data Package](https://specs.frictionlessdata.io/data-package/)
because these files are human documents rather than packaged datasets.

## Required metadata

```yaml
---
metadata_version: 1
title: Operator Distillation Workflow
type: experiment-workflow
category: experiments/block
status: active
created: 2026-09-11
modified: 2026-09-11
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---
```

| Field | Meaning |
| --- | --- |
| `metadata_version` | Version of this project metadata profile |
| `title` | Human-readable document name |
| `type` | Kind of document |
| `category` | Repository subject area, using a slash-separated path |
| `status` | Current document lifecycle state |
| `created` | Date the document was first created |
| `modified` | Date of the latest intentional content or metadata change |
| `authorship.created_by` | Who originally produced the prose |
| `curation.status` | Whether the current text has been reviewed by a human |
| `curation.reviewed_by` | Reviewer identity or role, when applicable |
| `curation.reviewed_on` | Date of the applicable review |

Dates use ISO 8601 calendar form `YYYY-MM-DD`. Git history may be used to
recover creation dates for legacy documents, but front matter remains the
explicit record.

## Controlled values

Document types currently used by the repository are:

- `index`
- `thesis-scope`
- `experiment-workflow`
- `methodology`
- `evaluation-framework`
- `reporting-framework`
- `report`
- `architecture`
- `metadata-reference`
- `research-note`

Lifecycle status is one of:

- `draft`: incomplete or still changing materially;
- `active`: current maintained documentation;
- `superseded`: replaced by a newer document; or
- `archived`: preserved historical documentation.

Authorship is one of `researcher`, `llm`, `collaborative`, or `unknown`.
`unknown` is appropriate when migrating a legacy document whose prose origin
cannot be established reliably.

Curation status is one of:

- `unreviewed`: no human review of the current text is recorded;
- `researcher-reviewed`: reviewed by the thesis researcher; or
- `supervisor-reviewed`: reviewed by the thesis supervisor.

`reviewed_by` and `reviewed_on` remain `null` while status is `unreviewed`.
After a substantive change to reviewed text, reset curation to `unreviewed`
until the new version is checked again.

## Experiment sources

Experiment workflow documents additionally record their executable notebook
and existing result artifacts:

```yaml
sources:
  notebooks:
    - notebooks/block/operator-distillation.ipynb
  artifacts:
    - data/results/notebook-block-study/operator-distillation-v4.json
```

These paths establish traceability but do not prove that the prose is correct.
Human curation and empirical artifact availability remain separate facts. A
planned artifact that does not exist is described in the document body rather
than listed as an existing source. An existing artifact consumed as an input,
rather than produced by the documented workflow, may be listed separately
under `reference_artifacts`.

## Relationship to the LLM wiki

This profile applies only to `docs/`. Maintained pages under `llm-wiki/wiki/`
continue to follow `llm-wiki/SCHEMA.md` and the richer provenance model
described in [the LLM-wiki metadata reference](knowledge-base/metadata.md).
