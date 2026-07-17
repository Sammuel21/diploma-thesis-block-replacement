# LLM-Wiki Metadata Reference

This document explains the YAML metadata used by maintained pages under
`llm-wiki/wiki/`. The authoritative specification is
`llm-wiki/SCHEMA.md`; this guide is optimized for readers and page authors.

## Why metadata is divided into blocks

Academic provenance requires several independent questions:

- What page is this and where is it in its lifecycle?
- Who wrote the wiki text?
- What kind of knowledge does it claim to contain?
- How was it verified?
- Which research areas does it concern?
- Which original sources support it?
- Which other wiki pages are related?

Combining these questions into a single `type`, `source`, or `confidence` field
would blur important distinctions. For example, a supervisor-provided paper is
high priority for the thesis, but that says nothing about whether a particular
wiki claim was checked against it.

## Complete example

```yaml
---
id: method-block-importance
title: Block Importance
summary: Measures the representational change associated with an MLP block.
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

## Identity and lifecycle

| Field | Meaning |
| --- | --- |
| `id` | Stable unique ID; also the Markdown filename |
| `title` | Human-readable page title |
| `summary` | One sentence used by `wiki/index.md` |
| `type` | Research object represented by the page |
| `status` | Current lifecycle stage |
| `created` | Original creation date |
| `updated` | Last substantive update date |

Allowed page types are:

- `source-summary`
- `concept`
- `entity`
- `method`
- `research-question`
- `hypothesis`
- `decision`
- `experiment`
- `finding`
- `comparison`
- `synthesis`

Allowed lifecycle states are:

- `inbox`: captured but not checked;
- `draft`: structured but incomplete;
- `review`: ready for researcher review;
- `verified`: reviewed and adequately supported for its stated role;
- `superseded`: replaced by another page or decision;
- `archived`: historical only.

Lifecycle status and scientific confidence are separate. A hypothesis can be a
well-maintained `verified` page while the hypothesis itself remains uncertain.

## Authorship

```yaml
authorship:
  created_by: collaborative
  contributors:
    - researcher
    - llm
```

`created_by` describes who produced the wiki prose:

- `researcher`
- `llm`
- `collaborative`

This does not describe the origin of a scientific idea. A page written by an
LLM can faithfully summarize prior work; a page written by the researcher can
still contain source-derived claims.

## Epistemic status

```yaml
epistemic:
  role: synthesis
  confidence: medium
  verification:
    - source-checked
```

`role` describes what kind of knowledge the page contains:

- `prior-work`
- `synthesis`
- `hypothesis`
- `empirical-finding`
- `decision`
- `mixed`

`confidence` is `low`, `medium`, `high`, or `not-assessed`. It records a
judgment and never replaces evidence.

`verification` is a list:

- `unverified`
- `source-checked`
- `experiment-backed`
- `reproduced`
- `supervisor-reviewed`

`unverified` cannot be combined with another verification value.

An experiment plan starts as `unverified`. It becomes `experiment-backed` only
after the recorded configuration and result artifacts exist and are linked
from the page.

## Research scope

```yaml
scope:
  topics:
    - block-importance
  granularities:
    - mlp-block
  pipeline_stages:
    - screening
```

`topics` is an extensible list of reused kebab-case terms.

Controlled granularities are `weight`, `neuron`, `mlp-block`,
`transformer-layer`, `model`, `moe`, and `cross-level`.

Controlled pipeline stages are `data`, `screening`, `selection`,
`replacement`, `integration`, `recovery`, `evaluation`, `analysis`, and
`infrastructure`.

These dimensions overlap intentionally. A BI method can concern an MLP block
as the scored object and model-level behavior as the downstream consequence.

## Evidence links

```yaml
sources:
  - source_id: src-example-2026
    locator: "Section 3.2, Figure 1"
    relation: supports
```

Every `source_id` must exist in `llm-wiki/raw/sources.yml`. A source relation is
one of `defines`, `supports`, `contradicts`, `motivates`, `extends`,
`implements`, `evaluates`, or `contextualizes`.

Locators should be precise enough for another reader to verify the claim.
Useful locators include page, section, equation, theorem, figure, table,
appendix, code symbol, or experiment artifact path.

## Relationships

```yaml
related:
  - "[[concept-block-replaceability]]"
supersedes: []
superseded_by: []
```

Relationships connect wiki objects; they are not citations. `supersedes` and
`superseded_by` preserve the history of changing decisions and understanding.

## Source collection metadata

Source collections and wiki pages use related but different metadata.

| Dimension | Question | Example |
| --- | --- | --- |
| `origin` | How did this material enter the project? | `supervisor-provided` |
| `evidence_kind` | What kind of material is it academically? | `scholarly-primary` |
| `priority` | How central is it to the thesis? | `core` |
| `mutability` | Can the underlying item change? | `immutable` |
| `ingestion_status` | Has this exact item been processed? | `registered` |

These fields prevent common category errors. `researcher-authored` describes
origin, `researcher-note` describes evidence kind, and `exploratory` describes
priority. None implies that the idea already exists in prior literature.

Collection `storage` is one of `repository-tracked`,
`repository-local-ignored`, `external-local`, or `remote`. Collection
`mutability` is `immutable`, `immutable-snapshot`, `mutable`, `append-only`, or
`mixed`. These operational fields determine how an agent may access and record
the source; they do not determine scientific quality.

## Claim-level provenance

Page metadata cannot disambiguate every sentence on a mixed page. Use explicit
body labels when needed:

- **Researcher hypothesis.** for an original testable proposition.
- **Synthesis.** for an interpretation assembled across inputs.
- **Empirical finding.** for an interpretation tied to preserved results.
- **Open question.** for an unresolved issue.

Source-derived claims still require registered source IDs and locators. Legacy
LLM summaries can help locate relevant material but cannot verify a claim
without checking the original source.

## Minimum review checklist

Before moving a page to `review` or `verified`, check:

1. filename and `id` agree;
2. required metadata is present and uses allowed values;
3. authorship and epistemic role are not conflated;
4. every cited source is individually registered;
5. important claims have useful locators;
6. original hypotheses and synthesis are visibly distinguished;
7. limitations and contradictory evidence are retained;
8. index, relationships, and operation log are updated.
