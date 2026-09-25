---
metadata_version: 1
title: Documentation
type: index
category: documentation
status: active
created: 2026-07-17
modified: 2026-09-25
authorship:
  created_by: unknown
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---

# Documentation

This directory contains documentation intended primarily for human readers.
It is deliberately separate from `llm-wiki/`, which is the incremental,
LLM-maintained research knowledge base.

Markdown files in this directory follow the
[documentation metadata profile](METADATA.md).

## Core scope

- [Thesis annotation](annotation.md) defines the baseline problem and goals.

## Agent implementation standards

- [Notebook implementation standard](agents/notebook-implementation.md)
  defines the researcher-authored Markdown boundary, observed notebook style,
  proportional verification, and implementation-model suitability.
- [Maintained code and workflow implementation standard](agents/maintained-code-and-workflows.md)
  defines source-package boundaries, portable execution, artifact and resume
  reliability, proportional verification, and implementation-model suitability.

## Knowledge-base documentation

- [Architecture](knowledge-base/architecture.md) explains the repository's
  raw, wiki, and human-documentation layers.
- [LLM-wiki metadata reference](knowledge-base/metadata.md) explains the
  richer provenance model used by maintained wiki pages.

## Methodology

- [Model compression evaluation framework](methodology/evaluation-framework.md)
  defines the current footprint, memory, quality, benchmark, and reporting
  contract.
- [Global-to-local MLP operator budget allocation](methodology/global-to-local-operator-budget-allocation.md)
  summarizes the numbered procedure for converting one model-level parameter
  target into block-specific replacement caps and reconciling unused budget.
- [Operator calibration data and training budget](methodology/operator-calibration-data-and-training-budget.md)
  explains activation-pair data feeding and distinguishes fixed-update,
  fixed-epoch, and operator-batch-size experiments.

## Infrastructure

- [TUKE Perun workflow infrastructure](infrastructure/perun.md) records the
  stable access, Slurm, storage, environment, scratch, and artifact contracts
  for unattended Perun workflows.
- [TUKE Perun project status](infrastructure/perun-status.md) tracks current
  readiness, available capacity, cumulative consumption, and open questions.
- [TUKE Perun experiment log](infrastructure/perun-log.md) is the append-only
  record of submitted jobs, allocations, outcomes, resource measurements, and
  artifact locations.

## Experiment workflows

- [Model compression experiments](experiments/model/README.md) explain the
  executable compression baseline, SwiGLU allocation, and allocation-search
  pipelines.
- [Block experiments](experiments/block/README.md) explain the isolated
  operator-fitting and singleton-replacement studies.

## Historical prototype

- [MVP archive](prototype/mvp/README.md) records the completed prototype,
  executable artifact locations, experiment evidence, and limitations.

## Publication rule

Material enters `docs/` only when it has been deliberately distilled for a
human audience. Working ideas, source summaries, hypotheses, and evolving
syntheses belong in `llm-wiki/`. Historical material belongs under
`docs/prototype/` and must be clearly identified as historical rather than
current methodology.
