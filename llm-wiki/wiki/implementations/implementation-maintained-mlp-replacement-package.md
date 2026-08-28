---
id: implementation-maintained-mlp-replacement-package
title: Maintained MLP Replacement Package
summary: Defines the responsibility boundaries of the maintained operator, compression, analysis, and evaluation code packages.
type: implementation
status: draft
created: 2026-08-28
updated: 2026-08-28

authorship:
  created_by: collaborative
  contributors:
    - researcher
    - llm

epistemic:
  role: decision
  confidence: high
  verification:
    - unverified

scope:
  topics:
    - implementation-scope
    - mlp-block-replacement
    - software-architecture
  granularities:
    - mlp-block
    - model
    - cross-level
  pipeline_stages:
    - screening
    - selection
    - replacement
    - integration
    - recovery
    - evaluation
    - analysis
    - infrastructure

sources: []
related:
  - "[[decision-working-experiment-code-standards]]"
  - "[[experiment-initial-block-compression-study]]"
supersedes: []
superseded_by: []
---

# Maintained MLP Replacement Package

## Purpose

This page records the project-defined responsibility boundaries inside
[`src/mlp_replacement/`](../../../src/mlp_replacement/). The structure separates
local replacement-operator work from model-level compression, diagnostic
analysis, and evaluation without introducing a package for every individual
experiment.

## Implemented Methods and Decisions

**Project decision.** Reusable code is grouped by the research operation it
performs:

- `operators/` defines and locally fits drop-in MLP replacements;
- `compression/` selects, integrates, and recovers replacements at model level;
- `analysis/` computes descriptive, screening, sensitivity, and interaction
  diagnostics; and
- `evaluation/` measures operator quality, language-model quality, footprint,
  and optional benchmark results.

Cross-cutting configuration, data, model-discovery, activation-capture, and run
logging modules remain at the package root. This is a maintainability decision,
not a scientific claim about the superiority of this organization.

## Architecture and Responsibilities

| Area | Current modules | Responsibility |
| --- | --- | --- |
| `operators/` | `modules.py`, `training.py`, `baselines.py` | Replacement architectures, local fitting, and reusable single-block operator baselines |
| `compression/` | `selection.py`, `surgery.py`, `recovery.py`, `workflows.py` | Layer policy, model mutation, replacement-only KL recovery, and end-to-end replacement workflows |
| `analysis/` | `activations.py`, `screening.py`, `sensitivity.py`, `interactions.py` | Activation geometry, BI-style screening, isolated replacement sensitivity, and multi-block interaction helpers |
| `evaluation/` | `operator.py`, `language_model.py`, `footprint.py`, `benchmarks.py` | Local reconstruction metrics, loss/PPL, structural and serialized footprint, and optional downstream benchmarks |
| Package root | `config.py`, `data.py`, `model.py`, `capture.py`, `runlog.py` | Shared configuration, datasets, model discovery/loading, activation capture, and JSON run records |

`analysis/interactions.py` is preserved as provisional interaction-analysis
support, but no maintained caller currently depends on it.

## Interfaces and Data Flow

The maintained flow is:

1. Root configuration, model, and data modules prepare the experiment.
2. `analysis/screening.py` may produce model- or MLP-level screening scores.
3. `compression/selection.py` resolves the selected layer indices.
4. `capture.py` and `operators/training.py` fit local replacement operators.
5. `compression/surgery.py` integrates the fitted modules.
6. `compression/recovery.py` performs replacement-only teacher-logit recovery.
7. `evaluation/` records quality and footprint outcomes.
8. `compression/workflows.py` coordinates these operations for maintained runs.

Notebooks remain responsible for research narrative, configuration choices,
visualization, and experiment-specific result tables. Shared domain operations
belong in `src/` only after they have a stable reusable role.

## Repository Locations and Entry Points

- Maintained package: [`src/mlp_replacement/`](../../../src/mlp_replacement/)
- Maintained single-experiment runner:
  [`pipelines/run_experiment.py`](../../../pipelines/run_experiment.py)
- Block notebooks: [`notebooks/block/`](../../../notebooks/block/)
- Model notebooks: [`notebooks/model/`](../../../notebooks/model/)

A future reusable model-compression baseline orchestrator belongs at
`src/mlp_replacement/compression/baselines.py`. No such module is claimed to be
implemented by this structural migration.

## Version and Maturity

The package is working research software. This restructuring changes module
ownership and import paths without intentionally changing algorithmic behavior.
The interfaces may still evolve as multi-block compression and allocation
experiments become concrete.

## Validation

Migration validation covers Python import resolution, notebook JSON structure,
remaining obsolete import paths, and wiki structure and links. It does not
constitute an experiment rerun or empirical validation of compression quality.

## Limitations

- The architecture groups current responsibilities but does not define a final
  public API.
- Notebook imports remain coupled to maintained module paths and must be linted
  when modules move.
- Package organization does not solve experiment caching, artifact design, or
  model-checkpoint persistence.
- `analysis/interactions.py` remains provisional until the multi-block study
  adopts or replaces it.

## Experiments Using This Implementation

- [[experiment-initial-block-compression-study]] owns the working block and
  model notebook progression.
- [[experiment-baseline-operator-analysis]] uses operator, sensitivity,
  recovery, and evaluation components.
- [[experiment-swiglu-operator-design-progression]] uses the generic operator
  definitions and isolated sensitivity evaluation.

## Relationships

- [[decision-working-experiment-code-standards]] governs when notebook logic
  should be promoted into this maintained package.
- [[experiment-initial-block-compression-study]] provides the current
  experiments whose reusable operations motivated these responsibility
  boundaries.

## Sources

No registered literature source is cited. This page records a project software
architecture decision and observed repository structure.
