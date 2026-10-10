---
id: implementation-maintained-mlp-replacement-package
title: Maintained MLP Replacement Package
summary: Maps reusable scientific code to maintained runners, completed homogeneous SwiGLU execution, and executor-specific jobs without generalizing their validation.
type: implementation
status: draft
created: 2026-08-28
updated: 2026-10-02

authorship:
  created_by: collaborative
  contributors:
    - researcher
    - llm

epistemic:
  role: mixed
  confidence: high
  verification:
    - experiment-backed

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
  - "[[method-discrete-width-curve-allocation]]"
  - "[[synthesis-homogeneous-swiglu-findings]]"
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

| Area | Representative modules | Responsibility |
| --- | --- | --- |
| `operators/` | `modules.py`, `training.py`, `baselines.py` | Replacement architectures, local fitting, and reusable single-block operator baselines |
| `compression/` | `allocation.py`, `selection.py`, `surgery.py`, `recovery.py`, `continuation.py`, `reconstruction.py`, `teacher_cache.py`, `adapters.py`, `quantization.py` | Width budgets, model mutation, recovery/checkpoint contracts, state reconstruction, teacher targets, LoRA, and library-backed quantization |
| `analysis/` | `activations.py`, `screening.py`, `sensitivity.py`, `interactions.py` | Activation geometry, BI-style screening, isolated replacement sensitivity, and multi-block interaction helpers |
| `evaluation/` | `operator.py`, `language_model.py`, `footprint.py`, `final_quality.py`, `bundles.py`, `mixed_precision.py`, `generation.py`, `quantized_bundles.py` | Local fidelity, pinned likelihood/tasks, footprint, inference bundles, dtype handling, and controlled generation measurements |
| Package root | `config.py`, `data.py`, `data_streams.py`, `model.py`, `capture.py`, `artifacts.py`, `runlog.py` | Configuration, finite streams, model loading, activation capture, artifact hashes, and JSON run records |

`analysis/interactions.py` is preserved as provisional interaction-analysis
support, but no maintained caller currently depends on it.

## Interfaces and Data Flow

The maintained flow is:

1. Root configuration, model, and data modules prepare the experiment.
2. `analysis/screening.py` may produce model- or MLP-level screening scores.
3. `compression/selection.py` resolves the selected layer indices.
4. `capture.py` and `operators/training.py` fit local replacement operators.
5. `compression/surgery.py` integrates the fitted modules.
6. Recovery helpers and the selected runner apply the declared teacher-logit
   objective, trainable scope, stream, and checkpoint protocol.
7. `evaluation/` records quality and footprint outcomes.
8. Maintained workflow runners coordinate experiment-specific phases; the
   older `compression/workflows.py` also provides reusable replacement flows.

These are responsibilities, not one universal job sequence. Curve-based
allocation fits candidate widths before choosing the joint topology.
SwiGLU-7's shared preparation plus training array is experiment-specific;
future jobs need not inherit that split.

Notebooks remain responsible for research narrative, configuration choices,
visualization, and experiment-specific result tables. Shared domain operations
belong in `src/` only after they have a stable reusable role. Maintained Python
process entry points and executor-specific submission files belong in the
top-level `workflows/` orchestration layer.

## Repository Locations and Entry Points

- Maintained package: [`src/mlp_replacement/`](../../../src/mlp_replacement/)
- Maintained single-experiment runner:
  [`workflows/runs/run_experiment.py`](../../../workflows/runs/run_experiment.py)
- TUKE Perun submission files:
  [`workflows/jobs/perun/`](../../../workflows/jobs/perun/)
- Homogeneous experiment runners:
  [`workflows/runs/model/swiglu/`](../../../workflows/runs/model/swiglu/)
- Completed independent SwiGLU-7 runner and dedicated job:
  [`swiglu_7.py`](../../../workflows/runs/model/swiglu/swiglu_7.py),
  [`swiglu_7.sbatch`](../../../workflows/jobs/perun/swiglu_7.sbatch)
- Block notebooks: [`notebooks/block/`](../../../notebooks/block/)
- Model notebooks: [`notebooks/model/`](../../../notebooks/model/)

A future reusable model-compression baseline orchestrator belongs at
`src/mlp_replacement/compression/baselines.py`. No such module is claimed to be
implemented by this structural migration.

## Version and Maturity

The package and workflow layer remain working research software. The original
migration established responsibility boundaries; later maintained runners
implemented the homogeneous SwiGLU progression. SwiGLU-7 has completed
production execution on PERUN. Quantization-1 code is present, but its
scientific GPU validation/run remains pending. These maturity statements do
not certify every module or launcher.

## Validation

Historical source/workflow migration checks concerned imports, configuration,
shell syntax, and repository links, not scientific correctness.

**Observed execution evidence, checked 2026-10-02.** The preserved SwiGLU-7
preparation and all 12 production records are completed, including final
evaluation. [[synthesis-homogeneous-swiglu-findings]] links their artifacts and
limits. The dedicated job stages code/inputs to manual SCRATCH, runs there,
and stages durable outputs to PROJECT. This evidence establishes that this
workflow executed; it is not an independent reproduction or validation of all
failure/restart paths.

## Limitations

- The architecture groups current responsibilities but does not define a final
  public API.
- Notebook imports remain coupled to maintained module paths and must be linted
  when modules move.
- The dedicated SwiGLU-7 job uses the verified manual-scratch contract. The
  generic `run_model.sbatch` still depends on `.activate_scratch`; its deployment
  readiness must not be inferred from the completed dedicated job.
- Exact continuation needs complete durable states, not the reduced analysis
  export. Hard node loss or an uncatchable kill can bypass final stage-out;
  directory/lock recovery remains a separate operational concern.
- `analysis/interactions.py` remains provisional until the multi-block study
  adopts or replaces it.

## Experiments Using This Implementation

- [[experiment-initial-block-compression-study]] owns the working block and
  model notebook progression.
- [[experiment-baseline-operator-analysis]] uses operator, sensitivity,
  recovery, and evaluation components.
- [[experiment-swiglu-operator-design-progression]] uses the generic operator
  definitions and isolated sensitivity evaluation.
- [[synthesis-homogeneous-swiglu-findings]] records completed model-level
  allocation/recovery studies using maintained runners and the dedicated job.

## Relationships

- [[decision-working-experiment-code-standards]] governs when notebook logic
  should be promoted into this maintained package.
- [[experiment-initial-block-compression-study]] provides the current
  experiments whose reusable operations motivated these responsibility
  boundaries.
- [[method-discrete-width-curve-allocation]] is implemented by
  `compression/allocation.py`; experiment-specific fitting and orchestration
  remain in the SwiGLU runners.
- [[synthesis-homogeneous-swiglu-findings]] records the bounded production
  evidence and routes to detailed reports without duplicating them here.

## Sources

No registered literature source is cited. This page records a project software
architecture decision and observed repository structure, inspected 2026-10-02.
The linked family synthesis supplies the original experiment evidence and
literature attribution; implementation presence alone is not a scientific
finding.
