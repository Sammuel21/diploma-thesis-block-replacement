---
id: experiment-baseline-operator-analysis
title: Baseline Operator Analysis
summary: Records the artifact-backed calibration, capacity, recovery, and isolated replacement-sensitivity analyses built on the fixed single-block baseline.
type: experiment
status: draft
created: 2026-08-18
updated: 2026-08-20

authorship:
  created_by: collaborative
  contributors:
    - researcher
    - llm

epistemic:
  role: mixed
  confidence: medium
  verification:
    - experiment-backed

scope:
  topics:
    - baseline-evaluation
    - calibration-scaling
    - capacity-sweep
    - compression-recovery
    - knowledge-distillation
    - mlp-block-replacement
    - operator-comparison
    - replacement-error-propagation
  granularities:
    - mlp-block
    - transformer-layer
    - model
    - cross-level
  pipeline_stages:
    - replacement
    - integration
    - recovery
    - evaluation
    - analysis

sources:
  - source_id: src-grafting-2025
    locator: "Sections 3.2-3.3"
    relation: contextualizes
    note: "Prior-work context for staged local fitting and integrated recovery; the project baseline is not a reproduction of self-grafting."

related:
  - "[[experiment-initial-block-compression-study]]"
  - "[[experiment-swiglu-operator-design-progression]]"
  - "[[method-block-importance]]"
  - "[[comparison-recovery-objective-trainable-scope]]"
  - "[[concept-replacement-error-propagation]]"
supersedes: []
superseded_by: []
---

# Baseline Operator Analysis

## Overview

**Mixed design and single-run evidence.** `baseline-testing.ipynb` defines the
fixed six-condition, layer-11 reference. `baseline-experiments.ipynb` extends
it with calibration-data, SwiGLU-width, recovery-budget, and full-depth
singleton-sensitivity analyses. Grafting provides prior-work context for local
operator fitting followed by integrated recovery, but this project baseline is
not a reproduction of its self-grafting control. [src-grafting-2025, Sections
3.2-3.3]

## Objective and Hypotheses

The notebook asks how calibration quantity affects local fit, how reduced
SwiGLU width trades footprint against quality, how equal replacement-only
recovery affects widths, and whether local distillability predicts the global
effect of replacing one block.

**Researcher hypotheses.** More data or width may reduce local error without a
proportional model-quality gain; layer position and operator family may govern
how residual error propagates. These propositions are not established beyond
the recorded run.

## Configuration

| Component | Recorded setting |
| --- | --- |
| Model and seed | `HuggingFaceTB/SmolLM2-1.7B`; seed 21 |
| Primary layer | zero-based layer 11 |
| Full-depth eligibility | layers 1-22; edge layers excluded |
| Sequence and batch size | 128 tokens; 2 sequences |
| Local calibration | 8, 16, 32, or 48 C4 batches |
| Local/model validation | 24 C4 batches / 24 WikiText-2 batches |
| References | ridge-fitted dense linear and affine; ridge $10^{-4}$ |
| SwiGLU ratios | 0.25, 0.50, 0.75, and 0.90 of teacher $d_{\mathrm{ff}}$ |
| Local fit | activation MSE; at most 64 epochs; minibatch 2,048; patience 3 |
| Recovery | forward KL; replacement parameters only; checkpoints 0-1,024 updates |

Calibration data, local validation, and recovery use document-disjoint C4
windows; model evaluation uses WikiText-2. A fixed local epoch count means a
larger calibration set also produces more optimizer updates. The current
scaling curve therefore combines data quantity with optimization exposure; it
is not an equal-update calibration-only comparison.

## Procedure

1. At layer 11, fit linear, affine, and 0.50-width SwiGLU candidates across
   calibration sizes and record held-out local metrics and model
   loss/perplexity.
2. At the largest calibration size, compare SwiGLU ratios 0.25-0.90 against
   the linear references.
3. Recover each fitted width with the same ordered teacher-logit batches and
   report common cumulative update checkpoints. Only replacement parameters
   are trainable; this is not full-model retraining.
4. Fit linear, affine, and 0.25/0.50/0.75 SwiGLU independently at each eligible
   layer and measure teacher-to-student KL with one replacement active.
5. Compare held-out local relative MSE with pre-recovery KL; retain their ratio
   only as a secondary within-protocol diagnostic.

Recovery streams one teacher-logit batch across all active width candidates
before discarding it. The smaller read-only validation-logit cache remains in
memory. This reduces recovery-cache memory without changing the stated
objective or ordered data exposure.

## Metrics and Project Definitions

**Project definition; no external citation required.** For teacher MLP $f_i$,
fitted substitute $g_i$, and held-out teacher input $h_i$:

$$
E_i^{\mathrm{local}}
=
\frac{\mathbb{E}\lVert g_i(h_i)-f_i(h_i)\rVert_2^2}
{\mathbb{E}\lVert f_i(h_i)\rVert_2^2+\varepsilon}.
$$

**Project evaluation definition using standard KL notation; no external
citation required.** If $S_i$ has only MLP $i$ replaced after local fitting:

$$
K_i^{\mathrm{pre}}
=
\mathbb{E}_x\left[D_{\mathrm{KL}}\left(
p_T(\cdot\mid x)\parallel p_{S_i}(\cdot\mid x)
\right)\right].
$$

**Project-proposed diagnostic; no external citation required.**

$$
R_i=\frac{K_i^{\mathrm{pre}}}{E_i^{\mathrm{local}}+\varepsilon}.
$$

$R_i$ is unbounded and has no universal threshold; it expresses global change
per unit of remaining local error only within a shared operator and evaluation
protocol. The two original quantities must remain visible. Additional metrics
include parameter/byte footprint, local MSE and cosine similarity, model loss,
perplexity, and pre/post-recovery KL.

## Direct Results and Interpretation

The following are **empirical findings from one configured run** in
[`baseline-experiments.json`](../../../data/results/notebook-block-study/baseline-experiments.json),
schema version 2:

- Untouched validation loss was 2.66997 and perplexity 14.4396 over 6,096
  predicted WikiText-2 tokens.
- Increasing calibration from 2,048 to 12,288 tokens reduced local relative
  MSE for all three tested families, but also increased learned-operator
  updates.
- At layer 11, increasing SwiGLU ratio from 0.25 to 0.90 reduced local relative
  MSE from 0.6471 to 0.6097; pre-recovery perplexity stayed between 15.1392 and
  15.1687 without a monotonic width ordering.
- After 1,024 replacement-only recovery updates, width-candidate perplexity was
  15.1169-15.1250, still above the untouched reference.
- Across 110 isolated operator-layer cases, mean post-recovery KL was highest
  at layers 1 (2.1012), 22 (1.8350), and 7 (1.4468); layer 20 was next at
  0.1941.
- Local relative MSE did not rank pre-recovery KL reliably across depth in
  this run.

These results separate local imitation, pre-recovery singleton sensitivity,
and fixed-protocol repairability. Pre-recovery KL is operator-conditioned; it
is not a universal block-importance or intrinsic-linearity score.

## Inputs, Artifacts, and Reproducibility

- [`baseline-experiments.ipynb`](../../../notebooks/block/baseline-experiments.ipynb)
- [`baseline-testing.ipynb`](../../../notebooks/block/baseline-testing.ipynb)
- [`src/mlp_replacement/baselines.py`](../../../src/mlp_replacement/baselines.py)
- [`src/mlp_replacement/recovery.py`](../../../src/mlp_replacement/recovery.py)
- [`baseline-experiments.json`](../../../data/results/notebook-block-study/baseline-experiments.json)

The checked local, ignored schema-2 artifact records configuration and result
tables but excludes activations, logits, fitted weights, optimizer state,
plots, and hardware telemetry. It contains four width rows, 24 recovery rows,
and 110 rows per full-depth KL table. Newer notebook additions expecting schema
3 remain pending rerun. The stage-level run/load path has not received an
independent numerical-parity test.

## Limitations

- One model and seed do not establish stability.
- Calibration quantity is confounded with update count.
- Operator families are not parameter-matched.
- Local $E_i$ uses held-out C4 while global $K_i$ uses WikiText-2, so their
  relationship includes cross-dataset transfer.
- Every full-depth row replaces one block independently; simultaneous errors
  may interact.
- Recovery matches update count, not computation per update.
- Candidate modules and activation sets still contribute substantial memory;
  streaming addresses only the large recovery-logit cache.

## Relationships

- [[experiment-initial-block-compression-study]] owns the fixed reference and
  broader stage map.
- [[experiment-swiglu-operator-design-progression]] extends the same evaluation
  conventions to more operator families.
- [[comparison-recovery-objective-trainable-scope]] defines the recovery
  objective and trainable scope.
- [[concept-replacement-error-propagation]] motivates the local-to-global and
  multi-block distinction.
- [[method-block-importance]] records forward-only screening alternatives.

## Sources

- `src-grafting-2025` - Sections 3.2-3.3, contextual comparison only
- Project-generated evidence -
  `data/results/notebook-block-study/baseline-experiments.json`, schema version
  2, created 2026-08-18 and checked 2026-08-20
