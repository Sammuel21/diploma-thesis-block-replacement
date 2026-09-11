---
metadata_version: 1
title: Model Compression Experiment Workflows
type: index
category: experiments/model
status: active
created: 2026-09-10
modified: 2026-09-11
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---

# Model Compression Experiment Workflows

This directory explains the model-wide compression studies. The retained
notebooks are the explanatory and artifact-analysis frontend; headless compute
entry points live under
[`workflows/runs/model/`](../../../workflows/runs/model/) and explicit run
choices live under
[`workflows/configs/model/`](../../../workflows/configs/model/). These
documents explain the research question, stage order, configuration choices,
and interpretation of each pipeline.

| Workflow | Main question | Status |
| --- | --- | --- |
| [Compression baseline](compression-baseline.md) | Do the improved calibration and initialization methods help under a simple fixed replacement pattern? | Notebook artifacts present; Python runner not yet executed |
| [SwiGLU compression](swiglu.md) | Does nonuniform block-width allocation improve a fixed model-wide compression budget? | Notebook artifacts present; Python runner not yet executed |
| [SwiGLU allocation study](swiglu-2.md) | Which block score and allocation temperature should determine the nonuniform widths? | Python runner implemented; no final artifact yet |

Reusable definitions belong under [methodology](../../methodology/). Results
and run metadata belong in JSON artifacts. The Python migrations require an
empirical notebook-versus-runner comparison before numerical parity is
considered established. Block-study workflows remain a separate later
migration.
