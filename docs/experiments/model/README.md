---
metadata_version: 1
title: Model Compression Experiment Workflows
type: index
category: experiments/model
status: active
created: 2026-09-10
modified: 2026-09-21
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
| [Compression baseline](baseline/compression-baseline.md) | Do the improved calibration and initialization methods help under a simple fixed replacement pattern? | Notebook artifacts present; Python runner not yet executed |
| [Block interaction](interaction/block-interaction.md) | How do locally fitted replacements interact across adjacent windows and block pairs? | Notebook artifact present; no Python runner |
| [Homogeneous SwiGLU experiments](swiglu/README.md) | What quality did model-wide SwiGLU replacement achieve, and why? See the [results overview](swiglu/swiglu-results.md) or [chronological progression](swiglu/swiglu-progression.md). | SwiGLU-1–5 artifacts present; SwiGLU-5 search and both 100M confirmations completed |

Reusable definitions belong under [methodology](../../methodology/). Results
and run metadata belong in JSON artifacts. The Python migrations require an
empirical notebook-versus-runner comparison before numerical parity is
considered established. Block-study workflows remain a separate later
migration.
