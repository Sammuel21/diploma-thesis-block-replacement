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

This directory explains how the model-wide compression notebooks work. The
notebooks remain the executable specification, while these documents explain
the research question, stage order, configuration choices, and interpretation
of each pipeline.

| Workflow | Main question | Status |
| --- | --- | --- |
| [Compression baseline](compression-baseline.md) | Do the improved calibration and initialization methods help under a simple fixed replacement pattern? | Executed artifacts are present |
| [SwiGLU compression](swiglu.md) | Does nonuniform block-width allocation improve a fixed model-wide compression budget? | Historical and optimized artifacts are present |
| [SwiGLU allocation study](swiglu-2.md) | Which block score and allocation temperature should determine the nonuniform widths? | Implemented, not yet executed |

Reusable definitions belong under [methodology](../../methodology/). Results
and run metadata belong in JSON artifacts. A future `experiments/block/`
directory can document isolated operator and single-block studies without
mixing them with model-compression pipelines.
