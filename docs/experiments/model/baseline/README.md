---
metadata_version: 1
title: Model Baseline Experiments
type: index
category: experiments/model/baseline
status: active
created: 2026-10-02
modified: 2026-10-02
authorship:
  created_by: llm
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---

# Model Baseline Experiments

This directory documents the model-level controls used to assess architectural
MLP replacement. Each experiment retains its own configuration, evaluation
protocol, and artifacts.

| Experiment | Purpose | Status |
| --- | --- | --- |
| [Compression baseline](compression-baseline.md) | Compare fitting methods under a fixed interleaved SwiGLU replacement pattern. | Notebook artifacts present; Python runner not yet executed |
| [Quantization-1](quantization-1.md) | Compare MLP-only INT8/INT4 compression with architectural editing in quality, measured bytes, speed, and recovery cost. | Implemented; full GPU scientific validation pending |

Matching [notebooks](../../../../notebooks/model/baseline/README.md),
[configurations](../../../../workflows/configs/model/baseline/README.md), and
[runners](../../../../workflows/runs/model/baseline/README.md) have separate
indexes. Quantization-1 is a headless workflow and has no notebook.

See the individual experiment documents for commands, artifact locations,
comparison denominators, and validation requirements. Historical baseline
results retain their original protocols; compatibility must be established
before comparing them with final SwiGLU evaluations.
