---
title: Model Baseline Experiments
status: active
created: 2026-10-02
modified: 2026-10-15
---

# Model Baseline Experiments

This directory documents the model-level controls used to assess architectural
MLP replacement. Each experiment retains its own configuration, evaluation
protocol, and artifacts.

| Experiment | Purpose | Status |
| --- | --- | --- |
| [Compression baseline](compression-baseline.md) | Compare fitting methods under a fixed interleaved SwiGLU replacement pattern. | Notebook artifacts present; Python runner not yet executed |
| [Quantization-1](quantization-1.md) | Compare MLP-only INT8/INT4 compression with architectural editing in quality, measured bytes, speed, and recovery cost. | Completed 2026-10-13; see its results |
| [Speed-1](speed-1.md) | Time dense, quantized, and SwiGLU-8 models in one job on one GPU, so their speeds are comparable. | Implemented 2026-10-15; not yet submitted |

Matching [notebooks](../../../../notebooks/model/baseline/README.md),
[configurations](../../../../workflows/configs/model/baseline/README.md), and
[runners](../../../../workflows/runs/model/baseline/README.md) have separate
indexes. Quantization-1 and Speed-1 are headless workflows and have no notebook.

See the individual experiment documents for commands, artifact locations,
comparison denominators, and validation requirements. Historical baseline
results retain their original protocols; compatibility must be established
before comparing them with final SwiGLU evaluations.
