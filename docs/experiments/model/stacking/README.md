---
title: Stacking Experiments
status: active
created: 2026-10-17
modified: 2026-10-17
---

# Stacking Experiments

This class holds model-wide experiments that chain compression methods, for
example SwiGLU replacement followed by weight quantization. Each variant is an
ordered list of stages applied to one model. Quantization alone appears here
as the 0% replacement row of the same pipeline and is labelled a
quantization-only baseline; [Quantization-1](../baseline/quantization-1.md)
stays the completed, unchanged baseline.

| Experiment | Main question | Status |
| --- | --- | --- |
| [Stacking-1](stacking-1.md) | Does SwiGLU replacement plus INT8/INT4/INT3 quantization reach better quality-per-byte points than either method alone, and does the order of operations matter? | Design settled 2026-10-17; not implemented |

Runners, configurations, and launchers will use matching `stacking/`
directories under `workflows/runs/model/`, `workflows/configs/model/`, and
`workflows/jobs/` when Stacking-1 is implemented. Reusable stage behavior
belongs in `src/mlp_replacement/`.
