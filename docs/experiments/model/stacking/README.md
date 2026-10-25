---
title: Stacking Experiments
status: active
created: 2026-10-17
modified: 2026-10-20
---

# Stacking Experiments

This class holds model-wide experiments that chain compression methods, for
example SwiGLU replacement followed by weight quantization. A stacked model
applies at least two kinds of redundancy reduction: structural (operator
replacement), numerical (quantization), or algebraic (low-rank). Local fitting
and recovery training are not reduction types. Each variant is an ordered list
of stages applied to one model. Quantization of the dense model alone is a
baseline, not stacking: INT8 and INT4 are in
[Quantization-1](../baseline/quantization-1.md), INT3 in
[Quantization-2](../baseline/quantization-2.md) (decision log, 2026-10-20).

| Experiment | Main question | Status |
| --- | --- | --- |
| [Stacking-1](stacking-1.md) | Does SwiGLU replacement plus INT8/INT4/INT3 quantization reach better quality-per-byte points than either method alone, and does the order of operations matter? | Ran 2026-10-20/21, 48 runs completed; analysis pending; the 3 order-B INT3 runs are invalid (2026-10-25) |

Runners, configurations, and launchers use matching `stacking/` directories
under `workflows/runs/model/` and `workflows/configs/model/`, and
`stacking_1.*` launchers under `workflows/jobs/`. Reusable stage behavior
belongs in `src/mlp_replacement/`.
