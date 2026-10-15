---
title: Block Operator Experiments
status: active
modified: 2026-09-19
---

# Block Operator Experiments

This class contains replacement-operator construction and fitting studies:

- [Operator study](operator.md) compares architecture families across depth
  using local and singleton model metrics.
- [Operator distillation](operator-distillation.md) studies calibration,
  optimization, teacher-derived initialization, and reduced-SwiGLU width.

Reusable implementations belong in `src/mlp_replacement/operators/`; these
documents describe the experiment designs and their recorded evidence.
