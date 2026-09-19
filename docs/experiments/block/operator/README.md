---
metadata_version: 1
title: Block Operator Experiments
type: index
category: experiments/block/operator
status: active
created: 2026-09-19
modified: 2026-09-19
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---

# Block Operator Experiments

This class contains replacement-operator construction and fitting studies:

- [Operator study](operator.md) compares architecture families across depth
  using local and singleton model metrics.
- [Operator distillation](operator-distillation.md) studies calibration,
  optimization, teacher-derived initialization, and reduced-SwiGLU width.

Reusable implementations belong in `src/mlp_replacement/operators/`; these
documents describe the experiment designs and their recorded evidence.
