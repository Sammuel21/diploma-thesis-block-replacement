---
metadata_version: 1
title: Block Baseline Experiments
type: index
category: experiments/block/baseline
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

# Block Baseline Experiments

This class contains controlled singleton replacement baselines:

- [Single-block replacement baseline](baseline-testing.md) compares the dense
  MLP, trivial controls, linear maps, and a correctly sized reduced SwiGLU.
- [Baseline experiments](baseline-experiments.md) extends the comparison over
  calibration size, width, recovery budget, layer, and operator family.

These studies characterize one inserted replacement at a time. They do not
constitute model-wide compression or global recovery.
