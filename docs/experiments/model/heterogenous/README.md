---
title: Heterogeneous Operator Experiments
status: active
created: 2026-10-01
modified: 2026-11-04
---

# Heterogeneous Operator Experiments

This class is reserved for model-wide experiments combining different MLP
replacement operator families and allocating capacity across layers.

A heterogeneous model is a hybrid of SwiGLU and non-SwiGLU operators: the same
allocator picks one item per layer from the dense MLP, SwiGLU widths, and new
operators (decision log, 2026-11-04). Each new operator family first gets a
block-level recipe-selection study comparable with SwiGLU-8 Part A, so model-level
comparisons are about the operator itself, not about how well it happened to be
tuned (decision log, 2026-10-11).

| Experiment | Main question | Status |
| --- | --- | --- |
| [Heterogeneous-1](heterogeneous-1.md) | How should each operator be fitted, which operators get picked, and can their pre-recovery prices be trusted? | Design settled 2026-11-04; not implemented |
| [Heterogeneous-2](heterogeneous-2.md) | How do hybrids compare with SwiGLU after recovery, with which retraining scope, and does local fitting matter? | Outline; designed after Heterogeneous-1 reports |

No scientific configuration or execution result exists yet.

| Area | Location |
| --- | --- |
| Documentation | This directory |
| Notebooks and reports | [notebooks/model/heterogenous/](../../../../notebooks/model/heterogenous/README.md) |
| Scientific configurations | [workflows/configs/model/heterogenous/](../../../../workflows/configs/model/heterogenous/README.md) |
| Python runners | [workflows/runs/model/heterogenous/](../../../../workflows/runs/model/heterogenous/README.md) |
| Local launchers | [workflows/jobs/local/heterogenous/](../../../../workflows/jobs/local/heterogenous/README.md) |
| Perun launchers | [workflows/jobs/perun/heterogenous/](../../../../workflows/jobs/perun/heterogenous/README.md) |

The local results area is `data/results/workflows/model/heterogenous/` and
remains ignored by Git under the existing data policy. Concrete experiment
identities and run layouts will be defined with their implementations.

The [homogeneous SwiGLU family](../swiglu/README.md) remains the reference
experiment class. Its notebooks, configurations, workflow identities, and
historical artifacts retain their existing locations.

Reusable operator, allocation, recovery, and evaluation behavior belongs in
`src/mlp_replacement/`. Isolated operator studies remain under the existing
`block/operator/` areas.
