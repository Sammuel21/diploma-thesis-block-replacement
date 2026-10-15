---
title: Heterogeneous Operator Experiments
status: active
created: 2026-10-01
modified: 2026-10-11
---

# Heterogeneous Operator Experiments

This class is reserved for model-wide experiments combining different MLP
replacement operator families and allocating capacity across layers.

Status: directory scaffold only. No experiment, scientific configuration, or
execution result is defined by this scaffold.

Planned requirement (decision log, 2026-10-11): each operator family gets a
block-level recipe-selection study comparable with SwiGLU-8 Part A, so local
fitting can be compared between the homogeneous and heterogeneous pipelines.
It is scheduled after Quantization-1 and SwiGLU-8 are deployed.

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
