---
metadata_version: 1
title: Heterogeneous Operator Experiments
type: index
category: experiments/model/heterogenous
status: active
created: 2026-10-01
modified: 2026-10-01
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---

# Heterogeneous Operator Experiments

This class is reserved for model-wide experiments combining different MLP
replacement operator families and allocating capacity across layers.

Status: directory scaffold only. No experiment, scientific configuration, or
execution result is defined by this scaffold.

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
