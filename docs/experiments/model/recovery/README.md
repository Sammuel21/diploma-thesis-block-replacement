---
title: Recovery Recipe Experiments
status: active
created: 2026-11-07
modified: 2026-11-07
---

# Recovery Recipe Experiments

This class holds studies of the recovery recipe itself: the training schedule
used when a compressed model is repaired by distillation, with or without
simulated quantization. Local fitting is not part of it (decision log,
2026-11-07).

| Experiment | Main question | Status |
| --- | --- | --- |
| [Recovery-1](recovery-1.md) | Does a 1-sqrt learning-rate decay beat the constant 3e-5 recipe at 100M tokens, for dense INT3 QAT, B 20% INT4, and SwiGLU alone at 20%? | Designed and implemented 2026-11-07, not run |

| Area | Location |
| --- | --- |
| Documentation | This directory |
| Scientific configurations | [workflows/configs/model/recovery/](../../../../workflows/configs/model/recovery/README.md) |
| Python runners | [workflows/runs/model/recovery/](../../../../workflows/runs/model/recovery/README.md) |
| Perun launchers | [workflows/jobs/perun/recovery/](../../../../workflows/jobs/perun/recovery/README.md) |

Results go to `PROJECT/perun-results/recovery-1/` on PERUN and to
`data/results/workflows/model/recovery-1/` locally, ignored by Git under the
existing data policy.
