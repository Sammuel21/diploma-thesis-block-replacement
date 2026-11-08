---
title: Recovery Recipe Experiments
status: active
created: 2026-11-07
modified: 2026-11-08
---

# Recovery Recipe Experiments

This class holds studies of the recovery recipe itself: the training schedule
used when a compressed model is repaired by distillation, with or without
simulated quantization. Local fitting is not part of it (decision log,
2026-11-07).

| Experiment | Main question | Status |
| --- | --- | --- |
| [Recovery-1](recovery-1.md) | Does a 1-sqrt learning-rate decay beat the constant 3e-5 recipe at 100M tokens, for dense INT3 QAT, B 20% INT4, and SwiGLU alone at 20%? | Run 2026-11-07 (array 145625): dense INT3 adopts the 20% decay; B and SwiGLU keep constant; `summary-001` after flag acceptance |
| [Recovery-2](recovery-2.md) | At 1B, does a 20% or 50% decay beat constant for the same three families, setting each family's 3B schedule? | Designed and decided 2026-11-08; not implemented |

| Area | Location |
| --- | --- |
| Documentation | This directory |
| Scientific configurations | [workflows/configs/model/recovery/](../../../../workflows/configs/model/recovery/README.md) |
| Python runners | [workflows/runs/model/recovery/](../../../../workflows/runs/model/recovery/README.md) |
| Perun launchers | [workflows/jobs/perun/recovery/](../../../../workflows/jobs/perun/recovery/README.md) |

Results go to `PROJECT/perun-results/recovery-N/` on PERUN and to
`data/results/workflows/model/recovery-N/` locally, ignored by Git under the
existing data policy.
