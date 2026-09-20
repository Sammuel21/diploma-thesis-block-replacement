---
metadata_version: 1
title: Homogeneous SwiGLU Experiment Progression
type: index
category: experiments/model/swiglu
status: active
created: 2026-09-19
modified: 2026-09-20
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---

# Homogeneous SwiGLU Experiment Progression

This class holds model-wide experiments in which every replaced MLP remains a
reduced-width SwiGLU. Heterogeneous operator selection is deliberately outside
this progression.

The chronological findings and rationale connecting all five experiments are
summarized in [SwiGLU experimental progression and results](swiglu-progression.md).

| Experiment | Main contribution | Execution status |
| --- | --- | --- |
| [SwiGLU](swiglu.md) | Initial homogeneous operator and nonuniform allocation study | Notebook artifacts present |
| [SwiGLU-2](swiglu-2.md) | Singleton-KL allocation policy and width search | Notebook artifact present |
| [SwiGLU-3](swiglu-3.md) | Calibration sweep, 20%-50% sparsity sweep, and 100M-token recovery | Completed artifact present |
| [SwiGLU-4](swiglu-4.md) | Recovery objective, learning rate, RMSNorm, and LoRA comparison | Completed artifact present |
| [SwiGLU-5](swiglu-5.md) | Efficient initialization/allocation search followed by gated confirmation | Search and both 100M confirmations completed |

The maintained runners and configurations use the matching class directories
under `workflows/runs/model/swiglu/` and `workflows/configs/model/swiglu/`.
Historical result artifacts keep their existing paths and workflow IDs.
