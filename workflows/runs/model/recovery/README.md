# Recovery Recipe Runners

Concrete Python entry points for recovery-recipe experiments.
[`recovery_1.py`](recovery_1.py) runs Recovery-1, the schedule screen: per family, a constant
trunk to 100M tokens and 1-sqrt decays branched from it. It reuses the start and training code
of the recorded runs (Quantization-2, Stacking-1 order B, SwiGLU-8) and the shared recovery
loop. [`recovery_2.py`](recovery_2.py) runs Recovery-2, the same screen at 1B with 50% and 20%
decays. It imports Recovery-1's starts, scoring, and checks unchanged, and copies each recorded
1B run's update layout: one segment for dense INT3, and SwiGLU-7's two segments for B and SwiGLU.

Scientific choices belong in
[`workflows/configs/model/recovery/`](../../../configs/model/recovery/README.md). Reusable
recovery and evaluation operations belong in `src/mlp_replacement/`. See the
[experiment documentation](../../../../docs/experiments/model/recovery/README.md) and
[workflow instructions](../../../AGENTS.md).
