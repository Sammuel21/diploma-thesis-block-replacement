# Recovery Recipe Runners

Concrete Python entry points for recovery-recipe experiments.
[`recovery_1.py`](recovery_1.py) runs Recovery-1, the schedule screen: per family, a constant
trunk to 100M tokens and 1-sqrt decays branched from it. It reuses the start and training code
of the recorded runs (Quantization-2, Stacking-1 order B, SwiGLU-8) and the shared recovery
loop.

Scientific choices belong in
[`workflows/configs/model/recovery/`](../../../configs/model/recovery/README.md). Reusable
recovery and evaluation operations belong in `src/mlp_replacement/`. See the
[experiment documentation](../../../../docs/experiments/model/recovery/README.md) and
[workflow instructions](../../../AGENTS.md).
