# Recovery Recipe Configurations

Versioned scientific configurations for recovery-recipe experiments.
[`recovery-1.json`](recovery-1.json) configures Recovery-1: its families, decays, rows, winner
rule, gates, and sha256 pins. It points at `quantization-2.json` and `stacking-1.json` for the
recipe, and the runner refuses to start unless their recipe blocks agree.
[`recovery-2.json`](recovery-2.json) configures Recovery-2 in the same way at 1B. It adds each
family's update layout and its Recovery-1 run, and it fixes the median reference.

Matching runners belong in
[`workflows/runs/model/recovery/`](../../../runs/model/recovery/README.md). See the
[experiment documentation](../../../../docs/experiments/model/recovery/README.md).
