# Recovery Recipe Configurations

Versioned scientific configurations for recovery-recipe experiments.
[`recovery-1.json`](recovery-1.json) configures Recovery-1: its families, decays, rows, winner
rule, gates, and sha256 pins. It points at `quantization-2.json` and `stacking-1.json` for the
recipe, and the runner refuses to start unless their recipe blocks agree.

Matching runners belong in
[`workflows/runs/model/recovery/`](../../../runs/model/recovery/README.md). See the
[experiment documentation](../../../../docs/experiments/model/recovery/README.md).
