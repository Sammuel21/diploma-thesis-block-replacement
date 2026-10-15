# Model Baseline Configurations

These files hold explicit scientific settings for the matching
[baseline runners](../../../runs/model/baseline/README.md).

| Configuration | Purpose | Experiment guide |
| --- | --- | --- |
| [compression.json](compression.json) | Fixed interleaved SwiGLU replacement and fitting-method comparisons. | [Compression baseline](../../../../docs/experiments/model/baseline/compression-baseline.md) |
| [quantization-1.json](quantization-1.json) | MLP-only INT8/INT4 PTQ, optional INT4 QAT, and final quality and deployment measurements using the frozen SwiGLU-7 reference. | [Quantization-1](../../../../docs/experiments/model/baseline/quantization-1.md) |
| [speed-1.json](speed-1.json) | The 12 saved bundles timed in one job, each pinned by its manifest sha256, with forward and reverse passes. | [Speed-1](../../../../docs/experiments/model/baseline/speed-1.md) |

Quantization-1 leaves the QAT token budget unset and excludes QAT from its
default run list. Enabling QAT requires an explicit positive budget.
Structural comparators are supplied model bundles; this workflow does not
create additional architectural experiments.

See the [baseline experiment index](../../../../docs/experiments/model/baseline/README.md)
for current execution status and the individual guides for evaluation
protocols and artifact identities. Keep historical configurations intact
when introducing a new experiment family.
