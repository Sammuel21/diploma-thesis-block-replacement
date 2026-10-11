# Model Baseline Runners

These entry points execute model-level compression controls. Reusable
compression, recovery, and evaluation operations belong in
`src/mlp_replacement/`.

| Runner | Configuration | Experiment guide |
| --- | --- | --- |
| [compression.py](compression.py) | [compression.json](../../../configs/model/baseline/compression.json) | [Compression baseline](../../../../docs/experiments/model/baseline/compression-baseline.md) |
| [quantization.py](quantization.py) | [quantization-1.json](../../../configs/model/baseline/quantization-1.json) | [Quantization-1](../../../../docs/experiments/model/baseline/quantization-1.md) |

The compression runner exposes historical, optimized, and combined stages;
its numerical parity with retained notebook artifacts remains unverified.
Quantization-1 exposes `prepare`, `run`, `evaluate`, and `report`. Its default
variants are dense BF16, MLP INT8 PTQ, and MLP INT4 PTQ. Optional INT4 QAT
requires an explicit positive training-token budget. Full GPU scientific
validation remains pending.

Quantization-1 has dedicated [local](../../../jobs/local/quantization_1.sh) and
[Perun](../../../jobs/perun/quantization_1.sbatch) launchers. It runs in the
shared `mlp-replacement` environment with TorchAO and MSLK added; the
[environment pins](../../../environments/quantization-1-requirements.txt)
record the expected stack.
Use its experiment guide for commands and artifact conventions.

See the [baseline experiment index](../../../../docs/experiments/model/baseline/README.md),
[configuration index](../../../configs/model/baseline/README.md), and
[workflow instructions](../../../AGENTS.md).
