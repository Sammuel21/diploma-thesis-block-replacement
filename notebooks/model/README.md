# Model-Compression Strategy Notebooks

This directory studies model-level compression strategies that replace multiple
MLP blocks and evaluate their combined footprint and quality effects. Secondarily,
the directory studies block interactions and sensitivity to multi-block replacement.

- `baseline/compression-baseline.ipynb` evaluates uniform interleaved replacement as an
  end-to-end compression baseline.
- `interaction/block-interaction.ipynb` studies how replacement errors interact across
  multiple blocks.
- `swiglu/swiglu.ipynb` develops the initial model-level strategies based on SwiGLU
  compression.
- `swiglu/swiglu-2.ipynb` compares importance signals, allocation temperatures,
  minimum-width policies, and full-budget finalists.
- `swiglu/swiglu-3.ipynb` reports the completed calibration-budget, global-sparsity,
  and long-recovery workflow from saved artifacts.
- `swiglu/swiglu-4.ipynb` reports the completed recovery-strategy comparison.
- `swiglu/swiglu-5.ipynb` is the load-only report shared by the bounded search
  and optional gated confirmation.

Single-block characterization and operator design remain under `notebooks/block/`.
Scoped implementation instructions are in [`../AGENTS.md`](../AGENTS.md).
