# Model-Compression Strategy Notebooks

This directory studies model-level compression strategies that replace multiple
MLP blocks and evaluate their combined footprint and quality effects. Secondarily,
the directory studies block interactions and sensitivity to multi-block replacement.

- `compression-baseline.ipynb` evaluates uniform interleaved replacement as an
  end-to-end compression baseline.
- `block-interaction.ipynb` studies how replacement errors interact across
  multiple blocks.
- `swiglu.ipynb` develops the initial model-level strategies based on SwiGLU
  compression.
- `swiglu-2.ipynb` compares importance signals, allocation temperatures,
  minimum-width policies, and full-budget finalists.
- `swiglu-3.ipynb` reports the completed calibration-budget, global-sparsity,
  and long-recovery workflow from saved artifacts.

Single-block characterization and operator design remain under `notebooks/block/`.
Scoped implementation instructions are in [`../AGENTS.md`](../AGENTS.md).
