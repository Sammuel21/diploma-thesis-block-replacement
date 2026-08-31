# Model-Compression Strategy Notebooks

This directory studies model-level compression strategies that replace multiple
MLP blocks and evaluate their combined footprint and quality effects. Secondarily,
the directory studies block interactions and sensitivty of multi-block replacement.

- `compression-baseline.ipynb` evaluates uniform interleaved replacement as an
  end-to-end compression baseline.
- `block-interaction.ipynb` studies how replacement errors interact across
  multiple blocks.
- `swiglu.ipynb` is reserved for model-level strategies based on SwiGLU
  compression.

Single-block characterization and operator design remain under `notebooks/block/`.
