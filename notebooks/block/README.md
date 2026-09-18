# Block-Compression Study Notebooks

These notebooks contain the maintained block-level characterization and
operator studies:

- `baseline-testing.ipynb` defines the initial single-block comparison.
- `baseline-experiments.ipynb` studies calibration, capacity, recovery, and
  behavior across blocks.
- `operator.ipynb` compares replacement-operator classes and block profiles.
- `operator-distillation.ipynb` studies calibration budgets, fitting choices,
  teacher-derived initialization, and reduced-SwiGLU widths.
- `activation-analysis.ipynb` is retained as an activation-geometry analysis,
  but it is excluded from the repository's notebook coding-style reference set.

The notebooks import scientific logic from `src/mlp_replacement/`. They should
be restartable and runnable from a clean kernel. Their default configuration is
intended for the remote RTX 4090 environment; the full post-gating covariance
eigendecomposition and the all-layer dense-linear fits are deliberately
expensive.

Generated summaries are written below `data/results/notebook-block-study/`,
which is local experiment storage and is ignored by Git. Raw activation tensors
are not persisted by these notebooks.

All outputs are exploratory until repeated with frozen model/data revisions,
additional layers and seeds, and the confirmation evaluation protocol.

Scoped implementation instructions are in [`../AGENTS.md`](../AGENTS.md).
