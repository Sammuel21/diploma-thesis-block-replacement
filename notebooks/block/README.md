# Block-Compression Study Notebooks

These notebooks form the exploratory beginning of the maintained thesis
experiments. Run them in this order:

1. `activation-analysis.ipynb` studies activation geometry without modifying
   the model.
2. `baseline-testing.ipynb` compares one-block replacement families at matched
   parameter budgets.
3. `degradation-analysis.ipynb` freezes the dense-linear baseline and varies
   BI-based or random layer selection and the number of replacements.

The notebooks import scientific logic from `src/mlp_replacement/`. They should
be restartable and runnable from a clean kernel. Their default configuration is
intended for the remote RTX 4090 environment; the full post-gating covariance
eigendecomposition and the all-layer dense-linear fits are deliberately
expensive.

Generated summaries are written below `data/results/notebook-block-study/`,
which is local experiment storage and is ignored by Git. Raw activation tensors
are not persisted by these notebooks.

The final degradation section simply reserves the later comparison with a
stronger operator. Its implementation should be added only after the
single-block results identify which operator and budget are worth carrying
forward.

All outputs are exploratory until repeated with frozen model/data revisions,
additional layers and seeds, and the confirmation evaluation protocol.
