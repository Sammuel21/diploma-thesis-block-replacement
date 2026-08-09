# Block-Compression Study Notebooks

These notebooks form the exploratory beginning of the maintained thesis
experiments. Run them in this order:

1. `activation-analysis.ipynb` studies activation geometry without modifying
   the model.
2. `baseline-testing.ipynb` evaluates one fixed SwiGLU whose intermediate
   width is 50% of the original MLP `d_ff`, against the original MLP and
   zero/mean controls at layer 11.
3. `degradation-analysis.ipynb` independently freezes a dense-linear diagnostic
   operator and varies BI-based or random layer selection and replacement count.

The notebooks import scientific logic from `src/mlp_replacement/`. They should
be restartable and runnable from a clean kernel. Their default configuration is
intended for the remote RTX 4090 environment; the full post-gating covariance
eigendecomposition and the all-layer dense-linear fits are deliberately
expensive.

Generated summaries are written below `data/results/notebook-block-study/`,
which is local experiment storage and is ignored by Git. Raw activation tensors
are not persisted by these notebooks.

The degradation notebook does not treat the dense-linear operator as the winner
of the single-block baseline. A later cross-operator replication may reuse its
selection protocol only after the narrow-SwiGLU result has been reviewed.

All outputs are exploratory until repeated with frozen model/data revisions,
additional layers and seeds, and the confirmation evaluation protocol.
