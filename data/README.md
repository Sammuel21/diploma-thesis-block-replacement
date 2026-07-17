# Data Policy

Data is ignored by default because model datasets, caches, activations, and
checkpoints can be large or externally reproducible.

The only current exception is `data/mvp/results/logs/*.json`. These small files
are versioned as the empirical evidence for the frozen MVP archive. See
`docs/prototype/mvp/` for their manifest and interpretation.
