# Notebook Guide

The notebooks are grouped by their role in the research workflow. The current
direction is **model-level screening and budget allocation** in `analyses/`,
followed by **block-level characterization and operator construction** in
`block/`. The `mvp/` directory is a frozen historical prototype rather than the
starting point for new experiments.

The intended conceptual flow is:

`importance.ipynb` -> `budget.ipynb` -> block/operator experiments -> integrated evaluation.

## `analyses/`: model-level screening and allocation

This directory studies decisions made across Transformer blocks. It determines
which blocks should be protected and how a whole-model compression target is
distributed into local MLP budgets; it does not select or train the final
replacement operator.

### [`analyses.ipynb`](analyses/analyses.ipynb)

A lightweight index and working map for project-wide analysis topics, including
importance estimation, budget allocation, and candidate operator classes. It is
primarily an organizational notebook rather than a complete experiment.

### [`importance.ipynb`](analyses/importance.ipynb)

Explores block- and MLP-level screening signals used to estimate importance
$I_\ell$. It compares BI, residual-aware MLP scoring, and dense-linear
approximation behavior, while keeping importance distinct from operator
approximability. Its output is an importance ranking or score vector for the
budget allocator, not a replacement-operator choice.

### [`budget.ipynb`](analyses/budget.ipynb)

Develops the global-to-local operator-budget allocation method discussed in the
thesis notes. It converts one whole-model sparsity target and block-importance
scores into inferred per-block capacity targets or caps. Local operator
selection, fitting, and any later unused-budget reconciliation remain separate
downstream steps.

## `block/`: block characterization and operator experiments

This directory studies what happens inside an individual MLP and how candidate
replacement operators behave. See [`block/README.md`](block/README.md) for the
current experimental sequence and execution notes.

### [`activation-analysis.ipynb`](block/activation-analysis.ipynb)

Analyzes captured MLP activations, covariance spectra, effective dimensionality,
and held-out reconstruction behavior. It provides diagnostic evidence about
activation geometry without replacing model blocks.

### [`baseline-testing.ipynb`](block/baseline-testing.ipynb)

Runs the practical single-block replacement baseline. It compares a SwiGLU
whose intermediate width is fixed at 50% of the original MLP `d_ff` with the
original MLP and simple controls, reporting both approximation quality and
parameter footprint.

### [`degradation-analysis.ipynb`](block/degradation-analysis.ipynb)

Studies model-quality degradation after independently fitted dense-linear
replacements are inserted into selected blocks. It compares BI-based and random
selection behavior and examines how degradation accumulates across multiple
replacements.

### [`operator.ipynb`](block/operator.ipynb)

A working notebook for operator-level surgery and candidate compression
strategies. It separates complete MLP replacement from replacement of internal
SwiGLU components and records possible operator families such as compact MLPs,
linear structures, hybrids, and low-rank factorizations.

## `mvp/`: frozen historical prototype

These notebooks preserve the original MVP workflow and its outputs for
traceability. They should be read as historical evidence and should not be
extended as the maintained implementation path; see
[`mvp/README.md`](mvp/README.md).

### [`introduction.ipynb`](mvp/introduction.ipynb)

Contains the original thesis introduction, early experiment configuration, and
prototype environment setup.

### [`methods.ipynb`](mvp/methods.ipynb)

Records early MVP methodology notes. It belongs to the prototype narrative and
is not the current global-to-local allocation specification.

### [`multi-block-replacement.ipynb`](mvp/multi-block-replacement.ipynb)

Contains the original end-to-end multi-block replacement workflow, including BI
screening, local fitting, recovery variants, and early search experiments. Its
implementation is retained for provenance rather than future extension.

### [`analysis.ipynb`](mvp/analysis.ipynb)

Loads and summarizes historical MVP runs, including fixed and search
experiments and BI-related observations under the earlier compute budget.

## Root-level notebooks

### [`data-load.ipynb`](data-load.ipynb)

Documents the early data-loading and conversion path from the provided sampling
utilities into calibration and validation batches expected by the model
pipeline. Confirm that its data assumptions still match the active experiment
before reuse.

### [`block-replacement.ipynb`](block-replacement.ipynb)

Currently an empty placeholder. It does not define an active experiment or
workflow; use the notebooks under `analyses/` and `block/` instead.
