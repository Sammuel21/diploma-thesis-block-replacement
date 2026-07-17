# Block level

This file documents the ideas and implementations of work performed on block-level granularity, targeting block specific MLP operator. Due to thesis constraint we focus only on research regarding the MLP operator.

## Block definition


## Block Importance

Several block importance estimation methods exist. Random selection is useful as baseline benchmarking method. The baseline informative method for block importance estimation we use in the thesis is Block Importance (BI) score of block $i$ defined as:

\[
\mathrm{BI}_i
= 1 - \mathbb{E}_{X,t}\left[
\frac{\mathbf{X}_{i,t}^{\top}\mathbf{X}_{i+1,t}}
{\lVert \mathbf{X}_{i,t}\rVert_2 \lVert \mathbf{X}_{i+1,t}\rVert_2}
\right]
\]

TODO: further block importance estimation strategies

## MLP Operator Importance Analysis

## Replacement

### Replacement Objective

The objective of the block replacement is to create a surrogate block that mimics the ability of the original block on two granularities:

1) Block-level: surrogate block should mimic the transformation of original block with minimal differences. For example activation distillation.
2) Model-level: model-wide error should not substaintially increase with replaced surrogate block. This objective tracks the mode-wide error metric (e.g. perplexity)

The end goal of model compression is to achieve maximum compression at cost of minimal performance degradation, hence model-level error is prioritized in granularity importance.

### Operator replacement

### Optimal Hybrid construction

### Low-Rank decomposition

### Exceptions

Both practice and research show that the $\ell \in \{0, L-1\}$ (first, last) blocks should not be augmented. This is due to the fact that the first layer directly processes token embeddings and last layer outputs logits to the softmax function. For this reason we exempt the first and last layers from replacement modifications and keep them as is.

## Mixture-of-Experts (MoE)

### Mixture-of-Novices-and-Experts (MoNE)

## Block Scoring/Eval
