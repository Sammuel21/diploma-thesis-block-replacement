### 1. Quantization Baseline

- add another baseline to the project: quanitzation
- usecase: easy to do, comparison against performed methods
- configs: with retraining / without retraining

### 2. Linear + plain MLP retry, MLP first (added 2026-11-13)

- Heterogeneous-1's linear + plain MLP fits mostly stalled because of their start, so the arm is inconclusive
- retry: train the plain MLP alone first, then ridge-fit the linear part to what it misses (optionally a short joint fit)
- answers: does a linear part add anything next to an MLP of the same width (redundancy), and its fair equal-size price
- scope: block level, layers 3, 11, 19, a few widths; a few GPU-hours (estimate)
- see heterogeneous-1.md, "Follow-up analysis (2026-11-13)"

### 3. Linear + SwiGLU same-width check, offline (added 2026-11-13)

- compare linear + SwiGLU with k neurons against SwiGLU with k neurons alone, from the stored prices
- no GPU; answers the redundancy question for the SwiGLU hybrid

### 4. Other operator families (added 2026-11-13)

- supervisor: be creative with operators; only zero, mean, linear, plain MLP, and two linear hybrids were tested
- RBF networks, KAN-style blocks (one fast KAN variant uses Gaussian RBFs; sources not yet registered)
- doubts: 2048-dimensional inputs make distance-based units weak; no teacher start; per-unit cost like a plain MLP neuron
- more promising: operators that reuse the teacher's weights, e.g. splitting its neurons into experts (MoE, MoNE)
- a new operator first needs a block-level screen against SwiGLU at equal size (about 9 GPU-hours per operator, from Heterogeneous-1)

