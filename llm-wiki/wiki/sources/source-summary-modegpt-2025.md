---
id: source-summary-modegpt-2025
title: "MoDeGPT: Modular Decomposition for Large Language Model Compression"
summary: MoDeGPT jointly compresses matrix pairs within Transformer modules and allocates nonuniform sparsity from layer importance using forward-only calibration.
type: source-summary
status: review
created: 2026-07-17
updated: 2026-07-27

authorship:
  created_by: collaborative
  contributors:
    - researcher
    - llm

epistemic:
  role: prior-work
  confidence: high
  verification:
    - source-checked

scope:
  topics:
    - modular-decomposition
    - structured-compression
    - matrix-decomposition
    - local-reconstruction
    - sparsity-allocation
    - block-importance
  granularities:
    - mlp-block
    - transformer-layer
    - model
    - cross-level
  pipeline_stages:
    - data
    - screening
    - selection
    - replacement
    - integration
    - recovery
    - evaluation
    - analysis

sources:
  - source_id: src-modegpt-2025
    locator: "Entire paper; especially Sections 3-5 and Appendices B.2, B.6-B.7, and B.10-B.11"
    relation: defines

related:
  - "[[method-modegpt-modular-decomposition]]"
  - "[[method-modegpt-global-sparsity-allocation]]"
  - "[[method-block-importance]]"
  - "[[method-quality-preservation-evaluation]]"
supersedes: []
superseded_by: []
---

# MoDeGPT: Modular Decomposition for Large Language Model Compression

## Bibliographic Identity

- Registered source: `src-modegpt-2025`
- Authors: Chi-Heng Lin, Shangqian Gao, James Seale Smith, Abhishek Patel,
  Shikhar Tuli, Yilin Shen, Hongxia Jin, and Yen-Chang Hsu
- Year: 2025
- Venue: ICLR 2025 Oral
- Organizations: Samsung Research America and Florida State University
- Source snapshot: published conference paper, OpenReview submission 9191

## Research Question

The paper asks whether structured LLM compression can be formulated as joint
matrix decomposition within functional Transformer modules. Its goal is to
reduce intermediate dimensions without the adapter overhead or severe rank
constraint of independently decomposing each weight matrix, and without
requiring backward propagation for the primary compression method.
[src-modegpt-2025, Sections 1 and 3.1]

## Method

MoDeGPT minimizes output reconstruction error for pairs of matrices within
three functional module types:

1. **Type I, MLP up/gate and down matrices:** use a Nyström approximation of
   the calibration activation-correlation matrix. Select intermediate channels
   by deterministic ridge leverage scores and recompute the down projection in
   closed form.
2. **Type II, query-key matrices:** select a shared reduced dimension for each
   attention head using a CR decomposition derived from query and key state
   correlations.
3. **Type III, value-output matrices:** solve the linear reconstruction problem
   with an SVD of the input-weighted matrix product.

The method then converts layer-level Block Importance scores into a nonuniform
sparsity distribution under a global average-sparsity constraint. The
entropically regularized mapping retains more parameters in high-importance
layers. Both decomposition statistics and importance scores are collected
from forward passes over calibration data. [src-modegpt-2025, Sections
3.1-3.3; Algorithms 1-3]

## Evidence

The paper evaluates OPT from 125M to 6.7B parameters, LLaMA-1 7B, LLaMA-2 7B,
13B, and 70B, and LLaMA-3 8B. Its standard calibration sets contain 128
length-2048 samples from WikiText-2 or Alpaca. Perplexity is evaluated on
WikiText-2 and zero-shot performance through LM Evaluation Harness tasks.
[src-modegpt-2025, Section 4.1; Table 3; Appendix B.2]

The main experiments compare 10%-50% structured compression against uniform
and magnitude pruning, SVD, SliceGPT, ShortGPT, SLEB, LLM-Pruner, LLM Surgeon,
and related methods. The paper also reports module-level ablations, calibration
size, sparsity allocation, compression time, peak memory, optional LoRA
recovery, and approximately equal-compute comparisons.
[src-modegpt-2025, Sections 4.2-4.5; Appendices B.4-B.7]

## Findings

The following findings are reported by the paper and have not been reproduced
in this thesis:

- For LLaMA-2 7B, MoDeGPT reports perplexities 5.48, 6.16, 7.51, 8.41, and
  11.88 at 10%, 20%, 30%, 40%, and 50% compression, compared with the dense
  baseline at 5.12. For LLaMA-2 13B the corresponding values are 4.83, 5.29,
  6.10, 6.95, and 8.95 versus 4.57. [src-modegpt-2025, Table 3]
- At 30% compression of LLaMA-2 7B, BI-based sparsity allocation reports
  perplexity 7.51 and average zero-shot accuracy 60.78%, compared with 9.06
  and 53.47% under uniform allocation. [src-modegpt-2025, Section 4.5;
  Table 9]
- MLP parameters constitute 66.84% of the compressed module parameter budget,
  so MLP compression causes most absolute perplexity degradation. After
  normalizing by parameter share, query-key compression is the most sensitive
  module type. [src-modegpt-2025, Section 4.5; Figure 4; Table 7]
- Increasing calibration size improves performance initially, but reported
  zero-shot gains diminish beyond 128 samples for the studied 30%-compressed
  LLaMA-2 7B setting. [src-modegpt-2025, Section 4.5; Figure 5]
- The paper reports that Alpaca calibration improves zero-shot task accuracy
  relative to WikiText-2 calibration, especially at higher compression ratios,
  while perplexity can move differently. [src-modegpt-2025, Sections 4.2-4.3;
  Tables 4-5]
- Optional LoRA recovery uses 8,000 length-1024 samples after 128 calibration
  samples. It yields small, task-dependent changes; tuning only MLP matrices is
  more effective on average than tuning all linear matrices in that experiment.
  [src-modegpt-2025, Appendix B.6; Table 20]

## Limitations

MoDeGPT compresses dimensions within the original MLP and attention module
families. It is not an arbitrary architecture-replacement method and does not
show that a linear operator can replace a complete nonlinear MLP block.

The local modular reconstruction objective is a proxy for model behavior. Low
calibration reconstruction error does not guarantee preservation of language
model loss or downstream tasks after all modules are compressed.

Calibration distribution matters. The difference between WikiText-2 and
Alpaca results shows that a fixed small calibration set is not universally
representative, even when gains saturate beyond 128 samples in one ablation.

The primary method avoids backward propagation but is not computationally
free. For LLaMA-2 7B, the paper reports 4 hours 9 minutes of compression on one
A100. MLP correlation computation reaches 23.33 GiB peak memory for a model
reported as 13.81 GiB, and value-output SVD dominates compression time.
[src-modegpt-2025, Sections 4.4-4.5; Tables 6-8]

The decomposition theory and experiments assume supported Transformer module
structures. Architecture-specific details such as grouped-query attention
require adaptations described in the appendix.

## Thesis Relevance

MoDeGPT supplies a strong closed-form baseline for MLP-level compression. Its
Type-I procedure reduces the existing intermediate width, whereas the thesis
prototype learns a separate replacement operator. Comparing them would test
whether architectural substitution adds value beyond calibration-aware
channel selection and down-projection reconstruction.

The source also connects block-level and model-level decisions. Local
decomposition determines how each module is compressed; BI determines how the
global compression budget is distributed across depth. This is more expressive
than using BI only to choose a top-k set of complete replacements.

The recovery appendix reinforces the need to separate calibration and recovery
budgets. MoDeGPT uses 128 calibration samples for compression and 8,000 samples
for optional LoRA recovery, and recovery does not uniformly improve every task.

The paper's module-size ablation suggests reporting both absolute damage and
damage normalized by parameters removed. Otherwise the MLP may appear most
sensitive primarily because it contains most of the compressible parameters.

## Claims Requiring Verification

- The Type-I MLP method must be implemented and evaluated on the thesis model
  before it can serve as an empirical baseline.
- BI-driven continuous sparsity allocation must be compared fairly with the
  prototype's discrete top-k replacement strategies under the same total
  parameter target.
- The reported 128-sample saturation point should not be adopted as a general
  calibration rule without a model-, task-, and method-specific ablation.
- Recovery comparisons require fixed trainable-parameter sets, token budgets,
  optimizers, and evaluation suites; sample counts alone are insufficient.
- Runtime and throughput claims remain hardware- and implementation-specific
  and outside the thesis's current primary evaluation scope.

## Relationships

- [[method-modegpt-modular-decomposition]] records the three module-specific
  forward-only compression algorithms.
- [[method-modegpt-global-sparsity-allocation]] records the model-level mapping
  from importance to per-layer sparsity.
- [[method-block-importance]] is the layer score reused for that allocation.
- [[method-quality-preservation-evaluation]] uses MoDeGPT's reported tasks and
  metrics as context for the project evaluation profiles.

## Sources

- `src-modegpt-2025` - entire paper, especially Sections 3-5 and Appendices
  B.2, B.6-B.7, and B.10-B.11
