---
id: source-summary-grafting-2025
title: Exploring Diffusion Transformer Designs via Grafting
summary: Grafting edits pretrained diffusion transformers through local activation distillation followed by model-level fine-tuning.
type: source-summary
status: review
created: 2026-07-17
updated: 2026-07-17

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
    - operator-grafting
    - activation-distillation
    - replacement-error-propagation
    - architectural-editing
    - hybrid-architectures
  granularities:
    - mlp-block
    - transformer-layer
    - model
    - cross-level
  pipeline_stages:
    - data
    - replacement
    - integration
    - recovery
    - evaluation
    - analysis

sources:
  - source_id: src-grafting-2025
    locator: "Entire paper; especially Sections 3-6 and Appendix B"
    relation: defines

related:
  - "[[method-two-stage-operator-grafting]]"
  - "[[concept-replacement-error-propagation]]"
supersedes: []
superseded_by: []
---

# Exploring Diffusion Transformer Designs via Grafting

## Bibliographic Identity

- Registered source: `src-grafting-2025`
- Authors: Keshigeyan Chandrasegaran, Michael Poli, Daniel Y. Fu, Dongjun Kim,
  Lea M. Hadzic, Manling Li, Agrim Gupta, Stefano Massaroli, Azalia
  Mirhoseini, Juan Carlos Niebles, Stefano Ermon, and Li Fei-Fei
- Year: 2025
- Venue: arXiv preprint 2506.05340v2
- Organizations: Stanford University, Liquid AI, Together AI, UC San Diego,
  Northwestern University, Google DeepMind, and Salesforce Research
- Source snapshot: arXiv version 2

## Research Question

The paper asks whether pretrained diffusion transformers can be used as
scaffolds for evaluating new architecture designs without training every design
from scratch. It focuses on two practical problems: initializing a replacement
operator and repairing the error that accumulates when many replacements are
integrated. [src-grafting-2025, Sections 1 and 3.1]

## Method

Grafting uses two stages:

1. **Activation distillation:** train each new operator as a local regression
   model that approximates the original operator's output on captured inputs.
2. **Lightweight fine-tuning:** integrate all initialized operators and train
   the edited model end to end on a smaller dataset to reduce accumulated
   model-level error.

The design space varies the original operator, replacement architecture,
replacement locations, and replacement ratio. Experiments replace MHA and MLP
operators in DiT-XL/2 and MHA operators in PixArt-Sigma. The paper also defines
a self-grafting control in which an operator is replaced by a randomly
initialized operator of the same architecture. [src-grafting-2025, Sections
3.1-3.3 and 4.1]

Stage 1 uses 8,000 samples and pre-extracted regression features. Operator
training is independent and can run in parallel. For the primary DiT-XL/2
experiments, Stage 2 uses 10% of ImageNet-1K, 50,000 steps, and batch size 256.
[src-grafting-2025, Section 4.1; Appendix B.1]

## Evidence

The main testbed edits all 28 MHA or MLP positions in DiT-XL/2 through
self-grafting and evaluates alternative operators at 50%, 75%, and 100%
replacement. Quality is assessed with FID, sFID, Inception Score, precision,
and recall over generated images. [src-grafting-2025, Sections 3-4]

For high-resolution text-to-image generation, the paper grafts Hyena-X into
PixArt-Sigma using 8,000 synthetic pairs for local initialization and 12,000
pairs for rank-64 LoRA recovery. It reports GenEval and H100 wall-clock
latency. A separate case study rewires pairs of sequential DiT blocks into
parallel blocks, merging their outputs through a linear projection.
[src-grafting-2025, Sections 5-6]

## Findings

The following findings are reported for diffusion transformers and have not
been reproduced in this thesis:

- The local regression objective is operator-dependent. L1 gives the best MHA
  self-grafting result, while L2 gives the best MLP result in the studied
  layers. [src-grafting-2025, Section 3.3; Table 1]
- Replacing every MHA or MLP with the same randomly initialized architecture
  causes catastrophic quality loss. Local initialization followed by
  fine-tuning on 10% of ImageNet-1K recovers near-baseline FID.
  [src-grafting-2025, Section 3.3; Table 2]
- Interleaved partial replacement is more robust than full replacement for
  local MHA alternatives. At 100% MHA replacement, all tested alternatives
  fail badly. [src-grafting-2025, Section 4.2; Table 4]
- MLP width is comparatively robust. Replacing every MLP with expansion ratio
  3 reduces MLP parameters and FLOPs by 25% and reports FID 2.66 versus 2.27
  for the baseline. Expansion ratio 6 preserves strong quality while
  increasing parameters. [src-grafting-2025, Section 4.2; Table 4]
- In the PixArt-Sigma experiment, replacing 50% of selected MHA operators
  reports a 1.43x single-forward-pass speedup on H100 with GenEval 47.78 versus
  49.75 for the baseline. [src-grafting-2025, Section 5; Table 5]
- Parallelizing every pair of sequential DiT blocks halves depth from 28 to 14
  while increasing parameters by 6%; the longer run reports FID 2.77.
  [src-grafting-2025, Section 6; Table 6]

## Limitations

The authors explicitly restrict their evidence to pretrained diffusion
transformers. Transfer to autoregressive language models is future work. The
paper also states that successful grafting does not establish that the same
architecture will perform well when trained from scratch. [src-grafting-2025,
Section 8]

The PixArt-Sigma experiment uses uncurated synthetic data, and the authors
observe localized artifacts that may reflect data quality or limited LoRA
capacity. Grafting also presupposes access to a pretrained teacher.
[src-grafting-2025, Sections 5 and 8]

**Synthesis.** The paper's use of the term `lightweight` is relative: the main
Stage 2 experiments use eight H100 GPUs for up to 50,000 or 100,000 steps. The
method is cheaper than DiT pretraining but is not evidence that a very small
LLM recovery budget will suffice.

## Thesis Relevance

Grafting is the closest registered source to the thesis's basic operator
replacement methodology. It directly supports separating local operator
approximation from model-level integration recovery. It also supplies four
experimental axes already anticipated by the thesis: replacement target,
replacement architecture, location strategy, and replacement ratio.

Self-grafting is an important control. Replacing an MLP with an identical
architecture separates failure caused by local optimization and integration
from failure caused by reduced capacity or a changed operator family.

The source also warns against assuming one universal local loss. Activation
distribution and operator architecture can change whether L1, L2, or another
regression objective is appropriate. This motivates block-level loss ablations
rather than treating MSE as methodologically settled.

The evidence does not establish that DiT results transfer to causal LLMs. In
particular, diffusion timesteps, image objectives, conditioning, and FID-based
evaluation differ from autoregressive token prediction and perplexity.

## Claims Requiring Verification

- Transfer of two-stage grafting to autoregressive LLM MLP blocks must be
  demonstrated by thesis experiments.
- The best local loss for LLM MLP activations remains unresolved.
- Self-grafting should be tested before attributing degradation to substitute
  capacity alone.
- Claims about runtime or deployment speed require separate implementation and
  hardware analysis and are outside the current thesis evaluation scope.

## Relationships

- [[method-two-stage-operator-grafting]] formalizes the local-then-global
  replacement workflow.
- [[concept-replacement-error-propagation]] records why individually accurate
  substitutes may still fail when composed inside the full model.

## Sources

- `src-grafting-2025` - entire paper, especially Sections 3-6 and Appendix B
