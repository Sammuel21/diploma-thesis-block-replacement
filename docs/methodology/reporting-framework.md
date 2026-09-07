# Compression Reporting Framework

## Purpose

This document defines supporting information to report alongside the model
quality and compression evaluations in
[evaluation-framework.md](evaluation-framework.md). Its purpose is to show how
much data and compute were required to create each compressed model.

## Reported Budgets

Report the following separately:

| Budget | Required information |
| --- | --- |
| Screening and activation capture | samples, tokens, forward passes, and wall-clock time |
| Local operator fitting | unique calibration pairs, total pair presentations, epochs, optimizer steps, and trainable parameters |
| Model-wide recovery | training tokens, epochs, optimizer steps, objective, and trainable parameters |
| Complete workflow | estimated FLOPs, GPU-hours, GPU type and count, wall-clock time, and peak memory |

Unique data and repeated training exposure are different quantities. For
example, processing the same calibration pairs for 64 epochs does not create
64 times more unique data, but it does require approximately 64 times more
training work.

## Relative Cost

Use two transparent ratios:

```text
training_data_fraction_pct = compression_training_tokens / pretraining_tokens * 100
compute_fraction_pct = compression_training_FLOPs / pretraining_FLOPs * 100
```

These are standard percentage ratios and do not represent a new method. Token
fraction is a data-budget comparison, not a substitute for compute fraction.
Local operator fitting trains only an MLP replacement, whereas model-wide
recovery processes the complete model; equal token counts therefore do not
imply equal compute.

Prefer reported pretraining FLOPs when available. Otherwise, clearly label the
denominator as an estimate and document its formula, parameter count, token
count, and assumptions. SmolLM2-1.7B reports 11 trillion pretraining tokens in
its [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B).

## Accounting Boundaries

Report two totals:

1. Production cost: the winning screening, operator-fitting, integration, and
   recovery path required to create one selected compressed model.
2. Research cost: all architecture sweeps, initialization comparisons,
   allocation policies, failed runs, and ablations performed during the study.

Do not present the complete research sweep as the cost of producing one model.
State whether evaluation runs are excluded from training cost, and apply that
choice consistently.

## Result Summary

Each final configuration should have one compact summary row containing:

| Method | Total parameter reduction | MLP parameter reduction | PPL change | KL | Production tokens | Optimizer steps | GPU-hours | Pretraining data fraction | Pretraining compute fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

Measured GPU-hours, wall-clock time, and peak memory should name the hardware.
Estimated FLOPs should be marked as estimates and should not be mixed with
measured hardware utilization.

## Prior-Work Context

[MiniTron](https://arxiv.org/abs/2407.14679) reports retraining with less than
3% of the original training data, up to 40 times fewer training tokens per
derived model, and a 1.8 times compute saving for producing its full model
family. [Grafting](https://arxiv.org/abs/2506.05340) reports edited diffusion
transformer designs using less than 2% of pretraining compute.

These results motivate the same style of reporting, but they are not direct
like-for-like baselines. MiniTron uses structured LLM pruning, while Grafting
studies diffusion transformers. Thesis claims should therefore compare the
accounting methodology and relative budget without implying identical models,
objectives, or hardware.
