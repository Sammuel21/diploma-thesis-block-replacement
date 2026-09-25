---
metadata_version: 1
title: Homogeneous SwiGLU Results Overview
type: experiment-synthesis
category: experiments/model/swiglu
status: active
created: 2026-09-21
modified: 2026-09-25
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
sources:
  artifacts:
    - data/results/notebook-model-study/swiglu-compression.json
    - data/results/notebook-model-study/swiglu-compression-optimized.json
    - data/results/notebook-model-study/swiglu-2.json
    - data/results/workflows/model/swiglu-3/run-001.json
    - data/results/workflows/model/swiglu-4/run-001.json
    - data/results/workflows/model/swiglu-5/search/run-001.json
    - data/results/workflows/model/swiglu-5/confirmation/run-001-target-0.2.json
    - data/results/workflows/model/swiglu-5/confirmation/run-001-target-0.5.json
    - data/results/workflows/model/swiglu-6/prepare-001.json
    - data/results/workflows/model/swiglu-6/protocol-001.json
    - data/results/workflows/model/swiglu-6/recovery-001-target-0.2.json
    - data/results/workflows/model/swiglu-6/recovery-001-target-0.5.json
    - data/results/workflows/model/swiglu-6/evaluation-001.json
---

# Homogeneous SwiGLU Results Overview

Compact lookup for the principal homogeneous SwiGLU results on the pinned
`HuggingFaceTB/SmolLM2-1.7B` revision. For experimental reasoning and the
handoff between studies, see the [progression](swiglu-progression.md). For full
methods and execution details, use the [individual experiment pages](README.md).

Layers 0 and 23 are protected; layers 1–22 are eligible for replacement. The
removal target applies to parameters in those eligible MLPs. Measured
whole-model removal is reported separately.

## Comparable recovery endpoints

These rows use the shared historical 24-batch WikiText validation prefix
(6,096 predicted tokens) and fixed C4 recovery-validation KL. They are the
most directly comparable quality measurements from SwiGLU-3 onward.

| Experiment | Method or configuration | Eligible-MLP removal | Whole-model removal | Recovery tokens | Sequence length | WikiText prefix PPL ↓ | Fixed C4 KL ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense teacher | Uncompressed reference | 0% | 0% | — | — | **14.4396** | 0 by definition |
| SwiGLU-3 | Ranked singleton-KL widths, `1e-5` | 20% | 12.94% | 100M | 128 | 16.6543 | 0.075510 |
| SwiGLU-3 | Ranked singleton-KL widths, `1e-5` | 30% | 19.41% | 100M | 128 | 18.0415 | 0.117218 |
| SwiGLU-3 | Ranked singleton-KL widths, `1e-5` | 40% | 25.88% | 100M | 128 | 19.5646 | 0.162198 |
| SwiGLU-3 | Ranked singleton-KL widths, `1e-5` | 50% | 32.35% | 100M | 128 | 22.0908 | 0.220696 |
| SwiGLU-4 | Matched 50% control, constant `3e-5` | 50% | 32.35% | 10M | 128 | 24.0373 | 0.266991 |
| SwiGLU-5 | Discrete widths, constant `3e-5` | 20% | 12.94% | 100M | 128 | 15.9491 | 0.074664 |
| SwiGLU-5 | Discrete widths, constant `3e-5` | 50% | 32.35% | 100M | 128 | 20.5286 | 0.194910 |
| **SwiGLU-6** | **Discrete widths, constant `3e-5`** | **20%** | **12.94%** | **1B** | **128** | **15.5993** | **0.062734** |
| **SwiGLU-6** | **Discrete widths, constant `3e-5`** | **50%** | **32.35%** | **1B** | **128** | **18.4678** | **0.142176** |

SwiGLU-6 exactly reproduced both historical SwiGLU-5 100M measurements before
continuing to 1B. From 100M to 1B, fixed KL fell by 15.98% at 20% removal and
27.06% at 50% removal.

## Earlier 50% construction results

These rows use the earlier teacher-KL protocol. Compare them with each other,
not numerically with the fixed C4 KL above.

| Experiment | Construction | Recovery budget | WikiText prefix PPL ↓ | Teacher KL ↓ |
| --- | --- | ---: | ---: | ---: |
| SwiGLU-1 | Random initialization, uniform widths | One short epoch | 46,023.14 | 8.0831 |
| SwiGLU-1 | Teacher-neuron initialization, uniform widths | One short epoch | 32.6286 | 0.7436 |
| SwiGLU-2 | Singleton-KL allocation at 25% probe width | One short epoch | **29.5973** | **0.6493** |

## Final full-corpus evaluation

SwiGLU-6 evaluates fixed 100M and 1B endpoints on complete WikiText-2
validation and test splits. The table below reports held-out test results and
the unweighted macro accuracy over the five zero-shot tasks in the next table.

| Model | Eligible-MLP removal | Whole-model removal | Recovery tokens | Test PPL, context 128 ↓ | Test PPL, context 2048 ↓ | Task macro accuracy ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 0% | 0% | — | **11.9820** | **7.2670** | **66.998%** |
| SwiGLU-6 20%, 100M | 20% | 12.94% | 100M | 13.4301 | 8.1157 | 63.259% |
| **SwiGLU-6 20%, 1B** | **20%** | **12.94%** | **1B** | **13.0840** | **7.9687** | **64.392%** |
| SwiGLU-6 50%, 100M | 50% | 32.35% | 100M | 17.5579 | 10.2235 | 56.549% |
| **SwiGLU-6 50%, 1B** | **50%** | **32.35%** | **1B** | **15.8293** | **9.4992** | **59.221%** |

The 1B endpoint improves test PPL at both evaluated context lengths. Relative
to 100M, macro accuracy rises by 1.133 percentage points at 20% removal and
2.672 points at 50% removal. Neither compressed endpoint reaches dense quality.

## Zero-shot benchmark accuracy

All tasks use the frozen `lm_eval==0.4.13` protocol, native zero-shot prompts,
batch size 1, and a 2,048-token context limit. PIQA, ARC-Easy, ARC-Challenge,
and HellaSwag report normalized accuracy; WinoGrande reports accuracy.

| Model | PIQA | ARC-Easy | ARC-Challenge | WinoGrande | HellaSwag | Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | **77.69%** | **73.32%** | **47.27%** | **65.35%** | **71.36%** | **67.00%** |
| SwiGLU-6 20%, 100M | 76.66% | 68.94% | 41.47% | 62.59% | 66.64% | 63.26% |
| **SwiGLU-6 20%, 1B** | **76.61%** | **71.34%** | **43.52%** | **62.75%** | **67.76%** | **64.39%** |
| SwiGLU-6 50%, 100M | 73.61% | 59.51% | 32.94% | 57.85% | 58.83% | 56.55% |
| **SwiGLU-6 50%, 1B** | **75.14%** | **63.89%** | **35.32%** | **59.83%** | **61.93%** | **59.22%** |

At 20% removal, 1B improves four of five tasks; PIQA changes by one fewer
correct example. At 50%, all five tasks improve. The paired 95% intervals
against dense exclude zero for every compressed-model task result. The
artifacts do not provide a direct paired 100M-versus-1B confidence interval or
training-seed uncertainty.

## Deployment footprint

Recovery length does not change model topology, so the 100M and 1B endpoints
at a given target have the same parameter footprint.

| Topology | Parameters | Parameters removed | Whole-model removal | BF16 bundle | Resident GPU parameter memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 1,711,376,384 | — | 0% | 3.191 GiB | 3.188 GiB |
| 20% eligible-MLP removal | 1,489,915,904 | 221,460,480 | 12.94% | 2.779 GiB | 2.778 GiB |
| 50% eligible-MLP removal | 1,157,728,256 | 553,648,128 | 32.35% | 2.160 GiB | 2.156 GiB |

These are model-bundle and fresh-process resident-weight measurements. They do
not include a serving KV cache and do not establish latency or throughput.

## Recorded SwiGLU-6 resource demand

| Target | New recovery time | New-token throughput | Peak RAM | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| 20% | 29.06 h | 9,510 tokens/s | 7.83 GiB | 13.62 GiB |
| 50% | 30.00 h | 9,212 tokens/s | 9.18 GiB | 14.71 GiB |

Both recoveries ran sequentially on one NVIDIA GeForce RTX 4090. The final
five-model evaluation took approximately 2 hours 54 minutes of wall time.

## Reading boundaries

- Historical prefix PPL and full-corpus PPL are different measurements and
  occupy separate tables.
- Early teacher KL and later fixed C4 recovery-validation KL are not one
  continuous metric series.
- Bootstrap intervals describe evaluation-example uncertainty, not variation
  across independent recovery seeds.
- The final test split was evaluated after architectures and endpoints were
  fixed; it was not used for training or model selection.
- The reported memory reductions cover resident model parameters. SwiGLU-6
  does not contain a serving-latency benchmark.
