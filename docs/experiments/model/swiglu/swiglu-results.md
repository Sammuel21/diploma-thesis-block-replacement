---
metadata_version: 1
title: Homogeneous SwiGLU Results Overview
type: experiment-synthesis
category: experiments/model/swiglu
status: active
created: 2026-09-21
modified: 2026-09-29
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
    - data/results/workflows/model/swiglu-7/prepare-001/result.json
    - data/results/workflows/model/swiglu-7/S7-0-target-0.2-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-0-target-0.3-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-0-target-0.4-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-0-target-0.5-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-1-target-0.2-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-1-target-0.3-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-1-target-0.4-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-1-target-0.5-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-2-target-0.2-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-2-target-0.3-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-2-target-0.4-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-2-target-0.5-run-001/result.json
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

## SwiGLU-7 native-8K production grid

SwiGLU-7 freshly rebuilt the discrete allocations and compared three retraining
scopes for one billion tokens at sequence length 8,192. All rows use the same
architecture at a given removal target, so scope changes training cost and
recovered weights rather than inference size.

| Target | Whole-model removal | Strategy | KL at 1B | Test PPL, context 8192 | Task macro |
| ---: | ---: | --- | ---: | ---: | ---: |
| 0% | 0% | Dense reference | 0 by definition | **6.9331** | **67.141%** |
| 20% | 12.94% | S7-0 replacement only | **0.060933** | **7.5553** | 63.953% |
| 20% | 12.94% | S7-1 transformer body | 0.077076 | 7.6578 | 64.169% |
| 20% | 12.94% | S7-2 MLP, RMSNorm, attention LoRA | 0.068545 | 7.6601 | **64.440%** |
| 30% | 19.41% | S7-0 replacement only | **0.085868** | **7.8683** | 62.115% |
| 30% | 19.41% | S7-1 transformer body | 0.100083 | 8.1236 | 62.271% |
| 30% | 19.41% | S7-2 MLP, RMSNorm, attention LoRA | 0.089550 | 7.9139 | **62.394%** |
| 40% | 25.88% | S7-0 replacement only | **0.116199** | **8.3096** | 60.131% |
| 40% | 25.88% | S7-1 transformer body | 0.127408 | 8.3522 | **60.197%** |
| 40% | 25.88% | S7-2 MLP, RMSNorm, attention LoRA | 0.124680 | 8.3351 | 60.132% |
| 50% | 32.35% | S7-0 replacement only | 0.153541 | 8.8551 | 58.489% |
| 50% | 32.35% | S7-1 transformer body | **0.152586** | 8.9214 | **58.679%** |
| 50% | 32.35% | S7-2 MLP, RMSNorm, attention LoRA | 0.154480 | **8.8428** | 58.555% |

Replacement-only recovery wins final KL at three of four targets and 17 of 24
full-corpus likelihood comparisons. S7-1 and S7-2 divide the four macro-task
wins, but their gains over S7-0 are at most 0.49 percentage points and have no
direct between-strategy confidence interval or training-seed replication.

| Strategy | Likelihood wins, 24 comparisons | Macro wins, 4 targets | Four-run H200 time | Mean trainable fraction |
| --- | ---: | ---: | ---: | ---: |
| **S7-0 replacement only** | **17** | 0 | **42.57 h** | **33.06%** |
| S7-1 transformer body | 0 | 2 | 51.56 h | 92.33% |
| S7-2 MLP, RMSNorm, attention LoRA | 7 | 2 | 52.65 h | 61.83% |

The supported homogeneous default is therefore S7-0. Broader retraining did
not produce a consistent likelihood improvement and cost 21% to 24% more H200
time across the four targets. This conclusion applies to the tested scopes and
single seed; it is not proof that broader adaptation can never help.

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
SwiGLU-7 additionally records 30% and 40% topologies with 1.379B and 1.268B
parameters and BF16 tensor files of 2.569 GiB and 2.363 GiB, respectively.

## Recorded SwiGLU-6 resource demand

| Target | New recovery time | New-token throughput | Peak RAM | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| 20% | 29.06 h | 9,510 tokens/s | 7.83 GiB | 13.62 GiB |
| 50% | 30.00 h | 9,212 tokens/s | 9.18 GiB | 14.71 GiB |

Both recoveries ran sequentially on one NVIDIA GeForce RTX 4090. The final
five-model evaluation took approximately 2 hours 54 minutes of wall time.

## Recorded SwiGLU-7 resource demand

| Strategy | Four-run H200 time | Longest run | Peak RAM | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| S7-0 replacement only | **42.57 h** | **10.91 h** | **8.88 GiB** | **33.52 GiB** |
| S7-1 transformer body | 51.56 h | 13.62 h | 22.57 GiB | 51.71 GiB |
| S7-2 MLP, RMSNorm, attention LoRA | 52.65 h | 13.89 h | 13.82 GiB | 48.06 GiB |

Independent preparation took 10.21 H200 hours. Preparation plus all 12
production runs consumed 156.99 measured H200 hours.

## Final homogeneous conclusion

The completed family supports a stable production recipe: independently fit
discrete per-layer widths, keep sensitive layers dense when the global budget
allows it, and recover only the replacement MLP parameters with constant
`3e-5` teacher KL. One billion recovery tokens materially improve every tested
compression target, but the dense quality gap grows with removal and is not
eliminated.

SwiGLU-7 closes the retraining-scope question for this experiment family.
Training the full transformer body or adding attention LoRA did not justify its
extra compute and memory. Future work can use replacement-only recovery as the
homogeneous control while moving the scientific question to heterogeneous
operators or another explicitly distinct compression axis.

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
- SwiGLU-6 and SwiGLU-7 are not a controlled sequence-length ablation.
  SwiGLU-7 independently refits its starting operators and uses native 8,192-
  token recovery, so cross-generation differences combine both changes.
- The local SwiGLU-7 analysis copy omits model weights and raw task records.
  Complete bundles remain in PERUN PROJECT; the local manifests retain their
  hashes.
