---
title: Speed-1 One-GPU Generation Speed Comparison
status: active
modified: 2026-10-15
---

# Speed-1

Speed-1 measures how fast the saved dense, quantized, and SwiGLU-8 models
generate text, with every model timed in **one job on one GPU**. The
Quantization-1 speeds came from a different H200 for each model, so they are
not comparable (decision log, 2026-10-11). Speed-1 removes that limitation.
Nothing is trained; the existing model bundles are only loaded and timed.

**Status (2026-10-15):** completed (job 110219); see [Results](#results-2026-10-15).

**Scope.** The decision log (2026-07-18) treats latency and throughput as
optional descriptive measurements. Whether speed becomes a Core axis, as the
draft [core-eval](../../../methodology/core-eval.md) proposes, is an open
decision. Speed-1 produces the numbers either way.

## What is measured

The Quantization-1 runtime protocol, unchanged: the runner calls
Quantization-1's `measure-runtime` command with the Quantization-1
configuration and prepared prompts. Each measurement runs in a fresh process.

| Workload | Batch | Prompt tokens | Generated tokens |
| --- | ---: | ---: | ---: |
| Short prompt, one request | 1 | 2,048 | 256 |
| Short prompt, four requests | 4 | 2,048 | 256 |
| Long prompt, one request | 1 | 7,936 | 256 |

Each workload is run twice as warm-up and then 10 times. The medians are
recorded for:
- time to first token;
- output tokens per second;
- decode tokens per second;
- peak GPU memory.

Resident GPU memory after loading is also recorded.

## Models and order

| Group | Models |
| --- | --- |
| Reference | Dense BF16 |
| Quantization-1 | INT8 PTQ, INT4 PTQ, INT4 QAT 1B |
| SwiGLU-8 fitted arm | 20, 30, 40, 50, 60, 70, 75, 80% |

The slice-only arm and INT4 QAT 100M are omitted. They have exactly the same
shapes as the fitted arm and as INT4, so their speeds are the same. Every
bundle is pinned by its `bundle.json` sha256 in
[speed-1.json](../../../../workflows/configs/model/baseline/speed-1.json) and
checked before any timing.

All 12 models are measured in a forward pass and then again in reverse order.
If the two passes disagree, the GPU or node drifted during the job. The
reported value for each model is the mean of its two pass medians. The
artifact records the largest disagreement between passes. The job also checks
that every measurement has the same execution fingerprint, which covers the
GPU, the software, the protocol and the prompts.

## Running it

From the repository root on a PERUN login node, with the environment from the
[PERUN runbook](../../../infrastructure/perun/perun-runbook.md) loaded:

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  --output="$PERUN_LOG_DIR/%x_%j.out" --error="$PERUN_LOG_DIR/%x_%j.err" \
  workflows/jobs/perun/speed_1.sbatch speed-001
```

The job reads the bundles in place from `PROJECT/perun-results/quantization-1/`
and `PROJECT/perun-results/swiglu-8/` and never writes there. It writes to
`PROJECT/perun-results/speed-1/speed-001/`:
- `result.json` (summary and checks);
- `run.json`;
- `comparison.md`;
- one `measurements/<pass>-<model>/runtime.json` per measurement.

If the job is interrupted, add `--resume` after the output name. Completed
measurements are kept; an unfinished one is repeated.

## Pre-run resource plan

| Field | Plan |
| --- | --- |
| Work decomposition | 1 job, 24 sequential measurements (12 models × 2 passes), no array |
| Per-task request | `gpu_short`, 1 H200, 8 CPUs, 64 GB RAM, 3 h limit |
| Expected elapsed | 0.8-1.2 h: about 2-3 min per measurement |
| GPU cost | 0.8-1.2 GPU-hours expected; 3 GPU-hour ceiling |
| CPU cost | 6-10 core-hours |
| Memory | Peak RAM below 10 GB (Quantization-1 measured 3.1-7.9 GB); peak VRAM below 10 GB |
| Storage | Inputs read in place (about 25 GB of bundles); output below 10 MB |
| Evidence | Quantization-1 runtime records: 15.6-22.6 s per 10 timed requests, 8-42 s to load a bundle; process start and prompt checks are estimated |
| Confidence | Medium; bundle loading from PROJECT storage is the main uncertainty |

## Limits

- Other users' jobs can share the node's CPUs and memory bandwidth. One job
  and two passes make drift visible but do not remove it.
- Peak GPU memory during generation includes the KV cache, so it grows with
  batch size and prompt length for every model alike.
- One GPU model (H200) and one software stack. The results describe this
  setup, not other hardware or inference engines.

## Results (2026-10-15)

Job 110219 at commit `b60371d`, one H200 on node gpu01, 2026-10-15
16:27-18:08 UTC. All 24 measurements share one execution fingerprint and one
GPU. Values are means of the two pass medians, from
`data/results/workflows/model/speed-1/speed-001/`.

| Model | TTFT 1 × 2,048 (ms) | TTFT 4 × 2,048 (ms) | TTFT 1 × 7,936 (ms) | Tokens/s 1 × 2,048 | Tokens/s 4 × 2,048 | Tokens/s 1 × 7,936 | Peak VRAM (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense BF16 | 19.8 | 72.4 | 80.2 | 163.6 | 648.6 | 162.2 | 5.22 |
| INT8 PTQ | 25.0 | 83.2 | 90.5 | 135.0 | 453.9 | 118.3 | 4.19 |
| INT4 PTQ | 20.5 | 75.0 | 82.8 | 153.5 | 624.5 | 152.8 | 3.70 |
| INT4 QAT 1B | 20.4 | 74.9 | 82.7 | 159.4 | 629.2 | 154.2 | 3.70 |
| SwiGLU 20% | 19.7 | 71.3 | 79.0 | 166.0 | 656.6 | 160.9 | 4.81 |
| SwiGLU 30% | 18.3 | 65.8 | 73.6 | 161.5 | 640.5 | 157.2 | 4.60 |
| SwiGLU 40% | 17.5 | 62.8 | 70.7 | 167.4 | 662.9 | 162.9 | 4.39 |
| SwiGLU 50% | 16.5 | 59.1 | 67.1 | 167.7 | 667.0 | 163.6 | 4.19 |
| SwiGLU 60% | 17.2 | 61.4 | 69.5 | 167.9 | 666.8 | 163.6 | 3.98 |
| SwiGLU 70% | 16.0 | 57.0 | 65.0 | 164.4 | 669.4 | 163.8 | 3.78 |
| SwiGLU 75% | 14.9 | 52.6 | 60.8 | 168.5 | 673.7 | 165.0 | 3.67 |
| SwiGLU 80% | 14.9 | 52.1 | 60.2 | 164.7 | 673.9 | 165.4 | 3.57 |

Peak VRAM is the largest workload (4 × 2,048 or 1 × 7,936, including the KV
cache). Tokens/s counts all 256 generated tokens per request.

| Check | Result |
| --- | --- |
| Same GPU and execution fingerprint for all 24 measurements | Yes |
| Time to first token, forward vs reverse pass | Within 0.5% for every model and workload |
| Tokens per second, forward vs reverse pass | Within 6.2%; differences below about 6% are noise |

**Findings.**

- **SwiGLU cuts time to first token.** At 1 × 7,936 tokens it is 16% faster
  than dense at 50% and 25% faster at 80%; at 4 × 2,048 it is 18% and 28%
  faster. The prompt-processing step is the most compute-bound one, so it
  follows the removed FLOPs.
- **SwiGLU does not speed up token generation.** Tokens per second stay within
  -3% to +4% of dense at every target, inside the 6% noise.
- **Quantization is never faster.** INT4 is 3-4% slower to first token and
  4-6% lower in tokens per second; INT8 is 13-26% slower to first token and
  17-30% lower in tokens per second.
- **SwiGLU 60% is slightly slower than 50%** on every workload (for example
  69.5 vs 67.1 ms). Observed, cause not investigated; its layer widths differ
  in shape from the neighbouring targets.

**Why generation does not speed up (inference, not measured).** Dense decodes
at about 6 ms per token. Reading its 3.2 GiB of weights once takes well under
1 ms on an H200, so decoding in this setup is limited by per-token overhead
(eager PyTorch, one kernel launch per operation, batch 1-4), not by FLOPs or
weight bytes. That overhead is the same for every model. An optimized engine
(CUDA graphs, fused kernels, larger batches) could change this for both
SwiGLU and INT4; Speed-1 does not test that.

**Measured resources.** Workflow time 1.69 h (6,067 s) against the planned
0.8-1.2 h. Each measurement took about 4.2 min instead of 2-3: generation
about 1-1.3 min, loading 10-15 s for quantized and 43-50 s for BF16 bundles,
and the rest in process start-up and input checks, which are not timed
separately. Peak process RSS 2.9-7.3 GiB. Output 72 KB compressed.
