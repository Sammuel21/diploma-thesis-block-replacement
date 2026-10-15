---
metadata_version: 1
title: Speed-1 One-GPU Generation Speed Comparison
type: experiment-workflow
category: experiments/model/baseline
status: active
created: 2026-10-15
modified: 2026-10-15
authorship:
  created_by: llm
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---

# Speed-1

Speed-1 measures how fast the saved dense, quantized, and SwiGLU-8 models
generate text, with every model timed in **one job on one GPU**. The
Quantization-1 speeds came from a different H200 for each model, so they are
not comparable (decision log, 2026-10-11). Speed-1 removes that limitation.
Nothing is trained; the existing model bundles are only loaded and timed.

**Status (2026-10-15):** implemented, not yet submitted.

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

## Results

Not yet run.
