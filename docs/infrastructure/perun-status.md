---
metadata_version: 1
title: TUKE Perun Project Status
type: report
category: infrastructure
status: active
created: 2026-09-25
modified: 2026-09-27
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---

# TUKE Perun Project Status

## Purpose

This page is the current operational snapshot for thesis experiments on TUKE
Perun. It tracks readiness, demonstrated project capacity, planned resources,
cumulative consumption, and unresolved infrastructure questions. Stable setup
instructions belong in the [Perun infrastructure guide](perun.md). Every
submitted job must first receive an append-only entry in the [Perun experiment
log](perun-log.md); this page is then updated from that evidence.

Snapshot date: **2026-09-27**. `Available` means demonstrated for this project
account, not merely documented as a cluster feature. Account names, QoS values,
usernames, credentials, and secret-bearing paths are not recorded.

## Readiness

| Requirement | Status | Current evidence or missing item |
| --- | --- | --- |
| VPN, SSH key, and login access | Available | Interactive access to `login01` observed. |
| Active Slurm association | Available | The project account and QoS accepted three GPU submissions. |
| Project allocation and concurrency limits | Partially verified | QoS exposes 8,424 allocated GPU-hours and no explicit per-user concurrent-GPU limit; scheduler availability still applies. |
| GPU partition | Available | Three jobs received one GPU on `gpu_long`; all failed during launcher startup before Python. |
| Persistent HOME/PROJECT storage | Available | HOME and the active 1 TB PROJECT allocation were observed with `perunfsusage`. |
| Manual scratch and PROJECT stage-out | Available | PERUN support confirmed the manual contract; CPU probe 91981 copied PROJECT to SCRATCH, ran there, copied and verified its result in PROJECT, and cleaned only its job directory. |
| Python/CUDA environment | Login-node verified | Python 3.12, PyTorch 2.11.0+cu128, Transformers 5.14.1, Datasets 5.0.0, and lm-eval 0.4.13 import successfully when the Conda library directory leads `LD_LIBRARY_PATH`. Compute-node execution remains unobserved. |
| Model and dataset availability | Unknown | The pinned SmolLM2 revision, tokenizer, C4 inputs, WikiText inputs, and evaluation datasets must be readable from a compute job. |
| SwiGLU-7 runner | Ready by static inspection | Preparation now builds its own allocation curves, fitted starts, 1B-token stream, monitoring data, evaluation protocol, and dense references. It declares no prior-experiment artifact inputs. |
| SwiGLU-7 PERUN job | Ready by static inspection | `swiglu_7.sbatch` implements the demonstrated manual staging contract, fixed task `0-11` mapping, persistent PROJECT outputs, locking, verified stage-out, and resume. GPU execution remains unobserved. |
| SwiGLU-7 prerequisite artifacts | None | No SwiGLU-1 through SwiGLU-6 runtime artifact is required. The pinned model and datasets must still be available from the compute allocation. |
| Recorded Perun execution | Manual pipeline verified | Three GPU startup failures and one CPU automatic-helper probe failure preceded successful manual CPU probe 91981. No SwiGLU-7 Python stage has run yet. |

## Available capacity

This table separates cluster documentation from capacity actually granted to
the project. Replace `Unknown` only with live, non-sensitive evidence.

| Resource | Cluster-documented capacity | Project-available capacity | Evidence date |
| --- | --- | --- | --- |
| GPU hardware | NVIDIA H200; up to 8 GPUs per GPU node | One GPU allocated per failed startup job; type not directly observed | 2026-09-26 |
| GPU partitions | `gpu_short` up to 2 days; `gpu_long` up to 4 days | `gpu_long` accepted three submissions | 2026-09-26 |
| Concurrent jobs/GPUs | Cluster and QoS dependent | Unknown | Not observed |
| CPU allocation | Partition and QoS dependent | Unknown | Not observed |
| HOME storage | 500 GB documented per user | 500 GB allocation; about 665 MB used when inspected | 2026-09-26 |
| PROJECT storage | Project-specific | 1 TB allocation; writable | 2026-09-27 |
| SCRATCH storage | Temporary; no fixed quota currently documented | User root provisioned and writable; manual job child verified | 2026-09-27 |

## Cumulative consumption

Totals include every logged completed, failed, timed-out, canceled, or
out-of-memory job. Allocation GPU-hours are GPU count multiplied by elapsed
allocation time; CPU core-hours are allocated CPUs multiplied by elapsed time.
These measure reserved capacity, not device utilization. Requested ceilings
are tracked separately and are not reported as consumption.

| Measure | Recorded total |
| --- | ---: |
| Submitted jobs | 5 |
| Successful jobs | 1 |
| Failed, canceled, timed-out, or OOM jobs | 4 |
| Requested GPU-hour ceiling | 144 |
| Allocation GPU-hours consumed | 0.000833 |
| CPU core-hours consumed | At least 0.006667; probe 91981 elapsed unavailable |
| Elapsed wall time | At least 3 seconds; probe 91981 elapsed unavailable |
| Maximum observed host memory | Not observed |
| Maximum observed GPU memory | Not observed |
| Durable output bytes | One small probe text file; exact bytes unrecorded |

## SwiGLU-7 plan

SwiGLU-7 comprises one shared preparation followed by 12 independent
training/evaluation jobs: three retraining scopes at four compression targets.
The runner is single-process and single-GPU; more GPUs do not accelerate one
trajectory.

| Resource | Planned request or location | Current status |
| --- | --- | --- |
| GPU | 1 H200-class GPU via `gpu_long` | One-GPU allocations accepted; Python/CUDA execution still unverified |
| CPU | 8 CPUs per task | Unmeasured |
| Host memory | 128 GB | Unmeasured; sized for S7-1 checkpoint and Adam-state assembly |
| Wall time | 48 hours per job | Within the documented `gpu_long` ceiling; estimated above the 8-24 hour trajectory bands, but unmeasured |
| Temporary work | `/mnt/scratch/$USER/job_<job-key>/tmp/mlp-replacement/...` | Manual directory contract verified by CPU probe |
| Durable job output | `PROJECT/perun-results/swiglu-7/<run-id>` | Launcher performs and verifies manual stage-out |
| Output reserve | Trainable state, optimizer state, and final bundle estimate, plus 15% | Exact bytes depend on strategy and target |
| Prepared data | About 16-20 GB durable plus 15-20 GB temporary fitting state | Estimate from configured tensor extents; unmeasured |
| Planned allocation cost | 124-266 H200 GPU-hours for preparation plus all 12 runs | Estimate from recorded RTX 4090 SwiGLU-5/6 timings and the changed 8K workload; not consumption |
| Expected VRAM | S7-0 25-50 GB; S7-1 35-70 GB; S7-2 30-60 GB | Planning bands only; H200 supplies 141 GB HBM |

Submit preparation first and verify its hashes and stage-out location. A `%4`
array concurrency cap gives an estimated 1.5-3 days of compute after
preparation, excluding queue time. If allocation policy permits, task 0
(S7-0/20%) can establish throughput and memory evidence before the remaining
tasks are released. See the [SwiGLU-7 experiment
guide](../experiments/model/swiglu/swiglu-7.md) for the estimate basis and
scientific contract.

## Open questions

- Which Python, PyTorch, CUDA, Transformers, Datasets, and `lm_eval` versions
  work on an allocated H200?
- Can compute nodes read the required model, cache, and dataset locations?
- Is the compute-node failure to resolve the numeric user ID related to scratch
  initialization?
- Are the current memory and wall-time requests sufficient for each SwiGLU-7
  retraining scope?

## Update procedure

After any submission:

1. Append the job to `perun-log.md`, including unsuccessful jobs.
2. Record requested allocation separately from measured consumption.
3. Update the readiness and available-capacity tables only when the job or a
   live account query establishes new evidence.
4. Recalculate cumulative totals from log entries; do not estimate missing
   measurements.
5. Advance the snapshot date and `modified` metadata date.
