---
metadata_version: 1
title: TUKE Perun Project Status
type: report
category: infrastructure/perun
status: active
created: 2026-09-25
modified: 2026-09-28
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

Snapshot date: **2026-09-28**. `Available` means demonstrated for this project
account, not merely documented as a cluster feature. Account names, QoS values,
usernames, credentials, and secret-bearing paths are not recorded.

## Readiness

| Requirement | Status | Current evidence or missing item |
| --- | --- | --- |
| VPN, SSH key, and login access | Available | Interactive access to `login01` observed. |
| Active Slurm association | Available | The project account and QoS accepted five GPU submissions. |
| Project allocation and concurrency limits | Partially verified | QoS exposes 8,424 allocated GPU-hours and no explicit per-user concurrent-GPU limit; scheduler availability still applies. |
| GPU partition | Available | Jobs 91995 and 92265 ran PyTorch on NVIDIA H200 GPUs on `gpu_long`; training is configured for `gpu_short`. |
| Persistent HOME/PROJECT storage | Available | HOME and the active 1 TB PROJECT allocation were observed with `perunfsusage`. |
| Manual scratch and PROJECT stage-out | Available | PERUN support confirmed the manual contract; CPU probe 91981 copied PROJECT to SCRATCH, ran there, copied and verified its result in PROJECT, and cleaned only its job directory. |
| Python/CUDA environment | Available | Job 92265 verified Python 3.12, PyTorch 2.11.0+cu128 on an H200 and completed the exact `HFLM` evaluation path after `accelerate` installation. |
| Model and dataset availability | Available for preparation | Job 92265 loaded the pinned model, tokenizer, recovery/evaluation data, and benchmark datasets and completed dense evaluation. |
| SwiGLU-7 runner | Preparation completed | Job 92265 completed and validated the independent `prepare-001` artifact. Training remains unmeasured. |
| SwiGLU-7 PERUN job | Preparation path validated | Manual staging, resume, execution, PROJECT stage-out, lock removal, and structured completion worked in job 92265. The 12-task training path remains unmeasured. |
| SwiGLU-7 prerequisite artifacts | None | No SwiGLU-1 through SwiGLU-6 runtime artifact is required. The pinned model and datasets must still be available from the compute allocation. |
| Recorded Perun execution | Preparation completed | After the earlier infrastructure and dependency failures, resume job 92265 completed SwiGLU-7 preparation and dense evaluation. |

## Available capacity

This table separates cluster documentation from capacity actually granted to
the project. Replace `Unknown` only with live, non-sensitive evidence.

| Resource | Cluster-documented capacity | Project-available capacity | Evidence date |
| --- | --- | --- | --- |
| GPU hardware | NVIDIA H200; up to 8 GPUs per GPU node | One NVIDIA H200 observed directly in job 91995 | 2026-09-28 |
| GPU partitions | `gpu_short` up to 2 days; `gpu_long` up to 4 days | `gpu_long` accepted five submissions; `gpu_short` training submission remains unmeasured | 2026-09-28 |
| Concurrent jobs/GPUs | Cluster and QoS dependent | Unknown | Not observed |
| CPU allocation | Partition and QoS dependent | 8 CPUs allocated with preparation job 91995; 1 CPU with probe 91981 | 2026-09-28 |
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
| Submitted jobs | 7 |
| Successful jobs | 2 |
| Failed, canceled, timed-out, or OOM jobs | 5 |
| Requested GPU-hour ceiling | 240 |
| Allocation GPU-hours consumed | At least 0.000833; jobs 91995 and 92265 elapsed unavailable |
| CPU core-hours consumed | At least 0.006667; jobs 91981, 91995, and 92265 elapsed unavailable |
| Elapsed wall time | At least 3 seconds; jobs 91981, 91995, and 92265 elapsed unavailable |
| Maximum observed host memory | 41.81 GiB process peak during preparation job 91995 |
| Maximum observed GPU memory | Not observed |
| Durable output bytes | One small probe record plus failed preparation artifacts; exact bytes unrecorded |

## SwiGLU-7 plan

SwiGLU-7 comprises one shared preparation followed by 12 independent
training/evaluation jobs: three retraining scopes at four compression targets.
The runner is single-process and single-GPU; more GPUs do not accelerate one
trajectory.

| Resource | Planned request or location | Current status |
| --- | --- | --- |
| GPU | 1 H200-class GPU via `gpu_short` | Python/CUDA execution verified on one H200; training placement remains unmeasured |
| CPU | 8 CPUs per task | Unmeasured |
| Host memory | 128 GB | Unmeasured; sized for S7-1 checkpoint and Adam-state assembly |
| Wall time | 48 hours per task | Equal to the documented `gpu_short` ceiling; estimated above the 8-24 hour trajectory bands, but unmeasured |
| Temporary work | `/mnt/scratch/$USER/job_<job-key>/tmp/mlp-replacement/...` | Manual directory contract verified by CPU probe |
| Durable job output | `PROJECT/perun-results/swiglu-7/<run-id>` | Launcher performs and verifies manual stage-out |
| Output reserve | Trainable state, optimizer state, and final bundle estimate, plus 15% | Exact bytes depend on strategy and target |
| Prepared data | About 16-20 GB durable plus 15-20 GB temporary fitting state | `prepare-001` completed; exact durable bytes unrecorded |
| Planned allocation cost | 124-266 H200 GPU-hours for preparation plus all 12 runs | Estimate from recorded RTX 4090 SwiGLU-5/6 timings and the changed 8K workload; not consumption |
| Expected VRAM | S7-0 25-50 GB; S7-1 35-70 GB; S7-2 30-60 GB | Planning bands only; H200 supplies 141 GB HBM |

Submit preparation first and verify its hashes and stage-out location. A `%12`
array concurrency cap permits all 12 one-GPU tasks to run simultaneously and
gives an estimated 12-24 hours of training-grid compute after preparation,
excluding queue time. Scheduler availability may start fewer than 12 tasks at
once. See the [SwiGLU-7 experiment
guide](../../experiments/model/swiglu/swiglu-7.md) for the estimate basis and
scientific contract.

## Open questions

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
