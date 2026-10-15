---
title: TUKE Perun Project Status
status: active
created: 2026-09-25
modified: 2026-10-15
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
| SwiGLU-7 runner | Completed | Job 92265 completed and validated the independent `prepare-001` artifact; all 12 production runs completed (see the [SwiGLU-7 report](../../experiments/model/swiglu/swiglu-7.md#measured-execution-cost)). |
| SwiGLU-7 PERUN job | Validated | Manual staging, resume, execution, PROJECT stage-out, lock removal, and structured completion worked in job 92265; the 12-task training array completed and its outputs are in PROJECT. |
| SwiGLU-7 prerequisite artifacts | None | No SwiGLU-1 through SwiGLU-6 runtime artifact is required. The pinned model and datasets must still be available from the compute allocation. |
| Recorded Perun execution | Completed | After the earlier infrastructure and dependency failures, SwiGLU-7, SwiGLU-8, Quantization-1, and Speed-1 completed on PERUN; see the consumption table below. |

## Available capacity

This table separates cluster documentation from capacity actually granted to
the project. Replace `Unknown` only with live, non-sensitive evidence.

| Resource | Cluster-documented capacity | Project-available capacity | Evidence date |
| --- | --- | --- | --- |
| GPU hardware | NVIDIA H200; up to 8 GPUs per GPU node | One NVIDIA H200 observed directly in job 91995 | 2026-09-28 |
| GPU partitions | `gpu_short` up to 2 days; `gpu_long` up to 4 days | `gpu_long` accepted five submissions; `gpu_short` training submission remains unmeasured | 2026-09-28 |
| Concurrent jobs/GPUs | Cluster and QoS dependent | 12 running jobs, inferred from `squeue`: with 12 SwiGLU-8 block-check tasks running, the rest waited with `AssocMaxJobsLimit`. A brief `AssocGrpNodeLimit` held the first wave at 10. Not confirmed with `scontrol`. A submit-time rule also exists: with 4 running SwiGLU-8 tasks, every new `sbatch`, including `--test-only`, was refused with "your total limit is 4 nodes (you have 4 short + 0 long = 4, requesting 1)". Waiting arrays appear to count once, which is why the 15- and 16-task arrays were accepted. Submit chained work while few jobs are active. Slurm association limits read with `scontrol show assoc_mgr` on 2026-10-12: `MaxJobs=12` (running at once) and `MaxSubmitJobs=48` (running plus pending, array tasks counted individually). The "4 nodes" refusal happened at 4 of 12 and 4 of 48. The QoS shows no Grp or Max limits (`scontrol show assoc_mgr qos=... flags=qos`). The refusal is therefore a separate, undocumented PERUN submission check, not a Slurm association or QoS limit (inferred; not confirmed by support) | 2026-10-12 |
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

The rows above are calculated from [perun-log.md](perun-log.md), which does
not yet contain the 12 SwiGLU-7 training jobs. The two figures below were
added on 2026-10-10 from other sources and are not included in the totals
above. They have not been reconciled with each other or with job-level
scheduler records.

| Measure | Value | Source and class |
| --- | ---: | --- |
| Project GPU-hours consumed, PERUN dashboard | About 150 | Researcher-reported dashboard value on 2026-10-10; project-level billing total |
| SwiGLU-7 preparation plus 12 runs, workflow timing | 156.99 H200-hours | Preparation 10.21 hours plus the 12 runs, from the [SwiGLU-7 report](../../experiments/model/swiglu/swiglu-7.md) and its result artifacts; workflow-timing proxy, not scheduler allocation |
| SwiGLU-8 block check, preparation, 16 runs and 4 stopped attempts, workflow timing | about 32.1 H200-hours | Block check 7.91, preparation 3.36, runs 20.18, stopped attempts 0.68, from the [SwiGLU-8 page](../../experiments/model/swiglu/swiglu-8.md) and its artifacts; workflow-timing proxy, not scheduler allocation |
| Quantization-1 five models, failed reloads and four re-evaluations, workflow timing | about 18.9 H200-hours, plus 0.03 h CPU preparation | Dense 0.29, INT8 and INT4 first attempts 0.02, QAT 100M 1.63, QAT 1B 15.83, evaluations 1.10, from the [Quantization-1 page](../../experiments/model/baseline/quantization-1.md#results-2026-10-13) and its artifacts; workflow-timing proxy, not scheduler allocation |
| Speed-1 one-GPU speed comparison, workflow timing | about 1.7 H200-hours | Job 110219, 24 measurements, from the [Speed-1 page](../../experiments/model/baseline/speed-1.md#results-2026-10-15) and its artifacts; workflow-timing proxy, not scheduler allocation |

## Open questions

- Is the compute-node failure to resolve the numeric user ID related to scratch
  initialization?

## Update procedure

After any submission:

1. Append the job to `perun-log.md`, including unsuccessful jobs.
2. Record requested allocation separately from measured consumption.
3. Update the readiness and available-capacity tables only when the job or a
   live account query establishes new evidence.
4. Recalculate cumulative totals from log entries; do not estimate missing
   measurements.
5. Advance the snapshot date and `modified` metadata date.
