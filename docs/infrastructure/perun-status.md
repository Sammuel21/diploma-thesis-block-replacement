---
metadata_version: 1
title: TUKE Perun Project Status
type: report
category: infrastructure
status: active
created: 2026-09-25
modified: 2026-09-25
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

Snapshot date: **2026-09-25**. `Available` means demonstrated for this project
account, not merely documented as a cluster feature. Account names, QoS values,
usernames, credentials, and secret-bearing paths are not recorded.

## Readiness

| Requirement | Status | Current evidence or missing item |
| --- | --- | --- |
| VPN, SSH key, and login access | Unknown | No account-specific observation has been recorded. |
| Active Slurm association | Unknown | Run `sacctmgr show user "$USER" withassoc format=account,qos`; record only whether a usable association exists. |
| Project allocation and concurrency limits | Unknown | Must be read from the live association/QoS and confirmed with the project owner. |
| GPU partition | Documented, not project-verified | Official documentation lists H200 nodes in `gpu_short` and `gpu_long`; live access depends on the association. |
| Persistent HOME/PROJECT storage | Documented, not project-verified | Run `perunfsusage` and select stable PROJECT locations for inputs and completed results. |
| Automatic scratch and result synchronization | Implemented, not run | The launcher uses `$TMPDIR` for disposable work and `$RESULTS_DIR` for job output. Stage-out location and failure behavior remain unobserved. |
| Python/CUDA environment | Missing evidence | No reproducible environment specification or successful PERUN import record is tracked. Supply the interpreter through `MLP_REPLACEMENT_PYTHON`. |
| Model and dataset availability | Unknown | The pinned SmolLM2 revision, tokenizer, C4 inputs, WikiText inputs, and evaluation datasets must be readable from a compute job. |
| SwiGLU-7 runner | Ready by static inspection | Preparation now builds its own allocation curves, fitted starts, 1B-token stream, monitoring data, evaluation protocol, and dense references. It declares no prior-experiment artifact inputs. |
| SwiGLU-7 PERUN job | Ready by static inspection | `swiglu_7.sbatch` provides one preparation job and deterministic tasks `0-11` for the 12-run grid, with isolated scratch work/output paths. |
| SwiGLU-7 prerequisite artifacts | None | No SwiGLU-1 through SwiGLU-6 runtime artifact is required. The pinned model and datasets must still be available from the compute allocation. |
| Recorded Perun execution | None | No submitted job or measured resource record appears in `perun-log.md`. |

## Available capacity

This table separates cluster documentation from capacity actually granted to
the project. Replace `Unknown` only with live, non-sensitive evidence.

| Resource | Cluster-documented capacity | Project-available capacity | Evidence date |
| --- | --- | --- | --- |
| GPU hardware | NVIDIA H200; up to 8 GPUs per GPU node | Unknown | Not observed |
| GPU partitions | `gpu_short` up to 2 days; `gpu_long` up to 4 days | Unknown | Not observed |
| Concurrent jobs/GPUs | Cluster and QoS dependent | Unknown | Not observed |
| CPU allocation | Partition and QoS dependent | Unknown | Not observed |
| HOME storage | 500 GB documented per user | Unknown free/used | Not observed |
| PROJECT storage | Project-specific | Unknown free/used | Not observed |
| SCRATCH storage | Temporary; no fixed quota currently documented | Unknown current use | Not observed |

## Cumulative consumption

Totals include every logged completed, failed, timed-out, canceled, or
out-of-memory job. Allocation GPU-hours are GPU count multiplied by elapsed
allocation time; CPU core-hours are allocated CPUs multiplied by elapsed time.
These measure reserved capacity, not device utilization. Requested ceilings
are tracked separately and are not reported as consumption.

| Measure | Recorded total |
| --- | ---: |
| Submitted jobs | 0 |
| Successful jobs | 0 |
| Failed, canceled, timed-out, or OOM jobs | 0 |
| Requested GPU-hour ceiling | 0 |
| Allocation GPU-hours consumed | 0 |
| CPU core-hours consumed | 0 |
| Elapsed wall time | 0 |
| Maximum observed host memory | Not observed |
| Maximum observed GPU memory | Not observed |
| Durable output bytes | 0 |

## SwiGLU-7 plan

SwiGLU-7 comprises one shared preparation followed by 12 independent
training/evaluation jobs: three retraining scopes at four compression targets.
The runner is single-process and single-GPU; more GPUs do not accelerate one
trajectory.

| Resource | Planned request or location | Current status |
| --- | --- | --- |
| GPU | 1 H200-class GPU via `gpu_long` | Project access unknown |
| CPU | 8 CPUs per task | Unmeasured |
| Host memory | 128 GB | Unmeasured; sized for S7-1 checkpoint and Adam-state assembly |
| Wall time | 48 hours per job | Within the documented `gpu_long` ceiling; estimated above the 8-24 hour trajectory bands, but unmeasured |
| Temporary work | `$TMPDIR/mlp-replacement/swiglu-7-<job-id>-<task>` | Launcher-managed and excluded from stage-out by `.rsyncignore` |
| Durable job output | `$RESULTS_DIR/swiglu-7/<run-id>`, then stable PROJECT storage | Stage-out and retained capacity unverified |
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

- Is the researcher able to log in through VPN and SSH?
- Which project association is active, and what non-sensitive concurrency and
  resource limits does it impose?
- Which Python, PyTorch, CUDA, Transformers, Datasets, and `lm_eval` versions
  work on an allocated H200?
- Can compute nodes read the required model, cache, and dataset locations?
- Where does automatic stage-out place `results_job_<job-id>` in practice?
- Does `.rsyncignore` exclude the temporary workflow subtree as intended?
- Does the epilog synchronize outputs after unsuccessful jobs?
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
