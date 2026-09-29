---
metadata_version: 1
title: TUKE Perun Resource Planning and Accounting
type: procedure
category: infrastructure/perun
status: active
created: 2026-09-29
modified: 2026-09-29
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---

# TUKE Perun Resource Planning and Accounting

## Purpose

Every experiment executed on TUKE Perun receives:

1. a pre-run resource estimate;
2. a record for every submitted job, including failures and retries; and
3. a post-run comparison of estimated and measured use.

This is the canonical accounting contract. Experiment documents own the
experiment-level analysis, [perun-log.md](perun-log.md) owns job-level facts,
and [perun-status.md](perun-status.md) owns cumulative project totals.

## Measurement classes

Do not merge these quantities:

| Class | Meaning |
| --- | --- |
| Requested ceiling | Maximum capacity requested from Slurm; not consumption |
| Expected consumption | Pre-run estimate based on comparable evidence |
| Scheduler allocation | Allocated resources multiplied by scheduler elapsed time |
| Workflow timing | Time recorded inside the scientific process |
| Observed utilization | CPU, RAM, GPU, and VRAM telemetry measured during execution |

If scheduler elapsed time is unavailable, report workflow timing as a proxy or
lower bound. Never label it scheduler allocation or billing consumption.

## Pre-run resource plan

Record this before submission:

| Field | Required content |
| --- | --- |
| Work decomposition | Stages, task count, array mapping, dependencies |
| Parallelism | Maximum concurrent tasks and whether tasks communicate |
| Per-task request | Partition, nodes, GPUs, CPUs, RAM, wall-time limit |
| Expected elapsed | Range per task and expected critical-path runtime |
| GPU cost | Expected GPU-hours and requested GPU-hour ceiling |
| CPU cost | Expected allocated CPU core-hours |
| Memory | Expected peak RAM and VRAM per task |
| Storage | Persistent inputs, peak SCRATCH, durable PROJECT output, local export |
| Evidence | Prior run, benchmark, analytical estimate, or explicit assumption |
| Confidence | Low, medium, or high, with the main uncertainty |

For heterogeneous stages or strategies, estimate each group separately.
Array concurrency changes time to completion but does not by itself change
total expected GPU-hours.

## Calculations

For each task:

- GPU-hours = allocated GPUs multiplied by elapsed hours.
- CPU core-hours = allocated CPU cores multiplied by elapsed hours.
- Requested GPU-hour ceiling = requested GPUs multiplied by the wall-time
  limit.

Sum across every task and attempt, including failed, timed-out, canceled, and
resumed jobs. Keep requested ceilings separate from expected and measured
consumption.

When available, also report:

- CPU time divided by allocated CPU core-time;
- mean and peak GPU utilization;
- peak process or job RSS;
- peak allocated and reserved VRAM;
- useful work per GPU-hour, such as recovery tokens per GPU-hour; and
- durable output bytes per completed run.

## Post-run record

After completion, collect:

| Field | Preferred evidence |
| --- | --- |
| State, exit code, reason | `scontrol` or scheduler record |
| Queue and allocation elapsed time | Scheduler record or PERUN support |
| Workflow elapsed time | Structured `run.json` or `result.json` |
| Peak RAM and VRAM | Job telemetry or structured workflow artifact |
| CPU/GPU utilization | Scheduler or explicit telemetry |
| Persistent output | Verified PROJECT path and `du` |
| Peak SCRATCH | Job telemetry; otherwise `Unknown` |
| Provenance | Git commit, configuration hash, prepared/input hash |

PERUN currently disables `sacct` for regular users. Preserve `scontrol`
records while available and use `Unknown` when neither the scheduler,
workflow, nor support provides a measurement.

## Required analysis

The final experiment report or notebook includes a compact **Compute and
Deployment** section containing:

- planned versus measured GPU-hours and elapsed time;
- allocated CPU core-hours and observed CPU utilization when available;
- requested RAM versus peak RAM;
- GPU model and requested GPUs versus peak VRAM and utilization;
- estimated versus observed SCRATCH, PROJECT, and export size;
- failures, retries, and their resource cost;
- throughput normalized by GPU where meaningful; and
- changes recommended for the next job request.

Do not infer runtime speed, utilization, or billing from parameter count,
requested resources, hardware specifications, or a single `nvidia-smi`
snapshot.

## Recording workflow

Before submission:

1. Put the resource plan in the experiment document.
2. Check the current capacity and cumulative use in `perun-status.md`.
3. Use the plan to choose the job request; preserve headroom explicitly.

After every submitted job:

1. Append or complete its entry in `perun-log.md`.
2. Update cumulative totals in `perun-status.md` from logged measurements.
3. Add the experiment-level resource analysis after the scientific artifacts
   and stage-out are verified.
4. Record estimate variance and reuse it when planning the next comparable
   experiment.

Agents handling an approved Perun planning or reporting task must follow this
sequence and preserve `Unknown` values rather than manufacturing measurements.

## Related documentation

- [PERUN operating contract](perun.md)
- [PERUN deployment blueprint](perun-deployment.md)
- [PERUN experiment log](perun-log.md)
- [PERUN project status](perun-status.md)
