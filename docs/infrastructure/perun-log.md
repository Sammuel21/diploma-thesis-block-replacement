---
metadata_version: 1
title: TUKE Perun Experiment Log
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

# TUKE Perun Experiment Log

## Purpose

This is the append-only operational history of thesis jobs submitted to TUKE
Perun. It records allocations, measured resource consumption, outcomes, and
artifact locations without duplicating scientific results. Stable operating
instructions belong in the [Perun infrastructure guide](perun.md); the current
aggregate view belongs in [Perun status](perun-status.md).

Record every submitted job, including jobs that fail before Python starts and
jobs ending in cancellation, timeout, or out-of-memory. Amend an existing entry
only to fill fields that were unavailable while the job was queued or running.
Never store usernames, account names, QoS values, credentials, tokens, SSH
material, or secret-bearing paths.

## Entry requirements

Each entry records:

- submission timestamp, Slurm job ID, state, exit code, and optional failure
  reason;
- experiment/workflow, stage, strategy, target, run identity, Git commit, and
  configuration or artifact hashes where available;
- partition, node count, GPU type/count, CPUs, requested memory, and requested
  wall time;
- queue wait, elapsed time, requested GPU-hour ceiling, allocation GPU-hours,
  CPU core-hours, MaxRSS, peak GPU memory, and efficiency measurements when
  exposed;
- output bytes, stage-out location category, structured artifact paths and
  hashes, checkpoint/resume state, and whether final outputs were verified; and
- a link to the experiment document rather than copied scientific metrics.

Use `Unknown` when the scheduler did not expose a measurement and `Not
applicable` only when the field cannot apply. Do not infer MaxRSS, GPU memory,
or consumed hours from requested limits.

After completion, collect the scheduler record with commands such as:

```bash
sacct -j <job-id> --format=JobID,JobName,State,ExitCode,Partition,AllocTRES,Elapsed,Timelimit,MaxRSS,ReqMem,Start,End
seff <job-id>
```

GPU peak memory must come from job telemetry or the workflow artifact, not from
the requested GPU model. Allocation GPU-hours are GPU count multiplied by
elapsed allocation time, and CPU core-hours are allocated CPUs multiplied by
elapsed time; neither is a utilization measurement. Record durable bytes only
after stage-out completes.

## Entry template

```markdown
## [YYYY-MM-DD HH:MM UTC] <job-id> | <workflow and run identity>

- Outcome: <state>; exit code <code>; reason <reason or none>
- Submission: <workflow/module>; <stage>; <strategy/target if applicable>
- Provenance: commit <hash>; config <path and SHA-256>; input/prepared SHA-256 <hash>
- Allocation: <partition>; <nodes> node; <GPU count/type>; <CPUs>; <requested memory>; <wall-time limit>
- Consumption: queue <duration>; elapsed <duration>; requested GPU-hour ceiling <value>; allocation GPU-hours <value>; CPU core-hours <value>; MaxRSS <value>; peak GPU memory <value>
- Efficiency: CPU <value>; memory <value>; GPU utilization <value or Unknown>
- Outputs: <durable bytes>; <HOME/PROJECT/stage-out category>; `result.json` <path/hash>; `run.json` <path/hash>; logs <paths>
- Continuation: <not resumable/resumable/completed>; checkpoint <path/hash or none>
- Verification: <artifact/hash/stage-out checks performed>
- Experiment document: <repository-relative link>
- Notes: <resource-planning or operational observation>
```

## Entries

Entries are appended below in submission order. An empty section means that no
Perun jobs have been recorded.
