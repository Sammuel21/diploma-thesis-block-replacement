---
metadata_version: 1
title: TUKE Perun Experiment Log
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

Capture the live or recently completed controller record with:

```bash
scontrol show job <job-id>
```

`sacct` is disabled for regular users on the current deployment. Request
historical accounting, MaxRSS, or other unavailable statistics from PERUN
support and add them to the entry when received.

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

Entries are appended below in submission order.

## [2026-09-26 19:11 UTC] 91205 | SwiGLU-7 preparation startup

- Outcome: FAILED; exit code `1:0`; reason `NonZeroExitCode`
- Submission: `workflows.runs.model.swiglu.swiglu_7`; preparation
- Provenance: commit `cf22d76`; Python did not start
- Allocation: `gpu_long`; 1 node (`gpu04`); 1 GPU, type Unknown; 8 CPUs; 128 GB requested memory; 48-hour limit
- Consumption: queue 0 seconds; elapsed 1 second; requested GPU-hour ceiling 48; allocation GPU-hours 0.000278; CPU core-hours 0.002222; MaxRSS Unknown; peak GPU memory Unknown
- Efficiency: CPU Unknown; memory Unknown; GPU utilization Unknown
- Outputs: no structured artifact; scheduler logs archived in PROJECT storage
- Continuation: not resumable; no checkpoint
- Verification: `scontrol show job` and scheduler stderr inspected
- Experiment document: [SwiGLU-7](../experiments/model/swiglu/swiglu-7.md)
- Notes: `.activate_scratch` was absent from the PROJECT submission directory.

## [2026-09-26 19:47 UTC] 91277 | SwiGLU-7 preparation startup

- Outcome: FAILED; exit code `1:0`; reason `NonZeroExitCode`
- Submission: `workflows.runs.model.swiglu.swiglu_7`; preparation
- Provenance: base commit `e858d32` with an uncommitted launcher edit; Python did not start
- Allocation: `gpu_long`; 1 node (`gpu01`); 1 GPU, type Unknown; 8 CPUs; 128 GB requested memory; 48-hour limit
- Consumption: queue 0 seconds; elapsed 1 second; requested GPU-hour ceiling 48; allocation GPU-hours 0.000278; CPU core-hours 0.002222; MaxRSS Unknown; peak GPU memory Unknown
- Efficiency: CPU Unknown; memory Unknown; GPU utilization Unknown
- Outputs: no structured artifact; scheduler logs archived in PROJECT storage
- Continuation: not resumable; no checkpoint
- Verification: `scontrol show job` and scheduler stderr inspected
- Experiment document: [SwiGLU-7](../experiments/model/swiglu/swiglu-7.md)
- Notes: the automatic helper was removed for this attempt; `RESULTS_DIR` remained undefined. The login shell also reported that the numeric user ID could not be resolved on the compute node.

## [2026-09-26 19:59 UTC] 91278 | SwiGLU-7 preparation startup

- Outcome: FAILED; exit code `1:0`; reason `NonZeroExitCode`
- Submission: `workflows.runs.model.swiglu.swiglu_7`; preparation
- Provenance: commit `e858d32`; Python did not start
- Allocation: `gpu_long`; 1 node (`gpu01`); 1 GPU, type Unknown; 8 CPUs; 128 GB requested memory; 48-hour limit
- Consumption: queue 0 seconds; elapsed 1 second; requested GPU-hour ceiling 48; allocation GPU-hours 0.000278; CPU core-hours 0.002222; MaxRSS Unknown; peak GPU memory Unknown
- Efficiency: CPU Unknown; memory Unknown; GPU utilization Unknown
- Outputs: no structured artifact; scheduler logs retained in PROJECT storage
- Continuation: not resumable; no checkpoint
- Verification: `scontrol show job`, scheduler stderr, account identity, Slurm prolog configuration, and expected scratch paths inspected
- Experiment document: [SwiGLU-7](../experiments/model/swiglu/swiglu-7.md)
- Notes: login shell plus `source .activate_scratch` still found no helper. No documented progress log or surviving per-job scratch directory was observed.

## [2026-09-27 11:51 cluster time] 91634 | Automatic-scratch CPU probe

- Outcome: FAILED; exit code `1:0`; reason `NonZeroExitCode`
- Submission: `workflows/jobs/perun/scratch_probe.sbatch`; automatic-helper probe
- Provenance: commit `a5f0d52`; no Python or scientific workflow
- Allocation: `cpu_short`; 1 node (`cn02`); no GPU; 1 CPU; 1 GB requested memory; five-minute limit
- Consumption: queue 0 seconds; elapsed 0 seconds reported; requested GPU-hour ceiling 0; allocation GPU-hours 0; CPU core-hours 0; MaxRSS Unknown; peak GPU memory Not applicable
- Efficiency: CPU Unknown; memory Unknown; GPU utilization Not applicable
- Outputs: no structured result; scheduler logs retained in PROJECT checkout
- Continuation: not resumable; no checkpoint
- Verification: `scontrol show job`, scheduler stdout, and scheduler stderr inspected
- Experiment document: [SwiGLU-7](../experiments/model/swiglu/swiglu-7.md)
- Notes: `.activate_scratch` was absent. PERUN support later confirmed that the helper is unavailable and staging must be manual.

## [2026-09-27 time unrecorded] 91981 | Manual PROJECT-SCRATCH pipeline probe

- Outcome: COMPLETED; exit code unrecorded; reason none
- Submission: `workflows/jobs/perun/scratch_probe.sbatch`; manual staging probe
- Provenance: commit `c237809`; no Python or scientific workflow
- Allocation: `cpu_short`; 1 node (`cn02`); no GPU; 1 CPU; 1 GB requested memory; five-minute limit
- Consumption: queue Unknown; elapsed Unknown; requested GPU-hour ceiling 0; allocation GPU-hours 0; CPU core-hours Unknown; MaxRSS Unknown; peak GPU memory Not applicable
- Efficiency: CPU Unknown; memory Unknown; GPU utilization Not applicable
- Outputs: one small text record in PROJECT `perun-results/job_91981/`; exact bytes unrecorded; scheduler logs in the PROJECT checkout
- Continuation: not resumable; no checkpoint
- Verification: result reported `status=ok`; persistent PROJECT file was read successfully; marker-protected job SCRATCH was absent after verified stage-out
- Experiment document: [SwiGLU-7](../experiments/model/swiglu/swiglu-7.md)
- Notes: Demonstrated manual checkout staging, execution from SCRATCH, explicit PROJECT stage-out, verification, and scoped cleanup without a GPU allocation.
