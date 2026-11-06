---
title: TUKE Perun Experiment Log
status: active
created: 2026-09-25
modified: 2026-11-06
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

- the pre-run estimate, its evidence basis and confidence, and a link to the
  experiment's resource plan;
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

- Plan: <estimate basis/confidence>; expected elapsed <value>; expected GPU-hours <value>; expected CPU core-hours <value>; RAM/VRAM <values>; SCRATCH/PROJECT/output <values>
- Outcome: <state>; exit code <code>; reason <reason or none>
- Submission: <workflow/module>; <stage>; <strategy/target if applicable>
- Provenance: commit <hash>; config <path and SHA-256>; input/prepared SHA-256 <hash>
- Allocation: <partition>; <nodes> node; <GPU count/type>; <CPUs>; <requested memory>; <wall-time limit>
- Consumption: queue <duration>; elapsed <duration>; requested GPU-hour ceiling <value>; allocation GPU-hours <value>; CPU core-hours <value>; MaxRSS <value>; peak GPU memory <value>
- Efficiency: CPU <value>; memory <value>; GPU utilization <value or Unknown>
- Outputs: <durable bytes>; <HOME/PROJECT/stage-out category>; `result.json` <path/hash>; `run.json` <path/hash>; logs <paths>
- Continuation: <not resumable/resumable/completed>; checkpoint <path/hash or none>
- Verification: <artifact/hash/stage-out checks performed>
- Variance: <actual-versus-plan differences, cause if known, and implication for the next request>
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
- Experiment document: [SwiGLU-7](../../experiments/model/swiglu/swiglu-7.md)
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
- Experiment document: [SwiGLU-7](../../experiments/model/swiglu/swiglu-7.md)
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
- Experiment document: [SwiGLU-7](../../experiments/model/swiglu/swiglu-7.md)
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
- Experiment document: [SwiGLU-7](../../experiments/model/swiglu/swiglu-7.md)
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
- Experiment document: [SwiGLU-7](../../experiments/model/swiglu/swiglu-7.md)
- Notes: Demonstrated manual checkout staging, execution from SCRATCH, explicit PROJECT stage-out, verification, and scoped cleanup without a GPU allocation.

## [2026-09-27 23:49 cluster time] 91995 | SwiGLU-7 preparation

- Outcome: FAILED; exit code Unknown because the controller record expired; reason `ModuleNotFoundError: No module named 'accelerate'`
- Submission: `workflows.runs.model.swiglu.swiglu_7`; preparation; identity `prepare-001`
- Provenance: commit Unknown; config `workflows/configs/model/swiglu/swiglu-7.json`; hashes retained in the structured artifact
- Allocation: `gpu_long`; 1 node (`gpu01`); 1 NVIDIA H200; 8 CPUs; 128 GB requested memory; 48-hour limit
- Consumption: queue Unknown; elapsed Unknown; requested GPU-hour ceiling 48; allocation GPU-hours Unknown; CPU core-hours Unknown; MaxRSS Unknown; peak GPU memory Unknown
- Efficiency: CPU Unknown; memory Unknown; GPU utilization Unknown
- Outputs: durable bytes Unknown; PROJECT stage-out; `result.json` and `run.json` under `perun-results/swiglu-7/prepare-001/`; scheduler logs `swiglu-7_91995.out` and `swiglu-7_91995.err`
- Continuation: resumable; completed preparation artifacts were retained; no optimizer checkpoint applies to this stage
- Verification: structured status was `failed`; persistent output was reported; the PROJECT lock was absent after verified stage-out
- Experiment document: [SwiGLU-7](../../experiments/model/swiglu/swiglu-7.md)
- Notes: PyTorch 2.11.0+cu128 ran on an H200 and the model and datasets loaded. Width curves completed before the first `HFLM` benchmark import exposed the missing optional dependency. Workflow telemetry reported 41.81 GiB peak process RAM. Install `accelerate>=0.26.0`, verify the exact `HFLM` import, and use `prepare --resume`.

## [2026-09-28 09:49 cluster time] 92265 | SwiGLU-7 preparation resume

- Outcome: COMPLETED; exit code Unknown because the controller record expired; reason none
- Submission: `workflows.runs.model.swiglu.swiglu_7`; preparation with `--resume`; identity `prepare-001`
- Provenance: commit Unknown; config `workflows/configs/model/swiglu/swiglu-7.json`; hashes retained in the structured artifact
- Allocation: `gpu_long`; 1 node (`gpu02`); 1 NVIDIA H200; 8 CPUs; 128 GB requested memory; 48-hour limit
- Consumption: queue Unknown; elapsed Unknown; requested GPU-hour ceiling 48; allocation GPU-hours Unknown; CPU core-hours Unknown; MaxRSS Unknown; peak GPU memory Unknown
- Efficiency: CPU Unknown; memory Unknown; GPU utilization Unknown
- Outputs: durable bytes Unknown; PROJECT `perun-results/swiglu-7/prepare-001/`; scheduler logs `swiglu-7_92265.out` and `swiglu-7_92265.err`
- Continuation: completed; no preparation resume remains necessary
- Verification: structured status `completed`; structured error `None`; PROJECT lock absent after verified stage-out
- Experiment document: [SwiGLU-7](../../experiments/model/swiglu/swiglu-7.md)
- Notes: The resumed job reused prior preparation artifacts, imported the `HFLM` backend successfully, completed the pinned dense benchmark evaluations, and finalized the shared preparation artifact.

## [2026-10-11 about 19:30 cluster time] 105196 | SwiGLU-8 Part A block check (array 0-14)

- Plan: SwiGLU-7 fit timings, medium confidence; expected 1-3 h per task and about 2-3 h in total; expected 16-31 GPU-hours; CPU core-hours 128-248; at most 50 GB RAM and 40 GB VRAM per task; SCRATCH about 2 GB per task; PROJECT under 0.1 GB
- Outcome: COMPLETED, all 15 tasks; exit codes Unknown; reason none
- Submission: `workflows.runs.model.swiglu.swiglu_8`; `block-check --task-index 0-14`, one Part A setting per task; outputs `block-check-001-task-00` to `-14`
- Provenance: commit Unknown (the PERUN checkout had pulled `llm-wiki-claude` through the parallel block check); config `workflows/configs/model/swiglu/swiglu-8.json` with `part_b` null; configuration and code hashes are recorded in every task artifact, and the merge verified that they are identical
- Allocation: `gpu_short`; 1 node per task (`gpu02`, `gpu03`, `gpu05`, `gpu12`); 1 NVIDIA H200, 8 CPUs, and 128 GB requested memory per task; 6-hour limit
- Consumption: tasks 0-9 started at once, 10-11 about 2 minutes later, and 12-14 after slots freed; the last task (2, `c1e-4-128`) ran for more than 59 minutes; workflow elapsed per task 0.02-1.04 h (artifact timestamps), 7.91 H200-hours summed; requested GPU-hour ceiling 90; allocation GPU-hours Unknown; CPU core-hours Unknown; MaxRSS Unknown; peak GPU memory Unknown
- Efficiency: CPU Unknown; memory Unknown; GPU utilization Unknown
- Outputs: PROJECT `perun-results/swiglu-8/block-check-001-task-00` to `-14`, and the login-node merge `block-check-001/result.json` (sha256 `a268f1b0537579580eac5fe908ac6fccb06cf6912688a57a4db972fa28c32724`); scheduler logs `swiglu-8_105196_<task>.out` and `.err`
- Continuation: completed; no checkpoint
- Verification: no task stderr contained `Traceback` or `AcceptanceFailure` during the run. The merge confirmed that all 15 tasks completed with identical configuration, code, environment, data streams, and rankings. The anchor checks P5, A1, and A2-A3 passed
- Variance: finished in about 1.1 h instead of the planned 2-3 h. Setup, scoring, and the low-learning-rate fits were faster than estimated. Concurrency was capped at 12 running jobs (`AssocMaxJobsLimit`), after a brief `AssocGrpNodeLimit` held the first wave at 10. Plan later arrays for 12 concurrent tasks
- Experiment document: [SwiGLU-8](../../experiments/model/swiglu/swiglu-8.md#part-a-results-2026-10-11)
- Notes: The outcome kept `c1e-3-128` for Part B, and the stage-2 indicator fired; see the decision log, 2026-10-11.

## [2026-10-11 evening, cluster time unrecorded] 105291 | SwiGLU-8 Part B preparation

- Plan: SwiGLU-7 preparation timing, medium-low confidence; expected 3-8 h; expected 3-8 GPU-hours; CPU core-hours 24-64; at most 60 GB RAM and 40 GB VRAM; PROJECT about 25 GB
- Outcome: COMPLETED; exit code Unknown; reason none
- Submission: `workflows.runs.model.swiglu.swiglu_8`; `prepare`; identity `prepare-001`; submitted together with training array 105292, which depended on it (`afterok`)
- Provenance: commit `c53cbe7` (`part_b` = `c1e-3-128`, block check sha256 `a268f1b0...`); config and code hashes recorded in the artifact
- Allocation: `gpu_short`; 1 node (`gpu01`); 1 NVIDIA H200; 8 CPUs; 128 GB requested memory; 24-hour limit
- Consumption: queue about 2 minutes; workflow elapsed 3.36 h (artifact timestamps 20:35-23:57 UTC); requested GPU-hour ceiling 24; allocation GPU-hours Unknown; CPU core-hours Unknown; MaxRSS Unknown; peak GPU memory Unknown
- Efficiency: CPU Unknown; memory Unknown; GPU utilization Unknown
- Outputs: PROJECT `perun-results/swiglu-8/prepare-001/` (`result.json`, `run.json`, `prepared/` with the 100M stream and the 16 starting models); scheduler logs `swiglu-8_105291.out` and `.err`
- Continuation: completed
- Verification: structured status `completed`; checks P1-P5 passed (training requires them); no PROJECT lock left after stage-out
- Variance: 3.36 h against the planned 3-8 h
- Experiment document: [SwiGLU-8](../../experiments/model/swiglu/swiglu-8.md)
- Notes: Harmless stderr lines: `fatal: not a git repository`, because the evaluation harness looks for git metadata in the scratch copy, and lm-eval's "already-initialized model" warnings.

## [2026-10-11 evening, cluster time unrecorded] 105292 | SwiGLU-8 Part B training (array 0-15)

- Plan: SwiGLU-7 S7-0 timing, high confidence; expected 1.3-1.6 h per task; expected 20-24 GPU-hours; 12 tasks at once, then 4
- Outcome: 12 tasks COMPLETED (all 8 slice-only runs, and fitted at 60-80%). Tasks 0-3 (fitted 20-50%) FAILED with `AcceptanceFailure`: the B2 reproduction check (1e-3) stopped them at 10M (tasks 0-2) and 25M (task 3). Relative differences from SwiGLU-7 were 0.02-0.21%
- Submission: `workflows.runs.model.swiglu.swiglu_8`; `train`; `--dependency=afterok:105291 --kill-on-invalid-dep=yes`
- Provenance: commit `c53cbe7`; prepared input `prepare-001/result.json`
- Allocation: `gpu_short`; 1 node, 1 NVIDIA H200, 8 CPUs, and 128 GB requested memory per task; 6-hour limit; nodes Unknown
- Consumption: workflow elapsed 1.18-1.33 h for each of the 12 completed tasks (0.91-1.05 h training, 0.23-0.24 h final evaluation) and 0.13-0.28 h for the 4 stopped tasks; requested GPU-hour ceiling 96; allocation GPU-hours Unknown; CPU core-hours Unknown; MaxRSS Unknown; peak GPU memory Unknown
- Efficiency: CPU Unknown; memory Unknown; GPU utilization Unknown
- Outputs: PROJECT `perun-results/swiglu-8/{fitted,slice}-target-<t>-run-001/`; the four stopped fitted runs are kept as `...-run-001-b2-stopped`, including their 10M or 25M checkpoint
- Continuation: completed tasks are final. The stopped tasks are rerun from token 0 after the B2 change (decision log, 2026-10-12); resuming is refused because the code changed
- Verification: no PROJECT lock remained after stage-out; status summary read with runbook block C5
- Variance: four tasks stopped early by an over-strict tolerance, not by an execution fault
- Experiment document: [SwiGLU-8](../../experiments/model/swiglu/swiglu-8.md)
- Notes: Token-0 values matched SwiGLU-7 exactly. The later drift is attributed to non-repeatable 8K-sequence GPU training, pending the rerun's own run-to-run comparison.

## [2026-10-12 cluster time unrecorded] 106606 | SwiGLU-8 Part B rerun of fitted 20-50% (array 0-3)

- Plan: S7-0 timing, high confidence; expected 1.3-1.6 h per task; about 6-8 GPU-hours
- Outcome: COMPLETED, all 4 tasks; exit codes Unknown; reason none
- Submission: `workflows.runs.model.swiglu.swiglu_8`; `train --array=0,1,2,3` after the stopped first attempts were renamed to `...-run-001-b2-stopped` (runbook 3.8)
- Provenance: commit `86291cf`, in which B2 records and is reviewed at 1% and B1 still stops; prepared input `prepare-001/result.json` unchanged
- Allocation: `gpu_short`; all 4 tasks on `gpu01`; 1 NVIDIA H200, 8 CPUs, and 128 GB requested memory each; 6-hour limit
- Consumption: workflow elapsed 1.26-1.32 h per task (artifact timestamps), about 5.2 H200-hours in total; allocation GPU-hours Unknown; CPU core-hours Unknown; MaxRSS Unknown; peak GPU memory Unknown
- Efficiency: CPU Unknown; memory Unknown; GPU utilization Unknown
- Outputs: PROJECT `perun-results/swiglu-8/fitted-target-{0.2,0.3,0.4,0.5}-run-001/`
- Continuation: completed
- Verification: B1 exact at token 0. B2: 34 of 36 values within 1%; two single-checkpoint flags (1.40% and 3.54%) reviewed as noise. Rerun against first attempt at 10M differs by 0.01-0.41%, which shows run-to-run noise
- Variance: none against the plan
- Experiment document: [SwiGLU-8](../../experiments/model/swiglu/swiglu-8.md)
- Notes: While the 4 tasks ran, every new `sbatch` (including `--test-only`) was refused with "your total limit is 4 nodes". This is a submission-time check separate from MaxJobs=12 and MaxSubmitJobs=48; see the status page.

## [2026-10-12 19:00 UTC] 106650 | Quantization-1 preparation

- Plan: CPU tokenization only; expected minutes; 1-hour limit; 4 CPU core-hours ceiling
- Outcome: COMPLETED; exit code Unknown; reason none
- Submission: `workflows.runs.model.baseline.quantization`; `prepare prepare-001`
- Provenance: commit `a16d776` (version-only preflight on CPU nodes)
- Allocation: `cpu_short`; 1 node; 4 CPUs; 32 GB requested memory; no GPU (`--gres=none`)
- Consumption: workflow elapsed 0.03 h (artifact timestamps 19:00-19:02 UTC); CPU core-hours Unknown; MaxRSS Unknown
- Efficiency: CPU Unknown; memory Unknown
- Outputs: PROJECT `perun-results/quantization-1/prepare-001/` (5 MB)
- Continuation: completed
- Verification: structured status `completed`; output staged out and checksum-verified
- Variance: none against the plan
- Experiment document: [Quantization-1](../../experiments/model/baseline/quantization-1.md)
- Notes: The five model tasks were submitted without a dependency after preparation had left the queue.

## [2026-10-12 19:22 UTC] 106658 | Quantization-1 models, tasks 0-3 (dense, INT8, INT4, QAT 100M)

- Plan: 0.5-2 h for each of tasks 0-2 and 1.5-3 h for task 3; 8-hour limit each
- Outcome: task 0 (dense) COMPLETED. Tasks 1-3 FAILED with `RuntimeError` at the fresh-process reload of the saved quantized bundle (`"cuda:0"` against `"cuda"` storage device in `load_quantized_bundle`), after the bundle was saved
- Submission: `workflows.runs.model.baseline.quantization`; `run-array --array=0-3`
- Provenance: commit `a16d776`; prepared input `prepare-001/result.json`
- Allocation: `gpu_short`; 1 NVIDIA H200, 8 CPUs, and 128 GB requested memory per task; nodes Unknown
- Consumption: workflow elapsed: dense 0.29 h; INT8 and INT4 0.01 h each; QAT 100M 1.63 h (training 1.51 h for 100M tokens); requested GPU-hour ceiling 32; allocation GPU-hours Unknown; peak GPU memory 15.4 GiB for dense, Unknown for the failed tasks
- Efficiency: CPU Unknown; memory Unknown; GPU utilization Unknown
- Outputs: PROJECT `perun-results/quantization-1/{dense,int8,int4,qat-int4-100m}-001/`; the QAT folder keeps its 100M checkpoint (about 13 GB)
- Continuation: the three saved quantized bundles were evaluated by jobs 108470-108472; a resume was impossible because the run fingerprint hashes all of `src/`
- Verification: dense status `completed`; QAT 100M training stream identical to SwiGLU-8's 100M stream (sha256)
- Variance: three tasks failed at evaluation start because of a loader bug; their models are intact
- Experiment document: [Quantization-1](../../experiments/model/baseline/quantization-1.md#results-2026-10-13)
- Notes: The loader bug had no test coverage; quantized reload was first exercised on GPU here.

## [2026-10-12 19:22 UTC] 106659 | Quantization-1 model, task 4 (QAT 1B)

- Plan: 13-20 h, about 25 h if fake quantization doubled the cost; 36-hour limit
- Outcome: FAILED with the same reload `RuntimeError` as tasks 1-3, after 1B training and bundle export
- Submission: `workflows.runs.model.baseline.quantization`; `run-array --array=4`
- Provenance: commit `a16d776`; prepared input `prepare-001/result.json`
- Allocation: `gpu_short`; 1 NVIDIA H200, 8 CPUs, 128 GB requested memory; node Unknown
- Consumption: workflow elapsed 15.83 h (training 14.98 h for 1B tokens, about 18.5k tokens/s); requested GPU-hour ceiling 36; allocation GPU-hours Unknown; peak GPU memory Unknown
- Efficiency: CPU Unknown; memory Unknown; GPU utilization Unknown
- Outputs: PROJECT `perun-results/quantization-1/qat-int4-1b-001/` (15 GB, including the 1B checkpoint of about 13 GB)
- Continuation: the saved bundle was evaluated by job 108473
- Verification: training stream identical to SwiGLU-7's 1B stream (sha256); training completed to exactly 1B tokens
- Variance: training time within the planned 13-20 h; failed at evaluation start
- Experiment document: [Quantization-1](../../experiments/model/baseline/quantization-1.md#results-2026-10-13)
- Notes: none

## [2026-10-13 18:32 UTC] 108470-108473 | Quantization-1 evaluation of the four saved quantized bundles

- Plan: final evaluation only, about 0.3 h each; 4-hour limit each
- Outcome: all 4 COMPLETED
- Submission: `workflows.runs.model.baseline.quantization`; `evaluate <name>-eval-001 --bundle <name>-001/model --label <variant>`, one job per bundle (INT8, INT4, QAT 100M, QAT 1B)
- Provenance: commit `461ee53` (loader uses an indexed CUDA device); prepared input `prepare-001/result.json`
- Allocation: `gpu_short`; 1 NVIDIA H200, 8 CPUs, and 128 GB requested memory per job; nodes Unknown
- Consumption: workflow elapsed 0.27-0.29 h each, 1.10 h in total; requested GPU-hour ceiling 16; peak GPU memory 13.9-14.4 GiB
- Efficiency: CPU Unknown; memory Unknown; GPU utilization Unknown
- Outputs: PROJECT `perun-results/quantization-1/{int8,int4,qat-int4-100m,qat-int4-1b}-eval-001/` (64 MB each)
- Continuation: completed; the report was built on the PC (`report-001`)
- Verification: structured status `completed`; staged out and checksum-verified
- Variance: none against the plan
- Experiment document: [Quantization-1](../../experiments/model/baseline/quantization-1.md#results-2026-10-13)
- Notes: The runtime fingerprints include each GPU's UUID, so speeds across these jobs are not comparable by rule; they are reported raw.

## [2026-10-15 16:26 UTC] 110219 | Speed-1 one-GPU speed comparison

- Plan: 24 sequential measurements (12 models × 2 passes); 0.8-1.2 h expected; 3-hour limit
- Outcome: COMPLETED; reason none
- Submission: `workflows.runs.model.baseline.speed`; `speed_1.sbatch speed-001`
- Provenance: commit `b60371d`; Quantization-1 `prepare-001` prompts and 12 bundles pinned by manifest sha256
- Allocation: `gpu_short`; 1 NVIDIA H200 on gpu01, 8 CPUs, 64 GB requested memory
- Consumption: workflow elapsed 1.69 h (6,067 s); requested GPU-hour ceiling 3; allocation GPU-hours Unknown; peak process RSS 7.3 GiB; peak generation VRAM 5.22 GiB
- Efficiency: CPU Unknown; memory Unknown; GPU utilization Unknown
- Outputs: PROJECT `perun-results/speed-1/speed-001/` (JSON only, 72 KB compressed)
- Continuation: completed
- Verification: structured status `completed`; one execution fingerprint and one GPU UUID across all 24 measurements; staged out and checksum-verified
- Variance: 1.69 h against 0.8-1.2 h; about 4.2 min per measurement instead of 2-3, mostly untimed process start-up and BF16 bundle loading (43-50 s)
- Experiment document: [Speed-1](../../experiments/model/baseline/speed-1.md#results-2026-10-15)
- Notes: The job prints no per-measurement progress; progress is visible only in the scratch `result.json`.

## [2026-10-17 17:30 UTC] 117283-117285 | SwiGLU-9 1B runs, block check, and reference KL

- Plan: train 5 × 9.5-11 h (14-hour limit); block check 3 × 0.8-2.7 h (6-hour limit); reference KL 0.1-0.3 h (1-hour limit); about 51-62 GPU-hours expected, ceiling 89 ([resource plan](../../experiments/model/swiglu/swiglu-9.md#pre-run-resource-plan)); medium-high confidence for training, low for the block check
- Outcome: all 9 tasks COMPLETED with structured status `completed`; no retries
- Submission: `workflows/jobs/perun/swiglu_9.sbatch`; `train --array=0-4` (117283), `block-check --array=0-2` (117284), `reference-kl` (117285); all with `--begin=now+10minutes`, so the three submissions were accepted before any task started; submission time read from the PERUN clock (UTC+2 assumed)
- Provenance: commit `ae67eab` (code identical to `4700b19`); configuration sha256 `6afcc214…`; every SwiGLU-7 and SwiGLU-8 input matched its pinned sha256 in the pre-submission check
- Allocation: `gpu_short`; 1 NVIDIA H200 and 8 CPUs per task; 64 GB requested memory (train, reference KL) or 128 GB (block check); nodes gpu03, gpu04, gpu05
- Consumption: workflow elapsed: train 9.37-10.62 h (50.0 h in total), block check 1.26 / 2.69 / 1.48 h (5.4 h), reference KL 96 s; about 55.5 GPU-hours workflow proxy; requested GPU-hour ceiling 89; allocation GPU-hours Unknown; queue wait about 1.3 h after the begin time
- Efficiency: CPU Unknown; peak RAM 5.9-8.9 GiB and peak VRAM 25.1-31.3 GiB per training run; block check and reference KL memory not recorded; GPU utilization Unknown
- Outputs: PROJECT `perun-results/swiglu-9/` (11 GB: five 1B bundles plus JSON), including the merged `block-check-001` and `part-b-summary-001` written on the login node
- Continuation: completed; checkpoints were deleted after the results and bundles were validated
- Verification: staged out and checksum-verified by the job script; no leftover locks; C1, B1, P4, D1, D2, P5, A1, and A2-A3 passed; 2 of 45 C2 values flagged (one transient at 50M in the slice-only 50% run, reviewed in the experiment document)
- Variance: training and block check within plan; reference KL much faster than planned (96 s against 6-18 min)
- Experiment document: [SwiGLU-9](../../experiments/model/swiglu/swiglu-9.md#part-b-results-2026-10-18)
- Notes: `scontrol` records had expired by the time they were requested, so scheduler state, exit codes, and elapsed times come from the job logs and the result records.

## [2026-10-20, about 21:30 UTC] 122867-122869 | Stacking-1 orders A and B, 48 runs

- Plan: about 22-40 GPU-hours expected, ceiling 120 ([resource plan](../../experiments/model/stacking/stacking-1.md#pre-run-resource-plan)):
  - order B: 6 × 1.5-3 h (6-hour limit), low confidence;
  - order A 100M and 1B: 33 + 9 × 0.3-0.5 h (2-hour limit), medium confidence.
- Outcome: all 48 COMPLETED with structured status `completed`; no retries.
  - The three order-B INT3 results were later found scientifically invalid (INT3 fake-quantizer mismatch, 2026-10-25). The jobs themselves ran as designed.
- Submission: `workflows/jobs/perun/stacking_1.sbatch group b` (122867, 6 tasks), `group a100m` (122868, 33 tasks), and `group a1b` (122869, 9 tasks).
  - Three array submissions in one sitting with `--begin=now+10minutes`, order B first (runbook 10.2).
- Provenance: commit `d7d7c1e`; configuration sha256 `61f30ea5…`; all 29 pinned sources matched before submission.
- Allocation: `gpu_short`; 1 NVIDIA H200, 8 CPUs, 128 GB requested memory per task; nodes Unknown.
- Consumption (workflow timing):
  - 18.6 h in total: order B 1.20-1.42 h per task (7.96 h), order A 100M 0.22-0.32 h (8.41 h), order A 1B 0.22-0.32 h (2.27 h).
  - Last log writes 21:57-23:09 UTC on 2026-10-20.
  - Allocation GPU-hours Unknown.
  - Order-B peak VRAM 25.9-36.6 GiB and peak RAM 6.1-9.0 GiB; order-A peak memory not recorded.
- Efficiency: CPU Unknown; memory Unknown; GPU utilization Unknown.
- Outputs: PROJECT `perun-results/stacking-1/{a,b}-*-001/`, 78 GB in total. Analysis copy on the PC in `data/results/workflows/model/stacking-1/`.
- Continuation: completed; order-B checkpoints removed at completion.
- Verification:
  - 48 of 48 `result.json` `completed`; no leftover locks.
  - Order A passed K1, K2, K4, and K5; order B passed K3, K4, and K5.
  - No check compared the INT3 fake quantizer with the deployed INT3, which is where the order-B INT3 defect was.
- Variance: below plan (18.6 against 22-40 GPU-hours). Order-A tasks took 0.22-0.32 h against 0.3-0.5 h; order-B tasks took 1.20-1.42 h against 1.5-3 h.
- Experiment documents: [Stacking-1](../../experiments/model/stacking/stacking-1.md); results in the [Results-2 snapshot](../../results/2026-11-01-results.md).
- Notes:
  - Recorded on 2026-11-02.
  - `sacct` refused (no scheduler records) and the `scontrol` records had expired, so the times come from the job logs and result records.
  - The submission time is the researcher's note (about 23:30 local time).

## [2026-10-21, started 09:31 UTC; submission time unrecorded] 123646-123648 | Quantization-2 INT3 PTQ, INT3 QAT 100M, and INT3 QAT 1B

- Plan: PTQ 0.3-0.5 h (2-hour limit), QAT 100M 1.6-3.5 h (6-hour limit), QAT 1B 16-32 h (36-hour limit); about 18-36 GPU-hours expected, ceiling 44 ([resource plan](../../experiments/model/baseline/quantization-2.md#pre-run-resource-plan)); medium confidence for PTQ, low for QAT
- Outcome: all 3 COMPLETED with structured status `completed`; no retries. Both QAT results were later found scientifically invalid (INT3 fake-quantizer mismatch, 2026-10-25); the jobs themselves ran as designed
- Submission: `workflows/jobs/perun/quantization_1.sbatch run <output> --config workflows/configs/model/baseline/quantization-2.json --prepared <quantization-1 prepare-001>/result.json`, variant `ptq-int3` (123646) or `qat-int3` with `--qat-tokens` 100000000 (123647) and 1000000000 (123648); three single submissions with `--job-name=quantization-2`, after Stacking-1 had finished
- Provenance: commit `d7d7c1e`; configuration sha256 `650e8ca1…`; Quantization-1's `prepare-001` reused; both QAT streams matched their pinned sha256 (`570409c1…` 100M, `f78bf805…` 1B)
- Allocation: `gpu_short`; 1 NVIDIA H200, 8 CPUs, 128 GB requested memory per job; nodes Unknown
- Consumption: workflow elapsed 0.35 / 1.78 / 14.25 h (16.4 h in total); all three started at 09:31 UTC on 2026-10-21 and ended at 09:52, 11:18, and 23:46 UTC; requested GPU-hour ceiling 44; allocation GPU-hours Unknown; peak GPU memory allocated 14.4 GiB (PTQ) and 49.6 GiB (each QAT run); peak process RSS 5.7 and 14.6 GiB
- Efficiency: CPU Unknown; memory Unknown; GPU utilization Unknown
- Outputs: PROJECT `perun-results/quantization-2/{int3,qat-int3-100m,qat-int3-1b}-001/` (2.3 GB each); analysis copy on the PC in `data/results/workflows/model/quantization-2/`
- Continuation: completed; the 1B checkpoint was not kept after validation
- Verification: "Verified persistent output" logged for all three; no tracebacks; no leftover locks; stream, scope, and bundle checks passed. No check compared the INT3 fake quantizer with the deployed INT3, which is where the defect was
- Variance: PTQ and QAT 100M within plan; QAT 1B faster than planned (14.25 h against 16-32 h)
- Experiment document: [Quantization-2](../../experiments/model/baseline/quantization-2.md#results-2026-10-25)
- Notes: `scontrol` returned "Invalid job id" for all three on 2026-10-25 and `sacct` is unavailable, so the times come from the job logs and result records. Incident record: [INT3 QAT incident](../../experiments/model/baseline/quantization-2.md#int3-qat-incident-2026-10-25).

## [2026-10-25, evening; submission time unrecorded] 132436 | Stacking-1 order-B INT3 reruns as `-002`

- Plan: 3 × 1.2-1.6 h (3-hour limit), 3.6-4.8 GPU-hours expected, ceiling 9 ([resource plan](../../experiments/model/stacking/stacking-1.md#reruns-with-the-fixed-quantizer-recorded-2026-10-25)); medium confidence.
- Outcome: all 3 COMPLETED with structured status `completed`; no retries.
  - K6 passed at the start: training-time INT3 weights bitwise equal to the deployed ones. At conversion, 0 projections mismatched.
- Submission: `stacking_1.sbatch group b --attempt 002 --array=1,3,5` (runbook 10.5).
  - Same sitting as Quantization-2's INT3 QAT 100M rerun (job 132434).
- Provenance: commit `61b2b0d`; configuration sha256 `61f30ea5…`.
- Allocation: `gpu_short`; 1 NVIDIA H200, 8 CPUs, 128 GB requested memory per task; nodes Unknown.
- Consumption (workflow timing):
  - 1.21-1.43 h per task, 4.01 h in total.
  - Last log writes 22:02-22:15 UTC.
  - Allocation GPU-hours Unknown.
  - Peak VRAM 25.9-33.3 GiB; peak RAM 6.1-9.0 GiB.
- Efficiency: CPU Unknown; memory Unknown; GPU utilization Unknown.
- Outputs: PROJECT `perun-results/stacking-1/b-fitted-{0.2,0.5,0.8}-100m-int3-002/`; size not recorded. The `-001` outputs are kept as evidence. Analysis copy on the PC.
- Continuation: completed; checkpoints removed at completion.
- Verification:
  - K3, K5, and K6 passed.
  - The outputs became the in-run twins of Stacking-1 order B at 1B, pinned by sha256 (2026-11-01).
- Variance: within plan (4.01 against 3.6-4.8 GPU-hours).
- Experiment document: [Stacking-1](../../experiments/model/stacking/stacking-1.md#int3-order-b-runs-invalid-2026-10-25).
- Notes: recorded on 2026-11-02 from the job logs and result records; scheduler records unavailable.

## [2026-11-01, submission time unrecorded] 140530, 140575, 140605 | INT4 QAT fix: Quantization-1 INT4 QAT 100M and 1B, Stacking-1 order-B INT4 100M, all as `-002`

- Plan:
  - QAT 100M: 1.7-2.3 h (4-hour limit).
  - Order B: 3 × 1.2-1.6 h (3-hour limit).
  - QAT 1B: 16-17 h (24-hour limit, decided after the 100M result).
  - About 21-24 GPU-hours expected ([resource plan](../../experiments/model/baseline/quantization-1.md#int4-qat-conversion-gap-2026-11-01)); medium confidence.
- Outcome: all 5 COMPLETED with structured status `completed`; no retries.
  - Every INT4 start and conversion check passed: 66 of 66 projections bitwise equal.
- Submission:
  - `quantization_1.sbatch run qat-int4-100m-002 --variant qat-int4 --qat-tokens 100000000` (140530; runbook 4.6).
  - `stacking_1.sbatch group b --attempt 002 --array=0,2,4` (140575; runbook 10.6), after 140530's start check.
  - `quantization_1.sbatch run qat-int4-1b-002 --qat-tokens 1000000000` (140605; runbook 4.7), after 140530 completed.
- Provenance: commit `b59da5c`. Quantization-1's default configuration and `prepare-001`. Both QAT streams equal `-001` (sha256).
- Allocation: `gpu_short`; 1 NVIDIA H200, 8 CPUs, 128 GB requested memory per task; nodes gpu17 and gpu18.
- Consumption (workflow timing):
  - QAT training elapsed: 1.55 h (100M) and 15.40 h (1B).
  - Order-B tasks: 1.41 / 1.41 / 1.22 h each, QAT recovery plus evaluation.
  - About 21.0 GPU-hours workflow proxy, a lower bound: Quantization-1 evaluation time is not recorded separately.
  - Allocation GPU-hours Unknown.
- Efficiency: order-B peak RAM 6.1-8.5 GiB and peak VRAM 28.4-38.3 GiB; Quantization-1 memory not read; CPU and GPU utilization Unknown.
- Outputs: PROJECT `perun-results/quantization-1/qat-int4-{100m,1b}-002/` (1.8 GB each) and `perun-results/stacking-1/b-fitted-{0.2,0.5,0.8}-100m-int4-002/` (1.4-1.7 GB). Analysis copy on the PC.
- Continuation: completed; checkpoints removed after validation.
- Verification: staged out; the in-run bitwise checks passed; streams matched `-001`.
- Variance: within plan. The 1B run is slightly faster than `-001` (15.40 against 15.83 h).
- Experiment documents:
  - [Quantization-1](../../experiments/model/baseline/quantization-1.md#int4-qat-conversion-gap-2026-11-01)
  - [Stacking-1](../../experiments/model/stacking/stacking-1.md#order-b-int4-reruns-with-the-fixed-quantizer-2026-11-01)
- Notes:
  - `scontrol` records had expired when requested on 2026-11-02, so times come from the result records.
  - Project storage was 279 of 1,000 GB on 2026-11-02.
  - A runbook 4.6 start-check path was corrected during the run (the live record is on scratch).

## [2026-11-01, about 20:00 UTC] 140758 | Stacking-1 order B at 1B, 6 runs

- Plan: 6 × 10-12.5 h (18-hour limit), 60-75 GPU-hours expected, ceiling 108 ([resource plan](../../experiments/model/stacking/stacking-1.md#pre-run-resource-plan-recorded-2026-11-01)); medium confidence.
- Outcome: all 6 COMPLETED with structured status `completed`; no retries.
  - K3, K6 (start and conversion, 66 of 66 projections bitwise equal), and `K7-stream` passed in all six.
  - `K7-start` passed in the three INT3 tasks.
  - The recorded `K7-review` flagged two INT3 tasks, within run-to-run noise.
- Submission: `stacking_1.sbatch group b1b --array=0-5 --time=18:00:00` (runbook 10.7), one array.
- Provenance:
  - Commit `d567d7a`, which adds only docs on top of `54ba71f`; configuration sha256 `7033abd5…`.
  - All nine pinned inputs, including the 4 GB SwiGLU-7 stream, hashed OK before submission.
  - The checkout was updated while job 140605 ran from `b59da5c`. That job runs from its SCRATCH copy of the code, and no file its resume check hashes changed.
- Allocation: `gpu_short`; 1 NVIDIA H200, 8 CPUs, 128 GB requested memory per task; nodes Unknown (`--test-only` projected gpu01).
- Consumption (workflow timing):
  - 9.78-12.11 h per task (training 9.52-11.88 h, evaluation 0.21-0.33 h), 66.9 h in total.
  - Last log writes 05:51-08:11 UTC on 2026-11-02.
  - Allocation GPU-hours Unknown; about 535 CPU core-hours (8 CPUs × workflow time, derived).
  - Peak VRAM 25.96-38.31 GiB; peak RAM 6.13-9.17 GiB.
- Efficiency: CPU Unknown; GPU utilization Unknown.
- Outputs: PROJECT `perun-results/stacking-1/b-fitted-{0.2,0.5,0.8}-1b-int{4,3}-001/`, 1.4-2.1 GB each, 10.0 GB in total. PROJECT held 279 of 1,000 GB afterwards. Analysis copy on the PC.
- Continuation: completed; checkpoints removed at completion.
- Verification: "Verified persistent output" logged for all six; no tracebacks; no leftover locks.
- Variance: within plan (66.9 against 60-75 GPU-hours). Peak VRAM was up to 1.7 GiB above the 36.6 GiB estimate, which does not matter on the H200.
- Experiment document: [Stacking-1](../../experiments/model/stacking/stacking-1.md#order-b-at-1b-results-2026-11-02).
- Notes:
  - The `scontrol` records had expired when requested on 2026-11-02, so the times come from the result records and job-log write times.
  - The submission time is from the researcher's session (about 21:00 local time).

## [2026-11-04, started 16:36 UTC; submission time unrecorded] 143434, 143435 | Heterogeneous-1 sitting 1, recipe selection and baseline table

- Plan: 24 tasks of 1.2-2.4 h (4-hour limit) and one of 1.2-1.7 h (3-hour limit), 31-52 GPU-hours expected, ceiling 99 ([resource plan](../../experiments/model/heterogenous/heterogeneous-1.md#pre-run-resource-plan)); medium confidence for teacher-start fits, low for random-start epoch counts.
- Outcome: all 25 tasks completed with structured status `completed`; no retries. Data identity and checks 2, 3, 5, and 6 passed; the manual sanity gate passed.
- Submission: `heterogeneous_1.sbatch select --array=0-23 --time=04:00:00 --mem=64G` and `heterogeneous_1.sbatch baseline-table --time=03:00:00 --mem=64G` (runbook 12.2).
- Provenance: commit `fbc9142`; configuration sha256 `7d20b93e…`; torch 2.11.0+cu128, transformers 5.14.1.
- Allocation: `gpu_short`; 1 NVIDIA H200, 8 CPUs, and 64 GB requested memory per task; nodes gpu01, gpu13, gpu20, gpu23 (from the job logs).
- Consumption (workflow timing):
  - `select`: 0.81-2.53 h per task, 45.9 h in total.
  - `baseline-table`: 1.44 h.
  - Allocation GPU-hours Unknown; about 379 CPU core-hours (8 CPUs × workflow time, derived).
- Efficiency: peak RAM, peak VRAM, CPU, and GPU utilization Unknown.
- Outputs: PROJECT `perun-results/heterogeneous-1/select-001-task-NN/`, `baseline-table-001/`, and `select-001/` from the login-node merge. Analysis copy on the PC.
- Continuation: completed; JSON outputs only, no resume.
- Verification: "Verified persistent output" logged for all 25; no tracebacks or acceptance failures in the job logs.
- Variance: within plan (47.3 against 31-52 GPU-hours). The longest task ran 0.13 h above its estimate.
- Experiment document: [Heterogeneous-1](../../experiments/model/heterogenous/heterogeneous-1.md#stage-1-results-2026-11-05).
- Notes:
  - The `scontrol` records had expired when requested, so times come from the result records.
  - PERUN held the array at 15 running tasks (`%15`, not requested), so it ran in two waves.

## [2026-11-05, about 22:21 UTC] 144574, 144575 | Heterogeneous-1 sitting 2, cost tables and single-block recovery

- Plan: 22 tasks of 0.7-1.4 h and 3 of 1-1.8 h (3-hour limit), 18-35.5 GPU-hours expected, ceiling 75 ([resource plan](../../experiments/model/heterogenous/heterogeneous-1.md#pre-run-resource-plan)).
- Outcome: all 25 tasks completed with structured status `completed`; no retries. Data identity, check 6, and check 8 passed; all 60 check 8 comparisons differ by exactly 0.
- Submission: `heterogeneous_1.sbatch cost-table --array=0-21` and `heterogeneous_1.sbatch recover --array=0-2`, both `--time=03:00:00 --mem=64G` (runbook 12.5).
- Provenance: commit `cff64fd`, which pins the reviewed recipes; configuration sha256 `671e58a9…`.
- Allocation: `gpu_short`; 1 NVIDIA H200, 8 CPUs, and 64 GB requested memory per task; nodes gpu04, gpu09, gpu15, gpu19, gpu26 (from the job logs).
- Consumption (workflow timing):
  - `cost-table`: 0.77-1.63 h per task, 23.2 h in total.
  - `recover`: 1.46-1.47 h per task, 4.4 h in total.
  - Allocation GPU-hours Unknown; about 220 CPU core-hours (8 CPUs × workflow time, derived).
- Efficiency: peak RAM, peak VRAM, CPU, and GPU utilization Unknown.
- Outputs: PROJECT `perun-results/heterogeneous-1/cost-table-001-layer-NN/` and `recovery-001-layer-{03,11,19}/`. Analysis copy on the PC.
- Continuation: completed; JSON outputs only, no resume.
- Verification: "Verified persistent output" logged for all 25; no tracebacks or acceptance failures in the job logs.
- Variance: within plan (27.6 against 18-35.5 GPU-hours). The longest `cost-table` task ran 0.23 h above its estimate, inside its limit.
- Experiment document: [Heterogeneous-1](../../experiments/model/heterogenous/heterogeneous-1.md#stage-2-results-2026-11-06).
- Notes:
  - The `scontrol` records had expired when requested on 2026-11-06, so times come from the result records.
  - The submission time is inferred from the first result record.

## [2026-11-06, about 07:13 UTC] 144734 | Heterogeneous-1 sitting 3, knapsack solve

- Plan: one task of 0.3-0.7 h (2-hour limit), no GPU ([resource plan](../../experiments/model/heterogenous/heterogeneous-1.md#pre-run-resource-plan)).
- Outcome: completed with structured status `completed`; no retries. Checks 1 and 4 passed.
- Submission: `heterogeneous_1.sbatch solve --partition=cpu_short --gres=none --cpus-per-task=4 --mem=16G --time=02:00:00` (runbook 12.6).
- Provenance: commit `cff64fd`; configuration sha256 `671e58a9…`.
- Allocation: `cpu_short`; 1 node (cn11); 4 CPUs; 16 GB requested memory; no GPU.
- Consumption (workflow timing): 0.31 h; about 1.2 CPU core-hours (derived); allocation Unknown.
- Efficiency: peak RAM and CPU utilization Unknown.
- Outputs: PROJECT `perun-results/heterogeneous-1/solve-001/`. Analysis copy on the PC.
- Continuation: completed.
- Verification: "Verified persistent output" logged; the job's error log is empty of failures.
- Variance: within plan.
- Experiment document: [Heterogeneous-1](../../experiments/model/heterogenous/heterogeneous-1.md#stage-4-results-2026-11-06).
- Notes: the `scontrol` record had expired when requested on the same day, so times come from the result record.

## [2026-11-06 17:29 UTC] 145235 | Heterogeneous-1 sitting 4, token-0 assembly

- Plan: 8 tasks of 0.2-1.5 h (6-hour limit), 4-8 GPU-hours expected, ceiling 48 ([resource plan](../../experiments/model/heterogenous/heterogeneous-1.md#pre-run-resource-plan)); least certain stage.
- Outcome: all 8 tasks COMPLETED, ExitCode 0:0, structured status `completed`; no retries. Checks 7 and 8 passed; all 8 check 7 and 177 check 8 comparisons differ by exactly 0.
- Submission: `heterogeneous_1.sbatch assemble --array=0-7 --time=06:00:00 --mem=128G` (runbook 12.7).
- Provenance: commit `cff64fd`; configuration sha256 `671e58a9…`.
- Allocation: `gpu_short`; 1 NVIDIA H200, 8 CPUs, and 128 GB requested memory per task; nodes gpu05 (6 tasks) and gpu11 (2 tasks).
- Consumption:
  - Scheduler: 0.14-1.40 h per task, 4.89 GPU-hours and about 39 CPU core-hours allocated.
  - Workflow timing: 4.86 h in total.
- Efficiency: peak RAM, peak VRAM, CPU, and GPU utilization Unknown.
- Outputs: PROJECT `perun-results/heterogeneous-1/assemble-001-target-T/`, then `summary-001/` from the login-node summary. The whole experiment holds 231 MB; PROJECT held 281 of 1,000 GB afterwards. Analysis copy on the PC.
- Continuation: completed; JSON outputs only, no resume.
- Verification: "Verified persistent output" logged for all 8; no tracebacks or acceptance failures in the job logs.
- Variance: within plan (4.89 against 4-8 GPU-hours). The shortest task ran below the per-task estimate.
- Experiment document: [Heterogeneous-1](../../experiments/model/heterogenous/heterogeneous-1.md#stage-5-results-2026-11-06).
- Notes: all tasks started within a second of submission. Scheduler times are cluster local time, UTC+1.
