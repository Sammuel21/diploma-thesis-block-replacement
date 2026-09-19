---
metadata_version: 1
title: SwiGLU Calibration and Token-Budget Recovery Study
type: experiment-workflow
category: experiments/model/swiglu
status: draft
created: 2026-09-15
modified: 2026-09-19
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
sources:
  notebooks:
    - notebooks/model/swiglu/swiglu-3.ipynb
  artifacts:
    - data/results/notebook-model-study/swiglu-2.json
    - data/results/workflows/model/swiglu-3/run-001.json
  reference_artifacts:
    - data/results/notebook-model-study/swiglu-compression-optimized.json
---

# SwiGLU Calibration and Token-Budget Recovery Study

Status: implemented and executed as a headless workflow. A completed schema-1
run artifact is present. The retained `swiglu-3.ipynb` provides the notebook
view of the same experiment family.

## Purpose and frozen starting point

This study starts from the completed `swiglu-2` winner,
`singleton_kl_w25_t1`. The policy rank-normalizes the 25%-width singleton-KL
score and uses removal propensities `exp(-rank / 1.0)`. Layers 0 and 23 remain
protected; layers 1 through 22 are eligible. The 50% case reuses the artifact's
exact widths, whose retained-width range is 21.2036%-71.0205%.

The pinned teacher is `HuggingFaceTB/SmolLM2-1.7B` at revision
`effd688a12921b4cc83e3312b6feb579f70f9c71` (24 layers, hidden size 2,048,
SwiGLU width 8,192). The inherited 50% model has 32.3510% whole-model parameter
removal. Its recorded pre-recovery allocation-selection metrics are KL
0.652182838569085 and perplexity 29.706515026355348. Those values identify the
starting artifact; the new workflow does not treat its short earlier recovery
as a new control trajectory.

## Pipeline at a glance

```text
swiglu-2 winner: singleton_kl_w25_t1
exact 50% widths + protected layers 0 and 23
|
v
1. NESTED CALIBRATION SWEEP
   98,304 -> 196,608 -> 393,216 pairs per eligible block
   same widths, teacher initialization, optimizer, and validation roles
   -> assemble each model and measure allocation-selection teacher KL
   -> select lowest KL; ties select the smaller calibration budget
|
v
2. GLOBAL SPARSITY SWEEP
   selected calibration budget at requested removals {50%, 40%, 30%, 20%}
   -> reuse selected 50% states; fit newly required widths for other targets
   -> record requested/realized MLP and whole-model removal
   -> evaluate all four pre-recovery models
|
v
3. INDEPENDENT RECOVERY TRAJECTORIES
   one fixed compressed starting state per sparsity target
   -> replacement-only KL on the same packed 100M-token C4 stream
   -> checkpoint every 1M tokens
   -> report the in-trajectory 10M milestone and selected <=100M checkpoint
|
v
run JSON: calibration fits + sparsity curve + recovery histories + provenance
assets: fitted operators, packed tokens, and resumable model checkpoints
```

The stages are dependent: the calibration winner supplies all sparsity fits,
and each sparsity allocation supplies one independent recovery trajectory.
The four trajectories do not pass trained weights to one another.

## Stage 1: nested calibration

The workflow fits the exact winning 50% widths independently with 98,304,
196,608, and 393,216 activation pairs per eligible block. Every fit begins from
the same teacher-neuron ranking: gate/up rows and down columns are copied for
the selected neurons. Replacement parameters, fitting operations, and AdamW
state are FP32. Local fitting uses learning rate `1e-3`, batch size 2,048,
weight decay 0, at most 64 epochs, patience 3, and the initial model as an
eligible validation-best checkpoint.

The original shard-00000 roles retain this order: 48 calibration batches, 24
operator-validation batches, 64 old-recovery batches, 24 recovery-validation
batches, 336 additional-calibration batches, and 12 allocation-selection
batches. New calibration windows are appended only after all those partitions.
Thus the old 98,304-pair data, validation data, and selection data do not move.
The larger sets add prefixes of the appended partition and are strictly nested.

Each assembled model is measured on the fixed C4 allocation-selection split
(teacher KL, loss, and perplexity) and 24 WikiText-2 validation batches. The
lowest selection KL chooses the calibration budget; an exact tie chooses the
smaller budget. Histories record validation NMSE, cosine, epochs, updates, and
time. No learning-rate, data-source, initialization, or architecture sweep is
part of this stage.

## Stage 2: sparsity curve

The selected calibration budget is used at requested eligible-MLP removals of
50%, 40%, 30%, and 20%. The score, ranks, temperature, eligible layers, and
teacher-neuron rankings stay frozen. The 50% operator states are reused; every
other width is initialized from the teacher and fitted independently.

One SmolLM2 SwiGLU neuron costs 6,144 parameters. Some requested aggregate
budgets are not divisible by that step. The allocator therefore rounds the
aggregate retained width to the nearest representable whole-neuron total,
reconciles per-layer widths deterministically, and records requested and
realized eligible-MLP and whole-model removal separately. It applies no
minimum-width floor. All four assembled models receive the same pre-recovery
C4 selection and WikiText validation measurements.

## Stage 3: recovery trajectories

Each sparsity model receives one continuous 100,000,000-token
replacement-only distillation trajectory. The 10M result is the state reached
en route to 100M, rather than a separate run. Across four sparsities this is
400M recovery tokens in total.

Recovery documents come from C4 shard
`en/c4-train.00001-of-01024.json.gz`, separate from all shard-00000 fitting and
validation roles. Documents are tokenized once, separated by EOS, and packed
into a finite 100M-token int32 memory map shared by the four models. The cache
never wraps; insufficient fresh data fails the run.

CUDA forwards use BF16 autocast while replacement master parameters and AdamW
states remain FP32. Attention, embeddings, and other original parameters are
frozen. The microbatch is two 128-token sequences, with eight-way gradient
accumulation for 2,048 valid positions per complete update. AdamW uses constant
learning rate `1e-5`, zero weight decay, and `foreach=False`. The exact 100M
endpoint may use a smaller final accumulation group.

The runner saves current resumable state every requested 1M tokens and reports
milestones at requested 10M and 100M token positions. Requests are rounded up
to the next optimizer boundary; both requested and actual counts are stored. Each
milestone distinguishes metrics for the current trajectory state from the
best fixed-recovery-validation checkpoint available under that token budget.
The pre-recovery state is eligible to remain best. A current resume checkpoint
contains replacement weights, optimizer state, CPU/CUDA RNG state, token
cursor, update count, widths, pinned model identity, and configuration/reference
fingerprint. Milestone checkpoints contain reconstructable replacement states.

## Artifacts and execution

The runner is `workflows.runs.model.swiglu.calibration_recovery`, configured by
`workflows/configs/model/swiglu/calibration-recovery.json`. It writes a unique incremental science
JSON, a sibling `.run.json`, an asset directory with local operator states, the
packed-token cache, current/best recovery state, and retained 10M and 100M
milestone states.
Existing paths are refused unless `--resume` is supplied with the same output.
Resume verifies the full effective configuration and both reference file
hashes before loading state.

From the repository root on a prepared Linux GPU host:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p data/results/workflows/model/swiglu-3
nohup python -u -m workflows.runs.model.swiglu.calibration_recovery \
  --output data/results/workflows/model/swiglu-3/run-001.json \
  > data/results/workflows/model/swiglu-3/run-001.log 2>&1 &
```

After an interruption, use the same output explicitly:

```bash
python -u -m workflows.runs.model.swiglu.calibration_recovery \
  --output data/results/workflows/model/swiglu-3/run-001.json --resume
```

`--smoke` applies the reduced budgets embedded in the same configuration. It
is intended to validate data access, model surgery, metrics, checkpointing, and
resume before the scientific run; its numbers are not thesis results.

## Resource and validation boundary

The full run captures three blocks at a time. At 393,216 pairs per block,
this requires roughly 9 GiB of BF16 CPU activation storage before transient
overhead. Smaller capture groups require more teacher passes while preserving
the calibration examples and local fitting budgets. This setting does not
control model-wide recovery batching; the smoke override remains two blocks.
Recovery holds a BF16
teacher and compressed student plus FP32 replacement parameters and optimizer
state. Checkpoint storage is also substantial because reconstructable FP32
replacement states are retained at 10M and 100M. The completed artifact records
about 44.9 hours across all three stages, 37.9 GiB peak host RAM, and a maximum
20.0 GiB peak VRAM across the four recovery trajectories.

The completed artifact records a Linux run on the RTX 4090 darthmachinus host,
so the configured workflow has demonstrated local GPU and memory fit. It does
not establish numerical parity with an independently executed notebook or
Perun readiness. A future Perun job may call the portable Python entry point,
but this workflow includes no Slurm harness.
