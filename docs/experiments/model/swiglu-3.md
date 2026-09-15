---
metadata_version: 1
title: SwiGLU Calibration and Token-Budget Recovery Study
type: experiment-workflow
category: experiments/model
status: draft
created: 2026-09-15
modified: 2026-09-15
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
sources:
  artifacts:
    - data/results/notebook-model-study/swiglu-2.json
  reference_artifacts:
    - data/results/notebook-model-study/swiglu-compression-optimized.json
---

# SwiGLU Calibration and Token-Budget Recovery Study

Status: implemented as a headless workflow, not yet executed or GPU-validated.
The retained `swiglu-3.ipynb` is not modified by this migration.

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

The runner is `workflows.runs.model.swiglu_3`, configured by
`workflows/configs/model/swiglu-3.json`. It writes a unique incremental science
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
nohup python -u -m workflows.runs.model.swiglu_3 \
  --output data/results/workflows/model/swiglu-3/run-001.json \
  > data/results/workflows/model/swiglu-3/run-001.log 2>&1 &
```

After an interruption, use the same output explicitly:

```bash
python -u -m workflows.runs.model.swiglu_3 \
  --output data/results/workflows/model/swiglu-3/run-001.json --resume
```

`--smoke` applies the reduced budgets embedded in the same configuration. It
is intended to validate data access, model surgery, metrics, checkpointing, and
resume before the scientific run; its numbers are not thesis results.

## Resource and validation boundary

The 393,216-pair capture for six blocks is expected to require roughly 18 GiB
of CPU activation storage before transient overhead. Recovery holds a BF16
teacher and compressed student plus FP32 replacement parameters and optimizer
state. Checkpoint storage is also substantial because reconstructable FP32
replacement states are retained at 10M and 100M. Actual RAM, VRAM, disk,
throughput, and wall time must be measured in a smoke run before scheduling the
full study.

This implementation has not been run on a project GPU or Perun. It makes no
numerical-parity, memory-fit, runtime, or cluster-readiness claim. A future
Perun job may call the portable Python entry point, but this migration adds no
Slurm harness.
