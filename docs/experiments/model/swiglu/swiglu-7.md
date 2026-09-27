---
metadata_version: 1
title: SwiGLU-7 - Retraining-Scope Analysis
type: experiment
category: experiments/model/swiglu
status: active
created: 2026-09-24
modified: 2026-09-27
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---

# SwiGLU-7 - Retraining-Scope Analysis

Implementation and PERUN launch support are present. No SwiGLU-7 scientific
result has been produced yet.

SwiGLU-7 asks whether broader retraining improves compressed-model quality. It
is a fixed production comparison, not another candidate search.

## Independence contract

SwiGLU-7 has no runtime dependency on a SwiGLU-1 through SwiGLU-6 result,
checkpoint, prepared stream, evaluation artifact, or fitted operator. Shared
maintained Python helpers are reused as code, but all run data and model states
are built from the pinned base model and datasets by the SwiGLU-7 preparation.

Preparation performs one fixed allocation method:

1. sample the pinned C4 calibration partitions at sequence length 128;
2. fit importance-selected replacement operators at seven widths for every
   eligible layer;
3. measure each singleton replacement against the dense teacher;
4. monotonicize each layer's width/KL curve;
5. minimize summed predicted KL under each 20%, 30%, 40%, and 50% retained-
   parameter budget; and
6. freshly fit and serialize the four selected token-zero starting models.

This is the method established by the earlier study, now recomputed inside
SwiGLU-7. There is no search over allocation strategies or initializations.
The 128-token local fitting protocol is deliberately separate from the 8,192-
token production recovery protocol: independence removes prior artifacts; it
does not silently change the calibrated operator-fitting variable.

The same preparation also builds its own finite, non-repeating one-billion-
token C4 recovery stream, monitoring batches, pinned WikiText corpora, task
protocol, and dense evaluation references. The durable preparation artifact
explicitly records an empty `prior_experiment_artifacts` list.

## Fixed comparison

| ID | Trainable scope | Frozen scope |
| --- | --- | --- |
| S7-0 | Reduced-width replacement SwiGLUs | Retained MLPs, attention, RMSNorms, tied embeddings/head |
| S7-1 | All transformer-body parameters | Tied embeddings/head |
| S7-2 | All MLPs and RMSNorms, plus rank-16 LoRA on every attention q/k/v/o projection | Tied embeddings/head and attention base weights |

Each scope runs independently at 20%, 30%, 40%, and 50% eligible-MLP parameter
removal: 12 trajectories in total. The three scopes at one target share the
same freshly fitted replacement state and token-zero RNG state. No trajectory
selects, initializes, or stops another trajectory.

Recovery is fixed at one billion tokens with online dense-teacher KL,
temperature 1, no cross-entropy term, fused AdamW, constant learning rate
`3e-5`, no weight decay, seed 21, BF16 forward computation, and FP32 trainable
parameters and optimizer state. Production sequence length and effective batch
are both 8,192 tokens. The exact 100M and 1B segment boundaries use final
partial sequences of 256 and 2,304 tokens instead of repeating stream data.

## Outputs and evaluation

The shared preparation creates:

- four fresh fitted starts and their independently recomputed width curves;
- a one-billion-token signed-int32 recovery stream;
- 128-token C4/WikiText monitoring batches;
- full WikiText-2 validation and test corpora;
- dense likelihood at contexts 128, 2,048, and 8,192; and
- dense zero-shot PIQA, ARC-Easy, ARC-Challenge, WinoGrande, and HellaSwag
  records under the frozen `lm_eval==0.4.13` protocol.

Each production job retains `result.json`, `run.json`, raw benchmark records,
and one inference `model/` bundle. A single verified optimizer checkpoint is
kept while a run is incomplete and removed only after successful result and
bundle validation. Output-owned paths are relative, so a complete output
directory can be relocated and resumed.

The analysis notebook is load-only. It still expects the superseded
SwiGLU-6-derived preparation fields and must be updated to the independent
SwiGLU-7 schema before results are analyzed; notebook editing requires a
separate explicit choice under the repository instructions.

## Files

- Configuration: `workflows/configs/model/swiglu/swiglu-7.json`
- Runner: `workflows/runs/model/swiglu/swiglu_7.py`
- Local launcher: `workflows/jobs/local/run_model.sh`
- Dedicated PERUN job: `workflows/jobs/perun/swiglu_7.sbatch`
- Report notebook: `notebooks/model/swiglu/swiglu-7.ipynb`

## PERUN prerequisites

Use a clean checkout and submit from its repository root. The environment must
provide a CUDA-enabled PyTorch build that supports H200, Transformers,
Datasets, NumPy, `lm_eval==0.4.13`, `safetensors`, and `psutil`. The exact
SmolLM2, C4, WikiText, and benchmark-dataset revisions are pinned in the
configuration. Compute
nodes must be able to download those revisions or read a pre-populated
Hugging Face cache.
The runner rejects the eager attention backend: native-8K execution must load
the model with PyTorch SDPA or a supported FlashAttention implementation.

Set the project-specific values only in the shell:

```bash
export PERUN_ACCOUNT="your-project-account"
export PERUN_QOS="your-project-qos"
export PERUN_PROJECT="/mnt/project/$PERUN_ACCOUNT"

scontrol show assoc_mgr users="$USER" accounts="$PERUN_ACCOUNT" flags=assoc
scontrol show assoc_mgr qos="$PERUN_QOS" flags=qos

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlp-replacement
export MLP_REPLACEMENT_PYTHON="$(command -v python)"
export HF_HOME="$PERUN_PROJECT/huggingface-cache"
mkdir -p "$HF_HOME"
```

Confirm the selected interpreter before allocating a long job:

```bash
"$MLP_REPLACEMENT_PYTHON" -c \
  'import torch, transformers, datasets, lm_eval, safetensors, psutil; print(torch.__version__, torch.version.cuda)'
```

## Deploy

Submit one preparation job:

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  workflows/jobs/perun/swiglu_7.sbatch prepare
```

The job uses one H200, eight CPUs, 128 GB RAM, and a 48-hour limit within
PERUN's documented [`gpu_long` four-day ceiling](https://wiki.perun.tuke.sk/slurm/partitions/).
The launcher manually copies the checkout to a unique directory below
`/mnt/scratch/$USER`, runs there, then copies and verifies durable preparation
data at `$PERUN_PROJECT/perun-results/swiglu-7/prepare-001`. PERUN support
confirmed that automatic scratch activation and synchronization are not
currently available. After a successful preparation:

1. inspect the scheduler `.out` and `.err` files in the repository;
2. inspect PROJECT `result.json` and `run.json`;
3. confirm `result.json` reports `status: completed`; and
4. retain the complete `prepare-001` directory and its internal structure.

Submit the fixed grid only after the prepared PROJECT path is complete. Each
task copies the whole preparation directory to its own SCRATCH directory before
Python starts. `%4` is a concurrency cap, not a scientific parameter; lower it
if scheduler availability or project policy requires it:

```bash
export S7_PREPARED="$PERUN_PROJECT/perun-results/swiglu-7/prepare-001/result.json"

sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --array=0-11%4 \
  workflows/jobs/perun/swiglu_7.sbatch train "$S7_PREPARED"
```

Array mapping is deterministic:

| Tasks | Strategy | Targets in task order |
| --- | --- | --- |
| 0-3 | S7-0 | 0.2, 0.3, 0.4, 0.5 |
| 4-7 | S7-1 | 0.2, 0.3, 0.4, 0.5 |
| 8-11 | S7-2 | 0.2, 0.3, 0.4, 0.5 |

Each task writes live checkpoints to its own SCRATCH output and stages them to
`$PERUN_PROJECT/perun-results/swiglu-7/<strategy>-target-<target>-run-001`.
Checkpoint-heavy writes therefore do not target persistent PROJECT storage
during training.

## Failure and continuation

On a Python exception, `TERM`, or `INT`, the launcher copies the latest
`result.json`, `run.json`, and verified checkpoint to PROJECT before exiting.
Resume preparation with:

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  workflows/jobs/perun/swiglu_7.sbatch prepare --resume
```

Preparation reuses completed durable stages, although work inside the active
local-fitting stage can be repeated. Resume one failed training task by its
original array index, for example task 6 (S7-1 at target 0.4):

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --array=6 \
  workflows/jobs/perun/swiglu_7.sbatch train --resume "$S7_PREPARED"
```

Training restores model, optimizer, RNG, and progress from the latest verified
25-million-token checkpoint, so at most the work since that checkpoint is
repeated. Never run two jobs for the same identity simultaneously.

If stage-out is interrupted or the node is lost, the launcher leaves a
`*.perun-lock` under the SwiGLU-7 PROJECT result root and, when available, the
marker-protected job directory in SCRATCH. Do not delete either blindly. Verify
the owner file, manually complete the SCRATCH-to-PROJECT copy, compare
`result.json`, and only then remove that exact scratch directory and lock. A
missing scratch directory after a hard node loss means recovery is limited to
the last already verified PROJECT copy.

## Planning cost

These are capacity-planning estimates, not measured H200 results. The closest
local evidence is the completed 128-token SwiGLU-6 replacement-only recovery:
one billion tokens took 104,611 seconds (29.1 hours) at 20% and 108,010 seconds
(30.0 hours) at 50% on an RTX 4090. SwiGLU-7 uses a faster H200 but adds much
more expensive 8K attention and, for S7-1/S7-2, broader gradient computation.

| Work | Estimated one-H200 allocation time |
| --- | ---: |
| Shared preparation | 4-10 GPU-hours |
| One S7-0 trajectory, including final evaluation | 8-18 GPU-hours |
| One S7-1 trajectory, including final evaluation | 12-24 GPU-hours |
| One S7-2 trajectory, including final evaluation | 10-22 GPU-hours |
| Entire preparation plus 12-run grid | **124-266 GPU-hours** |

With four concurrent GPUs, the compute portion is roughly 1.5-3 days after
queueing and stage-out; serial execution is roughly 5-11 days. The tracked
48-hour request gives each task substantial headroom. The requested ceiling
for all 13 jobs is 624 GPU-hours, but only elapsed allocation time counts as
consumption.

The [official NVIDIA H200 specifications](https://www.nvidia.com/en-us/data-center/h200/)
state 141 GB HBM and 4.8 TB/s bandwidth. One full H200 is
therefore the appropriate initial request. Planning bands are approximately
25-50 GB VRAM for S7-0, 35-70 GB for S7-1, and 30-60 GB for S7-2; these include
wide uncertainty for 8K activations and must be replaced by measured peaks.
The 128 GB host-memory request is mainly for FP32 checkpoint assembly and Adam
state. Preparation needs approximately 16-20 GB durable space plus 15-20 GB
temporary fit-state space. The largest training output reserve is expected to
be roughly 45-55 GB before the successful checkpoint is removed.

Record every job, including failures and cancellations, in the
[PERUN experiment log](../../../infrastructure/perun-log.md), then update the
[PERUN project status](../../../infrastructure/perun-status.md) from measured
elapsed time, MaxRSS, peak GPU memory, and output bytes.
