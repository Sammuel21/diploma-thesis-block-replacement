---
metadata_version: 1
title: SwiGLU-7 - Retraining-Scope Analysis
type: experiment
category: experiments/model/swiglu
status: active
created: 2026-09-24
modified: 2026-09-29
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
sources:
  notebooks:
    - notebooks/model/swiglu/swiglu-7.ipynb
  artifacts:
    - data/results/workflows/model/swiglu-7/prepare-001/result.json
    - data/results/workflows/model/swiglu-7/S7-0-target-0.2-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-0-target-0.3-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-0-target-0.4-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-0-target-0.5-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-1-target-0.2-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-1-target-0.3-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-1-target-0.4-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-1-target-0.5-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-2-target-0.2-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-2-target-0.3-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-2-target-0.4-run-001/result.json
    - data/results/workflows/model/swiglu-7/S7-2-target-0.5-run-001/result.json
---

# SwiGLU-7 - Retraining-Scope Analysis

The shared preparation and all 12 production runs completed successfully on
PERUN. Every run reached one billion recovery tokens, completed the frozen
WikiText and zero-shot evaluation, validated its inference bundle, and removed
its optimizer checkpoint after finalization.

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

## Results

The fixed comparison favors replacement-only recovery. S7-0 has the lowest
final validation KL at three of four targets and the lowest perplexity in 17 of
the 24 split, context, and target comparisons. S7-1 never wins a likelihood
comparison. S7-2 wins seven, mainly by small margins at the heaviest removal.

| Target | Strategy | Trainable fraction | KL at 1B | Test PPL at 8,192 | Five-task macro |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0% | Dense reference | — | 0 by definition | **6.9331** | **67.141%** |
| 20% | S7-0 replacement only | 22.30% | **0.060933** | **7.5553** | 63.953% |
| 20% | S7-1 transformer body | 93.24% | 0.077076 | 7.6578 | 64.169% |
| 20% | S7-2 MLP, RMSNorm, attention LoRA | 66.36% | 0.068545 | 7.6601 | **64.440%** |
| 30% | S7-0 replacement only | 34.30% | **0.085868** | **7.8683** | 62.115% |
| 30% | S7-1 transformer body | 92.70% | 0.100083 | 8.1236 | 62.271% |
| 30% | S7-2 MLP, RMSNorm, attention LoRA | 63.67% | 0.089550 | 7.9139 | **62.394%** |
| 40% | S7-0 replacement only | 36.50% | **0.116199** | **8.3096** | 60.131% |
| 40% | S7-1 transformer body | 92.06% | 0.127408 | 8.3522 | **60.197%** |
| 40% | S7-2 MLP, RMSNorm, attention LoRA | 60.52% | 0.124680 | 8.3351 | 60.132% |
| 50% | S7-0 replacement only | 39.13% | 0.153541 | 8.8551 | 58.489% |
| 50% | S7-1 transformer body | 91.31% | **0.152586** | 8.9214 | **58.679%** |
| 50% | S7-2 MLP, RMSNorm, attention LoRA | 56.76% | 0.154480 | **8.8428** | 58.555% |

The downstream differences between scopes are small and mixed. S7-1 wins the
macro average at 40% and 50%; S7-2 wins at 20% and 30%. The largest gain over
S7-0 is 0.49 percentage points. These are single-seed trajectories, and the
paired bootstrap intervals compare each model with dense rather than comparing
the three S7 scopes with each other. They are therefore secondary evidence and
do not outweigh the consistent likelihood result.

All 12 runs improved between 100M and 1B tokens. For S7-0, final KL fell by
19.9%, 23.7%, 27.3%, and 28.6% across the 20%, 30%, 40%, and 50% targets.
Long recovery remains most valuable when compression damage is larger.

The broader scopes used more compute and memory without establishing a quality
advantage:

| Strategy | Mean trainable fraction | Four-run H200 time | Peak RAM | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| S7-0 replacement only | 33.06% | **42.57 h** | **8.88 GiB** | **33.52 GiB** |
| S7-1 transformer body | 92.33% | 51.56 h | 22.57 GiB | 51.71 GiB |
| S7-2 MLP, RMSNorm, attention LoRA | 61.83% | 52.65 h | 13.82 GiB | 48.06 GiB |

Preparation took 10.21 H200 hours. Total measured allocation for preparation
and the 12 runs was 156.99 H200 hours.

The production decision is to retain replacement-only recovery as the default
for homogeneous SwiGLU compression. The experiment does not prove that broader
adaptation can never help; it shows that these two broader scopes do not justify
their extra cost under the tested one-billion-token protocol.

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

The analysis notebook is load-only. It reads the completed independent
SwiGLU-7 preparation and production artifacts, with SwiGLU-5 and SwiGLU-6 used
only for labeled historical comparisons.

## Files

- Configuration: `workflows/configs/model/swiglu/swiglu-7.json`
- Runner: `workflows/runs/model/swiglu/swiglu_7.py`
- Local launcher: `workflows/jobs/local/run_model.sh`
- Dedicated PERUN job: `workflows/jobs/perun/swiglu_7.sbatch`
- Report notebook: `notebooks/model/swiglu/swiglu-7.ipynb`

## PERUN prerequisites

Use a clean checkout and submit from its repository root. The environment must
provide a CUDA-enabled PyTorch build that supports H200, Transformers,
Datasets, NumPy, `lm_eval==0.4.13` with its Hugging Face backend,
`accelerate>=0.26.0`, `safetensors`, and `psutil`. A clean environment can
install the pinned harness backend with:

```bash
python -m pip install 'lm_eval[hf]==0.4.13'
```

Importing top-level `lm_eval` is not a sufficient check because `accelerate`
is an optional dependency loaded by the `HFLM` evaluation adapter. The
[pinned harness metadata](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/pyproject.toml)
declares `accelerate>=0.26.0` in its `hf` extra. The exact
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
export PERUN_LOG_DIR="$PERUN_PROJECT/perun-job-logs"

scontrol show assoc_mgr users="$USER" accounts="$PERUN_ACCOUNT" flags=assoc
scontrol show assoc_mgr qos="$PERUN_QOS" flags=qos

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlp-replacement
export MLP_REPLACEMENT_PYTHON="$(command -v python)"
export HF_HOME="$PERUN_PROJECT/huggingface-cache"
mkdir -p "$HF_HOME" "$PERUN_LOG_DIR"
```

Confirm the selected interpreter before allocating a long job:

```bash
"$MLP_REPLACEMENT_PYTHON" -c \
  'from importlib.metadata import version; from packaging.version import Version; import torch, transformers, datasets, lm_eval, safetensors, psutil; from lm_eval.models.huggingface import HFLM; accelerate_version = version("accelerate"); assert Version(accelerate_version) >= Version("0.26.0"); print(torch.__version__, torch.version.cuda, "accelerate", accelerate_version)'

"$MLP_REPLACEMENT_PYTHON" -m pip check
```

## Deploy

Submit one preparation job:

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  --output="$PERUN_LOG_DIR/%x_%j.out" \
  --error="$PERUN_LOG_DIR/%x_%j.err" \
  workflows/jobs/perun/swiglu_7.sbatch prepare
```

The job uses one H200, eight CPUs, 128 GB RAM, and the documented 48-hour
[`gpu_short` ceiling](https://wiki.perun.tuke.sk/slurm/partitions/). The limit
applies independently to each array task.
The launcher manually copies the checkout to a unique directory below
`/mnt/scratch/$USER`, runs there, then copies and verifies durable preparation
data at `$PERUN_PROJECT/perun-results/swiglu-7/prepare-001`. PERUN support
confirmed that automatic scratch activation and synchronization are not
currently available. After a successful preparation:

1. inspect the scheduler `.out` and `.err` files in `$PERUN_LOG_DIR`;
2. inspect PROJECT `result.json` and `run.json`;
3. confirm `result.json` reports `status: completed`; and
4. retain the complete `prepare-001` directory and its internal structure.

Submit the fixed grid only after the prepared PROJECT path is complete. Each
task copies the whole preparation directory to its own SCRATCH directory before
Python starts. `%12` permits all 12 independent one-GPU tasks to run
concurrently; scheduler availability can still keep some tasks pending:

```bash
export S7_PREPARED="$PERUN_PROJECT/perun-results/swiglu-7/prepare-001/result.json"

sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --array=0-11%12 \
  --output="$PERUN_LOG_DIR/%x_%A_%a.out" \
  --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
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
  --output="$PERUN_LOG_DIR/%x_%j.out" \
  --error="$PERUN_LOG_DIR/%x_%j.err" \
  workflows/jobs/perun/swiglu_7.sbatch prepare --resume
```

Preparation reuses completed durable stages, although work inside the active
local-fitting stage can be repeated. Resume one failed training task by its
original array index, for example task 6 (S7-1 at target 0.4):

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" --array=6 \
  --output="$PERUN_LOG_DIR/%x_%A_%a.out" \
  --error="$PERUN_LOG_DIR/%x_%A_%a.err" \
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

## Measured execution cost

The original estimate was 124 to 266 H200 hours for preparation and the full
grid. The observed total was 156.99 H200 hours, including 10.21 hours for
preparation. Individual production runs took 10.25 to 13.89 allocated hours,
well within the 48-hour job limit.

The largest measured resource use was 22.57 GiB host RAM and 51.71 GiB HBM.
One H200 was sufficient for every scope. All successful runs removed their
optimizer checkpoints after result and bundle validation, so durable output is
limited to the final inference bundle and structured records.

The repository copy under `data/results/workflows/model/swiglu-7/` is reduced
for analysis. It contains result and run records, scheduler logs, provenance,
and bundle metadata. The complete model weights and raw evaluation records
remain in PERUN PROJECT; their recorded hashes are preserved in the local
metadata.

Record any future reproduction or failure in the
[PERUN experiment log](../../../infrastructure/perun/perun-log.md). The
[PERUN project status](../../../infrastructure/perun/perun-status.md) contains
the broader environment and capacity record.
