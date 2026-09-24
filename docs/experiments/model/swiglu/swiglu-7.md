---
metadata_version: 1
title: SwiGLU-7 — Retraining-Scope Analysis
type: experiment
category: experiments/model/swiglu
status: active
created: 2026-09-24
modified: 2026-09-24
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---

# SwiGLU-7 — Retraining-Scope Analysis

Implementation is present; no SwiGLU-7 scientific results have been produced.
This experiment asks whether allowing the unchanged transformer body to adapt
improves the final quality of compressed SwiGLU models. It is a production
comparison across four compression targets and three retraining scopes, rather
than another architecture search.

## Fixed comparison

| ID | Retraining scope | Frozen parameters |
| --- | --- | --- |
| S7-0 | Reduced-width replacement SwiGLUs only | Retained MLPs, attention, RMSNorms, tied embeddings/head |
| S7-1 | Every transformer-body parameter: all MLPs, attention projections, and all RMSNorms including the final norm | Tied embeddings/head |
| S7-2 | All MLP and RMSNorm parameters directly, plus rank-16 LoRA on every attention q/k/v/o projection | Tied embeddings/head and attention base weights |

S7-0 is the control. S7-1 measures the largest credible adaptation scope without
training the vocabulary table. S7-2 tests cheaper attention adaptation while
retaining full MLP and normalization updates. Residual connections have no
parameters of their own. S7-2 LoRA uses alpha 32 and dropout 0 and is merged
into the attention weights before the final inference bundle is exported.

Each strategy is run independently at 20%, 30%, 40%, and 50% eligible-MLP
parameter removal. The grid therefore contains 12 fresh trajectories. Every
trajectory ends at exactly one billion cumulative recovery tokens. No result
from one strategy or target selects or terminates another run.

The recovery recipe is fixed across the grid: online dense-teacher KL at
temperature 1, no cross-entropy term, fused AdamW, constant learning rate
3e-5, weight decay 0, sequence length 8,192, one sequence and 8,192 effective
tokens per optimizer update, seed 21, BF16 forward operations, and FP32
trainable parameters and optimizer state. There is no warmup or learning-rate
decay. This isolates trainable scope as the planned treatment.

### Native-context recovery decision

SwiGLU-3 through SwiGLU-6 used 128-token recovery sequences and 2,048 effective
tokens per update. That geometry was appropriate for the earlier
replacement-only studies: it kept recovery inexpensive while the pretrained
attention stack remained frozen. SwiGLU-7 directly updates attention in S7-1
and adapts it through LoRA in S7-2. Training those branches only on 128-token
windows would limit their recovery evidence to short dependencies.

SwiGLU-7 therefore uses SmolLM2-1.7B's native 8,192-token context, recorded as
`max_position_embeddings` in the [official model
configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/blob/main/config.json),
for all three strategies. S7-0 uses the same geometry so it remains the internal
control for retraining scope. A single sequence is one optimizer update,
increasing the effective batch from 2,048 to 8,192 tokens and reducing the new
post-branch updates to 121,461. This intentionally starts a new recovery regime;
S7-0 is not an exact continuation control for the 128-token SwiGLU-6 trajectory.

The retained branch point is 5,001,216 cumulative tokens. Ordinary production
updates consume one complete 8,192-token sequence. To preserve the exact 100M
and 1B cumulative endpoints without repeating or inventing stream tokens, the
last update of those two recovery segments uses the remaining 4,352-token and
2,304-token sequence respectively. These two boundary updates are the only
shorter production sequences.

## Starting models

All branches begin at 5,001,216 cumulative recovery tokens.

- The 20% and 50% starts are exact retained S5-C2 endpoints, including their
  replacement optimizer moments and RNG state.
- The 30% and 40% allocations apply the recorded S5-C2 legacy-subset width
  curves at the intermediate budgets. Preparation refits only the selected
  operator widths and then performs online dense-teacher, replacement-only
  recovery to 5M tokens using the historical 128-token, 2,048-effective-token
  S5 recipe. This matches the retained 20% and 50% branch construction before
  every production trajectory switches to 8K.
- Expanded-scope runs preserve the replacement optimizer moments. Newly
  trainable parameters start with empty Adam state. Each result records this
  behavior and its exact trainable-parameter groups.

The 20% and 50% starts are exact historical branch states, but the subsequent
S7-0 trajectory is descriptive rather than an exact continuation comparison
with SwiGLU-6 because S7 uses 8K sequences and a larger effective batch. The new
30% and 40% starts have no historical endpoint to replay. Comparisons among
S7-0, S7-1, and S7-2 remain controlled because all three scopes at a target
branch from the same prepared checkpoint and use the same 8K recovery recipe.

Preparation is shared by the grid. It also copies the verified one-billion-token
stream, historical validation batches, frozen WikiText corpora, benchmark
protocol, dense benchmark records, and the two relevant SwiGLU-6 controls into
one relocatable prepared directory. Temporary fitted operators remain in
`--work-dir` and disappear with that directory after preparation.

## Storage and recovery

SwiGLU-7 is the first model workflow using the directory storage contract:

```text
--work-dir PATH
--output-dir PATH
--resume
```

`--work-dir` must be fresh, separate from `--output-dir`, and may be discarded.
The durable directory contains `result.json`, `run.json`, one final `model/`
bundle, detailed benchmark records, and a `checkpoint/` directory only while a
run is incomplete or failed. Paths owned by an output are stored relative to
that output.

Checkpoint replacement is atomic. The previous verified generation remains
until the new tensor payload and descriptor have both been committed and
verified, after which it is removed. Successful finalization validates the
completed JSON and final bundle before deleting optimizer state. A failed job
retains the latest verified checkpoint for `--resume`.

SwiGLU-1 through SwiGLU-6 keep their original CLI, checkpoint, and artifact
contracts.

## Final evaluation

Every one of the 12 final models receives the frozen SwiGLU-6 evaluation plus
native-context likelihood measurement:

- full WikiText-2 validation and test likelihood at contexts 128, 2,048, and
  8,192, with respective strides 64, 1,024, and 4,096, including the dense
  reference at every context;
- zero-shot PIQA, ARC-Easy, ARC-Challenge, WinoGrande, and HellaSwag;
- paired task differences against the same dense-model examples;
- BF16 bundle bytes, parameter count, native buffer bytes, and fresh-process
  resident CPU/GPU memory; and
- the fixed historical validation-prefix measurements along recovery.

The shared preparation evaluates the dense model at 8K once and carries forward
the frozen 128- and 2K-context dense records. The zero-shot task harness retains
its frozen 2,048-token limit so its paired dense records remain directly
comparable; these tasks do not supply the 8K claim. The WikiText likelihood
evaluation supplies that native-context measurement. The historical
recovery-validation cache remains 128 tokens for trajectory monitoring and
continuity, while final model selection is not performed from it.

The final test split is report-only. Architecture, targets, optimizer, learning
rate, token budget, and evaluation cohort are fixed before it is read. The
analysis notebook is load-only and does not load a model or dataset.

## Files

- Configuration: `workflows/configs/model/swiglu/swiglu-7.json`
- Runner: `workflows/runs/model/swiglu/swiglu_7.py`
- Local launcher: `workflows/jobs/local/run_model.sh`
- Perun launcher: `workflows/jobs/perun/run_model.sbatch`
- Report: `notebooks/model/swiglu/swiglu-7.ipynb`

The training environment must also contain `lm_eval==0.4.13`, `safetensors`,
and `psutil`, because each production job evaluates and exports its own final
model. The runner verifies these dependencies, the harness source hashes, the
frozen task definitions, prepared corpora, dense benchmark records, and a
conservative output-disk reserve before recovery begins.

## Prepare once

Preparation requires these completed source records and only their referenced
assets:

- the SwiGLU-5 search JSON and retained S5-C2 20%/50% endpoints;
- the SwiGLU-6 preparation JSON, 1B token stream, and legacy evaluation batches;
- the SwiGLU-6 frozen protocol JSON and both WikiText token files; and
- the SwiGLU-6 evaluation JSON and dense raw benchmark records.

The SwiGLU-6 model bundles and non-dense benchmark records are not inputs. They
do not need to be copied into a clean Perun submission checkout.

Local foreground execution is one command; it can be wrapped in one `nohup`:

```bash
bash workflows/jobs/local/run_model.sh swiglu-7 prepare \
  --output-dir data/results/workflows/model/swiglu-7/prepare-001
```

On Perun, submit from the clean repository root after exporting the project
account, QoS, interpreter, and external cache locations:

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  workflows/jobs/perun/run_model.sbatch swiglu-7 prepare
```

The launcher supplies a job-local work directory below `$TMPDIR` and a durable
output below `$RESULTS_DIR`. After stage-out, verify `result.json`, the recorded
hashes, and the actual `results_job_<job-id>` location. Move the complete
prepared directory to stable project storage accessible to later compute jobs.

## Run the grid

One job owns one strategy/target pair. The following is the 20% S7-0 integration
calibration and first intended scientific run:

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  --time=96:00:00 --mem=128G \
  workflows/jobs/perun/run_model.sbatch swiglu-7 train \
  --prepared /project/path/swiglu-7-prepare/result.json \
  --strategy S7-0 --target 0.2
```

This job establishes measured H200 throughput, peak RAM, peak GPU memory, and
stage-out behavior. It is not a reduced-budget smoke run. Inspect it with:

```bash
squeue -u "$USER"
sacct -j <job-id> --format=JobID,State,Elapsed,MaxRSS,ReqMem
seff <job-id>
```

Use the observed resource envelope before scheduling the remaining 11 jobs.
S7-1 has the largest memory requirement because its body weights, gradients,
Adam states, and 8K activations are all present during training. Confirm the
allocated H200 and host-memory limits with the first intended run before
submitting the remaining grid. The runner is single-process and single-GPU, so
requesting more GPUs does not accelerate one trajectory.

Submit the remaining fixed combinations by changing only `--strategy` and
`--target`. If a job is interrupted, stage its complete output directory back
into the new submission and run the same command with `--resume` and an explicit
`--output-dir`. Do not start two jobs against the same output directory.

## Result handling

`run.json` is the operational status record. `result.json` is the scientific
record and contains the recovery trajectory, trainable-parameter accounting,
final likelihood with same-context dense differences, task summaries,
footprint, environment, source hashes, and code hashes. Raw paired benchmark
samples live beneath the same output and are referenced relatively. `model/`
is the only retained inference-weight copy.

After all 12 outputs have been verified and moved to the canonical local result
paths used by the notebook, update `swiglu-progression.md` and
`swiglu-results.md` from measured evidence. They remain unchanged until then.
The notebook expects `prepare-001/result.json` and one directory named
`<strategy>-target-<target>-run-001/` for each branch beneath
`data/results/workflows/model/swiglu-7/`, for example
`S7-1-target-0.4-run-001/result.json`.
