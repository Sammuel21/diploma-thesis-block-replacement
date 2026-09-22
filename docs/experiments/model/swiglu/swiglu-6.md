---
metadata_version: 1
title: SwiGLU-6 — Long Recovery and Final Evaluation
type: experiment
category: experiments/model/swiglu
status: active
created: 2026-09-21
modified: 2026-09-22
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---

# SwiGLU-6 — Long Recovery and Final Evaluation

Implementation is present; no SwiGLU-6 scientific results have been produced.
This family completes the homogeneous chapter through longer recovery, final
held-out evaluation, and comparable deployment accounting. Earlier runners,
configurations, notebooks, and results remain unchanged. All commands below run
from the repository root and write to a new `swiglu-6/` directory.

## Fixed scientific contract

| Choice | SwiGLU-6 protocol |
| --- | --- |
| Models | Existing S5-C2 allocations at 20% and 50% eligible-MLP parameter removal |
| Initial state | Retained SwiGLU-5 search checkpoints at 5,001,216 tokens, including optimizer and RNG |
| Endpoints | Exactly 100M and 1B cumulative recovery tokens |
| Recovery | Replacement parameters only; online dense-teacher KL, temperature 1, no CE |
| Optimizer | Fused AdamW, constant LR 3e-5, weight decay 0 |
| Geometry | Sequence length 128; 8 sequences × 2 accumulated microbatches = 2,048 tokens per full update |
| Precision | BF16 forward; FP32 replacement parameters and Adam state |
| Final models | Dense plus both allocations at each of the two endpoints: five models |
| Model choice | Fixed endpoints; no best-checkpoint or final-test selection |

Token budgets include the inherited 5M search recovery. The new training-time
counter excludes that inherited work, which is recorded separately. The 100M
endpoint ends with a partial 256-token optimizer update. Subsequent segments
restart their update grid from that exact endpoint, without padding, replaying,
or skipping tokens. Both SwiGLU-6 segments end at their exact token budgets.

This preserves the historical selected allocations. SwiGLU-5's selection artifact
contains an endpoint-label issue: its displayed selection metric was associated
with an earlier budget, although the retained starting tensors are actual 5M
checkpoints. At 20% removal S5-C2 was a near runner-up to S5-C3 at the actual 5M
evaluation. SwiGLU-6 is confirmation of S5-C2, not a corrected architecture search.

## Files and environments

- Configuration: `workflows/configs/model/swiglu/swiglu-6.json`.
- Entry points: `swiglu_6_prepare`, `swiglu_6_recovery`, `swiglu_6_evaluate`
  under `workflows.runs.model.swiglu`.
- Report: `notebooks/model/swiglu/swiglu-6.ipynb`, which only reads artifacts.
- New reusable operations: `artifacts.py`, `data_streams.py`,
  `compression/continuation.py`, `evaluation/final_quality.py`, and
  `evaluation/bundles.py` under `src/mlp_replacement`.

Use the original recovery environment on the RTX 4090 host. The recorded
confirmations used Python 3.12.13, PyTorch 2.11.0+cu130, Transformers 5.14.1,
Datasets 5.0.0, and CUDA 13.0. The runner requires the three recorded package
versions and records hardware/environment details. Resume additionally checks
the complete recorded environment and maintained Python source hashes.
Do not upgrade that environment to install the evaluation harness.

Use a separate evaluation environment with the project dependencies, compatible
CUDA PyTorch/Transformers, `safetensors`, `psutil`, and **lm_eval==0.4.13**.
Record/freeze this environment before evaluation; resuming requires its recorded
identity. The adapter uses the pinned release's native task definitions and
[YAML loader](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/lm_eval/tasks/_yaml_loader.py).
Install the pinned harness with `python -m pip install 'lm_eval==0.4.13'` in
that separate environment. No environment is provisioned or GPU job launched
by adding these files.

The source JSON files alone are insufficient. Preserve and copy their referenced
`.assets` trees, especially both retained 5M checkpoints and the original 100M
packed token cache. Their hashes are checked before preparation and recovery.
Source paths can be changed in a new configuration before preparation; preserve
the referenced content and use that same configuration for all stages.

## Execution sequence

### Running as background jobs on darthmachinus

The SwiGLU-5 confirmations ran on the shared RTX 4090 Linux host. SwiGLU-6's
historical replay is specified for that host and its recovery environment. This
is a series of separate `nohup` processes, not one unattended chain: preparation,
protocol freeze, both 100M replay gates, both resumptions to 1B, and final
evaluation. Start each next process only after the previous artifact reaches
the expected status. The existing Perun `run_model.sbatch` launcher does not
dispatch SwiGLU-6.

Connect to the remote host, enter the repository root, and sync the checkout
containing the new SwiGLU-6 files. At implementation time those files are local
uncommitted changes, so `git pull` alone will not put them on the remote host;
transfer them or commit/push them through the normal repository workflow first.
The historical `data/results/` inputs are Git-ignored; they must already exist
on this host or be transferred separately, with their referenced `.assets`
directories. In the original recovery Python environment, verify the source
files and start preparation:

```bash
cd /path/to/Diplomova-Praca/development
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export S6_RESULTS=data/results/workflows/model/swiglu-6
mkdir -p "$S6_RESULTS"
if ! ls data/results/workflows/model/swiglu-3/run-001.json \
   data/results/workflows/model/swiglu-3/run-001.assets/recovery-data/tokens.int32 \
   data/results/workflows/model/swiglu-5/search/run-001.json \
   data/results/workflows/model/swiglu-5/search/run-001.assets/recovery/{0.2,0.5}/S5-C2/endpoint-000005001216.pt \
   data/results/workflows/model/swiglu-5/confirmation/run-001-target-{0.2,0.5}.json; then
  echo "Missing SwiGLU-3/5 source artifact or asset" >&2
  exit 1
fi
nohup python -u -m workflows.runs.model.swiglu.swiglu_6_prepare \
  --output "$S6_RESULTS/prepare-001.json" \
  > "$S6_RESULTS/prepare-001.log" 2>&1 < /dev/null &
echo "Preparation PID: $!"
```

The file check stops submission when historical assets have not been
transferred. Watch the job with
`tail -f "$S6_RESULTS/prepare-001.log"`; `ps -p <PID>` shows whether the
process is still running. A finished process is successful only if
`prepare-001.json` has `"status": "completed"`. The sibling `.run.json` gives
the last stage and failure information.
After reconnecting for a later stage, enter the repository root again and
re-export `PYTHONPATH` and `S6_RESULTS`; shell variables from a previous SSH
session do not persist.

Activate the **separate evaluation environment** (with `lm_eval==0.4.13`) and
freeze the benchmark protocol before recovery:

```bash
nohup python -u -m workflows.runs.model.swiglu.swiglu_6_evaluate \
  --freeze-only --prepared "$S6_RESULTS/prepare-001.json" \
  --output "$S6_RESULTS/protocol-001.json" \
  > "$S6_RESULTS/protocol-001.log" 2>&1 < /dev/null &
echo "Protocol PID: $!"
```

Once `protocol-001.json` is completed, return to the **original recovery
environment**. Start the 20% replay phase:

```bash
nohup python -u -m workflows.runs.model.swiglu.swiglu_6_recovery \
  --prepared "$S6_RESULTS/prepare-001.json" --target 0.2 \
  --output "$S6_RESULTS/recovery-001-target-0.2.json" \
  > "$S6_RESULTS/recovery-001-target-0.2-replay.log" 2>&1 < /dev/null &
echo "20% replay PID: $!"
```

Check `recovery-001-target-0.2.json`: its status should be
`paused_after_replay` and `results.replay_check.passed` should be `true`.
Then resume the same output in a new process:

```bash
nohup python -u -m workflows.runs.model.swiglu.swiglu_6_recovery \
  --prepared "$S6_RESULTS/prepare-001.json" --target 0.2 \
  --output "$S6_RESULTS/recovery-001-target-0.2.json" --resume \
  > "$S6_RESULTS/recovery-001-target-0.2-to-1b.log" 2>&1 < /dev/null &
echo "20% continuation PID: $!"
```

Wait for `"status": "completed"`, then repeat these two recovery commands
with target `0.5` and output `recovery-001-target-0.5.json`. Give its replay and
continuation separate log filenames. This keeps the two expensive trajectories
sequential on the single GPU. A later interruption resumes the **same** output
with `--resume` and a new log filename; the runner restores the last verified
checkpoint. Do not launch a second process against an output while the first is
still active.

Finally activate the evaluation environment and run the five-model report:

```bash
nohup python -u -m workflows.runs.model.swiglu.swiglu_6_evaluate \
  --prepared "$S6_RESULTS/prepare-001.json" \
  --protocol "$S6_RESULTS/protocol-001.json" \
  --recovery "$S6_RESULTS/recovery-001-target-0.2.json" \
  --recovery "$S6_RESULTS/recovery-001-target-0.5.json" \
  --output "$S6_RESULTS/evaluation-001.json" \
  > "$S6_RESULTS/evaluation-001.log" 2>&1 < /dev/null &
echo "Evaluation PID: $!"
```

Check the final artifact for `"status": "completed"`. Preserve and download
the entire `swiglu-6/` results directory, including its `.assets` trees;
the JSON alone is insufficient for model reload or benchmark audit. Log files
are operational records and do not replace the structured artifacts.

### Direct foreground commands

Run preparation in the recovery environment:

```bash
python -m workflows.runs.model.swiglu.swiglu_6_prepare --output data/results/workflows/model/swiglu-6/prepare-001.json
```

Preparation resolves immutable C4 and WikiText dataset revisions, materializes
the historical validation batches, and writes a shared 1B-token int32 stream.
It starts at C4 shard 00001 and proceeds in order, excluding calibration shard
00000. Documents receive EOS; the finite stream never wraps. The regenerated
first 100M tokens must be byte-identical to the historical packed cache.
Failure stops preparation. Its partial output is not a valid prepared artifact;
inspect/remove only that new failed artifact or choose a new output name to retry.

The stream alone occupies 4,000,000,000 bytes (about 3.73 GiB), excluding dataset
caches. Recovery checks free space conservatively for optimizer generations,
milestone weights and BF16 exports. Dataset cache growth must be budgeted
separately. Use a local disk with sufficient room and reliable atomic renames.

The completed SwiGLU-5 confirmations measured about 9,574 and 9,291 recovery
tokens/s on darthmachinus. If that rate holds, continuing each retained 5M state
to 1B takes about 29-30 hours, or about 59 hours for both targets sequentially.
Preparation, checkpoint writes and five-model final evaluation add unmeasured
time. This is a planning estimate, not a measured SwiGLU-6 runtime.

Before recovery, freeze the evaluation protocol in the evaluation environment:

```bash
python -m workflows.runs.model.swiglu.swiglu_6_evaluate --freeze-only --prepared data/results/workflows/model/swiglu-6/prepare-001.json --output data/results/workflows/model/swiglu-6/protocol-001.json
```

This records harness Python/YAML hashes, immutable task dataset revisions,
evaluation splits, and full WikiText validation/test token streams. It does not
score test data. Keep architecture choices and the endpoint cohort frozen.

Back in the recovery environment, run 20% first:

```bash
python -m workflows.runs.model.swiglu.swiglu_6_recovery --prepared data/results/workflows/model/swiglu-6/prepare-001.json --target 0.2 --output data/results/workflows/model/swiglu-6/recovery-001-target-0.2.json
```

At exactly 100M, the process compares historical C4 KL and the historical
WikiText validation prefix. It requires absolute differences no larger than
1e-4 KL and 0.01 perplexity, saves the endpoint, and exits with
`paused_after_replay`. Inspect that artifact's `results.replay_check`, then
continue the same output explicitly:

```bash
python -m workflows.runs.model.swiglu.swiglu_6_recovery --prepared data/results/workflows/model/swiglu-6/prepare-001.json --target 0.2 --output data/results/workflows/model/swiglu-6/recovery-001-target-0.2.json --resume
```

After the 20% trajectory completes, repeat these two commands with `--target 0.5`
and `recovery-001-target-0.5.json`. This is sequential execution on one GPU.
A failed replay is a diagnostic stop; the runner refuses continuation. Resolve
its cause before starting a separately identified experiment. Passing the
metric tolerances supports this replay comparison, not a general claim of
bitwise GPU determinism.

Recovery stores an initial checkpoint and then full resumable states every 25M
requested tokens, rounded to the next optimizer boundary within the segment.
It retains the current and previous verified generations. Each full state has
replacement weights, optimizer, RNG, cursor, updates, and committed result
history. Milestone weights at 100M/1B remain separately available; the final
1B optimizer checkpoint also remains. Hash-invalid or incomplete generations
are skipped in favor of the previous verified generation. Foreign fingerprints
are rejected. Resume after an interruption uses the same command and `--resume`.

In the evaluation environment, after both runs complete:

```bash
python -m workflows.runs.model.swiglu.swiglu_6_evaluate --prepared data/results/workflows/model/swiglu-6/prepare-001.json --protocol data/results/workflows/model/swiglu-6/protocol-001.json --recovery data/results/workflows/model/swiglu-6/recovery-001-target-0.2.json --recovery data/results/workflows/model/swiglu-6/recovery-001-target-0.5.json --output data/results/workflows/model/swiglu-6/evaluation-001.json
```

Add `--resume` to this command after interruption. Completed corpus/task units
are reused; saved task files and bundles are hash-checked. A failed export may
leave a `bundles/<model>.building` directory; inspect that new staging directory
before removing it and resuming. Existing completed bundles are never overwritten.
Freeze/code/environment/cohort mismatches reject resume rather than mixing results.

## Final evaluation and interpretation

Every final model is exported and reloaded as a complete BF16 bundle before
scoring. The bundle contains all inference tensors, tokenizer, base configuration,
per-layer replacement widths, aliases for tied embeddings, and file hashes.
Weights use BF16; model-defined buffers retain their native precision (for
example, FP32 rotary frequencies). Nonpersistent buffers are regenerated from
the saved configuration and their byte count is checked on reload.
Use `mlp_replacement.evaluation.bundles.load_bundle(path)` to reload it locally.
An unmodified generic `AutoModel.from_pretrained` is insufficient for the
nonuniform topology. Strict state loading, tensor equality, parameter count and
tied embeddings are checked by the project loader.

Full WikiText-2 validation and test are scored at contexts 128 and 2048, with
strides 64 and 1024 respectively. Each token after the first is a target exactly
once, including the final tail. NLL is pooled by predicted tokens before
computing perplexity. The rolling protocol is separate from the old 24-batch,
6,096-predicted-token prefix metric; historical results are not relabeled as
full-corpus measurements. Context 2048 evaluation does not change the length-128
recovery recipe. BF16 conversion deltas are measured on the old prefix.

The full zero-shot downstream suite uses native harness prompts without a chat
template, batch size 1, and context limit 2048. Primary metrics are `acc_norm`
for PIQA, ARC-Easy, ARC-Challenge and HellaSwag, and `acc` for WinoGrande.
The report gives each task, their unweighted macro mean, and paired differences
against dense. Per-task 95% percentile intervals use 10,000 paired-example
bootstrap resamples with seed 21. These describe evaluation-example uncertainty,
not recovery-seed variability. No separate training-seed replication is claimed.

For size, report unique parameters, BF16 parameter bytes, native buffer bytes, actual tensor
file bytes, and complete bundle bytes. Report eligible-MLP removal separately
from measured whole-model removal. Resident GPU allocated/reserved and host RSS
deltas come from a fresh subprocess loading only one bundle, without a forward
pass, optimizer, teacher, or KV cache. They are resident measurements, not peak
serving memory or evidence of inference speedup. SwiGLU-6 does not add a latency
benchmark or quantization baseline.

The report notebook reads only completed artifacts, produces recovery curves,
full-corpus/context tables, paired task differences, and quality/footprint views.
Open it after `evaluation-001.json` is complete. Preserve the protocol, configs,
source checkout, JSON files, task samples, and asset trees alongside results.

## Verification status

Local checks cover Python/config/notebook parsing, continuation boundary
arithmetic, complete rolling-target coverage, and checkpoint descriptor integrity.
No GPU training, model export/reload, harness inference, or scientific replay has
been executed in the implementation environment. Those integration checks remain
part of the intended run above; no reduced-budget workflow has been added.

Benchmark references: [WikiText](https://arxiv.org/abs/1609.07843),
[PIQA](https://arxiv.org/abs/1911.11641), [ARC](https://arxiv.org/abs/1803.05457),
[WinoGrande](https://arxiv.org/abs/1907.10641),
[HellaSwag](https://arxiv.org/abs/1905.07830), and the
[pinned harness release](https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.13).
