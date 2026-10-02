---
metadata_version: 1
title: Quantization-1 MLP Compression Baseline
type: experiment-workflow
category: experiments/model/baseline
status: active
created: 2026-10-02
modified: 2026-10-02
authorship:
  created_by: llm
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---

# Quantization-1

Quantization-1 asks how architectural MLP replacement compares with ordinary
weight quantization at similar storage budgets. It is a separate baseline
family. Existing SwiGLU configurations, runners, and results are preserved.

**Implementation status:** implemented; scientific execution and GPU backend
validation are pending. No numerical quality, memory, or speed benefit is
claimed from code inspection.

## What is compared

The pinned model is `HuggingFaceTB/SmolLM2-1.7B`, revision
`effd688a12921b4cc83e3312b6feb579f70f9c71`, with the same tokenizer revision.
Only the gate/up/down weights of MLP layers 1–22 are quantized: 66 projections.
Layers 0 and 23, attention, normalization, embeddings, and the tied output
head remain BF16. Activations and generation KV caches remain BF16.

| Variant | Quantization | Training | Nominal structural comparison |
| --- | --- | --- | --- |
| `dense-bf16` | None | None | Dense control |
| `ptq-int8` | Weight-only, symmetric per output channel | None | 50% eligible MLP removal at BF16 |
| `ptq-int4` | Weight-only, asymmetric groups of 128 input weights | None | 75% eligible MLP removal; 70% and 80% bracket it |
| `qat-int4` | Same deployed INT4 recipe, with fake-quantization-aware recovery | Explicit token budget | Recovered INT4 point |

**Standard storage arithmetic:** replacing a two-byte BF16 weight by an
eight-bit or four-bit value gives nominal eligible-weight savings of 50% or
75%, respectively. This bookkeeping is not a literature-derived compression
result and does not require a scientific citation. Scales, zero points,
padding, and container overhead change actual byte savings. Unquantized model
parts also reduce whole-model savings. Logical parameter count is unchanged
by quantization; do not label either variant as model parameter sparsity.

One INT4 model is evaluated, rather than inventing separate INT4 models for
70% and 80%. Comparisons report actual bytes, quality, speed, and training
cost. PTQ versus recovered architectural models is a useful deployment
comparison, but is not an equal-training-budget experiment.

## Backend and environment

The project decision is to use a fixed library recipe, with no calibration,
group-size search, custom quantizer, activation quantization, compilation, or
serving-engine tuning. Activation calibration and PTQ training counts are zero.

- TorchAO 0.17.0: `Int8WeightOnlyConfig(version=2, granularity=PerRow())` and
  `Int4WeightOnlyConfig(version=2, group_size=128, int4_packing_format=PLAIN)`.
  Both disable automatic Inductor configuration changes. See
  `src-torchao-0-17-0`, `torchao/quantization/quant_api.py`, the corresponding
  config classes and transform functions.
- INT4 PLAIN uses MSLK's BF16/INT4 rowwise kernel. Pin MSLK 1.1.0+cu128 with
  PyTorch 2.11.0+cu128, CUDA 12.8, and Python 3.12 on Linux. See
  `src-mslk-1-1-0`, release compatibility table; and
  `src-torchao-compatibility`, release 0.17.0 row.
- The dedicated environment keeps Transformers 5.14.1, Datasets 5.0.0,
  Triton 3.6.0, and LM Evaluation Harness 0.4.13, matching the observed
  SwiGLU-7 production software stack. The run records installed versions.

Install in a separate environment, not over the original SwiGLU environment:

```bash
python3.12 -m venv "$QUANTIZATION_ENV"
source "$QUANTIZATION_ENV/bin/activate"
python -m pip install -r workflows/environments/quantization-1-requirements.txt
python -m pip check
export MLP_REPLACEMENT_PYTHON="$(command -v python)"
```

Set `QUANTIZATION_ENV` to a persistent environment directory first. Source
inspection and version compatibility do not establish that kernels execute
correctly on a particular GPU; the intended full scientific run must verify
that. Unsupported kernels or incomplete conversion raise errors.

## Commands and artifacts

Configuration: [quantization-1.json](../../../../workflows/configs/model/baseline/quantization-1.json).
Runner: [quantization.py](../../../../workflows/runs/model/baseline/quantization.py).
The local launcher resolves paths from the checkout regardless of the caller's
working directory. Each example below owns a different output directory.

```bash
bash workflows/jobs/local/quantization_1.sh prepare \
  --output-dir data/results/workflows/model/quantization-1/prepare-001

bash workflows/jobs/local/quantization_1.sh run --variant dense-bf16 \
  --prepared data/results/workflows/model/quantization-1/prepare-001/result.json \
  --output-dir data/results/workflows/model/quantization-1/dense-001

bash workflows/jobs/local/quantization_1.sh run --variant ptq-int8 \
  --prepared data/results/workflows/model/quantization-1/prepare-001/result.json \
  --output-dir data/results/workflows/model/quantization-1/int8-001

bash workflows/jobs/local/quantization_1.sh run --variant ptq-int4 \
  --prepared data/results/workflows/model/quantization-1/prepare-001/result.json \
  --output-dir data/results/workflows/model/quantization-1/int4-001
```

`run` includes conversion/export, reload, footprint measurement, all quality
tasks, and fixed generation workloads. `prepare` freezes validation/test token
streams, a disjoint KL stream, and installed harness source hashes. It does
not generate the expensive recovery stream until QAT is selected.

Outputs contain atomic `result.json`, crash-aware `run.json`, the effective
configuration, raw benchmark samples, generation timings, and a final `model/`
bundle. The identity fingerprints scientific settings, input protocol, and
source code. Resume requires the original output and `--resume`; changed
scientific settings, source, prepared tokens, or checkpoint data are rejected.
The local launcher removes only the disposable work directory it creates;
caller-supplied `--work-dir` data remains caller-managed.

Quantized bundles use packed TorchAO tensor subclasses in `model.pt`, plus
configuration, tokenizer, recipe/backend metadata, and checksums. They have a
separate format from the historical BF16 safetensors bundles. Reload verifies
packed leaves, logical shapes, parameter scope, aliases, and physical storage.
Nonpersistent rotary buffers are regenerated from the saved configuration.
See `src-torchao-serialization-0-17`, serialization/deserialization flow.

### Supplied structural comparisons

The integration does not create new structural compression experiments. To
add an available 50%, 70%, or 80% model, supply its complete BF16 bundle and
the original result for revision/scope/budget provenance:

```bash
bash workflows/jobs/local/quantization_1.sh evaluate \
  --bundle "$STRUCTURAL_BUNDLE" --reference-result "$STRUCTURAL_RESULT" \
  --label 'SwiGLU 50%' \
  --prepared data/results/workflows/model/quantization-1/prepare-001/result.json \
  --output-dir data/results/workflows/model/quantization-1/structural-50-001
```

The new evaluation does not modify the source bundle or historical result.
The archived local SwiGLU-7 manifests are insufficient where weight files
were omitted: obtain the complete bundle from its retained production output.
Missing reference points remain explicitly pending.

```bash
bash workflows/jobs/local/quantization_1.sh report \
  --results data/results/workflows/model/quantization-1/dense-001/result.json \
            data/results/workflows/model/quantization-1/int8-001/result.json \
            data/results/workflows/model/quantization-1/int4-001/result.json \
  --output-dir data/results/workflows/model/quantization-1/report-001
```

Append completed structural/QAT evaluation results to `--results` as available.
The report emits JSON, CSV, and Markdown. Its JSON retains every likelihood
measurement, per-task paired confidence intervals, storage fields, and raw
runtime records. There is no combined quality/size/speed score.

## Evaluation protocol

**Project decision:** use the frozen SwiGLU-7 final evaluation subtree verbatim,
including dataset revisions, tokenizer behavior, task metrics, and seed 21.

- Full WikiText-2 validation and test PPL: contexts 128/2048/8192, strides
  64/1024/4096; every target token is scored once, including the final window.
- Full zero-shot PIQA, ARC Easy, ARC Challenge, WinoGrande, and HellaSwag:
  context limit 2048, batch size 1, no chat template, pinned datasets and
  harness. Keep raw document/prompt/target hashes. Reporting uses 10,000
  paired bootstrap resamples; these are example-level uncertainty, not
  multiple training seeds.
- Teacher KL: 24 consecutive EOS-packed C4 sequences of length 8192 from
  train shard 0; temperature 1, averaged over all valid positions. Teacher
  logits are computed online. This is a newly defined shared diagnostic;
  historical 128-token KL is not interchangeable with it.

Generation is a supporting descriptive measurement. It uses SDPA, eager
execution, fixed validation-derived prompts, BF16 KV caches, fresh caches per
request, and greedy output forced to 256 tokens. Workloads are batch/prompt
1/2048, 4/2048, and 1/7936; the last stays within the model's 8192 context.
Two warm-ups precede ten measured requests. CUDA is synchronized at first-token
and request boundaries. Tokenization, model load, queueing, and network time
are excluded. Report median/p95 and preserve every request's timing. See
`src-pytorch-benchmarking-guide`, warm-up and accelerator synchronization;
the exact workloads and aggregation are project decisions.

The fresh-process runtime worker has no teacher model resident. It measures
resident allocated/reserved VRAM and host RSS before inference, then peak VRAM,
TTFT, average subsequent-token latency, total latency, and tokens/second.
Execution fingerprints include software, attention backend, workloads, prompt
hash, measurement source hashes, and physical GPU UUID. A mismatch marks speed comparisons as incompatible;
quality measurements remain separately reported.

Physical storage counts deduplicate shared allocations and include scale/zero
metadata and padding. Report eligible/whole-model weight bytes, serialized
weight-file bytes, and complete bundle bytes separately. Container sizes can
differ between safetensors and TorchAO serialization. Optimizer checkpoints,
logs, and evaluation files do not contribute to model-bundle size.

## Optional QAT

`qat-int4` is absent from the default variant list and its shipped token budget
is null. Select it explicitly and supply the intended budget:

```bash
bash workflows/jobs/local/quantization_1.sh run --variant qat-int4 \
  --qat-tokens "$QAT_TOKEN_BUDGET" \
  --prepared data/results/workflows/model/quantization-1/prepare-001/result.json \
  --output-dir data/results/workflows/model/quantization-1/qat-int4-001
```

The alternative is to set `qat.target_tokens` in a separately retained config.
QAT uses native `QATConfig` prepare/convert with the same INT4 base recipe.
Only eligible MLP weights train; fake quantization is enabled from the start,
with FP32 master parameters/Adam state and BF16 forward autocast. Before final
conversion, master weights are cast to BF16 so the deployed metadata/kernel
format matches PTQ. See `src-torchao-0-17-0`, QAT prepare/convert API and
`qat/fake_quantize_config.py`, INT4 configuration inference.

Recovery uses the existing exact-token loop: teacher KL at temperature 1,
no CE, constant fused AdamW at 3e-5, no weight decay/warm-up, sequence length
8192, microbatch 1, accumulation 1. The finite ordered C4 stream begins at
shard 1, includes document EOS boundaries, and is never repeated.

Save one verified checkpoint at validation boundaries/milestones and the exact
endpoint. Requested milestones round up to the next optimizer boundary; the
endpoint may use a shorter final sequence. Validate every 25M tokens plus
SwiGLU monitoring milestones within budget. Only validation PPL/KL is used for
monitoring. Final test/task evaluation occurs after the declared final endpoint.

Resume restores eligible prepared state, optimizer, RNG, cursor, history, and
stream/configuration/source fingerprints. Work data is disposable; a Perun
resume regenerates and hash-verifies the same recovery stream if needed. The
checkpoint survives failure and is removed only after successful complete
evaluation. No best-test-checkpoint selection or extra LoRA scope is included.

## Perun resource and deployment contract

Use [quantization_1.sbatch](../../../../workflows/jobs/perun/quantization_1.sbatch),
not the historical generic launcher. It stages only code/configs and explicitly
supplied inputs; it does not depend on `.rsyncignore`. It locks each PROJECT
output, uses unique manual job scratch, copies durable results on success,
failure, TERM, or INT, and checksum-verifies the copy before scratch removal.

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  --output="$PERUN_LOG_DIR/%x_%j.out" --error="$PERUN_LOG_DIR/%x_%j.err" \
  --time="$QUANTIZATION_WALLTIME" \
  workflows/jobs/perun/quantization_1.sbatch run int8-001 \
  --variant ptq-int8 --prepared "$QUANTIZATION_PREPARED_RESULT"
```

The same launcher accepts `prepare OUTPUT_NAME` and `evaluate OUTPUT_NAME`.
Reporting can run locally from retained results. Persistent environment/cache
paths are supplied through `MLP_REPLACEMENT_PYTHON` and `HF_HOME`.

**Pre-run resource plan:** one independent process/GPU per variant, serial
stages, no distributed training. Tracked ceilings are one GPU, eight CPUs,
128 GiB host RAM, and 48 hours on gpu_short; they are conservative inherited
ceilings, not measured consumption. Set a justified wall time before submission.
Expected elapsed/GPU-hours/CPU-hours, peak RAM/VRAM, and scratch demand are
**Unknown**, pending the full intended run. Confidence is low until quantized
kernel and full-suite timing evidence exists. Retain packed model bundles, raw
task/generation records, and any incomplete QAT checkpoint on PROJECT; recovery
token scratch requires four bytes per requested token, excluding other data.

Follow [Perun resource accounting](../../../infrastructure/perun/perun-resources.md)
before submission and after every attempt. The runner records stage timings,
attempts, training tokens, peak VRAM/RSS, and workflow-time GPU-hour proxies.
Scheduler allocation and utilization remain Unknown without scheduler/telemetry
evidence. No job was submitted as part of this implementation.

The inherited per-job ceiling is 48 GPU-hours and 384 CPU core-hours; these
are standard allocated-resource/time calculations, not consumption estimates.
PTQ records zero training tokens and seconds. QAT records committed recovery
update time, validation/checkpoint time, and elapsed workflow time across
attempts; replayed work can make workflow cost exceed committed-update cost.

## Validation and limitations

Acceptance checks cover exact scope and frozen protected state, unchanged
logical parameter counts, tied embeddings, packed serialization/storage,
quality fingerprints, complete PPL coverage, task pairing, QAT state and budget
restoration, and incompatible-resume rejection. Focused CPU/static contract
tests do not replace the full scientific dense/INT8/INT4 runs or the declared
QAT run when enabled. No smoke or reduced-budget jobs are provided.

Run the focused checks in the pinned environment with
`python -m unittest discover -s tests -p test_quantization_contracts.py -v`.
Tensor/QAT checks are skipped if their dependencies are absent. Packed kernel
execution, serialization round trips, and full-model quality/latency remain
part of the intended scientific protocol.

This is a simple fixed-recipe quantization baseline, not a claim about the best
possible PTQ method. Quantization does not guarantee inference speedup, and eager
backend results do not establish optimized serving performance. Whole-model
quantization and quantizing architectural replacement models are outside this
experiment's initial scope.

## Sources

Entries below are registered in [the source registry](../../../../llm-wiki/raw/sources.yml).
Registration and selected API checks do not constitute full literature ingestion.

- `src-torchao-0-17-0`: [pinned source](https://github.com/pytorch/ao/tree/v0.17.0),
  `quant_api.py`, `qat/api.py`, and `qat/fake_quantize_config.py`.
- `src-torchao-compatibility`: [release table](https://github.com/pytorch/ao/issues/2919), 0.17.0 row.
- `src-mslk-1-1-0`: [compatibility table](https://github.com/meta-pytorch/MSLK#release-compatibility-table)
  and [official CUDA 12.8 wheels](https://download.pytorch.org/whl/cu128/mslk/).
- `src-torchao-serialization-0-17`: [serialization guide](https://docs.pytorch.org/ao/stable/eager_tutorials/serialization.html),
  serialization/deserialization flow; accessed for documentation version 0.17.
- `src-pytorch-benchmarking-guide`: [benchmarking guidance](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html),
  warm-up and accelerator synchronization.
