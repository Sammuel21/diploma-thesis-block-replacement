---
title: Quantization-2 Dense INT3 Baseline
status: active
created: 2026-10-20
modified: 2026-10-20
---

# Quantization-2

Quantization-2 is the dense INT3 baseline: the original SmolLM2-1.7B with its
MLP weights in layers 1-22 quantized to 3 bits, with no replacement. It is
the INT3 counterpart of [Quantization-1](quantization-1.md)'s INT4 runs, and
supplies the "quantization alone" INT3 points on the
[Stacking-1](../stacking/stacking-1.md) frontier plots. It is a baseline, not
a stacked model: only one kind of redundancy reduction is applied (decision
log, 2026-10-20).

**Status (2026-10-20):** implemented, not run. **Next step:** commit and
submit the three runs in the
[PERUN runbook](../../../infrastructure/perun/perun-runbook.md#11-quantization-2)
after Stacking-1 has finished, because changing `workflows/runs/` changes the
run fingerprint of every Stacking-1 task. The binding decisions are the
2026-10-20 rows in the [shared decision log](../../../agents/decisions.md).

## Runs

| Run | What happens | Training tokens | Output |
| --- | --- | ---: | --- |
| INT3 PTQ | Round the 66 MLP projections to INT3, then evaluate | 0 | `int3-001` |
| INT3 QAT 100M | Train the 66 projections for 100M tokens with INT3 simulated, convert to real INT3, evaluate | 100M | `qat-int3-100m-001` |
| INT3 QAT 1B | The same for 1B tokens | 1,000M | `qat-int3-1b-001` |

The INT4 counterparts are Quantization-1's `int4`, `qat-int4-100m`, and
`qat-int4-1b` runs. The 1B run gives Stacking-1's 1B stacked models an INT3
competitor with equal training tokens (decision log, 2026-10-20).

## Method

Everything except the bit width is Quantization-1's, enforced by the runner:
Quantization-2's `seed`, `model`, `backend`, `scope`, `data`, `qat`, and
`evaluation` blocks must equal Quantization-1's.

- **Scope:** the gate, up, and down projections of MLP layers 1-22 (66
  projections). Attention, embeddings, norms, and layers 0 and 23 stay BF16,
  and a check stops the run if anything outside the scope changes.
- **INT3 recipe** (TorchAO 0.17.0, `src-torchao-0-17-0`), shared with
  Stacking-1 in `src/mlp_replacement/compression/quantization.py`:
  weight-only, asymmetric, groups of 128 input weights, BF16 scales,
  round-to-nearest (`IntxWeightOnlyConfig(int3, PerGroup(128), ASYMMETRIC)`).
- **INT3 QAT:** Quantization-1's recipe (KL to the dense teacher, AdamW at
  3e-5 with a constant schedule, 8,192-token sequences, a validation row every
  25M tokens). INT3 is simulated with an explicit
  `IntxFakeQuantizeConfig(int3, group 128, asymmetric)`, because TorchAO
  infers fake quantization only for symmetric mapping. Gradients pass through
  rounding by a straight-through estimator (`src-qat-jacob-2018`, Section 3;
  `src-ste-bengio-2013`, Section 4). INT4 QAT uses TorchAO's dedicated INT4
  fake quantizer, so the two QAT points do not share one implementation.
- **Calculated bytes.** TorchAO stores one int8 per INT3 value, about 8.2
  bits per weight. Each INT3 result therefore records two sizes: the measured
  container, and the calculated size at 3.25 bits per weight (3 bits plus a
  16-bit scale and a 16-bit zero point per group, INT4's layout). The
  calculated size is used on the frontier, which is about 51.6% whole-model
  byte saving. This is standard storage arithmetic, not a cited result.
- **Weak PTQ.** Round-to-nearest is the weakest standard INT3 method. GPTQ
  reports that it "collapses completely" at 3 bits on OPT models
  (`src-gptq-2023`, Section 5), so INT3 QAT is the fair INT3 point. GPTQ is
  added only if INT3 PTQ is much worse than INT3 QAT (decision log,
  2026-10-17).

## Data and evaluation

- **Preparation:** Quantization-1's `prepare-001` is reused unchanged: its
  WikiText-2 validation and test tokens and its 24 × 8,192-token KL stream,
  which have the same sha256 as SwiGLU-8's, so the points are comparable with
  Stacking-1 and SwiGLU-8. The runner accepts it because the preparation
  fingerprint covers only blocks that Quantization-2 shares with
  Quantization-1.
- **QAT data:** each QAT run builds its stream from C4 shards 1 onward, as
  Quantization-1 did. The run **stops before training** unless the stream's
  sha256 equals the stream Quantization-1's INT4 QAT trained on at that
  budget: SwiGLU-8's 100M stream (`570409c1…`) and SwiGLU-7's 1B stream
  (`f78bf805…`) (decision log, 2026-10-20).
- **Evaluation:** Quantization-1's protocol: the 8K KL to the dense model,
  full WikiText-2 perplexity at 128, 2,048, and 8,192 tokens, and the five
  zero-shot tasks with the pinned harness.
- **Memory, no speed:** resident memory is measured in a fresh process with
  the protocol Quantization-1 and Stacking-1 use
  (`fresh-process-model-only-no-forward`). Speed is not measured, and INT3
  has no kernel (decision log, 2026-10-20).
- **Comparison report:** Quantization-1's `report` command accepts
  Quantization-2 runs against Quantization-1's dense control `dense-001`, with
  paired bootstrap task differences (10,000 resamples, seed 21). Each row
  records `frontier_basis`, `frontier_weight_bytes`, and
  `whole_model_frontier_byte_savings`: calculated for INT3, measured for
  every other format.

## Checks

Each check stops the run.

| Check | What it compares |
| --- | --- |
| Shared blocks | Quantization-2's configuration equals Quantization-1's in every shared block |
| Preparation | `prepare-001`'s fingerprints, harness sources, and token sha256 values (Quantization-1's existing check) |
| Stream | The QAT stream's sha256 equals the pinned stream for its budget |
| Scope | Exactly the 66 projections are INT3 tensors of the expected type, block size, and dtypes; every other weight is unchanged (Quantization-1's checks) |
| Bundle | The deployed footprint equals the bundle manifest after a fresh reload; tied embeddings are kept |

## Implementation

Implemented on 2026-10-20 on Quantization-1's runner. Quantization-1's
INT8 and INT4 recipes and its INT4 QAT path call the same functions as
before.

| File | Change |
| --- | --- |
| [quantization-2.json](../../../../workflows/configs/model/baseline/quantization-2.json) | Quantization-1's blocks, the INT3 variants, the two stream pins, and no speed workloads |
| [quantization.py](../../../../workflows/runs/model/baseline/quantization.py) | Workflow taken from the configuration; `ptq-int3` and `qat-int3`; QAT for any supported bit width; calculated INT3 bytes; the stream check; resident memory without speed when the configuration has no workloads; the report accepts both workflows |
| [quantization_1.sbatch](../../../../workflows/jobs/perun/quantization_1.sbatch) | The results folder follows the configuration's workflow; `run-array` stays Quantization-1 only |
| [test_quantization2_contracts.py](../../../../tests/test_quantization2_contracts.py) | Shared blocks, variant scope, stream pins, refused commands, and the report's frontier bytes |

**What was tested.** CPU only, on 2026-10-20, in the project's Python: the 7
new contract tests and Quantization-1's 11 tests pass (one TorchAO test
skipped), and the full suite passes (127 tests, 5 skipped). TorchAO is not
installed locally, so the INT3 PTQ and QAT tensor tests (Stacking-1's) were
skipped here; they passed with TorchAO 0.17.0 on 2026-10-17. GPU execution,
the 1B INT3 QAT speed, and full evaluation are untested and are part of the
intended runs.

**Commands** (local form; PERUN commands are in the runbook):

```bash
bash workflows/jobs/local/quantization_1.sh run --config workflows/configs/model/baseline/quantization-2.json \
  --prepared data/results/workflows/model/quantization-1/prepare-001/result.json \
  --variant qat-int3 --qat-tokens 100000000 \
  --output-dir data/results/workflows/model/quantization-2/qat-int3-100m-001
```

## Pre-run resource plan

Recorded 2026-10-20 under the
[PERUN resource contract](../../../infrastructure/perun/perun-resources.md).
Estimates, not measurements.

| Run | Expected | Time limit | Ceiling GPU-h | Basis | Confidence |
| --- | --- | --- | ---: | --- | --- |
| INT3 PTQ | 0.3-0.5 h | 2 h | 2 | Quantization-1 evaluations 0.27-0.29 h; INT3 dequantization adds an estimated 5-10 min | Medium |
| INT3 QAT 100M | 1.6-3.5 h | 6 h | 6 | Quantization-1 INT4 QAT 100M 1.63 h in total; INT3 fake-quantization overhead unknown | Low |
| INT3 QAT 1B | 16-32 h | 36 h | 36 | Quantization-1 INT4 QAT 1B 15.83 h in total; the same limit as that run | Low |
| **Total** | **about 18-36 GPU-h** | | **44** | | |

- **Per run:** one H200, 8 CPUs, 128 GB RAM, `gpu_short`; expected CPU use
  144-288 core-hours, ceiling 352.
- **Memory:** peak evaluation VRAM 14-16 GiB (Quantization-1 measured
  13.9-15.4). QAT VRAM was not recorded by Quantization-1.
- **Storage:** PROJECT about 7 GB (INT3 bundles at INT8 container size,
  about 2.3 GB each, plus JSON). SCRATCH for the 1B run: the 4 GB stream and
  up to two checkpoint generations of about 13 GB each.
- **Timeout:** a run that hits its limit resumes from its last 25M-token
  checkpoint, but only from unchanged code, configuration, and environment.

## Limits

- One run per variant.
- INT3 PTQ uses round-to-nearest, known to be weak at 3 bits.
- INT3 bytes are calculated, not measured; the measured INT3 storage is an
  8-bit container.
- INT3 and INT4 QAT use different TorchAO fake quantizers. A CPU probe on
  2026-10-17 showed that converting INT3 QAT weights moves about 0.5-0.7% of
  them by one step relative to their fake-quantized values
  ([Stacking-1](../stacking/stacking-1.md#implementation)).
- No speed is measured.

## Sources

Entries are registered in [the source registry](../../../../llm-wiki/raw/sources.yml);
the locators are those checked for Stacking-1 on 2026-10-17.

- `src-torchao-0-17-0`: INT3 recipe, `IntxUnpackedToInt8Tensor`, and the
  explicit fake quantizer.
- `src-qat-jacob-2018`: Section 3, simulated quantization during training.
- `src-ste-bengio-2013`: Section 4, straight-through estimator.
- `src-gptq-2023`: Section 5, round-to-nearest at 3 bits. Not implemented.
