---
title: Quantization-2 Dense INT3 Baseline
status: active
created: 2026-10-20
modified: 2026-10-25
---

# Quantization-2

Quantization-2 is the dense INT3 baseline: the original SmolLM2-1.7B with its
MLP weights in layers 1-22 quantized to 3 bits, with no replacement. It is
the INT3 counterpart of [Quantization-1](quantization-1.md)'s INT4 runs, and
supplies the "quantization alone" INT3 points on the
[Stacking-1](../stacking/stacking-1.md) frontier plots. It is a baseline, not
a stacked model: only one kind of redundancy reduction is applied (decision
log, 2026-10-20).

**Status (2026-10-25):** ran on PERUN on 2026-10-21/22 (jobs
123646-123648, commit `d7d7c1e`); all three runs completed. INT3 PTQ is
valid ([Results](#results-2026-10-25)). **Both INT3 QAT results are
invalid:** the INT3 training-time quantizer does not reproduce the deployed
INT3 format ([INT3 QAT incident](#int3-qat-incident-2026-10-25)). The
fixed quantizer was implemented on 2026-10-25
([INT3 QAT fix](#int3-qat-fix-2026-10-25)). **Next step:** submit
`qat-int3-100m-002` together with Stacking-1's three order-B INT3 reruns
(runbook 10.5 and 11.5, after the researcher's go); `qat-int3-1b-002` is
decided after that. The reruns are documented here when they finish. The
binding decisions are the 2026-10-20 and 2026-10-25 rows in the
[shared decision log](../../../agents/decisions.md).

## Runs

| Run | What happens | Training tokens | Output |
| --- | --- | ---: | --- |
| INT3 PTQ | Round the 66 MLP projections to INT3, then evaluate | 0 | `int3-001` |
| INT3 QAT 100M | Train the 66 projections for 100M tokens with INT3 simulated, convert to real INT3, evaluate | 100M | `qat-int3-100m-001` |
| INT3 QAT 1B | The same for 1B tokens | 1,000M | `qat-int3-1b-001` |
| INT3 QAT 100M, rerun (planned) | As above, with the fixed INT3 quantizer | 100M | `qat-int3-100m-002` |
| INT3 QAT 1B, rerun (decided after the 100M rerun) | As above, with the fixed INT3 quantizer | 1,000M | `qat-int3-1b-002` |

The two `-001` QAT runs are invalid and kept as evidence
([INT3 QAT incident](#int3-qat-incident-2026-10-25)). The INT4 counterparts are Quantization-1's `int4`, `qat-int4-100m`, and
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
  25M tokens). The `-001` runs simulated INT3 with TorchAO's
  `IntxFakeQuantizeConfig(int3, group 128, asymmetric)`, which does not
  reproduce the deployed INT3, so they are invalid
  ([INT3 QAT incident](#int3-qat-incident-2026-10-25)). The `-002` reruns
  use the project's `DeployedInt3FakeQuantizer`: its scale and zero point are
  the deployed recipe's (the deployed equations in [Cause](#cause),
  source-derived from `src-torchao-0-17-0`), computed on the BF16-cast weight,
  so the simulated weights equal the deployed weights bit for bit
  ([INT3 QAT fix](#int3-qat-fix-2026-10-25)). Gradients pass through rounding
  by a straight-through estimator (`src-qat-jacob-2018`, Section 3;
  `src-ste-bengio-2013`, Section 4), implemented by TorchAO's `_Round`. INT4
  QAT uses TorchAO's dedicated INT4 fake quantizer, so the two QAT points do
  not share one implementation.
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
| INT3 start (`-002` on) | After QAT preparation, before the first update: every projection's simulated INT3 weight is bitwise equal to the deployed recipe applied to the same weight (`results.int3_fake_quantization_check`) |
| INT3 conversion (`-002` on) | At conversion: every projection's deployed INT3 weight is bitwise equal to the simulated weight of the trained master just before conversion (`results.conversion.int3_fake_equals_deployed`) |

Recorded, not enforced: the token-0 8K KL of the simulated model (expected
near INT3 PTQ's 0.2226; the `-001` runs started at 6.229) and the first-step
gradient norm (`-001`: 308; INT4 QAT: 0.571). The 24-batch monitoring
perplexity is not a gate: it mostly measures damage at target position 1
(experiment-audit session, 2026-10-25).

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

### INT3 QAT fix (2026-10-25)

Implemented on 2026-10-25 after the
[INT3 QAT incident](#int3-qat-incident-2026-10-25), for Quantization-2 and
Stacking-1 order B together (decision log, 2026-10-25). INT4 QAT and every PTQ
path are unchanged.

| File | Change |
| --- | --- |
| [quantization.py](../../../../src/mlp_replacement/compression/quantization.py) | `DeployedInt3FakeQuantizer` (below); `prepare_qat` wraps each INT3 projection in TorchAO's `FakeQuantizedLinear` with this quantizer instead of `IntxFakeQuantizeConfig`; `int3_fake_quantization_check` compares simulated and deployed weights; `convert_qat` stops if a converted INT3 weight differs from the trained simulated weight |
| [quantization.py](../../../../workflows/runs/model/baseline/quantization.py) (runner) | Runs the check right after QAT preparation, records it, and stops before the first update on any mismatch |
| [test_stacking1_contracts.py](../../../../tests/test_stacking1_contracts.py) | A planted +9.375 outlier; trained masters off the BF16 grid; the old TorchAO quantizer is caught by both checks; all 66 projections of the pinned SmolLM2-1.7B weights (skipped without TorchAO or the cached snapshot) |

**The quantizer.** For each group of 128 weights, the scale and zero point
are chosen exactly as the deployed recipe chooses them
(`IntxUnpackedToInt8Tensor.from_hp`: `choose_qparams_affine`, asymmetric,
range −4…3, int8 zero point, BF16 scale), on the master weight cast to BF16
as `convert_qat` casts it. TorchAO's `_fake_quantize_affine` then quantizes and
dequantizes with the same internal functions as the deployed path, without
the int8 cast, which is exact for integers in −4…3. Its `_Round` passes
gradients straight through. The equations are the deployed ones in
[Cause](#cause). TorchAO's convert step reads the quantizer's `config` only to
pass learned ranges; the project quantizer has none, so conversion applies the
plain deployed recipe. `_fake_quantize_affine` and `_DTYPE_TO_QVALUE_BOUNDS`
are private TorchAO names, acceptable under the 0.17.0 pin.

**Effect on runs.** Every run fingerprint covers all maintained source, so no
earlier Quantization-1 or -2 run can be resumed (all completed). The reruns
write new `-002` outputs; the `-001` outputs stay as evidence. A `-002` run
resumes only from unchanged code, configuration, and environment, so the
PERUN checkout is not updated until the reruns finish.

**What was tested** (2026-10-25, CPU, on the researcher's PC):

- In a scratch environment with TorchAO 0.17.0 and PyTorch 2.11.0+cu128 (the
  pinned versions), the three quantization test files pass (36 tests). These
  include the real-weight test: all 66 projections of the cached
  `effd688a` snapshot are bitwise equal after preparation, with three trained
  masters (layers 1, 7, 22 `down_proj`) moved off the BF16 grid. The
  22 `down_proj` outlier stays above +9, and the conversion check passes.
- In the project's Python (no TorchAO), the full suite passes (130 tests, 8
  skipped).
- Not tested: GPU execution of the new quantizer and the checks. The in-run
  checks on PERUN are the authority there.

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

### Reruns with the fixed quantizer (recorded 2026-10-25)

Estimates, not measurements, based on the measured `-001` runs, which ran the
same stages; the two new weight checks add seconds (estimate). Stage 1 is
submitted together with Stacking-1's three order-B INT3 reruns
([Stacking-1 plan](../stacking/stacking-1.md#reruns-with-the-fixed-quantizer-recorded-2026-10-25)).

| Run | Stage | Expected | Time limit | Ceiling GPU-h | Basis | Confidence |
| --- | --- | --- | --- | ---: | --- | --- |
| INT3 QAT 100M `-002` | 1 | 1.7-2.2 h | 4 h | 4 | `qat-int3-100m-001` 1.78 h of workflow time | Medium |
| INT3 QAT 1B `-002` | 2, decided after stage 1 | 14-17 h | 24 h | 24 | `qat-int3-1b-001` 14.25 h | Medium |

- **Per run:** one H200, 8 CPUs, 128 GB RAM, `gpu_short`, one process.
  Expected CPU use 14-18 core-hours (100M) and 112-136 (1B); ceilings 32 and
  192.
- **Memory:** `-001` measured 49.6 GiB peak allocated VRAM and 14.6 GiB peak
  RSS.
- **Storage:** PROJECT about 2.3 GB per run; the `-001` outputs are kept.
  SCRATCH: the stream (0.4 GB at 100M, 4 GB at 1B) and up to two checkpoint
  generations of about 13 GB.
- **Stage 1 in total** (with Stacking-1): 4 single-GPU jobs, expected
  5.3-7.0 GPU-hours, ceiling 13. Headroom against the 8,424-GPU-hour
  allocation is ample.

## Results (2026-10-25)

The three runs ran on PERUN on 2026-10-21/22 from commit `d7d7c1e`, as jobs
123646 (PTQ), 123647 (QAT 100M), and 123648 (QAT 1B). All three ended
`completed`; the stream, scope, and bundle checks passed. The analysis copy
(JSON, bundle manifests, job logs) is in
`data/results/workflows/model/quantization-2/`. The reference rows are
Quantization-1's dense control and its INT4 evaluations (`*-eval-001`). The
values below were read from the `result.json` and task files; Quantization-1's
`report` command (paired bootstrap intervals) has not been run for
Quantization-2 yet.

| Model | 8K KL to dense | WikiText-2 test PPL, 128 / 2K / 8K | Task macro | Whole-model byte saving | Status |
| --- | ---: | --- | ---: | ---: | --- |
| Dense BF16 | 0 | 11.98 / 7.27 / 6.93 | 0.671 | 0 | Reference |
| INT4 PTQ | 0.090 | 13.20 / 7.79 / 7.42 | 0.650 | 47.5% (measured) | Valid |
| INT4 QAT 100M | 0.036 | 12.60 / 7.58 / 7.21 | 0.658 | 47.5% | Valid |
| INT4 QAT 1B | 0.035 | 12.59 / 7.61 / 7.24 | 0.666 | 47.5% | Valid |
| **INT3 PTQ** (`int3-001`) | **0.223** | **15.61 / 9.02 / 8.56** | **0.640** | **51.6% (calculated)** | **Valid** |
| INT3 QAT 100M (`qat-int3-100m-001`) | 0.269 | 15.96 / 9.36 / 9.49 | 0.620 | 51.6% | **Invalid** |
| INT3 QAT 1B (`qat-int3-1b-001`) | 1.864 | 15.16 / 14.03 / 69.48 | 0.629 | 51.6% | **Invalid** |

Task macro is the mean of `acc_norm` on PIQA, ARC-Easy, ARC-Challenge, and
HellaSwag and `acc` on WinoGrande. INT3 PTQ per task: PIQA 0.758, ARC-Easy
0.700, ARC-Challenge 0.430, WinoGrande 0.641, HellaSwag 0.670 (dense 0.774,
0.737, 0.472, 0.661, 0.713). The INT3 saving uses the calculated 1,657,999,360
weight bytes against the dense 3,422,752,768.

Compared with INT4 PTQ, INT3 PTQ saves 4.1 percentage points more of the
whole model's bytes, at about 2.5 times the KL (0.223 against 0.090) and a
task macro 1.0 point lower (0.640 against 0.650). Its 8K perplexity is 23%
above dense. This is one run per variant, without bootstrap intervals yet.
The GPTQ condition (decision log, 2026-10-17: add GPTQ only if INT3 PTQ is
much worse than INT3 QAT) cannot be evaluated until a valid INT3 QAT result
exists.

## INT3 QAT incident (2026-10-25)

**In short.** The INT3 QAT runs trained against a training-time ("fake")
quantizer that does not reproduce the deployed INT3 format. It nearly agrees
on ordinary weights but distorts the model's few large outlier weights.
Training improved this distorted model, and the final conversion restored the
outliers, so the deployed model was never trained in the form it ships. Both
INT3 QAT results are invalid, and so are Stacking-1's three order-B INT3
runs, which use the same code
([Stacking-1](../stacking/stacking-1.md#int3-order-b-runs-invalid-2026-10-25)).
INT4 QAT is unaffected: it uses TorchAO's matched INT4 pair.

### Evidence

Observed in the result artifacts. KL is measured on the shared 8K stream; the
token-0 row is the fake-quantized model before any update.

| KL to dense | INT4 (Quantization-1) | INT3 QAT 100M | INT3 QAT 1B |
| --- | ---: | ---: | ---: |
| PTQ model (final evaluation) | 0.0896 | 0.2226 | 0.2226 |
| Fake-quantized model, token 0 | 0.0897 | 6.229 | 6.229 |
| Fake-quantized model, last row | 0.0286 (100M), 0.0295 (1B) | 0.0977 | 0.0836 |
| Deployed model after conversion | 0.0360 (100M), 0.0352 (1B) | 0.2686 | 1.8644 |

At token 0 the INT3 fake-quantized model has a validation 8K perplexity of
19,594. By the last row it reaches 8.20 (100M) and 8.02 (1B), better than INT3
PTQ's 8.83, but the converted models lose that gain and more.

A CPU probe on 2026-10-25 ran TorchAO 0.17.0 in a scratch environment with the
project's own `weight_only_recipe(3)` (deployed) and `qat_prepare_config(3)`
(training-time, applied to the FP32 master weight as `prepare_qat` does). It
covered all 66 MLP projections of layers 1-22 of the pinned SmolLM2-1.7B
snapshot (`effd688a`).

- Ordinary groups nearly agree. 0.8% of the values differ, and the relative
  weight error is 0.219 deployed against 0.224 training-time. The
  training-time path keeps a median 86% of each group's largest weight
  (deployed: 106%), and 51% of the groups lose more than 10% of it.
- Outlier groups do not agree. The five largest weights:

| Projection | Weight | Deployed INT3 | Training-time INT3 |
| --- | ---: | ---: | ---: |
| 22 `down_proj` | +9.375 | +9.625 | −2.752 |
| 7 `down_proj` | −7.875 | −7.250 | −1.210 |
| 1 `gate_proj` | −7.719 | −7.312 | −4.379 |
| 1 `down_proj` | +7.594 | +7.562 | 0.000 |
| 6 `down_proj` | −5.625 | −5.312 | −1.772 |

The median of each projection's largest weight magnitude is 3.08; typical
weights are about 0.02. Not computed: the end-to-end model KL with these
weights on CPU. That the distorted outliers cause the token-0 KL of 6.23 is
inferred; the probe is consistent with it but does not prove it.

### Cause

The equations below are source-derived from the TorchAO 0.17.0 code
(`src-torchao-0-17-0`), written in this page's notation. For one group of 128
weights $w$ with minimum $m$ and maximum $M$:

- **Deployed** (`IntxWeightOnlyConfig` → `IntxUnpackedToInt8Tensor` →
  `choose_qparams_affine`, asymmetric; `quant_primitives.py`,
  `_choose_qparams_affine`):
  $s = (\max(M,0) - \min(m,0))/7$,
  $z = \mathrm{clamp}(-4 - \mathrm{round}(\min(m,0)/s), -4, 3)$,
  $q = \mathrm{clamp}(\mathrm{round}(w/s) + z, -4, 3)$, $\hat w = (q - z)s$.
  The grid covers the group's range.
- **Training-time** (`IntxFakeQuantizeConfig(torch.int3, group_size=128,
  is_symmetric=False)`; `qat/fake_quantizer.py`,
  `IntxFakeQuantizer._per_channel_or_group_forward`): the asymmetric branch
  calls `utils.py` `get_groupwise_affine_qparams`. Its defaults are the
  unsigned range 0…7 and a float-domain zero point
  (`_choose_qparams_affine_tinygemm`): $s = (M - m)/7$, $z_f = m + 4s$. The
  caller then casts $z_f$ to an integer by truncation and fake-quantizes with
  the signed range −4…3 in the integer domain:
  $q = \mathrm{clamp}(\mathrm{round}(w/s) + \mathrm{trunc}(z_f), -4, 3)$,
  $\hat w = (q - \mathrm{trunc}(z_f))s$.

Two conventions are mixed. In an ordinary group $\mathrm{trunc}(z_f) = 0$,
so the grid is centred on zero and covers about $[-8a/7, 6a/7]$ for a group
in $[-a, a]$: the top of the range is cut to 6/7 (the 86% above). In a group
with an outlier of several units, $z_f$ exceeds 1, the truncated zero point is
a nonzero integer, the grid shifts, and the outlier is clipped, zeroed, or
sign-flipped. For example, with the +9.375 weight, $s \approx 1.35$ and
$\mathrm{trunc}(z_f) = 5$, so $q = \mathrm{clamp}(7 + 5) = 3$ and
$\hat w = -2 s \approx -2.7$.

The project used this config because TorchAO's matched path refuses
asymmetric INT3 (the assert in `_infer_fake_quantize_configs`; test
`test_asymmetric_int3_cannot_use_the_inferred_qat_path`).

### Why the checks did not catch it

- The INT3 QAT contract test
  (`test_int3_qat_trains_only_the_selected_scope_and_converts`) checks the
  trainable scope, dtypes, and padding, and that conversion runs. It never
  compares training-time values with deployed values.
- The 2026-10-17 CPU probe found that "the library's two rounding paths
  differ", but on random Gaussian weights, which have no outliers, so the
  difference looked like a small one-step effect. It was recorded as a
  limitation instead of being explained.
- No run-time check compared the fake-quantized start with the PTQ model.
  INT4's matched pair never needed one, and Stacking-1's K3 compares the start
  before fake quantization. Stacking-1 recorded the conversion change
  (`conversion_ppl_delta`) without enforcing it, and Quantization-2 had no
  conversion check.
- Stacking-1's recorded order-B INT3 conversion changes (+10.0 and +30.8 PPL)
  were already on PERUN when Quantization-2 was submitted, but they were not
  inspected before submission.

### Consequences

- **Invalid:** `qat-int3-100m-001`, `qat-int3-1b-001`, and Stacking-1's
  `b-fitted-{0.2,0.5,0.8}-100m-int3-001`. Their artifacts stay as evidence;
  they are not results.
- **Cost:** 16.0 workflow GPU-hours here (1.78 + 14.25), plus about
  3.6-4.3 h in Stacking-1.
- **Still valid:** `int3-001`, and every INT8, INT4, and PTQ result.
- **Fix and reruns:** the INT3 training-time quantizer now takes its scale and
  zero point from the deployed path, with a real-weight equality test and
  in-run stops ([INT3 QAT fix](#int3-qat-fix-2026-10-25)). The reruns
  (`-002`) are documented here when they finish.

## Measured resource use (2026-10-25)

| Run | Workflow time | Plan | Peak GPU memory (allocated) | Peak process RSS |
| --- | ---: | --- | ---: | ---: |
| INT3 PTQ | 0.35 h | 0.3-0.5 h | 14.4 GiB | 5.7 GiB |
| INT3 QAT 100M | 1.78 h | 1.6-3.5 h | 49.6 GiB | 14.6 GiB |
| INT3 QAT 1B | 14.25 h | 16-32 h | 49.6 GiB | 14.6 GiB |
| **Total** | **16.4 h** | 18-36 h (ceiling 44) | | |

- Workflow timing is a stand-in. Scheduler allocation is Unknown: the
  `scontrol` records had expired by 2026-10-25, and `sacct` is unavailable.
- All three jobs started at 09:31 UTC on 2026-10-21 and ended at 09:52, 11:18,
  and 23:46 UTC. 1B training took 47,042 s, about 21.3k tokens/s.
- QAT peak VRAM was about 50 GiB; Quantization-1 did not record it.
- Storage: 2.3 GB per run on PROJECT; the 1B checkpoint was not kept.

## Limits

- One run per variant.
- INT3 PTQ uses round-to-nearest, known to be weak at 3 bits.
- INT3 bytes are calculated, not measured; the measured INT3 storage is an
  8-bit container.
- INT3 and INT4 QAT use different TorchAO fake quantizers. The 2026-10-17 CPU
  probe found that converting INT3 QAT weights moves only about 0.5-0.7% of
  them by one step, but this held only on random Gaussian weights. On the real
  weights, the INT3 fake quantizer distorts the outliers, and the INT3 QAT
  results are invalid ([INT3 QAT incident](#int3-qat-incident-2026-10-25)).
  From `-002` on, INT3 uses the project's quantizer, which is bitwise equal to
  the deployed INT3 and checked in every run
  ([INT3 QAT fix](#int3-qat-fix-2026-10-25)). INT4 still uses TorchAO's INT4
  quantizer; its conversion adds KL that is not yet explained
  (0.0286 → 0.0360 at 100M), an open question outside this fix.
- No speed is measured.

## Sources

Entries are registered in [the source registry](../../../../llm-wiki/raw/sources.yml);
the locators are those checked for Stacking-1 on 2026-10-17.

- `src-torchao-0-17-0`: INT3 recipe, `IntxUnpackedToInt8Tensor`, and the
  explicit fake quantizer. The incident analysis (2026-10-25) read the wheel's
  `qat/fake_quantizer.py` (`IntxFakeQuantizer._per_channel_or_group_forward`),
  `utils.py` (`get_groupwise_affine_qparams`), and `quant_primitives.py`
  (`_choose_qparams_affine_tinygemm`, `_choose_qparams_affine`).
- `src-qat-jacob-2018`: Section 3, simulated quantization during training.
- `src-ste-bengio-2013`: Section 4, straight-through estimator.
- `src-gptq-2023`: Section 5, round-to-nearest at 3 bits. Not implemented.
