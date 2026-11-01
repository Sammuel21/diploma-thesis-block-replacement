---
title: Stacking-1 - SwiGLU Replacement Plus Weight Quantization
status: active
created: 2026-10-17
modified: 2026-11-01
---

# Stacking-1 - SwiGLU Replacement Plus Weight Quantization

**Status:** design settled on 2026-10-17 in a grilling session after the
[supervisor meeting of 2026-10-16](../../../meetings/2026-10-16-supervisor-feedback.md).
Implemented on 2026-10-17 without any run (see [Implementation](#implementation)).
Revised on 2026-10-20 in a second grilling session: the dense INT3 runs moved
to the Quantization-2 baseline, leaving 48 stacked runs, and SwiGLU-9's fitted
80% model is pinned, so every run can be submitted at once. The
[pre-run resource plan](#pre-run-resource-plan) is recorded. **Next step:**
the pre-submission checks and the single submission in the
[PERUN runbook](../../../infrastructure/perun/perun-runbook.md#10-stacking-1),
after the researcher's go. The binding decisions are the 2026-10-16,
2026-10-17, and 2026-10-20 rows in the
[shared decision log](../../../agents/decisions.md); if this page and the log
disagree, the log wins until the conflict is raised.

**Update (2026-10-25):** all 48 runs ran on PERUN on 2026-10-20/21 and
completed. The results analysis and the job-log entry are still pending. The
three order-B INT3 runs are invalid
([INT3 order-B runs invalid](#int3-order-b-runs-invalid-2026-10-25)); the
other 45 runs are unaffected. The fixed INT3 quantizer was implemented the
same day ([INT3 QAT fix](#int3-qat-fix-2026-10-25)). **Next step:** rerun the
three order-B INT3 variants as `-002` (runbook 10.5), together with
Quantization-2's INT3 QAT 100M rerun, after the researcher's go.

**Update (2026-11-01):** order B goes to 1B tokens: six runs
`b-fitted-{0.2,0.5,0.8}-1b-int{4,3}` (group `b1b`), implemented and tested
on CPU, not yet run ([Order B at 1B](#order-b-at-1b-plan-2026-11-01)). Its
decision row is proposed, not yet approved. **Next step:** the researcher
approves the row and commits; submission follows runbook 10.7.

## Questions

1. Does SwiGLU replacement followed by INT8, INT4, or INT3 weight quantization
   reach quality-per-byte points that neither method reaches alone?
2. Does the order matter: quantizing after recovery, or recovering while
   quantization is simulated?
3. Does local fitting still matter once the model is quantized?

## Why this experiment

In the [results snapshot](../../../results/2026-10-15-results.md),
quantization beats SwiGLU replacement at every size it reaches (up to 47.5%
whole-model byte saving), and SwiGLU reaches the frontier only beyond that.
The supervisor expected this (INT8 is close to free, INT4 often too) and
suggested stacking the two methods instead of only competing them.

## Win criterion

A model wins if it is on the frontier of a metric's plot: nothing else saves
at least as many bytes with better quality. There is one plot per metric (8K
KL, test perplexity, task macro accuracy), and each plot includes SwiGLU
alone, quantization alone, and the stacked models. Quantization alone,
including INT3, must be on the same plot; otherwise a stacked model could look
like a winner only because its strongest equal-size competitor is missing.
Quantization alone comes from the baselines: INT8 and INT4 from
[Quantization-1](../baseline/quantization-1.md), INT3 PTQ and INT3 QAT at 100M
and 1B from [Quantization-2](../baseline/quantization-2.md) (decision log,
2026-10-20). A stacked model
applies at least two kinds of redundancy reduction (structural, numerical, or
algebraic); quantizing the dense model alone is a baseline, not stacking.

Approximate whole-model byte savings (offline arithmetic: eligible MLPs are
about 64.8% of the BF16 bytes; INT8 costs about 8 bits, INT4 4.25, INT3 3.25
bits per weight):

| Variant | Saving |
| --- | ---: |
| INT4 alone (measured) | 47.5% |
| SwiGLU 50% + INT8 | about 48.6% |
| SwiGLU 20% + INT4 | about 51.0% |
| INT3 alone | about 51.6% |
| SwiGLU 80% alone (measured) | 51.8% |
| SwiGLU 50% + INT4 | about 56% |
| SwiGLU 80% + INT4 | about 61% |

The three variants near 51% are the cleanest equal-size comparison. The INT3
value is reproduced by a contract test from Quantization-1's measured byte
counts.

## Pipeline

Each variant is an ordered stage list set in the configuration. Only stages
that are used are implemented; low-rank factorization is a possible future
stage (structural, numerical, and algebraic redundancy as three stage kinds).

| Variant | Stages |
| --- | --- |
| Order A: quantize after recovery | replace → fit → recover (BF16) → PTQ → evaluate |
| Order B: train while quantized | replace → fit → recover with simulated quantization (QAT) → convert to real integers → evaluate |
| Order C (deferred) | replace → fit → recover (BF16) → QAT → convert → evaluate |

In the configuration, the stages that already ran in SwiGLU-7, -8, or -9
(replace, fit, recover) are recorded as upstream provenance; the runner
executes `load_recovered|load_start → pad → ptq|qat_recover → convert →
export → evaluate`.

**Quantized scope:** every MLP linear layer in layers 1-22, as in
Quantization-1, including layers the allocator kept dense. Attention,
embeddings, norms, and layers 0 and 23 stay BF16.

## Run grid

48 runs, stacked models only (decision log, 2026-10-20). **Order A**,
conversion and evaluation only:

| SwiGLU models | INT8 | INT4 | INT3 | Runs |
| --- | :---: | :---: | :---: | ---: |
| SwiGLU-8 100M fitted, 20-80% (8 targets) | ✓ | ✓ | ✓ | 24 |
| SwiGLU-8 100M slice-only, 20/50/80% | ✓ | ✓ | ✓ | 9 |
| 1B fitted, 20/50/80% (SwiGLU-7's S7-0 at 20/50%, SwiGLU-9's fitted 80%) | ✓ | ✓ | ✓ | 9 |

The slice-only rows answer question 3 at no training cost. The 1B rows are
fitted only: fitted versus slice-only at 1B is SwiGLU-9's question, and
Stacking-1 does not test gap closing (decision log, 2026-10-17). They run
from SwiGLU-9's fitted 80% model, pinned on 2026-10-20 after SwiGLU-9
finished.

**Order B**, QAT recovery from SwiGLU-8's token-0 starts, 100M tokens, 6 runs:

| Fitted models | INT4 | INT3 |
| --- | :---: | :---: |
| 20/50/80% | ✓ | ✓ |

Order B trains SwiGLU-8's S7-0 scope (the replacement modules only); every
projection in layers 1-22 is fake-quantized, including the dense-kept layers,
which stay frozen as they are in order A.

**Quantization alone** is not part of Stacking-1. The dense INT3 PTQ and
INT3 QAT runs planned here on 2026-10-17 moved to
[Quantization-2](../baseline/quantization-2.md), which runs
on Quantization-1's runner and adds INT3 QAT at 1B (decision log,
2026-10-20). After the INT3 QAT fix, the 1B run is not rerun: longer
training barely helps the fixed 100M dense INT3 QAT model, and against the
1B stacked models of equal or smaller byte saving a 1B run could not change
the KL or perplexity comparison (decision log, 2026-10-26;
[Quantization-2](../baseline/quantization-2.md#no-int3-qat-1b-rerun-2026-10-26)).

**Why order B stays at 100M.** Order A already has 1B rows because the 1B
SwiGLU models exist and quantizing them costs only an evaluation. Order B at
1B would cost about 15-20 GPU-hours per run, 90-120 in total. Dense INT4 QAT
gained almost nothing from 1B tokens over 100M in Quantization-1 (8K KL
0.0352 against 0.0360), so 1B order-B runs are justified only if the 100M
results show that order A and order B differ. That decision is taken after the
100M results (decision log, 2026-10-20). *Update (2026-11-01):* they differ
at every width, so order B goes to 1B
([Order B at 1B](#order-b-at-1b-plan-2026-11-01)).

**Why the fitted models, and why slice-only only at 100M.** SwiGLU-9 found
that local fitting no longer matters after 1B recovery tokens (8K KL gap
0.3-2.8%), but it still does at 100M (10-18%). Quantizing both arms at 1B
would therefore be redundant, while the 100M slice-only rows are the only test
of whether quantization changes fitting's head start. Every model used here
already exists, so the choice of arm costs no GPU time (decision log,
2026-10-20).

**Order comparison:** A and B both use 100M training tokens in total. Order C
runs only if A and B differ clearly, and is then compared with B at equal
total tokens (100M + 100M against 200M). Whether any stacking variant goes to
1B is decided after the 100M results. INT8 is PTQ only, as in Quantization-1.

## Quantization method

- **Same family as Quantization-1** (TorchAO 0.17.0): weight-only; INT8
  symmetric per output channel; INT4 and INT3 asymmetric with groups of 128
  input weights; round-to-nearest. INT4 uses Quantization-1's exact recipes
  and QAT path.
- **INT3 in TorchAO 0.17.0** (`src-torchao-0-17-0`, checked in the v0.17.0
  tag and in the copy installed on PERUN on 2026-10-17):
  `IntxWeightOnlyConfig(weight_dtype=torch.int3, granularity=PerGroup(128),
  mapping_type=ASYMMETRIC, version=2)` gives `IntxUnpackedToInt8Tensor`
  weights with values in [-4, 3], one int8 per value, a BF16 scale and an
  int8 zero point per group. Its linear layer dequantizes the weight and calls
  an ordinary BF16 matrix multiply; there is no INT3 kernel.
- **INT3 QAT uses a project fake quantizer.** TorchAO infers fake
  quantization from `IntxWeightOnlyConfig` only for symmetric mapping (an
  assert in `_infer_fake_quantize_configs`). The `-001` order-B INT3 runs
  therefore prepared with `IntxFakeQuantizeConfig(torch.int3, group_size=128,
  is_symmetric=False)`, which does not reproduce the deployed INT3; they are
  invalid ([INT3 order-B runs invalid](#int3-order-b-runs-invalid-2026-10-25)).
  From `-002` on, INT3 QAT uses `DeployedInt3FakeQuantizer`, whose simulated
  weights are bitwise equal to the deployed INT3 weights, and converts with the
  asymmetric PTQ recipe ([INT3 QAT fix](#int3-qat-fix-2026-10-25)). The method
  is simulated quantization during training (`src-qat-jacob-2018`, Section 3),
  with gradients passed through rounding by a straight-through estimator
  (`src-ste-bengio-2013`, Section 4).
- **Calculated bytes.** INT3 bytes are calculated from 3.25 bits per weight
  (3 bits plus a 16-bit scale and a 16-bit zero point per group, the INT4
  layout) and labelled as calculated; measured INT3 storage is about 8.2 bits
  per weight and is reported only as the container size. The same arithmetic
  is computed for INT8 (one byte per weight plus 3 bytes per output row, the
  BF16 scale and int8 zero point TorchAO's `Int8Tensor` stores) and INT4
  (4.25 bits per weight), where it must equal the measured bytes (check K4).
  A contract test reproduces Quantization-1's measured INT8 and INT4 bytes
  exactly. This is standard storage arithmetic, not a cited result.
- **INT3 PTQ uses the weakest standard method.** 3-bit results are usually
  reported with error-compensating methods such as GPTQ (`src-gptq-2023`) or
  AWQ (`src-awq-2024`); GPTQ's Section 5 reports that round-to-nearest
  "collapses completely" at 3 bits on OPT models. INT3 QAT is the fair INT3
  point. GPTQ is added only if INT3 PTQ is much worse than INT3 QAT; it would
  need its own environment decision.

## Width alignment

Groups of 128 need every quantized row length to be a multiple of 128. In a
SwiGLU block, the down projection's row length is the layer's width. The
allocator gives one layer per model a boundary width that is not a multiple:

| Target | Layer | Width | Padded width |
| --- | ---: | ---: | ---: |
| 20% | 9 | 4,915 | 4,992 |
| 30% | 4 | 5,324 | 5,376 |
| 40% | 6 | 3,686 | 3,712 |
| 60% | 21 | 5,529 | 5,632 |
| 70% | 18 | 3,379 | 3,456 |
| 80% | 21 | 2,252 | 2,304 |

The 50% and 75% models need no padding (observed in the SwiGLU-8 bundle
manifests). For INT4 and INT3 only, those layers get whole zero neurons
appended: a zero gate row, a zero up row, and a zero down column. A zero gate
and up row give a zero intermediate value, so the function is unchanged and
the architectural edit is not affected. INT8 is per-row and is never padded.
The real and padded widths are reported, and the padding weights and bytes
are counted. BF16 SwiGLU models stay untouched.

For INT3, zero stays exactly zero after quantization, and the appended zero
columns leave the real columns' scales unchanged, because TorchAO's affine
quantizer always keeps zero inside each group's range. INT4's quantizer
(MSLK's `int4_row_quantize_zp`) does not force zero into the range. Its
all-zero gate and up groups still dequantize to exactly zero (−8·s + 8s), but
the zero columns join the last group of each padded down-projection row. That
can widen that one group's range, at most one group per row in one layer.
Check K5 verifies on a real batch that the padded neurons output exactly zero.

In order B, the padded neurons are a fixed point of training, and a gradient
mask keeps them exactly zero under fake quantization; trainable-parameter
counts include the padding and are reported with it.

## Evaluation

The full SwiGLU-8 final evaluation and Quantization-1's 8K KL, so every point
is comparable with the existing frontier. The frozen protocol and data come
from SwiGLU-8's `prepare-001`, checked by sha256: full WikiText-2 validation
and test perplexity at 128/2,048/8,192 tokens (strides 64/1,024/4,096); the
five zero-shot tasks with the pinned harness, paired bootstrap differences
against SwiGLU-8's dense records (10,000 resamples, seed 21), macro accuracy,
and the likelihood comparison to dense; the 8K teacher KL on the 24 × 8,192
stream; and resident memory in a fresh process. The runner calls the same
functions with the same arguments as `swiglu_7.evaluate_final_model` and
`swiglu_8.final_teacher_kl`; a contract test pins the shared settings to
`swiglu-8.json`.

Each result also records the monitoring perplexity on SwiGLU-8's 128-token
monitoring batches: before quantization and after conversion for PTQ (the
quantization delta), and against the last fake-quantized row for QAT (the
conversion delta, like SwiGLU-8's BF16 conversion delta). No speed is
measured (decision log, 2026-10-17).

## Checks

Each check stops the run if it fails, except where noted.

| Check | What it compares | Tolerance |
| --- | --- | --- |
| K1 | Order A: the source bundle manifest's sha256 equals the configuration pin and the source result's record | exact |
| K2 | Order A: the monitoring perplexity of the loaded BF16 input equals the source run's `monitoring_after_bf16` | relative 1e-4 |
| K3 | Order B: the token-0 monitoring KL and perplexity of the start, before padding and fake quantization, equal SwiGLU-8's token-0 row | relative 1e-4 |
| K4 | INT8 and INT4: calculated eligible bytes equal measured eligible bytes | exact |
| K5 | Padded models: the deployed bundle's padded neurons output exactly zero gate and up values on one monitoring batch; order B also checks that the padded master weights are still zero after training (`K5-training`) | exact |
| K6 | Order-B INT3 and, from the INT4 `-002` reruns on, INT4 (decision log, 2026-11-01): after QAT preparation, before the first update, every projection's simulated weight equals the deployed recipe applied to the same weight (`K6-start`); at conversion, every deployed weight equals the trained simulated weight (`convert_qat`, recorded in `results.conversion.int{3,4}_fake_equals_deployed`). INT4 compares the weight `Int4Tensor` stores, not the MSLK kernel's arithmetic; see [Quantization-1](../baseline/quantization-1.md#int4-qat-conversion-gap-2026-11-01) | bitwise |
| K7 | Order B at 1B: the first 100M tokens retrace the 100M run ([Order B at 1B](#order-b-at-1b-plan-2026-11-01)). All six: the 1B stream starts with SwiGLU-8's 100M stream (`K7-stream`). INT3: before the first update, the fake-quantized token-0 row equals the 100M `-002` twin's (`K7-start`); the 10-100M rows are compared with the twin's (`K7-review`, recorded, never stops) | exact; relative 1e-4; 1% review |

The monitoring perplexity after padding (before quantization) is recorded but
not enforced, because a matrix multiply over a longer row may sum in a
different order. The conversion change (`conversion_ppl_delta`), the token-0
validation KL, and the first-step gradient norm are also recorded, not
enforced. The 24-batch monitoring perplexity mostly measures damage at target
position 1 (experiment-audit session, 2026-10-25), so it is not a gate. The existing checks of Quantization-1's code also apply:
frozen weights outside the scope, unchanged logical dimensions, tied
embeddings, and the packed-bundle round trip.

## Implementation

Implemented on 2026-10-17; nothing was run on PERUN.

| File | Role |
| --- | --- |
| [stacking-1.json](../../../../workflows/configs/model/stacking/stacking-1.json) | Grid, stage lists, formats, and sha256 pins of SwiGLU-8's preparation and the 14 source models; shared blocks equal to `swiglu-8.json` and `quantization-1.json` |
| [stacking_1.py](../../../../workflows/runs/model/stacking/stacking_1.py) | `run --variant`, `list-variants`, `inputs`, `measure-bundle` |
| [quantization.py](../../../../src/mlp_replacement/compression/quantization.py) | INT3 recipe and scope check, replacement projection names, QAT for INT4 and INT3 with a trainable subset, padding, gradient mask, calculated bytes |
| [quantized_bundles.py](../../../../src/mlp_replacement/evaluation/quantized_bundles.py) | Packed bundles may record SwiGLU topology (padded and real widths) and calculated bytes; INT3 accepted |
| [bundles.py](../../../../src/mlp_replacement/evaluation/bundles.py) | Resident-memory measurement takes a loader, so packed bundles can be measured |
| [stacking_1.sbatch](../../../../workflows/jobs/perun/stacking_1.sbatch), [stacking_1.sh](../../../../workflows/jobs/local/stacking_1.sh) | PERUN and local launchers |
| [test_stacking1_contracts.py](../../../../tests/test_stacking1_contracts.py) | Configuration, grid, pins, padding, gradient mask, bytes, recipes, INT3 PTQ and QAT, packed-bundle round trip |

Quantization-1's INT8 and INT4 recipes, recipe records, and QAT behaviour are
unchanged; its QAT contract test, which needs TorchAO, passes. SwiGLU-7, -8,
and -9 code is not modified; Stacking-1 imports their helpers.

**Commands.** One variant per process; the variant id fixes everything else.

```bash
python -m workflows.runs.model.stacking.stacking_1 list-variants --group a100m   # a100m, a1b, b, b1b
bash workflows/jobs/local/stacking_1.sh run --variant a-fitted-0.2-100m-int3 \
  --prepared data/results/workflows/model/swiglu-8/prepare-001/result.json \
  --source-root data/results/workflows/model \
  --output-dir data/results/workflows/model/stacking-1/a-fitted-0.2-100m-int3-001
```

Variant ids are `a-{arm}-{target}-{100m|1b}-int{8,4,3}` and
`b-fitted-{target}-{100m|1b}-int{4,3}`. Each output holds `result.json`, `run.json`, the
packed `model/` bundle, and the benchmark records; QAT runs keep a 25M-token
checkpoint while running, deleted after the result is validated. A completed
bundle is reused on resume, and the run fingerprint covers the configuration,
SwiGLU-8's preparation, the source pins, and all maintained code.

**What was tested.** CPU only, on 2026-10-17: the 14 new contract tests pass
in the project's Python (TorchAO tests skipped) and all 14 pass with TorchAO
0.17.0 in a separate scratch environment, together with Quantization-1's 11
tests; the full suite passes (119 tests, 5 skipped). A CPU probe showed that
converting INT3 QAT weights changes about 0.5-0.7% of them by one step
relative to their fake-quantized values on random Gaussian weights; the
library's two rounding paths differ, and matching the scale precision did not
remove it. GPU execution, MSLK kernels, autocast on INT3 weights, and full
evaluation are untested and are part of the intended runs.

*Correction (2026-10-25):* the probe's random Gaussian weights have no
outliers, so the difference looked small. On the real SmolLM2 weights, the
INT3 fake quantizer distorts the outlier weights (for example 9.375 becomes
−2.752 while the deployed INT3 keeps 9.625). The difference was a defect, not
a limitation
([INT3 order-B runs invalid](#int3-order-b-runs-invalid-2026-10-25)).

On 2026-10-20 the dense group was removed and SwiGLU-9's fitted 80% model was
pinned from the copy of its run on the researcher's PC (`result.json`
sha256 `69b86261…`, bundle manifest `1f384bb4…`; the result's own manifest
record equals the bundle digest). The pre-submission check compares both
values with the files on PERUN. CPU only: the 15 contract tests give 11
passes and 4 TorchAO skips in the project's Python, and the full suite passes
(120 tests, 5 skipped).

### INT3 QAT fix (2026-10-25)

The shared INT3 QAT code was fixed for Stacking-1 order B and Quantization-2
together (decision log, 2026-10-25). The quantizer, the tests, and what was
tested are described in
[Quantization-2's fix record](../baseline/quantization-2.md#int3-qat-fix-2026-10-25).
Order A, order-B INT4, and every PTQ path are unchanged.

| File | Change |
| --- | --- |
| [quantization.py](../../../../src/mlp_replacement/compression/quantization.py) | `DeployedInt3FakeQuantizer`: scale and zero point from the deployed INT3 recipe, so simulated weights equal deployed weights bit for bit; `int3_fake_quantization_check`; `convert_qat` stops if a converted INT3 weight differs from the trained simulated weight |
| [stacking_1.py](../../../../workflows/runs/model/stacking/stacking_1.py) | Check `K6-start` right after QAT preparation in order B (INT3; INT4 from 2026-11-01), before the first update; the conversion half runs inside `convert_qat` |
| [stacking_1.sbatch](../../../../workflows/jobs/perun/stacking_1.sbatch) | Optional `--attempt NNN` (default `001`), placed before `--resume`; the output is `VARIANT-NNN`, so a rerun never overwrites the invalid `-001` evidence |
| [test_stacking1_contracts.py](../../../../tests/test_stacking1_contracts.py) | Planted outlier, trained masters off the BF16 grid, the old TorchAO quantizer caught by both checks, and the pinned SmolLM2 weights |

**Effect on runs.** The run fingerprint covers all maintained code, so no
`-001` run can be resumed (all completed). The three reruns are
`b-fitted-{0.2,0.5,0.8}-100m-int3-002`, group-b array tasks 1, 3, and 5. A
`-002` task resumes only from unchanged code, configuration, and environment,
so the PERUN checkout is not updated until the reruns finish. The Stacking-1
analysis takes order-B INT3 from `-002`.

## Pre-run resource plan

Recorded 2026-10-17 under the [PERUN resource contract](../../../infrastructure/perun/perun-resources.md)
and revised on 2026-10-20 for the 48-run grid. Estimates, not measurements.

| Group | Tasks | Per task, expected | GPU-h expected | Time limit | Ceiling GPU-h | Basis | Confidence |
| --- | ---: | --- | ---: | --- | ---: | --- | --- |
| A 100M (`a100m`) | 33 | 0.3-0.5 h | 10-17 | 2 h | 66 | SwiGLU-8 final evaluation 849 s with the 8K KL; Quantization-1 evaluations 0.27-0.29 h; INT3 dequantization adds an estimated 5 min | Medium |
| A 1B (`a1b`) | 9 | 0.3-0.5 h | 2.5-4.5 | 2 h | 18 | same | Medium |
| B (`b`) | 6 | 1.5-3 h | 9-18 | 6 h | 36 | SwiGLU-8 recovery 1.03 h per 100M; Quantization-1 QAT 1.51 h; INT3 fake-quantization overhead unknown | Low |
| **Total** | **48** | | **about 22-40** | | **120** | | |

- **Per task:** one H200, 8 CPUs, 128 GiB RAM, `gpu_short`; one process,
  no communication between tasks. Expected CPU use 170-320 core-hours;
  requested ceiling 960 core-hours.
- **Parallelism:** at most 12 running tasks (`MaxJobs`), 48 submitted
  (`MaxSubmitJobs`, array tasks counted individually), and new submissions
  refused while 4 of your jobs run (perun-status.md). The 48 tasks are
  exactly the submission limit, so they go in one sitting while none of your
  jobs is queued: three array submissions with a delayed start, order B
  first because its tasks are the longest (runbook 10.2). Critical path about
  3-4 h once running.
- **Memory:** peak VRAM 14-16 GiB for evaluation (Quantization-1 measured
  13.9-15.4); under 40 GiB for QAT (reasoning, not a measurement).
- **Storage:** PROJECT about 70-100 GB of packed bundles (INT3 is stored at
  INT8 size, 1.4-2.1 GB each). SCRATCH per task: the source bundle (up to
  3.4 GB) and output (up to 2.2 GB); QAT tasks add the 0.4 GB stream and up to
  two checkpoint generations. About 265 of 8,424 project GPU-hours were used
  before Stacking-1 (about 210 recorded on 2026-10-17 plus SwiGLU-9's 55.5
  workflow-timing hours).

### Reruns with the fixed quantizer (recorded 2026-10-25)

Estimates, not measurements, based on the three `-001` order-B INT3 tasks,
which ran the same stages; the two new weight checks add seconds (estimate).
Submitted in one sitting with Quantization-2's INT3 QAT 100M rerun
([Quantization-2 plan](../baseline/quantization-2.md#reruns-with-the-fixed-quantizer-recorded-2026-10-25)).

| Group | Tasks | Per task, expected | GPU-h expected | Time limit | Ceiling GPU-h | Basis | Confidence |
| --- | ---: | --- | ---: | --- | ---: | --- | --- |
| B INT3 `-002` (array tasks 1, 3, 5) | 3 | 1.2-1.6 h | 3.6-4.8 | 3 h | 9 | `-001` stage times 1.42, 1.35, 1.20 h (recovery 0.94-1.11 h, evaluation 0.25-0.31 h) | Medium |

- **Per task:** one H200, 8 CPUs, 128 GB RAM, `gpu_short`; one process, no
  communication between tasks. Expected CPU use 29-38 core-hours; ceiling 72.
- **Memory:** `-001` measured 25.9-33.3 GiB peak VRAM and 6.1-8.9 GiB peak
  RAM.
- **Storage:** PROJECT 1.4-2.1 GB per packed INT3 bundle; the `-001` outputs
  are kept. SCRATCH: the 0.4 GB stream and up to two checkpoint generations of
  the replacement weights.
- **Stage 1 in total** (with Quantization-2's 100M rerun): 4 single-GPU jobs,
  expected 5.3-7.0 GPU-hours, ceiling 13. The earlier "4 running jobs"
  submit refusal no longer applies (perun-status.md, 2026-10-25); the limits
  are now 20 running jobs and 50 queued tasks.

## INT3 order-B runs invalid (2026-10-25)

`b-fitted-0.2-100m-int3-001`, `b-fitted-0.5-100m-int3-001`, and
`b-fitted-0.8-100m-int3-001` are invalid. Order B prepares with
`prepare_qat` and converts with `convert_qat`. At 3 bits these use TorchAO's
generic INT3 fake quantizer, which does not reproduce the deployed INT3 and
distorts the outlier weights. The cause and the real-weight evidence are in
[Quantization-2's incident record](../baseline/quantization-2.md#int3-qat-incident-2026-10-25).
Order A quantizes once with `apply_ptq` and never uses the fake quantizer, so
all 42 order-A runs, including the 14 INT3 ones, are unaffected, and so are
the three order-B INT4 runs.

Observed in the result artifacts. KL is the 8K teacher KL, PPL is WikiText-2
test perplexity at 8K, and macro is the five-task mean.

| Width reduction | A-INT4 KL / PPL / macro | B-INT4 | A-INT3 | B-INT3 (invalid) |
| --- | --- | --- | --- | --- |
| 20% | 0.164 / 8.28 / 0.613 | 0.102 / 8.00 / 0.625 | 0.288 / 9.46 / 0.600 | 0.302 / 9.68 / 0.569 |
| 50% | 0.318 / 10.52 / 0.540 | 0.255 / 9.92 / 0.554 | 0.432 / 11.91 / 0.540 | 0.484 / 13.03 / 0.501 |
| 80% | 0.707 / 17.49 / 0.456 | 0.579 / 15.71 / 0.463 | 0.875 / 21.48 / 0.446 | 1.011 / 25.01 / 0.429 |

- **Start:** the order-B INT3 runs start training with a validation KL of
  5.165, 5.882, and 6.639; the INT4 runs start at 0.247, 0.540, and 1.292.
- **Conversion change** (recorded `conversion_ppl_delta`, not enforced):
  INT4 −0.03, +0.18, and +1.38 PPL; INT3 −2.41, +10.04, and +30.77 PPL.
  *Interpretation:* at 20% the conversion helps, because the frozen dense-kept
  layers get their outliers back.
- **Outcome:** order B beats order A at every width for INT4, but loses for
  INT3. Until the INT3 order-B runs are rerun with a fixed quantizer, there is
  no evidence on whether the order matters at INT3.
- **Cost:** about 3.6-4.3 workflow GPU-hours (1.20-1.42 h per order-B task).
- **Reruns:** the fix is implemented ([INT3 QAT fix](#int3-qat-fix-2026-10-25));
  the `-002` reruns are documented here when they finish. For reference, the
  `-001` first-step gradient norms were 5.37, 53.50, and 199.56 (INT4: 0.16,
  1.27, 3.16); a fixed run is expected to look INT4-like (recorded, not
  enforced).

## Order-B INT4 reruns with the fixed quantizer (2026-11-01)

**What and why.** The `-001` order-B INT4 runs trained with TorchAO's INT4
fake quantizer. That quantizer rounds the FP32 master weight, while deployment
rounds its BF16 cast ([Quantization-1](../baseline/quantization-1.md#int4-qat-conversion-gap-2026-11-01);
decision log, 2026-11-01). The three runs were rerun as `-002` with the
deployed-path INT4 quantizer.
- Job 140575, array tasks 0, 2 and 4; commit `b59da5c`; 3 h limit; all
  completed.
- K6 passed at the start and at conversion: 66 of 66 projections bitwise
  equal.
- Values were read on PERUN; the files are not home yet.

| Target | 8K KL `-001` → `-002` | Change | Test PPL 8K `-001` → `-002` | Macro `-001` → `-002` | Order A 100M INT4 KL | Order A 1B INT4 KL |
| ---: | --- | ---: | --- | --- | ---: | ---: |
| 20% | 0.1022 → 0.1023 | +0.1% | 8.00 → 8.00 | 0.6251 → 0.6230 | 0.1642 | 0.1409 |
| 50% | 0.2554 → 0.2485 | -2.7% | 9.92 → 9.88 | 0.5535 → 0.5505 | 0.3185 | 0.2386 |
| 80% | 0.5791 → 0.5645 | -2.5% | 15.71 → 15.39 | 0.4632 → 0.4678 | 0.7066 | 0.5225 |

- **Much smaller than in dense INT4 QAT (-19%), as predicted.** Only the
  replacement modules train with FP32 masters. The dense-kept projections stay
  frozen BF16, on the grid, so they never mismatched.
- **Prediction check:** the change in KL is -0.0069 at 50% and -0.0146 at 80%,
  with none at 20%. An estimate made before the rerun, comparing `-001` INT4
  with the bit-exact INT3 `-002`, gave 0.005-0.016 at 50% and 80% and none at
  20%.
- **Macro accuracy moves by at most 0.5 points.** It goes down at 20% and 50%
  and up at 80%; single runs.
- **The order comparison is unchanged.** Order B still beats order A at 100M at
  every width.
- **Against order A at 1B:** order B at 100M wins at 20% but not at 50% or 80%.
- From here on, order-B INT4 uses the `-002` runs, including as the 100M
  twins of order B at 1B.

## Order B at 1B (plan, 2026-11-01)

Planned and implemented on 2026-11-01; nothing has run on PERUN. The decision
row is proposed and waits for the researcher's approval.

**Why.** At 100M tokens order B beats order A at every width. 8K teacher KL
from the result artifacts:

| Width reduction | INT4: A 100M / B 100M / A 1B | INT3: A 100M / B 100M (`-002`) / A 1B |
| --- | --- | --- |
| 20% | 0.164 / 0.102 / 0.141 | 0.288 / 0.161 / 0.266 |
| 50% | 0.318 / 0.255 / 0.239 | 0.432 / 0.288 / 0.360 |
| 80% | 0.707 / 0.579 / 0.523 | 0.875 / 0.606 / 0.717 |

- The order-B validation KL still fell 9-16% over the last 50M tokens in all
  six runs, while dense INT4 QAT gained 2% from 1B tokens (0.0360 against
  0.0352).
- At INT3, order B at 100M already beats order A at 1B. The 1B runs add the
  frontier above 52% byte saving and show how large the gap becomes.
- At INT4, order A at 1B leads at 50% and 80%, so only an equal-token run
  can tell whether the order matters there.
- The B INT4 values above come from the `-001` runs, which used the INT4
  quantizer that the 2026-11-01 fix replaced; the `-002` reruns (runbook 10.6)
  replace them.

**Design.** Only the token budget, the recipe, and the stream change against
the 100M runs.

| Item | Choice | Reason |
| --- | --- | --- |
| Runs | `b-fitted-{0.2,0.5,0.8}-1b-int{4,3}`, group `b1b`, outputs `-001` | 1B runs use the 20/50/80% targets (decision log, 2026-10-16) |
| Start | SwiGLU-8's fitted token-0 starts, as at 100M | At 20% and 50% it equals SwiGLU-7's start at token 0 (SwiGLU-8's B1 check); SwiGLU-9's fitted 80% started from the same file |
| Recipe | `recovery_1b`, equal to SwiGLU-7's recovery block: 1B tokens, segments ending at 100M and 1B, validation every 25M, perplexity at 10/25/50/100/250/500M and 1B | The recipe of the order-A 1B sources (SwiGLU-7's S7-0 at 20/50%, SwiGLU-9's fitted 80%) |
| Stream | SwiGLU-7's 1B stream (sha256 `f78bf805…`), pinned through its preparation (`98d330de…`) | The order-A 1B sources trained on it; its first 100M tokens are SwiGLU-8's stream |
| Quantizers | The deployed-path INT3 and INT4 fake quantizers, with K6 | Decision log, 2026-10-25 and 2026-11-01 |
| Rest | Trainable scope, padding, conversion, export, and evaluation as at 100M | |

Each 1B run pairs with order A at 1B of the same target and bits: the same
start, stream, recipe, and token count. It also pairs with its own 100M
order-B run, which shows what the extra 900M tokens give.

**Checks.** K3-K6 apply as at 100M. K3 still compares the start with
SwiGLU-8's 100M run, because the start file comes from it. The new check K7
follows from the design: a 1B run has the same start, recipe, and stream as
its 100M twin, and its first segment ends exactly at 100M. Its first 100M
tokens therefore follow the same checkpoint schedule (a contract test checks
this), and the run should retrace the twin there, apart from GPU drift.

- `K7-stream`, all six runs, stops on failure: the first 100M tokens of the
  1B stream hash to SwiGLU-8's 100M stream.
- `K7-start`, INT3, stops on failure: before the first update, the
  fake-quantized token-0 row (validation KL and monitoring perplexity) equals
  the twin's within relative 1e-4. This catches a wrong quantizer, padding,
  or start in the first minutes.
- `K7-review`, INT3, recorded and never stopping: the 10, 25, 50, 75, and
  100M rows are compared with the twin's against the 1% review limit, as
  SwiGLU-8's B2 and SwiGLU-9's C2. A flagged value is investigated before any
  result is used. It does not stop the run, because SwiGLU-9 saw a +43.7%
  transient at 50M that recovered within 25M tokens.
- INT4 has no in-run twin: its twins are the 100M INT4 `-002` reruns, which
  have not run yet and cannot be pinned. The analysis makes the same
  comparison once they are home.

Monitoring perplexity enters only the token-0 equality check, never a quality
judgement (experiment-audit session, 2026-10-25).

**Analysis.** For each target and bit width: 8K KL, test perplexity, and task
macro of B at 1B against A at 1B and against B at 100M; frontier membership
by the [win criterion](#win-criterion). There is one run per variant, so
there is no QAT noise estimate. The nearest reference is SwiGLU-9's 0.31%
twin noise for BF16 recovery at 1B.

**Implementation.**

| File | Change |
| --- | --- |
| [stacking-1.json](../../../../workflows/configs/model/stacking/stacking-1.json) | `recovery_1b`; group `b1b`; `sources.swiglu7_prepare_sha256`; `sources.b1b_twins`, the three INT3 `-002` results by sha256 |
| [stacking_1.py](../../../../workflows/runs/model/stacking/stacking_1.py) | Group `b1b`; the recipe per group; SwiGLU-7's stream with `K7-stream`; twin validation (completed, same recipe and stream, K6 passed) and K7; `inputs` stages the 1B stream and the twin instead of the 100M stream |
| [stacking_1.sbatch](../../../../workflows/jobs/perun/stacking_1.sbatch) | Accepts `group b1b` |
| [test_stacking1_contracts.py](../../../../tests/test_stacking1_contracts.py) | The recipe equals SwiGLU-7's; the first-100M schedule is identical; the staging list; `K7-stream`; twin validation (a twin without K6, like an invalid `-001` run, is rejected); K7 stops only at token 0; the pins match the downloaded artifacts |

Group `b` keeps its variant ids, identities, and behaviour. Its run
fingerprint changes with the code, which affects no completed run. The INT4
`-002` reruns (runbook 10.6) run from `b59da5c`, before this change, so the
PERUN checkout is updated only after they finish.

**What was tested.** CPU only, 2026-11-01:

- The project's Python: the full suite passes (139 tests, 10 skipped; 132
  before).
- The TorchAO 0.17.0 probe environment: all 25 Stacking-1 and 13 quantization
  contract tests pass, none skipped.
- On the downloaded artifacts: the three twin pins, the SwiGLU-7 preparation
  pin, and the stream path match, and the twin validation accepts all three
  twins and reads their 0-100M rows.
- `list-variants --group b1b` and `inputs` print the expected lists, and the
  runbook pre-check prints OK for every local file (the 4 GB stream is not
  local).
- `bash -n` passes on the launcher.

Not tested: GPU execution, staging of the 4 GB stream, K7 on real runs, and
the run time.

### Pre-run resource plan (recorded 2026-11-01)

Estimates, not measurements, under the
[PERUN resource contract](../../../infrastructure/perun/perun-resources.md).

| Group | Tasks | Per task, expected | GPU-h expected | Time limit | Ceiling GPU-h | Basis | Confidence |
| --- | ---: | --- | ---: | --- | ---: | --- | --- |
| B 1B (`b1b`) | 6 | 10-12.5 h | 60-75 | 18 h | 108 | Order-B 100M QAT training 0.95-1.15 h per 100M tokens (six runs), times ten; evaluation 0.21-0.31 h; SwiGLU-7 and SwiGLU-9 BF16 1B training 9.1-10.4 h | Medium: linear extrapolation, node-to-node variance unknown |

- **Per task:** one H200, 8 CPUs, 128 GB RAM, `gpu_short` (2-day maximum); one
  process, no communication between tasks. Expected CPU use 480-600
  core-hours; ceiling 864.
- **Parallelism:** one 6-task array. All six can run at once (20 running jobs
  and 50 queued tasks; perun-status.md, 2026-10-25), so the critical path is
  about one task once they start.
- **Memory:** the 100M order-B runs peaked at 25.9-36.6 GiB VRAM and 6.1-9.0
  GiB RAM. The model, batch, and sequence length are the same, so the same is
  expected (reasoning, not a measurement).
- **Storage:** SCRATCH needs about 20 GB per task: the 4 GB stream, the start
  and inputs, and the checkpoint space that the 100M runs' storage preflight
  estimated (8.4-14.8 GB). PROJECT needs about 9-11 GB of packed bundles (the
  100M bundles were 1.33-2.12 GB); PROJECT held 253 of 1,000 GB on
  2026-10-26.
- **Time limit:** 18 h is about 45% above the upper estimate. A task that
  runs out of time stages out at the TERM signal, 30 minutes before the limit,
  and resumes from its last 25M-token checkpoint (runbook 10.7).
- **Share:** under 1% of the project's 8,424 GPU-hours.

## Limits

- One run per variant.
- INT3 PTQ uses round-to-nearest, which is known to be weak at 3 bits.
- INT3 bytes are calculated, not measured; measured INT3 storage is an 8-bit
  container.
- INT3 QAT uses the project's fake quantizer (from `-002` on), INT4 QAT
  TorchAO's dedicated INT4 one, so the two QAT points do not share one
  fake-quantization implementation. The `-001` INT3 runs used TorchAO's
  generic quantizer, which does not reproduce the deployed INT3, and are
  invalid ([INT3 order-B runs invalid](#int3-order-b-runs-invalid-2026-10-25)).
- For INT4, the padding zeros can widen the range of one group per row in one
  layer per padded model.
- Order B fake-quantizes the dense-kept layers without training them, as order
  A quantizes them without training.
- Speed is not evaluated; the quantized kernels are not optimized, and INT3
  has no kernel (decision log, 2026-10-16 and 2026-10-17).

## Sources

Entries are registered in [the source registry](../../../../llm-wiki/raw/sources.yml).
Registration and the checked locators do not constitute full ingestion. The
paper locators were found through automated fetches of the ar5iv HTML
renderings on 2026-10-17 and are not yet confirmed in the PDFs.

- `src-torchao-0-17-0`: [pinned source](https://github.com/pytorch/ao/tree/v0.17.0),
  INT3 locators listed in the registry note (`IntxWeightOnlyConfig`,
  `IntxUnpackedToInt8Tensor`, `_infer_fake_quantize_configs`,
  `IntxFakeQuantizeConfig`, `QATConfig`).
- `src-mslk-1-1-0`: INT4 kernels; `int4_row_quantize_zp` read in the copy
  installed on PERUN.
- `src-qat-jacob-2018`: Jacob et al., CVPR 2018,
  [arXiv 1712.05877](https://arxiv.org/abs/1712.05877v1), Section 3, Eq. (12),
  simulated quantization during training.
- `src-ste-bengio-2013`: Bengio, Léonard, and Courville,
  [arXiv 1308.3432](https://arxiv.org/abs/1308.3432v1), Section 4,
  straight-through estimator.
- `src-gptq-2023`: Frantar et al., ICLR 2023,
  [arXiv 2210.17323](https://arxiv.org/abs/2210.17323v2), Section 5,
  Tables 3-5, round-to-nearest against GPTQ at 3 and 4 bits. Not implemented.
- `src-awq-2024`: Lin et al., MLSys 2024,
  [arXiv 2306.00978](https://arxiv.org/abs/2306.00978v6), Section 5.2,
  Table 4, INT3 with groups of 128. Not implemented.
