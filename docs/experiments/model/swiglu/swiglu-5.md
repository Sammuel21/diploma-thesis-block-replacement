---
metadata_version: 1
title: SwiGLU-5 Efficient Global-Recovery Search and Confirmation
type: experiment-workflow
category: experiments/model/swiglu
status: draft
created: 2026-09-19
modified: 2026-09-20
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
sources:
  notebooks:
    - notebooks/model/swiglu/swiglu-5.ipynb
  reference_artifacts:
    - data/results/workflows/model/swiglu-3/run-001.json
---

# SwiGLU-5 Efficient Global-Recovery Search and Confirmation

Status: both runners and the load-only report are implemented. The bounded
search has not completed, and confirmation must not run until the search
artifact has been reviewed.

## Purpose

SwiGLU-5 asks whether better local initialization, discrete width allocation,
and one composition-aware refit can close more of the dense-model KL and
perplexity gap than the exact SwiGLU-3 operators when every candidate receives
the meaningful SwiGLU-4 recovery configuration: constant `3e-5`, pure
teacher-to-student KL at temperature 1, zero weight decay, and
replacement-only AdamW.

This remains a homogeneous SwiGLU experiment. A later heterogeneous workflow
may spend small operators on easy blocks and give additional SwiGLU width to
sensitive blocks; SwiGLU-5 does not test that allocation class.

## Compatibility and comparison levels

The search requires a completed schema-1 SwiGLU-3 artifact and verifies the
pinned model/tokenizer revisions, 24-layer topology, hidden and MLP widths,
eligible/protected layers, 393,216-pair calibration selection, exact 20% and
50% allocations and states, packed C4 stream fingerprint, sequence length,
2,048-token effective updates, evaluation partitions, and BF16/FP32 precision
contract. Recorded asset paths may relocate, but content hashes and stream
fingerprints may not change.

The artifact keeps three roles separate:

1. imported published SwiGLU-3 evidence, retaining its original `1e-5`
   recovery configuration;
2. `S5-C0`, which reconstructs the exact SwiGLU-3 operators but trains them
   with the fixed SwiGLU-5 recovery configuration; and
3. `S5-C1` through `S5-C4`, which change initialization, allocation, or local
   capture context while holding recovery and token order fixed.

Search-time 5M comparisons against SwiGLU-3 are limited to its retained
validation trajectory. Equal-budget WikiText comparisons become available at
10M and 100M in confirmation.

## Candidate construction

For each 20% and 50% eligible-MLP removal target:

| ID | Initialization | Width allocation |
| --- | --- | --- |
| `S5-C0` | Exact SwiGLU-3 operator states | Exact SwiGLU-3 ranked widths |
| `S5-C1` | Output-aware reconstruction | SwiGLU-3 ranked widths |
| `S5-C2` | Legacy teacher subset | Discrete width curves |
| `S5-C3` | Output-aware reconstruction | Discrete width curves |
| `S5-C4` | Composition-aware refit | Widths of the best pre-recovery parent |

Output-aware initialization copies selected gate/up rows and down columns,
initializes one down-projection bias from the mean output residual, fits down
weight plus bias for at most eight epochs, and then applies the established
full FP32 local fit. This is a project adaptation inspired by MoDeGPT's
module-output reconstruction, not a reproduction claim.

The discrete allocator fits 25%, 37.5%, 50%, 62.5%, 75%, 87.5%, and dense
anchors for every eligible layer, monotonicizes noisy singleton-KL curves, and
solves a multiple-choice knapsack. The additive curve objective is a
project-proposed hypothesis; integrated model KL and WikiText perplexity remain
authoritative. Bias parameters are included in requested-versus-realized
budget reporting.

`S5-C4` freezes the chosen parent as an input-capture context, derives dense
local targets at those student-distribution inputs, fits every replacement
without mutating the capture context, and installs the states only after the
single recapture round is complete.

## Search recovery and compute guard

The dense final normalized hidden states for the first requested 5M stream
tokens are stored in atomic BF16 shards. A frozen copy of the tied output head
reconstructs teacher logits after the dense Transformer is released. Fixed
sample checks replay the cache-capture batch geometry and must keep
cached-versus-online teacher KL at or below `1e-5`.
Before local fitting begins, a storage preflight bounds two non-overlapping
peaks: width-curve fitting and recovery with the teacher cache, selected
candidate states, and at most four transient candidate checkpoints during an
atomic write. It adds a 15% reserve and fails early if free space is
insufficient; the cache builder still independently enforces its 25 GiB
minimum.

The runner profiles `4x4`, `8x2`, and `16x1` sequence/accumulation geometries,
selects the largest below 22 GiB peak VRAM, and retains `torch.compile` only
when it is at least 10% faster with loss divergence no greater than `1e-4`.
Each throughput trial measures 16 optimizer updates (32,768 token positions)
so one-time startup work does not dominate the runtime projection. All trials
use disposable states, and the pre-profile RNG state is restored before the
tournament. Training, evaluation, and checkpoint time are recorded separately.
The 20% and 50% targets run sequentially. During qualification, the runner
keeps only `S5-C0`, the currently leading two challengers, and the candidate
being atomically written. Displaced candidates are removed immediately.
Width-fit tensors are deleted as soon as no untrained candidate references
them, while their histories, metrics, hashes, and allocation recipes remain in
the JSON. Only one reconstructable 5M winner endpoint per target survives. The
approximately 19 GiB teacher-hidden cache is temporary and is deleted after
its final consumer.

Every candidate reaches the 2M requested boundary. The best two challengers
per target and `S5-C0` continue to the exact optimizer boundary for 5M
requested tokens. Challengers must have 5M WikiText perplexity no worse than
`S5-C0`; final selection uses fixed T=1 KL, a `1e-6` tie interval, then
perplexity and candidate ID. `S5-C0` remains eligible throughout and is the
fallback when no challenger passes.

Before recovery, the workflow projects the complete 38M candidate-token
tournament from measured preparation time and recovery throughput with a 15%
reserve. By default, a projection above six hours writes
`budget_guard_rejected` and stops without silently shrinking the design.
`--allow-over-budget` preserves the projection record but permits an explicitly
authorized longer run.

## Confirmation

Confirmation is a separate process for one explicit target. It verifies the
search and original SwiGLU-3 artifacts, restores the exact selected 5M
replacement, optimizer, RNG, update count, and token cursor, then continues the
same finite stream to 100M with online dense-teacher inference. It re-profiles
online-teacher batching, records fixed KL every 5M, and performs full
evaluation at 10M, 25M, 50M, and 100M. Milestones retain metrics rather than
model or optimizer snapshots. A successful run keeps only FP32 replacement
weights for the final state and the best fixed-validation-KL state; when the
final state is also best, both records refer to the same file. If the original
5M search endpoint remains best, confirmation references that retained search
endpoint instead of copying it.

The 10M and 100M tables pair exact boundaries with the matching SwiGLU-3
target. The source artifact has no observed 10M runtime checkpoint, so that
single runtime difference is explicitly unavailable rather than inferred.

## Commands

Run the bounded search first on darthmachinus:

```bash
python -m workflows.runs.model.swiglu.swiglu_5_search \
  --config workflows/configs/model/swiglu/swiglu-5-search.json \
  --source data/results/workflows/model/swiglu-3/run-001.json \
  --output data/results/workflows/model/swiglu-5/search/<unique>.json
```

A failed search before recovery training may continue from its original output:

```bash
python -m workflows.runs.model.swiglu.swiglu_5_search \
  --config workflows/configs/model/swiglu/swiglu-5-search.json \
  --source data/results/workflows/model/swiglu-3/run-001.json \
  --output data/results/workflows/model/swiglu-5/search/run-001.json \
  --resume \
  --allow-over-budget
```

After analyzing that completed artifact, run one selected target:

```bash
python -m workflows.runs.model.swiglu.swiglu_5_confirmation \
  --config workflows/configs/model/swiglu/swiglu-5-confirmation.json \
  --search-artifact <completed-search.json> \
  --target 0.2 \
  --output data/results/workflows/model/swiglu-5/confirmation/<unique>.json
```

Search resume is limited to failures before recovery training and does not add
large recovery checkpoints. Confirmation does not support resume, and neither
process has a smoke mode. This low-storage policy does not change candidate
training, selection, token order, or confirmation continuation. No SwiGLU-5
Perun launcher is provided.

## Reporting and interpretation boundary

[`swiglu-5.ipynb`](../../../../notebooks/model/swiglu/swiglu-5.ipynb) loads
JSON artifacts only. It reports the imported SwiGLU-3 rows, dense reference,
pre-recovery candidates, layer widths, recovery curves, selection decisions,
runtime decomposition, and optional confirmation comparisons. Before a search
artifact exists, this document describes an implemented experimental contract,
not an empirical result. A null result in which `S5-C0` remains best is valid.
