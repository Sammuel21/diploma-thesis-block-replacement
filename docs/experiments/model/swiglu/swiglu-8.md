---
metadata_version: 1
title: SwiGLU-8 - Local-Fitting Attribution and Homogeneous Control Spectrum
type: experiment-workflow
category: experiments/model/swiglu
status: draft
created: 2026-10-10
modified: 2026-10-10
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
reference_artifacts:
  - data/results/workflows/model/swiglu-7/prepare-001/result.json
  - data/results/workflows/model/swiglu-7/S7-0-target-0.2-run-001/result.json
  - data/results/workflows/model/swiglu-7/S7-0-target-0.3-run-001/result.json
  - data/results/workflows/model/swiglu-7/S7-0-target-0.4-run-001/result.json
  - data/results/workflows/model/swiglu-7/S7-0-target-0.5-run-001/result.json
---

# SwiGLU-8 - Local-Fitting Attribution and Homogeneous Control Spectrum

**Status:** design settled on 2026-10-10. Not implemented, no runs. The binding
decisions are recorded in the [shared decision log](../../../agents/decisions.md);
this page explains them. If this page and the log disagree, the log wins until
the conflict is raised with the researcher.

## Questions

1. Does local fitting still improve the compressed model after recovery, or
   does recovery make it unnecessary?
2. Can the local fit itself be improved, including by capturing calibration
   pairs from 8,192-token context instead of 128?
3. What does homogeneous SwiGLU achieve across 20-80% eligible-MLP removal
   at a 100M-token recovery budget? These runs are the homogeneous control
   spectrum for the heterogeneous-operator chapter.

## Why this experiment

**Local fitting is the step that makes the method "replacement".** The
pipeline scores teacher neurons, copies the selected ones into a narrower
SwiGLU (the *slice*), and then locally fits that slice to the teacher MLP's
input/output pairs before model-level recovery. Without the local fit, the
method is structured pruning followed by distillation, which is close to
Minitron. [src-minitron-2024, Section 2.2; Sections 3 and 4.3]

**Its contribution after recovery has never been measured.** SwiGLU-7's
records show that local fitting reduces block-level NMSE by a median of about
30% relative to the slice. Recovery, however, trains the same weights again.
Every SwiGLU experiment recorded the slice's NMSE before fitting, but none
evaluated an unfitted slice in the assembled model, before or after recovery.
Grafting's full self-grafting experiment compares random initialization,
local fitting alone, and local fitting plus fine-tuning, but not fine-tuning
without local fitting. Its "no local fitting" baseline is also random weights,
not a teacher slice. [src-grafting-2025, Section 3.3, Table 2]

**The current fit may not be at its practical ceiling.** SwiGLU-7's 135 local
fits show two patterns:

| Width | Behavior | Diagnosis |
| --- | --- | --- |
| 0.25 | Validation error improves slowly; median best epoch about 22, latest 47 | Optimization-limited, close to a plateau |
| 0.5 and wider | Validation error stops improving after 1-3 epochs while training error keeps falling | Overfitting |

The local-fitting learning rate (constant `1e-3`) was carried over from
random-initialization fitting and never tuned for teacher-initialized fits.
The earlier learning-rate study, SwiGLU-4, tuned *recovery*, not local
fitting. Calibration pairs come from 128-token sequences, while recovery and
use run at 8,192 tokens. If, as is common in language models, the first
token of each sequence carries very large activations, 128-token capture
over-represents those tokens in the squared-error loss. This is a hypothesis
that the 8K recipe tests; it has not been checked for SmolLM2.

## Design overview

| Part | What it does | Decides |
| --- | --- | --- |
| A. Block check | Compares local-fitting recipes on three layers and two widths | Which fitting recipe the model runs use |
| B. Model runs | Slice-only and fitted models at eight targets, recovered to 100M tokens | Whether local fitting matters after recovery; the homogeneous control spectrum |

Part B's token-0 checkpoint is the before-recovery comparison; there is no
separate stage for it.

## Part A - block-level fit check

**Scope:** layers 3, 11, and 19 (early, tolerant middle, sensitive late) at
widths 0.25 and 0.5. Layers 1, 7, and 22 are excluded because their very large
activations make them atypical.

**Stage 1 recipes**, each fitted at every layer-width combination:

| Recipe | Learning rate | Schedule | Purpose |
| --- | --- | --- | --- |
| Current, fitted twice with different seeds | `1e-3` | Constant | Reference and seed-to-seed variation |
| Lower constant rate | `3e-4`, `1e-4` | Constant | Overfitting at wider widths |
| Decay only | `1e-3` | Cosine over 64 epochs | Slow fits at 0.25 |
| Lower rate plus decay | `3e-4`, `1e-4` | Cosine over 64 epochs | Both effects together |
| Exact down-projection refit | None | None | Best possible output matrix for the chosen neurons, by ordinary least squares |
| 8K-context calibration | `1e-3` | Constant | Context length only; other settings match the current recipe |

That is nine fits per combination and 54 in total. Shared settings follow
SwiGLU-7: AdamW, batch 2,048, zero weight decay, at most 64 epochs,
early-stopping patience 3, 393,216 calibration pairs, importance-ranked
teacher-subset initialization, MSE objective. Constant and cosine variants are
tested separately so that the effect of the rate and the effect of the
schedule are not confounded.

**Stage 2**, only if stage 1 leaves the wider widths overfitting:

- a penalty keeping weights near the copied teacher slice, at a weak and a
  strong setting (the L2-SP method; Li et al., ICML 2018, to be registered
  before the thesis cites it); and
- 786,432 calibration pairs, twice the SwiGLU-7 count.

**Evaluation:** single-block KL, meaning the teacher KL when only that block
is replaced. It is measured at 128 tokens on SwiGLU-7's allocation-selection
partition and at 8,192 tokens on Quantization-1's 8K KL definition, so the
128-token measurement does not favor the 128-token fit. NMSE is reported
alongside, but it is not the selection criterion, because a few
large-magnitude tokens dominate it.

**Adoption rule:** a new recipe is used in Part B only if it reduces
single-block KL by more than 5% and by more than the seed-to-seed variation
of the current recipe, on average across the six combinations, without being
clearly worse in any. Otherwise Part B uses the current recipe.

## Part B - model runs at 100M tokens

**Arms:** slice-only (teacher slice, no local fitting) and fitted (the recipe
selected by Part A). Within a target, both arms use the same neurons and the
same widths; only local fitting differs.

**Targets:** 20, 30, 40, 50, 60, 70, 75, and 80% eligible-MLP removal. The 75%
target matches Quantization-1's nominal INT4 storage reduction; 70% and 80%
bracket it.

**Widths and allocation:**

- Widths 0.125 and 0.0625 are added to SwiGLU-7's options (0.25 to 1.0 in
  steps of 0.125). SwiGLU-only allocation cannot exceed 75% removal without
  them.
- At 20-50%, the fitted arm keeps SwiGLU-7's exact widths, so the comparison
  with SwiGLU-7 isolates fitting. The implementation checks whether
  re-solving the allocation with the extended widths would change these
  targets and reports the result.
- At 60-80%, widths come from the discrete allocator over width curves
  extended to the new widths. If Part A adopts a new recipe, all width curves
  are refitted with it so that allocation uses one recipe; otherwise
  SwiGLU-7's curves are extended with the two new widths.

**Recovery:** identical to SwiGLU-7's replacement-only recipe (S7-0). Only
replacement SwiGLUs train; layers kept at full width, attention, norms, and
embeddings stay frozen. AdamW, constant `3e-5`, zero weight decay, temperature
1 KL with no cross-entropy, 8,192-token sequences with an effective batch of
8,192 tokens, seed 21, and the same ordered C4 recovery stream starting at
shard 1. Because the learning rate is constant, a 100M-token run is
identical to the first 100M tokens of a 1B-token run.

**Checkpoints:** 0, 10, 25, 50, 75, and 100M tokens. Monitoring KL is measured
at all six and WikiText prefix perplexity at all but 75M, matching SwiGLU-7.

**Final evaluation at 100M:** the full SwiGLU-7 final evaluation (full
WikiText-2 validation and test perplexity at contexts 128, 2,048, and 8,192,
plus the five zero-shot tasks) and Quantization-1's 8K KL. The quantization
baseline and the heterogeneous comparison use these metrics.

**Interpretation, fixed in advance:**

| Slice-only versus fitted at 100M | Conclusion |
| --- | --- |
| Gap shrinks to within about 2% | Local fitting does not matter after recovery |
| Gap stays clearly larger and is not shrinking | Local fitting matters |
| Anything else | Inconclusive at 100M |

The 2% threshold is a judgment call and is reported as such. There is no
repeat run: both arms use seed 21 and the identical ordered recovery stream,
so the comparison is paired.

**Promotion:** only clear 100M results that serve as thesis headline
comparisons are promoted to 1B-token recovery.

## Implementation requirements from the design

1. **Fitted arm at 20-50% (decided 2026-10-10).** SwiGLU-7 deleted its
   intermediate checkpoints after the 1B runs, so it has no 100M weights on
   which to run the full final evaluation. The fitted arm is therefore run to
   100M at all eight targets. At 20-50% these runs must reproduce SwiGLU-7's
   first 100M of monitoring values when the current recipe is kept, which
   serves as a reproduction check. If Part A adopts a new recipe, they are new
   runs anyway.
2. **8K data overlap.** Quantization-1's 8K KL uses C4 train shard 0, which is
   also SwiGLU-7's calibration source. The 8K calibration sequences in Part A
   must be drawn disjointly from the 8K KL evaluation sequences.
3. **Shared neuron selection (decided 2026-10-10).** SwiGLU-7's artifacts
   record widths but not which neurons were selected. SwiGLU-8 therefore
   selects neurons once per target with SwiGLU-7's importance score, saves the
   selection in its own artifacts, and builds both arms from it. Equality with
   SwiGLU-7's selection is checked indirectly by the reproduction check below.
4. **Shared 8K KL code (decided 2026-10-10).** Quantization-1's 8K KL
   evaluation (`teacher-kl-t1-all-valid-positions-c4-shard0-24x8192-v1`) moves
   from `workflows/runs/model/baseline/quantization.py` into shared code in
   `src/`, used by both experiments, with identical behavior. The move happens
   before either experiment runs.

## Acceptance checks

These are part of the real runs, not separate smoke runs. Tolerances are
stated in the implementation plan.

1. **Part A:** the current recipe at layers 3, 11, and 19 reproduces the
   local-fit error SwiGLU-7 recorded for the same layer and width.
2. **Part B:** while the current recipe is kept, the fitted runs at 20-50%
   reproduce SwiGLU-7's monitoring values over the first 100M tokens.

If either check fails, work stops for investigation before continuing.

## Implementation boundaries

- All parts run on PERUN.
- Run SwiGLU-8 before the planned refactor changes `src/`, so its source
  fingerprints and SwiGLU-7 reproduction are not affected by unrelated
  changes.
- Compute uses a headless runner in `workflows/runs/model/swiglu/` with a
  configuration in `workflows/configs/model/swiglu/` and a PERUN job script.
  Analysis, including Part A plots, uses a load-only
  `notebooks/model/swiglu/swiglu-8.ipynb`.
- No smoke or reduced-budget runs.

## Resources

Rough estimates from SwiGLU-7's workflow timings; not a resource plan and
not billing:

| Item | Estimated H200-hours |
| --- | ---: |
| Part A, stage 1 (54 fits plus evaluation) | 2-4 |
| Part A, stage 2, if needed | 1-2 |
| Width curves extended to 0.125 and 0.0625 (full refit if a new recipe is adopted) | 2-6 |
| Part B, 16 runs to 100M (two arms at eight targets) at about 1.2-1.5 hours each, plus about 15 minutes of final evaluation per run | 23-28 |
| **Total** | **about 28-40** |

The formal pre-run plan required by
[PERUN resource accounting](../../../infrastructure/perun/perun-resources.md)
is completed during implementation planning, before any submission, and
compared with measured use afterward.

## Sources and attribution

- `src-grafting-2025`, Section 3.3, Table 2: local fitting followed by
  fine-tuning; no fine-tuning-only condition.
- `src-minitron-2024`, Section 2.2; Sections 3 and 4.3: width pruning followed
  by distillation, without local fitting.
- L2-SP (Li et al., ICML 2018) and cosine learning-rate decay (Loshchilov and
  Hutter, 2017) are standard published methods. They are not yet registered;
  register them before the thesis cites them.
- The exact down-projection refit is ordinary least squares and needs no
  citation. Its relation to MoDeGPT's down-projection reconstruction
  [src-modegpt-2025, Sections 3.1-3.2] is noted when it is reported.
- The fitting observations come from
  `data/results/workflows/model/swiglu-7/prepare-001/result.json`, checked
  2026-10-10.
