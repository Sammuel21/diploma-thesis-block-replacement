---
title: Deep Analysis 2026-11-12 - Implementation Review
status: draft
created: 2026-11-12
modified: 2026-11-12
---

# 4. Implementation: correctness, leanness, structure

Part of the [deep analysis of 2026-11-12](README.md). It covers your topics
1.2 and 2.1-2.3. Evidence labels are defined in
[file 1](01-direction-and-design.md).

## Short answer

- **Correctness: I found no error in the code I read.** That is about 3,900
  lines: the allocator, both quantizers, the retraining loops, the operators
  and their fitting, the KL and perplexity evaluation, model surgery, stream
  building, and the first half of the newest runner. The stronger evidence
  is not my reading; it is that your runs reproduce each other bit for bit
  before they start.
- **Leanness: the scientific core is lean; the bloat is in three specific
  places.** Copied launcher boilerplate (about 1,200 lines), three notebook
  ports that have never run (4,293 lines), and two generations of duplicated
  retraining code. There is almost no dead code in `src/`.
- **Structure: one real problem.** The shared helpers of the project live
  inside finished experiment files. `recovery_2.py` imports from six other
  experiments. Every new experiment makes an old one harder to leave alone.
- **What I did not read:** the large runners (`swiglu_7/8/9`, `stacking_1`,
  `quantization`, `heterogeneous_1/2`), `teacher_cache.py`, `data.py`,
  `capture.py`, `config.py`, the notebooks, and the SwiGLU-2 to 6 runners.
  I make no correctness claim about them.

## 4.1 What was done

| Check | How | Label |
| --- | --- | --- |
| Read core modules in full | `allocation.py`, `quantization.py`, `recovery.py`, `operators/modules.py`, `operators/training.py`, `evaluation/teacher_kl.py`, `evaluation/final_quality.py`, `compression/selection.py`, `compression/surgery.py`, `data_streams.py`; `recovery_2.py` lines 1-330; one launcher; the check helpers in `swiglu_8.py` | **[read]** |
| Structural scan | A script parsed every Python file under `src/`, `workflows/`, `tests/`: function-body duplicates, same-name near-duplicates, names in `src/` with no user, shared lines across launchers, imports between runners | **[ran]** |
| Test suite | `python -m unittest discover -s tests` on this machine: **237 tests, all pass, 10 skipped, 11 seconds**. Run with local Python 3.12 and PyTorch 2.11, which is not the pinned PERUN environment | **[ran]** |

The script is kept outside git with the session material
([file 6](06-session-findings.md), "Where the raw material is").

## 4.2 Correctness

### What I checked and found right

| Piece | What I checked | Result |
| --- | --- | --- |
| Distillation loss | Direction and scaling of the KL | Teacher-to-student KL, averaged per token, times T². Correct |
| Gradient accumulation | Weighting of micro-batches | Each is weighted by its token share of the update. Correct |
| Learning-rate schedule | The 1-sqrt decay and where the position is read | Uses the token count after the update; constant until the decay start. Matches the description |
| Knapsack | Exactness of the dynamic program | Keeps the best partial solution per used budget, exact integer units, deterministic ties. Returns the optimum of its objective |
| Isotonic smoothing | Pool-adjacent-violators for a non-increasing curve | Correct |
| Boundary width | Budget safety | Re-checked after the step; raises if exceeded |
| Neuron importance | The score | RMS activation of the neuron times the norm of its output column: the size of that neuron's contribution to the output. Sensible and correctly accumulated in double precision |
| Least-squares refit | Normal equations | Accumulated in FP64, solved by Cholesky, fails loudly if singular |
| INT3 / INT4 training-time quantizers | Agreement with the deployed weights | Enforced bitwise after preparation and again at conversion |
| Zero-neuron padding | Function unchanged | A padded neuron has zero gate, up, and down weights; a gradient mask keeps it zero |
| Byte accounting | Shared storage | Each physical allocation is counted once, so tied embeddings are not counted twice |
| Rolling perplexity | Coverage | Every target is scored exactly once; the function refuses to return otherwise |
| Token streams | Repetition and identity | Refuses to repeat documents; every stream is identified by sha256 |

### Small things worth knowing

None of these has affected a result that I can see. They are listed so that
they are known, not because they need fixing now.

| # | Where | Observation | Consequence |
| --- | --- | --- | --- |
| C1 | `operators/training.py`, both fit functions | Early stopping counts from the *starting* weights. A fit that starts well and gets slightly worse for three epochs returns its start | This is what stalled linear + plain MLP in Heterogeneous-1 (88 of 252 fits). It is documented there. Any future operator with a strong closed-form start will hit it |
| C2 | `swiglu_8.py`, `relative_difference` | Divides by the reference. A reference of exactly zero raises an error instead of failing the check cleanly | Not reachable with today's references (KL values are non-zero; the teacher self-check uses an absolute bound) |
| C3 | `allocation.py` | Since Heterogeneous-1 a "SwiGLU width curve point" can hold any operator, and a layer's menu is sorted by width while the tie-break compares widths only | Two different operators with equal width and exactly equal cost would tie without a rule. Practically impossible with measured costs. The names now mislead |
| C4 | `data_streams.py`, `build_verified_stream` | Unlike its sibling, it leaves its temporary file behind on failure | Housekeeping only |
| C5 | `evaluation/final_quality.py` | A paired bootstrap for task accuracy exists and per-example outputs are stored, but the snapshots compare macro accuracy between models with "no interval" | See 4.2.1 |

### 4.2.1 A reporting point that the code already supports

The 2026-11-01 snapshot decides which models are "on the macro frontier" by
differences as small as 0.2 points (for example A 50% INT3 1B at 56.59%
against B 50% INT3 1B at 56.39%). With these task sizes, a difference of
that size is well inside example-sampling error.

An illustrative calculation **[calc; the 15% disagreement rate between two
models is my assumption, not a measurement]**: with about 1,170 to 10,040
examples per task, the paired standard error of a five-task macro difference
is around 0.4 points, so differences under about 0.8 points are not
resolved.

Your 2026-11-02 row says no bootstrap intervals are planned, because the
main effects are far above noise. That is true for KL. For the macro
frontier it is not. The cheap fix needs no GPU: either treat macro
differences under one point as ties when marking the frontier, or run the
existing paired bootstrap between pairs of models on the stored samples.
Each run records its per-example files by path and sha256; they are not in
the local copy of the results, so the bootstrap would run where they are.
This is your decision; I raise it because it is new evidence relative to
that row.

### Why the in-run checks matter more than this review

Every run since SwiGLU-8 stops unless it reproduces a recorded value from an
earlier run within 1e-4, on the same data, before it trains. Recovery-2's
three starts matched their recorded runs with a relative difference of
exactly zero. A wrong loss, a wrong stream, a wrong start, or a wrong
quantizer would break that chain. No reading of mine is as strong as that.
It is the best engineering decision in the repository.

The limit of that protection: it guards *consistency with the past*, not
*correctness of the first run*. The INT3 quantizer was wrong from its first
use and no reproduction check could see it; the bitwise deployed-weight
check, added afterwards, is what closes that hole.

## 4.3 Leanness

### The numbers [ran]

| Area | Python lines | Note |
| --- | ---: | --- |
| `src/mlp_replacement/` | 7,466 | The reusable science |
| `workflows/runs/` | 24,462 | Experiment runners |
| `tests/` | 4,922 | 237 tests |
| Launchers (`.sbatch`, `.sh`) | about 2,100 code lines | 12 PERUN, 3 local |

| Finding | Size |
| --- | --- |
| Function bodies duplicated exactly across files | 7 groups, 69 lines |
| Same-name functions that are 60-96% similar | 18 pairs, about 650 overlapping lines |
| The three `train` functions in `swiglu_7/8/9` | about 380 lines each, 54-67% similar |
| Names in `src/` with no user anywhere | 66 lines (5 small functions) |
| Launcher lines that also appear in three or more other launchers | about 1,230 of 1,930 PERUN code lines (64%); 77-82% in the three newest |
| Notebook ports that have never produced a result | 4,293 lines in three files |

### Reading the numbers

**`src/` is clean.** After the 2026-10-15 cleanup, almost everything in it
is used. The five unused names are trivial (`convert_int4_qat`,
`prepare_int4_qat` outside tests, `cosine_distance_score`,
`fit_layer_replacement`, `experiment_config_from_dict`).

**Exact duplication is negligible.** 69 lines in 24,000. The agents have
been reusing code, not copying it.

**The real bloat, in order of size:**

1. **The three never-run ports (4,293 lines).** `swiglu_initial.py`,
   `swiglu_2_allocation.py`, and `baseline/compression.py` are 17.5% of the
   runner code. The 2026-10-10 row keeps them, "each marked 'never run; the
   notebook is authoritative'". **I found no such marking** in the three
   files or in the `workflows` READMEs (searched for "never run",
   "authoritative", "not run"); their docstrings say only "Headless
   migration of ...". The 2026-10-15 row deletes "code and configuration
   that never produced a result and has no caller". The two rows pull in
   different directions for these files. A reader of the appendix cannot
   tell that a sixth of the runner code has never executed. Recommendation:
   either add the marking the row requires, or delete them and let git keep
   them. I would delete: untested code that looks maintained is a liability
   in a repository an opponent will read.
2. **Launcher boilerplate (about 1,230 lines).** This is the redundancy you
   asked about on 2026-10-20. Each launcher repeats the same blocks:
   environment checks, PROJECT and scratch resolution, lock directory,
   stage-in of code, the stage-out trap with byte verification, signal
   handling. Only about 30-40 lines per launcher are specific to the
   experiment (name, array mapping, pinned inputs, the Python call).
3. **Two generations of retraining code in `recovery.py` (about 300 lines).**
   The KL computation appears three times, and `recover_replacements_by_tokens`
   is a near-copy of the newer `recover_trainable_by_tokens` restricted to
   one parameter group and a constant rate.
4. **`recovery_2.py` against `recovery_1.py`.** Ten functions are 60-100%
   similar. The second experiment was made by copying the first and
   importing twelve names from it.

### What to do about each

| Bloat | Recommendation | Risk |
| --- | --- | --- |
| Never-run ports | Mark or delete (see above) | None to results; they produced none |
| Launchers | **Do not touch the launchers of finished experiments.** For new ones, put the common blocks in one sourced shell file and keep each launcher to its specific 30-40 lines | Low, because old launchers stay as evidence. A launcher change is never "mechanical" (your own rule), so the first new launcher needs the same care as any |
| Retraining loops | Make the old loop a thin call into the new one, *only if* a finished experiment that uses it is ever re-run and reproduces exactly. Otherwise leave it | Medium: optimizer construction differs slightly; bit-identity would have to be shown |
| Recovery-1/2 overlap | Leave. Both are finished | None |

This follows the intent you stated on 2026-10-20: do not archive, because
experiments must stay reproducible; parametrize instead. The honest scope of
"parametrize" is new code. Rewriting finished runners to share more code
would put their reproducibility at risk for a cosmetic gain.

**A trend worth praising.** The newest runners are the smallest:
Heterogeneous-2 is 568 lines and Recovery-2 is 634, against 2,093 to 2,418
for SwiGLU-7 and 8. The "minimal viable implementation" rule of 2026-11-04
is visible in the code. Heterogeneous-2 having no save, load, or resume code
because a run takes under two hours is exactly the right call.

## 4.4 Structure

### The directory layout is good

`src/` for reusable science, `workflows/configs|runs|jobs` by experiment
class, notebooks as load-only views, tests as contracts. The split by
`model/` and `block/` and then by class matches the docs tree. An outsider
can find things. I would not change it.

One wart: the class folder is spelled `heterogenous` while the experiments
inside are `heterogeneous_1`. Your sessions show this already caused a
failed submission **[read: L6]**. Renaming touches module paths that are
recorded in result artifacts, so it is not free. Decide it once: rename at
the final cleanup before submission, or leave it and say so in the class
README.

### The one structural problem: experiments used as libraries

`recovery_2.py` imports **[ran]**:

| From | Names |
| --- | --- |
| `swiglu_7.py` | `code_hashes`, `cpu_tree`, `group_metadata`, `load_trainable_state`, `output_path`, `persist`, `resolve_storage`, `trainable_state`, `utc_now` |
| `swiglu_8.py` | `comparison`, `enforce`, `equality_check`, `is_digest`, `pinned_json`, `record_checks`, `record_failure`, `release_memory`, `verified_stream` |
| `swiglu_9.py` | `finished_outputs`, `load_swiglu8_prepared`, `write_once` |
| `recovery_1.py` | twelve names |
| `stacking_1.py`, `baseline/quantization.py`, `swiglu/shared.py` | used as modules |

Across the tree, other runners import 22 names from `swiglu_7.py`, 31 from
`swiglu_8.py`, and 13 from `swiglu5/fitting.py`. Most of them are not about
SwiGLU at all. `enforce`, `comparison`, `pinned_json`, `persist`, and
`utc_now` are the project's general tools for checks, pinned inputs, and
run records. They are about 10 lines each and they sit inside files of
2,000 lines.

The project now has three generations of shared code: `common.py` (the
first), `swiglu/shared.py` (the second), and "whatever the last SwiGLU
runner defined" (the third). Some helpers exist several times: a UTC
timestamp in three places, `persist` in four, an atomic JSON writer in
three, `storage_preflight` and `code_hashes` twice each.

**Why this matters:**

- **Finished experiments are no longer frozen.** Your rule says their code
  "must keep producing the same results". But `swiglu_8.py` is now a
  dependency of five later experiments, so it will keep being opened.
- **Reading cost.** To follow `recovery_2.py`, a reader, or an agent paying
  tokens, must open seven files. This is a direct cause of long, expensive
  sessions.
- **The fingerprint grows.** Each run records hashes of the code it used;
  that set now spans most of the tree.

**What I would do, and when.** Move the generic helpers (about 35 small
functions) into neutral modules beside `common.py`, for example one for
checks and pins and one for run records, and have the old files import them
back so nothing else changes. This is a pure move. Do it at the next natural
boundary, before the first runner of a new class (Stacking-2,
Quantization-3, or a mixture-of-experts runner), not in the middle of
anything. It is the one refactor I think pays for itself, because every
future experiment will otherwise deepen the chain.

**What I would not do.** Build a general "experiment framework" with stages
and registries. The 2026-10-17 row describes stacking variants as an
"ordered stage list", and you asked on 2026-11-01 whether a framework could
replace the code; the answer you got ("mostly no") was right. Twelve
experiments with different questions do not share enough shape.

## 4.5 Tests

**What they are.** 237 tests in 19 files, all named `*_contracts.py`. Two
kinds:

- **Behaviour on toy models** (for example a toy three-family schedule
  screen that trains for 40 tokens). These catch logic errors cheaply and
  are worth their lines.
- **Text checks on files** (for example asserting that a launcher contains
  `#SBATCH --time=30:00:00` or that a config pins a given sha256). These
  freeze decided constants. They are useful as a tripwire against an agent
  silently changing a decided value, which is a real risk in this project.
  They are also brittle and they test nothing about behaviour.

**Assessment [view].** The suite fits its purpose: it guards decisions, not
science. It runs in 11 seconds, so it costs nothing to keep. Two cautions:

- A green suite says "the decided constants are still in place and the toy
  paths run". It does not say a new runner is correct. The in-run gates do
  that. Agents should not report "tests pass" as evidence about results.
- 4,900 lines of tests for 24,000 lines of runners is a lot of text for
  agents to keep consistent. When an experiment is finished and its
  launcher will never change, its text checks have done their job.

## 4.6 Efficiency

Short, because I found little to gain **[read: measured-use sections]**:

- **Throughput** is 74-97M tokens per GPU-hour on one H200 and repeats
  within 1.5% between experiments. The teacher runs online at every step,
  which is the right choice: caching its outputs for 1B tokens over a
  49,152-word vocabulary is not feasible.
- **Design-level savings are already taken:** trunk-and-branch schedule
  screens, reuse of 1B models across experiments, no checkpoints where runs
  are short.
- **Memory requests** are generous (128 GB requested, 10-27 GiB used in
  Recovery-2). Harmless on PERUN.
- **The expensive thing is not compute.** By your own audit of 2026-10-15,
  the cost is session length and agent tokens. Structure (4.4) and
  documentation ([file 5](05-documentation-and-agents.md)) are where
  efficiency is to be found.

## 4.7 Summary table

| Topic | Verdict | Most useful action |
| --- | --- | --- |
| 2.1 Correctness | No error found in what I read; the reproduction gates are strong evidence for the rest | Keep the gates; treat macro differences under one point as ties |
| 2.2 Leanness | Core lean; three concentrated sources of bloat | Decide the never-run ports; share launcher blocks for new launchers only |
| 2.3 Structure | Good layout; helpers trapped in experiment files | One pure move of generic helpers before the next experiment class |
