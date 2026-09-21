---
metadata_version: 1
title: Homogeneous SwiGLU Experimental Progression
type: experiment-synthesis
category: experiments/model/swiglu
status: active
created: 2026-09-20
modified: 2026-09-21
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
sources:
  notebooks:
    - notebooks/model/swiglu/swiglu.ipynb
    - notebooks/model/swiglu/swiglu-2.ipynb
    - notebooks/model/swiglu/swiglu-3.ipynb
    - notebooks/model/swiglu/swiglu-4.ipynb
    - notebooks/model/swiglu/swiglu-5.ipynb
  artifacts:
    - data/results/notebook-model-study/swiglu-compression.json
    - data/results/notebook-model-study/swiglu-compression-optimized.json
    - data/results/notebook-model-study/swiglu-2.json
    - data/results/workflows/model/swiglu-3/run-001.json
    - data/results/workflows/model/swiglu-4/run-001.json
    - data/results/workflows/model/swiglu-5/search/run-001.json
    - data/results/workflows/model/swiglu-5/confirmation/run-001-target-0.2.json
    - data/results/workflows/model/swiglu-5/confirmation/run-001-target-0.5.json
---

# Homogeneous SwiGLU Experimental Progression

## Purpose and evidence boundary

This page is the chronological synthesis of the homogeneous SwiGLU experiment
family. It explains what each experiment changed, what was observed, and why
the next experiment was designed. For the outcomes grouped by finding rather
than timeline, see the [results overview](swiglu-results.md). The individual
workflow pages remain the authoritative method descriptions:

- [SwiGLU](swiglu.md), referred to here as **SwiGLU-1**;
- [SwiGLU-2](swiglu-2.md);
- [SwiGLU-3](swiglu-3.md);
- [SwiGLU-4](swiglu-4.md); and
- [SwiGLU-5](swiglu-5.md).

All five experiments use the pinned `HuggingFaceTB/SmolLM2-1.7B` model. Layers
0 and 23 are protected, while layers 1 through 22 are eligible for homogeneous
SwiGLU replacement. The dense WikiText-2 validation reference is loss 2.66997
and perplexity 14.4396.

Perplexity is the safest metric for following the family across generations
because the same WikiText validation definition is retained. Early `teacher_kl`
rows and the later fixed recovery-validation KL are evaluated on different
partitions and must not be treated as one continuous series. KL comparisons in
this document are therefore made only when the artifacts use the same declared
evaluation split.

## Executive conclusion

The experimental progression moved through five distinct bottlenecks:

```text
feasible local replacement
  -> informative layer allocation
  -> sufficient global-recovery budget
  -> efficient recovery configuration
  -> discrete, globally optimized layer widths
```

The final evidence supports four main conclusions:

1. **Teacher-derived initialization made homogeneous replacement viable.**
   Increasing calibration data helped, but random initialization still left a
   severely damaged assembled model.
2. **Model-level sensitivity is more useful than generic block influence for
   distributing a fixed parameter budget.** SwiGLU-2's singleton-KL policy
   improved the original 50% result at the same total parameter count.
3. **Global recovery helps substantially but does not erase structural
   compression loss.** More tokens continued to improve perplexity, especially
   at 50% eligible-MLP removal, but neither 100M-token trajectory reached dense
   quality or KL zero.
4. **SwiGLU-5's decisive methodological improvement is discrete width
   allocation, not output reconstruction, RMSNorm, or LoRA.** It preserved
   some sensitive MLPs at full width and concentrated compression in tolerant
   layers. This improved both short-budget and 100M-token results.

At 100M tokens, the confirmed discrete-allocation model improves on SwiGLU-3:

| Eligible-MLP removal | Whole-model removal | SwiGLU-3 PPL | SwiGLU-5 PPL | PPL reduction | SwiGLU-3 KL | SwiGLU-5 KL | KL reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20% | 12.94% | 16.6543 | **15.9491** | 0.7052 (4.23%) | 0.075510 | **0.074664** | 0.000846 (1.12%) |
| 50% | 32.35% | 22.0908 | **20.5286** | 1.5621 (7.07%) | 0.220696 | **0.194910** | 0.025786 (11.68%) |

Relative to the dense PPL reference, these changes close 31.84% of the
remaining SwiGLU-3 PPL gap at 20% removal and 20.42% at 50% removal. The models
are materially better, but the remaining gaps of 1.5095 and 6.0891 PPL show
that this is not dense-equivalent recovery.

## Chronological overview

| Experiment | Question | Principal change | Main finding | Handoff |
| --- | --- | --- | --- | --- |
| SwiGLU-1 | Can 22 dense MLPs be replaced by narrower SwiGLUs at a fixed global budget? | Larger calibration and teacher-neuron initialization | Initialization dominated the first quality improvement; BI allocation did not beat uniform | Search for a model-level allocation signal |
| SwiGLU-2 | Which score should determine per-layer widths? | Singleton model damage, temperature sweep, minimum-width ablation | Aggressive-width singleton KL selected the best 50% allocation | Test calibration scale, sparsity, and long recovery |
| SwiGLU-3 | How do sparsity and recovery tokens affect quality? | 393,216-pair local fits and four 100M-token trajectories | Recovery remained useful through long budgets, but was expensive and left a dense gap | Improve the recovery recipe before changing structure again |
| SwiGLU-4 | Which optimizer, objective, and trainable scope recover best? | LR/schedule/objective, RMSNorm, and LoRA tournament | Constant `3e-5`, T=1 KL, replacement-only recovery was the best efficient recipe | Hold recovery fixed and improve model construction |
| SwiGLU-5 | Can initialization, allocation, or composition improve the quality frontier? | Output reconstruction, discrete width curves, composition-aware refit, efficient search and confirmation | Discrete allocation won; confirmed gains persisted to 100M | Use discrete allocation as the homogeneous baseline and move future novelty to heterogeneous operators or stronger allocation models |

## SwiGLU-1: establishing feasibility

### Question and design

The first workflow replaced every eligible MLP with a reduced-width SwiGLU and
compared uniform, canonical block-influence, and residual-aware block-influence
allocations at 50% eligible-MLP removal. It decomposed the fitting improvement
into:

1. the historical 12,288-pair random initialization;
2. 98,304 pairs with random initialization; and
3. 98,304 pairs with importance-ranked teacher-neuron initialization.

### Observed progression

| 50% configuration | Post-recovery PPL | Post-recovery teacher KL |
| --- | ---: | ---: |
| Historical random, uniform widths | 46,023.14 | 8.0831 |
| More data only, best policy | 136.03 | 2.0954 |
| Complete method, uniform widths | **32.6286** | **0.7436** |
| Complete method, canonical BI | 36.1295 | 0.8296 |
| Complete method, residual-aware BI | 36.3267 | 0.8322 |

The scale of the change establishes that teacher-derived initialization, not
the initial BI allocator, made simultaneous model-wide replacement viable.
More calibration data alone improved the random model dramatically but did not
bring it into a useful range. With the complete fitting method, uniform widths
were better than either BI-derived allocation.

### Why SwiGLU-2 followed

Block influence measures how much a block affects the uncompressed model. It
does not directly measure how much damage occurs when that MLP is replaced at
a particular width. The result therefore motivated allocation scores based on
the replacement operation itself.

## SwiGLU-2: learning where parameters matter

### Question and design

SwiGLU-2 fitted every eligible block at 25% and 50% retained width and measured
the resulting singleton model damage. It screened 19 score/temperature
policies, tested minimum-width floors, promoted several families, fully fitted
the finalists, and recovered every finalist independently.

The winning `singleton_kl_w25_t1` policy used singleton teacher KL at the
aggressive 25%-width probe. Its layer widths ranged from 21.20% to 71.02% while
retaining the same aggregate 50% eligible-MLP budget as the uniform control.

### Result

| 50% policy | Pre-recovery PPL | Post-recovery PPL | Pre teacher KL | Post teacher KL |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 32.7345 | 32.6286 | 0.7469 | 0.7436 |
| Singleton KL at 25%, T=1 | **29.7065** | **29.5973** | **0.6522** | **0.6493** |

At the same parameter count, the winning policy reduced post-recovery PPL by
3.0313 (9.29%) and teacher KL by 0.0943 (12.68%) relative to uniform. The
one-epoch recovery itself changed the winning PPL by only 0.1092, showing that
the allocation and local construction carried most of this experiment's gain.

### Why SwiGLU-3 followed

SwiGLU-2 established a better allocation but did not answer whether local
fitting had saturated or whether substantially more global-recovery tokens
could close the remaining gap. SwiGLU-3 retained the winning policy and varied
calibration size, compression, and recovery length.

## SwiGLU-3: calibration, sparsity, and long recovery

### Calibration result

The 50% allocation was refitted with nested calibration budgets. Increasing
the budget improved both the allocation-selection KL and WikiText PPL:

| Calibration pairs per block | Allocation-selection KL | Pre-recovery PPL |
| ---: | ---: | ---: |
| 98,304 | 0.402510 | 29.6812 |
| 196,608 | 0.388924 | 29.2105 |
| 393,216 | **0.373938** | **28.9146** |

The workflow therefore selected 393,216 pairs. The gain was real but much
smaller than the original teacher-initialization gain, and the largest budget
also multiplied local-fitting cost.

### Sparsity and token result

Each compression target then received an independent 100M-token trajectory at
constant `1e-5`, pure T=1 KL, and 2,048 effective tokens per optimizer update.

| Eligible-MLP removal | Whole-model removal | Pre-recovery PPL | 10M PPL | 100M PPL | 10M fixed KL | 100M fixed KL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20% | 12.94% | 17.6876 | 17.1226 | **16.6543** | 0.093124 | **0.075510** |
| 30% | 19.41% | 20.2696 | 19.1386 | **18.0415** | 0.148346 | **0.117218** |
| 40% | 25.88% | 23.6724 | 21.3715 | **19.5646** | 0.208286 | **0.162198** |
| 50% | 32.35% | 28.9146 | 24.6861 | **22.0908** | 0.286172 | **0.220696** |

Increasing tokens improved every compression target. The absolute recovery
gain was larger at aggressive compression, where there was more damage to
repair, but even 100M tokens did not remove the dense gap. The fixed validation
KL reached its best recorded value near 95M rather than exactly at 100M for all
four targets, while endpoint PPL could still improve. This established that
validation KL and downstream PPL are related but not interchangeable selection
signals.

### Compute consequence

The four recoveries consumed 400M candidate-token positions. The artifact
records 156,709 seconds for recovery and approximately 44.9 hours for the full
calibration, sparsity, and recovery pipeline. Long recovery was effective, but
it was too expensive to use as the first-stage evaluator of many new methods.

### Why SwiGLU-4 followed

Before constructing new compressed models, the project needed to determine
whether the SwiGLU-3 recovery recipe itself was leaving inexpensive quality on
the table. SwiGLU-4 held the 50% starting state fixed and changed only global
recovery choices.

## SwiGLU-4: optimizing the recovery recipe

### Equal-budget 10M comparison

| Strategy | Fixed KL | PPL | Trainable parameters | Runtime |
| --- | ---: | ---: | ---: | ---: |
| SwiGLU-3 / reproduced control, constant `1e-5` | 0.286172 | 24.6861 | 553.65M | 3,882 s for reproduced control |
| Constant `3e-5`, T=1 KL | **0.266991** | **24.0373** | 553.65M | 3,845 s |
| Replacements plus RMSNorm | 0.266596 | 24.0115 | 553.69M | 3,930 s |
| Replacement-only MLP LoRA | 0.309782 | 25.5665 | 6.49M | 4,710 s |
| Transformer-assisted LoRA | 0.265047 | 23.9864 | 560.92M | 6,076 s |

The constant `3e-5` configuration produced the meaningful efficient gain:
0.01918 lower KL and 0.6488 lower PPL than the exact `1e-5` control. RMSNorm
improved PPL by only another 0.0259. Replacement-only LoRA was worse, while
transformer-assisted LoRA improved PPL by just 0.0509 over replacement-only
`3e-5` recovery despite training more parameters and taking substantially
longer.

### Decision carried into SwiGLU-5

The project adopted constant `3e-5`, pure teacher-to-student KL at T=1, zero
weight decay, and replacement-only AdamW. RMSNorm and LoRA were excluded from
the next search. This preserved the useful recovery gain without changing the
thesis question into broad full-Transformer adaptation.

## SwiGLU-5: improving the compressed structure

### Search questions

SwiGLU-5 held the recovery recipe fixed and compared:

- the exact SwiGLU-3 construction under the stronger recovery recipe;
- output-reconstructed initialization at the legacy widths;
- discrete per-layer width allocation with legacy initialization;
- output reconstruction combined with discrete allocation; and
- one composition-aware refit on compressed-model inputs.

Every candidate reached 2M tokens. The control and the two strongest
challengers at each target continued to 5M, after which one endpoint was
retained for confirmation.

### What changed before recovery

Discrete allocation improved the assembled model before any global recovery:

| Target | Construction | Pre-recovery fixed KL | Pre-recovery PPL |
| ---: | --- | ---: | ---: |
| 20% | Legacy ranked widths | 0.123033 | 17.6876 |
| 20% | Discrete widths | **0.110401** | **16.9060** |
| 50% | Legacy ranked widths | 0.407479 | 28.9146 |
| 50% | Discrete widths | **0.397470** | **28.1810** |

Unlike the continuous ranked allocation, the discrete solver can retain the
original dense MLP at a 100% anchor. At 20% removal it kept 11 of the 22
eligible MLPs dense; at 50% it kept layers 21 and 22 dense. It compressed the
remaining layers more aggressively while preserving the same aggregate
parameter budget. Consequently only 332.19M parameters were trainable in the
20% replacements and 452.98M in the 50% replacements, although the retained
eligible-MLP totals remained 885.84M and 553.65M respectively. Dense retained
layers stayed frozen during replacement-only recovery.

This result is the clearest evidence that the earlier smooth ranked allocation
was the principal structural bottleneck: sensitive layers benefit from being
left exact, while tolerant layers can absorb disproportionately more removal.

### True 5M finalist results

| Target | Candidate | Fixed KL at 5M | PPL at 5M |
| ---: | --- | ---: | ---: |
| 20% | Legacy allocation control | 0.096449 | 17.2055 |
| 20% | Discrete allocation | 0.091872 | 16.4044 |
| 20% | Reconstruction plus discrete allocation | **0.091744** | **16.3751** |
| 50% | Legacy allocation control | 0.284632 | 24.4594 |
| 50% | Discrete allocation | **0.281350** | **23.3052** |
| 50% | Reconstruction plus discrete allocation | 0.282249 | 23.3200 |

Output reconstruction alone did not qualify for 5M continuation. When combined
with discrete allocation, it was only marginally different: slightly better at
20% and slightly worse at 50%. The evidence therefore does not support adding
its bias and two-phase fitting complexity to the default homogeneous method.

Composition-aware refinement also did not win the fixed-KL criterion. At 50%
and 2M it had worse KL than discrete allocation (0.305023 versus 0.300836) but
slightly better PPL (24.0432 versus 24.1282). This is an interesting objective
trade-off, not evidence of a confirmed long-budget improvement, because the
candidate did not continue to 5M or 100M.

### Historical 20% selection caveat

The completed search artifact has a milestone-labeling defect: the 2M full
evaluation was also tagged with the future 5M request, so historical winner
selection consumed the 2M row. It selected discrete allocation at both targets.
The genuine 5M evidence instead makes the combined strategy the 20% winner by
0.000128 KL and 0.0292 PPL. The 50% selection is unchanged.

Both completed confirmations therefore continue discrete allocation. The 20%
trajectory is a valid confirmed near-runner-up, not confirmation of the exact
5M winner. The current runner and notebook correct the lookup semantics, while
the historical artifacts remain unchanged. Rebuilding the pruned 20% combined
candidate merely to chase this small search margin is not necessary for the
main conclusion that discrete allocation improved the family.

### Confirmation against SwiGLU-3

| Target | Budget | SwiGLU-3 KL | SwiGLU-5 KL | KL change | SwiGLU-3 PPL | SwiGLU-5 PPL | PPL change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20% | 10M | 0.093124 | **0.086453** | -0.006671 | 17.1226 | **16.2993** | -0.8233 |
| 20% | 100M | 0.075510 | **0.074664** | -0.000846 | 16.6543 | **15.9491** | -0.7052 |
| 50% | 10M | 0.286172 | **0.260746** | -0.025427 | 24.6861 | **23.0215** | -1.6646 |
| 50% | 100M | 0.220696 | **0.194910** | -0.025786 | 22.0908 | **20.5286** | -1.5621 |

The 20% KL advantage narrows substantially by 100M, but its PPL advantage
persists. The 20% trajectory reaches its best fixed validation KL at 75M
(0.073419), then fluctuates, while PPL improves from 16.0834 at 50M to 15.9491
at 100M. This target is approaching a recovery plateau, but the two metrics do
not identify exactly the same stopping point.

The 50% trajectory still benefits materially from the second half of recovery:
PPL falls from 21.2478 at 50M to 20.5286 at 100M. Its best fixed KL occurs at
95M (0.193434), close to the final 0.194910. More recovery remains more valuable
at aggressive compression, although returns are diminishing.

The direct SwiGLU-3 comparison combines the new allocation with the `3e-5`
recovery and faster execution geometry. It must not attribute the complete
100M difference to allocation alone. The search-time control isolates
allocation under the same SwiGLU-5 recovery recipe, while SwiGLU-4 separately
isolates the learning-rate gain at 10M.

### Efficiency result

The system optimizations changed wall time without changing the 100M-token or
48,829-update scientific budgets:

| Target | SwiGLU-3 elapsed | SwiGLU-5 elapsed | Recorded speedup | SwiGLU-3 throughput | SwiGLU-5 throughput |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20% | 11.42 h | 2.90 h | 3.93x | 2,433 tok/s | 9,574 tok/s |
| 50% | 10.44 h | 2.99 h | 3.49x | 2,662 tok/s | 9,291 tok/s |

The confirmations selected 8 sequences with 2 accumulation steps instead of
SwiGLU-3's 2 sequences with 8 accumulation steps, used fused CUDA AdamW, and
retained 2,048 tokens per update. Peak RAM fell from 37.92 GiB to 9.46 GiB at
20% and 11.42 GiB at 50%. Peak VRAM fell at 20% but rose slightly at 50%, from
14.30 to 14.71 GiB, while remaining within the configured envelope.

The search itself did not satisfy the original six-hour objective. Recorded
stage time totals approximately 6.97 hours after the explicit over-budget
override. Width-curve fitting consumed 4.84 hours, candidate assembly 1.18
hours, and candidate recovery only 0.90 hours. The important search bottleneck
is therefore local width-curve construction, not global candidate recovery.
The cached-teacher path passed equivalence validation at mean KL
`2.46e-9`, achieved about 10,580 cached tokens/s, and deleted its temporary
approximately 19 GiB cache after use. `torch.compile` was not used because the
host lacked a discoverable C compiler; the eager path supplied the recorded
results.

## Family-level interpretation

### What actually improved quality

The evidence ranks the investigated mechanisms approximately as follows:

1. teacher-derived initialization over random initialization;
2. adequate global-recovery tokens;
3. width allocation based on measured replacement damage;
4. the constant `3e-5` recovery configuration;
5. larger local calibration budgets; and
6. output reconstruction, RMSNorm, or LoRA, which were neutral, inconsistent,
   or inefficient in the tested forms.

This ordering is qualitative because the interventions were not all evaluated
in one factorial experiment. The controlled comparisons inside each workflow,
not the ranking itself, are the primary evidence.

### What tokens and optimizer updates mean here

SwiGLU-3 and SwiGLU-5 both use 2,048 effective tokens per optimizer update.
Their 100M endpoints therefore contain the same 48,829 updates. SwiGLU-5's
runtime gain is not caused by performing fewer scientific updates; it processes
each equal-sized update much faster. Tokens define the amount of recovery data,
updates define how many parameter changes occur, and batching geometry defines
how efficiently the fixed update is executed.

The trajectories show that more tokens generally improve PPL, but not at a
constant rate and not monotonically for fixed validation KL. Token count is
therefore a budget, not a guarantee of proportional improvement.

### Recommended homogeneous baseline

For subsequent homogeneous SwiGLU work, the evidence-backed baseline is:

- discrete per-layer width allocation with a dense 100% option;
- legacy teacher-subset initialization unless a new reconstruction method
  demonstrates a material integrated-model gain;
- constant `3e-5`, pure T=1 teacher KL, zero weight decay;
- replacement-only fused AdamW;
- 2,048 effective tokens per update with profiled microbatch geometry; and
- both fixed teacher KL and WikiText perplexity for selection and reporting.

At 20% removal, the combined reconstruction candidate remains the literal 5M
winner, but its margin is too small and its 100M behavior is unconfirmed. The
completed discrete-allocation confirmation is the stronger empirical baseline.

### Remaining limits

The family does not establish universal behavior across models or datasets.
The main limitations are:

- one model revision and one experiment seed;
- a small fixed WikiText validation evaluation of 24 batches / 6,096 predicted
  tokens;
- homogeneous SwiGLU replacements only;
- no statistical confidence intervals across independent runs;
- one historical 20% winner-selection defect, explicitly bounded above;
- direct SwiGLU-3 versus SwiGLU-5 comparisons that intentionally combine a
  structural change with the recovery and systems improvements established in
  SwiGLU-4 and SwiGLU-5; and
- substantial remaining distance from dense quality, especially at 50%
  eligible-MLP removal.

The natural next experiment class is heterogeneous operator allocation: simple
operators may be sufficient for tolerant blocks, leaving more parameter budget
for sensitive SwiGLU blocks. The current results support carrying forward the
discrete global-budget perspective, but they do not themselves demonstrate that
a heterogeneous operator family will improve quality.

## Compact 50% progression

The following table shows the clearest chronological 50% PPL improvements.
Budgets must remain visible because these are not all equal-compute rows.

| Stage | Recovery budget | WikiText PPL | What changed |
| --- | ---: | ---: | --- |
| Dense teacher | none | 14.4396 | Uncompressed reference |
| SwiGLU-1 complete uniform | one short epoch | 32.6286 | Teacher-derived local initialization |
| SwiGLU-2 winner | one short epoch | 29.5973 | Singleton-KL ranked allocation |
| SwiGLU-3 pre-recovery | 0 | 28.9146 | 393,216-pair local fitting |
| SwiGLU-3 | 10M | 24.6861 | Long replacement-only recovery |
| SwiGLU-4 efficient winner | 10M | 24.0373 | Constant `3e-5` recovery |
| SwiGLU-5 discrete allocation | 10M | 23.0215 | Discrete layer widths plus tuned recovery |
| SwiGLU-3 | 100M | 22.0908 | Original allocation, long budget |
| SwiGLU-5 discrete allocation | 100M | **20.5286** | Discrete widths, tuned and accelerated recovery |

This progression shows genuine cumulative improvement, but it also identifies
the unresolved scientific boundary: at 32.35% whole-model parameter removal,
the best homogeneous result still has a 6.0891 PPL gap to the dense model.
