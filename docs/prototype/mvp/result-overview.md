Below is a long, self-contained summary you can paste into another LLM for presentation generation.

---

# Diploma Thesis MVP Summary: Block Replacement Prototype

My diploma thesis investigates whether selected MLP blocks inside a pretrained large language model can be replaced by smaller drop-in modules while preserving as much model quality as possible. The broader motivation is model compression: transformer MLP blocks contain a large share of parameters and compute, so replacing them with lighter approximations could reduce model size or runtime cost.

The current MVP focuses on the prototype workflow rather than final compression performance. The goal was to build an experimental infrastructure that can:

1. Select target MLP blocks for replacement.
2. Train replacement operators from activation data.
3. Insert replacements into the original model.
4. Optionally apply lightweight recovery training.
5. Evaluate the quality degradation at both block level and model level.
6. Log all experiment outputs for later analysis.

The prototype is implemented around `SmolLM2-1.7B` and currently uses a simple linear replacement operator for the MLP block output function.

---

# Methodological Design

The methodology is inspired mainly by ideas from Minitron, Grafting, and related compression work such as MoDeGPT. The common principle is that instead of retraining a model from scratch, we use a smaller calibration/recovery dataset to identify important components, replace or prune selected parts, and then apply a lightweight repair step.

In this MVP, the workflow is:

1. Load the pretrained model.
2. Prepare calibration data from C4.
3. Prepare evaluation data from Wikitext2.
4. Collect input/output activation pairs for each target MLP block.
5. Fit a smaller replacement operator to imitate the original MLP block.
6. Replace one or more MLP blocks in the model.
7. Optionally perform model-level recovery using cached teacher logits.
8. Evaluate language modeling loss and perplexity.
9. Store all results in structured JSON logs.
10. Parse and analyze the logs in a separate analysis notebook.

The calibration and evaluation data are intentionally separated. Calibration data is used for block fitting, BI-score screening, and recovery. Evaluation data is used only to measure final model degradation.

---

# Current Model and Data Setup

The current MVP uses:

- Model: `HuggingFaceTB/SmolLM2-1.7B`
- Device: CUDA GPU, RTX 3060 Laptop GPU with 6GB VRAM
- Sequence length: 128
- Batch size: 2
- Calibration dataset: C4
- Evaluation dataset: Wikitext2
- Block fitting budget: 24 calibration batches
- Evaluation budget: 24 evaluation batches
- High recovery budget: 512 recovery batches

The recovery budget was increased after realizing that the original setup had very little recovery data. Initially, recovery used only:

```text
24 batches × batch size 2 × sequence length 128 = 6,144 tokens
```

The later high-budget recovery setup uses:

```text
512 batches × batch size 2 × sequence length 128 = 131,072 tokens
```

This is around 21.3 times more recovery tokens than the original setup. Even this is still far smaller than the budgets used in papers such as Minitron, where recovery can involve billions of tokens.

---

# Replacement Operator

The current implemented replacement operator is a simple linear layer:

```text
hidden_size -> hidden_size
```

It is trained to imitate the original MLP block output from collected activation pairs.

For a target MLP block, the prototype collects:

```text
X = input activations to the original MLP block
Y = output activations from the original MLP block
```

Then the replacement operator is trained with MSE loss:

```text
MSE(replacement(X), original_MLP(X))
```

This gives a local block imitation objective. The original MLP block itself would have zero MSE by definition, so replacement MSE measures how badly the approximation deviates from the original block.

At the moment, no smaller MLP replacement operator is implemented yet. That is planned as the next methodological extension because the linear operator is likely too weak to approximate nonlinear MLP behavior.

---

# Block Importance Screening

The prototype computes a BI-style score for each MLP block based on activation behavior.

The current BI score is based on cosine distance between the MLP input and output activations:

```text
BI = mean(1 - cosine_similarity(X, Y))
```

Intuitively, a low BI score means the block output is more similar to its input, while a high BI score means the block performs a stronger transformation.

The MVP tests two BI-based selection directions:

- `asc`: choose low-BI blocks first
- `desc`: choose high-BI blocks first

The hypothesis was:

```text
Low-BI blocks may be easier to replace with a simple linear operator.
High-BI blocks may be more important or more nonlinear, so replacing them should hurt more.
```

This hypothesis mostly held at fixed `k=5`, but failed in an interesting way for `k=6`, showing that BI alone is not sufficient.

---

# Replacement Strategies

The experimental plan includes several axes:

1. One-shot replacement vs iterative replacement.
2. Random-k selection vs BI-based top-k selection.
3. Low-BI selection vs high-BI selection.
4. No recovery vs recovery after replacement.
5. Linear operator vs smaller MLP operator.

Currently implemented and tested:

- One-shot replacement.
- Random-k replacement.
- BI-based replacement.
- Low-BI and high-BI selection.
- Recovery on/off.
- Search over `k = 1..6`.
- Linear operator.

Not yet implemented:

- Iterative replacement.
- Smaller MLP replacement operator.
- Full global best subset search.
- More advanced recovery strategies.

The current search is not an exhaustive subset search. It is a BI-prefix search. For example, for each `k`, it selects the first `k` layers according to the BI ranking.

---

# Recovery Training

The recovery step is model-level knowledge distillation after replacement.

The original pretrained model is treated as a teacher. Before replacement, teacher logits are cached on calibration/recovery batches. After replacement, the modified student model is trained to match the cached teacher logits.

The recovery objective is KL divergence between student and teacher output distributions:

```text
KL(student_logits || teacher_logits)
```

Only the replacement modules are unfrozen during recovery. The rest of the model remains frozen.

This means recovery is not full model retraining. It is a lightweight repair step designed to recalibrate the inserted replacement operators after they are placed inside the full model.

This distinction is important methodologically:

```text
Block fitting = local MLP imitation using activation MSE.
Recovery = model-level distillation using teacher logits.
Evaluation = final language modeling loss and perplexity.
```

---

# Logging and Analysis Infrastructure

The MVP now includes structured experiment logging. Each experiment stores:

- Configuration values.
- Selection strategy.
- BI rank order.
- Whether recovery was applied.
- Target layer indices.
- Final model loss.
- Final model perplexity.
- Per-block replacement MSE.
- Per-layer replacement metadata.

The analysis notebook parses logs into dataframes:

- `run_df`: one row per fixed experiment run.
- `block_df`: one row per replaced block.
- `search_df`: one row per search configuration.
- `search_block_df`: one row per replaced block inside search experiments.

This separation makes it possible to analyze both model-level and block-level behavior.

---

# Baseline Model Quality

The baseline model before any replacement has approximately:

```text
Base loss: 3.0152
Base perplexity: 20.39
```

All replacement experiments are interpreted relative to this baseline.

The main metric used in analysis is:

```text
delta_loss = model_loss - base_loss
```

Important interpretation:

```text
Higher delta_loss is worse.
Lower delta_loss is better.
```

A positive delta means the replacement damaged the model.

---

# Fixed-k Experimental Findings

For fixed `k=5`, several replacement scenarios were evaluated.

The random-k replacement selected layers roughly like:

```text
[5, 9, 13, 20, 22]
```

This caused moderate degradation:

```text
Loss around 3.85
PPL around 47
Delta loss around 0.83
```

BI-based low-BI replacement selected layers roughly like:

```text
[2, 3, 18, 19, 20]
```

This caused larger degradation than random-k, but less than high-BI replacement:

```text
Loss around 4.28-4.33
PPL around 72-76
Delta loss around 1.27-1.32
```

BI-based high-BI replacement selected layers roughly like:

```text
[8, 9, 10, 12, 23]
```

This caused stronger degradation:

```text
Loss around 4.67
PPL around 107
Delta loss around 1.65
```

This supports the basic hypothesis that high-BI blocks are harder or more damaging to replace with a simple linear operator.

However, the low-BI strategy was not always safe, especially when increasing `k`.

---

# Search Experiment Findings

A search experiment was run for:

```text
k = 1, 2, 3, 4, 5, 6
BI order = asc / desc
Recovery = on / off
```

This gives:

```text
2 BI orders × 2 recovery settings × 6 values of k = 24 configurations
```

The search evaluates how model degradation changes as more blocks are replaced.

For high-BI descending selection, degradation increased steadily but remained bounded:

```text
k=1 delta_loss ≈ 1.15
k=2 delta_loss ≈ 1.22
k=3 delta_loss ≈ 1.32
k=4 delta_loss ≈ 1.53
k=5 delta_loss ≈ 1.66
k=6 delta_loss ≈ 2.54 without recovery
k=6 delta_loss ≈ 2.39 with high-budget recovery
```

For low-BI ascending selection, results were reasonable until `k=5`, but became catastrophic at `k=6`:

```text
k=1 delta_loss ≈ 0.20
k=2 delta_loss ≈ 0.45
k=3 delta_loss ≈ 0.85-0.99
k=4 delta_loss ≈ 0.96
k=5 delta_loss ≈ 1.27-1.32
k=6 delta_loss ≈ 10-11
```

This is one of the most important empirical observations from the MVP.

---

# Important Interpretation: Why Low-BI Failed at k=6

The low-BI `k=5` selection was approximately:

```text
[2, 3, 18, 19, 20]
```

The low-BI `k=6` selection added layer 1:

```text
[1, 2, 3, 18, 19, 20]
```

Adding this early layer caused catastrophic global degradation.

This shows that low BI does not automatically mean a block is safe to replace. A block may have low input-output transformation strength but still be positioned early in the model, where even moderate approximation error propagates through many later layers.

This is a key methodological insight:

```text
Block replaceability is not determined only by local BI score.
It also depends on layer position, downstream error propagation, and approximation error.
```

A later layer can have high local replacement MSE but less opportunity to propagate damage. An early layer can have moderate local error but cause severe global failure.

This motivates future screening criteria that combine:

- BI score.
- Layer depth.
- Local replacement MSE.
- Possibly sensitivity of final model loss.
- Possibly iterative validation after each replacement.

---

# Recovery Findings

The recovery stage was tested with both the original low budget and the later higher budget.

With the original low recovery budget, recovery had little or inconsistent effect. Sometimes recovered results were almost identical to unrecovered results.

This was expected after realizing that the original recovery used only about 6,144 tokens, which is extremely small compared with recovery budgets in related papers.

With the increased recovery budget of 131,072 tokens, recovery improved some cases, especially the catastrophic or larger-degradation cases, but it still did not fully repair the model.

Examples:

```text
Low-BI k=6:
without recovery delta_loss ≈ 10.81
with high-budget recovery delta_loss ≈ 9.82
```

```text
High-BI k=6:
without recovery delta_loss ≈ 2.54
with high-budget recovery delta_loss ≈ 2.39
```

So recovery helped somewhat, but not enough to make linear replacement robust.

This suggests that the main bottleneck is not only recovery budget. The linear replacement operator is probably too weak to approximate the original nonlinear MLP blocks.

---

# Methodological Lessons

The MVP produced several useful methodological lessons.

First, local block imitation and global model quality are related but not identical. A replacement can have a measurable block MSE, but the final effect on model loss depends heavily on where the block sits in the network.

Second, BI score is useful for screening, but insufficient as the only criterion. Low-BI blocks were often less damaging than high-BI blocks, but the catastrophic low-BI `k=6` result shows that layer position and propagation matter.

Third, recovery training requires enough data to be meaningful. The original recovery budget was too small. Increasing the recovery budget improved some results, but recovery cannot fully compensate for a weak replacement operator.

Fourth, one-shot replacement is harsh because multiple approximations are inserted at once. Iterative replacement may allow better control because each replacement can be evaluated or recovered before moving to the next one.

Fifth, linear replacements are a useful stress-test and baseline, but likely not expressive enough for final compression. A smaller MLP replacement is the next important experiment.

---

# What Has Been Achieved Infrastructure-Wise

The MVP now has a working experimental pipeline for block replacement.

Implemented components include:

- Model loading.
- Tokenizer setup.
- GPU execution.
- C4 calibration loader.
- Wikitext2 evaluation loader.
- MLP block path discovery.
- Activation collection with hooks.
- BI score computation.
- Target layer resolver for manual, random-k, and top-k-BI selection.
- Linear replacement operator.
- Local replacement training using MSE.
- One-shot multi-block replacement.
- Optional model-level KD recovery.
- Teacher-logit caching.
- Evaluation with loss and perplexity.
- JSON experiment logging.
- Analysis notebook with parsing functions.
- Search over multiple `k` values.
- Comparison of recovery budgets.

This infrastructure is likely the most important MVP contribution so far because it makes future experiments systematic and reproducible.

---

# What Has Been Achieved Empirically

The MVP established several empirical findings:

1. Replacing MLP blocks with simple linear operators causes measurable degradation.
2. High-BI blocks are generally more damaging to replace than low-BI blocks.
3. Random-k replacement can sometimes outperform naive low-BI selection.
4. Low-BI selection is not automatically safe.
5. Early-layer replacement can cause catastrophic error propagation.
6. Recovery with too few tokens is nearly ineffective.
7. Increasing the recovery budget helps but does not solve the problem.
8. Linear operators are probably too weak for robust block replacement.
9. Per-block replacement MSE varies strongly across layers.
10. Model-level degradation cannot be explained by local replacement MSE alone.

---

# Current Limitations

The current MVP has several limitations that should be clearly stated in the presentation.

The replacement operator is only linear, so it is a very constrained approximation of the original MLP block.

The recovery budget is still much smaller than in large-scale compression papers.

The experiments use only one model, `SmolLM2-1.7B`.

The current search is not a true global subset search. It is a prefix search over BI rankings.

The iterative replacement strategy has not yet been implemented.

Recovery only trains the replacement modules, not the whole model or surrounding layers.

The evaluation budget is small and should be considered preliminary.

---

# Planned Next Steps

The next steps are:

1. Implement a smaller MLP replacement operator.
2. Compare linear vs smaller MLP replacement.
3. Implement iterative replacement.
4. Compare one-shot vs iterative replacement.
5. Improve screening by combining BI, depth, and local replacement MSE.
6. Test larger recovery budgets if compute allows.
7. Add clearer visualizations for search curves and block-level errors.
8. Expand the analysis notebook into presentation-ready plots.
9. Document the full workflow for thesis writing.
10. Eventually compare parameter savings against quality degradation.

The most important next experiment is likely replacing the linear operator with a smaller MLP, because current results suggest that the linear operator is the main bottleneck.

---

# Suggested Presentation Narrative

A good presentation structure would be:

1. Motivation: LLMs are large; MLP blocks are expensive; replacing blocks could compress the model.
2. Thesis idea: learn smaller drop-in approximations for selected MLP blocks.
3. Inspiration: Minitron, Grafting, MoDeGPT, lightweight calibration and recovery.
4. MVP workflow: activation collection, block fitting, replacement, recovery, evaluation.
5. Infrastructure built: loaders, hooks, operators, resolver, logging, analysis.
6. Experiment design: random vs BI selection, low-BI vs high-BI, recovery on/off, search over k.
7. Results: replacement degrades quality, high-BI worse than low-BI at fixed k.
8. Key surprise: low-BI k=6 catastrophic because early layer 1 was added.
9. Recovery: small recovery budget ineffective; larger budget helps but not enough.
10. Conclusion: MVP validates the pipeline and reveals that replaceability depends on BI, depth, propagation, and operator expressivity.
11. Next work: smaller MLP operator and iterative replacement.
