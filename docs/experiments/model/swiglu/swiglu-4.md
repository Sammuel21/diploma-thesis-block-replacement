---
metadata_version: 1
title: SwiGLU Global-Recovery Analysis
type: experiment-workflow
category: experiments/model/swiglu
status: draft
created: 2026-09-18
modified: 2026-09-19
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
sources:
  reference_artifacts:
    - data/results/workflows/model/swiglu-3/run-001.json
  artifacts:
    - data/results/workflows/model/swiglu-4/run-001.json
---

# SwiGLU Global-Recovery Analysis

Status: completed on darthmachinus. The schema-1 artifact and populated
load-only reporting notebook are present.

## Purpose and fixed starting state

This experiment isolates global recovery for the homogeneous SwiGLU workflow.
It does not change the allocator, local operator fit, sparsity, or replacement
architecture. Every trajectory begins independently from the exact completed
`swiglu-3` 50%-eligible-MLP-removal state: the 22 locally fitted operators at
393,216 calibration pairs, the `singleton_kl_w25_t1` allocation, and protected
layers 0 and 23. The source model has 50% removal among eligible MLPs and
32.3510% whole-model parameter removal.

The runner reads the source operator states and the original 100M packed C4
token stream in place. It checks the model revision, allocation policy, widths,
data fingerprint, batching, and numerical-precision contract before training.
The fixed selection metric for every candidate is recovery-validation teacher
KL at temperature 1, including candidates trained with temperature 2 or with a
causal-language-model loss component.

## Pipeline at a glance

```text
swiglu-3 artifact
├── 50% eligible-MLP removal
├── 22 fitted SwiGLU operators (393,216 calibration pairs)
└── fixed 100M-token recovery stream
             │
             ▼
Load dense teacher + reconstruct the fixed pre-recovery student
             │
             ▼
┌─────────────────────────────────────────────────┐
│ 1. CONFIGURATION TOURNAMENT                     │
├─────────────────────────────────────────────────┤
│ C0  constant 1e-5, KL T=1               → 10M │
│ C1  constant 3e-5, KL T=1                →  5M │
│ C2  cosine 1e-5, KL T=1                  →  5M │
│ C3  cosine 3e-5, KL T=1                  →  5M │
│                                                 │
│ Compare all at 5M using validation KL at T=1   │
│ Best optimizer candidate continues to 10M      │
│                                                 │
│ C4  winning optimizer + KL T=2           →  5M │
│ C5  winning optimizer + 90% KL/10% CE    →  5M │
│                                                 │
│ Best objective challenger may continue to 10M  │
│ Compare with incumbent → configuration winner  │
└────────────────────────┬────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────┐
│ 2. RMSNORM SCOPE                                │
├─────────────────────────────────────────────────┤
│ New students start from the pre-recovery state  │
│                                                 │
│ R1  replacements + RMSNorm at LR ×1       → 1M │
│ R2  replacements + RMSNorm at LR ×10      → 1M │
│                                                 │
│ Best validation-KL candidate continues to 10M  │
└────────────────────────┬────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────┐
│ 3. LoRA SCOPES                                  │
├─────────────────────────────────────────────────┤
│ New students start from the pre-recovery state  │
│                                                 │
│ L1  frozen replacements + MLP LoRA        → 10M │
│ L2  trainable replacements +              → 10M │
│     attention/protected-MLP LoRA                 │
│                                                 │
│ Embeddings and tied LM head remain frozen       │
└────────────────────────┬────────────────────────┘
                         ▼
Final comparison
├── published swiglu-3 result at 10M
├── reproduced C0
├── configuration winner
├── RMSNorm winner
├── L1
└── L2
```

Every C/R/L trajectory begins independently from the same compressed
pre-recovery model. Later stages inherit the winning configuration, not the
winner's trained weights.

At each requested 1M-token checkpoint:

```text
train
  → measure fixed recovery-validation KL at T=1
  → record KL / CE / total loss / learning rates / time
  → save current.pt
  → update best.pt when validation improves
  → at 0, 1M, 2M, 5M, and 10M:
       evaluate WikiText + allocation-selection data
```

The storage lifecycle is:

```text
run-001.json          persistent metrics and histories
run-001.run.json      persistent execution status
run-001.log           persistent nohup output

run-001.scratch/      temporary recovery checkpoints
└── trajectories/<candidate>/{current.pt,best.pt}

successful run  → delete run-001.scratch/
interrupted run → retain scratch for --resume
```

## Stage 1: configuration tournament

The tournament uses one 50% sparsity target. `C0` is the direct `swiglu-3`
control: replacement-only AdamW, pure KL at temperature 1, constant learning
rate `1e-5`, and zero weight decay. It runs to the requested 10M milestone.
The first round compares it at 5M against:

| ID | Learning rate | Schedule | Training objective | Initial budget |
| --- | ---: | --- | --- | ---: |
| C1 | `3e-5` | constant | KL, T=1 | 5M |
| C2 | `1e-5` | 1% warmup, cosine to 10% | KL, T=1 | 5M |
| C3 | `3e-5` | 1% warmup, cosine to 10% | KL, T=1 | 5M |

The lowest fixed-T=1 validation KL at the equal 5M milestone selects the
optimizer configuration. If necessary, that candidate continues along the
same trajectory to 10M. Its optimizer settings then define two independent
objective candidates from the original pre-recovery state:

| ID | Training objective | Initial budget |
| --- | --- | ---: |
| C4 | pure KL, T=2 | 5M |
| C5 | 90% KL at T=1 + 10% causal CE | 5M |

The best objective challenger at 5M continues to 10M and is compared with the
pure-KL incumbent at 10M. Weight decay `0.01` is represented as disabled `C6`
configuration metadata. It is deliberately outside the main run because the
learning-rate, schedule, and objective questions have higher priority and the
extra candidate would add at least 5M tokens.

## Stage 2: RMSNorm scope

This stage starts two new trajectories with the winning configuration. The
replacement operators remain fully trainable, and the
`post_attention_layernorm` preceding each replaced MLP is added to the
trainable scope. `R1` uses the replacement learning rate for RMSNorm; `R2`
uses ten times that rate. Both qualify at 1M tokens, and the lower fixed-T=1
validation-KL candidate continues to 10M.

RMSNorm is a trainable-scope experiment, not a learning-rate schedule. Its two
parameter groups share the winning schedule but have different base learning
rates. The selected norm weights use FP32 master parameters, like the
replacement operators.

## Stage 3: LoRA scopes

Both LoRA trajectories use rank 16, alpha 32, zero adapter dropout, and adapter
learning rate `1e-4`. They inherit the winning objective and schedule and run
independently to 10M.

- `L1` freezes the compressed replacement matrices and trains LoRA adapters on
  their gate, up, and down projections. This measures parameter-efficient
  recovery of the replacement subset.
- `L2` keeps the replacement operators fully trainable and adds LoRA to the
  otherwise frozen attention q/k/v/o projections in all 24 blocks and to the
  original gate/up/down MLP projections in protected layers 0 and 23. This is
  called transformer-wide assisted LoRA rather than literal whole-model LoRA.

Input embeddings and the LM head are excluded. The LM head is the final
hidden-state-to-vocabulary projection. SmolLM2 ties its weight storage to the
input embedding matrix, so adapting either changes the same vocabulary table.
That high-capacity table could directly absorb KL or perplexity error without
showing that the compressed Transformer blocks recovered. Its exclusion keeps
this experiment focused and avoids special tied-weight adapter handling. The
runner verifies the tie at runtime and records the exclusion and targeted
module paths in the artifact.

## Tokens, checkpoints, and comparison with swiglu-3

The data order, sequence length 128, two-sequence microbatch, eight-step
accumulation, BF16 forward autocast, FP32 trainable weights and optimizer state,
recovery-validation split, allocation-selection split, and WikiText-2 split
match `swiglu-3`. Requested checkpoints occur every 1M tokens through 10M.
Requests are rounded to complete optimizer boundaries, and both requested and
actual token counts are stored. Full WikiText and allocation-selection metrics
are recorded at requested 0, 1M, 2M, 5M, and 10M; fixed validation KL, training
KL/CE/total loss, parameter-group learning rates, optimizer updates, elapsed
time, and throughput are retained at every checkpoint.

`C0` therefore reproduces the `swiglu-3` 10M methodology at its actual
10,000,384-token boundary. The final table contains the published `swiglu-3`
10M row, the new `C0`, the configuration winner, the RMSNorm winner, and both
LoRA scopes. The maximum planned training is about 76M token positions: up to
45M for configuration selection, 11M for RMSNorm, and 20M for LoRA.

## Artifact and interruption behavior

The runner is `workflows.runs.model.swiglu.swiglu_4_recovery_analysis`, configured by
`workflows/configs/model/swiglu/swiglu-4-recovery-analysis.json`. Its persistent scientific products are
the main JSON and sibling `.run.json`; the shell log remains the external nohup
log. The main JSON contains all histories, milestones, trainable parameter
counts, target paths, memory measurements, runtime, source fingerprints,
selection decisions, and final comparisons needed by the notebook.

No model weights, token cache, or `assets/` directory remains after a successful
run. During execution, `current.pt` and `best.pt` files live under a sibling
`.scratch` directory so an interrupted trajectory can resume with its model,
optimizer, RNG, and token cursor intact. Scratch for rejected candidates is
removed after its selection decision is committed; all remaining scratch is
removed after successful completion. A failed or interrupted run retains it.

From the repository root on darthmachinus:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p data/results/workflows/model/swiglu-4
nohup python -u -m workflows.runs.model.swiglu.swiglu_4_recovery_analysis \
  --source data/results/workflows/model/swiglu-3/run-001.json \
  --output data/results/workflows/model/swiglu-4/run-001.json \
  > data/results/workflows/model/swiglu-4/run-001.log 2>&1 &
```

Resume an interrupted run with the same source and output:

```bash
python -u -m workflows.runs.model.swiglu.swiglu_4_recovery_analysis \
  --source data/results/workflows/model/swiglu-3/run-001.json \
  --output data/results/workflows/model/swiglu-4/run-001.json \
  --resume
```

`--smoke` reduces the token budgets and evaluation batches but still requires
the source operators and packed token cache. Smoke results only test execution,
checkpointing, and artifact construction.

## Notebook view and validation boundary

`notebooks/model/swiglu/swiglu-4.ipynb` is a load-only view. It discovers or accepts a
completed artifact and reconstructs the configuration, RMSNorm, LoRA, and final
comparison tables and charts without loading a model or dataset. This keeps the
headless job and notebook logically equivalent while avoiding accidental
retraining in an interactive session.

The completed run reproduced the published SwiGLU-3 10M row exactly in `C0`.
The constant-`3e-5`, pure-T=1 replacement-only configuration `C1` improved
fixed validation KL from 0.286172 to 0.266991 and WikiText perplexity from
24.6861 to 24.0373. RMSNorm changed the result only slightly (`R1`: KL
0.266596, PPL 24.0115). Replacement-only LoRA was substantially worse (`L1`:
KL 0.309782, PPL 25.5665), while transformer-assisted LoRA was the best listed
10M result (`L2`: KL 0.265047, PPL 23.9864) but trained 560,922,624 parameters
and took 6,076 seconds. These results motivate SwiGLU-5's use of the `C1`
recovery configuration and its exclusion of RMSNorm and LoRA from the bounded
search matrix.
