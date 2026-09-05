# Operator Calibration Data and Training Budget

Status: methodology note. The reworked method is implemented in the notebook,
but it has not been executed or empirically validated yet.

## Mental model

Local operator fitting is supervised regression. The frozen teacher MLP
receives an input vector `x` and produces an output vector `y`. The replacement
operator is trained to predict `y` from `x`.

```text
C4 documents
    -> 128-token sequences
    -> frozen language model
    -> layer-11 MLP input-output pairs (x, y)
    -> groups of pairs for operator training
    -> one weight update per operator batch
```

The two batch sizes have different roles:

| Setting | Meaning |
| --- | --- |
| Capture batch size | Number of text sequences passed through the teacher together |
| Operator batch size | Number of captured activation pairs averaged before one student weight update |

With capture batch size 2 and sequence length 128, one capture batch creates
`2 * 128 = 256` activation pairs. The MLP processes tokens independently, but
each input vector already contains context created by earlier attention layers.

## Training quantities

- Calibration pairs are the unique supervised examples available for fitting.
- Operator batch size controls how many pairs contribute to one gradient.
- An optimizer step is one update of the replacement weights.
- An epoch is one pass over every calibration pair.

The following equations are standard training bookkeeping and require no
external citation:

```text
calibration pairs = capture batches * capture batch size * sequence length
steps per epoch = ceil(calibration pairs / operator batch size)
total steps = epochs * steps per epoch
pair presentations = calibration pairs * epochs
```

More calibration data supplies more unique examples. It does not by itself
guarantee more learning: the optimizer must also receive enough steps to learn
from those examples.

## Fixed baseline and validation partition

The notebook preserves the original data setting before drawing additional
calibration samples:

```text
48 original calibration batches
    -> 24 fixed operator-validation batches
    -> 336 additional calibration batches
```

With capture batch size 2 and sequence length 128, this gives 12,288 original
calibration pairs and 6,144 fixed validation pairs. Every experiment uses the
same validation pairs. Larger calibration conditions append samples drawn only
after that validation partition, so additional training data cannot displace or
overlap the original validation data.

The original baseline is recomputed in the notebook with a 4,096-wide SwiGLU,
which is 50% of the teacher MLP's `d_ff=8,192`. The older 1,024-wide result used
50% of `d_model=2,048` and is not a valid baseline for this study.

## Legacy method: fixed optimizer-update budget

The earlier notebook derived a budget of 384 optimizer steps from its smallest
dataset:

```text
12,288 pairs / 2,048 pairs per operator batch = 6 steps per epoch
64 epochs * 6 steps per epoch = 384 total steps
```

It then reduces the epoch count as calibration data grows:

| Calibration pairs | Operator batch size | Epochs | Total steps | Pair presentations |
| ---: | ---: | ---: | ---: | ---: |
| 12,288 | 2,048 | 64 | 384 | 786,432 |
| 24,576 | 2,048 | 32 | 384 | 786,432 |
| 49,152 | 2,048 | 16 | 384 | 786,432 |
| 98,304 | 2,048 | 8 | 384 | 786,432 |

This method asks whether more unique calibration data helps under the same
training budget. It is a valid fixed-compute comparison, but it does not
estimate the best fit available from each dataset. In the 98,304-pair run,
validation error was still decreasing when the 384-step budget ended.

## Current method: fixed maximum epoch count

The primary calibration-data study gives every dataset a maximum of
64 epochs while keeping the operator batch size at 2,048:

| Calibration pairs | Operator batch size | Maximum epochs | Maximum steps | Pair presentations |
| ---: | ---: | ---: | ---: | ---: |
| 12,288 | 2,048 | 64 | 384 | 786,432 |
| 24,576 | 2,048 | 64 | 768 | 1,572,864 |
| 49,152 | 2,048 | 64 | 1,536 | 3,145,728 |
| 98,304 | 2,048 | 64 | 3,072 | 6,291,456 |

Here, increasing calibration data also increases total training. Each dataset
may be learned equally thoroughly instead of exchanging repetition for unique
examples. Use validation-based checkpoint selection and optionally early
stopping, so 64 is a maximum rather than a required final checkpoint.

This method asks how calibration size affects the best observed operator fit.
The current fixed-384-step curve can remain as a secondary compute-efficiency
comparison.

## Separate operator-batch-size study

After the calibration-size study, use the 98,304-pair dataset and compare
operator batch sizes without changing other settings:

| Operator batch size | Steps per epoch | Steps over 8 epochs | Pair presentations |
| ---: | ---: | ---: | ---: |
| 2,048 | 48 | 384 | 786,432 |
| 1,024 | 96 | 768 | 786,432 |
| 512 | 192 | 1,536 | 786,432 |
| 256 | 384 | 3,072 | 786,432 |

All conditions see the same pairs the same number of times. Smaller batches
provide more frequent but noisier updates and usually take longer because the
optimizer updates all replacement parameters more often.

The notebook screens all four batch sizes for 8 epochs, then uses the selected
batch size in the combined calibration-size study with a maximum of 64 epochs.
The data, seed, initialization, learning rate, scheduler, weight decay, and loss
remain fixed during the initial comparison.

## Implementation impact

The fixed-epoch and batch-size experiments are implemented in
`notebooks/block/operator-distillation.ipynb`. Its result artifact now records
the epoch limits, batch-size candidates, completed updates, pair presentations,
and selected-checkpoint metrics.

No change to `src/mlp_replacement/` is required for the basic experiments. The
maintained training interface already supports configurable epochs, batch
size, learning rate, scheduler, weight decay, and early stopping. Source-level
changes would only be needed for later improvements such as validation at
fixed optimizer-step intervals or step-based early stopping.

## Interpretation rule

Use both data exposure and optimizer updates when describing a run. An epoch
count alone is not a fixed compute measure because larger datasets contain
more steps per epoch. An optimizer-step count alone is also incomplete because
different batch sizes process different numbers of pairs per step.

The final report should therefore record calibration pairs, operator batch
size, epochs, optimizer steps, pair presentations, selected checkpoint, and
validation metrics.
