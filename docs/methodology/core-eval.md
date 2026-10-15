# core-eval

We compare storage, computation, runtime, and preserved quality, grouping
metrics by what they explain. Removing width reduces computation, so runtime is
measured alongside storage. Methods are compared on two frontiers:

- **Size frontier:** actual weight bytes against quality.
- **Speed frontier:** TTFT and batched throughput against quality.

Quality is shown separately through teacher KL at 8K (primary, least noisy),
WikiText-2 PPL, and benchmark accuracy.

Runtime is observed on a fixed setup, not optimized. Diagnostics and cost
records support the comparison.

## 1. Size and footprint

| Metric | Meaning | Priority |
| --- | --- | --- |
| Total parameters | Remaining model structure; quantization does not reduce this | Core |
| Eligible-MLP removal (%) | Structural reduction within the targeted MLPs | Core |
| Actual weight bytes | Storage including quantization metadata; comparable across precisions | Core |
| Whole-model byte saving (%) | Budget used to match methods (e.g. INT8 32.3% vs SwiGLU 50% at 32.4%) | Core |
| Serialized checkpoint size | Size of the saved deployable model | Supporting |

## 2. Computation and speed

| Metric | Meaning | Priority |
| --- | --- | --- |
| FLOPs or MACs | Estimated arithmetic for a declared workload; weight-only quantization preserves matrix dimensions and logical MAC count | Core |
| TTFT on a long prompt (ms) | Time until the first output token; dominated by prompt processing | Core |
| Batched throughput (tokens/s) | Total output rate at batch 4 | Core |
| Subsequent-token latency (ms/token) | Average generation time per token after the first, batch 1 | Supporting |
| Total request latency (ms) | Time to finish the entire response | Optional summary |

Workloads are fixed by Quantization-1: batch 1 with a 2,048-token prompt, batch
4 with 2,048, and batch 1 with 7,936, each generating 256 tokens. Batch 4 is a
small fixed workload, not evidence of high-concurrency serving; the 7,936-token
TTFT is the most compute-bound measurement. Compare timings only from one job on
one GPU; the speed protocol treats runs on different physical GPUs as not
comparable. State the operation-count convention and precision, and whether
throughput includes prompt processing. No custom kernel optimization is required.

## 3. Quality and benchmarks

| Metric | Meaning | Priority |
| --- | --- | --- |
| Held-out teacher KL at 8K | Preservation of the teacher's output distribution | Core |
| WikiText-2 test PPL at 8K | Prediction quality on held-out text | Core |
| Five-task macro accuracy | Unweighted summary of downstream performance | Core summary |
| Individual task accuracies | PIQA, ARC-Easy, ARC-Challenge, WinoGrande, HellaSwag; exposes uneven damage | Core detail |
| Single-block KL | Damage from replacing one block; decides fitting recipes and allocation | Diagnostic |
| Local fitting NMSE | Approximation error for an individual block | Diagnostic |

Report quality against the same dense baseline; retain paired task uncertainty.

## 4. GPU memory and production cost

| Metric | Meaning | Priority |
| --- | --- | --- |
| Resident model VRAM | GPU memory occupied by the loaded model before inference | Supporting |
| Peak inference VRAM | Maximum memory for a specified inference workload; neither method shrinks the KV cache | Supporting |
| Peak training VRAM | Memory needed for fitting or recovery | Optional |
| Production GPU-hours | GPU time needed to produce one model, excluding the research sweep | Supporting cost |
| GPU utilization / power | Hardware activity or energy behavior | Outside default scope |

Record calibration and recovery tokens alongside production cost; name the GPU
and distinguish measured time from estimates.
