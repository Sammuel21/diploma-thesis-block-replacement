---
title: Research Compute Environments
status: active
created: 2026-10-15
modified: 2026-10-15
---

# Research Compute Environments

This page records the local MVP machine and the shared remote GPU environment:
their validated setup and the capacity constraints relevant to thesis
experiments. It is an implementation-context record, not evidence for
scientific or deployment-performance claims. PERUN is the primary compute
environment (decision log, 2026-10-10) and is documented separately in
[TUKE Perun workflow infrastructure](perun/perun.md).

The content was moved here on 2026-10-15 from the wiki page
`implementation-compute-environments`, last updated 2026-07-27. The
observations below date from that time.

## Privacy boundary

This record intentionally excludes hostnames, network addresses, account names,
system-owner identity, hardware serial information, and information about other
users. It records only the capabilities and constraints needed to interpret
project experiments.

## Local MVP environment

**Project observation.** The historical MVP ran locally on an NVIDIA RTX 3060
Laptop GPU with 6 GiB VRAM. This limited available device memory and made larger
recovery runs memory-sensitive. The documented MVP hardware record is in
[`prototype/mvp/result-overview.md`](../prototype/mvp/result-overview.md).

The local CPU, RAM, disk, driver, and software-stack specifications have not
been formally inventoried. Do not infer a complete hardware comparison from the
GPU model alone.

## Shared remote experiment environment

**Project observation.** The remote Linux GPU environment was prepared for
maintained-pipeline experiments with the following verified capabilities:

- one NVIDIA RTX 4090 with 24 GiB VRAM;
- 24 logical CPU cores and 62 GiB host RAM;
- a user-local Miniconda environment named `mlp-replacement`;
- Python 3.12, PyTorch 2.11 with CUDA 13.0 support, NumPy, Transformers, and
  Datasets installed inside that environment;
- a user-controlled Hugging Face cache separate from source code, result data,
  and future exported models; and
- a successful one-batch end-to-end smoke run of the maintained one-shot linear
  replacement workflow, including data loading, replacement, recovery, and
  JSON result logging.

No system-wide package, CUDA-driver, or scheduler installation was required for
this setup. The observed filesystem capacity is operational state, not a fixed
project property, and must be rechecked before large runs.

## Comparison and experiment implications

| Aspect | Local MVP | Shared remote environment | Implication |
| --- | --- | --- | --- |
| GPU VRAM | 6 GiB | 24 GiB | The remote GPU provides four times the observed VRAM capacity, making larger recovery budgets and model variants more practical. |
| Host memory | Not formally recorded | 62 GiB | CPU-backed activation and teacher-logit caches are less constrained remotely, but still require explicit budget checks. |
| Execution role | Prototype development | Pilot and thesis experiment execution | Use the remote environment for controlled experiment runs; preserve configurations and JSON logs. |
| Resource ownership | Personal local machine | Shared access assumed | Check GPU and host availability before long runs and coordinate rather than assuming exclusive access. |

This comparison is a capacity-planning aid only. It is not a controlled
cross-hardware performance comparison and must not support latency, throughput,
or energy claims.

## Operational constraints

- No Slurm command was available during setup, so no scheduler-backed queue or
  reservation mechanism is assumed.
- The GPU is host-wide hardware. A process can consume its VRAM and compute
  capacity even when started by a different permitted user.
- The project should use one experiment per process, inspect resource state
  before long runs, and retain an independent JSON record for every run.
- Useful pre-run observations are `nvidia-smi`, `free -h`, and `df -h ~`.
- A user-local cache improves repeated-run efficiency but consumes shared disk;
  it is disposable and should remain separate from saved experiment artifacts.

## Validation

The remote setup was directly exercised on 2026-07-21 through a one-batch smoke
configuration of the maintained pipeline. It completed model loading, C4 and
WikiText2 data loading, local linear-operator fitting, in-place replacement,
teacher-logit recovery, model validation, parameter-footprint measurement, and
atomic JSON logging.

The smoke artifact is retained in user-managed external experiment storage and
is not a thesis result.

## Limitations

- The remote environment has no confirmed scheduler or resource-allocation
  policy; coordination remains a human responsibility.
- Peak GPU and host-memory measurements were not captured automatically by
  the run logger at the time of this record.
- The record does not include a complete inventory of the local MVP machine.
- The remote environment is suitable for controlled project runs, but it is not
  a general deployment or latency-measurement platform.
