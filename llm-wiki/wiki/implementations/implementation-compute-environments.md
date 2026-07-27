---
id: implementation-compute-environments
title: Research Compute Environments
summary: Records the local MVP and shared remote execution environments, their validated setup, and capacity constraints relevant to thesis experiments.
type: implementation
status: review
created: 2026-07-21
updated: 2026-07-27

authorship:
  created_by: collaborative
  contributors:
    - researcher
    - llm

epistemic:
  role: mixed
  confidence: high
  verification:
    - unverified

scope:
  topics:
    - compute-environment
    - experiment-reproducibility
    - hardware-constraints
    - remote-execution
  granularities:
    - model
    - cross-level
  pipeline_stages:
    - infrastructure
    - evaluation

sources: []
related:
  - "[[concept-model-compression-evaluation-axes]]"
  - "[[decision-primary-compression-evaluation-scope]]"
  - "[[experiment-initial-block-compression-study]]"
  - "[[decision-working-experiment-code-standards]]"
supersedes: []
superseded_by: []
---

# Research Compute Environments

## Purpose

This page records the project execution environments relevant to reproducibility
and capacity planning. It is an implementation-context record, not evidence for
scientific or deployment-performance claims.

## Privacy Boundary

This record intentionally excludes hostnames, network addresses, account names,
system-owner identity, hardware serial information, and information about other
users. It records only the capabilities and constraints needed to interpret
project experiments.

## Local MVP Environment

**Project observation.** The historical MVP ran locally on an NVIDIA RTX 3060
Laptop GPU with 6 GiB VRAM. This limited available device memory and made larger
recovery runs memory-sensitive. The documented MVP hardware record is in
[`docs/prototype/mvp/result-overview.md`](../../../docs/prototype/mvp/result-overview.md).

The local CPU, RAM, disk, driver, and software-stack specifications have not
been formally inventoried. Do not infer a complete hardware comparison from the
GPU model alone.

## Shared Remote Experiment Environment

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
this setup. The currently observed filesystem capacity is operational state,
not a fixed project property, and must be rechecked before large runs.

## Comparison and Experiment Implications

| Aspect | Local MVP | Shared remote environment | Implication |
| --- | --- | --- | --- |
| GPU VRAM | 6 GiB | 24 GiB | The remote GPU provides four times the observed VRAM capacity, making larger recovery budgets and model variants more practical. |
| Host memory | Not formally recorded | 62 GiB | CPU-backed activation and teacher-logit caches are less constrained remotely, but still require explicit budget checks. |
| Execution role | Prototype development | Pilot and thesis experiment execution | Use the remote environment for controlled experiment runs; preserve configurations and JSON logs. |
| Resource ownership | Personal local machine | Shared access assumed | Check GPU and host availability before long runs and coordinate rather than assuming exclusive access. |

This comparison is a capacity-planning aid only. It is not a controlled
cross-hardware performance comparison and must not support latency, throughput,
or energy claims.

## Operational Constraints

- No Slurm command was available during setup, so no scheduler-backed queue or
  reservation mechanism is assumed.
- The GPU is host-wide hardware. A process can consume its VRAM and compute
  capacity even when started by a different permitted user.
- The project should use one experiment per process, inspect resource state
  before long runs, and retain an independent JSON record for every run.
- Useful pre-run observations are `nvidia-smi`, `free -h`, and `df -h ~`.
- A user-local cache improves repeated-run efficiency but consumes shared disk;
  it is disposable and should remain separate from saved experiment artifacts.

## Validation and Known Behavior

The remote setup was directly exercised through a one-batch smoke configuration
of the maintained pipeline. It completed model loading, C4 and WikiText2 data
loading, local linear-operator fitting, in-place replacement, teacher-logit
recovery, model validation, parameter-footprint measurement, and atomic JSON
logging.

The smoke artifact is retained in user-managed external experiment storage and
is not a thesis result. Before citing an experiment in the thesis, register its
configuration and result artifact through a canonical experiment page.

## Limitations and Technical Debt

- The remote environment has no confirmed scheduler or resource-allocation
  policy; coordination remains a human responsibility.
- Peak GPU and host-memory measurements are not yet captured automatically by
  the maintained run logger.
- The current record does not include a complete inventory of the local MVP
  machine.
- The remote environment is suitable for controlled project runs, but it is not
  a general deployment or latency-measurement platform.

## Experiments Using This Implementation

- One-shot linear smoke run, 2026-07-21: completed as a pipeline integration
  check; not promoted to a canonical experiment page or scientific finding.

## Relationships

- [[concept-model-compression-evaluation-axes]] distinguishes parameter,
  storage, runtime-memory, and quality measurements that this environment can
  support under a controlled protocol.
- [[decision-primary-compression-evaluation-scope]] explains why this page
  records capacity constraints without treating this hardware as a latency or
  deployment benchmark.
- [[experiment-initial-block-compression-study]] records the working notebooks
  intended for the shared remote GPU environment.
- [[decision-working-experiment-code-standards]] separates execution capacity
  from the maturity and hardening level of experimental code.

## Sources

No registered literature source is cited. This page records project observations
and researcher-provided terminal outputs. The local MVP GPU observation is
linked to the preserved prototype overview above.
