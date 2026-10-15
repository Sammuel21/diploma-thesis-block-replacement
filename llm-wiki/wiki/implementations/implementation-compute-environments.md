---
id: implementation-compute-environments
title: Research Compute Environments
summary: Pointer to the docs record of the local MVP and shared remote GPU environments; PERUN is documented separately.
type: implementation
status: review
created: 2026-07-21
updated: 2026-10-15

epistemic:
  role: mixed
  confidence: high
  verification:
    - unverified

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

## Overview

**Pointer page.** The record of the local MVP machine (RTX 3060 Laptop, 6 GiB)
and the shared remote GPU environment (RTX 4090, 24 GiB), with its privacy
boundary, setup, capacity comparison, and operational constraints, moved on
2026-10-15 to
[`docs/infrastructure/compute-environments.md`](../../../docs/infrastructure/compute-environments.md).
PERUN, the primary compute environment, is documented in
[`docs/infrastructure/perun/perun.md`](../../../docs/infrastructure/perun/perun.md).

These records support capacity planning only. They are not evidence for
latency, throughput, energy, or deployment claims.

## Relationships

- [[concept-model-compression-evaluation-axes]] distinguishes parameter,
  storage, runtime-memory, and quality measurements that these environments can
  support under a controlled protocol.
- [[decision-primary-compression-evaluation-scope]] explains why the hardware
  is not treated as a latency or deployment benchmark.
- [[experiment-initial-block-compression-study]] records the working notebooks
  intended for the shared remote GPU environment.
- [[decision-working-experiment-code-standards]] separates execution capacity
  from the maturity and hardening level of experimental code.

## Sources

No registered literature source is cited. The moved record is based on project
observations and researcher-provided terminal outputs.
