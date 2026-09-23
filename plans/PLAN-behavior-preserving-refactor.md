# Behavior-preserving refactor implementation ledger

## Baseline and scope

- Baseline: `dc11a031a07b217e1661adf0cb1f29b1ccb28393`.
- Original branch: `llm-wiki`; working tree clean before implementation.
- Refactor branch: `refactor/maintained-workflows`.
- Approved scope: teacher-cache separation; shared evaluation, reconstruction and matching helpers; SwiGLU-5 decomposition; SwiGLU-6 dependency cleanup.
- Scientific loops, configurations, schemas, notebooks, scheduler files, historical artifacts, and wiki remain unchanged.
- Existing SwiGLU-6 runs require their original source snapshot: source hashes remain enforced. No checkpoint migration or scientific job launch.

## Stage status

| Stage | Status | Evidence |
| --- | --- | --- |
| 0 Baseline | Recorded | Clean tree and immutable reference hashes below |
| 1 Teacher cache | Pending | |
| 2 Shared operations | Pending | |
| 3 SwiGLU-5 modules | Pending | |
| 4 Documentation and verification | Pending | |

## Reference artifacts

All five records are schema 1 and marked completed. These hashes describe the original bytes, not numerical parity.

| Artifact | SHA-256 |
| --- | --- |
| `data/results/workflows/model/swiglu-3/run-001.json` | `105e6e0588fb0281cfa382d5c47daa976c1ab00dbf504a132676cf037364d3da` |
| `data/results/workflows/model/swiglu-4/run-001.json` | `b3b00b92a2b291c1bd36e7c9d1ef72bdeaa9b4914492b77f5bc7f344c04695ac` |
| `data/results/workflows/model/swiglu-5/search/run-001.json` | `28f7e5624bb490f1dc1bd87d63695917ddd684dc95464642da9b04689680715b` |
| `data/results/workflows/model/swiglu-5/confirmation/run-001-target-0.2.json` | `940677d6853e9df98b0136e84ea939247bdd43a8591aa27faf27056e2310b180` |
| `data/results/workflows/model/swiglu-5/confirmation/run-001-target-0.5.json` | `47f1bc98b32d4bd8b1935daaf7880c7cb94ad018f74bfa02c575b6bcc33614e4` |

## Validation and limitations

- Structural comparison and focused contract checks are required.
- Numerical equivalence is pending the next intended scientific run.
- No smoke or reduced-budget experiments are authorized.
- Wiki log reconstruction is skipped: latest wiki/log operation is 2026-09-21 in cfa2318.
