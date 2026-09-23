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
| 1 Teacher cache | Implemented; runtime checks pending | 12 cache symbols and autocast moved verbatim; 16 retained recovery definitions unchanged |
| 2 Shared operations | Implemented; runtime checks pending | Matching helpers extracted; S6 imports package reconstruction; distinct legacy evaluators/writers retained |
| 3 SwiGLU-5 modules | Implemented; runtime checks pending | 61 relocated definitions match stage 2 verbatim; explicit facade preserves 64 original definitions and six constants |
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

### Stage 1 checks

- Exact normalized-newline source comparison: all moved cache definitions and all retained recovery definitions unchanged.
- Cache module depends on model utilities, never on recovery optimization; old recovery exports retained.
- Added five CPU tensor/file contract tests in `tests/test_teacher_cache_contracts.py`.
- `git diff --check` passed. Python execution unavailable: `py -0p` reports no installed Pythons; WSL enumeration is inaccessible. No dependency installation attempted.
- The explicitly requested ledger is force-tracked as a single exception to the existing ignored `plans/` directory.

### Stage 2 checks and bounded adaptations

- Shared evaluation, state snapshots, operator loading, and FP32 surgery retain their original bodies.
- SwiGLU-3 fingerprint differences were only formatting/docstring/trailing argument comma; restoration differed only in a local variable name. Both were consolidated.
- SwiGLU-3 teacher-cache evaluation and its `blocks_by_layer` temporary-surgery signature remain unchanged. Raw S4 fingerprints, raw teacher-cache JSON, and both recovery loops remain distinct.
- The legacy atomic Torch writer only adds a function-local Torch import. `json_value` is re-exported from runlog; ExperimentLog methods are unchanged.
- Student reconstruction substitutes explicit hidden-size/allocation inputs for context/candidate lookup. Module loading, initialization order, insertion, dtype and return order remain the original operations. Compatibility wrapper and S6 call-site adaptations are explicit.
- Added eight focused artifact/reconstruction checks. Python execution remains pending in the configured research environment.

### Stage 3 checks

- All 61 definitions moved into the six internal modules match their stage-2 source bodies exactly, including nested callbacks, timers and cleanup. The already documented reconstruction wrapper is the sole S5 body adaptation from the original baseline.
- Static project import-name resolution found no missing exports. The internal module graph follows the approved dependency directions; internal cache/evaluation/reconstruction helpers import their actual owners.
- The 89-line compatibility module explicitly exposes all 64 originally defined functions/classes and six protocol constants. Both S5 CLI files and S6 preparation remain unchanged.
- Added six standard-library structural checks and three runtime import checks. These are authored and reviewed, not executed: Python remains unavailable.
