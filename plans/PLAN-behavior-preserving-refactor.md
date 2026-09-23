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
| 4 Documentation and verification | Documentation and available inspection complete; execution incomplete | Ownership references updated; source/import inspection and five artifact hashes checked; Python unavailable |

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

## Baseline surfaces and relocation map

All locations in the baseline column refer to `dc11a031a07b217e1661adf0cb1f29b1ccb28393`; exact original definitions remain recoverable with `git show`. The structural checks enumerate every original maintained definition, rather than storing a second source copy.

| Baseline location | Current owner / compatibility boundary |
| --- | --- |
| `compression/recovery.py:17` cache dataclasses; `:134` shard loader and cache functions | `compression/teacher_cache.py`; explicit recovery re-exports |
| `compression/recovery.py:829` two-argument autocast | `model.py`; recovery re-export |
| `swiglu/shared.py:28` matching artifact/configuration helpers | `artifacts.py`, `config.py`, workflow `common.py`; shared re-exports |
| `swiglu/shared.py:424` one-argument autocast and mixed evaluators | `evaluation/mixed_precision.py`; shared re-exports |
| `swiglu/shared.py:580` saved-operator loading | `compression/reconstruction.py`; shared re-export |
| `swiglu_5.py:177` replacement state; `:192` FP32 insertion | `compression/reconstruction.py` and `compression/surgery.py`; facade exports |
| `swiglu_5.py:204` context and remaining scientific/orchestration helpers | Six `swiglu5/` modules; explicit facade exports |
| `swiglu_5.py:2363` search; `:3030` confirmation | `swiglu5/search.py`, `swiglu5/confirmation.py` |
| `swiglu_5.py:2778` blank student | Explicit-input package constructor plus compatibility wrapper |

Existing CLI surfaces are `workflows.runs.model.swiglu.swiglu_5_search`, `swiglu_5_confirmation`, `swiglu_6_prepare`, `swiglu_6_recovery`, and `swiglu_6_evaluate`. S5 CLI files and S6 preparation are byte-for-byte equal after newline normalization; the S6 `main` definitions are unchanged. Argument definitions, defaults, configuration paths, and exception boundaries therefore remain the baseline implementation. Help execution is pending. The compression package exports are unchanged; recovery, shared, runlog, and the S5 facade preserve the documented compatibility imports.

## Final verification evidence (2026-09-23)

| Check | Result and limit |
| --- | --- |
| Original definition source comparison | 87 of 89 relocated definitions are verbatim. Exceptions: local Torch import and explicit-input reconstruction/wrapper, documented above. All retained definitions are unchanged except the two bounded S6 reconstruction call adaptations. |
| SwiGLU-5 split | All 61 stage-3 moves are verbatim relative to stage 2; all six constants match the baseline. |
| Duplicate inspection | 13 matching definitions consolidated (152 original lines); only docstrings, formatting, a trailing argument comma, and the documented restoration local-variable name differed. |
| Import inspection | All project `from` bindings resolve statically; inspected module graph has no cycles, no package dependency on workflows, and no S6 import of either S5 implementation path. Runtime resolution remains pending. |
| Historical JSON | All five original SHA-256 hashes still match; schema 1/completed statuses, S4 source milestone, S5 candidate IDs, allocation fields, winner endpoints and confirmation retained-state records inspected read-only. No model weights loaded. |
| Protected scope | No configuration, notebook, scheduler, checkpoint directory, historical artifact, dependency, or wiki changes. Original loops, selection rules, source hashing and continuation policy retained. |
| Whitespace | `git diff --check` passed for each implementation stage and the final patch. |
| Python execution | `py -3 -m unittest discover -s tests -v` exits 1: `No installed Python found!`. No AST parse, runtime import, help execution, or tensor-check pass is claimed. |
| Numerical equivalence | Pending the next intended scientific run; no scientific job, profiling, recovery, or reduced-budget workflow launched. |

The source comparisons preserve data order, RNG operations, parameter/optimizer ordering, precision/reductions, callback timing, model modes, tie rules, guards, token boundaries, commit/prune ordering and resource lifetimes as written. They do not establish numerical equivalence.

### Focused checks ready for the research environment

Twenty-five unittest methods in six files cover artifact bytes/fingerprints and writer distinctions (4), replacement state and surgery (4), teacher-cache indexing/manifests/release and raw serialization (6), structural preservation (6), runtime imports (3), and historical schemas/adapters (2). Historical checks explicitly skip if ignored local JSON references are unavailable. The structural suite requires the preserved baseline commit in local Git history.

Run from the repository root in the configured research Python environment:

```bash
python -m unittest discover -s tests -v
```

The standard-library structural checks can be run alone with `python -m unittest discover -s tests -p test_refactor_structure.py -v`. The tensor/import checks require the existing research dependencies. No dependency changes or substitute environment were introduced.

### Deferred findings

- S3 teacher-cache evaluation keeps its distinct empty-input behavior. Its narrower saved-operator loader and `blocks_by_layer` temporary-replacement helper also remain distinct. The planned matching consolidations were completed; these unmatched variants were not adapted.
- Legacy normalized JSON/fingerprints/Torch writes, strict durable writes, cache raw JSON, older distinct fingerprints, and `ExperimentLog.write` remain separate.
- Both existing confirmation artifacts begin `full_evaluations` with `actual_tokens=2000896`, whereas their initial validation history records the retained search endpoint at `5001216`. This inconsistency predates the refactor; metadata and lookup behavior are preserved. Investigate separately before interpreting that row as a token-aligned scientific observation.
- Runtime verification is incomplete because the configured research Python environment is unavailable locally. Do not treat the authored tests as passed or this branch as numerically validated.

## Line accounting

Counts are physical source lines with normalized newlines. Definition spans include signatures/decorators, docstrings and internal blank lines, but exclude surrounding separators/imports. Relocation is not deletion.

| Category | Lines |
| --- | ---: |
| Original definition spans relocated/adapted (89 definitions, including the constructor extraction) | 3,899 |
| Original protocol constants relocated | 58 |
| Duplicate definition spans removed (11 S3, one S4, one recovery hash helper) | 152 |
| Compatibility facade, wrapper and re-export statements | 119 |
| Added focused check files | 655 |

The compatibility count consists of the 89-line S5 facade, six-line reconstruction wrapper, and 24 lines of re-export statements in recovery/shared/runlog (whole mixed import statements counted). Other import wiring, module docstrings and separators are reflected in the whole-file totals. Maintained production Python totals **18,709 -> 18,810 lines (+101)** across `src/` and `workflows/`, excluding tests and Markdown. The result improves ownership and removes demonstrated duplication; it does not claim a net source-line reduction.

## Stage commits and rollback

| Boundary | Local commit |
| --- | --- |
| Original baseline / original branch `llm-wiki` | `dc11a031a07b217e1661adf0cb1f29b1ccb28393` |
| Stage 0 ledger | `24ab2f9` |
| Stage 1 cache extraction | `1c52911` |
| Stage 2 shared operations | `77531f8` |
| Stage 3 S5 decomposition | `d916250` |
| Stage 4 documentation / verification closure | Commit titled `Document refactor ownership and verification limits` |

These are local, separately reviewable commits on `refactor/maintained-workflows`; nothing has been published. Before publication, return to the preserved original branch/checkout if abandoning the refactor. After publication, revert affected stages in reverse dependency order while preserving unrelated later work. Never use destructive reset/clean, rewrite historical artifacts, or delete checkpoint directories for rollback. Existing runs retain their actual original source snapshot, even when it differs from this baseline. New S6 source hashes require new outputs; checkpoint migration remains out of scope.

The append-only wiki log is untouched. Its maintained 2026-09-21 registration entry confirms recent wiki operations, not a complete development timeline or currency of every scientific wiki page.
