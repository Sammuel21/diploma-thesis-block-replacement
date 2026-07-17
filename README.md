# Compressing Large Language Models via Replacement of MLP Blocks

Diploma thesis development repository, FMFI UK Bratislava.

The thesis studies whether selected Transformer MLP blocks can be replaced by
smaller drop-in operators trained from calibration activations, and how those
replacements affect model quality and compression. The formal baseline scope is
defined in [docs/annotation.md](docs/annotation.md).

## Repository map

| Path | Responsibility |
| --- | --- |
| `llm-wiki/` | LLM-maintained, source-linked research knowledge base |
| `docs/` | Distilled, human-oriented documentation and historical records |
| `docs/prototype/mvp/` | Frozen documentation and manifest for the completed MVP |
| `notebooks/mvp/` | Historical MVP execution and analysis notebooks |
| `scripts/intro/` | Historical MVP helper implementation |
| `configs/` | MVP configuration modules; future configuration structure is not yet finalized |
| `data/mvp/results/logs/` | Versioned MVP experiment evidence |
| `src/` | Reserved for the maintained implementation after code consolidation |
| `pipelines/` | Reserved for maintained experiment orchestration |

The executable MVP paths remain unchanged to avoid breaking notebook imports.
New maintained code should not be added to the legacy MVP modules until the
codebase consolidation phase defines its target architecture.

## Navigation

- [Human documentation index](docs/README.md)
- [MVP archive](docs/prototype/mvp/README.md)
- [LLM-wiki orientation](llm-wiki/README.md)
- [LLM-wiki schema](llm-wiki/SCHEMA.md)
- [Repository agent instructions](AGENTS.md)

## Preservation

The pre-restructuring implementation is preserved by the remote branch
`origin/mvp-archive-freeze` at commit
`e8e6615ecf1119ec666237e5dbb7de898bb18211`. The archive manifest records
artifacts that are not part of that original commit, including the experiment
logs that were previously ignored by Git.
