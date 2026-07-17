# Block Replacement MVP Archive

## Status

This directory documents the completed MVP prototype. It is a historical
record, not the canonical specification for future methodology or production
code.

The executable files remain at their original repository paths because the
notebooks import `scripts.intro` and `configs.intro_config` directly. Moving
those files would damage the very state this archive is intended to preserve.

## Frozen checkpoint

- Remote branch: `origin/mvp-archive-freeze`
- Local annotated tag: `mvp-v1` (not yet pushed)
- Commit: `e8e6615ecf1119ec666237e5dbb7de898bb18211`
- Current restructuring branch: `llm-wiki`
- Archive manifest: [manifest.yml](manifest.yml)

The frozen branch preserves the committed implementation. The manifest and
versioned JSON logs complete the local empirical record that was previously
excluded by `.gitignore`.

## Artifact map

| Artifact | Location |
| --- | --- |
| Main workflow notebook | `notebooks/mvp/multi-block-replacement.ipynb` |
| Analysis notebook | `notebooks/mvp/analysis.ipynb` |
| Introductory notebooks | `notebooks/mvp/introduction.ipynb`, `notebooks/mvp/methods.ipynb` |
| Prototype helpers | `scripts/intro/` and `scripts/utils.py` |
| Experiment configuration | `configs/intro_config.py` |
| Experiment logs | `data/mvp/results/logs/` |
| Long historical result summary | [result-overview.md](result-overview.md) |
| Experiment index | [experiments.md](experiments.md) |
| Early design notes | [notes/](notes/) |

## MVP boundary

The prototype established an end-to-end workflow for:

1. loading a pretrained causal language model;
2. preparing calibration and evaluation data;
3. collecting MLP input/output activations;
4. computing BI-style block scores;
5. selecting blocks manually, randomly, or by BI ranking;
6. fitting linear replacement operators with local activation MSE;
7. inserting several replacements in one shot;
8. optionally applying teacher-logit recovery;
9. evaluating loss and perplexity; and
10. logging and analyzing fixed and BI-prefix search experiments.

The MVP is evidence that the workflow is viable. It is not evidence that the
linear operator, BI score, recovery objective, or search procedure is optimal.

## Preservation policy

- Do not refactor historical notebooks or helper modules in place.
- Correct factual archive errors through a clearly dated erratum.
- Build maintained implementations under the future `src/` and `pipelines/`
  architecture after that architecture is reviewed.
- Treat notebook outputs and `result-overview.md` as historical observations,
  not as source literature or automatically verified thesis claims.
- Cite an experiment log when promoting an MVP observation into the wiki.

## Known reproducibility limitations

- The checkpoint has no dependency lockfile.
- C4, Wikitext2, and the pretrained model are fetched from external services.
- Notebook execution is stateful and some results depend on cell order.
- The logs do not record every software, hardware, seed, and dataset revision.
- The implementation was designed as a prototype and has not yet been
  separated into a maintained package and orchestration layer.
