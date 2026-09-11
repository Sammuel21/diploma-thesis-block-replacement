---
metadata_version: 1
title: MVP Experiment Index
type: index
category: prototype/mvp
status: archived
created: 2026-07-17
modified: 2026-09-11
authorship:
  created_by: unknown
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
sources:
  notebooks: []
  artifacts:
    - data/mvp/results/logs/mvp_log_1.json
    - data/mvp/results/logs/mvp_log_2.json
    - data/mvp/results/logs/mvp_log_3_search.json
    - data/mvp/results/logs/mvp_log_4_search_high_recovery_budget.json
---

# MVP Experiment Index

This index maps the preserved JSON logs to the prototype experiment rounds.
The JSON files are the evidence; prose summaries are interpretations.

| Log | Date | Recorded scope |
| --- | --- | --- |
| `mvp_log_1.json` | 2026-05-06 | Initial fixed one-shot replacement runs |
| `mvp_log_2.json` | 2026-05-08 | Expanded fixed one-shot comparisons |
| `mvp_log_3_search.json` | 2026-05-08 | BI-prefix search over several values of `k` |
| `mvp_log_4_search_high_recovery_budget.json` | 2026-05-11 | BI-prefix search with the increased recovery budget |

All files are stored in `data/mvp/results/logs/`. Their SHA-256 checksums are
recorded in [manifest.yml](manifest.yml).

The experiment family varied:

- the number of replaced blocks;
- BI ranking direction (`asc` or `desc`);
- fixed versus search-style runs; and
- recovery enabled or disabled.

The search is a prefix search over a BI ranking, not an exhaustive search over
all possible layer subsets. This distinction must be retained in future thesis
descriptions.
