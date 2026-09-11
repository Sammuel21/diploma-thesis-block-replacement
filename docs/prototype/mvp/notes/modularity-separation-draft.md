---
metadata_version: 1
title: High-Level Pipeline Draft
type: research-note
category: prototype/mvp/notes
status: archived
created: 2026-05-06
modified: 2026-09-11
authorship:
  created_by: unknown
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---

# High-level pipeline draft

1. Setup
   load model, tokenizer, loaders, config

2. Screening
   compute block-level signals
   example: BI scores, block MSE probes, activation stats later

3. Selection
   choose target layers from screening outputs
   example: random-k, BI-low-k, BI-high-k, manual

4. Surgery / Replacement
   collect block I/O
   train replacement operators
   insert replacements

5. Recovery
   optional model-level KD repair

6. Evaluation
   model loss/PPL, block MSE logs, compression stats

7. Logging
   store config, targets, metrics, artifacts, notes
