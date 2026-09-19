---
metadata_version: 1
title: Block Experiment Workflows
type: index
category: experiments/block
status: active
created: 2026-09-11
modified: 2026-09-19
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---

# Block Experiment Workflows

This directory explains the isolated-operator and singleton-replacement
notebooks. The notebooks remain the current executable specifications, while
these documents explain each research question, stage order, configuration,
artifact, and methodological status.

| Workflow | Main question | Status |
| --- | --- | --- |
| [Block baselines](baseline/README.md) | What do controlled singleton baselines reveal about calibration, capacity, and recovery? | Historical artifacts present |
| [Operator studies](operator/README.md) | Which replacement architectures and fitting choices work locally and after singleton integration? | Historical artifacts present |

These are not simultaneous model-compression workflows. Even when a notebook
covers all eligible layers, it inserts and evaluates one replacement at a time.
The model-wide pipelines are documented under
[model experiments](../model/).

For the planned Perun migration, the Python jobs should own model loading,
activation capture, fitting, recovery, evaluation, and artifact writing. The
notebooks should load those artifacts and provide tables and visualizations.
Reusable definitions remain under [methodology](../../methodology/).
