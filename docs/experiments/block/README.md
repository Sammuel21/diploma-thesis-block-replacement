---
metadata_version: 1
title: Block Experiment Workflows
type: index
category: experiments/block
status: active
created: 2026-09-11
modified: 2026-09-11
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
| [Single-block replacement baseline](baseline-testing.md) | How does one correctly sized 50% SwiGLU compare with simple learned and input-independent controls at layer 11? | Artifact present; marked exploratory-unverified |
| [Baseline experiments](baseline-experiments.md) | How do calibration size, SwiGLU width, recovery budget, layer, and operator family affect isolated replacements? | Historical artifact present; notebook and artifact schemas differ |
| [Operator study](operator.md) | How do several replacement architectures behave locally and after singleton model integration across depth? | Historical artifact present |
| [Operator distillation](operator-distillation.md) | Which data, optimization, and initialization choices improve local SwiGLU fitting? | Current schema-5 artifact present |

These are not simultaneous model-compression workflows. Even when a notebook
covers all eligible layers, it inserts and evaluates one replacement at a time.
The model-wide pipelines are documented under
[model experiments](../model/).

For the planned Perun migration, the Python jobs should own model loading,
activation capture, fitting, recovery, evaluation, and artifact writing. The
notebooks should load those artifacts and provide tables and visualizations.
Reusable definitions remain under [methodology](../../methodology/).
