---
title: Analyses
status: active
created: 2026-11-12
modified: 2026-11-14
---

# Analyses

Dated, non-binding review snapshots of the project, written for a human
reader. Each analysis lives in its own `YYYY-MM-DD-<topic>/` folder with its
own `README.md` as the entry point. An analysis records findings and
recommendations at one point in time; nothing in it is a decision until it
is entered in the [shared decision log](../agents/decisions.md).

When an analysis has been acted on or replaced, set its README `status` to
`superseded` or `archived` rather than deleting it. Record the writing
model with its version (for example `Fable 5.2`), since later versions may
differ.

| Date | Analysis | Status | Scope | Written by |
| --- | --- | --- | --- | --- |
| 2026-11-12 | [Deep analysis](2026-11-12-deep-analysis/README.md) | draft | Whole project: direction, reporting, future work, code, documentation, session history; open questions in `07-questions.md` | Fable 5.2 |
| 2026-11-13 | [Deep analysis, part 2](2026-11-13-deep-analysis-part-2/README.md) | draft | After the 2026-11-13 meeting: whether the local fit predicts global damage (offline study on stored artifacts), mixture-of-experts prior work and pricing, a design from the annotation alone, what transfers and what is tuning, the reporting model and measured speed, the TorchAO defect; proposed decision rows in `01-meeting-3.md`. Added 2026-11-14: further operator families, local losses, and a matched MoE model (`08-operators-losses-moe-model.md`) | Fable 5.2 |
