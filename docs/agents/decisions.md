---
metadata_version: 1
title: Shared Decision Log
type: decision-log
category: agents/decisions
status: active
created: 2026-10-10
modified: 2026-10-10
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---

# Shared Decision Log

This log records settled project decisions so that every agent, in every tool,
works from the same understanding. It is a lightweight equivalent of
architecture decision records: one row per decision, with its date, its
content, and the reason or source.

Methodology decisions that need literature or experiment evidence keep their
full record in the wiki decision pages under `llm-wiki/wiki/research/`. This
log only links to them.

## Rules

- Agents read this log before planning, steering, or implementation work and
  treat entries as binding until the researcher changes them.
- Agents propose new entries when a decision settles, for example at the end
  of a [grilling](grilling.md) session, and add them only with the
  researcher's approval. Confirming a grilling summary counts as that
  approval.
- A reversed decision is not deleted. Add a new row that supersedes it and
  mark the old row as superseded.
- Record only decisions made by the researcher or supervisor, not agent
  recommendations.

## Decisions

| Date | Decision | Reason / source |
| --- | --- | --- |
| 2026-06-10 | The MoE/MoNE chapter is agreed with the supervisor as an addition; its go/no-go is pending. | [Supervisor meeting](../meetings/2026-06-10-thesis-direction-result.md) |
| 2026-07-18 | Footprint-quality trade-offs are the primary evaluation scope; latency and throughput are optional descriptive measurements. | [Wiki decision page](../../llm-wiki/wiki/research/decision-primary-compression-evaluation-scope.md), agreed with the supervisor |
| 2026-10-02 | *Superseded on 2026-10-10 by the SwiGLU-8 family entry.* The homogeneous SwiGLU family is closed; S7-0 replacement-only recovery is the homogeneous control. | [SwiGLU results](../experiments/model/swiglu/swiglu-results.md) |
| 2026-10-10 | Quantization-1 is a baseline control only, not a compression axis combined with replacement. | Researcher decision |
| 2026-10-10 | Agents use the adapted grilling protocol; `grill-with-docs` and its `domain-modeling` skill are not adopted. | Researcher decision; [grilling protocol](grilling.md) |
| 2026-10-10 | Claude Code is the primary agent. The setup stays tool-agnostic: `AGENTS.md` is the single rules file, with no `CLAUDE.md`. | Alignment grilling session |
| 2026-10-10 | Agents do not implement until the researcher gives an explicit go; they then carry out the agreed scope and ask before going beyond it. When unsure, they ask rather than assume. | Alignment grilling session |
| 2026-10-10 | No delegation to sub-agents or smaller models unless the researcher asks. The GPT model-tier tables are to be removed in the refactoring session. | Alignment grilling session |
| 2026-10-10 | Submission is due 2027-05-07. Writing starts this semester; the Diploma Seminar requires one completed chapter. Supervisor meetings are monthly. | Researcher information |
| 2026-10-10 | The thesis is written in English by the researcher. AI is used for fact-checking, short drafts, and rephrasing, not for writing whole sections. | Alignment grilling session |
| 2026-10-10 | The repository goes into the thesis submission appendix as a file and will likely be public, so it must be readable by the opponent. Result files stay out of git and are included in the appendix archive. | Alignment grilling session |
| 2026-10-10 | PERUN is the primary compute environment; the supervisor's machine is optional for simple workflows. Both execution paths stay supported. | Alignment grilling session |
| 2026-10-10 | The refactor aims for smaller code with identical behavior; "no action needed" is an acceptable outcome. It is cleanup only; features for new experiment classes are added separately. | Alignment grilling session |
| 2026-10-10 | Code for finished experiments is kept and must keep producing the same results. Restructuring is allowed; deletion is not. | Alignment grilling session |
| 2026-10-10 | The never-run notebook ports (`swiglu_initial.py`, `swiglu_2_allocation.py`, `baseline/compression.py`) are kept, each marked "never run; the notebook is authoritative." | Alignment grilling session |
| 2026-10-10 | Tests guard maintained-code contracts. Tests that only freeze an old file layout are removed. A green suite is required before work is reported as done. | Alignment grilling session |
| 2026-10-10 | `llm-wiki/` holds sources and method attribution, `docs/` holds experiment reports, and notebooks hold analysis and figures. | Alignment grilling session |
| 2026-10-10 | Agent alignment succeeds when an agent can determine the current state from one or two entry files, each fact has one source of truth, nothing looks current that is not, and instructions are short enough to follow. | Alignment grilling session |
| 2026-10-10 | When a decision-log entry conflicts with another document, agents point out the conflict and ask instead of silently choosing one. | Alignment grilling session |
| 2026-10-10 | The homogeneous SwiGLU family is closed except for SwiGLU-8. S7-0 replacement-only recovery remains the homogeneous recovery recipe and the 1B-token reference. | SwiGLU-8 grilling session; supersedes the 2026-10-02 family entry |
| 2026-10-10 | SwiGLU-8 tests whether local fitting still improves the compressed model after recovery and whether the local fit can be improved, including fitting on 8K-token context. Compute uses a headless runner in `workflows/runs/model/swiglu/`; analysis and plots, including the block-level check, use a load-only `notebooks/model/swiglu/swiglu-8.ipynb`. | SwiGLU-8 grilling session |
| 2026-10-10 | SwiGLU-8 block check: layers 3, 11, and 19 at widths 0.25 and 0.5. Stage 1 compares the current recipe (two seeds); constant and cosine-decay learning rates of 1e-3, 3e-4, and 1e-4; an exact least-squares down-projection refit; and 8K-context calibration pairs. All use at most 64 epochs and early-stopping patience 3. Stage 2 (keep-near-teacher penalty, 786,432 pairs) runs only if overfitting remains. Recipes are judged by single-block KL at 128 and 8K tokens, with NMSE also reported. | SwiGLU-8 grilling session |
| 2026-10-10 | A new fitting recipe is used in SwiGLU-8 model runs only if it reduces single-block damage by more than 5% and by more than the seed-to-seed variation, on average across the six layer-width combinations, without being clearly worse in any. | SwiGLU-8 grilling session |
| 2026-10-10 | SwiGLU-8 model runs compare slice-only replacements (no local fitting) with fitted replacements; only fitting differs. Targets are 20, 30, 40, 50, 60, 70, 75, and 80% eligible-MLP removal. Widths 0.125 and 0.0625 are added to the allocation options; at 20-50% the fitted arm keeps SwiGLU-7's widths. | SwiGLU-8 grilling session |
| 2026-10-10 | SwiGLU-8 recovery uses the S7-0 recipe to exactly 100M tokens, with checkpoints at 0, 10, 25, 50, 75, and 100M, and the full SwiGLU-7 final evaluation plus Quantization-1's 8K KL at 100M. A gap that shrinks to within about 2% means fitting does not matter after recovery; a stable, larger gap means it does; anything else is reported as inconclusive at 100M. | SwiGLU-8 grilling session |
| 2026-10-10 | Only clear 100M results that will serve as thesis headline comparisons are promoted to 1B-token recovery. | SwiGLU-8 grilling session |
| 2026-10-10 | All SwiGLU-8 stages run on PERUN. | SwiGLU-8 grilling session |
| 2026-10-10 | SwiGLU-8's 20-80% runs at 100M are the homogeneous control spectrum for the heterogeneous-operator chapter. | SwiGLU-8 grilling session |
| 2026-10-10 | The SwiGLU-8 fitted arm is run to 100M at all eight targets, including 20-50%, because SwiGLU-7 kept no intermediate checkpoints for the 100M final evaluation. At 20-50% these runs also check that SwiGLU-7's first 100M of monitoring values are reproduced. | SwiGLU-8 grilling session |
| 2026-10-10 | SwiGLU-8 selects neurons once per target, saves the selection in its own artifacts, and builds both the slice-only and fitted arms from it. Equality with SwiGLU-7's selection is checked indirectly through the reproduction check. | SwiGLU-8 grilling session |
| 2026-10-10 | SwiGLU-8 acceptance checks: in Part A, the current recipe must reproduce SwiGLU-7's recorded local-fit error at the same layer and width; in Part B, while the recipe is unchanged, the fitted runs at 20-50% must reproduce SwiGLU-7's monitoring values over the first 100M tokens. Both must match within a tolerance stated in the implementation plan; if either fails, work stops for investigation. | SwiGLU-8 grilling session |
| 2026-10-10 | Quantization-1's 8K KL evaluation moves from its runner into shared code in `src/` and is used by both Quantization-1 and SwiGLU-8. Its behavior must stay identical; the move happens before either experiment runs. | SwiGLU-8 grilling session |
| 2026-10-10 | SwiGLU-8 has no repeat (noise-floor) run. Both arms use seed 21 and the identical ordered recovery token stream, so the comparison is paired; the "about 2%" interpretation threshold is reported as a judgment call. | SwiGLU-8 grilling session |
