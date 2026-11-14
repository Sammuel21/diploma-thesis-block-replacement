---
title: Supervisor Meeting 2026-11-13 - Results-3 Feedback and Next Steps
status: draft
created: 2026-11-13
modified: 2026-11-14
---

# Supervisor meeting, 2026-11-13

**Draft, not yet in `docs/meetings/`.** Written from the researcher's own
digest of the meeting (2026-11-13, in the Part 2 analysis prompt). The
supervisor's words are the researcher's paraphrase. Points marked "to confirm"
were not stated in the digest. Move this file to `docs/meetings/` once it is
correct; the decision rows it mentions are proposals in
[file 1 of the analysis](../01-meeting-3.md), not log entries.

The researcher presented the [Results-3 snapshot](../../../results/2026-11-13-results.md):
the recovery schedule screens (Recovery-1, Recovery-2) and the first trained
heterogeneous models (Heterogeneous-2 Step 1).

## Feedback points

| # | Point | What was said | Where it goes |
| --- | --- | --- | --- |
| 1 | Schedule at 1B and the 3B comparison | With a learning-rate decay, dense INT3 QAT at 1B (8K KL 0.0542, 51.6% saving) is clearly ahead of SwiGLU 20% with INT4 (0.0778, 50.9%). The 3B comparison will not be pursued. | Proposed decision row 1. Stacking-2 and Quantization-3 are not run |
| 2 | Heterogeneous replacement | Discussed: with the knapsack choosing by measured KL, the heterogeneous workflow collapses into pure SwiGLU on the dense model. | Heterogeneous-1 and Heterogeneous-2 Step 1 stand as the result. Whether Step 2 (80% at 1B) is dropped: to confirm |
| 3 | Local fitting as a first filter | The supervisor suggested returning to local fitting on calibration data: for hybrid operators, the local fit could serve as a first, cheap filter for choosing a configuration, instead of the knapsack. The researcher objected that an operator may lose on local MSE and still win on KL at the model output; the supervisor agreed that this can happen. He pointed out that whether the local fit on calibration data is predictive is one of the annotation's own questions. | A study of how well local-fit selection agrees with KL-based selection. Design open; first offline evidence in [file 2 of the analysis](../02-local-fit-as-selector.md) |
| 4 | Mixture-of-experts models | Hybrid workflows may collapse on a dense model and still be useful in MoE or MoNE-style models, where a layer has many experts instead of one MLP. Besides heterogeneous replacement, the SwiGLU workflow and any newly defined method could be run there. | MoE work moves up from a bonus; scope to be designed. Researcher's concern: pricing every expert for the knapsack may be too expensive, and local fitting may help there |
| 5 | Inference speed | As in the previous meeting: a deployment environment such as vLLM, configured so that speed and latency are comparable across experiment classes, would allow a speed frontier (speed against quality) next to the size frontier. | Conflicts with the 2026-10-25 and 2026-11-01 decision rows (no speed comparisons between methods). To resolve; options in [file 6 of the analysis](../06-reporting-model.md) |
| 6 | Other models | Running the defined methodology on several model families, sizes, and dense and MoE models, as a last step and a bonus that shows transferability. | Unchanged in priority (2026-11-07 row); now discussed with the supervisor, which that row left open |
| 7 | Operator families (added 2026-11-14 from the researcher's later note) | Hybrid operators were discussed. So far only zero, mean, linear, plain MLP, and the two linear hybrids were tested; the supervisor encouraged more creative operators (wording from backlog item 4, to confirm). | [Backlog item 4](../../../backlog/backlog-ideas.md); candidates, local losses, and a matched MoE model in [file 8 of the analysis](../08-operators-losses-moe-model.md) |

## Not stated in the digest (to confirm)

| Snapshot question | Status |
| --- | --- |
| Close heterogeneous replacement on dense SmolLM2 without the 80% run at 1B? | To confirm |
| Peak learning-rate screen (1e-4) before more long runs? | To confirm |
| Framing as an evaluation thesis with three results? | To confirm |
| Weekly Friday meetings? | To confirm |
| Date of the next meeting | To confirm |

## Researcher's decision ideas after the meeting

1. The 3B configuration comparison is not run; INT3 with decay wins clearly.
2. Return to local-fitting analyses to test the annotation's hypothesis. Rough
   idea: select by local loss, and compare with the knapsack.
3. MoE workflows: test heterogeneous replacement on one or more MoE models.
4. (Added 2026-11-14.) Widen the operator set and pick by local-fit loss,
   with several losses beside MSE, on dense and MoE models. Use an MoE model
   comparable to SmolLM2-1.7B as the MoE reference; other models afterwards
   if compute remains.
