---
title: Heterogeneous-2 - Hybrid Models After Recovery
status: draft
created: 2026-11-04
modified: 2026-11-04
---

# Heterogeneous-2 - Hybrid Models After Recovery

**Status:** outline only. It is designed after
[Heterogeneous-1](heterogeneous-1.md) reports, because its targets and operators
come from those results. The plan below is the 2026-11-04 entry in the
[shared decision log](../../../agents/decisions.md), which wins if the two
disagree. Nothing here is implemented.

## Questions

1. How do hybrids compare with homogeneous SwiGLU on KL, perplexity, and tasks?
2. Which retraining scope works best for hybrids?
3. Does local fitting matter for hybrids, and does it differ from SwiGLU?
4. Which removal targets benefit?

## Planned order

Everything depends on Heterogeneous-1. Where a hybrid equals SwiGLU-8's model at
a target, there is nothing to run: the finding is that no other operator was
worth using there.

```text
P  prepare fitted and no-fit starts per target

After P, in parallel:
  M  100M main comparison (S7-0)
  L  learning rates 1e-5 and 1e-4 to 100M, one target
  F  forced runs, if required

After M and L (the researcher picks the rate), in parallel:
  B  1B: S7-0, S7-1, S7-2 at 20/50/80%
  W  SwiGLU S7-1, S7-2 at 80%

After B and W (the researcher picks the scope):
  N  no-fit arm at 100M and 1B

After N: final tables, charts, report
```

W needs nothing from Heterogeneous-1 unless the learning rate changes: it starts
from SwiGLU-8's existing 80% start.

## Main comparison

100M at targets where the hybrid differs, paired with SwiGLU-8 (same recovery
stream, seed 21, full final evaluation). 1B at 20/50/80%, where the S7-0 run is
the main 1B comparison, against SwiGLU-7's 20% and 50% and SwiGLU-9's 80% models.
A 100M checkpoint saved inside the 1B runs avoids duplicate work.

## Retraining scope

The three scopes from [SwiGLU-7](../swiglu/swiglu-7.md): S7-0 trains only replaced
MLPs, S7-1 the transformer body, and S7-2 all MLPs and norms plus attention LoRA.

**Why it may differ for hybrids.** A zero or mean layer has nothing to train, so
under S7-0 only other replaced layers can compensate. SwiGLU-7's S7-0 lead shrank
with removal, and scope was never tested above 50%, which is where new operators
are most likely to appear.

**Why it runs at 1B.** In SwiGLU-7 the three scopes sit within run-to-run noise at
100M and only separate at 1B
([table](../swiglu/swiglu-7.md#scope-ranking-against-recovery-budget-read-only-check-2026-11-04)).
A 100M comparison would mostly measure noise. This makes it an exception to the
2026-10-10 promotion rule, which needs a clear 100M result first. It should be
mentioned to the supervisor, since it costs about 110 GPU-hours.

Main runs keep S7-0 until scope is chosen, so a hybrid and SwiGLU differ only in
their operators. The researcher chooses from the analysis, with no fixed rule, as
for SwiGLU-7. Differences within measured noise are reported as unresolved
(decision log, 2026-11-02).

**Chart.** One figure per target: SwiGLU in the left panel, the hybrid in the
right, recovery tokens across, monitoring KL up, one line per scope. Use KL, not
monitoring perplexity.

## Learning rate

SwiGLU-4 compared only 1e-5 and 3e-5 for replacements, so nothing above 3e-5 was
tried. One hybrid at one target runs to 100M at 1e-5 and 1e-4 for the whole model,
against the 3e-5 main run. It comes before the 1B runs so they do not spend about
110 GPU-hours on a rate that is then replaced. 100M is still early, so this is a
sanity check.

## No-fit arm

The same test as SwiGLU-8's fitted against slice-only: the same hybrid built twice,
with the same layers, operators, sizes, and tokens, one copy with local fitting and
one without.

| Operator | What "no fit" means |
| --- | --- |
| SwiGLU | Copied teacher neurons |
| Linear + SwiGLU | Copied neurons, linear part starting at zero |
| Linear | Zero or random start |
| Plain MLP, linear + plain MLP | Random start |
| Zero, mean | Nothing to fit |

It runs after scope is chosen, as in SwiGLU, where SwiGLU-7 settled scope before
SwiGLU-8 and SwiGLU-9 tested fitting. It runs at 100M where the hybrid differs and
at 1B for 20/50/80%, because SwiGLU-8 showed that 100M alone cannot conclude this.
If S7-0 wins, the gap compares directly with SwiGLU's 10-24% at 100M and 0.3-2.8%
at 1B. If another scope wins, one S7-0 no-fit run restores that comparison. Report
the gap and SwiGLU-9's head-start tokens.

## Conditional parts

- **Forced runs:** a few degradation-curve points at 100M, about 1.3 GPU-hours
  each. Required if Heterogeneous-1 shows a ranking swap or picks no new operator.
- **90% bonus:** 100M only, under the 2026-10-16 rule. It needs a new SwiGLU
  control with width 256.

## Rough cost

Maximum if the hybrid differs at every target; not a PERUN resource plan.

| Stage | GPU-hours |
| --- | ---: |
| 100M fitted and no-fit, up to 8 targets | about 22 |
| 1B three scopes at 20/50/80% | about 111 |
| 1B SwiGLU S7-1, S7-2 at 80% | about 26 |
| 1B no-fit at 20/50/80% | about 33 |
| Learning-rate check | about 3 |
| **Maximum** | **about 195** |

## To settle when designing

- Saving and loading models with non-SwiGLU layers.
- The start for "no fit" linear operators (zero or random).
- Which degradation-curve points run at 100M.
- Whether a fitted noise twin is needed for hybrids.
