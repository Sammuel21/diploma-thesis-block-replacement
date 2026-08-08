---
id: method-global-to-local-operator-budget-allocation
title: Global-to-Local Operator Budget Allocation
summary: Converts a whole-model parameter-sparsity target into importance-aware per-block replacement-budget caps without choosing local operators.
type: method
status: draft
created: 2026-08-08
updated: 2026-08-08

authorship:
  created_by: collaborative
  contributors:
    - researcher
    - llm

epistemic:
  role: mixed
  confidence: not-assessed
  verification:
    - unverified

scope:
  topics:
    - global-sparsity-allocation
    - importance-screening
    - operator-budget
    - nonuniform-compression
    - budget-reconciliation
  granularities:
    - mlp-block
    - transformer-layer
    - model
    - cross-level
  pipeline_stages:
    - screening
    - selection
    - replacement
    - integration
    - recovery
    - evaluation
    - analysis

sources:
  - source_id: src-modegpt-2025
    locator: "Section 3.3; Equations 10-11"
    relation: motivates

related:
  - "[[method-block-importance]]"
  - "[[method-modegpt-global-sparsity-allocation]]"
  - "[[concept-replacement-error-propagation]]"
  - "[[decision-primary-compression-evaluation-scope]]"
  - "[[concept-moe-parameter-accounting]]"
supersedes: []
superseded_by: []
---

# Global-to-Local Operator Budget Allocation

## Overview

**Project-proposed method.** This method converts one whole-model parameter-
sparsity target into an initial maximum parameter budget for every eligible
MLP replacement. It is a first-stage screener and allocator: it decides how
much capacity each block may initially receive, not which replacement
architecture, fitting algorithm, or recovery procedure should be used.

The fixed method skeleton covers parameter accounting, feasibility, budget
conservation, cap semantics, and actual-footprint reporting. Importance
estimation and downstream operator construction remain replaceable policies.
The method is currently unverified and makes no empirical claim that its
nonuniform allocation outperforms a uniform budget.

## Method Boundary and Contract

The allocator receives:

- a dense model and a declared set of eligible MLP blocks;
- one whole-model parameter-sparsity target;
- one scalar importance value per eligible block;
- score-direction and normalization rules;
- a positive allocation temperature; and
- optional protected-block and local budget bounds.

It returns an initial cap vector $C^{(0)}$ and the accounting needed to verify
that the caps satisfy the global constraint. It does not select an operator.
A downstream block-specific method may replace a complete SwiGLU MLP, replace
only internal components, or use another parameterized construction, provided
that the resulting trainable and stored parameters are counted consistently.

## Global Parameter-Budget Formalization

**Standard parameter-accounting notation.** Let $E$ be the declared set of
eligible MLP blocks, $P_0$ the original whole-model parameter count, and
$F_\ell$ the original parameter count of eligible MLP block $\ell$.
Parameters outside $E$ are fixed for this allocation:

$$
\begin{aligned}
P_{\mathrm{fixed}}
    &= P_0 - \sum_{\ell \in E} F_\ell, \\
P^\star
    &= (1-s_{\mathrm{global}})P_0, \\
B^\star
    &= P^\star-P_{\mathrm{fixed}}, \\
R^\star
    &= \sum_{\ell \in E}F_\ell-B^\star.
\end{aligned}
$$

Here, $s_{\mathrm{global}}$ is the desired fraction of the original
whole-model parameters removed, $P^\star$ is the corresponding maximum final
model size, $B^\star$ is the total budget available to eligible replacements,
and $R^\star$ is the removal quota that must be distributed across them. These
bookkeeping identities require no external citation.

**Standard parameter-accounting notation.** For every block, let
$C_{\min,\ell}$ be its minimum permitted retained budget and $H_\ell$ its
hard retained-budget ceiling:

$$
\begin{aligned}
0
    &\le C_{\min,\ell}
    \le C_\ell^{(0)}
    \le H_\ell
    \le F_\ell,
    \qquad \ell \in E, \\
\sum_{\ell \in E}C_\ell^{(0)}
    &= B^\star.
\end{aligned}
$$

A protected block inside $E$ uses
$C_{\min,\ell}=H_\ell=F_\ell$; a permanently fixed block can instead be
excluded from $E$ and counted in $P_{\mathrm{fixed}}$. The global
target is feasible only when:

$$
\sum_{\ell \in E}C_{\min,\ell}
\le B^\star
\le \sum_{\ell \in E}H_\ell.
$$

An infeasible target must be rejected or explicitly revised rather than
silently changing the requested sparsity.

## Configurable Importance Interface

**Project-proposed interface definition.** The method accepts one raw score
from a configurable importance estimator:

$$
I_\ell = \operatorname{ImportanceEstimator}(\ell).
$$

Before allocation, a direction adapter must ensure that a larger value always
means "protect this block more." $I_\ell$ may come from canonical complete-layer
BI, residual-aware MLP influence, an ablated model-loss signal, or another
declared screener. The allocation skeleton does not assume that any one score
also measures linear approximability or determines the local operator family.

**Project-proposed reference normalization.** For $n=\lvert E\rvert>1$, the
default reference rule converts scores to ascending percentile ranks:

$$
z_\ell
= \frac{\operatorname{rank}_{\mathrm{asc}}(I_\ell)-1}{n-1}.
$$

Ties receive their average rank. For one eligible block, define $z_\ell=0$.
This normalization is invariant to score units and preserves ordering, but it
deliberately discards the magnitude of differences between raw scores.
Alternative normalization rules remain configurable and must be reported.

## Reference Cap Allocation

**Project-proposed reference allocation.** For temperature $\tau>0$, define
a positive removal propensity and its parameter-size-weighted share:

$$
\begin{aligned}
a_\ell
    &= \exp\!\left(-\frac{z_\ell}{\tau}\right), \\
q_\ell
    &= \frac{F_\ell a_\ell}
            {\sum_{j \in E}F_j a_j}.
\end{aligned}
$$

$\exp(x)=e^x$ is the exponential function. The negative sign gives a
high-importance block a smaller removal propensity. When all $F_\ell$ are
equal, $\mathbf{q}=\operatorname{softmax}(-\mathbf{z}/\tau)$; with unequal
sizes it is a size-weighted softmax. The $q_\ell$ values, not the local
retention ratios, sum to one.

**Project-proposed reference allocation (continued).** Without active bounds,
assign removals and initial caps as:

$$
\begin{aligned}
R_\ell
    &= R^\star q_\ell, \\
C_\ell^{(0)}
    &= F_\ell-R_\ell, \\
\rho_\ell
    &= \frac{C_\ell^{(0)}}{F_\ell}.
\end{aligned}
$$

$\rho_\ell$ is a local retention-cap ratio. Since a downstream replacement
may use fewer than $C_\ell^{(0)}$ parameters, $1-\rho_\ell$ is a minimum
assigned local sparsity, not necessarily the realized sparsity. Neither
$\rho_\ell$ nor the realized local sparsities are probability weights, and
they need not sum to one. If earlier notation writes
$C_\ell^{(0)}=w_\ell(F_\ell)$, $w_\ell$
denotes this derived local mapping; $q_\ell$ is the separately normalized
global share.

Temperature controls concentration. Large $\tau$ approaches uniform local
sparsity, while smaller values direct more removal toward lower-importance
blocks. No numerical temperature is currently claimed to be optimal.

### Bound-Aware Redistribution

**Standard parameter-accounting notation.** The retained-budget interval
corresponds to removal bounds:

$$
\begin{aligned}
R_{\min,\ell}
    &= F_\ell-H_\ell, \\
R_{\max,\ell}
    &= F_\ell-C_{\min,\ell}.
\end{aligned}
$$

If an unconstrained $R_\ell$ violates a bound, clamp it to the violated boundary,
remove that block from the free set, subtract its fixed removal from the
remaining quota, and recompute the weighted shares over the remaining blocks.
Repeat until every block is feasible and the total removal remains $R^\star$.
This active-set or water-filling procedure produces bounded caps without
discarding or inventing global budget.

## Fixed Skeleton and Configurable Policies

| Component | Fixed contract | Configurable choice |
| --- | --- | --- |
| Global target | Whole-model parameter count and exact feasibility accounting | Target sparsity and eligible block set |
| Importance | One direction-corrected scalar per block | BI, residual-aware MLP influence, ablation score, or another estimator |
| Normalization | Report the transformation from $I_\ell$ to $z_\ell$ | Rank normalization is the reference; alternatives may preserve magnitude |
| Allocation | Conserve $B^\star$ and respect hard bounds | Temperature and a declared alternative allocator |
| Local construction | Return actual parameter use no greater than the assigned cap | Operator family, internal MLP surgery, fitting loss, and training procedure |
| Reconciliation | Never exceed the global budget or hard local ceilings | Need or marginal-utility estimator and candidate increments |
| Recovery | Report its data and optimization budget separately | Distillation, supervised recovery, or no recovery |

This separation permits importance estimators and local compression methods to
be studied independently. In particular, failure of BI to predict one
operator family's approximation error would not invalidate the budget-
allocation interface; it would challenge that score as an allocation input.

## Local Construction and Realized Sparsity

**Project-proposed downstream contract.** A local construction method receives
$C_\ell^{(0)}$ and returns a replacement $g_\ell$ with actual parameter count
$P_\ell$:

$$
\begin{aligned}
P_\ell
    &\le C_\ell^{(0)}, \\
P_{\mathrm{actual}}
    &= P_{\mathrm{fixed}}+\sum_{\ell \in E}P_\ell, \\
s_{\mathrm{actual}}
    &= 1-\frac{P_{\mathrm{actual}}}{P_0}.
\end{aligned}
$$

In cap-only mode, $P_{\mathrm{actual}}\le P^\star$, so the final model may be
smaller than the nominal target and
$s_{\mathrm{actual}}\ge s_{\mathrm{global}}$. The global target is therefore
a hard maximum footprint, not a guarantee that downstream discrete operator
families will consume every available parameter.

## Optional Budget Reconciliation and Recovery

**Project-proposed accounting definition.** After local construction, pool the
unused assigned capacity:

$$
U=\sum_{\ell \in E}\left(C_\ell^{(0)}-P_\ell\right).
$$

Reconciliation is optional and is intended for near-exact matched-budget
experiments. The initial cap is provisional; $H_\ell$ remains the
non-negotiable hard ceiling. A local method may expose feasible expansion
candidate $k$ with additional parameter cost $\Delta P_{\ell,k}$ and estimated
quality improvement $\widehat{\Delta Q}_{\ell,k}$.

**Project-proposed prioritization interface.** For candidate $k$ at block
$\ell$, define marginal expected benefit per added parameter:

$$
u_{\ell,k}
= \frac{\widehat{\Delta Q}_{\ell,k}}{\Delta P_{\ell,k}}.
$$

The reconciliation stage repeatedly selects the highest-priority feasible
increment, revises the affected cap without exceeding $H_\ell$, and subtracts
the increment from $U$. It stops when the pool is empty or no candidate fits.
Any unspendable remainder stays unused and must be reported; parameters must
not be added merely to match a nominal count.

Candidate priority is a research variable:

- importance-only redistribution is the simplest baseline;
- held-out local-error improvement asks whether added capacity improves the
  isolated approximation;
- importance-weighted local improvement combines sensitivity and unmet local
  approximation need; and
- singleton model-loss improvement is more directly model-facing but requires
  substantially more evaluation.

Budget reconciliation changes architecture or local capacity. Model-level
fine-tuning does not itself consume a parameter surplus, so recovery occurs
only after the final architecture and reconciled caps have been chosen.

## Evaluation Protocol

To isolate the allocator, hold the downstream operator family, fitting data,
training procedure, and recovery budget fixed while changing only the score
or allocation policy. Compare:

- uniform local sparsity at the same aggregate parameter budget;
- fixed-seed random permutations of importance ranks;
- each candidate importance estimator passed through the same allocator;
- cap-only and reconciled variants; and
- the fixed $d_{\mathrm{model}}/2$ replacement baseline at a matched aggregate
  footprint.

If discrete operator widths prevent exact matching, report the mismatch and
compare actual rather than nominal parameter counts. Each run should report:

- $P_0$, $P^\star$, $B^\star$, and $R^\star$;
- every $I_\ell$, $z_\ell$, $q_\ell$, bound, and initial cap;
- actual local parameter use and cap-utilization ratio;
- pooled, reallocated, and unspent budget;
- final whole-model parameter count and realized sparsity;
- held-out local approximation measurements; and
- model-level loss, perplexity, and the declared quality evaluations.

The allocator is useful only if it improves the footprint-quality trade-off
relative to matched-budget baselines. A favorable allocation at one nominal
budget does not establish optimality.

## Evidence and Rationale

MoDeGPT is a source-checked precedent for converting layer-importance scores
into nonuniform sparsities under a global compression constraint. Its
source-derived optimization and softmax solution are documented separately in
[[method-modegpt-global-sparsity-allocation]]. [src-modegpt-2025, Section 3.3;
Equations 10-11]

**Synthesis.** That precedent supports separating model-level allocation from
local compression realization. The project method extends the separation to
replacement-budget caps, configurable importance estimators, actual cap
utilization, and optional leftover reconciliation. These additions are
project proposals, not claims attributed to MoDeGPT.

No result from the exploratory importance notebook is promoted here as an
empirical finding. Any future claim about correlation, predictive validity, or
allocation performance requires a preserved experiment configuration and
result artifact.

## Limitations and Open Issues

Importance is not the same as approximability. A block may strongly influence
the residual stream while remaining easy to approximate, or show modest
directional influence while requiring a complex substitute.

Rank normalization makes heterogeneous scores easy to pass into one allocator
but discards score magnitude. The appropriate temperature, normalization, and
local bounds may depend on the model, calibration distribution, global target,
and downstream operator family.

Independent cap assignment does not model how errors from several replacements
interact or shift downstream inputs. Local fit and marginal-utility estimates
must therefore be validated at model level, especially after reconciliation.

Continuous caps may not be exactly realizable by discrete widths or operator
families. Cap-only runs can exceed the requested sparsity; reconciled runs can
still leave an unspendable remainder. Both cases require actual-footprint
reporting.

Parameter count alone does not determine checkpoint bytes, resident memory,
latency, or throughput. Quantization and systems effects require their own
accounting rather than being folded silently into $s_{\mathrm{global}}$.

The core formalization treats one dense MLP per Transformer layer as the
allocation unit. A mixture-of-experts extension must first specify whether
the target concerns total stored expert parameters, routing-dependent active
parameters, or both. Expert-level importance and routing behavior are
additional variables rather than automatic consequences of this allocator.

## Relationships

- [Global-to-local MLP operator budget allocation](../../../docs/methodology/global-to-local-operator-budget-allocation.md)
  provides the compact thesis-facing version of this method.
- [[method-block-importance]] defines source-derived and project-adapted
  importance candidates that may instantiate the configurable $I_\ell$ input.
- [[method-modegpt-global-sparsity-allocation]] documents the prior-work
  precedent and its distinct direct layer-sparsity formulation.
- [[concept-replacement-error-propagation]] explains why locally assigned caps
  and errors require integrated model-level evaluation.
- [[decision-primary-compression-evaluation-scope]] requires final comparisons
  to use actual footprint and quality rather than nominal allocation alone.
- [[concept-moe-parameter-accounting]] defines the additional stored-versus-
  active accounting required before extending the method to MoE models.

## Sources

- `src-modegpt-2025` - Section 3.3 and Equations 10-11
