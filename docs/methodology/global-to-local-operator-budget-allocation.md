# Global-to-Local MLP Operator Budget Allocation

Status: methodology draft. This is a project-proposed allocation framework,
not an empirical finding or a claim of optimality.

The method converts one whole-model parameter-sparsity target into an initial
parameter cap for every eligible MLP replacement. It decides **how much local
capacity is available**, not which replacement operator should be used.

The parameter identities below are standard bookkeeping. The importance
normalization, cap allocator, and unused-budget reconciliation are
project-proposed definitions. MoDeGPT motivates nonuniform allocation under a
global constraint but does not define this exact replacement-cap method.

## Variables

| Symbol | Meaning |
| --- | --- |
| $E,\ell$ | Eligible MLP-block set and one block index |
| $P_0,F_\ell$ | Original whole-model and local MLP parameter counts |
| $P_{\mathrm{fixed}}$ | Parameters outside the eligible allocation set |
| $s_{\mathrm{global}},P^\star$ | Global removed fraction and maximum target model size |
| $B^\star,R^\star$ | Eligible replacement budget and global removal quota |
| $I_\ell,z_\ell$ | Raw and normalized block-importance scores |
| $\tau$ | Allocation temperature controlling budget concentration |
| $a_\ell,q_\ell,R_\ell$ | Removal propensity, normalized share, and assigned local removal |
| $C_{\min,\ell},C_\ell^{(0)},H_\ell$ | Minimum budget, initial cap, and hard ceiling |
| $P_\ell,P_{\mathrm{actual}},s_{\mathrm{actual}}$ | Local use, final model size, and realized sparsity |
| $U$ | Unused parameter budget after local construction |
| $\Delta P_{\ell,k},\widehat{\Delta Q}_{\ell,k},u_{\ell,k}$ | Candidate cost, expected improvement, and marginal utility |

## Procedure

### 1) Convert the global target into an eligible-operator budget

$$
\begin{aligned}
P_{\mathrm{fixed}}
    &= P_0-\sum_{\ell\in E}F_\ell, \\
P^\star
    &= (1-s_{\mathrm{global}})P_0, \\
B^\star
    &= P^\star-P_{\mathrm{fixed}}, \\
R^\star
    &= \sum_{\ell\in E}F_\ell-B^\star.
\end{aligned}
$$

With minimum and maximum local budgets $C_{\min,\ell}$ and $H_\ell$, the
target is feasible only when

$$
\sum_{\ell\in E}C_{\min,\ell}
\le B^\star
\le \sum_{\ell\in E}H_\ell.
$$

Reject or revise an infeasible global target rather than silently changing
the requested sparsity.

### 2) Screen and normalize block importance

Obtain one configurable score $I_\ell$ per block and orient it so that a
larger value always means "protect this block more." BI, residual-aware MLP
influence, or a more expensive ablation score can instantiate this interface.

The reference normalization uses ascending percentile ranks:

$$
z_\ell
=\frac{\operatorname{rank}_{\mathrm{asc}}(I_\ell)-1}
       {\lvert E\rvert-1}.
$$

For one eligible block, set $z_\ell=0$. Rank normalization makes score scales
comparable but discards the magnitude of differences between blocks.

### 3) Allocate the removal quota and produce local caps

$$
\begin{aligned}
a_\ell
    &= \exp\!\left(-\frac{z_\ell}{\tau}\right), \\
q_\ell
    &= \frac{F_\ell a_\ell}{\sum_{j\in E}F_j a_j}, \\
R_\ell
    &= R^\star q_\ell, \\
C_\ell^{(0)}
    &= F_\ell-R_\ell.
\end{aligned}
$$

High-importance blocks receive smaller removal shares and therefore larger
caps. Large $\tau$ approaches uniform local sparsity; smaller $\tau$ produces
more unequal allocation. The $q_\ell$ values sum to one, while local retention
ratios $C_\ell^{(0)}/F_\ell$ do not.

If a proposed cap violates a protected-block or local-budget bound, clamp it
and redistribute the remaining quota over the still-feasible blocks until the
global budget is conserved.

### 4) Construct local operators under the assigned caps

Each block-specific method may choose its operator family and fitting
procedure, subject only to

$$
P_\ell\le C_\ell^{(0)}.
$$

The downstream method may replace the complete MLP or selected internal
components. The allocator remains independent of that choice.

The realized model footprint is

$$
\begin{aligned}
P_{\mathrm{actual}}
    &= P_{\mathrm{fixed}}+\sum_{\ell\in E}P_\ell, \\
s_{\mathrm{actual}}
    &= 1-\frac{P_{\mathrm{actual}}}{P_0}.
\end{aligned}
$$

### 5) Reconcile unused parameters when exact budget matching matters

A local method may use fewer parameters than its cap. Pool the difference:

$$
U=\sum_{\ell\in E}\left(C_\ell^{(0)}-P_\ell\right).
$$

Reconciliation is optional. Candidate expansions can be prioritized by
estimated quality improvement per added parameter:

$$
u_{\ell,k}
=\frac{\widehat{\Delta Q}_{\ell,k}}{\Delta P_{\ell,k}}.
$$

Importance-only redistribution is the simplest baseline. Better policies can
include unmet local approximation error or singleton model-loss improvement.
Never exceed $H_\ell$ or the remaining global budget. If no feasible
candidate can use the remainder, leave it unspent and report the actual
footprint. Perform model-level recovery only after the final architecture is
fixed.

### 6) Evaluate the allocation strategy

Hold the local operator family, fitting data, training procedure, and recovery
budget fixed while changing the allocator. Compare against:

- uniform local sparsity at the same aggregate budget;
- fixed-seed random importance rankings;
- alternative importance estimators through the same allocator;
- cap-only versus reconciled allocation; and
- the fixed $d_{\mathrm{model}}/2$ replacement baseline at a matched actual
  footprint.

Report allocated caps, actual use, cap utilization, unused and reallocated
budget, final model parameters, local approximation error, and model-level
quality.

## Key Considerations

- **Importance is not approximability.** A sensitive block can still be easy
  to approximate, and a low-influence block can require a nonlinear operator.
- **A cap is not actual use.** In cap-only mode the target is a maximum
  footprint; the realized model may be smaller.
- **Unused parameters need an explicit policy.** Reallocate them only when a
  feasible expansion has measurable expected value.
- **Continuous budgets meet discrete operators.** Exact parameter matching
  may be impossible for fixed-width or discrete operator families.
- **Calibration matters.** Importance rankings and local fit can change with
  the calibration distribution.
- **Blocks interact.** Independent local fits do not capture downstream error
  propagation after several replacements.
- **Parameter count is only one axis.** It does not by itself determine
  checkpoint bytes, resident memory, latency, or throughput.
- **MoE requires separate accounting.** Stored and routing-dependent active
  parameters must not be merged into one sparsity number.

## Knowledge-Base Traceability

- [Complete maintained method](../../llm-wiki/wiki/methods/method-global-to-local-operator-budget-allocation.md)
- [Block Importance and MLP screening adaptations](../../llm-wiki/wiki/methods/method-block-importance.md)
- [MoDeGPT global sparsity allocation](../../llm-wiki/wiki/methods/method-modegpt-global-sparsity-allocation.md)
- [Replacement error propagation](../../llm-wiki/wiki/concepts/concept-replacement-error-propagation.md)
- [Compression evaluation framework](evaluation-framework.md)

Registered prior-work motivation: `src-modegpt-2025`, Section 3.3 and
Equations 10-11.
