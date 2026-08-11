---
id: method-hybrid-operator-replacement
title: Hybrid Linear-Nonlinear MLP Replacement
summary: Defines a project-proposed drop-in MLP replacement that adds a linear branch to a compact nonlinear correction while leaving capacity allocation as future work.
type: method
status: draft
created: 2026-08-09
updated: 2026-08-11

authorship:
  created_by: collaborative
  contributors:
    - researcher
    - llm

epistemic:
  role: hypothesis
  confidence: not-assessed
  verification:
    - unverified

scope:
  topics:
    - mlp-block-replacement
    - hybrid-operator
    - low-rank-linear
    - nonlinear-residual
    - operator-budget
  granularities:
    - mlp-block
    - transformer-layer
    - model
  pipeline_stages:
    - replacement
    - integration
    - recovery
    - evaluation
    - analysis

sources: []
related:
  - "[[experiment-initial-block-compression-study]]"
  - "[[experiment-swiglu-operator-design-progression]]"
  - "[[method-global-to-local-operator-budget-allocation]]"
supersedes: []
superseded_by: []
---

# Hybrid Linear-Nonlinear MLP Replacement

## Overview

**Project-proposed method.** A hybrid operator is a drop-in replacement for
one dense Transformer MLP that combines a linear approximation with a compact
nonlinear correction. The stable core idea is the additive decomposition; the
choice of branch architectures, training schedule, and internal capacity split
remain experimental variables.

This object belongs under `methods/` because it defines a reusable replacement
architecture and construction procedure. It is not yet a finding: no preserved
experiment currently shows that the hybrid outperforms a purely linear or
purely nonlinear replacement.

## Operator Definition

**Project-proposed operator definition.** For normalized MLP input $x$ at
block $\ell$, define:

$$
\widehat{f}_\ell(x)=L_\ell(x)+G_\ell(x).
$$

Both branches accept and return the model width $d=d_{\mathrm{model}}$, so
their sum can replace the original MLP contribution without changing the
surrounding residual interface.

- $L_\ell$ represents behavior that can be modeled linearly.
- $G_\ell$ is a compact nonlinear branch intended to correct error left by
  $L_\ell$.

This decomposition is an architectural hypothesis rather than a claim that
the teacher MLP has one unique or identifiable linear and nonlinear split. The
equation is project-proposed and requires no external citation.

## Branch Choices and Parameter Accounting

The linear branch has two main forms:

$$
L_\ell(x)=A_\ell x
$$

or

$$
L_\ell(x)=U_\ell(V_\ell x).
$$

A dense $A_\ell$ has fixed shape $d\times d$ and no configurable hidden
width. The factorized form introduces rank $r_L$ through
$V_\ell:\mathbb{R}^{d}\rightarrow\mathbb{R}^{r_L}$ and
$U_\ell:\mathbb{R}^{r_L}\rightarrow\mathbb{R}^{d}$.

The nonlinear branch may be an ungated compact MLP,

$$
G_\ell(x)=W_{2,\ell}\,\phi(W_{1,\ell}x),
$$

or a small SwiGLU whose intermediate width is $r_G$. A small linear branch
plus a small SwiGLU is therefore one concrete member of the hybrid family, not
a separate top-level method.

**Standard parameter-counting notation.** Ignoring biases, the branch costs
are:

| Branch | Configurable capacity | Parameters |
| --- | --- | ---: |
| Dense linear | none beyond fixed model width | $d^2$ |
| Factorized linear | rank $r_L$ | $2dr_L$ |
| Ungated nonlinear MLP | hidden width $r_G$ | $2dr_G$ |
| SwiGLU nonlinear branch | intermediate width $r_G$ | $3dr_G$ |

These counts follow directly from matrix dimensions and require no external
citation. Exact stored state must additionally include any configured biases
or buffers.

## Candidate Construction Procedure

**Project-proposed reference procedure.** A first controlled implementation
may use the following stages:

1. Capture teacher MLP input-output pairs from the frozen dense model.
2. Fit the selected linear branch against the teacher output.
3. Form residual targets $y-L_\ell(x)$ and fit the nonlinear branch against
   those residuals.
4. Optionally fine-tune both branches jointly against the original teacher
   output.
5. Insert the fitted hybrid into one block and evaluate complete-model loss
   and perplexity before any model-level recovery.
6. In a later recovery comparison, give every candidate the same model-level
   distillation data, objective, trainable scope, and optimization budget.

The staged fit gives the nonlinear branch an explicit correction target, but
it is not yet established as superior to joint training from initialization.
That comparison remains an experimental control.

## Initial Baseline Experiments

The smallest useful hybrid study should compare:

- the original teacher MLP;
- the applicable zero and mean controls;
- the linear branch alone;
- the nonlinear branch alone;
- their hybrid sum; and
- the fixed reduced-width SwiGLU reference from
  [[experiment-initial-block-compression-study]].

Pure-branch and hybrid candidates must use the same activation pairs and
model-validation batches. Exact parameter counts and theoretical weight bytes
must be reported. A fixed reference at a different footprint remains a useful
anchor, but a matched-budget conclusion requires candidates with equal or
explicitly reconciled actual footprints.

The primary local metric is held-out relative MSE, supported by cosine
similarity and token-error summaries. The model-facing measurements are
singleton loss and perplexity changes. Local improvement alone does not
establish model-level superiority.

## Pending Internal Capacity Allocation

**Researcher direction; unresolved future work.** A future extension may give
block $\ell$ a total replacement cap $B_\ell$ and divide its capacity between
linear rank $r_L$ and nonlinear width $r_G$. The intended idea is to control
the complexity of the two branches under one total footprint, not to multiply
their outputs by mixture coefficients.

No allocation rule is currently selected. In particular, the project has not
decided:

- whether the linear branch should be dense or factorized;
- whether the nonlinear branch should be an ungated MLP or SwiGLU;
- whether $B_\ell$ is an exact target or only a maximum cap;
- how continuous shares should be projected to feasible integer ranks and
  widths;
- whether a small grid, a local approximation signal, or a model-level signal
  should select the split; or
- how unused capacity should be treated when no exact configuration exists.

[[method-global-to-local-operator-budget-allocation]] may eventually provide
the total block cap $B_\ell$. It deliberately does not decide the internal
linear/nonlinear split. Likewise, BI or another block-importance score may
influence the total cap but does not by itself measure the required nonlinear
capacity.

The allocation direction must remain labelled pending until a concrete rule,
controls, and evaluation protocol are chosen. It must not be described as an
optimal hybrid construction before such evidence exists.

## Candidate Research Questions and Hypotheses

**Researcher hypotheses.** Suitable falsifiable questions include:

1. At matched actual parameter count, does a hybrid reduce held-out activation
   error more than either pure branch?
2. Does the hybrid also reduce singleton model-loss degradation, or is its
   local advantage downstream-irrelevant?
3. Do blocks with worse full-dense linear approximation receive more benefit
   from adding nonlinear capacity?
4. Does residual-first branch fitting outperform direct joint training under
   the same optimization budget?

These propositions are unverified and require preserved experiment artifacts
before any answer becomes an empirical finding.

## Limitations and Open Questions

- The decomposition is not identifiable: a nonlinear branch may also learn
  linear behavior, and the two branches may compete or duplicate capacity.
- A low-rank factorization changes both parameter count and attainable linear
  rank, so its error cannot be interpreted as pure nonlinearity.
- Dense linear and compact nonlinear branches have different feasible
  parameter grids; approximate matching must use actual rather than nominal
  counts.
- Calibration-set activation error may not reflect sensitive downstream
  directions or shifted inputs after several replacements.
- A candidate that improves before recovery may not retain its advantage after
  equal model-level distillation, and the reverse is also possible.
- No branch family, rank, width, allocation rule, or training schedule is
  currently claimed to be optimal.

## Repository Status

- [`operator.ipynb`](../../../notebooks/block/operator.ipynb) records the
  operator-family design space.
- [`budget.ipynb`](../../../notebooks/analyses/budget.ipynb) records the
  internal allocation direction as future work.
- [`HybridReplacement`](../../../src/mlp_replacement/operators/modules.py) is
  an exploratory low-rank-linear plus compact-nonlinear implementation. Its
  presence does not verify the method or settle the allocation design.

## Relationships

- [[experiment-initial-block-compression-study]] provides the fixed
  single-block reference and evaluation measurements against which an initial
  hybrid should be compared.
- [[experiment-swiglu-operator-design-progression]] places the hybrid between
  generic whole-MLP screening and later teacher-tailored branch allocation.
- [[method-global-to-local-operator-budget-allocation]] defines an upstream
  block-cap interface; the unresolved internal hybrid split is a separate
  downstream decision.

## Sources

No external source is cited. This page formalizes a researcher-proposed
operator direction from current project notebooks and discussion; it does not
claim literature priority or empirical effectiveness.
