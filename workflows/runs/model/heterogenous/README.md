# Heterogeneous Operator Runners

Concrete Python entry points for heterogeneous experiments.
[`heterogeneous_1.py`](heterogeneous_1.py) runs Heterogeneous-1, and
[`heterogeneous_2.py`](heterogeneous_2.py) runs Heterogeneous-2's Step 1, reusing its
assembly and SwiGLU-8's recovery loop.

Scientific choices belong in
[`workflows/configs/model/heterogenous/`](../../../configs/model/heterogenous/README.md).
Reusable fitting, allocation, recovery, and evaluation operations belong in
`src/mlp_replacement/`. See the
[experiment documentation](../../../../docs/experiments/model/heterogenous/README.md)
and [workflow instructions](../../../AGENTS.md).
