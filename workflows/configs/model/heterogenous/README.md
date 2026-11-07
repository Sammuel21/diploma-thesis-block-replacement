# Heterogeneous Operator Configurations

Versioned scientific configurations for heterogeneous experiments.
[`heterogeneous-1.json`](heterogeneous-1.json) configures Heterogeneous-1; its
blocks shared with SwiGLU-8 must stay equal to `swiglu-8.json`, and `recipes`
stays null until the researcher has reviewed the adoption table.
[`heterogeneous-2.json`](heterogeneous-2.json) configures Heterogeneous-2's Step 1: its run
table, reading rules, and pinned assemblies. It points at `heterogeneous-1.json` for
everything else, so it inherits SwiGLU-8's recipe and does not repeat it.

Matching runners belong in
[`workflows/runs/model/heterogenous/`](../../../runs/model/heterogenous/README.md).
See the [experiment documentation](../../../../docs/experiments/model/heterogenous/README.md)
for the class layout. Existing SwiGLU configurations retain their identities
and locations.
