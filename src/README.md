# Maintained Source Package

`mlp_replacement/` is the maintained implementation of the thesis workflow.
The historical MVP remains under `scripts/intro/` and `notebooks/mvp/` and is
not imported by this package.

The package follows a functional-core design. Configuration and result objects
are immutable dataclasses, model mutation is isolated in `surgery.py`, and
experiment orchestration is isolated in `workflows.py`. Stateful experiment
logging and artifact persistence are intentionally deferred.
