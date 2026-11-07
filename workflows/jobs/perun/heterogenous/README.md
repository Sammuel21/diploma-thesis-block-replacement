# Heterogeneous Operator Perun Launchers

Perun Slurm launchers for heterogeneous operator experiments.
[`heterogeneous_1.sbatch`](heterogeneous_1.sbatch) runs Heterogeneous-1 and
[`heterogeneous_2.sbatch`](heterogeneous_2.sbatch) runs Heterogeneous-2's Step 1; see the
[job README](../README.md#heterogeneous-2-workflow).

Scientific configuration and computation belong in
`workflows/configs/model/heterogenous/` and
`workflows/runs/model/heterogenous/`. Follow the
[Perun job conventions](../README.md) and
[infrastructure guide](../../../../docs/infrastructure/perun/perun.md)
when implementing a launcher.
