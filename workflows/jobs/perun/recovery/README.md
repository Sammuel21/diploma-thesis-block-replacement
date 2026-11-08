# Recovery Perun Launchers

Perun Slurm launchers for recovery-recipe experiments.
[`recovery_1.sbatch`](recovery_1.sbatch) runs Recovery-1's schedule screen; see the
[job README](../README.md#recovery-1-workflow).
[`recovery_2.sbatch`](recovery_2.sbatch) runs Recovery-2's 1B decay-length screen; see the
[job README](../README.md#recovery-2-workflow).

Scientific configuration and computation belong in `workflows/configs/model/recovery/` and
`workflows/runs/model/recovery/`. Follow the [Perun job conventions](../README.md) and
[infrastructure guide](../../../../docs/infrastructure/perun/perun.md) when implementing a
launcher.
