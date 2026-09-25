---
metadata_version: 1
title: TUKE Perun Workflow Infrastructure
type: architecture
category: infrastructure
status: active
created: 2026-09-11
modified: 2026-09-25
authorship:
  created_by: collaborative
curation:
  status: unreviewed
  reviewed_by: null
  reviewed_on: null
---

# TUKE Perun Workflow Infrastructure

## Purpose

This document is the infrastructure baseline for migrating thesis notebooks
into unattended Python workflows intended for the TUKE Perun supercomputer.
Read it before designing or modifying a Perun-targeted workflow or Slurm job.

It records three different kinds of information:

- behavior stated by the official TUKE Perun documentation;
- project rules adopted for this repository; and
- procedures for validating cluster- or account-specific behavior.

Current project readiness, available capacity, and cumulative use belong in
the [Perun status](perun-status.md). Individual submitted jobs and their
measured allocations belong in the append-only [Perun experiment
log](perun-log.md).

The official documentation was last checked on 2026-09-25. Perun policies and
available software can change, so recheck the linked pages before changing
partitions, resource limits, storage assumptions, or environment setup.

Perun is distinct from the previously used shared RTX 4090 remote environment.
That machine was a directly accessed Linux host. Perun is a Slurm-managed HPC
system.

## Operating model

The expected lifecycle is:

1. Connect to the Perun VPN and then to a login node over SSH.
2. Place the repository, persistent environment, and reusable caches in
   appropriate HOME or PROJECT storage.
3. Inspect the assigned account, QoS, partitions, and storage usage.
4. Submit a batch file from the repository root with `sbatch`.
5. Let Perun's prolog stage the submit directory to job-local Lustre scratch.
6. Run one configured workflow inside the Slurm allocation.
7. Let Perun's epilog synchronize new or modified outputs to a
   `results_job_<job-id>/` directory on persistent storage.
8. Inspect the job record and download compact artifacts for notebook analysis.

Login nodes are the access, setup, transfer, and submission boundary. Expensive
experiment execution belongs in a Slurm allocation.

## Access and data transfer

Perun documents two equivalent login nodes:

- `login01.perun.tuke.sk`
- `login02.perun.tuke.sk`

Remote access requires a valid Perun account, VPN access, and a registered SSH
public key. Never commit a private key, VPN profile, password, access token,
username, or account-specific credential to this repository.

The official data-transfer guide supports `scp`, `rsync`, and SFTP. Prefer Git
for tracked source and configuration, and use `rsync` for large or repeatedly
updated result directories because it transfers only changed data.

## Account, QoS, and partitions

Account and QoS values are project-specific and must not be hardcoded in the
tracked job files. Inspect the values assigned to the current user:

```bash
sacctmgr show user "$USER" withassoc format=account,qos
```

The documented partitions as of 2026-09-11 are:

| Partition | Hardware | Documented wall-time limit | Intended use |
| --- | --- | --- | --- |
| `cpu_short` | CPU nodes | 2 days | Short CPU work |
| `cpu_long` | CPU nodes | 4 days | Long CPU work |
| `gpu_short` | GPU nodes with NVIDIA H200 GPUs | 2 days | Short GPU and AI work |
| `gpu_long` | GPU nodes with NVIDIA H200 GPUs | 4 days | Long GPU and AI work |

The partition guide reports up to eight GPUs per GPU node. That hardware limit
does not establish the allocation, concurrency, or GPU-hour limits of the
researcher's project. Inspect live partition state with:

```bash
sinfo -s
scontrol show partitions
```

Request a realistic wall time. Shorter requests may schedule sooner. Request a
GPU explicitly with `--gres=gpu:N`; joining a GPU partition alone does not
allocate one.

## Slurm submission contract

The maintained Perun files live under `workflows/jobs/perun/`. Their `#SBATCH`
directives establish default resources, while project-specific account and QoS
values are supplied to `sbatch`:

```bash
sbatch --account="$PERUN_ACCOUNT" --qos="$PERUN_QOS" \
  workflows/jobs/perun/smoke.sbatch
```

Command-line Slurm options can override tracked defaults without editing the
job file. Current workflows use one GPU and one Python process per job. Do not
request multiple GPUs unless the invoked Python workflow explicitly implements
distributed execution.

Submit from the repository root. Perun stages the directory from which
`sbatch` is invoked, so submission from `workflows/jobs/perun/` would not stage
the rest of the repository. Use repository-relative configuration and output
paths so the same workflow works in the staged copy.

## Automatic scratch and storage

The maintained jobs explicitly run:

```bash
source .activate_scratch
```

The official example guide says the prolog copies the submit directory into a
per-job Lustre scratch directory and creates `.activate_scratch`. Sourcing the
helper changes the working directory to the staged copy and defines scratch
environment variables. The epilog then synchronizes new or modified results
out of scratch and cleans the job area.

The staging guide says hidden directories such as `.git/` and `.venv/`, Python
cache directories, and existing `.out` or `.err` files are excluded. A Python
environment required by a job must therefore live outside a hidden environment
directory inside the repository.

Perun's current documentation defines `SCRATCH_DIR` as the staged job root,
`TMPDIR` as its temporary directory, and `RESULTS_DIR` as the intended output
directory. It also documents `.rsyncignore` for files that must not be staged
back. Future long workflows therefore place disposable state below
`$TMPDIR/mlp-replacement/`, which the repository-root `.rsyncignore` excludes,
and durable state below `$RESULTS_DIR`. The launcher removes only its exact
per-job temporary subtree; it never removes the scratch root.

Perun documents three general storage roles:

| Storage | Repository use |
| --- | --- |
| HOME | Source, small configuration, personal environments, and small persistent files |
| PROJECT | Shared datasets, reusable caches, and results that project members must retain |
| SCRATCH | Temporary high-throughput job input and output |

Inspect storage usage with `perunfsusage`. Large Hugging Face caches and Conda
package caches can exhaust persistent quotas; keep them outside the repository
and monitor their size.

### Unresolved scratch documentation differences

Two current official pages disagree on details:

- the complete submission guide describes synchronized results under
  `~/results_job_<job-id>/`, while the dedicated scratch guide shows
  `<submit-directory>/results_job_<job-id>/`; and
- the dedicated scratch guide states a seven-day cleanup period, while the
  general storage page states that scratch data is deleted 60 days after last
  access.

These statements may refer to different scratch areas or documentation
versions, but that distinction is not explicit. Until observed on the cluster:

- check both documented result locations after the smoke job;
- record the actual location in this document;
- copy important results to HOME, PROJECT, or the local workstation promptly;
  and
- never treat scratch as persistent storage.

## Python environment

Perun documents both Conda and standard Python virtual environments. For this
GPU research stack, a project-specific Conda environment is a reasonable
starting point. Keep its reproducible specification in the repository, but
keep the installed environment itself in persistent storage outside the staged
repository.

The current job files accept an absolute interpreter path through
`MLP_REPLACEMENT_PYTHON`:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlp-replacement
export MLP_REPLACEMENT_PYTHON="$(command -v python)"
```

The job files request export of the submission environment and invoke that
interpreter directly. This avoids hardcoding a user-specific Conda installation
path in version control. Before submission, verify at least the Python,
PyTorch, CUDA, Transformers, and Datasets imports in the selected environment.

Before relying on an environment for a long job, validate:

- the NVIDIA driver and CUDA versions exposed to jobs;
- compatibility of the selected PyTorch build with the H200 nodes;
- outbound network access from compute nodes; and
- whether Hugging Face models and datasets must be downloaded on a login node
  and consumed from a pre-populated cache.

Record the observed versions and access behavior in the [Perun experiment
log](perun-log.md), then update [Perun status](perun-status.md) before the rest
of a grid is submitted. Authentication tokens, if required, must be
provided through the user environment or an approved secret mechanism and
must never be stored in a JSON configuration or job file.

## Repository workflow requirements

Perun-targeted migrations follow these project boundaries:

- `src/mlp_replacement/` owns reusable scientific operations;
- `workflows/runs/block/` is intended for executable block-study workflows;
- `workflows/runs/model/` is intended for executable model-wide workflows;
- `workflows/jobs/perun/` owns Slurm resource requests and process launch only;
- `workflows/configs/` owns explicit migrated-workflow choices and budgets;
- the root `configs/` directory remains the configuration source for the
  existing generic experiment runner; and
- notebooks remain the explanatory and artifact-analysis frontend.

A Python workflow should be runnable without notebook state, use one explicit
configuration, emit a unique machine-readable artifact, and exit after one
experiment so model, activation, teacher-logit, and CUDA allocator state are
released. A Slurm array may coordinate independent configurations, but it does
not make one workflow distributed.

Do not place scientific selection, fitting, recovery, or evaluation logic in
an `.sbatch` file. Do not claim notebook parity until the migrated Python job
has the same inputs, operations, metrics, and artifact semantics as its source
notebook.

## Artifact contract

The generic runner writes one crash-aware JSON record beneath
`data/results/perun/` in the staged repository. The migrated model-wide
workflows instead write a notebook-compatible science artifact and a sibling
`.run.json` operational sidecar. The sidecar contains the resolved workflow
configuration, environment information, completed-stage summaries, final
artifact path, and failure details when Python starts successfully and later
raises an exception. Neither form overwrites an existing output path.

Existing experiment artifacts beneath `data/results/` are runtime data and are
excluded from Git. A fresh Perun clone therefore does not contain historical
controls or upstream reference artifacts required by optimized-only runs.
Before submission, transfer each prerequisite into the submitted repository
tree or pass its staged path explicitly. For `compression-baseline` and
`swiglu`, `--stage all` can instead regenerate the historical and optimized
artifacts in dependency order. `swiglu-2` always requires a completed optimized
`swiglu` schema-3 artifact. Verify that every prerequisite path is inside the
directory submitted to `sbatch`, unless the path names persistent storage that
is directly visible from compute nodes.

Use a clean checkout for submission. Perun copies the whole submit directory,
including untracked data, so a checkout containing the complete local results
archive wastes staging time and scratch capacity. Transfer only the upstream
artifacts needed by the intended run, and keep previous `results_job_*`
directories outside the submit tree.

Slurm `.out` and `.err` files remain operational logs. They are useful for
diagnosis but do not replace the structured artifact. Future workflows that
produce checkpoints, plot data, or larger tables must give those artifacts
unique paths and document whether they are required for reproducibility or
only convenient for analysis.

### Long-workflow storage contract

SwiGLU-1 through SwiGLU-6 retain their historical artifact contracts. SwiGLU-7
is the first long-running runner to accept `--work-dir`, `--output-dir`, and
optional `--resume`:

- work data is disposable, cannot contain the output directory, and is never
  referenced by durable JSON;
- output data is relocatable and contains relative paths to its own files;
- one verified optimizer checkpoint is retained during normal execution, with
  an old and new generation coexisting only during atomic replacement;
- failure retains the latest checkpoint, while successful finalization removes
  it after the final bundle and result have been validated; and
- activations, teacher outputs, and validation caches stay in CPU/GPU memory
  unless a later scientific design explicitly requires serialization.

The local and Perun launchers call the same Python module. They differ only in
environment setup, resource declarations, and default work/output paths.

## First-session checklist

Before the first job:

```bash
sacctmgr show user "$USER" withassoc format=account,qos
sinfo -s
perunfsusage
```

Then:

1. Confirm VPN, SSH, and repository access.
2. Confirm the active project and its account and QoS values.
3. Create or activate the persistent Python environment.
4. Verify the Python package imports and record the interpreter path.
5. Confirm model and dataset cache access.
6. Change to the repository root.
7. Submit the intended configured workflow.
8. Monitor the job and inspect its final resource record.
9. Locate the synchronized JSON, `.out`, and `.err` artifacts.
10. Download and inspect the JSON before submitting further workloads.

Useful job-management commands are:

```bash
squeue -u "$USER"
scontrol show job <job-id>
sacct -j <job-id> --format=JobID,State,Elapsed,MaxRSS,ReqMem
scancel <job-id>
```

Record an `OUT_OF_MEMORY` or `TIMEOUT` result as a resource-planning outcome;
do not silently rerun with substantially larger resources without preserving
the failed job's configuration and logs.

## Official documentation

- [How to connect](https://wiki.perun.tuke.sk/connect/)
- [VPN access](https://wiki.perun.tuke.sk/vpn_windows/)
- [Creating an SSH key](https://wiki.perun.tuke.sk/perun/ssh/)
- [Data transfer](https://wiki.perun.tuke.sk/data_transfer/)
- [Example Slurm scripts and automatic scratch](https://wiki.perun.tuke.sk/perun/slurm/example/)
- [Available partitions](https://wiki.perun.tuke.sk/slurm/partitions/)
- [Job states and reason codes](https://wiki.perun.tuke.sk/slurm/states/)
- [Dedicated scratch guide](https://wiki.perun.tuke.sk/slurm/scratch/)
- [Storage overview](https://wiki.perun.tuke.sk/perun/System_overview/storage/)
- [`perunfsusage` manual](https://wiki.perun.tuke.sk/env/perunfsusage/)
- [Conda guide](https://wiki.perun.tuke.sk/perun/env/conda/)
- [Python virtual-environment guide](https://wiki.perun.tuke.sk/perun/env/pve/)
