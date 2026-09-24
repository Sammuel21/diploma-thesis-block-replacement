# Local workflow jobs

This directory contains thin launchers for direct Linux hosts such as
darthmachinus. Scientific configuration and execution remain under
`workflows/configs/` and `workflows/runs/`; this directory owns only process
launch and local path defaults.

Run an allow-listed model workflow from any directory inside the checkout:

```bash
bash workflows/jobs/local/run_model.sh swiglu-2
```

Arguments after the workflow name are forwarded unchanged. Set
`MLP_REPLACEMENT_PYTHON` to an absolute interpreter path when `python3` is not
the intended environment. Set `MLP_REPLACEMENT_RUN_ID` to choose the default
output name.

The launcher runs in the foreground. A long run can still use one `nohup`:

```bash
nohup bash workflows/jobs/local/run_model.sh <workflow> [arguments...] \
  > <workflow>.log 2>&1 < /dev/null &
```

The current allow-listed workflows retain their historical `--output`
interfaces. Future long-running workflows use `--work-dir` for disposable
state and `--output-dir` for durable state. Their local defaults are
`data/work/<workflow>/<run-id>` and
`data/results/workflows/model/<workflow>/<run-id>` respectively. The Python
runner owns validation, checkpointing, and resume behavior. The launcher
removes only the default work directory it constructs; a caller-supplied work
directory remains the runner's responsibility. The launcher never implements
scientific work.
