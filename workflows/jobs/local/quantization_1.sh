#!/bin/bash
# Direct Linux launcher; scientific choices belong to the versioned config.
set -euo pipefail
REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${MLP_REPLACEMENT_PYTHON:-python3}"
export PYTHONPATH="$REPOSITORY_ROOT/src:$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$REPOSITORY_ROOT"
HAS_WORK_DIR=0
for ARGUMENT in "$@"; do
    case "$ARGUMENT" in --work-dir|--work-dir=*) HAS_WORK_DIR=1;; esac
done
if (( HAS_WORK_DIR )); then
    exec "$PYTHON_BIN" -m workflows.runs.model.baseline.quantization "$@"
fi

# Only launcher-created work is disposable; durable output is always explicit.
WORK_ROOT="$REPOSITORY_ROOT/data/work/quantization-1"
mkdir -p "$WORK_ROOT"
WORK_ROOT="$(realpath "$WORK_ROOT")"
WORK_DIR="$(mktemp -d "$WORK_ROOT/run.XXXXXXXX")"
PYTHON_PID=""
cleanup_work() {
    [[ ! -L "$WORK_DIR" && "$(dirname "$(realpath "$WORK_DIR")")" == "$WORK_ROOT" ]] || return 1
    rm -rf -- "$WORK_DIR"
}
stop_run() {
    if [[ -n "$PYTHON_PID" ]]; then
        kill -TERM "$PYTHON_PID" 2>/dev/null || true
        wait "$PYTHON_PID" 2>/dev/null || true
    fi
    exit "$1"
}
trap cleanup_work EXIT
trap 'stop_run 143' TERM
trap 'stop_run 130' INT
"$PYTHON_BIN" -m workflows.runs.model.baseline.quantization --work-dir "$WORK_DIR" "$@" &
PYTHON_PID=$!
wait "$PYTHON_PID"
