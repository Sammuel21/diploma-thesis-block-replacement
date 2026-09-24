#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

if (( $# < 1 )); then
    echo "Usage: $0 WORKFLOW [WORKFLOW_ARGUMENTS...]" >&2
    echo "Workflows: compression-baseline, swiglu, swiglu-2" >&2
    exit 2
fi

WORKFLOW_NAME="$1"
shift
STORAGE_CONTRACT="artifact"
case "$WORKFLOW_NAME" in
    compression-baseline)
        WORKFLOW_MODULE="workflows.runs.model.baseline.compression"
        ;;
    swiglu)
        WORKFLOW_MODULE="workflows.runs.model.swiglu.swiglu_initial"
        ;;
    swiglu-2)
        WORKFLOW_MODULE="workflows.runs.model.swiglu.swiglu_2_allocation"
        ;;
    *)
        echo "Unsupported model workflow: $WORKFLOW_NAME" >&2
        exit 2
        ;;
esac

# Existing workflows keep their historical --output contract. A future
# long-running entry sets STORAGE_CONTRACT="directories" in the case above.
if [[ "$STORAGE_CONTRACT" == "artifact" ]]; then
    HAS_OUTPUT=0
    for argument in "$@"; do
        if [[ "$argument" == "--output" || "$argument" == --output=* ]]; then
            HAS_OUTPUT=1
            break
        fi
    done
    if (( HAS_OUTPUT == 0 )); then
        RUN_ID="${MLP_REPLACEMENT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
        set -- "$@" --output "data/results/workflows/model/${WORKFLOW_NAME}/${RUN_ID}.json"
    fi
else
    RUN_ID="${MLP_REPLACEMENT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
    HAS_WORK_DIR=0
    HAS_OUTPUT_DIR=0
    for argument in "$@"; do
        if [[ "$argument" == "--work-dir" || "$argument" == --work-dir=* ]]; then
            HAS_WORK_DIR=1
        fi
        if [[ "$argument" == "--output-dir" || "$argument" == --output-dir=* ]]; then
            HAS_OUTPUT_DIR=1
        fi
    done
    if (( HAS_WORK_DIR == 0 )); then
        WORK_DIR="data/work/${WORKFLOW_NAME}/${RUN_ID}"
        cleanup_work_dir() {
            case "$WORK_DIR" in
                data/work/"$WORKFLOW_NAME"/*) rm -rf -- "$WORK_DIR" ;;
                *)
                    echo "Refusing unsafe temporary cleanup path: $WORK_DIR" >&2
                    ;;
            esac
        }
        trap cleanup_work_dir EXIT
        trap 'exit 143' TERM
        trap 'exit 130' INT
        set -- "$@" --work-dir "$WORK_DIR"
    fi
    if (( HAS_OUTPUT_DIR == 0 )); then
        OUTPUT_DIR="data/results/workflows/model/${WORKFLOW_NAME}/${RUN_ID}"
        set -- "$@" --output-dir "$OUTPUT_DIR"
    fi
fi

PYTHON_BIN="${MLP_REPLACEMENT_PYTHON:-python3}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

echo "Checkout directory: $PWD"
echo "Workflow: $WORKFLOW_NAME"
echo "Module: $WORKFLOW_MODULE"
echo "Python: $PYTHON_BIN"
if [[ "$STORAGE_CONTRACT" == "directories" ]]; then
    if (( HAS_WORK_DIR == 0 )); then
        echo "Temporary workflow directory: $WORK_DIR"
    else
        echo "Temporary workflow directory: forwarded --work-dir"
    fi
    if (( HAS_OUTPUT_DIR == 0 )); then
        echo "Durable output directory: $OUTPUT_DIR"
    else
        echo "Durable output directory: forwarded --output-dir"
    fi
fi

"$PYTHON_BIN" -u -m "$WORKFLOW_MODULE" "$@"
