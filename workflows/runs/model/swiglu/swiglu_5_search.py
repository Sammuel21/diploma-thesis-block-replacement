"""CLI entry point for the compute-bounded SwiGLU-5 search."""

from __future__ import annotations

import argparse
from pathlib import Path

from workflows.runs.model._common import load_workflow_config

from ._swiglu_5 import (
    SEARCH_WORKFLOW,
    prepare_search_context,
    run_search,
)


DEFAULT_CONFIG = Path("workflows/configs/model/swiglu/swiglu-5-search.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the bounded SwiGLU-5 global-recovery tournament"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    settings = load_workflow_config(args.config, SEARCH_WORKFLOW)
    context = prepare_search_context(
        settings,
        args.config,
        args.source,
        args.output,
    )
    try:
        run_search(context)
    except BaseException as error:
        context.artifact["status"] = "failed"
        context.artifact["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        context.persist("failed")
        raise
    print(f"Wrote {context.output}", flush=True)


if __name__ == "__main__":
    main()
