"""CLI entry point for one gated SwiGLU-5 100M confirmation."""

from __future__ import annotations

import argparse
from pathlib import Path

from workflows.runs.model._common import load_workflow_config

from ._swiglu_5 import (
    CONFIRMATION_WORKFLOW,
    prepare_confirmation_context,
    run_confirmation,
)


DEFAULT_CONFIG = Path("workflows/configs/model/swiglu/swiglu-5-confirmation.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continue one selected SwiGLU-5 endpoint to 100M tokens"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--search-artifact", type=Path, required=True)
    parser.add_argument("--target", type=float, required=True, choices=(0.2, 0.5))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = load_workflow_config(args.config, CONFIRMATION_WORKFLOW)
    context = prepare_confirmation_context(
        settings,
        args.config,
        args.search_artifact,
        args.target,
        args.output,
        args.resume,
    )
    try:
        run_confirmation(context)
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
