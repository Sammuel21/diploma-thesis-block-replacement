import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from mlp_replacement.config import experiment_config_from_dict
from mlp_replacement.data import build_data_loaders
from mlp_replacement.model import load_model_and_tokenizer
from mlp_replacement.runlog import ExperimentLog
from mlp_replacement.compression.workflows import run_replacement_experiment


def parse_args():
    """Read the JSON configuration path supplied to the pipeline."""

    parser = argparse.ArgumentParser(description="Run one MLP replacement experiment")
    parser.add_argument("config", type=Path, help="Path to an experiment JSON configuration")
    parser.add_argument("--output", type=Path, help="Single JSON run-log path")
    return parser.parse_args()


def default_output_path(config_path):
    """Create a unique default result path from the configuration filename."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path("data/results") / f"{config_path.stem}-{timestamp}.json"


def main():
    """Load resources and execute one complete replacement experiment."""

    args = parse_args()
    config = experiment_config_from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    run_log = ExperimentLog(args.output or default_output_path(args.config), config.to_dict())

    try:
        run_log.begin("load_model_and_tokenizer")
        model, tokenizer = load_model_and_tokenizer(config.model)
        parameter = next(model.parameters())
        run_log.record("resources", {
            "model_class": type(model).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "resolved_model_revision": getattr(model.config, "_commit_hash", None),
            "resolved_tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
            "device": str(parameter.device),
            "dtype": str(parameter.dtype),
        })

        run_log.begin("build_data_loaders")
        loaders = build_data_loaders(
            tokenizer,
            config.data,
            include_recovery=config.recovery.enabled,
        )
        run_log.record("data_roles", {
            "operator_training_batches": len(loaders.calibration),
            "operator_validation_batches": len(loaders.operator_validation),
            "recovery_training_batches": len(loaders.recovery) if loaders.recovery is not None else 0,
            "recovery_validation_batches": (
                len(loaders.recovery_validation) if loaders.recovery_validation is not None else 0
            ),
            "model_validation_batches": len(loaders.model_validation),
            "final_test_batches": len(loaders.test) if loaders.test is not None else 0,
        })

        result = run_replacement_experiment(model, loaders, config, run_log=run_log)
        run_log.complete(result)
    except BaseException as error:
        run_log.fail(error)
        raise

    print(f"Completed experiment: {run_log.path}")


if __name__ == "__main__":
    main()
