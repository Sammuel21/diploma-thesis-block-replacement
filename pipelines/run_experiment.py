from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mlp_replacement import (
    experiment_config_from_dict,
    load_model_and_tokenizer,
    run_replacement_experiment,
)
from mlp_replacement.data import build_data_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one MLP replacement experiment")
    parser.add_argument("config", type=Path, help="Path to an experiment JSON configuration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = experiment_config_from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    model, tokenizer = load_model_and_tokenizer(config.model)
    loaders = build_data_loaders(
        tokenizer,
        config.data,
        include_recovery=config.recovery.enabled,
    )
    result = run_replacement_experiment(model, loaders, config)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()

