"""Headless migration of ``notebooks/model/baseline/compression-baseline.ipynb``."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from workflows.runs.model.common import (
    default_artifact_path,
    json_records,
    load_artifact,
    load_workflow_config,
    release_cuda,
    require_new_path,
    resolve_path,
    start_run_log,
    write_artifact,
)

from mlp_replacement.compression.workflows import run_replacement_experiment
from mlp_replacement.config import (
    CaptureConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OperatorConfig,
    RecoveryConfig,
    SelectionConfig,
    WorkflowConfig,
)
from mlp_replacement.data import (
    DataLoaders,
    build_data_loaders,
    contiguous_token_windows,
    load_text_dataset,
    make_token_loader,
    sample_partitioned_windows,
)
from mlp_replacement.evaluation.footprint import parameter_footprint
from mlp_replacement.evaluation.language_model import evaluate_language_model
from mlp_replacement.model import load_model_and_tokenizer
from mlp_replacement.runlog import environment_record, json_value


WORKFLOW = "compression-baseline"
SCHEMA_HISTORICAL = 1
SCHEMA_OPTIMIZED = 2
DEFAULT_HISTORICAL_REFERENCE = Path(
    "data/results/model-compression-baselines/compression-baseline.json"
)
DEFAULT_CONFIG = Path("workflows/configs/model/baseline/compression.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the model-wide fixed-allocation SwiGLU baseline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Explicit notebook-equivalent workflow configuration",
    )
    parser.add_argument(
        "--stage",
        choices=("historical", "optimized", "all"),
        default="optimized",
        help="Notebook section to execute (default: optimized)",
    )
    parser.add_argument(
        "--historical-artifact",
        type=Path,
        default=DEFAULT_HISTORICAL_REFERENCE,
        help="Schema-1 control used by the optimized stage",
    )
    parser.add_argument(
        "--historical-output",
        type=Path,
        help="Output for a historical/all run; must not already exist",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output for the optimized stage; must not already exist",
    )
    return parser.parse_args()


def base_configs(settings: dict, model_revision: str | None = None):
    data_values = settings["data"]
    operator_values = settings["operator"]
    recovery_values = settings["recovery"]
    selection_values = settings["selection"]
    seed = int(settings["seed"])
    model_revision = model_revision or settings["model_revision"]
    model = ModelConfig(
        model_id="HuggingFaceTB/SmolLM2-1.7B",
        revision=model_revision,
        tokenizer_revision=model_revision,
        device="auto",
        dtype="auto",
    )
    data = DataConfig(
        sequence_length=int(data_values["sequence_length"]),
        batch_size=int(data_values["batch_size"]),
        num_calibration_batches=int(data_values["historical_calibration_batches"]),
        num_operator_validation_batches=int(data_values["operator_validation_batches"]),
        num_recovery_batches=int(data_values["recovery_batches"]),
        num_recovery_validation_batches=int(data_values["recovery_validation_batches"]),
        num_model_validation_batches=int(data_values["model_validation_batches"]),
        num_test_batches=int(data_values["test_batches"]),
        seed=seed,
    )
    capture = CaptureConfig(**settings["capture"])
    operator = OperatorConfig(
        kind=operator_values["kind"],
        intermediate_ratio=float(operator_values["intermediate_ratio"]),
        bias=bool(operator_values["bias"]),
        epochs=int(operator_values["epochs"]),
        learning_rate=float(operator_values["learning_rate"]),
        batch_size=int(operator_values["batch_size"]),
        weight_decay=float(operator_values["weight_decay"]),
        scheduler=operator_values["scheduler"],
        early_stopping_patience=operator_values["early_stopping_patience"],
        seed=seed,
    )
    recovery = RecoveryConfig(**recovery_values)
    selections = {
        name: SelectionConfig(
            strategy="interleaved",
            interleave_stride=int(selection_values["interleave_stride"]),
            interleave_offset=int(selection_values["interleave_offset"]),
            protected_prefix=int(boundaries["protected_prefix"]),
            protected_suffix=int(boundaries["protected_suffix"]),
            application_order="layer",
            seed=seed,
        )
        for name, boundaries in selection_values["variants"].items()
    }
    workflow = WorkflowConfig(strategy="one_shot")
    experiments = {
        name: ExperimentConfig(
            model=model,
            data=data,
            capture=capture,
            selection=selection,
            operator=operator,
            recovery=recovery,
            workflow=workflow,
        )
        for name, selection in selections.items()
    }
    return model, data, capture, operator, recovery, selections, workflow, experiments


def result_frames(results, dense_row):
    evaluation_rows = []
    block_rows = []
    selection_rows = []
    for variant, result in results.items():
        removed = result.footprint_before.parameters - result.footprint_after.parameters
        reduction = 100 * removed / result.footprint_before.parameters
        selection_rows.append(
            {
                "variant": variant,
                "eligible_blocks": len(result.selection.eligible_indices),
                "selected_blocks": len(result.selection.indices),
                "selected_layers": list(result.selection.indices),
            }
        )
        for stage, metrics, teacher_kl in (
            (
                "before_recovery",
                result.pre_recovery_validation_metrics,
                result.pre_recovery_validation_kl,
            ),
            (
                "after_recovery",
                result.final_validation_metrics,
                result.post_recovery_validation_kl,
            ),
        ):
            evaluation_rows.append(
                {
                    "variant": variant,
                    "stage": stage,
                    "selected_layers": list(result.selection.indices),
                    "parameters": result.footprint_after.parameters,
                    "trainable_parameters": result.footprint_after.trainable_parameters,
                    "theoretical_weight_bytes": result.footprint_after.theoretical_weight_bytes,
                    "removed_parameters": removed,
                    "parameter_reduction_pct": reduction,
                    "teacher_kl": teacher_kl,
                    "loss": metrics.loss,
                    "perplexity": metrics.perplexity,
                    "predicted_tokens": metrics.predicted_tokens,
                    "batches": metrics.batches,
                }
            )
        for block in result.blocks:
            block_rows.append(
                {
                    "variant": variant,
                    "initialization": block.operator_initialization,
                    "layer": block.layer_index,
                    "original_parameters": block.original_parameters,
                    "replacement_parameters": block.replacement_parameters,
                    "retained_pct": 100
                    * block.replacement_parameters
                    / block.original_parameters,
                    "validation_mse": block.operator_validation_mse,
                    "validation_nmse": block.operator_validation_nmse,
                    "validation_cosine": block.operator_validation_cosine,
                    "best_epoch": block.best_operator_epoch,
                }
            )
    evaluations = pd.concat(
        [pd.DataFrame([dense_row]), pd.DataFrame(evaluation_rows)], ignore_index=True
    )
    return evaluations, pd.DataFrame(selection_rows), pd.DataFrame(block_rows)


def run_historical(output: Path, settings: dict) -> dict:
    output = require_new_path(output)
    (
        model_config,
        data_config,
        capture,
        unused_operator,
        recovery,
        unused_selections,
        unused_workflow,
        experiments,
    ) = base_configs(settings)
    variants = tuple(experiments)
    run_log = start_run_log(
        output,
        WORKFLOW,
        {"stage": "historical", "workflow_config": settings},
    )
    try:
        run_log.begin("load_model_and_data")
        model, tokenizer = load_model_and_tokenizer(model_config)
        device = next(model.parameters()).device
        loaders = build_data_loaders(tokenizer, data_config, include_recovery=True)
        footprint = parameter_footprint(model)
        dense_metrics = evaluate_language_model(
            model,
            loaders.model_validation,
            device,
            data_config.num_model_validation_batches,
        )
        dense_row = {
            "variant": "dense_reference",
            "stage": "initial",
            "selected_layers": [],
            "parameters": footprint.parameters,
            "trainable_parameters": footprint.trainable_parameters,
            "theoretical_weight_bytes": footprint.theoretical_weight_bytes,
            "removed_parameters": 0,
            "parameter_reduction_pct": 0.0,
            "teacher_kl": 0.0,
            "loss": dense_metrics.loss,
            "perplexity": dense_metrics.perplexity,
            "predicted_tokens": dense_metrics.predicted_tokens,
            "batches": dense_metrics.batches,
        }
        run_log.record("dense_reference", dense_row)

        results = {}
        for index, variant in enumerate(variants):
            if index:
                del model
                release_cuda(torch)
                model, unused_tokenizer = load_model_and_tokenizer(model_config)
            run_log.begin(f"experiment_{variant}")
            results[variant] = run_replacement_experiment(
                model, loaders, experiments[variant]
            )
            run_log.record(
                f"experiment_{variant}",
                {"selected_layers": list(results[variant].selection.indices)},
            )

        evaluations, selections_df, blocks_df = result_frames(results, dense_row)
        # Preserve the notebook's historical schema, which predates NMSE/cosine.
        blocks_df = blocks_df.drop(
            columns=["initialization", "validation_nmse", "validation_cosine"]
        )
        serialization = pd.DataFrame(
            [
                {
                    "variant": variant,
                    "compressed_model_path": None,
                    "serialized_model_bytes": None,
                }
                for variant in variants
            ]
        )
        compressed_model_root = resolve_path(
            Path("data/results/compressed-models/interleaved-swiglu-050")
        )
        artifact = {
            "schema_version": SCHEMA_HISTORICAL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": environment_record(),
            "configuration": {
                "model": {
                    **asdict(model_config),
                    "resolved_revision": getattr(model.config, "_commit_hash", None),
                },
                "experiments": {
                    name: config.to_dict() for name, config in experiments.items()
                },
                "save_compressed_model": False,
                "compressed_model_paths": {
                    name: str(compressed_model_root / name.replace("_", "-"))
                    for name in variants
                },
            },
            "results": {
                "evaluations": json_records(evaluations),
                "selections": json_records(selections_df),
                "block_fitting": json_records(blocks_df),
                "serialization": json_records(serialization),
                "workflows": json_value(results),
            },
        }
        write_artifact(output, artifact)
        run_log.complete({"artifact_path": str(output), "schema_version": 1})
        print(f"Saved historical compression baseline to {output}", flush=True)
        return artifact
    except BaseException as error:
        run_log.fail(error)
        raise


def optimized_loaders(
    tokenizer,
    data_config: DataConfig,
    base_calibration_batches: int,
    optimized_calibration_batches: int,
):
    partition_batches = {
        "calibration": base_calibration_batches,
        "operator_validation": data_config.num_operator_validation_batches,
        "recovery": data_config.num_recovery_batches,
        "recovery_validation": data_config.num_recovery_validation_batches,
        "additional_calibration": (
            optimized_calibration_batches - base_calibration_batches
        ),
    }
    calibration_data = load_text_dataset(data_config.calibration_source)
    windows = sample_partitioned_windows(
        calibration_data,
        tokenizer,
        {
            name: batches * data_config.batch_size
            for name, batches in partition_batches.items()
        },
        data_config.sequence_length,
        data_config.seed,
        data_config.calibration_source.text_column,
    )
    calibration_sequences = windows["calibration"] + windows["additional_calibration"]
    validation_data = load_text_dataset(data_config.model_validation_source)
    model_validation_batches = data_config.num_model_validation_batches
    if model_validation_batches is None:
        raise ValueError("The compression baseline requires bounded validation")
    validation_sequences = contiguous_token_windows(
        validation_data,
        tokenizer,
        model_validation_batches * data_config.batch_size,
        data_config.sequence_length,
        data_config.model_validation_source.text_column,
    )
    loaders = DataLoaders(
        calibration=make_token_loader(calibration_sequences, data_config.batch_size),
        operator_validation=make_token_loader(
            windows["operator_validation"], data_config.batch_size
        ),
        recovery=make_token_loader(windows["recovery"], data_config.batch_size),
        recovery_validation=make_token_loader(
            windows["recovery_validation"], data_config.batch_size
        ),
        model_validation=make_token_loader(validation_sequences, data_config.batch_size),
        test=None,
    )
    del calibration_data, validation_data
    return loaders, partition_batches


def run_optimized(
    output: Path,
    historical_path: Path,
    settings: dict,
    historical: dict | None = None,
) -> dict:
    output = require_new_path(output)
    historical_path = resolve_path(historical_path)
    historical = historical or load_artifact(
        historical_path, SCHEMA_HISTORICAL, "historical compression baseline"
    )
    resolved_revision = historical["configuration"]["model"].get(
        "resolved_revision"
    )
    if not resolved_revision:
        raise ValueError("Historical artifact does not record a model revision")
    (
        model_config,
        base_data,
        capture,
        unused_operator,
        recovery,
        selections,
        workflow,
        unused_experiments,
    ) = base_configs(settings, resolved_revision)
    data_values = settings["data"]
    operator_values = settings["operator"]
    seed = int(settings["seed"])
    base_calibration_batches = int(data_values["historical_calibration_batches"])
    optimized_calibration_batches = int(data_values["optimized_calibration_batches"])
    data_config = replace(
        base_data, num_calibration_batches=optimized_calibration_batches
    )
    operator_configs = {
        methodology: OperatorConfig(
            kind=operator_values["kind"],
            initialization=initialization,
            intermediate_ratio=float(operator_values["intermediate_ratio"]),
            bias=bool(operator_values["bias"]),
            epochs=int(operator_values["epochs"]),
            learning_rate=float(operator_values["learning_rate"]),
            batch_size=int(operator_values["batch_size"]),
            weight_decay=float(operator_values["weight_decay"]),
            scheduler=operator_values["scheduler"],
            early_stopping_patience=operator_values["early_stopping_patience"],
            seed=seed,
        )
        for methodology, initialization in operator_values["initializations"].items()
    }
    variants = tuple(selections)
    configs = {
        methodology: {
            variant: ExperimentConfig(
                model=model_config,
                data=data_config,
                capture=capture,
                selection=selections[variant],
                operator=operator,
                recovery=recovery,
                workflow=workflow,
            )
            for variant in variants
        }
        for methodology, operator in operator_configs.items()
    }
    run_log = start_run_log(
        output,
        WORKFLOW,
        {
            "stage": "optimized",
            "historical_artifact": str(historical_path),
            "workflow_config": settings,
        },
    )
    try:
        run_log.begin("load_model_and_data")
        model, tokenizer = load_model_and_tokenizer(model_config)
        loaders, partition_batches = optimized_loaders(
            tokenizer,
            data_config,
            base_calibration_batches,
            optimized_calibration_batches,
        )
        run_log.record(
            "load_model_and_data",
            {
                "resolved_revision": getattr(model.config, "_commit_hash", None),
                "partition_batches": partition_batches,
            },
        )
        results = {methodology: {} for methodology in operator_configs}
        first = True
        for methodology in operator_configs:
            for variant in variants:
                if not first:
                    del model
                    release_cuda(torch)
                    model, unused_tokenizer = load_model_and_tokenizer(model_config)
                first = False
                stage = f"experiment_{methodology}_{variant}"
                run_log.begin(stage)
                result = run_replacement_experiment(
                    model, loaders, configs[methodology][variant]
                )
                results[methodology][variant] = result
                run_log.record(stage, {"selected_layers": list(result.selection.indices)})

        evaluation_rows = []
        selection_rows = []
        block_rows = []
        for methodology, methods in results.items():
            for variant, result in methods.items():
                removed = result.footprint_before.parameters - result.footprint_after.parameters
                reduction = 100 * removed / result.footprint_before.parameters
                selection_rows.append(
                    {
                        "methodology": methodology,
                        "variant": variant,
                        "eligible_blocks": len(result.selection.eligible_indices),
                        "selected_blocks": len(result.selection.indices),
                        "selected_layers": list(result.selection.indices),
                    }
                )
                for stage, metrics, teacher_kl in (
                    (
                        "before_recovery",
                        result.pre_recovery_validation_metrics,
                        result.pre_recovery_validation_kl,
                    ),
                    (
                        "after_recovery",
                        result.final_validation_metrics,
                        result.post_recovery_validation_kl,
                    ),
                ):
                    evaluation_rows.append(
                        {
                            "methodology": methodology,
                            "variant": variant,
                            "stage": stage,
                            "selected_layers": list(result.selection.indices),
                            "parameters": result.footprint_after.parameters,
                            "trainable_parameters": result.footprint_after.trainable_parameters,
                            "theoretical_weight_bytes": result.footprint_after.theoretical_weight_bytes,
                            "removed_parameters": removed,
                            "parameter_reduction_pct": reduction,
                            "teacher_kl": teacher_kl,
                            "loss": metrics.loss,
                            "perplexity": metrics.perplexity,
                            "predicted_tokens": metrics.predicted_tokens,
                            "batches": metrics.batches,
                        }
                    )
                for block in result.blocks:
                    block_rows.append(
                        {
                            "methodology": methodology,
                            "variant": variant,
                            "initialization": block.operator_initialization,
                            "layer": block.layer_index,
                            "original_parameters": block.original_parameters,
                            "replacement_parameters": block.replacement_parameters,
                            "retained_pct": 100
                            * block.replacement_parameters
                            / block.original_parameters,
                            "validation_mse": block.operator_validation_mse,
                            "validation_nmse": block.operator_validation_nmse,
                            "validation_cosine": block.operator_validation_cosine,
                            "best_epoch": block.best_operator_epoch,
                        }
                    )

        optimized_evaluations = pd.DataFrame(evaluation_rows)
        optimized_selections = pd.DataFrame(selection_rows)
        optimized_blocks = pd.DataFrame(block_rows)
        historical_evaluations = pd.DataFrame(
            historical["results"]["evaluations"]
        ).assign(methodology="historical_random")
        historical_blocks = pd.DataFrame(
            historical["results"]["block_fitting"]
        ).assign(
            methodology="historical_random",
            initialization="random_weights",
            validation_nmse=float("nan"),
            validation_cosine=float("nan"),
        )
        comparison_evaluations = pd.concat(
            [
                historical_evaluations.query("variant != 'dense_reference'"),
                optimized_evaluations,
            ],
            ignore_index=True,
        )
        comparison_blocks = pd.concat(
            [historical_blocks, optimized_blocks], ignore_index=True
        )
        artifact = {
            "schema_version": SCHEMA_OPTIMIZED,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": environment_record(),
            "configuration": {
                "model": {
                    **asdict(model_config),
                    "resolved_revision": getattr(model.config, "_commit_hash", None),
                },
                "historical_control": {
                    "artifact_path": str(historical_path),
                    "schema_version": historical["schema_version"],
                },
                "methodologies": {
                    "historical_random": {
                        "calibration_pairs": base_calibration_batches
                        * base_data.batch_size
                        * base_data.sequence_length,
                        "initialization": "random_weights",
                    },
                    "new_data_random": {
                        "calibration_pairs": optimized_calibration_batches
                        * data_config.batch_size
                        * data_config.sequence_length,
                        "initialization": "random_weights",
                    },
                    "complete_method": {
                        "calibration_pairs": optimized_calibration_batches
                        * data_config.batch_size
                        * data_config.sequence_length,
                        "initialization": "importance_teacher_subset",
                        "importance_metric": (
                            "rms_intermediate_activation_times_"
                            "down_projection_column_l2"
                        ),
                    },
                },
                "partition_order": list(partition_batches),
                "experiments": {
                    methodology: {
                        variant: config.to_dict()
                        for variant, config in methods.items()
                    }
                    for methodology, methods in configs.items()
                },
            },
            "results": {
                "optimized_evaluations": json_records(optimized_evaluations),
                "optimized_selections": json_records(optimized_selections),
                "optimized_block_fitting": json_records(optimized_blocks),
                "comparison_evaluations": json_records(comparison_evaluations),
                "comparison_block_fitting": json_records(comparison_blocks),
                "workflows": json_value(results),
            },
        }
        write_artifact(output, artifact)
        run_log.complete({"artifact_path": str(output), "schema_version": 2})
        print(f"Saved optimized compression baseline to {output}", flush=True)
        return artifact
    except BaseException as error:
        run_log.fail(error)
        raise


def main() -> None:
    args = parse_args()
    settings = load_workflow_config(args.config, WORKFLOW)
    historical = None
    historical_path = args.historical_artifact
    if args.stage in {"historical", "all"}:
        historical_path = (
            args.historical_output
            or (args.output if args.stage == "historical" else None)
            or default_artifact_path(WORKFLOW, "historical")
        )
        historical = run_historical(historical_path, settings)
    if args.stage in {"optimized", "all"}:
        output = args.output or default_artifact_path(WORKFLOW, "optimized")
        run_optimized(output, historical_path, settings, historical)


if __name__ == "__main__":
    main()
