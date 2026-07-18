from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn

from .capture import collect_module_io
from .config import ExperimentConfig
from .data import DataLoaders
from .evaluation.footprint import ParameterFootprint, parameter_footprint
from .evaluation.language_model import LanguageModelMetrics, evaluate_language_model
from .model import BlockRef, discover_mlp_blocks, get_mlp_block, resolve_dtype
from .operators.training import OperatorFitResult, OperatorTrainingEpoch, fit_replacement_operator
from .recovery import (
    RecoveryResult,
    TeacherCache,
    cache_teacher_logits,
    recover_replacements,
)
from .screening import ScreeningResult, compute_bi_scores
from .selection import LayerSelection, select_layers
from .surgery import ReplacementRecord, apply_replacements


@dataclass(frozen=True)
class BlockReplacementResult:
    layer_index: int
    path: str
    operator_kind: str
    operator_history: tuple[OperatorTrainingEpoch, ...]
    best_operator_epoch: int
    operator_validation_mse: float
    original_parameters: int
    replacement_parameters: int
    model_metrics_after: LanguageModelMetrics | None = None

    @property
    def removed_parameters(self) -> int:
        return self.original_parameters - self.replacement_parameters


@dataclass(frozen=True)
class WorkflowResult:
    strategy: str
    selection: LayerSelection
    screening: ScreeningResult | None
    baseline_metrics: LanguageModelMetrics
    final_metrics: LanguageModelMetrics
    footprint_before: ParameterFootprint
    footprint_after: ParameterFootprint
    blocks: tuple[BlockReplacementResult, ...]
    recovery_steps: tuple[RecoveryResult, ...]


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _fit_layer(
    model: nn.Module,
    ref: BlockRef,
    loaders: DataLoaders,
    config: ExperimentConfig,
    device: torch.device,
) -> OperatorFitResult:
    storage_device = torch.device(config.capture.storage_device)
    storage_dtype = resolve_dtype(config.capture.storage_dtype, storage_device)
    training_pairs = collect_module_io(
        model=model,
        module_path=ref.path,
        loader=loaders.calibration,
        max_batches=config.data.num_calibration_batches,
        device=device,
        storage_device=storage_device,
        storage_dtype=storage_dtype,
    )
    validation_pairs = collect_module_io(
        model=model,
        module_path=ref.path,
        loader=loaders.operator_validation,
        max_batches=config.data.num_operator_validation_batches,
        device=device,
        storage_device=storage_device,
        storage_dtype=storage_dtype,
    )
    return fit_replacement_operator(
        training_pairs=training_pairs,
        validation_pairs=validation_pairs,
        config=config.operator,
        device=device,
    )


def _teacher_caches(
    model: nn.Module,
    loaders: DataLoaders,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[TeacherCache, TeacherCache | None]:
    if loaders.recovery is None:
        raise ValueError("Recovery is enabled but no recovery loader was provided")
    training_cache = cache_teacher_logits(
        model,
        loaders.recovery,
        config.data.num_recovery_batches,
        device,
        config.recovery.cache_dtype,
    )
    validation_cache = None
    if loaders.recovery_validation is not None and config.data.num_recovery_validation_batches > 0:
        validation_cache = cache_teacher_logits(
            model,
            loaders.recovery_validation,
            config.data.num_recovery_validation_batches,
            device,
            config.recovery.cache_dtype,
        )
    return training_cache, validation_cache


def _block_result(
    layer_index: int,
    path: str,
    fit: OperatorFitResult,
    record: ReplacementRecord,
    config: ExperimentConfig,
    model_metrics_after: LanguageModelMetrics | None = None,
) -> BlockReplacementResult:
    return BlockReplacementResult(
        layer_index=layer_index,
        path=path,
        operator_kind=config.operator.kind,
        operator_history=fit.history,
        best_operator_epoch=fit.best_epoch,
        operator_validation_mse=fit.best_validation_mse,
        original_parameters=record.original_parameters,
        replacement_parameters=record.replacement_parameters,
        model_metrics_after=model_metrics_after,
    )


def run_one_shot_replacement(
    model: nn.Module,
    loaders: DataLoaders,
    selection: LayerSelection,
    config: ExperimentConfig,
    screening: ScreeningResult | None = None,
) -> WorkflowResult:
    """Fit every replacement against the same dense model, then apply them together."""

    device = _model_device(model)
    footprint_before = parameter_footprint(model)
    baseline = evaluate_language_model(
        model, loaders.evaluation, device, config.data.num_evaluation_batches
    )
    training_cache = validation_cache = None
    if config.recovery.enabled:
        training_cache, validation_cache = _teacher_caches(model, loaders, config, device)

    refs = {ref.index: ref for ref in discover_mlp_blocks(model)}
    target_paths = {index: refs[index].path for index in selection.indices}
    fits: dict[int, OperatorFitResult] = {}
    replacements: dict[int, nn.Module] = {}
    for layer_index in selection.indices:
        fit = _fit_layer(model, refs[layer_index], loaders, config, device)
        fits[layer_index] = fit
        replacements[layer_index] = fit.module

    manifest = apply_replacements(model, replacements)
    records = {record.layer_index: record for record in manifest.records}
    del refs
    recovery_steps: tuple[RecoveryResult, ...] = ()
    if config.recovery.enabled:
        recovery = recover_replacements(
            student=model,
            training_cache=training_cache,
            validation_cache=validation_cache,
            target_paths=[records[index].path for index in selection.indices],
            config=config.recovery,
            device=device,
        )
        recovery_steps = (recovery,)

    final = evaluate_language_model(
        model, loaders.evaluation, device, config.data.num_evaluation_batches
    )
    blocks = tuple(
        _block_result(index, target_paths[index], fits[index], records[index], config)
        for index in selection.indices
    )
    return WorkflowResult(
        strategy="one_shot",
        selection=selection,
        screening=screening,
        baseline_metrics=baseline,
        final_metrics=final,
        footprint_before=footprint_before,
        footprint_after=parameter_footprint(model),
        blocks=blocks,
        recovery_steps=recovery_steps,
    )


def run_iterative_replacement(
    model: nn.Module,
    loaders: DataLoaders,
    selection: LayerSelection,
    config: ExperimentConfig,
    screening: ScreeningResult | None = None,
) -> WorkflowResult:
    """Fit, apply, optionally recover, and evaluate one selected MLP at a time."""

    device = _model_device(model)
    footprint_before = parameter_footprint(model)
    baseline = evaluate_language_model(
        model, loaders.evaluation, device, config.data.num_evaluation_batches
    )
    dense_caches: tuple[TeacherCache, TeacherCache | None] | None = None
    if config.recovery.enabled and config.workflow.iterative_teacher == "dense":
        dense_caches = _teacher_caches(model, loaders, config, device)

    replaced_paths: list[str] = []
    block_results: list[BlockReplacementResult] = []
    recovery_results: list[RecoveryResult] = []

    for layer_index in selection.indices:
        ref = get_mlp_block(model, layer_index)
        target_path = ref.path
        fit = _fit_layer(model, ref, loaders, config, device)
        step_caches = dense_caches
        if config.recovery.enabled and config.workflow.iterative_teacher == "previous":
            step_caches = _teacher_caches(model, loaders, config, device)

        manifest = apply_replacements(model, {layer_index: fit.module})
        record = manifest.records[0]
        replaced_paths.append(record.path)
        del ref

        if config.recovery.enabled:
            if step_caches is None:
                raise RuntimeError("Iterative recovery teacher cache was not created")
            recovery_paths = (
                [record.path]
                if config.workflow.iterative_recovery_scope == "current"
                else list(replaced_paths)
            )
            recovery_results.append(
                recover_replacements(
                    student=model,
                    training_cache=step_caches[0],
                    validation_cache=step_caches[1],
                    target_paths=recovery_paths,
                    config=config.recovery,
                    device=device,
                )
            )

        step_metrics = evaluate_language_model(
            model, loaders.evaluation, device, config.data.num_evaluation_batches
        )
        block_results.append(
            _block_result(layer_index, target_path, fit, record, config, step_metrics)
        )

    final = block_results[-1].model_metrics_after if block_results else baseline
    if final is None:
        raise RuntimeError("Iterative workflow did not produce final metrics")
    return WorkflowResult(
        strategy="iterative",
        selection=selection,
        screening=screening,
        baseline_metrics=baseline,
        final_metrics=final,
        footprint_before=footprint_before,
        footprint_after=parameter_footprint(model),
        blocks=tuple(block_results),
        recovery_steps=tuple(recovery_results),
    )


def run_replacement_experiment(
    model: nn.Module,
    loaders: DataLoaders,
    config: ExperimentConfig,
    bi_scores: Mapping[int, float] | None = None,
) -> WorkflowResult:
    refs = discover_mlp_blocks(model)
    screening = None
    scores = bi_scores
    if config.selection.strategy == "top_k_bi" and scores is None:
        screening = compute_bi_scores(
            model=model,
            loader=loaders.calibration,
            max_batches=config.data.num_calibration_batches,
            device=_model_device(model),
        )
        scores = screening.scores

    selection = select_layers(
        available_indices=[ref.index for ref in refs],
        config=config.selection,
        bi_scores=scores,
    )
    if config.workflow.strategy == "one_shot":
        return run_one_shot_replacement(model, loaders, selection, config, screening)
    if config.workflow.strategy == "iterative":
        return run_iterative_replacement(model, loaders, selection, config, screening)
    raise ValueError(f"Unsupported workflow strategy: {config.workflow.strategy}")
