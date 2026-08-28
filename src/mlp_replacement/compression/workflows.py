from dataclasses import dataclass

import torch

from ..analysis.screening import ScreeningResult, compute_bi_scores
from ..capture import collect_module_io, collect_modules_io
from ..evaluation.footprint import ParameterFootprint, parameter_footprint
from ..evaluation.language_model import LanguageModelMetrics, evaluate_language_model
from ..model import discover_mlp_blocks, resolve_dtype
from ..operators.training import OperatorTrainingEpoch, fit_replacement_operator
from .recovery import (
    RecoveryResult,
    cache_teacher_logits,
    mean_cache_loss,
    recover_replacements,
)
from .selection import LayerSelection, select_layers
from .surgery import apply_replacements


@dataclass(frozen=True)
class BlockReplacementResult:
    """Summarize local fitting and parameter reduction for one replaced MLP."""

    layer_index: int
    path: str
    operator_kind: str
    operator_history: tuple[OperatorTrainingEpoch, ...]
    best_operator_epoch: int
    operator_validation_mse: float
    original_parameters: int
    replacement_parameters: int

    @property
    def removed_parameters(self):
        """Return the parameter reduction achieved by this block replacement."""

        return self.original_parameters - self.replacement_parameters


@dataclass(frozen=True)
class WorkflowResult:
    """Collect model-level and block-level results from one replacement workflow."""

    strategy: str
    selection: LayerSelection
    screening: ScreeningResult | None
    baseline_validation_metrics: LanguageModelMetrics
    pre_recovery_validation_metrics: LanguageModelMetrics
    pre_recovery_validation_kl: float | None
    final_validation_metrics: LanguageModelMetrics
    post_recovery_validation_kl: float | None
    baseline_test_metrics: LanguageModelMetrics | None
    final_test_metrics: LanguageModelMetrics | None
    footprint_before: ParameterFootprint
    footprint_after: ParameterFootprint
    blocks: tuple[BlockReplacementResult, ...]
    recovery_steps: tuple[RecoveryResult, ...]


def fit_layer_replacement(model, ref, loaders, config, device):
    """Capture one MLP's data and fit its configured replacement operator."""

    storage_device = torch.device(config.capture.storage_device)
    storage_dtype = resolve_dtype(config.capture.storage_dtype, storage_device)
    training_pairs = collect_module_io(
        model, ref.path, loaders.calibration, config.data.num_calibration_batches,
        device, storage_device, storage_dtype
    )
    validation_pairs = collect_module_io(
        model, ref.path, loaders.operator_validation, config.data.num_operator_validation_batches,
        device, storage_device, storage_dtype
    )
    intermediate_width = (
        int(ref.module.up_proj.out_features)
        if config.operator.kind == "swiglu"
        else None
    )
    return fit_replacement_operator(
        training_pairs,
        validation_pairs,
        config.operator,
        device,
        intermediate_width,
    )


def create_teacher_caches(model, loaders, config, device):
    """Cache dense predictions for recovery training and validation."""

    if loaders.recovery is None:
        raise ValueError("Recovery is enabled but no recovery loader was provided")
    training_cache = cache_teacher_logits(model, loaders.recovery, config.data.num_recovery_batches,
                                           device, config.recovery.cache_dtype)
    validation_cache = None
    if loaders.recovery_validation is not None and config.data.num_recovery_validation_batches > 0:
        validation_cache = cache_teacher_logits(model, loaders.recovery_validation,
                                                 config.data.num_recovery_validation_batches,
                                                 device, config.recovery.cache_dtype)
    return training_cache, validation_cache


def create_block_result(layer_index, path, fit, record, config):
    """Combine local fitting and structural information into one block result."""

    return BlockReplacementResult(
        layer_index=layer_index,
        path=path,
        operator_kind=config.operator.kind,
        operator_history=fit.history,
        best_operator_epoch=fit.best_epoch,
        operator_validation_mse=fit.best_validation_mse,
        original_parameters=record.original_parameters,
        replacement_parameters=record.replacement_parameters,
    )


def run_one_shot_replacement(model, loaders, selection, config, screening=None, run_log=None):
    """Fit every replacement against the same dense model, then apply them together."""

    device = next(model.parameters()).device
    footprint_before = parameter_footprint(model)
    if run_log is not None:
        run_log.record("footprint_before", footprint_before)
        run_log.begin("baseline_model_validation")
    baseline_validation = evaluate_language_model(
        model, loaders.model_validation, device, config.data.num_model_validation_batches
    )
    baseline_test = None
    if loaders.test is not None:
        if run_log is not None:
            run_log.begin("baseline_final_test")
        baseline_test = evaluate_language_model(model, loaders.test, device, config.data.num_test_batches)
    if run_log is not None:
        run_log.record("baseline_metrics", {
            "model_validation": baseline_validation,
            "final_test": baseline_test,
        })

    training_cache = validation_cache = None
    if config.recovery.enabled:
        if run_log is not None:
            run_log.begin("teacher_cache")
        training_cache, validation_cache = create_teacher_caches(model, loaders, config, device)
        if run_log is not None:
            run_log.record("teacher_cache", {
                "training_batches": len(training_cache),
                "validation_batches": len(validation_cache) if validation_cache is not None else 0,
                "dtype": config.recovery.cache_dtype,
            })

    refs = {ref.index: ref for ref in discover_mlp_blocks(model)}
    target_paths = {index: refs[index].path for index in selection.indices}
    storage_device = torch.device(config.capture.storage_device)
    storage_dtype = resolve_dtype(config.capture.storage_dtype, storage_device)
    if run_log is not None:
        run_log.begin("operator_activation_capture")
    training_pairs_by_path = collect_modules_io(
        model,
        target_paths.values(),
        loaders.calibration,
        config.data.num_calibration_batches,
        device,
        storage_device,
        storage_dtype,
    )
    validation_pairs_by_path = collect_modules_io(
        model,
        target_paths.values(),
        loaders.operator_validation,
        config.data.num_operator_validation_batches,
        device,
        storage_device,
        storage_dtype,
    )
    if run_log is not None:
        run_log.record("operator_activation_capture", {
            "layers": selection.indices,
            "training_tokens_per_layer": next(
                iter(training_pairs_by_path.values())
            ).num_tokens,
            "validation_tokens_per_layer": next(
                iter(validation_pairs_by_path.values())
            ).num_tokens,
        })
    fits = {}
    replacements = {}
    operator_progress = {}
    for layer_index in selection.indices:
        if run_log is not None:
            run_log.begin(f"operator_fit_layer_{layer_index}")
        ref = refs[layer_index]
        intermediate_width = (
            int(ref.module.up_proj.out_features)
            if config.operator.kind == "swiglu"
            else None
        )
        fit = fit_replacement_operator(
            training_pairs_by_path[ref.path],
            validation_pairs_by_path[ref.path],
            config.operator,
            device,
            intermediate_width,
        )
        fits[layer_index] = fit
        replacements[layer_index] = fit.module
        operator_progress[str(layer_index)] = {
            "path": refs[layer_index].path,
            "kind": config.operator.kind,
            "history": fit.history,
            "best_epoch": fit.best_epoch,
            "best_validation_mse": fit.best_validation_mse,
        }
        if run_log is not None:
            run_log.record("operators", operator_progress)

    del training_pairs_by_path, validation_pairs_by_path

    if run_log is not None:
        run_log.begin("apply_replacements")
    manifest = apply_replacements(model, replacements)
    if run_log is not None:
        run_log.record("replacement_manifest", manifest)
    records = {record.layer_index: record for record in manifest.records}
    del refs
    if run_log is not None:
        run_log.begin("pre_recovery_model_validation")
    pre_recovery_validation = evaluate_language_model(
        model,
        loaders.model_validation,
        device,
        config.data.num_model_validation_batches,
    )
    pre_recovery_validation_kl = (
        mean_cache_loss(
            model,
            validation_cache,
            config.recovery.temperature,
            device,
        )
        if validation_cache is not None
        else None
    )
    if run_log is not None:
        run_log.record(
            "pre_recovery_metrics",
            {
                "model_validation": pre_recovery_validation,
                "teacher_kl": pre_recovery_validation_kl,
            },
        )
    recovery_steps = ()
    if config.recovery.enabled:
        if run_log is not None:
            run_log.begin("model_recovery")
        recovery_paths = [records[index].path for index in selection.indices]
        recovery = recover_replacements(model, training_cache, validation_cache,
                                          recovery_paths, config.recovery, device)
        recovery_steps = (recovery,)
        if run_log is not None:
            run_log.record("recovery", recovery_steps)

    if run_log is not None:
        run_log.begin("final_model_validation")
    final_validation = evaluate_language_model(
        model, loaders.model_validation, device, config.data.num_model_validation_batches
    )
    post_recovery_validation_kl = (
        mean_cache_loss(
            model,
            validation_cache,
            config.recovery.temperature,
            device,
        )
        if validation_cache is not None
        else None
    )
    final_test = None
    if loaders.test is not None:
        if run_log is not None:
            run_log.begin("final_test")
        final_test = evaluate_language_model(model, loaders.test, device, config.data.num_test_batches)
    if run_log is not None:
        run_log.record("final_metrics", {
            "model_validation": final_validation,
            "teacher_kl": post_recovery_validation_kl,
            "final_test": final_test,
        })

    blocks = tuple(
        create_block_result(index, target_paths[index], fits[index], records[index], config)
        for index in selection.indices
    )
    footprint_after = parameter_footprint(model)
    if run_log is not None:
        run_log.record("footprint_after", footprint_after)
    return WorkflowResult(
        strategy="one_shot",
        selection=selection,
        screening=screening,
        baseline_validation_metrics=baseline_validation,
        pre_recovery_validation_metrics=pre_recovery_validation,
        pre_recovery_validation_kl=pre_recovery_validation_kl,
        final_validation_metrics=final_validation,
        post_recovery_validation_kl=post_recovery_validation_kl,
        baseline_test_metrics=baseline_test,
        final_test_metrics=final_test,
        footprint_before=footprint_before,
        footprint_after=footprint_after,
        blocks=blocks,
        recovery_steps=recovery_steps,
    )


def run_iterative_replacement(model, loaders, selection, config, screening=None):
    """Reject the deprecated iterative replacement workflow."""

    raise RuntimeError(
        "Iterative replacement is deprecated and disabled. Use workflow.strategy='one_shot'. "
        "Minitron found no benefit from repeated within-stage importance/pruning iterations, "
        "and this replacement variant multiplies local fitting and recovery cost."
    )


def run_replacement_experiment(model, loaders, config, bi_scores=None, run_log=None):
    """Resolve screening and selection before dispatching the configured workflow."""

    refs = discover_mlp_blocks(model)
    screening = None
    scores = bi_scores
    if config.selection.strategy == "top_k_bi" and scores is None:
        device = next(model.parameters()).device
        if run_log is not None:
            run_log.begin("bi_screening")
        screening = compute_bi_scores(
            model, loaders.calibration, config.data.num_calibration_batches,
            device, scope=config.selection.bi_scope
        )
        scores = screening.scores
        if run_log is not None:
            run_log.record("screening", screening)
    elif config.selection.strategy == "top_k_bi" and run_log is not None:
        run_log.record("screening", {
            "metric": f"external_{config.selection.bi_scope}_bi",
            "scores": scores,
            "num_batches": None,
        })

    available_indices = [ref.index for ref in refs]
    selection = select_layers(available_indices, config.selection, scores)
    if run_log is not None:
        run_log.record("selection", selection)
    if config.workflow.strategy == "one_shot":
        return run_one_shot_replacement(model, loaders, selection, config, screening, run_log)
    if config.workflow.strategy == "iterative":
        return run_iterative_replacement(model, loaders, selection, config, screening)
    raise ValueError(f"Unsupported workflow strategy: {config.workflow.strategy}")
