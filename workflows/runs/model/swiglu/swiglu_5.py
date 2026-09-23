"""Compatibility exports for SwiGLU-5; implementation lives in ``swiglu5``."""

from .swiglu5.context import (
    CANDIDATE_DEFINITIONS,
    CONFIRMATION_WORKFLOW,
    METRIC_DEFINITIONS,
    RECOVERY_PROTOCOL_DEFINITION,
    SCHEMA_VERSION,
    SEARCH_WORKFLOW,
    SwiGLU5Context,
    asset_directory,
    default_output,
    import_swiglu3_evidence,
    load_search_resources,
    packed_source_cache,
    prepare_confirmation_context,
    prepare_search_context,
    record_stage_runtime,
    storage_preflight,
    strict_source_assets,
    utc_now,
    validate_search_resume_artifact,
)

from .swiglu5.fitting import (
    build_width_curves,
    capture_dense_pairs,
    curve_lookup,
    discrete_allocation,
    ensure_fit_for_width,
    fit_key,
    fit_operator,
    history_rows,
    load_fit_operator,
    operator_config,
    operator_state_path,
    selected_neurons,
    singleton_kl,
)

from .swiglu5.candidates import (
    blank_candidate_student,
    build_c1_c3_candidates,
    build_composition_candidates,
    candidate_modules,
    candidate_parameter_summary,
    dense_targets,
    evaluate_candidate,
    legacy_candidate,
    load_candidate_student,
)

from .swiglu5.recovery import (
    calibrate_recovery,
    checkpoint_rng,
    clone_teacher_head,
    find_full_evaluation,
    profile_trial,
    recover_candidate,
    requested_actual_map,
    restore_rng,
)

from .swiglu5.search import (
    initial_candidates_complete,
    pending_candidate_fit_keys,
    prune_search_candidate_checkpoints,
    prune_search_local_fit_states,
    prune_teacher_hidden_cache,
    rank_qualifier_challengers,
    retain_search_winner_endpoint,
    run_search,
    runtime_guard,
    select_finalists,
    select_winner,
    width_curves_complete,
)

from .swiglu5.confirmation import (
    calibrate_online_confirmation,
    confirmation_candidate,
    load_selected_search_checkpoint,
    online_profile_trial,
    run_confirmation,
    source_comparison_row,
)

from mlp_replacement.compression.reconstruction import load_replacement_state, replacement_state
from mlp_replacement.compression.surgery import temporary_fp32_replacements
