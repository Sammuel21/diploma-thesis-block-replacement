"""Model-level MLP replacement and recovery workflows."""

from .allocation import (
    DiscreteSwiGLUAllocation,
    SwiGLUWidthCurvePoint,
    allocate_discrete_swiglu_widths,
    build_swiglu_width_curve,
)
from .recovery import (
    TeacherFinalHiddenCache,
    build_teacher_final_hidden_cache,
    cached_hidden_distillation_loss,
    validate_teacher_final_hidden_cache,
)

__all__ = [
    "DiscreteSwiGLUAllocation",
    "SwiGLUWidthCurvePoint",
    "TeacherFinalHiddenCache",
    "allocate_discrete_swiglu_widths",
    "build_swiglu_width_curve",
    "build_teacher_final_hidden_cache",
    "cached_hidden_distillation_loss",
    "validate_teacher_final_hidden_cache",
]
