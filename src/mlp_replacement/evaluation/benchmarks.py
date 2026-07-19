from dataclasses import dataclass
from typing import Any, Mapping


BENCHMARK_PROFILES = {
    "smoke": ("piqa",),
    "routine": ("piqa", "arc_easy", "winogrande"),
    "confirmation": ("piqa", "arc_easy", "winogrande", "arc_challenge", "hellaswag"),
    "extended-knowledge": (
        "piqa",
        "arc_easy",
        "winogrande",
        "arc_challenge",
        "hellaswag",
        "mmlu",
    ),
    "conditional-math": (
        "piqa",
        "arc_easy",
        "winogrande",
        "arc_challenge",
        "hellaswag",
        "gsm8k",
    ),
}


@dataclass(frozen=True)
class BenchmarkResult:
    """Store benchmark task results and the harness version that produced them."""

    tasks: tuple[str, ...]
    results: Mapping[str, Any]
    harness_version: str


def resolve_benchmark_tasks(profile, additions=()):
    """Expand a named evaluation profile and any explicitly requested additions."""

    try:
        base = BENCHMARK_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark profile: {profile}") from exc
    return tuple(dict.fromkeys((*base, *additions)))


def evaluate_benchmarks(model, tokenizer, profile, additions=(), batch_size="auto", limit=None, device=None):
    """Run a pinned lm-evaluation-harness installation without reimplementing tasks."""

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:
        raise ImportError(
            "Benchmark evaluation requires the 'benchmarks' optional dependency"
        ) from exc

    if profile != "smoke" and limit is not None:
        raise ValueError("Example limits are only valid for the smoke profile")

    tasks = resolve_benchmark_tasks(profile, additions)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device)
    output = lm_eval.simple_evaluate(model=lm, tasks=list(tasks), limit=limit)
    version = getattr(lm_eval, "__version__", "unknown")
    return BenchmarkResult(tasks, output, version)
