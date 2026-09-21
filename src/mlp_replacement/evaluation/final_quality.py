"""Full-corpus likelihood and paired downstream reporting for final models."""

import math
from pathlib import Path


def rolling_windows(token_count, context_length, stride):
    """Yield (begin, end, first_target) with each global target scored once."""

    if context_length < 2 or not 1 <= stride < context_length:
        raise ValueError("Rolling evaluation requires 0 < stride < context length")
    end = min(context_length, token_count)
    if end > 1:
        yield 0, end, 1
    while end < token_count:
        previous_end = end
        end = min(token_count, end + stride)
        begin = max(0, end - context_length)
        yield begin, end, previous_end - begin


def evaluate_rolling_likelihood(model, token_ids, context_length, stride, device):
    """Score the complete token stream, including its final partial window."""

    import torch
    import torch.nn.functional as F

    total_nll = 0.0
    targets = 0
    windows = 0
    model.eval()
    with torch.inference_mode():
        for begin, end, first_target in rolling_windows(len(token_ids), context_length, stride):
            ids = torch.as_tensor(token_ids[begin:end], dtype=torch.long, device=device).unsqueeze(0)
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            nll = F.cross_entropy(
                logits[0, first_target - 1:-1].float(),
                ids[0, first_target:],
                reduction="sum",
            )
            total_nll += float(nll.item())
            targets += end - begin - first_target
            windows += 1
            del logits, ids, nll
    if targets != len(token_ids) - 1 or targets < 1 or not math.isfinite(total_nll):
        raise ValueError("Rolling likelihood failed its coverage/finite-loss contract")
    loss = total_nll / targets
    return {"loss": loss, "perplexity": math.exp(loss), "predicted_tokens": targets,
            "windows": windows, "context_length": context_length, "stride": stride}


def plain_values(value):
    """Normalize harness NumPy scalars and arrays for strict JSON output."""

    if isinstance(value, dict):
        return {str(key): plain_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain_values(item) for item in value]
    if hasattr(value, "tolist"):
        return plain_values(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if type(value).__module__ == "torch" and type(value).__name__ in ("dtype", "device"):
        return str(value)
    return value


def evaluate_pinned_task(model, tokenizer, task, config, task_config):
    """Run one full zero-shot task against its frozen dataset revision."""

    from importlib.metadata import version
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    installed = version("lm_eval")
    if installed != config["harness_version"]:
        raise ValueError(f"Expected lm_eval {config['harness_version']}, found {installed}")
    # Supply a native task-config mapping: no unversioned dataset download or
    # changed prompt is permitted during the final evaluation.
    wrapper = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1,
                   max_length=config["benchmark_context_length"], device="cuda")
    seed = config["seed"]
    output = lm_eval.simple_evaluate(
        model=wrapper, tasks=[task_config], num_fewshot=0, batch_size=1,
        limit=None, log_samples=True, apply_chat_template=False,
        random_seed=seed, numpy_random_seed=seed, torch_random_seed=seed,
        fewshot_random_seed=seed,
        bootstrap_iters=config["bootstrap_resamples"],
    )
    if task not in output["results"] or not output.get("samples", {}).get(task):
        raise ValueError(f"Task {task} did not emit full metrics and paired samples")
    return plain_values(output)


def paired_accuracy_difference(dense_rows, student_rows, metric, resamples=10000, seed=21):
    """Bootstrap paired examples without treating them as training-seed runs."""

    import numpy as np

    def indexed(rows):
        result = {}
        for row in rows:
            key = (row["doc_id"], row.get("filter", "none"))
            if key in result:
                raise ValueError("Duplicate benchmark sample identity")
            result[key] = row
        return result

    dense = indexed(dense_rows)
    student = indexed(student_rows)
    if dense.keys() != student.keys() or not dense:
        raise ValueError("Paired benchmark samples do not match")
    differences = []
    for key in sorted(dense):
        for field in ("doc_hash", "prompt_hash", "target_hash"):
            if dense[key].get(field) != student[key].get(field):
                raise ValueError(f"Paired benchmark {field} differs")
        differences.append(float(student[key][metric]) - float(dense[key][metric]))
    values = np.asarray(differences)
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples)
    for index in range(resamples):
        estimates[index] = values[rng.integers(0, len(values), size=len(values))].mean()
    return {"student_minus_dense": float(values.mean()),
            "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "examples": len(values), "resamples": resamples, "seed": seed}
