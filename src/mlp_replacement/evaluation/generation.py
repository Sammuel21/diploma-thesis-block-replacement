"""Fixed-shape, model-only generation measurements with BF16 KV caches."""

import gc
import os
from statistics import median
from time import perf_counter


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def greedy_request(model, input_ids, generated_tokens, verify_cache=False):
    """Time prefill/first token, then a complete forced-length cached decode."""

    import torch

    torch.cuda.synchronize()
    started = perf_counter()
    output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                   use_cache=True, logits_to_keep=1)
    next_token = output.logits[:, -1:].argmax(dim=-1)
    cache = output.past_key_values
    del output
    torch.cuda.synchronize()
    first = perf_counter()
    for index in range(1, generated_tokens):
        output = model(input_ids=next_token, past_key_values=cache, use_cache=True,
                       logits_to_keep=1)
        next_token = output.logits[:, -1:].argmax(dim=-1)
        cache = output.past_key_values
        del output
    torch.cuda.synchronize()
    finished = perf_counter()
    if verify_cache:
        if not cache.layers or any(value.dtype != torch.bfloat16 for layer in cache.layers
                                   for value in (layer.keys, layer.values)):
            raise ValueError("Generation requires a BF16 inference KV cache")
    del cache, next_token
    batch_size = input_ids.shape[0]
    total = finished - started
    decode = finished - first
    return {
        "time_to_first_token_ms": (first - started) * 1000,
        "mean_subsequent_token_latency_ms": decode * 1000 / (generated_tokens - 1),
        "generation_latency_ms": total * 1000,
        "output_tokens_per_second": batch_size * generated_tokens / total,
        "decode_tokens_per_second": batch_size * (generated_tokens - 1) / decode,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def measure_generation(model, prompt_tokens, workloads, warmups=2, repetitions=10):
    import torch

    if warmups < 1 or repetitions < 2:
        raise ValueError("Generation measurement requires warmups and repeated requests")
    model.eval()
    results = []
    with torch.inference_mode():
        for workload in workloads:
            prompt = int(workload["prompt_tokens"])
            generated = int(workload["generated_tokens"])
            batch_size = int(workload["batch_size"])
            if generated < 2 or prompt + generated > model.config.max_position_embeddings:
                raise ValueError("Generation workload exceeds the model context")
            if len(prompt_tokens) < prompt or batch_size < 1:
                raise ValueError("Invalid fixed generation prompt")
            ids = torch.tensor(list(prompt_tokens[:prompt]), dtype=torch.long, device="cuda")
            ids = ids.unsqueeze(0).repeat(batch_size, 1)
            for index in range(warmups):
                greedy_request(model, ids, generated, verify_cache=index == 0)
            samples = []
            for index in range(repetitions):
                torch.cuda.reset_peak_memory_stats()
                samples.append(greedy_request(model, ids, generated))
            metrics = samples[0].keys()
            results.append({
                **workload, "warmups": warmups, "repetitions": repetitions,
                "samples": samples,
                "peak_gpu_allocated_bytes": max(row["peak_gpu_allocated_bytes"] for row in samples),
                "peak_gpu_reserved_bytes": max(row["peak_gpu_reserved_bytes"] for row in samples),
                "median": {key: median(row[key] for row in samples) for key in metrics},
                "p95": {key: percentile([row[key] for row in samples], 0.95) for key in metrics},
            })
            del ids
    return {"protocol": "fixed-length-greedy-bf16-kv-eager-v1",
            "timing": "CUDA-synchronized wall time at first-token and request boundaries",
            "excludes": ["tokenization", "model_loading", "network", "queueing"],
            "workloads": results}


def measure_bundle_runtime(directory, prompt_tokens, settings):
    """Use only in a fresh process; capture resident memory before inference."""

    import psutil
    import torch
    from mlp_replacement.evaluation.quantized_bundles import load_inference_bundle

    process = psutil.Process(os.getpid())
    torch.cuda.init()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    before = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved(), process.memory_info().rss)
    started = perf_counter()
    model, tokenizer, manifest = load_inference_bundle(
        directory, attention_implementation=settings["attention_implementation"],
    )
    gc.collect()
    torch.cuda.synchronize()
    resident = {
        "protocol": "fresh-process-model-only-no-forward",
        "gpu_allocated_delta_bytes": torch.cuda.memory_allocated() - before[0],
        "gpu_reserved_delta_bytes": torch.cuda.memory_reserved() - before[1],
        "host_rss_delta_bytes": process.memory_info().rss - before[2],
        "load_seconds": perf_counter() - started,
    }
    speed = measure_generation(model, prompt_tokens, settings["workloads"],
                               settings["warmups"], settings["repetitions"])
    result = {"resident_memory": resident, "generation": speed,
              "attention_implementation": model.config._attn_implementation,
              "host_rss_after_generation_bytes": process.memory_info().rss}
    if os.name != "nt":
        import resource
        result["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return result
