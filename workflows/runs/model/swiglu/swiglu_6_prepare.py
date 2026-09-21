"""Prepare the shared SwiGLU-6 stream and immutable source provenance."""

import argparse
import shutil
from pathlib import Path

from workflows.runs.model.common import resolve_path
from mlp_replacement.artifacts import file_digest
from mlp_replacement.data_streams import build_verified_stream
from .swiglu_6 import DEFAULT_CONFIG, asset_directory, load_settings, new_artifact, persist, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = load_settings(args.config)
    output, artifact = new_artifact(args.output, settings, "prepare")
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer

        search, source, provenance = sources(settings)
        artifact["provenance"] = provenance
        assets = asset_directory(output)
        assets.mkdir(parents=True)
        required_bytes = int(settings["data"]["target_tokens"] * 4 * 1.15)
        if shutil.disk_usage(assets).free < required_bytes:
            raise RuntimeError(f"Packed stream requires at least {required_bytes} free bytes, plus dataset cache")
        values = settings["data"]
        info = HfApi().dataset_info(values["recovery_dataset"], revision=values["recovery_revision"], files_metadata=True)
        revision = info.sha
        inventory = {item.rfilename: item for item in info.siblings}
        used_shards = []
        artifact["dataset_revision"] = revision
        persist(output, artifact, "tokenizing")
        tokenizer = AutoTokenizer.from_pretrained(settings["model"]["model_id"], revision=settings["model"]["tokenizer_revision"])
        from .shared import build_local_data
        from .swiglu_6 import legacy_context
        from mlp_replacement.compression.continuation import save_tensor_atomic

        context = legacy_context(search, tokenizer)
        context.settings["data"]["local_source"]["revision"] = revision
        wiki_revision = HfApi().dataset_info("Salesforce/wikitext").sha
        context.settings["data"]["model_validation_source"]["revision"] = wiki_revision
        legacy = build_local_data(context)
        legacy_path = assets / "legacy-evaluation-batches.pt"
        save_tensor_atomic(legacy_path, {
            key: list(legacy[key]) for key in ("recovery_validation", "model_validation")
        })
        artifact["legacy_evaluation"] = {"path": legacy_path.name, "sha256": file_digest(legacy_path),
                                          "c4_revision": revision, "wikitext_revision": wiki_revision}
        del legacy, context

        def records():
            for index in range(values["first_shard"], values["shard_count"]):
                name = f"en/c4-train.{index:05d}-of-{values['shard_count']:05d}.json.gz"
                entry = inventory[name]
                lfs = entry.lfs
                used_shards.append({"path": name, "git_blob": entry.blob_id,
                                    "sha256": (lfs.get("sha256") if isinstance(lfs, dict)
                                               else getattr(lfs, "sha256", None))})
                yield from load_dataset(values["recovery_dataset"], revision=revision,
                                        data_files={"train": name}, split="train", streaming=True)

        def progress(tokens, documents):
            artifact["progress"] = {"tokens": tokens, "documents": documents}
            persist(output, artifact, "tokenizing")
            print(f"Prepared {tokens:,} recovery tokens", flush=True)

        stream = build_verified_stream(records(), tokenizer, assets / "recovery-tokens.bin",
                                       values["target_tokens"], provenance["prefix"]["path"],
                                       values["historical_prefix_tokens"], progress)
        artifact["results"] = {"stream": stream, "source_shards": used_shards,
                               "dataset_revision": revision}
        artifact["status"] = "completed"
        persist(output, artifact, "completed")
    except BaseException as error:
        artifact["status"] = "failed"
        artifact["error"] = f"{type(error).__name__}: {error}"
        persist(output, artifact, "failed")
        raise


if __name__ == "__main__":
    main()
