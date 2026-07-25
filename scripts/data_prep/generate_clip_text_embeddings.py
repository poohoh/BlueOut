"""Attach CLIP text embeddings to annotation JSON files (iconart format).

Each JSON file in ``datasets/annotations/<dataset>/`` contains an ``objects``
list (strings) that describes the boxes stored in ``boxes``.  This script
computes a 768-d CLIP embedding for every string and stores the results in a
parallel list (same length) under ``text_embedding_before`` (configurable).

Usage example::

    python scripts/generate_clip_text_embeddings.py \
        --annotations-root datasets/annotations \
        --instance-key objects \
        --output-key text_embedding_before

Running the script multiple times has no effect unless ``--overwrite`` is
specified.  The annotation structure itself is not otherwise modified.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor


def encode_tensor_as_base64(arr: torch.Tensor | np.ndarray) -> str:
    """Encode a tensor/array of dtype float32 into a base64 string."""

    if isinstance(arr, torch.Tensor):
        arr = arr.detach().to(torch.float32).cpu().numpy()
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return base64.b64encode(arr.tobytes()).decode("utf-8")


def get_clip_text_embedding(
    model: CLIPModel,
    processor: CLIPProcessor,
    text: str,
    device: torch.device,
) -> torch.Tensor:
    """Return the 768-d CLIP text embedding for ``text``."""

    if not text:
        raise ValueError("Text prompt is empty; cannot compute embedding.")

    # Match ID's usage: use CLIPProcessor and include a placeholder pixel_values
    inputs = processor(text=[text], padding=True, return_tensors="pt")
    # Move tensors to device
    inputs["input_ids"] = inputs["input_ids"].to(device)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].to(device)
    # Placeholder pixel values (not used by text tower but included in ID code path)
    inputs["pixel_values"] = torch.ones(1, 3, 224, 224, device=device)
    with torch.no_grad():
        outputs = model(**inputs)
        if not hasattr(outputs, "text_model_output"):
            raise RuntimeError("CLIP model output missing 'text_model_output'.")
        embedding = outputs.text_model_output.pooler_output[0]  # (768,)
    return embedding.detach().to(torch.float32)


def iter_json_files(root: Path, pattern: str) -> Iterable[Path]:
    """Yield JSON file paths under ``root`` that match ``pattern``."""

    if root.is_file() and root.suffix == ".json":
        yield root
        return

    for path in sorted(root.rglob(pattern)):
        if path.suffix == ".json" and path.is_file():
            yield path


def process_file(
    json_path: Path,
    instance_key: str,
    output_key: str,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
    overwrite: bool,
    verbose: bool,
) -> bool:
    """Process one JSON file (iconart schema). Return True if modified."""

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    instances = data.get(instance_key)
    if not isinstance(instances, list):
        if verbose:
            print(f"[skip] '{json_path}' has no list under '{instance_key}'.")
        return False

    if instances and not isinstance(instances[0], str):
        if verbose:
            print(f"[skip] '{json_path}' expects strings under '{instance_key}'.")
        return False

    existing = data.get(output_key)
    if (
        not overwrite
        and isinstance(existing, list)
        and len(existing) == len(instances)
        and all(isinstance(x, str) for x in existing)
    ):
        if verbose:
            print(f"[skip] '{json_path}' already has '{output_key}'.")
        return False

    embeddings: list[str] = []
    for text in instances:
        # Preserve original text exactly (no strip) to mirror ID behavior
        if not text:
            embeddings.append("")
            continue
        embedding = get_clip_text_embedding(clip_model, clip_processor, text, device)
        embeddings.append(encode_tensor_as_base64(embedding))

    data[output_key] = embeddings

    tmp_path = json_path.with_suffix(".tmp.json")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, json_path)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add CLIP text embeddings to annotation JSONs")
    parser.add_argument(
        "--annotations-root",
        type=str,
        required=True,
        help="Directory (or single JSON file) containing annotations.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Glob pattern used when scanning directories (default: *.json)",
    )
    parser.add_argument(
        "--instance-key",
        type=str,
        default="objects",
        help="Key containing the list of object labels (default: objects)",
    )
    parser.add_argument(
        "--output-key",
        type=str,
        default="text_embedding_before",
        help="Field name used to store the base64 embedding (default: text_embedding_before)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute embeddings even if the output key already exists",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="HuggingFace model identifier for CLIP (default: openai/clip-vit-large-patch14)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (e.g., cuda:0). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    annotations_root = Path(args.annotations_root).resolve()
    if not annotations_root.exists():
        raise FileNotFoundError(f"annotations_root does not exist: {annotations_root}")

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    if args.verbose:
        print(f"Loading CLIP model '{args.model}' on {device} ...")

    clip_model = CLIPModel.from_pretrained(args.model).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(args.model)

    json_files = list(iter_json_files(annotations_root, args.pattern))
    if args.verbose:
        print(f"Found {len(json_files)} JSON files under {annotations_root}.")

    updated = 0
    for path in json_files:
        if args.verbose:
            print(f"Processing {path} ...", end=" ")
        try:
            modified = process_file(
                path,
                instance_key=args.instance_key,
                output_key=args.output_key,
                clip_model=clip_model,
                clip_processor=clip_processor,
                device=device,
                overwrite=args.overwrite,
                verbose=args.verbose,
            )
        except Exception as exc:  # noqa: BLE001
            if args.verbose:
                print(f"error: {exc}")
            else:
                print(f"[error] {path}: {exc}")
            continue

        if modified:
            updated += 1
            if args.verbose:
                print("updated")
        elif args.verbose:
            print("skipped")

    print(f"Done. Updated {updated} file(s) out of {len(json_files)}.")


if __name__ == "__main__":
    main()
