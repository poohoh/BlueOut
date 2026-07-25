"""Extract InstanceDiffusion ID modules for Stage 1 global U-Net training.

Stage 1 training (`configs/outpainting/stage1_global_unet.yaml`)
requires `checkpoints/pretrained/InstanceDiffusion/instancediffusion_modules.pth`,
which contains only the InstanceDiffusion-specific modules of the UNet:

- `position_net.*` (UniFusion grounding tokenizer)
- `scaleu_b_*` / `scaleu_s_*` (ScaleU gates)
- `*.fuser.*` (GatedSelfAttentionDense fusers)

This matches the key classification used by the Stage 1 global U-Net loader.

Usage:

    python scripts/extract_instancediffusion_modules.py \
        --src /path/to/instancediffusion_sd15.pth \
        --out checkpoints/pretrained/InstanceDiffusion/instancediffusion_modules.pth

`--src` accepts the official InstanceDiffusion checkpoint or any checkpoint
whose (possibly nested) state dict contains the UNet weights. The official
full training checkpoint stores non-tensor objects alongside the weights, so
loading it requires passing `--trust-source` (full unpickling; only use with
checkpoints you trust).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

UNET_PREFIX = "model.diffusion_model."


def is_id_module_key(key: str) -> bool:
    return (
        key.startswith("position_net.")
        or key.startswith("scaleu_b_")
        or key.startswith("scaleu_s_")
        or (".fuser." in key)
    )


def load_unet_state_dict(path: Path, trust_source: bool = False) -> Dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as exc:
        if not trust_source:
            raise RuntimeError(
                f"{path} contains non-tensor objects and cannot be loaded with "
                "weights_only=True (the official InstanceDiffusion training checkpoint "
                "is like this). Re-run with --trust-source if you trust this file."
            ) from exc
        obj = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    # Unwrap common nesting: {"state_dict": ...} / {"model": ...}
    for nested_key in ("state_dict", "model"):
        if isinstance(obj, dict) and nested_key in obj and isinstance(obj[nested_key], dict):
            obj = obj[nested_key]
    if not isinstance(obj, dict) or not all(isinstance(k, str) for k in obj):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    # Strip a LatentDiffusion-style UNet prefix if present.
    if any(k.startswith(UNET_PREFIX) for k in obj):
        obj = {k[len(UNET_PREFIX):]: v for k, v in obj.items() if k.startswith(UNET_PREFIX)}
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True,
                        help="Source InstanceDiffusion checkpoint (.pth).")
    parser.add_argument(
        "--out", type=Path,
        default=Path("checkpoints/pretrained/InstanceDiffusion/instancediffusion_modules.pth"),
        help="Output path for the extracted ID-module state dict.",
    )
    parser.add_argument(
        "--trust-source", action="store_true",
        help="Allow full unpickling for checkpoints that fail weights_only loading. "
             "Only use with checkpoints from a trusted source.",
    )
    args = parser.parse_args()

    state_dict = load_unet_state_dict(args.src, trust_source=args.trust_source)
    modules = {k: v.clone() for k, v in state_dict.items() if is_id_module_key(k)}
    if not modules:
        raise ValueError(
            f"No InstanceDiffusion module keys (position_net/scaleu/fuser) found in {args.src}. "
            "Is this an InstanceDiffusion UNet checkpoint?"
        )

    print(f"[extract] source keys: {len(state_dict)}, ID-module keys: {len(modules)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(modules, args.out)
    print(f"[extract] saved: {args.out}")


if __name__ == "__main__":
    main()
