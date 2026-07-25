import argparse
import json
import os
import re
from typing import Iterable, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# LAVIS provides BLIP / BLIP-2 models and preprocessors
from lavis.models import load_model_and_preprocess


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def is_image(name: str) -> bool:
    return os.path.splitext(name.lower())[1] in IMAGE_EXTS


def regex_cleanup(text: str) -> str:
    """Apply the same regex cleanups used in the original script."""
    patterns = [
        r'^((a )?(\w+\s+)*(?:painting|drawing|mural|artwork|view|portrait|photo|photograph|watercolor|illustration|picture|digital art) of )+',
        r'in (a )?(\w+\s+)*(?:painting|drawing|mural|artwork|view|portrait|photo|photograph|watercolor|illustration|picture|digital art)\s?',
        r'^(?:painting|photograph|digital art|black and white photograph) - ',
    ]
    for p in patterns:
        text = re.sub(p, "", text)
    return text.strip()


def load_caption_model(device: torch.device):
    """Load captioning model and preprocessors.

    Switched to BLIP (base_coco) per request. Previous BLIP-2 config is
    kept below as commented lines for easy toggling.
    """
    # BLIP - 1 (active)
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=True, device=device
    )

    # BLIP - 2 (previous)
    # model, vis_processors, _ = load_model_and_preprocess(
    #     name="blip2_opt", model_type="caption_coco_opt6.7b", is_eval=True, device=device
    # )

    return model, vis_processors



def _anchor_subpath(path: str, anchor: str = "datasets") -> str:
    """Return subpath of `path` after the first occurrence of `anchor`.

    If `anchor` is not found, fall back to basename of `path`.
    """
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)
    if anchor in parts:
        idx = parts.index(anchor)
        sub = os.path.join(*parts[idx + 1 :]) if idx + 1 < len(parts) else ""
        return sub or os.path.basename(path)
    return os.path.basename(path)


def iter_image_dirs(root: str) -> Iterable[Tuple[str, List[str]]]:
    """Yield (dirpath, [image_file_names]) for directories containing images under `root`."""
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        images = [fn for fn in filenames if is_image(fn)]
        if images:
            yield dirpath, sorted(images)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Caption images in directories (no tar). Mirrors input tree under "
            "datasets/caption by default. By default stores one <leaf>_blip2_prompts.json "
            "in the parent directory of each leaf image folder (use --per-dir folder for legacy)."
        )
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Directory containing images (recursively)",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Explicit output root. If unset, outputs mirror the source tree "
            "under --output-base (default datasets/caption)."
        ),
    )
    parser.add_argument(
        "--output-base",
        default=os.path.join("datasets", "caption"),
        help=(
            "Base directory where the source directory structure is mirrored. "
            "Used only when --output-root is not provided."
        ),
    )
    parser.add_argument(
        "--per-dir",
        choices=["json", "folder"],
        default="json",
        help=(
            "json: write <leaf>_blip2_prompts.json in the parent dir of each leaf (default). "
            "folder: write blip2_prompts.json inside each leaf directory (legacy)."
        ),
    )
    parser.add_argument("--resume", action="store_true", help="Append to existing outputs and skip processed files")
    parser.add_argument("--max-samples", type=int, default=None, help="Debug: cap number of samples per directory")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    model, vis_processors = load_caption_model(device)

    data_root = os.path.abspath(args.data_root)

    # Determine base output directory, mirroring the source tree under datasets/caption by default.
    if args.output_root:
        out_root = args.output_root
    else:
        subpath = _anchor_subpath(data_root, anchor="datasets")
        out_root = os.path.join(args.output_base, subpath)
    os.makedirs(out_root, exist_ok=True)

    found_any = False

    for dirpath, image_names in iter_image_dirs(data_root):
        found_any = True

        # Determine output placement per directory mode
        rel_dir = os.path.relpath(dirpath, start=data_root)
        if args.per_dir == "folder":
            # Legacy behavior: write inside the leaf directory mirror
            target_dir = os.path.normpath(os.path.join(out_root, rel_dir))
            os.makedirs(target_dir, exist_ok=True)
            out_path = os.path.join(target_dir, "blip2_prompts.json")
        else:
            # Default behavior: write one file per leaf in its parent's mirror
            leaf = os.path.basename(os.path.normpath(dirpath))
            parent_rel = os.path.relpath(os.path.dirname(dirpath), start=data_root)
            target_dir = os.path.normpath(os.path.join(out_root, parent_rel))
            os.makedirs(target_dir, exist_ok=True)
            out_path = os.path.join(target_dir, f"{leaf}_blip2_prompts.json")

        processed = set()
        mode = "a" if args.resume and os.path.exists(out_path) else "w"
        if args.resume and os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as r:
                for line in r:
                    try:
                        processed.add(json.loads(line).get("file"))
                    except Exception:
                        # ignore malformed lines
                        continue

        with open(out_path, mode, encoding="utf-8") as fout:
            count = 0
            desc = rel_dir if rel_dir != "." else os.path.basename(data_root.rstrip(os.sep))
            for name in tqdm(image_names, desc=desc):
                if args.max_samples is not None and count >= args.max_samples:
                    break

                if args.resume and name in processed:
                    continue

                src_path = os.path.join(dirpath, name)
                try:
                    raw_img = Image.open(src_path).convert("RGB")
                except Exception:
                    # skip unreadable images
                    continue

                image = vis_processors["eval"](raw_img).unsqueeze(0).to(device)
                with torch.inference_mode():
                    caption = model.generate({"image": image})[0]
                caption = regex_cleanup(caption)

                record = {"file": name, "prompt": caption}
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

                count += 1

    if not found_any:
        raise FileNotFoundError(f"No images found under {data_root}")


if __name__ == "__main__":
    main()
