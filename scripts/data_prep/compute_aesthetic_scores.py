"""
Compute CLIP Aesthetic scores for LAION high-resolution dataset.

This script processes all images in datasets/images/laion-high-resolution/
and saves aesthetic scores to datasets/AES_score/laion-high-resolution/
with the same directory structure.

Usage:
  python test/scripts/compute_aesthetic_scores.py \
    --input-root datasets/images/laion-high-resolution \
    --output-root datasets/AES_score/laion-high-resolution \
    --aesthetic-weights assets/aesthetic/aesthetic_v2_clip_vit_l_14_linear.pt \
    --device auto \
    --batch-size 32 \
    --resume

Features:
- Chunk-by-chunk processing for memory efficiency
- Batch processing within each chunk
- Resume capability (skips already processed files)
- Progress tracking with tqdm
- JSON output with filename -> aesthetic_score mapping
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

# Require tqdm for progress bars
try:
    from tqdm.auto import tqdm
except Exception as e:
    raise RuntimeError("tqdm is required for progress display. Please install with: pip install tqdm") from e


_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}


def require_cuda_device(device_arg: str) -> torch.device:
    requested = str(device_arg).strip().lower()
    if requested == 'auto':
        requested = 'cuda'
    if requested == 'cpu':
        raise RuntimeError(
            "compute_aesthetic_scores.py requires CUDA for aesthetic score computation; "
            "--device cpu is not allowed."
        )
    if requested != 'cuda' and not requested.startswith('cuda:'):
        raise ValueError(f"Unsupported --device value for GPU-only aesthetic scoring: {device_arg!r}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for compute_aesthetic_scores.py, but torch.cuda.is_available() is False. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}. "
            "Check that the selected GPU is visible in this environment."
        )
    device = torch.device(requested)
    if device.type != 'cuda':
        raise RuntimeError(f"Resolved non-CUDA device unexpectedly: {device}")
    if device.index is None:
        device = torch.device('cuda:0')
    if device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are visible. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}."
        )
    torch.cuda.set_device(device)
    return device


class ClipBackbone:
    """CLIP model for encoding images to embeddings."""

    def __init__(self, arch: str = 'ViT-L-14', pretrained: str = 'openai', device: torch.device = torch.device('cpu')):
        try:
            import open_clip
        except Exception as e:
            raise RuntimeError("open_clip_torch is required for CLIP. pip install open_clip_torch") from e

        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=device)
        model.eval()

        self.model = model
        self.preprocess = preprocess
        self.device = device

    @torch.no_grad()
    def encode_images_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode a batch of PIL images to normalized CLIP embeddings."""
        if not images:
            return torch.empty(0, self.model.visual.output_dim, device=self.device)

        # Preprocess all images
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)

        # Encode
        features = self.model.encode_image(batch)

        # L2 normalize
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        return features


class AestheticPredictor:
    """Linear head on CLIP image embeddings: score = w^T x + b"""

    def __init__(self, weight: torch.Tensor, bias: float = 0.0, device: torch.device = torch.device('cpu')):
        self.w = weight.to(device).float()
        self.b = float(bias)
        self.device = device

    @classmethod
    def from_file(cls, path: str, device: torch.device):
        """Load aesthetic predictor weights from file."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Aesthetic weights not found: {path}")

        if p.suffix.lower() in ('.pt', '.pth'):
            obj = torch.load(p, map_location='cpu')

            def _tensor(x):
                return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

            w = None
            b = 0.0

            if isinstance(obj, dict):
                # Direct format
                if 'weight' in obj:
                    w = _tensor(obj['weight']).view(-1)
                    b = float(obj.get('bias', 0.0))
                # sklearn-like format
                elif 'coef_' in obj:
                    w = _tensor(obj['coef_']).view(-1)
                    b = float(obj.get('intercept_', 0.0))
                # Nested state_dict
                elif 'state_dict' in obj and isinstance(obj['state_dict'], dict):
                    sd = obj['state_dict']
                    for k, v in sd.items():
                        if k.endswith('weight') and w is None:
                            w = _tensor(v).view(-1)
                        elif k.endswith('bias'):
                            b = float(_tensor(v).view(-1)[0].item())
                    if w is None:
                        raise ValueError('No weight found in state_dict')
                else:
                    # Try to find any weight/bias keys
                    for k, v in obj.items():
                        if isinstance(v, (torch.Tensor, list, tuple)) and 'weight' in k and w is None:
                            w = _tensor(v).view(-1)
                        if isinstance(v, (torch.Tensor, float, int)) and 'bias' in k:
                            b = float(_tensor(v).view(-1)[0].item())
                    if w is None:
                        raise ValueError("Unsupported .pt aesthetic weights format: expected 'weight'/'bias' or a state_dict")
            elif isinstance(obj, (list, tuple)) and len(obj) >= 1:
                w = _tensor(obj[0]).view(-1)
                if len(obj) > 1:
                    b = float(_tensor(obj[1]).view(-1)[0].item())
            else:
                # Raw tensor
                w = _tensor(obj).view(-1)

            return cls(weight=w, bias=float(b), device=device)

        elif p.suffix.lower() in ('.npz', '.npy'):
            import numpy as np
            obj = np.load(p, allow_pickle=True)
            if isinstance(obj, np.lib.npyio.NpzFile):
                w = obj['weight']
                b = obj['bias'][()] if 'bias' in obj else 0.0
            else:
                w = obj
                b = 0.0
            w_t = torch.from_numpy(w).float()
            return cls(weight=w_t, bias=float(b), device=device)

        else:
            raise ValueError(f"Unsupported aesthetic weights extension: {p.suffix}")

    @torch.no_grad()
    def predict_batch(self, clip_features: torch.Tensor) -> List[float]:
        """Predict aesthetic scores for a batch of CLIP features."""
        if clip_features.size(0) == 0:
            return []

        # features: [batch_size, feature_dim]
        scores = torch.matmul(clip_features.float(), self.w.float()) + self.b
        return scores.cpu().tolist()


def get_chunk_dirs(input_root: str) -> List[str]:
    """Get all chunk directories sorted by name."""
    root_path = Path(input_root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    chunk_dirs = []
    for p in sorted(root_path.iterdir()):
        if p.is_dir() and p.name.startswith('chunk_'):
            chunk_dirs.append(p.name)

    return chunk_dirs


def get_image_files(chunk_path: Path) -> List[Path]:
    """Get all image files in a chunk directory."""
    if not chunk_path.is_dir():
        return []

    image_files = []
    for p in sorted(chunk_path.iterdir()):
        if p.is_file() and p.suffix.lower() in _IMG_EXTS:
            image_files.append(p)

    return image_files


def load_existing_scores(output_file: Path) -> Dict[str, float]:
    """Load existing aesthetic scores from JSON file."""
    if not output_file.is_file():
        return {}

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_scores(output_file: Path, scores: Dict[str, float]):
    """Save aesthetic scores to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)


@torch.no_grad()
def process_chunk(
    chunk_name: str,
    input_root: Path,
    output_root: Path,
    clip_model: ClipBackbone,
    aesthetic_predictor: AestheticPredictor,
    batch_size: int = 32,
    resume: bool = True
) -> Tuple[int, int]:
    """Process a single chunk and return (processed_count, total_count)."""

    chunk_input_path = input_root / chunk_name
    output_file = output_root / f"{chunk_name}_aesthetic_scores.json"

    # Get all image files
    image_files = get_image_files(chunk_input_path)
    if not image_files:
        print(f"[Warning] No images found in {chunk_input_path}")
        return 0, 0

    # Load existing scores if resuming
    existing_scores = load_existing_scores(output_file) if resume else {}

    # Filter out already processed files
    pending_files = []
    for img_file in image_files:
        if img_file.name not in existing_scores:
            pending_files.append(img_file)

    if not pending_files:
        print(f"[Info] {chunk_name}: All {len(image_files)} images already processed")
        return len(image_files), len(image_files)

    print(f"[Info] {chunk_name}: Processing {len(pending_files)}/{len(image_files)} images")

    # Process in batches
    all_scores = existing_scores.copy()
    processed_count = 0

    for i in tqdm(range(0, len(pending_files), batch_size),
                  desc=f"{chunk_name}", unit="batch", dynamic_ncols=True):
        batch_files = pending_files[i:i + batch_size]

        # Load and preprocess images
        batch_images = []
        batch_names = []

        for img_file in batch_files:
            try:
                img = Image.open(img_file).convert('RGB')
                batch_images.append(img)
                batch_names.append(img_file.name)
            except Exception as e:
                print(f"[Warning] Failed to load {img_file}: {e}")
                continue

        if not batch_images:
            continue

        # Process entire batch efficiently
        try:
            # Encode entire batch with CLIP (GPU efficient!)
            clip_features = clip_model.encode_images_batch(batch_images)
            aesthetic_scores = aesthetic_predictor.predict_batch(clip_features)

            # Store all batch results
            for img_file, score in zip(batch_files, aesthetic_scores):
                all_scores[img_file.name] = score
                processed_count += 1

            # Save after each batch (balance between safety and efficiency)
            save_scores(output_file, all_scores)

        except Exception as e:
            print(f"[Error] Failed to process batch: {e}")
            # Fallback: process individually for this batch
            for img_file, img in zip(batch_files, batch_images):
                try:
                    clip_features = clip_model.encode_images_batch([img])
                    aesthetic_scores = aesthetic_predictor.predict_batch(clip_features)
                    score = aesthetic_scores[0]
                    all_scores[img_file.name] = score
                    processed_count += 1
                except Exception as e2:
                    print(f"[Error] Failed to process {img_file.name}: {e2}")
                    continue
            # Save after fallback processing
            save_scores(output_file, all_scores)

    total_processed = len(existing_scores) + processed_count
    print(f"[Info] {chunk_name}: Completed {total_processed}/{len(image_files)} images")

    return total_processed, len(image_files)


def main():
    parser = argparse.ArgumentParser(description="Compute CLIP Aesthetic scores for LAION dataset")
    parser.add_argument('--input-root', type=str, required=True,
                       help='Input root directory (e.g., datasets/images/laion-high-resolution)')
    parser.add_argument('--output-root', type=str, required=True,
                       help='Output root directory (e.g., datasets/AES_score/laion-high-resolution)')
    parser.add_argument('--aesthetic-weights', type=str, required=True,
                       help='Path to aesthetic predictor weights')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto|cuda|cuda:N; CPU fallback is disabled')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing images')
    parser.add_argument('--clip-arch', type=str, default='ViT-L-14',
                       help='CLIP architecture')
    parser.add_argument('--clip-pretrained', type=str, default='openai',
                       help='CLIP pretrained weights')
    parser.add_argument('--resume', action='store_true',
                       help='Resume processing (skip already computed scores)')
    parser.add_argument('--chunks', type=str, nargs='+', default=None,
                       help='Process specific chunks only (e.g., chunk_00000 chunk_00001)')

    args = parser.parse_args()

    # Setup device
    device = require_cuda_device(args.device)

    print(f"[Info] Using CUDA device: {device} (visible={torch.cuda.device_count()})")

    # Setup paths
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    # Load models
    print(f"[Info] Loading CLIP model: {args.clip_arch} ({args.clip_pretrained})")
    clip_model = ClipBackbone(arch=args.clip_arch, pretrained=args.clip_pretrained, device=device)

    print(f"[Info] Loading aesthetic predictor: {args.aesthetic_weights}")
    aesthetic_predictor = AestheticPredictor.from_file(args.aesthetic_weights, device=device)

    # Get chunks to process
    if args.chunks:
        chunk_dirs = args.chunks
        print(f"[Info] Processing specified chunks: {len(chunk_dirs)} chunks")
    else:
        chunk_dirs = get_chunk_dirs(str(input_root))
        print(f"[Info] Found {len(chunk_dirs)} chunks to process")

    if not chunk_dirs:
        print("[Warning] No chunks found to process")
        return

    # Process each chunk
    total_processed = 0
    total_images = 0
    start_time = time.time()

    for chunk_name in chunk_dirs:
        try:
            processed, total = process_chunk(
                chunk_name=chunk_name,
                input_root=input_root,
                output_root=output_root,
                clip_model=clip_model,
                aesthetic_predictor=aesthetic_predictor,
                batch_size=args.batch_size,
                resume=args.resume
            )
            total_processed += processed
            total_images += total

        except Exception as e:
            print(f"[Error] Failed to process {chunk_name}: {e}")
            continue

    # Summary
    elapsed_time = time.time() - start_time
    print(f"\n=== Summary ===")
    print(f"Total processed: {total_processed}/{total_images} images")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    if total_processed > 0:
        print(f"Processing rate: {total_processed/elapsed_time:.2f} images/second")
    print(f"Output directory: {output_root}")


if __name__ == '__main__':
    main()
