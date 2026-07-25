"""
IconArt test dataloader for progressive outpainting (inference)

- Loads only IconArt images and BLIP/BLIP-2 captions
- Resizes each image so the long edge is 1K, then center-crops by r (keep 1-r)
- Matches the interface used by test_iconart_controlnet_compel.py:
  { 'orig': PIL.Image, 'crop': PIL.Image, 'txt': str, 'data_keys': str }

Additionally returns:
- center_bbox: (left, top, width, height) on 'orig' for metric masking

Notes
- dataset_root and image_size are accepted for compatibility but unused here.
- Images root is assumed at `datasets/images/iconart`.
- Captions file is assumed at `datasets/caption/iconart/blip2_prompts.json` (JSONL).
- Long-edge resize is downscale-only: images with long edge <= target are kept as-is.
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset, DataLoader


def read_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # skip malformed lines
                continue


class IconArtProgressiveTestDataset(Dataset):
    def __init__(
        self,
        dataset_root: str = "datasets/iconart",  # unused (compat)
        image_size: int = 512,  # unused (compat)
        transform=None,  # unused (compat)
        images_root: str = "datasets/images/iconart",
        captions_file: str = "datasets/caption/iconart/blip2_prompts.json",
        r: float = 0.333,
        target_long_edge: int = 1408,
    ):
        """
        Dataset for IconArt progressive outpainting tests.

        Returns per-item:
        - orig: PIL.Image (RGB) resized to long edge = target_long_edge
        - crop: PIL.Image (RGB) center crop after deleting r/2 per side
        - txt: str (caption prompt)
        - data_keys: str (file stem for naming)
        - center_bbox: (left, top, width, height) on 'orig' for metrics
        """
        self.images_root = images_root.rstrip('/')
        self.captions_file = captions_file
        # normalize r and long-edge target
        try:
            r = float(r)
        except Exception:
            r = 0.333
        self.r = max(1e-6, min(0.999, r))
        self.target_long_edge = int(target_long_edge)
        self.transform = transform
        self.image_size = image_size

        # Build entries: [{'img_path': ..., 'prompt': ..., 'name': ...}, ...]
        self.data_items: List[Dict[str, str]] = []
        missing = 0
        total = 0
        if os.path.isfile(self.captions_file):
            for obj in read_jsonl(self.captions_file):
                fname = obj.get('file')
                prompt = obj.get('prompt', '')
                if not fname:
                    continue
                total += 1
                img_path = os.path.join(self.images_root, fname)
                if os.path.isfile(img_path):
                    name = os.path.splitext(os.path.basename(fname))[0]
                    self.data_items.append({
                        'img_path': img_path,
                        'prompt': prompt,
                        'name': name,
                    })
                else:
                    missing += 1
        else:
            raise FileNotFoundError(f"Captions file not found: {self.captions_file}")

        print(f"IconArt Test Dataset loaded: {len(self.data_items)} images (missing {missing}/{total})")

    def __len__(self):
        return len(self.data_items)

    def load_image(self, path: str) -> Image.Image:
        return Image.open(path).convert('RGB')

    def _resize_long_edge(self, img: Image.Image) -> Image.Image:
        """
        Downscale-only long-edge resize to `self.target_long_edge` while preserving aspect ratio.

        - If the current long edge is <= target, return the image unchanged (no upscaling).
        - If the current long edge is > target, scale both sides by the same factor so that
          max(width, height) == target.
        """
        w, h = img.size
        long_edge = max(w, h)

        # Downscale-only: keep as-is if already within target
        if long_edge <= self.target_long_edge:
            return img

        scale = self.target_long_edge / float(long_edge)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        return img.resize((new_w, new_h), resample=resample)

    @staticmethod
    def _center_crop_by_ratio(img: Image.Image, r: float) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """
        Remove r/2 on each side; keep (1-r) center.
        Returns (crop, bbox) where bbox=(left, top, width, height) in img coords.
        """
        w, h = img.size
        left = int(round((w * r) / 2.0))
        top = int(round((h * r) / 2.0))
        right = w - left
        bottom = h - top
        crop = img.crop((left, top, right, bottom))
        return crop, (left, top, right - left, bottom - top)

    def __getitem__(self, idx: int):
        info = self.data_items[idx]
        img_path = info['img_path']
        prompt = info['prompt']
        name = info['name']

        # Load and resize long edge to 1K
        image = self.load_image(img_path)
        image = self._resize_long_edge(image)
        # r-based center crop (keep 1-r center area)
        center_crop, bbox = self._center_crop_by_ratio(image, self.r)

        return {
            'orig': image,           # PIL.Image (long edge = target_long_edge)
            'crop': center_crop,     # PIL.Image (center keep = 1 - r)
            'txt': prompt,           # str
            'data_keys': name,       # str
            'center_bbox': bbox,     # (left, top, width, height) on 'orig'
        }


def collate_fn_iconart(batch):
    # Keep PIL objects as-is; batch_size is 1 in tests, but support >1
    if len(batch) == 1:
        return batch[0]
    else:
        return {
            'orig': [item['orig'] for item in batch],
            'crop': [item['crop'] for item in batch],
            'txt': [item['txt'] for item in batch],
            'data_keys': [item['data_keys'] for item in batch],
            'center_bbox': [item['center_bbox'] for item in batch],
        }


def create_progressive_condition_dataloader(
    # compatibility args (accepted, but unused for paths)
    dataset_root: str = "datasets/iconart",
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = False,
    image_size: int = 512,  # unused
    images_root: str = "datasets/images/iconart",
    captions_file: str = "datasets/caption/iconart/blip2_prompts.json",
    r: float = 0.333,
    target_long_edge: int = 1408,
):
    """
    Create a DataLoader for IconArt progressive outpainting tests.

    Returns (dataloader, dataset) to mirror training dataloader API.
    """
    dataset = IconArtProgressiveTestDataset(
        dataset_root=dataset_root,
        image_size=image_size,
        images_root=images_root,
        captions_file=captions_file,
        r=r,
        target_long_edge=target_long_edge,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_iconart,
        pin_memory=True,
    )

    return loader, dataset


if __name__ == "__main__":
    # Quick smoke test (does not save files)
    dl, ds = create_progressive_condition_dataloader(batch_size=1, num_workers=0, shuffle=False)
    sample = next(iter(dl))
    print(f"orig size: {sample['orig'].size}, crop size: {sample['crop'].size}")
    print(f"bbox: {sample['center_bbox']}")
    print(f"name: {sample['data_keys']}, prompt: {sample['txt'][:72]}...")
