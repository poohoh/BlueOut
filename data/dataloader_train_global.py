"""
Progressive multi-dataset loader (TRAIN) with explicit conditioning keys for GlobalUNetOutpaintDiffusion.

- Returns fixed-size model-ready tensors (BHWC) only — no legacy metadata
- Requires per-sample annotation JSON (caption)

Model-facing keys
- image: HWC float [-1,1] 512x512 — local GT (first_stage_key)
- mask: HWC float {0,1} 512x512 — local mask (1=to fill)
- masked_image: HWC float [-1,1] 512x512 — local input with unknown region zeroed
- global_image: HWC float [-1,1] 512x512 - global branch RGB (global U-Net input/logging)
- global_mask: HWC float {0,1} 512x512 — letterboxed GLOBAL inpainting mask (1=to fill, 0=known)
- window_mask: HWC float {0,1} 512x512 — letterboxed PATCH‑window mask (현재 로컬 패치 위치만 1)
- txt: str — local prompt for UNet cross-attn (defaults to generic quality prompt if not provided)
- global_prompt: str — global prompt from annotation caption
- annotation_path: str — path to annotation JSON

IconArt is excluded by default (intended for test only).
"""

import os
import json
from pathlib import Path
import base64
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from ProOut.aug.geometry import (
    mask_creation,
    canvas_sweep,
    random_outline_attach,
    get_sticked_window_coordinates,
    hint_pad,
)
from ProOut.aug.safety import pil_maximum_size, pil_minimum_size


def get_attmask_w_box(att_masks: torch.Tensor, idx: int, box, image_size: int) -> torch.Tensor:
    """InstanceDiffusion-compatible bbox attention mask helper.

    The original helper lives in ``InstanceDiffusion.utils.input`` but importing
    that module eagerly pulls in checkpoint/model dependencies.  This keeps the
    same indexing behavior without forcing those unrelated imports in the
    training environment.
    """

    x1 = int(np.round(float(box[0]) * image_size))
    y1 = int(np.round(float(box[1]) * image_size))
    x2 = int(np.round(float(box[2]) * image_size))
    y2 = int(np.round(float(box[3]) * image_size))
    att_masks[idx][x1:x2, y1:y2] = 1
    return att_masks


def _decode_tensor_from_string_id(arr_str: str, use_tensor: bool = True):
    """InstanceDiffusion-style minimal base64 -> float32 decoder.

    Mirrors InstanceDiffusion/dataset/decode_item.py: decode_tensor_from_string.
    No validation or length checks are performed.
    """
    arr = np.frombuffer(base64.b64decode(arr_str), dtype='float32').copy()
    if use_tensor:
        arr = torch.from_numpy(arr)
    return arr


def _decode_b64_embedding_list(raw_list: List[str]) -> torch.Tensor:
    """Decode a list of base64 float32 vectors into a stacked tensor (N, D)."""
    if not raw_list:
        return torch.empty(0, 768, dtype=torch.float32)
    vecs = [_decode_tensor_from_string_id(s, use_tensor=True) for s in raw_list]
    return torch.stack(vecs, dim=0).to(torch.float32)


def _letterbox_apply_params_to_boxes(boxes: List[List[float]], sx: float, sy: float, x_off: float, y_off: float) -> List[List[float]]:
    """Apply letterbox params (sx, sy, offsets) to normalized xyxy boxes.

    boxes: list of [x0,y0,x1,y1] normalized to pre-letterbox image
    returns: list of boxes normalized to letterboxed target (0..1)
    """
    # sx, sy: 컨텐츠가 차지하는 비율
    # x_off, y_off: 전체 가로 및 세로에 대한 왼쪽 및 위쪽의 패딩 비율
    out: List[List[float]] = []
    for b in boxes:
        x0, y0, x1, y1 = map(float, b)
        x0p = x_off + sx * x0
        y0p = y_off + sy * y0
        x1p = x_off + sx * x1
        y1p = y_off + sy * y1
        # clamp to [0,1]
        x0p = 0.0 if x0p < 0.0 else (1.0 if x0p > 1.0 else x0p)
        y0p = 0.0 if y0p < 0.0 else (1.0 if y0p > 1.0 else y0p)
        x1p = 0.0 if x1p < 0.0 else (1.0 if x1p > 1.0 else x1p)
        y1p = 0.0 if y1p < 0.0 else (1.0 if y1p > 1.0 else y1p)
        out.append([x0p, y0p, x1p, y1p])
    return out

# Allowed image extensions
_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}


def _gather_entries(
    images_root: str,
    include_datasets: List[str],
    annotations_root: Optional[str] = None,
    sketch_root: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Collect annotation file paths only (lazy metadata loading).

    Unlike the previous eager implementation, this function no longer opens
    each JSON to read fields. It only discovers and records JSON paths so
    that __getitem__ can load metadata on demand.
    """
    entries: List[Dict[str, Any]] = []

    annotations_root_path = Path(annotations_root).resolve() if annotations_root else None
    sketch_root_path = Path(sketch_root).resolve() if sketch_root else None

    # Basic directory sanity checks (keep behavior consistent)
    if annotations_root_path is None or not annotations_root_path.is_dir():
        raise ValueError("annotations_root must point to a directory containing annotations.")

    target_datasets = include_datasets if include_datasets else [p.name for p in annotations_root_path.iterdir() if p.is_dir()]

    for dataset_name in target_datasets:
        dataset_ann_dir = annotations_root_path / dataset_name
        if not dataset_ann_dir.is_dir():
            raise FileNotFoundError(f"Annotation directory missing for dataset '{dataset_name}': {dataset_ann_dir}")

        # Record JSON paths (sorted for deterministic order)
        for json_path in sorted(dataset_ann_dir.rglob('*.json')):
            entries.append({'annotation_path': str(json_path)})

    if not entries:
        raise ValueError("No training entries were gathered. Check annotations and include_datasets.")
    return entries


class ProgressiveConditionDataset(Dataset):
    def __init__(
        self,
        dataset_root: str = "datasets",  # kept for compatibility (unused)
        mapping_file: str = "",
        image_size: int = 512,
        transform=None,
        use_2d_only: bool = True,  # kept for compatibility (unused)
        excluded_categories: List[str] = None,  # kept for compatibility (unused)
        images_root: str = "datasets/images",
        annotations_root: Optional[str] = "datasets/annotations",
        sketch_root: Optional[str] = None,
        include_datasets: List[str] = None,
        n_max_instances: int = 30,
        embedding_key: str = "text_embedding_before",
        use_instance_attn_mask: bool = False,
    ):
        # Normalize roots (allow relative from repo root)
        self.images_root = images_root
        self.annotations_root = annotations_root
        self.sketch_root = sketch_root
        self.image_size = image_size
        self.n_max_instances = int(n_max_instances)
        self.embedding_key = str(embedding_key)
        self.use_instance_attn_mask = bool(use_instance_attn_mask)

        if include_datasets is None:
            include_datasets = ['humanart', 'laion-high-resolution', 'wikiart']

        # Build entries from multiple datasets
        self.data_items: List[Dict[str, Any]] = _gather_entries(
            images_root=self.images_root,
            include_datasets=include_datasets,
            annotations_root=self.annotations_root,
            sketch_root=self.sketch_root,
        )

        print(f"Progressive Multi-Dataset loaded: {len(self.data_items)} images")

        # not used for model-facing tensors (we prepare tensors explicitly)
        self.transform = transform

    def __len__(self):
        return len(self.data_items)

    def load_image(self, img_path: str) -> Image.Image:
        # img_path is stored as absolute or repo-relative path
        image = Image.open(img_path).convert('RGB')

        return image

    def _prepare_modal_image(self, image: Image.Image, fill: float = 0.5) -> torch.Tensor:
        """Letterbox image to dataset window size and map to [-1, 1]."""
        tensor = TF.to_tensor(image)  # (3, H, W) in [0,1]
        padded = hint_pad(tensor, window_size=self.image_size, fill=[fill, fill, fill])
        padded = padded * 2.0 - 1.0
        return padded.permute(1, 2, 0).contiguous()  # H, W, C

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Lazy metadata loading: open annotation JSON on demand
        info = self.data_items[idx]
        ann_path = info['annotation_path']
        with open(ann_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        # Resolve image path and caption from annotation
        rel_path = meta.get('image_path') or meta.get('img_path') or meta.get('path')
        if not rel_path:
            raise ValueError(f"Annotation file missing image_path: {ann_path}")
        images_root_path = Path(self.images_root).resolve()
        global_img_path = str(images_root_path / rel_path)

        global_prompt = meta.get('caption') or meta.get('prompt')
        if not global_prompt:
            raise ValueError(f"Annotation missing caption: {ann_path}")

        # Extract object-level annotations and embeddings together (aligned by minimal length)
        raw_texts = meta.get('objects', [])
        raw_boxes = meta.get('boxes', [])
        raw_embeds = meta.get(self.embedding_key, [])

        if not len(raw_texts) == len(raw_boxes) == len(raw_embeds):
            raise ValueError(f"Length of texts, boxes, and embeds should be same")

        triplet_len = len(raw_texts)
        ann_texts: List[str] = [str(t) for t in raw_texts[:triplet_len]]
        ann_bboxes: List[List[float]] = [
            [float(b[0]), float(b[1]), float(b[2]), float(b[3])] for b in raw_boxes[:triplet_len]
        ]
        embeds_list = _decode_b64_embedding_list(raw_embeds[:triplet_len])

        # Default local prompt: general qualitative keywords.
        default_local_prompt = "harmonized painting, high resolution, best quality, high quality, harmonized simple background"

        # Keep local prompt default
        local_prompt = default_local_prompt

        # Load original (PIL)
        global_image = self.load_image(global_img_path)


        # ProOut augmentation to obtain: local patch + masks + global hint
        # Additionally, return the true local GT patch and letterbox params
        # global_mask_lb: letterboxed GLOBAL inpainting mask (1=to fill)
        # window_mask_lb: letterboxed PATCH‑window mask (현재 로컬 패치 위치만 1)
        local_rgb, local_mask, hint_rgb, global_mask_lb, window_mask_lb, local_gt, lb_params, window_bbox_lb = self.augment_data(global_image)

        # Map annotation boxes to letterbox coordinates (normalized)
        sx, sy, x_off, y_off = lb_params
        ann_bboxes_lb = _letterbox_apply_params_to_boxes(ann_bboxes, sx, sy, x_off, y_off)

        # Normalize to [-1,1]
        # local_gt_chw is the true GT local patch (from original image)
        local_gt_chw = local_gt * 2.0 - 1.0             # (3,512,512)
        hint_rgb_chw = hint_rgb * 2.0 - 1.0             # (3,512,512)
        
        # Local masked_image: keep known region only (unknown zero) computed from GT
        local_masked_chw = local_gt_chw * (1.0 - local_mask).unsqueeze(0)

        # Convert to HWC for model input interface (BHWC after collate)
        # Use GT local patch as the model target (first_stage_key: 'image')
        local_image_hwc = local_gt_chw.permute(1, 2, 0).contiguous()       # H,W,3
        local_mask_hwc = local_mask.unsqueeze(-1).contiguous()             # H,W,1
        local_masked_hwc = local_masked_chw.permute(1, 2, 0).contiguous()  # H,W,3

        # Global reference tensors
        # - Keep augmented canvas (hint_rgb) for logging/conditioning
        # - Use GLOBAL inpainting mask (1=to fill) letterboxed to KxK for global U-Net input
        global_mask_hwc = global_mask_lb.unsqueeze(-1).contiguous()        # H,W,1 (global inpaint mask)
        window_mask_hwc = window_mask_lb.unsqueeze(-1).contiguous()        # H,W,1 (patch-window mask)
        global_image_hwc = hint_rgb_chw.permute(1, 2, 0).contiguous()      # H,W,3 in [-1,1]
        # Build FULL global image via the same letterbox function (hint_pad)
        global_image_full_hwc = self._prepare_modal_image(global_image, fill=0.5)

        # Letterbox-valid region mask on KxK (1 inside original image area, 0 on padding bars).
        # Use the same normalized letterbox params (sx, sy, x_off, y_off) that map pre-letterbox
        # coordinates to the KxK canvas.
        K = self.image_size
        x0_valid = int(round(x_off * K))
        y0_valid = int(round(y_off * K))
        x1_valid = int(round((x_off + sx) * K))
        y1_valid = int(round((y_off + sy) * K))
        # Clamp to [0, K]
        x0_valid = max(0, min(K, x0_valid))
        y0_valid = max(0, min(K, y0_valid))
        x1_valid = max(0, min(K, x1_valid))
        y1_valid = max(0, min(K, y1_valid))
        letterbox_valid_hw = torch.zeros((K, K), dtype=torch.float32)
        if x1_valid > x0_valid and y1_valid > y0_valid:
            letterbox_valid_hw[y0_valid:y1_valid, x0_valid:x1_valid] = 1.0
        letterbox_valid_hwc = letterbox_valid_hw.unsqueeze(-1).contiguous()  # H,W,1 {0,1}

        # Use direct (x,y,K) from augment_data and letterbox params to produce normalized xyxy on KxK.
        ref_window_bbox = window_bbox_lb.view(1, 4).to(torch.float32)


        # Pack InstanceDiffusion-style fixed-size tensors (N_max padding)
        N = self.n_max_instances
        # Only real instance tokens (no extra special/content token)
        ref_boxes = torch.zeros(N, 4, dtype=torch.float32)
        ref_masks = torch.zeros(N, dtype=torch.float32)
        ref_pos = torch.zeros(N, 768, dtype=torch.float32)

        # fill up to N instance tokens (N may be 0)
        n_ann = min(triplet_len, N)
        if n_ann > 0:
            ref_boxes[:n_ann] = torch.tensor(ann_bboxes_lb[:n_ann], dtype=torch.float32).clamp_(0.0, 1.0)
            ref_masks[:n_ann] = 1.0
            ref_pos[:n_ann] = embeds_list[:n_ann]

        # Build attention masks [N, 64, 64] from normalized boxes (ID-compatible)
        # If disabled, do not include the key so downstream skips masking entirely.
        att_masks = None
        if self.use_instance_attn_mask:
            att_masks = torch.zeros((N, 64, 64), dtype=torch.float32)
            try:
                for i in range(N):
                    # Only fill masks for valid instances
                    if ref_masks[i] > 0.5:
                        att_masks = get_attmask_w_box(att_masks, i, ref_boxes[i].cpu().numpy(), 64)
            except Exception as exc:
                raise RuntimeError(f"Failed to build attn masks from boxes for {global_img_path}: {exc}") from exc

        sample: Dict[str, Any] = {
            # Unet inputs (BHWC after collate)
            'image': local_image_hwc,              # local GT
            'mask': local_mask_hwc,                # local mask
            'masked_image': local_masked_hwc,      # local masked input

            # Global U-Net input
            'global_mask': global_mask_hwc,
            'window_mask': window_mask_hwc,
            'global_image': global_image_hwc,
            'global_image_full': global_image_full_hwc,
            'letterbox_valid_mask': letterbox_valid_hwc,

            # Prompts
            'txt': local_prompt,                   # local prompt (UNet)
            'global_prompt': global_prompt,        # global prompt (global U-Net)
            'annotation_path': ann_path,
            'relative_path': str(Path(rel_path)),

            # grounding inputs (fixed-size, up to N_max instances; no special token)
            'ref_boxes': ref_boxes,                          # [N,4]
            'ref_masks': ref_masks,                          # [N]
            'ref_positive_embeddings': ref_pos,              # [N,768]
            'ref_window_bbox': ref_window_bbox,              # [1,4] normalized xyxy on letterboxed KxK
        }

        if att_masks is not None:
            sample['ref_att_masks'] = att_masks                      # [N,64,64]

        # Also expose texts aligned to boxes for logging/visualization.
        # Keep only the valid portion (n_ann) and pass through (list[str]).
        # Collate will keep this as a list of lists.
        sample['ref_texts'] = ann_texts[:n_ann]

        # Reduce CPU RAM footprint: store tensors as float16 in the DataLoader queue
        for k in (
            'image', 'mask', 'masked_image', 'global_mask', 'window_mask', 'global_image', 'global_image_full',
            'ref_boxes', 'ref_masks', 'ref_positive_embeddings', 'ref_window_bbox'
        ):
            val = sample[k]
            if isinstance(val, torch.Tensor):
                sample[k] = val.half()

        return sample
    
    def augment_data(self, global_image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[float, float, float, float], torch.Tensor]:
        """
        Apply ProOut augmentation to build training sample components.

        Returns:
            local_rgb:      (3,K,K) float in [0,1] from canvas (for reference)
            local_mask:     (K,K) float in {0,1}
            hint_rgb:       (3,K,K) float in [0,1]
            global_mask_lb: (K,K) float in {0,1} - letterboxed GLOBAL inpaint mask (1=to fill)
            window_mask_lb: (K,K) float in {0,1} - letterboxed PATCH-window mask
            local_gt:       (3,K,K) float in [0,1] from original image (true GT)
            lb_params:      (sx, sy, x_off, y_off) for letterbox mapping (pre->post)
            window_bbox_lb: (4,) float in [0,1] normalized xyxy of the PATCH window on letterboxed KxK
        """
        # 1) Safety resize (similar to ProOut/vis.py)
        img = pil_maximum_size(global_image, max_edge_size=2048)
        img = pil_minimum_size(img, min_edge_size=max(1280, int(self.image_size)))

        # To tensor [0,1], CHW
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)  # (3,H,W)

        # 2) Initial canvas and global mask by removing borders with r=0.5
        _, H, W = img_t.shape
        r = 0.5
        remove_h = int(round(H * r // 2))
        remove_w = int(round(W * r // 2))
        canvas = canvas_sweep(img_t, remove_h, remove_w, [0.5])       # (3,H,W) [0,1]
        gmask = mask_creation(H, W, remove_h, remove_w)               # (1,H,W) {0,1}

        # 3) Random attachments (Algorithm 2): simulate arbitrary tau
        # Use the configured training resolution as the attachment patch size.
        K = int(self.image_size)
        N = 32
        q = 0.9
        u = 4
        e = 0.15
        canvas, gmask = random_outline_attach(
            img_t, canvas, gmask,
            remove_h, remove_w,
            max_attach_num=N,
            attach_window=K,
            spare_pixels=u,
            min_crop_ratio=q,
            escape_threshold=e,
        )

        # 4) Position local window (Algorithm 3)
        d = 0.05
        x, y = get_sticked_window_coordinates(canvas, gmask, model_window=K, crop=K, escape_threshold=d)
        local_rgb = canvas[:, y:y+K, x:x+K].clone()     # (3,K,K) from canvas
        local_mask = gmask[0, y:y+K, x:x+K].clone()     # (K,K)
        local_gt = img_t[:, y:y+K, x:x+K].clone()       # (3,K,K) true GT from original

        # 5) Build global hint (letterbox-like pad to square KxK)
        hint_rgb = hint_pad(canvas, window_size=K, fill=[0.5])        # (3,K,K)
        
        # Masks (both letterboxed to KxK)
        global_mask_lb = hint_pad(gmask, window_size=K)[0]            # (K,K)
        # Ensure masks are strictly binary. hint_pad uses bilinear+antialias resizing,
        # which can introduce fractional values at boundaries.
        global_mask_lb = (global_mask_lb > 0.5).to(dtype=global_mask_lb.dtype)
        patch_mask = torch.zeros_like(gmask)                           # (1,H,W)
        patch_mask[:, y:y+K, x:x+K] = 1.0
        window_mask_lb = hint_pad(patch_mask, window_size=K)[0]        # (K,K)
        window_mask_lb = (window_mask_lb > 0.5).to(dtype=window_mask_lb.dtype)

        # Letterbox params (normalized) that map pre-letterbox coords -> target square
        if H > W:
            sx = W / float(H)
            sy = 1.0
            x_off = (1.0 - sx) * 0.5
            y_off = 0.0
        else:
            sx = 1.0
            sy = H / float(W)
            x_off = 0.0
            y_off = (1.0 - sy) * 0.5
        
        # Compute window bbox normalized to pre-letterbox image
        x0 = x / float(W)
        y0 = y / float(H)
        x1 = (x + K) / float(W)
        y1 = (y + K) / float(H)

        # Map to letterboxed normalized coordinates
        x0p = x_off + sx * x0
        y0p = y_off + sy * y0
        x1p = x_off + sx * x1
        y1p = y_off + sy * y1
        window_bbox_lb = torch.tensor([
            max(0.0, min(1.0, x0p)),
            max(0.0, min(1.0, y0p)),
            max(0.0, min(1.0, x1p)),
            max(0.0, min(1.0, y1p)),
        ], dtype=torch.float32)

        return local_rgb, local_mask, hint_rgb, global_mask_lb, window_mask_lb, local_gt, (sx, sy, x_off, y_off), window_bbox_lb



def collate_fn_condition(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}

    tensor_keys = [
        'image',
        'mask',
        'masked_image',
        'global_mask',
        'window_mask',
        'global_image',
        'global_image_full',
        'letterbox_valid_mask',
        'ref_boxes',
        'ref_masks',
        'ref_positive_embeddings',
        'ref_window_bbox',
    ]

    # Optionally include attention masks if dataset provided them
    if 'ref_att_masks' in batch[0]:
        tensor_keys.append('ref_att_masks')

    for key in tensor_keys:
        first_value = batch[0][key]
        if isinstance(first_value, torch.Tensor):
            output[key] = torch.stack([item[key] for item in batch], dim=0)

    # string / misc fields (keep as list)
    output['txt'] = [item['txt'] for item in batch]
    output['global_prompt'] = [item['global_prompt'] for item in batch]
    output['annotation_path'] = [item['annotation_path'] for item in batch]
    output['relative_path'] = [item['relative_path'] for item in batch]
    
    # list[list[str]] with per-instance texts (aligned to boxes order)
    output['ref_texts'] = [item.get('ref_texts', []) for item in batch]


    return output


def create_progressive_condition_dataloader(
    dataset_root: str = "datasets",  # kept for compatibility (unused)
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    image_size: int = 512,
    use_2d_only: bool = True,  # kept for compatibility (unused)
    excluded_categories: List[str] = None,  # kept for compatibility (unused)
    images_root: str = "datasets/images",
    annotations_root: Optional[str] = "datasets/annotations",
    sketch_root: Optional[str] = None,
    include_datasets: List[str] = None,
    n_max_instances: int = 30,
    embedding_key: str = "text_embedding_before",
    prefetch_factor: int = 1,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    use_instance_attn_mask: bool = False,
):
    dataset = ProgressiveConditionDataset(
        dataset_root=dataset_root,
        image_size=image_size,
        use_2d_only=use_2d_only,
        excluded_categories=excluded_categories,
        images_root=images_root,
        annotations_root=annotations_root,
        sketch_root=sketch_root,
        include_datasets=include_datasets,
        n_max_instances=n_max_instances,
        embedding_key=embedding_key,
        use_instance_attn_mask=use_instance_attn_mask,
    )

    dl_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_condition,
        pin_memory=pin_memory,
    )
    # prefetch_factor and persistent_workers are valid only when num_workers>0
    if num_workers and num_workers > 0:
        dl_kwargs['persistent_workers'] = bool(persistent_workers)
        # prefetch_factor must be >=1
        pf = int(prefetch_factor) if prefetch_factor and prefetch_factor > 0 else 1
        dl_kwargs['prefetch_factor'] = pf

    # Prefer pinning directly to CUDA device when supported.  Training is CUDA-only.
    if bool(pin_memory):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this training dataloader; refusing to fall back to CPU.")
        try:
            loader = DataLoader(dataset, pin_memory_device='cuda', **dl_kwargs)
        except TypeError:
            loader = DataLoader(dataset, **dl_kwargs)
    else:
        loader = DataLoader(dataset, **dl_kwargs)
    return loader, dataset


if __name__ == "__main__":
    print("\n=== Progressive Multi-Dataset Condition Dataset Test ===")
    ds = ProgressiveConditionDataset(image_size=512)
    s = ds[0]
    print(f"image(HWC): {tuple(s['image'].shape)}")
    print(f"mask(HWC): {tuple(s['mask'].shape)} range=({s['mask'].min().item():.1f},{s['mask'].max().item():.1f})")
    print(f"masked(HWC): {tuple(s['masked_image'].shape)}")
    print(f"global_image(HWC): {tuple(s['global_image'].shape)}")
    print(f"global_mask(HWC): {tuple(s['global_mask'].shape)}")
    print(f"txt: {s['txt'][:48]}...")

    # Prepare debug helpers and output root (no single-sample saving)
    try:
        import numpy as _np
        from PIL import Image as _Image
        from PIL import ImageDraw as _ImageDraw

        def _to_uint8_rgb(img_hwc_minus1_1: torch.Tensor) -> _np.ndarray:
            arr = img_hwc_minus1_1.detach().cpu().numpy()
            arr = _np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
            return (arr * 255.0).round().astype(_np.uint8)

        def _to_uint8_mask(mask_hwc_01: torch.Tensor) -> _np.ndarray:
            arr = mask_hwc_01.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[..., 0]
            return (arr * 255.0).round().astype(_np.uint8)

        debug_root = os.environ.get("OUTPAINTING_DEBUG_ROOT", os.path.join("debug", "data_augment_reference"))
        os.makedirs(debug_root, exist_ok=True)

        def _overlay_boxes(img_uint8: _np.ndarray, boxes_norm: _np.ndarray, masks: _np.ndarray, color_cycle=None, special_index: int = None) -> _Image:
            """Draw normalized xyxy boxes on uint8 HWC RGB image and return PIL image."""
            if color_cycle is None:
                color_cycle = ["red", "lime", "yellow", "cyan", "magenta", "orange", "blue", "white"]
            pil = _Image.fromarray(img_uint8)
            draw = _ImageDraw.Draw(pil)
            H, W = img_uint8.shape[:2]
            if boxes_norm.ndim == 2 and boxes_norm.shape[-1] == 4:
                n = boxes_norm.shape[0]
                # draw all except special first
                for i in range(n):
                    if special_index is not None and i == special_index:
                        continue
                    if i < masks.shape[0] and float(masks[i]) <= 0.5:
                        continue
                    x0, y0, x1, y1 = boxes_norm[i]
                    x0 = int(round(float(x0) * W)); y0 = int(round(float(y0) * H))
                    x1 = int(round(float(x1) * W)); y1 = int(round(float(y1) * H))
                    draw.rectangle([x0, y0, x1, y1], outline=color_cycle[i % len(color_cycle)], width=2)
                # draw special on top in white bold
                if special_index is not None and 0 <= special_index < n:
                    if special_index < masks.shape[0] and float(masks[special_index]) > 0.5:
                        x0, y0, x1, y1 = boxes_norm[special_index]
                        x0 = int(round(float(x0) * W)); y0 = int(round(float(y0) * H))
                        x1 = int(round(float(x1) * W)); y1 = int(round(float(y1) * H))
                        draw.rectangle([x0, y0, x1, y1], outline="white", width=3)
            return pil
    except Exception as e:
        print(f"[warn] Debug setup failed: {e}")

    print("\n=== DataLoader Test ===")
    # Disable shuffle to align saved indices with dataset ordering
    dl, dset = create_progressive_condition_dataloader(batch_size=2, num_workers=0, shuffle=False)
    b = next(iter(dl))
    print(f"B image: {tuple(b['image'].shape)}")
    print(f"B mask: {tuple(b['mask'].shape)}")
    print(f"B masked: {tuple(b['masked_image'].shape)}")
    print(f"B global_image: {tuple(b['global_image'].shape)}")
    print(f"B global_mask: {tuple(b['global_mask'].shape)}")

    # Debug-save batch[0] visuals for quick inspection
    try:
        b0 = {k: (v[0] if isinstance(v, torch.Tensor) else v[0]) for k, v in b.items()}
        _Image.fromarray(_to_uint8_rgb(b0['image'])).save(os.path.join(debug_root, 'batch0_image.png'))
        _Image.fromarray(_to_uint8_rgb(b0['masked_image'])).save(os.path.join(debug_root, 'batch0_masked_image.png'))
        _Image.fromarray(_to_uint8_mask(b0['mask'])).save(os.path.join(debug_root, 'batch0_mask.png'))
        _Image.fromarray(_to_uint8_rgb(b0['global_image'])).save(os.path.join(debug_root, 'batch0_global_image.png'))
        _Image.fromarray(_to_uint8_mask(b0['global_mask'])).save(os.path.join(debug_root, 'batch0_global_mask.png'))

        with open(os.path.join(debug_root, 'batch0_prompts.txt'), 'w', encoding='utf-8') as f:
            f.write(f"txt: {b0['txt']}\n")
            f.write(f"global_prompt: {b0['global_prompt']}\n")

        # Save global U-Net inputs (boxes overlay + stats)
        try:
            boxes0 = b0['ref_boxes'].detach().cpu().float().numpy()  # (N,4)
            masks0 = b0['ref_masks'].detach().cpu().float().numpy()  # (N,) or (N,1)
            if masks0.ndim == 2 and masks0.shape[1] == 1:
                masks0 = masks0[:, 0]
            img_global0 = _to_uint8_rgb(b0['global_image'])
            special_idx0 = None
            _overlay_boxes(img_global0, boxes0, masks0, special_index=special_idx0).save(os.path.join(debug_root, 'batch0_ref_boxes_overlay.png'))

            # embeddings stats (valid only)
            pos0 = b0['ref_positive_embeddings'].detach().cpu().float()
            n_valid0 = int((torch.as_tensor(masks0) > 0.5).sum().item())
            with open(os.path.join(debug_root, 'batch0_ref_stats.txt'), 'w', encoding='utf-8') as f:
                f.write(f"ref_boxes shape: {tuple(b0['ref_boxes'].shape)}\n")
                f.write(f"ref_masks shape: {tuple(b0['ref_masks'].shape)}\n")
                f.write(f"ref_pos shape: {tuple(b0['ref_positive_embeddings'].shape)}\n")
                f.write(f"valid_tokens: {n_valid0}\n")
                if n_valid0 > 0:
                    norms = pos0[:n_valid0].norm(dim=1).numpy()
                    f.write(f"ref_pos L2 norms (first {min(n_valid0,8)}): {norms[:8].round(3).tolist()}\n")
                # No special content token

            # Save attention masks visualization (sum over instances and content-only)
            try:
                attm0 = b0['ref_att_masks'].detach().cpu().float().numpy()  # (N,64,64)
                # Sum mask (any instance region)
                sum_mask = (attm0.sum(axis=0) > 0).astype(_np.uint8) * 255  # (64,64)
                _Image.fromarray(sum_mask).resize((512, 512), resample=_Image.NEAREST).save(
                    os.path.join(debug_root, 'batch0_ref_att_masks_sum.png')
                )
                # No special content token mask to export
            except Exception as e:
                print(f"[warn] Debug-save (attn masks, batch0) failed: {e}")
            # Save raw embeddings for first sample (optional, small):
            _np.save(os.path.join(debug_root, 'batch0_ref_pos.npy'), pos0.numpy())
        except Exception as e:
            print(f"[warn] Debug-save (ref net inputs, batch0) failed: {e}")

        print(f"Saved batch[0] debug images to: {debug_root}")
    except Exception as e:
        print(f"[warn] Debug-save (batch0) failed: {e}")

    # Debug-save multiple samples from the DataLoader (up to 10)
    try:
        max_save = 10
        saved_dir = os.path.join(debug_root, 'loader_samples')
        os.makedirs(saved_dir, exist_ok=True)

        saved = 0
        for batch in dl:
            B = batch['image'].shape[0]
            for i in range(B):
                if saved >= max_save:
                    break
                prefix = f"loader_{saved:03d}"
                _Image.fromarray(_to_uint8_rgb(batch['image'][i])).save(os.path.join(saved_dir, f"{prefix}_image.png"))
                _Image.fromarray(_to_uint8_rgb(batch['masked_image'][i])).save(os.path.join(saved_dir, f"{prefix}_masked_image.png"))
                _Image.fromarray(_to_uint8_mask(batch['mask'][i])).save(os.path.join(saved_dir, f"{prefix}_mask.png"))
                _Image.fromarray(_to_uint8_rgb(batch['global_image'][i])).save(os.path.join(saved_dir, f"{prefix}_global_image.png"))
                _Image.fromarray(_to_uint8_mask(batch['global_mask'][i])).save(os.path.join(saved_dir, f"{prefix}_global_mask.png"))

                with open(os.path.join(saved_dir, f"{prefix}_prompts.txt"), 'w', encoding='utf-8') as f:
                    # lists of strings of length B
                    f.write(f"txt: {batch['txt'][i]}\n")
                    f.write(f"global_prompt: {batch['global_prompt'][i]}\n")

                # Save global U-Net inputs overlay + stats per sample (lightweight)
                try:
                    boxes_i = batch['ref_boxes'][i].detach().cpu().float().numpy()
                    masks_i = batch['ref_masks'][i].detach().cpu().float().numpy()
                    if masks_i.ndim == 2 and masks_i.shape[1] == 1:
                        masks_i = masks_i[:, 0]
                    img_global_i = _to_uint8_rgb(batch['global_image'][i])
                    special_idx_i = None
                    _overlay_boxes(img_global_i, boxes_i, masks_i, special_index=special_idx_i).save(os.path.join(saved_dir, f"{prefix}_ref_boxes_overlay.png"))
                    n_valid_i = int((torch.as_tensor(masks_i) > 0.5).sum().item())
                    with open(os.path.join(saved_dir, f"{prefix}_ref_stats.txt"), 'w', encoding='utf-8') as f:
                        f.write(f"ref_boxes shape: {tuple(batch['ref_boxes'][i].shape)}\n")
                        f.write(f"ref_masks shape: {tuple(batch['ref_masks'][i].shape)}\n")
                        f.write(f"ref_pos shape: {tuple(batch['ref_positive_embeddings'][i].shape)}\n")
                        f.write(f"valid_tokens: {n_valid_i}\n")
                        # No special content token

                    # Save attn masks (sum and content) per sample
                    try:
                        attm_i = batch['ref_att_masks'][i].detach().cpu().float().numpy()  # (N,64,64)
                        sum_mask_i = (attm_i.sum(axis=0) > 0).astype(_np.uint8) * 255
                        _Image.fromarray(sum_mask_i).resize((512, 512), resample=_Image.NEAREST).save(
                            os.path.join(saved_dir, f"{prefix}_ref_att_masks_sum.png")
                        )
                        # No special content token mask to export
                    except Exception as e:
                        print(f"[warn] Debug-save (attn masks, loader {saved}) failed: {e}")
                except Exception as e:
                    print(f"[warn] Debug-save (ref net inputs, loader {saved}) failed: {e}")

                saved += 1
            if saved >= max_save:
                break
        print(f"Saved {saved} loader samples to: {saved_dir}")
    except Exception as e:
        print(f"[warn] Debug-save (loader samples) failed: {e}")

    print("Done.")
