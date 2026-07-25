"""
Ours (Multi-GPU): full-model parallel outpainting entrypoint.

Canonical BlueOut setting:
- cond_only local bank injection is fixed in code
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.models.attention import GatedSelfAttentionDense
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
# Reuse AlignNoise AAM modules/functions as-is.
sys.path.insert(0, os.path.join(REPO_ROOT, "previous_works/AlignNoise"))

from common.file_utils import create_output_structure, parse_order, stack_side_by_side_centered  # noqa: E402
from common.image_processing import to_image_uint8, torch_letterbox, torch_unletter  # noqa: E402
from data.dataloader_test import create_progressive_condition_dataloader  # noqa: E402
from diffusers_local.models import INSTDIFFTextBoundingboxProjectionBBoxOnly, build_main_unet_no_scaleu  # noqa: E402
from diffusers_local.pipelines.instdiff_inpaint_pipeline import convert_ldm_unet_checkpoint  # noqa: E402
from diffusers_local.pipelines import StableDiffusionINSTDIFFInpaintPipelineBBoxOnly  # noqa: E402
from diffusers_local.training import DiffusersGlobalUNetFuserWriter, DiffusersUNetFuserBankInjector  # noqa: E402
from utils.attn_utils import fn_smoothing_func  # noqa: E402
from utils.ptp_utils import AttendExciteAttnProcessor, AttentionStore  # noqa: E402


@dataclass(frozen=True)
class PatchWindow:
    name: str
    x: int
    y: int


@dataclass(frozen=True)
class GlobalLayoutCondition:
    texts: List[str]
    boxes_norm: np.ndarray
    positive_embeddings: torch.Tensor
    orig_w: int
    orig_h: int
    crop_left: int
    crop_top: int


@dataclass
class RefBankCache:
    cond_per_step: List[Dict[int, torch.Tensor]]
    uncond_per_step: Optional[List[Dict[int, torch.Tensor]]]


@dataclass
class AAMEmptyAttnMonitor:
    rank: int
    verbose: bool = True
    events: List[Dict[str, Any]] = field(default_factory=list)
    _warned_keys: Set[Tuple[str, str, int]] = field(default_factory=set, init=False, repr=False)

    def record(
        self,
        *,
        sample_id: str,
        branch: str,
        outer_iter: int,
        timestep: int,
        error_type: str,
    ) -> None:
        event = {
            "sample_id": str(sample_id),
            "branch": str(branch),
            "outer_iter": int(outer_iter),
            "timestep": int(timestep),
            "error_type": str(error_type),
            "rank": int(self.rank),
        }
        self.events.append(event)

        warn_key = (str(sample_id), str(branch), int(timestep))
        if self.verbose and warn_key not in self._warned_keys:
            self._warned_keys.add(warn_key)
            tqdm.write(
                "[Warn][AAMEmptyAttn] "
                f"sample={sample_id} branch={branch} timestep={timestep} "
                f"iter={outer_iter} error={error_type}"
            )

    def build_payload(self) -> Dict[str, Any]:
        by_branch: Dict[str, int] = {}
        by_sample: Dict[str, Dict[str, Any]] = {}

        for event in self.events:
            branch = str(event["branch"])
            sample_id = str(event["sample_id"])
            by_branch[branch] = int(by_branch.get(branch, 0)) + 1

            sample_bucket = by_sample.setdefault(
                sample_id,
                {
                    "count": 0,
                    "branches": {},
                    "timesteps": [],
                },
            )
            sample_bucket["count"] = int(sample_bucket["count"]) + 1
            sample_branches = sample_bucket["branches"]
            sample_branches[branch] = int(sample_branches.get(branch, 0)) + 1
            sample_bucket["timesteps"].append(int(event["timestep"]))

        for sample_bucket in by_sample.values():
            sample_bucket["timesteps"] = sorted(set(int(t) for t in sample_bucket["timesteps"]))

        return {
            "event_count": int(len(self.events)),
            "sample_count": int(len(by_sample)),
            "by_branch": by_branch,
            "by_sample": by_sample,
            "events": self.events,
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.build_payload()
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")


def _sanitize_path_stem(stem: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(stem).strip())
    return s if s else "sample"


def _save_u8_rgb(path: Path, image_u8: np.ndarray) -> None:
    Image.fromarray(np.ascontiguousarray(image_u8), mode="RGB").save(path)


def _compose_masked_input_u8(rgb_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(rgb_u8).copy()
    out[np.ascontiguousarray(mask_u8) > 0] = 0
    return out


def _build_gt_canvas(
    *,
    orig_u8: Optional[np.ndarray],
    center_bbox: Optional[Tuple[int, int, int, int]],
    canvas_w: int,
    canvas_h: int,
    cx: int,
    cy: int,
) -> Optional[np.ndarray]:
    if orig_u8 is None or center_bbox is None:
        return None
    if orig_u8.ndim != 3 or orig_u8.shape[-1] != 3:
        return None

    left, top, _cw, _ch = [int(v) for v in center_bbox]
    off_x = int(cx) - int(left)
    off_y = int(cy) - int(top)

    canvas = np.zeros((int(canvas_h), int(canvas_w), 3), dtype=np.uint8)
    src_h, src_w = int(orig_u8.shape[0]), int(orig_u8.shape[1])

    dst_x0 = max(0, off_x)
    dst_y0 = max(0, off_y)
    dst_x1 = min(int(canvas_w), off_x + src_w)
    dst_y1 = min(int(canvas_h), off_y + src_h)
    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return canvas

    src_x0 = dst_x0 - off_x
    src_y0 = dst_y0 - off_y
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = orig_u8[src_y0:src_y1, src_x0:src_x1]
    return canvas


def _letterbox_u8_rgb(image_u8: np.ndarray, target_size: int) -> np.ndarray:
    img_t = torch.from_numpy(np.ascontiguousarray(image_u8)).to(dtype=torch.float32) / 255.0
    img_lb, _s, _p = torch_letterbox(img_t, target_size=int(target_size), mode="bilinear")
    return (img_lb.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8).cpu().numpy()


def _center_crop_or_pad_u8(image_u8: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    src_h, src_w = int(image_u8.shape[0]), int(image_u8.shape[1])
    target_w = int(target_w)
    target_h = int(target_h)
    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    src_x0 = max(0, (src_w - target_w) // 2)
    src_y0 = max(0, (src_h - target_h) // 2)
    src_x1 = min(src_w, src_x0 + target_w)
    src_y1 = min(src_h, src_y0 + target_h)
    crop = np.ascontiguousarray(image_u8[src_y0:src_y1, src_x0:src_x1])
    if crop.size == 0:
        return out

    crop_h, crop_w = int(crop.shape[0]), int(crop.shape[1])
    dst_x0 = max(0, (target_w - crop_w) // 2)
    dst_y0 = max(0, (target_h - crop_h) // 2)
    out[dst_y0 : dst_y0 + crop_h, dst_x0 : dst_x0 + crop_w] = crop
    return out


def _snap_overlap_to_grid(
    *,
    target_px: float,
    patch_size: int,
    center_size: int,
    grid: int = 8,
    axis: str = "x",
) -> int:
    patch_size = int(patch_size)
    center_size = int(center_size)
    grid = int(grid)
    if patch_size <= (2 * grid):
        raise ValueError(f"patch_size must be > 2*grid ({2 * grid}), got patch_size={patch_size}, grid={grid}")

    min_required = int(math.floor((patch_size - center_size) / 2.0)) + 1
    min_required = max(1, min_required)

    max_from_canvas = int((3 * patch_size - center_size) // 2)
    lower_req = max(grid, min_required)
    upper_req = min(patch_size - grid, max_from_canvas)

    lower = int(math.ceil(float(lower_req) / float(grid))) * grid
    upper = int(math.floor(float(upper_req) / float(grid))) * grid
    if lower > upper:
        raise ValueError(
            f"Cannot satisfy {axis}-overlap constraints with 8px grid: "
            f"patch_size={patch_size}, center_size={center_size}, lower={lower}, upper={upper}"
        )

    ceil_snap = int(math.ceil(float(target_px) / float(grid))) * grid
    ov = min(max(ceil_snap, lower), upper)
    ov = min(max(ov, lower), upper)
    return int(ov)


def _draw_layout_condition_u8(
    base_u8: np.ndarray,
    layout_boxes_norm: np.ndarray,
    window_bbox_norm: Optional[np.ndarray] = None,
) -> np.ndarray:
    img = Image.fromarray(np.ascontiguousarray(base_u8), mode="RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    def _to_px(box: np.ndarray) -> Tuple[int, int, int, int]:
        x0 = int(round(float(np.clip(box[0], 0.0, 1.0)) * float(w - 1)))
        y0 = int(round(float(np.clip(box[1], 0.0, 1.0)) * float(h - 1)))
        x1 = int(round(float(np.clip(box[2], 0.0, 1.0)) * float(w - 1)))
        y1 = int(round(float(np.clip(box[3], 0.0, 1.0)) * float(h - 1)))
        if x1 <= x0:
            x1 = min(w - 1, x0 + 1)
        if y1 <= y0:
            y1 = min(h - 1, y0 + 1)
        return x0, y0, x1, y1

    for box in np.asarray(layout_boxes_norm, dtype=np.float32):
        draw.rectangle(_to_px(box), outline=(255, 96, 96), width=2)

    if window_bbox_norm is not None:
        draw.rectangle(_to_px(np.asarray(window_bbox_norm, dtype=np.float32)), outline=(96, 224, 96), width=3)

    return np.asarray(img, dtype=np.uint8)


def _save_global_debug_views(
    *,
    out_dir: Path,
    stage_idx: int,
    masked_input_u8: np.ndarray,
    ground_truth_u8: Optional[np.ndarray],
    layout_boxes_norm: np.ndarray,
    generated_u8: np.ndarray,
) -> None:
    stage_tag = f"g{int(stage_idx):02d}"
    gt_u8 = masked_input_u8 if ground_truth_u8 is None else ground_truth_u8
    layout_u8 = _draw_layout_condition_u8(gt_u8, layout_boxes_norm, window_bbox_norm=None)

    _save_u8_rgb(out_dir / f"{stage_tag}_masked_input.png", masked_input_u8)
    _save_u8_rgb(out_dir / f"{stage_tag}_ground_truth.png", gt_u8)
    _save_u8_rgb(out_dir / f"{stage_tag}_layout_condition.png", layout_u8)
    _save_u8_rgb(out_dir / f"{stage_tag}_generated.png", generated_u8)


def _save_local_debug_input_views(
    *,
    out_dir: Path,
    stage_idx: int,
    local_idx: int,
    local_key: str,
    masked_input_u8: np.ndarray,
    ground_truth_u8: np.ndarray,
) -> str:
    tag = f"g{int(stage_idx):02d}_i{int(local_idx):02d}_{str(local_key)}"
    _save_u8_rgb(out_dir / f"{tag}_masked_input.png", masked_input_u8)
    _save_u8_rgb(out_dir / f"{tag}_ground_truth.png", ground_truth_u8)
    return tag


def _save_local_debug_output_view(*, out_dir: Path, tag: str, generated_u8: np.ndarray) -> None:
    _save_u8_rgb(out_dir / f"{tag}_generated.png", generated_u8)


def plan_windows_band_ratio(
    patch_size: int,
    overlap_ratio_x: float,
    overlap_ratio_y: float,
    center_w: int,
    center_h: int,
    order: List[str],
) -> Tuple[Dict[str, PatchWindow], int, int, int, int]:
    assert 0.0 < float(overlap_ratio_x) < 1.0 and 0.0 < float(overlap_ratio_y) < 1.0
    ovx_px = int(round(patch_size * float(overlap_ratio_x)))
    ovy_px = int(round(patch_size * float(overlap_ratio_y)))
    assert 0 < ovx_px < patch_size and 0 < ovy_px < patch_size

    sx = patch_size - ovx_px
    sy = patch_size - ovy_px
    canvas_w = 3 * patch_size - 2 * ovx_px
    canvas_h = 3 * patch_size - 2 * ovy_px

    x0, x1, x2 = 0, sx, 2 * sx
    y0, y1, y2 = 0, sy, 2 * sy

    dx = patch_size - center_w
    cx = x1 + (dx // 2) if dx >= 0 else x1 - ((-dx + 1) // 2)
    dy = patch_size - center_h
    cy = y1 + (dy // 2) if dy >= 0 else y1 - ((-dy + 1) // 2)

    if not (0 <= cx <= (canvas_w - center_w) // 2):
        max_ovx_px = (3 * patch_size - center_w) // 2
        max_ratio_x = max(0.0, min(0.999, max_ovx_px / float(patch_size)))
        raise AssertionError(
            f"center_w={center_w} doesn't fit canvas_w={canvas_w}. reduce overlap_ratio_x to {max_ratio_x}."
        )
    if not (0 <= cy <= (canvas_h - center_h) // 2):
        max_ovy_px = (3 * patch_size - center_h) // 2
        max_ratio_y = max(0.0, min(0.999, max_ovy_px / float(patch_size)))
        raise AssertionError(
            f"center_h={center_h} doesn't fit canvas_h={canvas_h}. reduce overlap_ratio_y to {max_ratio_y}."
        )

    gap_x = patch_size - 2 * ovx_px
    gap_y = patch_size - 2 * ovy_px
    if center_w <= gap_x:
        min_overlap_ratio_x = (patch_size - center_w) / (2.0 * patch_size)
        min_overlap_ratio_x = max(0.0, min(0.999, min_overlap_ratio_x))
        raise AssertionError(
            f"center_w={center_w} is too small to overlap with edge patches. "
            f"Gap between patches: {gap_x}. Minimum center_w: {gap_x + 1}. "
            f"minimum overlap ratio: {min_overlap_ratio_x}"
        )
    if center_h <= gap_y:
        min_overlap_ratio_y = (patch_size - center_h) / (2.0 * patch_size)
        min_overlap_ratio_y = max(0.0, min(0.999, min_overlap_ratio_y))
        raise AssertionError(
            f"center_h={center_h} is too small to overlap with edge patches. "
            f"Gap between patches: {gap_y}. Minimum center_h: {gap_y + 1}. "
            f"minimum overlap ratio: {min_overlap_ratio_y}"
        )

    positions = {
        "C": (x1, y1),
        "N": (x1, y0),
        "S": (x1, y2),
        "W": (x0, y1),
        "E": (x2, y1),
        "NW": (x0, y0),
        "NE": (x2, y0),
        "SW": (x0, y2),
        "SE": (x2, y2),
    }

    valid = {"N", "S", "W", "E", "NW", "NE", "SW", "SE"}
    if any(k not in valid for k in order):
        raise ValueError(f"order contains invalid keys: {order}")

    windows = {k: PatchWindow(k, *v) for k, v in positions.items()}
    return windows, canvas_w, canvas_h, cx, cy


class PriorityComposer:
    """Canvas compositor with selectable first-wins / blend modes."""

    def __init__(self, canvas_h: int, canvas_w: int):
        self.filled_by = np.full((canvas_h, canvas_w), -1, dtype=np.int16)
        self.blend_weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    def mark_center(self, x: int, y: int, h: int, w: int) -> None:
        self.filled_by[y : y + h, x : x + w] = 0
        self.blend_weight[y : y + h, x : x + w] = 1.0

    def composite(
        self,
        canvas: np.ndarray,  # (H,W,3) uint8
        out_patch: np.ndarray,  # (H,W,3) uint8
        mask_fill: np.ndarray,  # (H,W) uint8 in {0,255}; 255 == fill
        x: int,
        y: int,
        pid: int,
        mode: str = "first_wins",
        blend_feather_radius: int = 2,
        blend_feather_strength: float = 0.35,
    ) -> None:
        mode_key = str(mode).strip().lower()
        if mode_key == "first_wins":
            self.composite_first_wins(
                canvas=canvas,
                out_patch=out_patch,
                mask_fill=mask_fill,
                x=x,
                y=y,
                pid=pid,
            )
            return
        if mode_key == "blend":
            self.composite_blend(
                canvas=canvas,
                out_patch=out_patch,
                mask_fill=mask_fill,
                x=x,
                y=y,
                pid=pid,
                blend_feather_radius=blend_feather_radius,
                blend_feather_strength=blend_feather_strength,
            )
            return
        raise ValueError(f"Unknown final composite mode: {mode}. expected 'first_wins' or 'blend'")

    def composite_first_wins(
        self,
        canvas: np.ndarray,  # (H,W,3) uint8
        out_patch: np.ndarray,  # (H,W,3) uint8
        mask_fill: np.ndarray,  # (H,W) uint8 in {0,255}; 255 == fill
        x: int,
        y: int,
        pid: int,
    ) -> None:
        h, w = int(out_patch.shape[0]), int(out_patch.shape[1])
        region_filled = self.filled_by[y : y + h, x : x + w]
        region_canvas = canvas[y : y + h, x : x + w]

        eligible = (mask_fill > 0) & (region_filled == -1)
        if not np.any(eligible):
            return

        region_canvas[eligible] = out_patch[eligible]
        region_filled[eligible] = np.int16(pid)

    def composite_blend(
        self,
        canvas: np.ndarray,  # (H,W,3) uint8
        out_patch: np.ndarray,  # (H,W,3) uint8
        mask_fill: np.ndarray,  # (H,W) uint8 in {0,255}; 255 == fill
        x: int,
        y: int,
        pid: int,
        *,
        blend_feather_radius: int = 2,
        blend_feather_strength: float = 0.35,
    ) -> None:
        h, w = int(out_patch.shape[0]), int(out_patch.shape[1])
        region_filled = self.filled_by[y : y + h, x : x + w]
        region_canvas = canvas[y : y + h, x : x + w]
        region_weight = self.blend_weight[y : y + h, x : x + w]

        fill = mask_fill > 0
        if not np.any(fill):
            return

        weights = fill.astype(np.float32)
        radius = max(0, int(blend_feather_radius))
        strength = float(np.clip(float(blend_feather_strength), 0.0, 1.0))
        if radius > 0 and strength > 0.0:
            mask_u8 = np.ascontiguousarray(fill.astype(np.uint8) * np.uint8(255))
            blurred = np.asarray(
                Image.fromarray(mask_u8, mode="L").filter(ImageFilter.GaussianBlur(radius=float(radius))),
                dtype=np.float32,
            ) / 255.0
            # Allow a thin seam band to mix with already-filled neighbors.
            seam_band = (~fill) & (region_filled >= 0) & (blurred > 0.0)
            if np.any(seam_band):
                weights[seam_band] = np.maximum(weights[seam_band], blurred[seam_band] * strength)

        apply = weights > 0.0
        if not np.any(apply):
            return

        old_w = region_weight[apply]
        add_w = weights[apply]
        total_w = old_w + add_w

        canvas_f32 = region_canvas.astype(np.float32)
        patch_f32 = out_patch.astype(np.float32)
        mixed = (canvas_f32[apply] * old_w[:, None] + patch_f32[apply] * add_w[:, None]) / np.maximum(
            total_w[:, None], 1e-6
        )
        region_canvas[apply] = np.clip(np.round(mixed), 0, 255).astype(np.uint8)
        region_weight[apply] = total_w

        newly_filled = fill & (region_filled == -1)
        if np.any(newly_filled):
            region_filled[newly_filled] = np.int16(pid)


def build_local_input(
    canvas: np.ndarray,
    composer: PriorityComposer,
    x: int,
    y: int,
    patch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    local_rgb = canvas[y : y + patch_size, x : x + patch_size].copy()
    local_filled = composer.filled_by[y : y + patch_size, x : x + patch_size]
    local_mask = (local_filled == -1).astype(np.uint8) * np.uint8(255)
    return local_rgb, local_mask


def _resolve_torch_dtype(precision: str) -> torch.dtype:
    precision = str(precision).lower()
    if precision == "fp16":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if precision == "bf16":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return torch.float32


def _set_instdiff_fuser_state(unet: torch.nn.Module, *, enabled: bool, scale: float) -> int:
    """
    Align InstanceDiffusion fuser behavior with the diffusers pipeline's inference path.

    In the Stage 1 global U-Net baseline, the global branch runs with:
    - fuser enabled
    - scale = instdiff_alpha (default 1.0)

    This runner bypasses the pipeline and calls UNets directly, so we must set these
    flags explicitly to avoid mismatched behavior.
    """
    updated = 0
    for module in unet.modules():
        if type(module) is GatedSelfAttentionDense:
            # `enabled` exists on diffusers' GatedSelfAttentionDense.
            setattr(module, "enabled", bool(enabled))
            module.scale = float(scale)
            updated += 1
    return updated


def _freeze_(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


def _decode_b64_embedding_list(raw_list: List[str]) -> torch.Tensor:
    if not raw_list:
        return torch.empty((0, 768), dtype=torch.float32)

    vecs: List[torch.Tensor] = []
    for s in raw_list:
        arr = np.frombuffer(base64.b64decode(s), dtype=np.float32)
        if arr.size != 768:
            raise ValueError(f"Expected embedding dim=768, got {arr.size}")
        vecs.append(torch.from_numpy(arr))

    return torch.stack(vecs, dim=0).to(torch.float32)


def _load_iconart_layout_json(
    annotations_root: Path,
    stem: str,
) -> Tuple[List[str], np.ndarray, torch.Tensor]:
    ann_path = annotations_root / f"{stem}.json"
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing required annotation json: {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    texts = list(meta.get("objects", []) or [])
    boxes = meta.get("boxes", []) or []
    embeds_b64 = meta.get("text_embedding_before", []) or []
    n = min(len(texts), len(boxes))
    if n <= 0:
        # Some samples legitimately have no objects/boxes; treat as empty layout
        # but keep the file-exists requirement to catch missing annotations.
        return [], np.zeros((0, 4), dtype=np.float32), torch.empty((0, 768), dtype=torch.float32)

    boxes_np = np.asarray(boxes[:n], dtype=np.float32)
    if boxes_np.ndim != 2 or boxes_np.shape[1] != 4:
        raise ValueError(f"Invalid boxes shape in {ann_path}: {boxes_np.shape}")

    embeds = (
        _decode_b64_embedding_list(list(embeds_b64[:n]))
        if len(embeds_b64) >= n
        else torch.empty((0, 768), dtype=torch.float32)
    )

    return [str(x) for x in texts[:n]], boxes_np, embeds


@torch.no_grad()
def _encode_prompts(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompts: List[str],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tok = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tok.input_ids.to(device)
    attn_mask = tok.attention_mask.to(device)
    enc = text_encoder(input_ids=input_ids, attention_mask=attn_mask)
    return enc.last_hidden_state.to(dtype=dtype)


@torch.no_grad()
def _encode_vae(vae: AutoencoderKL, images_bchw: torch.Tensor) -> torch.Tensor:
    latents = vae.encode(images_bchw).latent_dist.sample()
    scale = getattr(vae.config, "scaling_factor", None)
    if scale is None:
        raise AttributeError("VAE is missing config.scaling_factor; do not hardcode latent scaling.")
    return latents * float(scale)


@torch.no_grad()
def _decode_vae(vae: AutoencoderKL, latents_bchw: torch.Tensor) -> torch.Tensor:
    scale = getattr(vae.config, "scaling_factor", None)
    if scale is None:
        raise AttributeError("VAE is missing config.scaling_factor; do not hardcode latent scaling.")
    images = vae.decode(latents_bchw / float(scale)).sample
    return images.clamp(-1, 1)


def _collect_transformer_blocks(unet: torch.nn.Module) -> List[torch.nn.Module]:
    blocks: List[torch.nn.Module] = []
    for module in unet.modules():
        tbs = getattr(module, "transformer_blocks", None)
        if tbs is None:
            continue
        for block in list(tbs):
            blocks.append(block)
    return blocks


def _collect_fusers(unet: torch.nn.Module) -> List[torch.nn.Module]:
    fusers: List[torch.nn.Module] = []
    for block in _collect_transformer_blocks(unet):
        fuser = getattr(block, "fuser", None)
        if fuser is not None:
            fusers.append(fuser)
    return fusers


def _load_global_unet_from_ckpt(
    global_checkpoint: str,
    *,
    instdiff_model: str,
    torch_dtype: torch.dtype,
) -> UNet2DConditionModel:
    unet = UNet2DConditionModel.from_pretrained(
        instdiff_model,
        subfolder="unet",
        torch_dtype=torch_dtype,
    )

    old_conv_in = unet.conv_in
    unet.conv_in = torch.nn.Conv2d(
        9,
        old_conv_in.out_channels,
        kernel_size=old_conv_in.kernel_size,
        stride=old_conv_in.stride,
        padding=old_conv_in.padding,
    ).to(dtype=torch_dtype)
    unet.config["in_channels"] = 9

    unet.position_net = INSTDIFFTextBoundingboxProjectionBBoxOnly(
        positive_len=768,
        out_dim=768,
    ).to(dtype=torch_dtype)

    raw = torch.load(global_checkpoint, map_location="cpu")
    clean: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if k.startswith("inner."):
            clean[k[len("inner.") :]] = v
        else:
            clean[k] = v

    converted = convert_ldm_unet_checkpoint(clean, {"layers_per_block": 2})

    if "conv_in.weight" not in converted or "conv_in.bias" not in converted:
        raise RuntimeError("Converted checkpoint is missing 'conv_in.weight'/'conv_in.bias'.")

    ckpt_w = converted.pop("conv_in.weight")
    ckpt_b = converted.pop("conv_in.bias")
    with torch.no_grad():
        if ckpt_w.shape[1] == 9:
            unet.conv_in.weight.copy_(ckpt_w.to(dtype=torch_dtype))
        else:
            unet.conv_in.weight.zero_()
            unet.conv_in.weight[:, : ckpt_w.shape[1]].copy_(ckpt_w.to(dtype=torch_dtype))
        unet.conv_in.bias.copy_(ckpt_b.to(dtype=torch_dtype))

    allowed = set(unet.state_dict().keys())
    filtered = {k: v for k, v in converted.items() if k in allowed}
    filtered["conv_in.weight"] = unet.conv_in.weight.detach().clone()
    filtered["conv_in.bias"] = unet.conv_in.bias.detach().clone()

    required_model_keys = {k for k in allowed if k.startswith("position_net.") or ".fuser." in k}
    missing_required = sorted(required_model_keys - set(filtered.keys()))
    if missing_required:
        preview = "\n".join(f"  - {k}" for k in missing_required[:20])
        raise RuntimeError(
            "Converted checkpoint is missing required Stage 1 global U-Net keys (position_net / fuser). "
            f"missing_required={len(missing_required)}\n{preview}"
        )

    missing, unexpected = unet.load_state_dict(filtered, strict=False)
    if len(missing) > 0:
        print(f"[GlobalUNet][warn] missing keys: {len(missing)} (unexpected: {len(unexpected)})")
        if len(missing) < 20:
            for k in missing:
                print(f"  - {k}")

    return unet


def _load_main_adapters_from_ckpt(
    main_unet: UNet2DConditionModel,
    patch_unifusion: torch.nn.Module,
    main_checkpoint: str,
) -> None:
    ckpt = torch.load(main_checkpoint, map_location="cpu")
    if "patch_unifusion" not in ckpt or "main_fusers" not in ckpt:
        raise RuntimeError(
            f"Invalid main checkpoint format: expected keys 'patch_unifusion' and 'main_fusers'. file={main_checkpoint}"
        )

    patch_unifusion.load_state_dict(ckpt["patch_unifusion"], strict=True)
    fusers = _collect_fusers(main_unet)
    fusers_sd = ckpt["main_fusers"]
    if len(fusers) != len(fusers_sd):
        raise RuntimeError(
            f"main_fusers count mismatch: model={len(fusers)} ckpt={len(fusers_sd)} (file={main_checkpoint})"
        )

    for i, sd in enumerate(fusers_sd):
        fusers[i].load_state_dict(sd, strict=True)


def _prepare_global_grounding(
    *,
    layout: GlobalLayoutCondition,
    canvas_w: int,
    canvas_h: int,
    cx: int,
    cy: int,
    patch_size: int,
    lb_scale: float,
    lb_pad: Tuple[int, int],
    default_positive: torch.Tensor,
    max_instances: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embed_dim = int(default_positive.shape[-1])
    M = int(max_instances)

    def _make_empty() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes = torch.zeros((1, M, 4), device=device, dtype=dtype)
        masks = torch.zeros((1, M), device=device, dtype=dtype)
        pos = default_positive.view(1, 1, -1).to(device=device, dtype=dtype).expand(1, M, -1).clone()
        return boxes, masks, pos

    # Allow empty layout when the annotation file exists but contains no objects/boxes.
    if layout.boxes_norm.size == 0:
        return _make_empty()

    boxes = np.asarray(layout.boxes_norm, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        return _make_empty()

    n_total = int(boxes.shape[0])
    if n_total <= 0:
        return _make_empty()

    x0 = boxes[:, 0] * float(layout.orig_w)
    y0 = boxes[:, 1] * float(layout.orig_h)
    x1 = boxes[:, 2] * float(layout.orig_w)
    y1 = boxes[:, 3] * float(layout.orig_h)

    canvas_offset_x = float(cx) - float(layout.crop_left)
    canvas_offset_y = float(cy) - float(layout.crop_top)

    x0c = x0 + canvas_offset_x
    y0c = y0 + canvas_offset_y
    x1c = x1 + canvas_offset_x
    y1c = y1 + canvas_offset_y

    x0c = np.clip(x0c, 0.0, float(canvas_w))
    y0c = np.clip(y0c, 0.0, float(canvas_h))
    x1c = np.clip(x1c, 0.0, float(canvas_w))
    y1c = np.clip(y1c, 0.0, float(canvas_h))

    good = (x1c > x0c + 1e-6) & (y1c > y0c + 1e-6)
    keep_idx = np.nonzero(good)[0].tolist()
    if not keep_idx:
        return _make_empty()

    keep_idx = keep_idx[:M]

    pad_left, pad_top = lb_pad
    x0lb = (x0c[keep_idx] * lb_scale + float(pad_left)) / float(patch_size)
    y0lb = (y0c[keep_idx] * lb_scale + float(pad_top)) / float(patch_size)
    x1lb = (x1c[keep_idx] * lb_scale + float(pad_left)) / float(patch_size)
    y1lb = (y1c[keep_idx] * lb_scale + float(pad_top)) / float(patch_size)

    x0lb = np.clip(x0lb, 0.0, 1.0)
    y0lb = np.clip(y0lb, 0.0, 1.0)
    x1lb = np.clip(x1lb, 0.0, 1.0)
    y1lb = np.clip(y1lb, 0.0, 1.0)

    good_lb = (x1lb > x0lb + 1e-6) & (y1lb > y0lb + 1e-6)
    if not np.any(good_lb):
        return _make_empty()

    final_idx = [keep_idx[i] for i, g in enumerate(good_lb.tolist()) if g]
    boxes_lb = np.stack([x0lb[good_lb], y0lb[good_lb], x1lb[good_lb], y1lb[good_lb]], axis=-1)
    n_valid = len(final_idx)

    boxes_t = torch.zeros((1, M, 4), device=device, dtype=dtype)
    masks_t = torch.zeros((1, M), device=device, dtype=dtype)
    pos_t = torch.zeros((1, M, embed_dim), device=device, dtype=dtype)

    boxes_t[0, :n_valid] = torch.from_numpy(boxes_lb).to(device=device, dtype=dtype)
    masks_t[0, :n_valid] = 1.0

    if layout.positive_embeddings.numel() and layout.positive_embeddings.shape[0] >= max(final_idx) + 1:
        pos_t[0, :n_valid] = layout.positive_embeddings[final_idx].to(device=device, dtype=dtype)

    return boxes_t, masks_t, pos_t


def _register_attention_control(unet: torch.nn.Module, attention_store: AttentionStore) -> Dict[str, object]:
    """
    AlignNoise attention processor registration (reused pattern).
    """
    original = dict(unet.attn_processors)
    attn_procs: Dict[str, object] = {}
    attn_layer_count = 0

    for name in original.keys():
        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        else:
            place_in_unet = "mid"

        attn_layer_count += 1
        attn_procs[name] = AttendExciteAttnProcessor(attnstore=attention_store, place_in_unet=place_in_unet)

    unet.set_attn_processor(attn_procs)
    attention_store.num_att_layers = attn_layer_count
    return original


@contextmanager
def _attention_control_ctx(unet: torch.nn.Module, attention_store: AttentionStore):
    original = _register_attention_control(unet, attention_store)
    try:
        yield
    finally:
        unet.set_attn_processor(original)


def _compute_aam_loss_alignnoise(
    attention_store: AttentionStore,
    *,
    mask_latent: torch.Tensor,  # [1,1,h,w], 1==out
    valid_latent: torch.Tensor,  # [1,1,h,w], 1==valid
    attention_res: int,
    smooth_attentions: bool,
    tau_out: float,
    lambda_tau: float,
    empty_attn_monitor: Optional[AAMEmptyAttnMonitor] = None,
    sample_id: str = "",
    branch: str = "",
    outer_iter: int = -1,
    timestep: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    AlignNoise-style AAM loss:
    AS_out/(AS_src+eps) + lambda_tau * relu(tau_out - AS_out)
    """
    try:
        self_attention_maps = attention_store.aggregate_attention(from_where=("up", "down", "mid"), is_cross=False)
    except (KeyError, RuntimeError) as exc:
        if empty_attn_monitor is not None:
            empty_attn_monitor.record(
                sample_id=sample_id or "unknown",
                branch=branch or "unknown",
                outer_iter=int(outer_iter),
                timestep=int(timestep),
                error_type=type(exc).__name__,
            )
        z = torch.tensor(0.0, device=mask_latent.device, dtype=mask_latent.dtype, requires_grad=True)
        return z, z, z
    h, w = int(self_attention_maps.shape[0]), int(self_attention_maps.shape[1])

    mask = F.interpolate(mask_latent, size=(h, w), mode="nearest").squeeze(0).squeeze(0).bool()
    valid = F.interpolate(valid_latent, size=(h, w), mode="nearest").squeeze(0).squeeze(0) > 0.5
    query_mask = mask & valid

    per_token_maps: List[torch.Tensor] = []
    for coord_x in range(h):
        for coord_y in range(w):
            if not bool(query_mask[coord_x, coord_y]):
                continue
            attn_map = self_attention_maps[coord_x, coord_y].view(attention_res, attention_res).contiguous()
            if smooth_attentions:
                attn_map = fn_smoothing_func(attn_map)
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)
            per_token_maps.append(attn_map)

    if not per_token_maps:
        z = torch.tensor(0.0, device=self_attention_maps.device, requires_grad=True)
        return z, z, z

    combined = torch.stack(per_token_maps, dim=0).sum(dim=0)
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-6)

    mask_resized = (
        F.interpolate(mask.float().unsqueeze(0).unsqueeze(0), size=(attention_res, attention_res), mode="nearest")
        .squeeze(0)
        .squeeze(0)
    )
    valid_resized = (
        F.interpolate(valid.float().unsqueeze(0).unsqueeze(0), size=(attention_res, attention_res), mode="nearest")
        .squeeze(0)
        .squeeze(0)
        > 0.5
    )

    out_sel = (mask_resized > 0.5) & valid_resized
    src_sel = (mask_resized <= 0.5) & valid_resized
    if int(out_sel.sum().item()) == 0 or int(src_sel.sum().item()) == 0:
        z = torch.tensor(0.0, device=self_attention_maps.device, requires_grad=True)
        return z, z, z

    as_out = combined[out_sel].mean()
    as_src = combined[src_sel].mean()
    aam_loss = as_out / (as_src + 1e-6) + float(lambda_tau) * torch.relu(
        torch.tensor(float(tau_out), device=as_out.device) - as_out
    )
    return aam_loss, as_src, as_out


def _optimize_global_latents_with_aam(
    *,
    global_unet: UNet2DConditionModel,
    global_writer: DiffusersGlobalUNetFuserWriter,
    scheduler: DDIMScheduler,
    latents_init: torch.Tensor,  # [1,4,h,w] scaled by init_noise_sigma
    mask_latent: torch.Tensor,  # [1,1,h,w]
    masked_image_latents: torch.Tensor,  # [1,4,h,w]
    valid_latent: torch.Tensor,  # [1,1,h,w]
    prompt_embeds: torch.Tensor,  # [1,tok,dim]
    ref_boxes: torch.Tensor,  # [1,N,4]
    ref_masks: torch.Tensor,  # [1,N]
    ref_positive_embeddings: torch.Tensor,  # [1,N,dim]
    steps: int,
    aam_iters: int,
    aam_lr: float,
    aam_denoising_steps: int,
    aam_stop_loss: float,
    aam_res: int,
    aam_smooth: bool,
    aam_tau_out: float,
    aam_lambda_tau: float,
    sample_id: str,
    empty_attn_monitor: Optional[AAMEmptyAttnMonitor] = None,
) -> torch.Tensor:
    """
    AlignNoise-style initial-noise optimization over the Stage 1 global branch.
    """
    if int(aam_iters) <= 0:
        return latents_init

    device = latents_init.device
    dtype = latents_init.dtype

    attention_store = AttentionStore(attn_res=(int(aam_res), int(aam_res)))
    log_var = torch.zeros_like(latents_init, requires_grad=True)
    mu = torch.zeros_like(latents_init, requires_grad=True)
    opt = torch.optim.Adam([log_var, mu], lr=float(aam_lr), eps=1e-3)

    scheduler.set_timesteps(int(steps), device=device)
    timesteps = scheduler.timesteps
    k_steps = max(1, int(aam_denoising_steps))
    t_steps = [timesteps[j] for j in range(min(k_steps, len(timesteps)))]

    cross_attention_kwargs = {
        "instdiff": {
            "boxes": ref_boxes,
            "positive_embeddings": ref_positive_embeddings,
            "masks": ref_masks,
        }
    }

    with _attention_control_ctx(global_unet, attention_store), torch.enable_grad():
        for aam_iter_idx in range(int(aam_iters)):
            opt.zero_grad(set_to_none=True)

            latents_curr = latents_init * torch.exp(0.5 * log_var) + mu
            loss_terms: List[torch.Tensor] = []

            for t in t_steps:
                attention_store.reset()
                global_writer.clear()

                latent_model_input = scheduler.scale_model_input(latents_curr, t)
                model_input = torch.cat([latent_model_input, mask_latent, masked_image_latents], dim=1)
                _ = global_unet(
                    model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                loss, _as_src, _as_out = _compute_aam_loss_alignnoise(
                    attention_store,
                    mask_latent=mask_latent,
                    valid_latent=valid_latent,
                    attention_res=int(aam_res),
                    smooth_attentions=bool(aam_smooth),
                    tau_out=float(aam_tau_out),
                    lambda_tau=float(aam_lambda_tau),
                    empty_attn_monitor=empty_attn_monitor,
                    sample_id=str(sample_id),
                    branch="cond",
                    outer_iter=int(aam_iter_idx),
                    timestep=int(t.item()),
                )
                loss_terms.append(loss)
                global_writer.clear()

            if not loss_terms:
                break

            loss_all = torch.stack(loss_terms, dim=0).mean()
            if float(loss_all.detach().item()) < float(aam_stop_loss):
                break

            global_unet.zero_grad(set_to_none=True)
            loss_all.backward()
            opt.step()

    global_writer.clear()
    return (latents_init * torch.exp(0.5 * log_var.detach()) + mu.detach()).to(device=device, dtype=dtype)


def _optimize_global_latents_with_negative_aam(
    *,
    global_unet: UNet2DConditionModel,
    global_writer: DiffusersGlobalUNetFuserWriter,
    scheduler: DDIMScheduler,
    latents_init: torch.Tensor,  # [1,4,h,w] scaled by init_noise_sigma
    mask_latent: torch.Tensor,  # [1,1,h,w]
    masked_image_latents: torch.Tensor,  # [1,4,h,w]
    valid_latent: torch.Tensor,  # [1,1,h,w]
    prompt_embeds: torch.Tensor,  # [1,tok,dim] (uncond prompt)
    ref_boxes: torch.Tensor,  # [1,N,4]
    ref_masks: torch.Tensor,  # [1,N]
    ref_positive_embeddings: torch.Tensor,  # [1,N,dim]
    steps: int,
    neg_aam_iters: int,
    neg_aam_lr: float,
    neg_aam_denoising_steps: int,
    neg_aam_stop_loss: float,
    aam_res: int,
    aam_smooth: bool,
    neg_aam_tau_src: float,
    neg_aam_lambda_tau: float,
    sample_id: str,
    empty_attn_monitor: Optional[AAMEmptyAttnMonitor] = None,
) -> torch.Tensor:
    """Negative-AAM optimization for CFG uncond branch on global stage."""
    if int(neg_aam_iters) <= 0:
        return latents_init

    device = latents_init.device
    dtype = latents_init.dtype

    attention_store = AttentionStore(attn_res=(int(aam_res), int(aam_res)))
    log_var = torch.zeros_like(latents_init, requires_grad=True)
    mu = torch.zeros_like(latents_init, requires_grad=True)
    opt = torch.optim.Adam([log_var, mu], lr=float(neg_aam_lr), eps=1e-3)

    scheduler.set_timesteps(int(steps), device=device)
    timesteps = scheduler.timesteps
    k_steps = max(1, int(neg_aam_denoising_steps))
    t_steps = [timesteps[j] for j in range(min(k_steps, len(timesteps)))]

    cross_attention_kwargs = {
        "instdiff": {
            "boxes": ref_boxes,
            "positive_embeddings": ref_positive_embeddings,
            "masks": ref_masks,
        }
    }

    with _attention_control_ctx(global_unet, attention_store), torch.enable_grad():
        for neg_aam_iter_idx in range(int(neg_aam_iters)):
            opt.zero_grad(set_to_none=True)

            latents_curr = latents_init * torch.exp(0.5 * log_var) + mu
            loss_terms: List[torch.Tensor] = []

            for t in t_steps:
                attention_store.reset()
                global_writer.clear()

                latent_model_input = scheduler.scale_model_input(latents_curr, t)
                model_input = torch.cat([latent_model_input, mask_latent, masked_image_latents], dim=1)
                _ = global_unet(
                    model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                _aam_loss_unused, as_src, as_out = _compute_aam_loss_alignnoise(
                    attention_store,
                    mask_latent=mask_latent,
                    valid_latent=valid_latent,
                    attention_res=int(aam_res),
                    smooth_attentions=bool(aam_smooth),
                    tau_out=0.0,
                    lambda_tau=0.0,
                    empty_attn_monitor=empty_attn_monitor,
                    sample_id=str(sample_id),
                    branch="neg",
                    outer_iter=int(neg_aam_iter_idx),
                    timestep=int(t.item()),
                )
                neg_loss = as_src / (as_out + 1e-6) + float(neg_aam_lambda_tau) * torch.relu(
                    torch.tensor(float(neg_aam_tau_src), device=as_src.device, dtype=as_src.dtype) - as_src
                )
                loss_terms.append(neg_loss)
                global_writer.clear()

            if not loss_terms:
                break

            loss_all = torch.stack(loss_terms, dim=0).mean()
            if float(loss_all.detach().item()) < float(neg_aam_stop_loss):
                break

            global_unet.zero_grad(set_to_none=True)
            loss_all.backward()
            opt.step()

    global_writer.clear()
    return (latents_init * torch.exp(0.5 * log_var.detach()) + mu.detach()).to(device=device, dtype=dtype)


def compute_window_bbox_letterboxed(
    win: PatchWindow,
    patch_size: int,
    scale: float,
    pad: Tuple[int, int],
) -> torch.Tensor:
    pad_left, pad_top = pad
    x0 = (win.x * scale + pad_left) / float(patch_size)
    y0 = (win.y * scale + pad_top) / float(patch_size)
    x1 = ((win.x + patch_size) * scale + pad_left) / float(patch_size)
    y1 = ((win.y + patch_size) * scale + pad_top) / float(patch_size)

    return torch.tensor(
        [
            max(0.0, min(1.0, x0)),
            max(0.0, min(1.0, y0)),
            max(0.0, min(1.0, x1)),
            max(0.0, min(1.0, y1)),
        ],
        dtype=torch.float32,
    )


def _build_parallel_local_inputs(
    init_canvas: np.ndarray,
    composer: PriorityComposer,
    windows: Dict[str, PatchWindow],
    order: List[str],
    patch_size: int,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    keys: List[str] = []
    rgbs: List[np.ndarray] = []
    masks: List[np.ndarray] = []

    for key in order:
        win = windows[key]
        local_rgb, local_mask = build_local_input(init_canvas, composer, win.x, win.y, patch_size)
        if int(np.count_nonzero(local_mask)) == 0:
            continue
        keys.append(key)
        rgbs.append(local_rgb)
        masks.append(local_mask)

    if not rgbs:
        return [], np.zeros((0, patch_size, patch_size, 3), dtype=np.uint8), np.zeros((0, patch_size, patch_size), dtype=np.uint8)

    return keys, np.stack(rgbs, axis=0), np.stack(masks, axis=0)


def _repeat_bank_tokens(
    bank_step: Dict[int, torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[int, torch.Tensor]:
    repeated: Dict[int, torch.Tensor] = {}
    for idx, tokens in bank_step.items():
        t = tokens
        if t.shape[0] == 1 and batch_size > 1:
            t = t.expand(batch_size, -1, -1).contiguous()
        elif t.shape[0] != batch_size:
            raise RuntimeError(
                f"Cannot repeat bank tokens for block {idx}: bank.B={t.shape[0]} target.B={batch_size}"
            )
        repeated[idx] = t.to(device=device, dtype=dtype, non_blocking=True)
    return repeated


@torch.no_grad()
def _sample_global_and_cache_banks(
    *,
    global_unet: UNet2DConditionModel,
    global_writer: DiffusersGlobalUNetFuserWriter,
    scheduler: DDIMScheduler,
    global_masked_latents: torch.Tensor,
    global_mask_latent: torch.Tensor,
    global_prompt_embeds: torch.Tensor,
    global_neg_embeds: torch.Tensor,
    ref_boxes: torch.Tensor,
    ref_masks: torch.Tensor,
    ref_positive_embeddings: torch.Tensor,
    steps: int,
    global_guidance_scale: float,
    eta: float,
    generator: torch.Generator,
    latents_init: Optional[torch.Tensor],
    latents_init_uncond: Optional[torch.Tensor],
    cfg_base: str,
    cache_bank_features: bool,
    cache_uncond_bank_features: bool,
) -> Tuple[torch.Tensor, torch.Tensor, RefBankCache]:
    device = global_masked_latents.device
    dtype = global_masked_latents.dtype

    scheduler.set_timesteps(int(steps), device=device)
    timesteps = scheduler.timesteps

    if latents_init is None:
        init_sigma = float(getattr(scheduler, "init_noise_sigma", 1.0))
        global_latents = randn_tensor(
            global_masked_latents.shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        global_latents = global_latents * init_sigma
    else:
        if tuple(latents_init.shape) != tuple(global_masked_latents.shape):
            raise RuntimeError(
                "latents_init shape mismatch: "
                f"got={tuple(latents_init.shape)} expected={tuple(global_masked_latents.shape)}"
            )
        global_latents = latents_init.to(device=device, dtype=dtype).clone()
    if latents_init_uncond is not None:
        if tuple(latents_init_uncond.shape) != tuple(global_masked_latents.shape):
            raise RuntimeError(
                "latents_init_uncond shape mismatch: "
                f"got={tuple(latents_init_uncond.shape)} expected={tuple(global_masked_latents.shape)}"
            )
        global_latents_uncond = latents_init_uncond.to(device=device, dtype=dtype).clone()
    else:
        global_latents_uncond = global_latents.clone()

    extra_step_kwargs = {"eta": float(eta), "generator": generator}
    cond_bank_cache_per_step: List[Dict[int, torch.Tensor]] = []
    uncond_bank_cache_per_step: Optional[List[Dict[int, torch.Tensor]]] = (
        [] if bool(cache_uncond_bank_features) else None
    )
    use_global_cfg = float(global_guidance_scale) != 1.0
    cfg_base_mode = str(cfg_base).strip().lower()
    if cfg_base_mode not in {"uncond", "cond"}:
        raise ValueError(f"Unknown cfg_base: {cfg_base}. expected one of ['uncond', 'cond']")

    bsz = int(global_latents.shape[0])
    assert bsz == 1, f"Expected global batch size 1, got {bsz}"

    for step_idx, t in enumerate(timesteps):
        cond_step_bank: Dict[int, torch.Tensor] = {}
        uncond_step_bank: Dict[int, torch.Tensor] = {}

        if use_global_cfg and step_idx == 0 and latents_init_uncond is not None:
            latent_model_input_uncond = scheduler.scale_model_input(global_latents_uncond, t)
            global_model_input_uncond = torch.cat(
                [latent_model_input_uncond, global_mask_latent, global_masked_latents],
                dim=1,
            )
            cross_attention_kwargs_uncond = {
                "instdiff": {
                    "boxes": ref_boxes,
                    "positive_embeddings": ref_positive_embeddings,
                    "masks": torch.zeros_like(ref_masks),
                }
            }

            global_writer.clear()
            noise_u = global_unet(
                global_model_input_uncond,
                t,
                encoder_hidden_states=global_neg_embeds,
                cross_attention_kwargs=cross_attention_kwargs_uncond,
            ).sample
            if cache_bank_features and uncond_bank_cache_per_step is not None:
                for i in range(global_writer.num_blocks):
                    bank_list = global_writer.bank.get(i, [])
                    if not bank_list:
                        raise RuntimeError(f"Global U-Net uncond bank missing for block {i} at timestep {int(t)}")
                    tokens = torch.cat(bank_list, dim=1) if len(bank_list) > 1 else bank_list[0]
                    uncond_step_bank[i] = tokens.detach().to(device=device, dtype=torch.float16).contiguous()
            global_writer.clear()

            latent_model_input_cond = scheduler.scale_model_input(global_latents, t)
            global_model_input_cond = torch.cat(
                [latent_model_input_cond, global_mask_latent, global_masked_latents],
                dim=1,
            )
            cross_attention_kwargs_cond = {
                "instdiff": {
                    "boxes": ref_boxes,
                    "positive_embeddings": ref_positive_embeddings,
                    "masks": ref_masks,
                }
            }

            global_writer.clear()
            noise_c = global_unet(
                global_model_input_cond,
                t,
                encoder_hidden_states=global_prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs_cond,
            ).sample
            if cache_bank_features:
                for i in range(global_writer.num_blocks):
                    bank_list = global_writer.bank.get(i, [])
                    if not bank_list:
                        raise RuntimeError(f"Global U-Net cond bank missing for block {i} at timestep {int(t)}")
                    tokens = torch.cat(bank_list, dim=1) if len(bank_list) > 1 else bank_list[0]
                    cond_step_bank[i] = tokens.detach().to(device=device, dtype=torch.float16).contiguous()
            global_writer.clear()

            if cfg_base_mode == "cond":
                global_noise_pred = noise_c + float(global_guidance_scale) * (noise_c - noise_u)
            else:
                global_noise_pred = noise_u + float(global_guidance_scale) * (noise_c - noise_u)

            # Keep the step-0 update anchored on the conditional optimized latent.
            global_latents = scheduler.step(global_noise_pred, t, global_latents, **extra_step_kwargs).prev_sample
            global_latents_uncond = global_latents
            cond_bank_cache_per_step.append(cond_step_bank)
            if uncond_bank_cache_per_step is not None:
                uncond_bank_cache_per_step.append(uncond_step_bank)
            continue

        if use_global_cfg:
            latent_model_input = torch.cat([global_latents] * 2, dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            mask_latent_cfg = torch.cat([global_mask_latent] * 2, dim=0)
            masked_image_latents_cfg = torch.cat([global_masked_latents] * 2, dim=0)
            global_model_input = torch.cat([latent_model_input, mask_latent_cfg, masked_image_latents_cfg], dim=1)

            encoder_hidden_states_cfg = torch.cat([global_neg_embeds, global_prompt_embeds], dim=0)

            boxes_cfg = torch.cat([ref_boxes, ref_boxes], dim=0)
            pos_cfg = torch.cat([ref_positive_embeddings, ref_positive_embeddings], dim=0)
            masks_cfg = torch.cat([ref_masks, ref_masks], dim=0)
            masks_cfg[:bsz] = 0

            cross_attention_kwargs_cfg = {
                "instdiff": {
                    "boxes": boxes_cfg,
                    "positive_embeddings": pos_cfg,
                    "masks": masks_cfg,
                }
            }

            global_writer.clear()
            global_noise_pred_2b = global_unet(
                global_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states_cfg,
                cross_attention_kwargs=cross_attention_kwargs_cfg,
            ).sample

            noise_u, noise_c = global_noise_pred_2b.chunk(2, dim=0)
            global_noise_pred = noise_u + float(global_guidance_scale) * (noise_c - noise_u)

            if cache_bank_features:
                for i in range(global_writer.num_blocks):
                    bank_list = global_writer.bank.get(i, [])
                    if not bank_list:
                        raise RuntimeError(f"Global U-Net bank missing for block {i} at timestep {int(t)}")
                    tokens = torch.cat(bank_list, dim=1) if len(bank_list) > 1 else bank_list[0]
                    cond_step_bank[i] = tokens[bsz:].detach().to(device=device, dtype=torch.float16).contiguous()
                    if uncond_bank_cache_per_step is not None:
                        uncond_step_bank[i] = tokens[:bsz].detach().to(device=device, dtype=torch.float16).contiguous()

            global_writer.clear()
        else:
            latent_model_input = scheduler.scale_model_input(global_latents, t)
            global_model_input = torch.cat([latent_model_input, global_mask_latent, global_masked_latents], dim=1)

            cross_attention_kwargs = {
                "instdiff": {
                    "boxes": ref_boxes,
                    "positive_embeddings": ref_positive_embeddings,
                    "masks": ref_masks,
                }
            }

            global_writer.clear()
            global_noise_pred = global_unet(
                global_model_input,
                t,
                encoder_hidden_states=global_prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample

            if cache_bank_features:
                for i in range(global_writer.num_blocks):
                    bank_list = global_writer.bank.get(i, [])
                    if not bank_list:
                        raise RuntimeError(f"Global U-Net bank missing for block {i} at timestep {int(t)}")
                    tokens = torch.cat(bank_list, dim=1) if len(bank_list) > 1 else bank_list[0]
                    cond_step_bank[i] = tokens.detach().to(device=device, dtype=torch.float16).contiguous()

            global_writer.clear()

        global_latents = scheduler.step(global_noise_pred, t, global_latents, **extra_step_kwargs).prev_sample
        cond_bank_cache_per_step.append(cond_step_bank)
        if uncond_bank_cache_per_step is not None:
            uncond_bank_cache_per_step.append(uncond_step_bank)

    return global_latents, timesteps, RefBankCache(
        cond_per_step=cond_bank_cache_per_step,
        uncond_per_step=uncond_bank_cache_per_step,
    )


def _extract_bank_cache_from_writer(
    global_writer: DiffusersGlobalUNetFuserWriter,
    num_steps: int,
    bsz: int,
    *,
    cache_bank_features: bool,
    cache_uncond: bool,
    do_cfg: bool,
    device: torch.device,
) -> RefBankCache:
    """Extract per-step bank cache from accumulated writer.bank after a full pipeline run.

    During pipeline execution with CFG, each denoising step calls the UNet once with
    batch=[uncond, cond] (size 2*bsz). The writer captures the fuser output for each
    step, so writer.bank[block_idx] is a list of num_steps tensors, each [2*bsz, N, C].
    """
    cond_per_step: List[Dict[int, torch.Tensor]] = []
    uncond_per_step: Optional[List[Dict[int, torch.Tensor]]] = [] if cache_uncond else None

    if not cache_bank_features:
        return RefBankCache(
            cond_per_step=[{} for _ in range(num_steps)],
            uncond_per_step=[{} for _ in range(num_steps)] if cache_uncond else None,
        )

    for step_idx in range(num_steps):
        cond_step: Dict[int, torch.Tensor] = {}
        uncond_step: Dict[int, torch.Tensor] = {}
        for block_idx in range(global_writer.num_blocks):
            bank_list = global_writer.bank.get(block_idx, [])
            if step_idx >= len(bank_list):
                raise RuntimeError(
                    f"Writer bank has {len(bank_list)} entries for block {block_idx}, "
                    f"but expected at least {num_steps} (step_idx={step_idx})."
                )
            tokens = bank_list[step_idx]  # [2*bsz, N, C] if CFG, [bsz, N, C] otherwise
            if do_cfg:
                cond_step[block_idx] = tokens[bsz:].detach().to(device=device, dtype=torch.float16).contiguous()
                if uncond_per_step is not None:
                    uncond_step[block_idx] = tokens[:bsz].detach().to(device=device, dtype=torch.float16).contiguous()
            else:
                cond_step[block_idx] = tokens.detach().to(device=device, dtype=torch.float16).contiguous()
        cond_per_step.append(cond_step)
        if uncond_per_step is not None:
            uncond_per_step.append(uncond_step)

    return RefBankCache(cond_per_step=cond_per_step, uncond_per_step=uncond_per_step)


def _build_local_inputs_for_keys(
    canvas: np.ndarray,
    composer: PriorityComposer,
    windows: Dict[str, PatchWindow],
    keys: List[str],
    patch_size: int,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    selected_keys: List[str] = []
    rgbs: List[np.ndarray] = []
    masks: List[np.ndarray] = []

    for key in keys:
        win = windows[key]
        local_rgb, local_mask = build_local_input(canvas, composer, win.x, win.y, patch_size)
        if int(np.count_nonzero(local_mask)) == 0:
            continue
        selected_keys.append(key)
        rgbs.append(local_rgb)
        masks.append(local_mask)

    if not rgbs:
        return (
            [],
            np.zeros((0, patch_size, patch_size, 3), dtype=np.uint8),
            np.zeros((0, patch_size, patch_size), dtype=np.uint8),
        )

    return selected_keys, np.stack(rgbs, axis=0), np.stack(masks, axis=0)


@torch.no_grad()
def _prepare_local_condition_latents(
    *,
    vae: AutoencoderKL,
    local_rgbs_u8: np.ndarray,
    local_masks_u8: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if int(local_rgbs_u8.shape[0]) == 0:
        patch_size = int(local_rgbs_u8.shape[1]) if local_rgbs_u8.ndim == 4 else 512
        lat = int(patch_size) // 8
        return (
            torch.zeros((0, 4, lat, lat), device=device, dtype=dtype),
            torch.zeros((0, 1, lat, lat), device=device, dtype=dtype),
        )

    local_rgb = torch.from_numpy(local_rgbs_u8).to(device=device, dtype=torch.float32)
    local_rgb = local_rgb.permute(0, 3, 1, 2) / 127.5 - 1.0
    local_rgb = local_rgb.to(dtype=dtype)

    local_mask = torch.from_numpy(local_masks_u8).to(device=device, dtype=torch.float32)
    local_mask = (local_mask / 255.0).unsqueeze(1).to(dtype=dtype)

    local_masked_rgb = local_rgb * (1.0 - local_mask)
    local_masked_latents = _encode_vae(vae, local_masked_rgb)
    local_mask_latent = F.interpolate(local_mask, size=local_masked_latents.shape[-2:], mode="nearest")
    return local_masked_latents, local_mask_latent


def _build_local_window_bbox_batch(
    *,
    local_keys: List[str],
    windows: Dict[str, PatchWindow],
    patch_size: int,
    lb_scale: float,
    lb_pad: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    bboxes: List[torch.Tensor] = []
    for key in local_keys:
        bboxes.append(
            compute_window_bbox_letterboxed(
                win=windows[key],
                patch_size=patch_size,
                scale=lb_scale,
                pad=lb_pad,
            )
        )
    if not bboxes:
        return torch.zeros((0, 1, 4), device=device, dtype=dtype)
    return torch.stack(bboxes, dim=0).to(device=device, dtype=dtype).unsqueeze(1)


@torch.no_grad()
def _run_global_stage(
    *,
    pipe: StableDiffusionINSTDIFFInpaintPipelineBBoxOnly,
    global_writer: DiffusersGlobalUNetFuserWriter,
    sample_id: str,
    canvas_u8: np.ndarray,
    composer: PriorityComposer,
    canvas_w: int,
    canvas_h: int,
    cx: int,
    cy: int,
    patch_size: int,
    prompt: str,
    global_guidance_scale: float,
    eta: float,
    steps: int,
    global_seed: int,
    layout: GlobalLayoutCondition,
    layout_max_instances: int,
    use_bank_features: bool,
    capture_global_uncond_bank: bool,
    aam_iters: int,
    aam_lr: float,
    aam_denoising_steps: int,
    aam_stop_loss: float,
    aam_res: int,
    aam_smooth: bool,
    aam_tau_out: float,
    aam_lambda_tau: float,
    neg_aam_uncond: bool,
    neg_aam_iters: int,
    neg_aam_lr: float,
    neg_aam_denoising_steps: int,
    neg_aam_stop_loss: float,
    neg_aam_tau_src: float,
    neg_aam_lambda_tau: float,
    neg_aam_cfg_base: str,
    global_mask_dilate_px: int,
    disable_global_layout_adapter: bool = False,
    empty_attn_monitor: Optional[AAMEmptyAttnMonitor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float, Tuple[int, int], torch.Tensor, RefBankCache, np.ndarray, np.ndarray]:
    """Run global stage using the 04 pipeline for denoising, with writer bank capture."""
    device = pipe._execution_device
    weight_dtype = pipe.text_encoder.dtype
    global_unet = pipe.unet
    vae = pipe.vae

    # ── 1. Canvas → letterbox ──
    canvas_t = torch.from_numpy(canvas_u8).to(device=device, dtype=torch.float32)
    canvas_t = canvas_t.permute(2, 0, 1) / 127.5 - 1.0

    global_mask_canvas = torch.from_numpy((composer.filled_by == -1).astype(np.float32)).to(device=device)
    global_rgb_lb, lb_scale, lb_pad = torch_letterbox(canvas_t, target_size=patch_size, mode="bilinear")
    # Use bilinear for mask too (matches hint_pad in 04 dataloader) to align
    # the mask boundary with the image content boundary after downsampling.
    global_mask_lb, mask_scale, mask_pad = torch_letterbox(global_mask_canvas, target_size=patch_size, mode="bilinear")
    if abs(float(lb_scale) - float(mask_scale)) > 1e-7 or lb_pad != mask_pad:
        raise RuntimeError("letterbox mismatch between RGB and mask")

    if global_mask_lb.ndim == 2:
        global_mask_lb = global_mask_lb.unsqueeze(0)
    global_mask_lb = (global_mask_lb > 0.5).to(dtype=weight_dtype, device=device)
    dilate_px = max(0, int(global_mask_dilate_px))
    if dilate_px > 0:
        # Expand outpaint mask slightly to absorb boundary gradients after bilinear letterbox.
        global_mask_lb = F.max_pool2d(
            global_mask_lb.unsqueeze(0),
            kernel_size=2 * dilate_px + 1,
            stride=1,
            padding=dilate_px,
        ).squeeze(0)
    global_rgb_lb = global_rgb_lb.to(dtype=weight_dtype, device=device)

    # ── 2. Convert to PIL via 04-identical path ──
    # Pre-mask the canvas so boundary gradient pixels are zeroed (04 passes
    # pre-masked global_image to the pipeline; the pipeline then applies
    # masked_image = image * (1 - mask) which is idempotent for already-masked
    # content, avoiding dark-line artifacts from partially blended pixels).
    masked_rgb_lb = global_rgb_lb * (1.0 - global_mask_lb)
    # Match tensor_to_pil_rgb quantization used in 04.
    _rgb_np = ((masked_rgb_lb.permute(1, 2, 0).cpu().float().numpy() + 1.0) * 0.5).clip(0.0, 1.0)
    image_pil = Image.fromarray(((_rgb_np * 255.0) + 0.5).astype(np.uint8), mode="RGB")
    _mask_np = global_mask_lb.squeeze(0).cpu().float().numpy().clip(0.0, 1.0)
    mask_pil = Image.fromarray(((_mask_np * 255.0) + 0.5).astype(np.uint8), mode="L")

    # ── 3. Prepare mask/image latents through pipe preprocessors ──
    # Using the same image_processor/mask_processor path that the pipeline's
    # __call__ uses internally ensures AAM and denoising see identical
    # mask_latent / masked_image_latents (no tensor↔PIL round-trip mismatch).
    image_t = pipe.image_processor.preprocess(image_pil).to(device=device, dtype=weight_dtype)
    mask_t = pipe.mask_processor.preprocess(mask_pil).to(device=device, dtype=weight_dtype)
    masked_image_t = image_t * (1.0 - mask_t)

    gen_vae = torch.Generator(device=device).manual_seed(int(global_seed))
    mask_latent, masked_image_latents = pipe.prepare_mask_latents(
        mask=mask_t,
        masked_image=masked_image_t,
        batch_size=1,
        height=patch_size,
        width=patch_size,
        dtype=weight_dtype,
        device=device,
        generator=gen_vae,
        do_classifier_free_guidance=False,
    )

    # Valid region on letterboxed canvas (exclude padding from AAM).
    valid_canvas = torch.ones((canvas_h, canvas_w), device=device, dtype=weight_dtype)
    valid_lb, valid_scale, valid_pad = torch_letterbox(valid_canvas, target_size=patch_size, mode="nearest")
    if abs(float(lb_scale) - float(valid_scale)) > 1e-7 or lb_pad != valid_pad:
        raise RuntimeError("letterbox mismatch between RGB and valid mask")
    if valid_lb.ndim == 2:
        valid_lb = valid_lb.unsqueeze(0)
    valid_latent = F.interpolate(
        valid_lb.unsqueeze(0).to(device=device, dtype=weight_dtype),
        size=mask_latent.shape[-2:],
        mode="nearest",
    )

    # ── 4. Prepare grounding (reused for both AAM and pipeline) ──
    default_pos = global_unet.position_net.null_positive_feature.detach()
    ref_boxes, ref_masks_t, ref_pos = _prepare_global_grounding(
        layout=layout,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        cx=cx,
        cy=cy,
        patch_size=patch_size,
        lb_scale=float(lb_scale),
        lb_pad=lb_pad,
        default_positive=default_pos,
        max_instances=int(layout_max_instances),
        device=device,
        dtype=weight_dtype,
    )

    # Convert grounding to pipeline format (List[List[float]] + Tensor)
    n_valid = int(ref_masks_t[0].sum().item())
    instdiff_boxes: Optional[List[List[float]]] = (
        ref_boxes[0, :n_valid].cpu().tolist() if n_valid > 0 else None
    )
    instdiff_embeddings: Optional[torch.Tensor] = (
        ref_pos[0, :n_valid] if n_valid > 0 else None
    )
    layout_boxes_norm = (
        ref_boxes[0, :n_valid].detach().float().cpu().numpy().astype(np.float32)
        if n_valid > 0
        else np.zeros((0, 4), dtype=np.float32)
    )

    # ── 5. Initial noise ──
    init_sigma = float(getattr(pipe.scheduler, "init_noise_sigma", 1.0))
    gen_noise = torch.Generator(device=device).manual_seed(int(global_seed))
    noise_unscaled = randn_tensor(
        masked_image_latents.shape,
        generator=gen_noise,
        device=device,
        dtype=weight_dtype,
    )
    latents_init = noise_unscaled * init_sigma

    # Shared prompt embeddings for global U-Net sampling.
    prompt_embeds_cond, _ = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    prompt_embeds_uncond, _ = pipe.encode_prompt(
        prompt="",
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )

    # ── 6. AAM optimization (cond branch) ──
    latents_opt_cond = latents_init
    if int(aam_iters) > 0:
        max_objs = 30
        aam_boxes = torch.zeros(max_objs, 4, device=device, dtype=weight_dtype)
        aam_text_emb = torch.zeros(max_objs, global_unet.config.cross_attention_dim, device=device, dtype=weight_dtype)
        aam_masks = torch.zeros(max_objs, device=device, dtype=weight_dtype)
        if n_valid > 0:
            aam_boxes[:n_valid] = ref_boxes[0, :n_valid]
            aam_text_emb[:n_valid] = ref_pos[0, :n_valid]
            aam_masks[:n_valid] = 1

        latents_opt_cond = _optimize_global_latents_with_aam(
            global_unet=global_unet,
            global_writer=global_writer,
            scheduler=pipe.scheduler,
            latents_init=latents_init,
            mask_latent=mask_latent,
            masked_image_latents=masked_image_latents,
            valid_latent=valid_latent,
            prompt_embeds=prompt_embeds_cond,
            ref_boxes=aam_boxes.unsqueeze(0),
            ref_masks=aam_masks.unsqueeze(0),
            ref_positive_embeddings=aam_text_emb.unsqueeze(0),
            steps=int(steps),
            aam_iters=int(aam_iters),
            aam_lr=float(aam_lr),
            aam_denoising_steps=int(aam_denoising_steps),
            aam_stop_loss=float(aam_stop_loss),
            aam_res=int(aam_res),
            aam_smooth=bool(aam_smooth),
            aam_tau_out=float(aam_tau_out),
            aam_lambda_tau=float(aam_lambda_tau),
            sample_id=str(sample_id),
            empty_attn_monitor=empty_attn_monitor,
        )

    # ── 7. Negative-AAM optimization (CFG uncond branch) ──
    run_neg_aam_uncond = bool(neg_aam_uncond) and float(global_guidance_scale) != 1.0
    latents_opt_uncond = latents_init
    if run_neg_aam_uncond and int(neg_aam_iters) > 0:
        max_objs = int(ref_masks_t.shape[1])
        neg_boxes = torch.zeros(max_objs, 4, device=device, dtype=weight_dtype)
        neg_text_emb = torch.zeros(max_objs, global_unet.config.cross_attention_dim, device=device, dtype=weight_dtype)
        neg_masks = torch.zeros(max_objs, device=device, dtype=weight_dtype)
        latents_opt_uncond = _optimize_global_latents_with_negative_aam(
            global_unet=global_unet,
            global_writer=global_writer,
            scheduler=pipe.scheduler,
            latents_init=latents_init,
            mask_latent=mask_latent,
            masked_image_latents=masked_image_latents,
            valid_latent=valid_latent,
            prompt_embeds=prompt_embeds_uncond,
            ref_boxes=neg_boxes.unsqueeze(0),
            ref_masks=neg_masks.unsqueeze(0),
            ref_positive_embeddings=neg_text_emb.unsqueeze(0),
            steps=int(steps),
            neg_aam_iters=int(neg_aam_iters),
            neg_aam_lr=float(neg_aam_lr),
            neg_aam_denoising_steps=int(neg_aam_denoising_steps),
            neg_aam_stop_loss=float(neg_aam_stop_loss),
            aam_res=int(aam_res),
            aam_smooth=bool(aam_smooth),
            neg_aam_tau_src=float(neg_aam_tau_src),
            neg_aam_lambda_tau=float(neg_aam_lambda_tau),
            sample_id=str(sample_id),
            empty_attn_monitor=empty_attn_monitor,
        )

    # ── 8. Global denoising + global U-Net bank cache ──
    global_writer.clear()
    pipe.set_scale(0.0 if bool(disable_global_layout_adapter) else 1.0)
    pipe.enable_fuser(not bool(disable_global_layout_adapter))
    if run_neg_aam_uncond:
        gen_pipe = torch.Generator(device=device).manual_seed(int(global_seed))
        global_latents, timesteps, bank_cache = _sample_global_and_cache_banks(
            global_unet=global_unet,
            global_writer=global_writer,
            scheduler=pipe.scheduler,
            global_masked_latents=masked_image_latents,
            global_mask_latent=mask_latent,
            global_prompt_embeds=prompt_embeds_cond,
            global_neg_embeds=prompt_embeds_uncond,
            ref_boxes=ref_boxes,
            ref_masks=ref_masks_t,
            ref_positive_embeddings=ref_pos,
            steps=int(steps),
            global_guidance_scale=float(global_guidance_scale),
            eta=float(eta),
            generator=gen_pipe,
            latents_init=latents_opt_cond,
            latents_init_uncond=latents_opt_uncond,
            cfg_base=str(neg_aam_cfg_base),
            cache_bank_features=bool(use_bank_features),
            cache_uncond_bank_features=bool(capture_global_uncond_bank),
        )
        global_writer.clear()
    else:
        latents_for_pipe = (latents_opt_cond / init_sigma).to(device=device, dtype=weight_dtype)
        gen_pipe = torch.Generator(device=device).manual_seed(int(global_seed))
        out = pipe(
            prompt=prompt,
            image=image_pil,
            mask_image=mask_pil,
            height=patch_size,
            width=patch_size,
            num_inference_steps=int(steps),
            guidance_scale=float(global_guidance_scale),
            instdiff_scheduled_sampling_alpha=1.0,
            instdiff_scheduled_sampling_beta=0.0,
            instdiff_boxes=instdiff_boxes,
            instdiff_positive_embeddings=instdiff_embeddings,
            eta=float(eta),
            generator=gen_pipe,
            latents=latents_for_pipe,
            output_type="latent",
        )
        pipe.scheduler.set_timesteps(int(steps), device=device)
        timesteps = pipe.scheduler.timesteps
        do_cfg = float(global_guidance_scale) != 1.0
        bank_cache = _extract_bank_cache_from_writer(
            global_writer=global_writer,
            num_steps=len(timesteps),
            bsz=1,
            cache_bank_features=bool(use_bank_features),
            cache_uncond=bool(capture_global_uncond_bank),
            do_cfg=do_cfg,
            device=device,
        )
        global_writer.clear()
        global_latents = out.images  # raw latents (output_type="latent")

    # ── 9. Decode ──
    global_gen_lb = _decode_vae(vae, global_latents)[0]
    global_gen_canvas = torch_unletter(
        global_gen_lb,
        scale=float(lb_scale),
        pad=lb_pad,
        orig_size=(canvas_h, canvas_w),
        mode="bilinear",
    ).clamp(-1, 1)

    masked_input_u8 = np.asarray(image_pil, dtype=np.uint8)
    return global_gen_lb, global_gen_canvas, float(lb_scale), lb_pad, timesteps, bank_cache, masked_input_u8, layout_boxes_norm


@torch.no_grad()
def _sample_local_parallel(
    *,
    main_unet: UNet2DConditionModel,
    patch_unifusion: torch.nn.Module,
    main_fuser_injector: DiffusersUNetFuserBankInjector,
    scheduler: DDIMScheduler,
    local_latents_init: torch.Tensor,
    local_masked_latents: torch.Tensor,
    local_mask_latent: torch.Tensor,
    local_window_bbox: torch.Tensor,
    local_prompt_embeds: torch.Tensor,
    local_neg_embeds: torch.Tensor,
    timesteps_ref: torch.Tensor,
    total_steps: int,
    bank_cache: RefBankCache,
    use_bank_features: bool,
    bank_injection_mode: str,
    disable_patch_token: bool,
    guidance_scale: float,
    eta: float,
    generator: torch.Generator,
    progress_desc: Optional[str] = None,
) -> torch.Tensor:
    device = local_latents_init.device
    dtype = local_latents_init.dtype

    valid_modes = {"cond_only", "both", "split_uc_cond"}
    mode = str(bank_injection_mode).strip().lower()
    if mode not in valid_modes:
        raise ValueError(f"Unknown bank injection mode: {bank_injection_mode}. expected one of {sorted(valid_modes)}")

    scheduler.set_timesteps(int(total_steps), device=device)
    timesteps = timesteps_ref.to(device=device)
    if bool(use_bank_features) and len(timesteps) != len(bank_cache.cond_per_step):
        raise RuntimeError(
            f"Timestep/bank cache mismatch: timesteps={len(timesteps)} bank_cache={len(bank_cache.cond_per_step)}"
        )
    if bool(use_bank_features) and float(guidance_scale) != 1.0 and mode == "split_uc_cond":
        if bank_cache.uncond_per_step is None:
            raise RuntimeError("split_uc_cond requires uncond global U-Net bank cache, but cache is missing.")
        if len(timesteps) != len(bank_cache.uncond_per_step):
            raise RuntimeError(
                "Timestep/uncond-bank cache mismatch: "
                f"timesteps={len(timesteps)} uncond_bank_cache={len(bank_cache.uncond_per_step)}"
            )

    local_latents = local_latents_init
    bsz = int(local_latents.shape[0])

    pos_null = (
        patch_unifusion.null_positive_feature.view(1, 1, -1)
        .expand(bsz, 1, -1)
        .to(device=device, dtype=dtype)
    )

    extra_step_kwargs = {"eta": float(eta), "generator": generator}

    step_iter = tqdm(timesteps, desc=progress_desc, dynamic_ncols=True) if progress_desc else timesteps
    for step_idx, t in enumerate(step_iter):
        if bool(use_bank_features):
            cond_bank_step = _repeat_bank_tokens(
                bank_cache.cond_per_step[step_idx],
                batch_size=bsz,
                device=device,
                dtype=dtype,
            )
            inject_mask: Optional[torch.Tensor] = None
            bank_step_to_inject: Dict[int, torch.Tensor] = cond_bank_step

            if float(guidance_scale) != 1.0:
                if mode == "cond_only":
                    inject_mask = torch.zeros((2 * bsz,), device=device, dtype=torch.bool)
                    inject_mask[bsz:] = True
                elif mode == "both":
                    inject_mask = None
                elif mode == "split_uc_cond":
                    assert bank_cache.uncond_per_step is not None
                    uncond_bank_step = _repeat_bank_tokens(
                        bank_cache.uncond_per_step[step_idx],
                        batch_size=bsz,
                        device=device,
                        dtype=dtype,
                    )
                    cond_keys = set(cond_bank_step.keys())
                    uncond_keys = set(uncond_bank_step.keys())
                    if cond_keys != uncond_keys:
                        raise RuntimeError(
                            "split_uc_cond key mismatch between cond/uncond banks at step "
                            f"{step_idx}: cond={sorted(cond_keys)} uncond={sorted(uncond_keys)}"
                        )
                    bank_step_to_inject = {
                        k: torch.cat([uncond_bank_step[k], cond_bank_step[k]], dim=0) for k in sorted(cond_keys)
                    }
                    inject_mask = None

            main_fuser_injector.update(bank_step_to_inject)
            main_fuser_injector.set_inject_mask(inject_mask)
        else:
            main_fuser_injector.clear()

        if float(guidance_scale) != 1.0:
            latent_model_input = torch.cat([local_latents] * 2, dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            mask_latent_cfg = torch.cat([local_mask_latent] * 2, dim=0)
            masked_image_latents_cfg = torch.cat([local_masked_latents] * 2, dim=0)
            local_model_input = torch.cat([latent_model_input, mask_latent_cfg, masked_image_latents_cfg], dim=1)

            local_embeds_cfg = torch.cat([local_neg_embeds, local_prompt_embeds], dim=0)
            bbox_cfg = torch.cat([local_window_bbox, local_window_bbox], dim=0)
            patch_masks_cfg = torch.ones((2 * bsz, 1), device=device, dtype=dtype)
            # Local patch token control:
            # - disable_patch_token=True: UC/COND both use null patch token (mask=0)
            # - otherwise keep canonical cond_only behavior (UC null, COND enabled)
            if bool(disable_patch_token):
                patch_masks_cfg.zero_()
            elif mode == "cond_only":
                patch_masks_cfg[:bsz] = 0
            pos_null_cfg = torch.cat([pos_null, pos_null], dim=0)

            patch_kwargs_cfg = {
                "instdiff": {
                    "boxes": bbox_cfg,
                    "positive_embeddings": pos_null_cfg,
                    "masks": patch_masks_cfg,
                }
            }

            local_noise_pred_2b = main_unet(
                local_model_input,
                t,
                encoder_hidden_states=local_embeds_cfg,
                cross_attention_kwargs=patch_kwargs_cfg,
            ).sample

            noise_u, noise_c = local_noise_pred_2b.chunk(2, dim=0)
            local_noise_pred = noise_u + float(guidance_scale) * (noise_c - noise_u)
        else:
            latent_model_input = scheduler.scale_model_input(local_latents, t)
            local_model_input = torch.cat([latent_model_input, local_mask_latent, local_masked_latents], dim=1)

            patch_masks = torch.zeros((bsz, 1), device=device, dtype=dtype) if bool(disable_patch_token) else torch.ones(
                (bsz, 1), device=device, dtype=dtype
            )
            patch_kwargs = {
                "instdiff": {
                    "boxes": local_window_bbox,
                    "positive_embeddings": pos_null,
                    "masks": patch_masks,
                }
            }

            local_noise_pred = main_unet(
                local_model_input,
                t,
                encoder_hidden_states=local_prompt_embeds,
                cross_attention_kwargs=patch_kwargs,
            ).sample

        local_latents = scheduler.step(local_noise_pred, t, local_latents, **extra_step_kwargs).prev_sample
        main_fuser_injector.clear()

    return local_latents


def _compute_local_latent_canvas_geometry(
    *,
    local_keys: List[str],
    windows: Dict[str, PatchWindow],
    patch_size: int,
    downscale: int = 8,
) -> Tuple[List[Tuple[int, int]], int, int, int]:
    if int(patch_size) % int(downscale) != 0:
        raise ValueError(f"patch_size must be divisible by downscale={downscale}, got patch_size={patch_size}")

    patch_lat = int(patch_size) // int(downscale)
    offsets: List[Tuple[int, int]] = []
    max_x1 = 0
    max_y1 = 0
    misaligned: List[str] = []
    for key in local_keys:
        win = windows[key]
        if (int(win.x) % int(downscale)) != 0 or (int(win.y) % int(downscale)) != 0:
            misaligned.append(f"{key}(x={win.x},y={win.y})")
        x0 = int(math.floor((float(win.x) / float(downscale)) + 0.5))
        y0 = int(math.floor((float(win.y) / float(downscale)) + 0.5))
        offsets.append((x0, y0))
        max_x1 = max(max_x1, x0 + patch_lat)
        max_y1 = max(max_y1, y0 + patch_lat)

    if misaligned:
        preview = ", ".join(misaligned[:6])
        extra = "" if len(misaligned) <= 6 else f", ... (+{len(misaligned) - 6} more)"
        print(
            "[Warn][B3-9] window offsets are not 8px-aligned; overlap averaging will use rounded latent coordinates. "
            f"misaligned={len(misaligned)} ({preview}{extra})"
        )

    return offsets, max_x1, max_y1, patch_lat


def _crop_canvas_to_patch_latents(
    *,
    canvas: torch.Tensor,  # [1,C,H,W]
    offsets_xy: List[Tuple[int, int]],
    patch_lat: int,
) -> torch.Tensor:
    if canvas.ndim != 4 or int(canvas.shape[0]) != 1:
        raise ValueError(f"Expected canvas [1,C,H,W], got {tuple(canvas.shape)}")

    out: List[torch.Tensor] = []
    for x0, y0 in offsets_xy:
        out.append(canvas[0, :, y0 : y0 + patch_lat, x0 : x0 + patch_lat])
    if not out:
        return torch.zeros((0, int(canvas.shape[1]), patch_lat, patch_lat), device=canvas.device, dtype=canvas.dtype)
    return torch.stack(out, dim=0)


def _make_center_weight_map(
    patch_lat: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a 2D Gaussian weight map [1, 1, patch_lat, patch_lat].

    sigma = patch_lat * 0.25 so that edge weights are ~13.5% of center.
    """
    sigma = patch_lat * 0.25
    coords = torch.arange(patch_lat, device=device, dtype=torch.float32) - (patch_lat - 1) / 2.0
    g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    g2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)  # [patch_lat, patch_lat]
    g2d = g2d / g2d.max()  # normalize peak to 1
    return g2d.to(dtype=dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, patch_lat, patch_lat]


def _avg_fuse_patch_latents_to_canvas(
    *,
    patches: torch.Tensor,  # [B,C,ph,pw]
    offsets_xy: List[Tuple[int, int]],
    canvas_lat_w: int,
    canvas_lat_h: int,
) -> torch.Tensor:
    if patches.ndim != 4:
        raise ValueError(f"Expected patches [B,C,H,W], got {tuple(patches.shape)}")
    bsz, ch, ph, pw = [int(x) for x in patches.shape]
    if ph != pw:
        raise ValueError(f"Expected square patch latents, got {(ph, pw)}")
    if bsz != int(len(offsets_xy)):
        raise ValueError(f"patches batch mismatch: patches={bsz} offsets={len(offsets_xy)}")

    device = patches.device
    dtype = patches.dtype
    value = torch.zeros((1, ch, int(canvas_lat_h), int(canvas_lat_w)), device=device, dtype=dtype)
    count = torch.zeros((1, 1, int(canvas_lat_h), int(canvas_lat_w)), device=device, dtype=dtype)

    for i, (x0, y0) in enumerate(offsets_xy):
        value[0, :, y0 : y0 + ph, x0 : x0 + pw] += patches[i]
        count[0, 0, y0 : y0 + ph, x0 : x0 + pw] += 1

    return value / count.clamp_min(1)


@torch.no_grad()
def _sample_local_multidiffusion(
    *,
    main_unet: UNet2DConditionModel,
    patch_unifusion: torch.nn.Module,
    main_fuser_injector: DiffusersUNetFuserBankInjector,
    scheduler: DDIMScheduler,
    local_latents_init: torch.Tensor,
    local_masked_latents: torch.Tensor,
    local_mask_latent: torch.Tensor,
    local_window_bbox: torch.Tensor,
    local_prompt_embeds: torch.Tensor,
    local_neg_embeds: torch.Tensor,
    timesteps_ref: torch.Tensor,
    total_steps: int,
    bank_cache: RefBankCache,
    use_bank_features: bool,
    bank_injection_mode: str,
    disable_patch_token: bool,
    guidance_scale: float,
    eta: float,
    generator: torch.Generator,
    local_keys: List[str],
    windows: Dict[str, PatchWindow],
    patch_size: int,
    progress_desc: Optional[str] = None,
    blend_mode: str = "uniform",
) -> torch.Tensor:
    """
    B3-9: MultiDiffusion-style local sampling.

    After each scheduler step (x_t -> x_{t-1}), average overlapping latent regions between
    patches using their known (x,y) window positions.

    blend_mode:
        "uniform"         – equal weight per patch (original 1/N averaging).
        "center_weighted" – Gaussian center-weight so each patch dominates its center
                            and fades at edges, reducing boundary seams.

    Note (temporary): Blending is all-to-all.
    Every patch contributes to overlap averaging, but only outpaint mask (mask==1) pixels are blended.
    """
    if blend_mode not in ("uniform", "center_weighted"):
        raise ValueError(f"Unknown blend_mode={blend_mode!r}. expected 'uniform' or 'center_weighted'")
    device = local_latents_init.device
    dtype = local_latents_init.dtype

    valid_modes = {"cond_only", "both", "split_uc_cond"}
    mode = str(bank_injection_mode).strip().lower()
    if mode not in valid_modes:
        raise ValueError(f"Unknown bank injection mode: {bank_injection_mode}. expected one of {sorted(valid_modes)}")

    scheduler.set_timesteps(int(total_steps), device=device)
    timesteps = timesteps_ref.to(device=device)
    if bool(use_bank_features) and len(timesteps) != len(bank_cache.cond_per_step):
        raise RuntimeError(
            f"Timestep/bank cache mismatch: timesteps={len(timesteps)} bank_cache={len(bank_cache.cond_per_step)}"
        )
    if bool(use_bank_features) and float(guidance_scale) != 1.0 and mode == "split_uc_cond":
        if bank_cache.uncond_per_step is None:
            raise RuntimeError("split_uc_cond requires uncond global U-Net bank cache, but cache is missing.")
        if len(timesteps) != len(bank_cache.uncond_per_step):
            raise RuntimeError(
                "Timestep/uncond-bank cache mismatch: "
                f"timesteps={len(timesteps)} uncond_bank_cache={len(bank_cache.uncond_per_step)}"
            )

    local_latents = local_latents_init
    bsz = int(local_latents.shape[0])
    if bsz != int(len(local_keys)):
        raise RuntimeError(f"local_latents batch mismatch: latents={bsz} local_keys={len(local_keys)}")

    offsets_xy, canvas_lat_w, canvas_lat_h, patch_lat = _compute_local_latent_canvas_geometry(
        local_keys=local_keys,
        windows=windows,
        patch_size=int(patch_size),
        downscale=8,
    )
    if patch_lat != int(local_latents.shape[-1]) or patch_lat != int(local_latents.shape[-2]):
        raise RuntimeError(
            "Unexpected local latent spatial size: "
            f"expected {(patch_lat, patch_lat)}, got {tuple(local_latents.shape[-2:])}"
        )

    use_center_weight = (blend_mode == "center_weighted")
    if use_center_weight:
        cw_map = _make_center_weight_map(patch_lat, device=device, dtype=dtype)  # [1,1,pl,pl]

    ch = int(local_latents.shape[1])
    value = torch.zeros((1, ch, canvas_lat_h, canvas_lat_w), device=device, dtype=dtype)
    count = torch.zeros((1, 1, canvas_lat_h, canvas_lat_w), device=device, dtype=dtype)

    pos_null = (
        patch_unifusion.null_positive_feature.view(1, 1, -1)
        .expand(bsz, 1, -1)
        .to(device=device, dtype=dtype)
    )
    extra_step_kwargs = {"eta": float(eta), "generator": generator}

    step_iter = tqdm(timesteps, desc=progress_desc, dynamic_ncols=True) if progress_desc else timesteps
    for step_idx, t in enumerate(step_iter):
        if bool(use_bank_features):
            cond_bank_step = _repeat_bank_tokens(
                bank_cache.cond_per_step[step_idx],
                batch_size=bsz,
                device=device,
                dtype=dtype,
            )
            inject_mask: Optional[torch.Tensor] = None
            bank_step_to_inject: Dict[int, torch.Tensor] = cond_bank_step

            if float(guidance_scale) != 1.0:
                if mode == "cond_only":
                    inject_mask = torch.zeros((2 * bsz,), device=device, dtype=torch.bool)
                    inject_mask[bsz:] = True
                elif mode == "both":
                    inject_mask = None
                elif mode == "split_uc_cond":
                    assert bank_cache.uncond_per_step is not None
                    uncond_bank_step = _repeat_bank_tokens(
                        bank_cache.uncond_per_step[step_idx],
                        batch_size=bsz,
                        device=device,
                        dtype=dtype,
                    )
                    cond_keys = set(cond_bank_step.keys())
                    uncond_keys = set(uncond_bank_step.keys())
                    if cond_keys != uncond_keys:
                        raise RuntimeError(
                            "split_uc_cond key mismatch between cond/uncond banks at step "
                            f"{step_idx}: cond={sorted(cond_keys)} uncond={sorted(uncond_keys)}"
                        )
                    bank_step_to_inject = {
                        k: torch.cat([uncond_bank_step[k], cond_bank_step[k]], dim=0) for k in sorted(cond_keys)
                    }
                    inject_mask = None

            main_fuser_injector.update(bank_step_to_inject)
            main_fuser_injector.set_inject_mask(inject_mask)
        else:
            main_fuser_injector.clear()

        if float(guidance_scale) != 1.0:
            latent_model_input = torch.cat([local_latents] * 2, dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            mask_latent_cfg = torch.cat([local_mask_latent] * 2, dim=0)
            masked_image_latents_cfg = torch.cat([local_masked_latents] * 2, dim=0)
            local_model_input = torch.cat([latent_model_input, mask_latent_cfg, masked_image_latents_cfg], dim=1)

            local_embeds_cfg = torch.cat([local_neg_embeds, local_prompt_embeds], dim=0)
            bbox_cfg = torch.cat([local_window_bbox, local_window_bbox], dim=0)
            patch_masks_cfg = torch.ones((2 * bsz, 1), device=device, dtype=dtype)
            # Local patch token control:
            # - disable_patch_token=True: UC/COND both use null patch token (mask=0)
            # - otherwise keep canonical cond_only behavior (UC null, COND enabled)
            if bool(disable_patch_token):
                patch_masks_cfg.zero_()
            elif mode == "cond_only":
                patch_masks_cfg[:bsz] = 0
            pos_null_cfg = torch.cat([pos_null, pos_null], dim=0)

            patch_kwargs_cfg = {
                "instdiff": {
                    "boxes": bbox_cfg,
                    "positive_embeddings": pos_null_cfg,
                    "masks": patch_masks_cfg,
                }
            }

            local_noise_pred_2b = main_unet(
                local_model_input,
                t,
                encoder_hidden_states=local_embeds_cfg,
                cross_attention_kwargs=patch_kwargs_cfg,
            ).sample

            noise_u, noise_c = local_noise_pred_2b.chunk(2, dim=0)
            local_noise_pred = noise_u + float(guidance_scale) * (noise_c - noise_u)
        else:
            latent_model_input = scheduler.scale_model_input(local_latents, t)
            local_model_input = torch.cat([latent_model_input, local_mask_latent, local_masked_latents], dim=1)

            patch_masks = torch.zeros((bsz, 1), device=device, dtype=dtype) if bool(disable_patch_token) else torch.ones(
                (bsz, 1), device=device, dtype=dtype
            )
            patch_kwargs = {
                "instdiff": {
                    "boxes": local_window_bbox,
                    "positive_embeddings": pos_null,
                    "masks": patch_masks,
                }
            }

            local_noise_pred = main_unet(
                local_model_input,
                t,
                encoder_hidden_states=local_prompt_embeds,
                cross_attention_kwargs=patch_kwargs,
            ).sample

        local_latents_next = scheduler.step(local_noise_pred, t, local_latents, **extra_step_kwargs).prev_sample

        # MultiDiffusion blending (mask-aware, out-only, all-to-all):
        # - Blend only pixels that are in the inpaint/outpaint region (mask==1) for each patch.
        # - Keep src/known pixels (mask==0) untouched to reduce source↔generated boundary artifacts.
        value.zero_()
        count.zero_()
        for i, (x0, y0) in enumerate(offsets_xy):
            m_ch = local_mask_latent[i].to(dtype=dtype)  # [1,pl,pl] in {0,1}
            m = m_ch[0]  # [pl,pl]
            w = m if not use_center_weight else (cw_map[0, 0] * m)
            value[0, :, y0 : y0 + patch_lat, x0 : x0 + patch_lat] += local_latents_next[i] * w
            count[0, 0, y0 : y0 + patch_lat, x0 : x0 + patch_lat] += w

        avg_out = value / count.clamp_min(torch.finfo(count.dtype).tiny)
        fused: List[torch.Tensor] = []
        for i, (x0, y0) in enumerate(offsets_xy):
            m_ch = local_mask_latent[i].to(dtype=dtype)  # [1,pl,pl]
            patch_avg = avg_out[0, :, y0 : y0 + patch_lat, x0 : x0 + patch_lat]
            patch_next = local_latents_next[i]
            fused.append(patch_next * (1.0 - m_ch) + patch_avg * m_ch)

        local_latents = torch.stack(fused, dim=0)

        main_fuser_injector.clear()

    return local_latents


def _select_local_schedule_and_banks(
    *,
    timesteps: torch.Tensor,
    bank_cache: RefBankCache,
    use_forward_diffusion: bool,
    forward_strength: float,
) -> Tuple[torch.Tensor, RefBankCache]:
    if not bool(use_forward_diffusion):
        return timesteps, bank_cache

    strength = float(forward_strength)
    if not (0.0 < strength <= 1.0):
        raise ValueError(f"--forward_strength must be in (0, 1], got {forward_strength}")

    n_steps = int(len(timesteps))
    if n_steps <= 0:
        raise RuntimeError("timesteps must be non-empty.")

    skip_steps = int(math.floor((1.0 - strength) * n_steps))
    skip_steps = max(0, min(skip_steps, n_steps - 1))
    if skip_steps == 0:
        return timesteps, bank_cache

    sliced_cond = bank_cache.cond_per_step[skip_steps:]
    sliced_uncond = bank_cache.uncond_per_step[skip_steps:] if bank_cache.uncond_per_step is not None else None
    return timesteps[skip_steps:], RefBankCache(cond_per_step=sliced_cond, uncond_per_step=sliced_uncond)


@torch.no_grad()
def _prepare_local_init_latents(
    *,
    vae: AutoencoderKL,
    scheduler_local: DDIMScheduler,
    global_gen_canvas: torch.Tensor,
    windows: Dict[str, PatchWindow],
    local_keys: List[str],
    patch_size: int,
    local_masked_latents: torch.Tensor,
    local_timesteps: torch.Tensor,
    use_forward_diffusion: bool,
    prefuse_to_canvas: bool,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    offsets_xy: Optional[List[Tuple[int, int]]] = None
    canvas_lat_w: Optional[int] = None
    canvas_lat_h: Optional[int] = None
    patch_lat: Optional[int] = None
    if bool(prefuse_to_canvas) and len(local_keys) > 1:
        offsets_xy, canvas_lat_w, canvas_lat_h, patch_lat = _compute_local_latent_canvas_geometry(
            local_keys=local_keys,
            windows=windows,
            patch_size=int(patch_size),
            downscale=8,
        )

    if bool(use_forward_diffusion):
        local_init_clean: List[torch.Tensor] = []
        for key in local_keys:
            win = windows[key]
            crop = global_gen_canvas[:, win.y : win.y + patch_size, win.x : win.x + patch_size]
            if crop.shape[-2:] != (patch_size, patch_size):
                raise RuntimeError(
                    f"Invalid local crop size for key={key}: got {tuple(crop.shape[-2:])}, expected {(patch_size, patch_size)}"
                )
            local_init_clean.append(crop)

        local_init_clean_t = torch.stack(local_init_clean, dim=0).to(device=device, dtype=dtype)
        local_init_latents_clean = _encode_vae(vae, local_init_clean_t)

        if int(local_timesteps.numel()) <= 0:
            raise RuntimeError("local_timesteps must be non-empty for forward diffusion init.")
        t0_scalar = local_timesteps[0].to(device=device).to(dtype=torch.long)

        if (
            bool(prefuse_to_canvas)
            and offsets_xy is not None
            and canvas_lat_w is not None
            and canvas_lat_h is not None
            and patch_lat is not None
        ):
            # Do NOT average-fuse clean latents to a shared canvas (causes unwanted smoothing).
            # Instead, keep per-patch clean latents and use a shared noise canvas so overlapping
            # regions still receive consistent noise.
            gen_local_init = torch.Generator(device=device).manual_seed(int(seed))
            noise_canvas = randn_tensor(
                (1, int(local_init_latents_clean.shape[1]), int(canvas_lat_h), int(canvas_lat_w)),
                generator=gen_local_init,
                device=device,
                dtype=dtype,
            )
            noise_patches = _crop_canvas_to_patch_latents(
                canvas=noise_canvas,
                offsets_xy=offsets_xy,
                patch_lat=int(patch_lat),
            )
            t0 = t0_scalar.repeat(local_init_latents_clean.shape[0])
            return scheduler_local.add_noise(local_init_latents_clean, noise_patches, t0)

        t0 = t0_scalar.repeat(local_init_latents_clean.shape[0])
        gen_local_init = torch.Generator(device=device).manual_seed(int(seed))
        local_noise = randn_tensor(
            local_init_latents_clean.shape,
            generator=gen_local_init,
            device=device,
            dtype=dtype,
        )
        return scheduler_local.add_noise(local_init_latents_clean, local_noise, t0)

    init_sigma = float(getattr(scheduler_local, "init_noise_sigma", 1.0))
    gen_local_init = torch.Generator(device=device).manual_seed(int(seed))
    if bool(prefuse_to_canvas) and offsets_xy is not None and canvas_lat_w is not None and canvas_lat_h is not None and patch_lat is not None:
        noise_canvas = randn_tensor(
            (1, int(local_masked_latents.shape[1]), int(canvas_lat_h), int(canvas_lat_w)),
            generator=gen_local_init,
            device=device,
            dtype=dtype,
        )
        noisy_canvas = noise_canvas * float(init_sigma)
        return _crop_canvas_to_patch_latents(canvas=noisy_canvas, offsets_xy=offsets_xy, patch_lat=int(patch_lat))

    local_latents_init = randn_tensor(
        local_masked_latents.shape,
        generator=gen_local_init,
        device=device,
        dtype=dtype,
    )
    return local_latents_init * init_sigma


def _split_shards(total: int, world_size: int) -> Tuple[List[int], List[int]]:
    """Return (sizes, starts) for contiguous sharding of range(total) across ranks."""
    total = int(total)
    world_size = max(1, int(world_size))
    base = total // world_size
    rem = total % world_size

    sizes = [base + (1 if r < rem else 0) for r in range(world_size)]
    starts: List[int] = []
    cur = 0
    for sz in sizes:
        starts.append(cur)
        cur += int(sz)
    return sizes, starts


@torch.no_grad()
def _prepare_local_init_latents_mgpu(
    *,
    vae: AutoencoderKL,
    scheduler_local: DDIMScheduler,
    global_gen_canvas: torch.Tensor,
    windows: Dict[str, PatchWindow],
    local_keys: List[str],
    patch_size: int,
    local_masked_latents: torch.Tensor,
    local_timesteps: torch.Tensor,
    use_forward_diffusion: bool,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    # Global canvas geometry for shared noise between patches.
    offsets_xy_local: List[Tuple[int, int]],
    canvas_lat_w: int,
    canvas_lat_h: int,
    patch_lat: int,
) -> torch.Tensor:
    """Local init latents with globally shared noise canvas (MultiDiffusion multi-GPU)."""
    if int(local_masked_latents.shape[0]) == 0:
        return torch.zeros((0, 4, int(patch_lat), int(patch_lat)), device=device, dtype=dtype)

    if bool(use_forward_diffusion):
        local_init_clean: List[torch.Tensor] = []
        for key in local_keys:
            win = windows[key]
            crop = global_gen_canvas[:, win.y : win.y + patch_size, win.x : win.x + patch_size]
            if crop.shape[-2:] != (patch_size, patch_size):
                raise RuntimeError(
                    f"Invalid local crop size for key={key}: got {tuple(crop.shape[-2:])}, expected {(patch_size, patch_size)}"
                )
            local_init_clean.append(crop)

        local_init_clean_t = torch.stack(local_init_clean, dim=0).to(device=device, dtype=dtype)
        local_init_latents_clean = _encode_vae(vae, local_init_clean_t)

        if int(local_timesteps.numel()) <= 0:
            raise RuntimeError("local_timesteps must be non-empty for forward diffusion init.")
        t0_scalar = local_timesteps[0].to(device=device).to(dtype=torch.long)

        gen_local_init = torch.Generator(device=device).manual_seed(int(seed))
        noise_canvas = randn_tensor(
            (1, int(local_init_latents_clean.shape[1]), int(canvas_lat_h), int(canvas_lat_w)),
            generator=gen_local_init,
            device=device,
            dtype=dtype,
        )
        noise_patches = _crop_canvas_to_patch_latents(
            canvas=noise_canvas,
            offsets_xy=offsets_xy_local,
            patch_lat=int(patch_lat),
        )
        t0 = t0_scalar.repeat(int(local_init_latents_clean.shape[0]))
        return scheduler_local.add_noise(local_init_latents_clean, noise_patches, t0)

    init_sigma = float(getattr(scheduler_local, "init_noise_sigma", 1.0))
    gen_local_init = torch.Generator(device=device).manual_seed(int(seed))
    noise_canvas = randn_tensor(
        (1, int(local_masked_latents.shape[1]), int(canvas_lat_h), int(canvas_lat_w)),
        generator=gen_local_init,
        device=device,
        dtype=dtype,
    )
    noisy_canvas = noise_canvas * float(init_sigma)
    return _crop_canvas_to_patch_latents(canvas=noisy_canvas, offsets_xy=offsets_xy_local, patch_lat=int(patch_lat))


@torch.no_grad()
def _sample_local_multidiffusion_allreduce(
    *,
    main_unet: UNet2DConditionModel,
    patch_unifusion: torch.nn.Module,
    main_fuser_injector: DiffusersUNetFuserBankInjector,
    scheduler: DDIMScheduler,
    local_latents_init: torch.Tensor,
    local_masked_latents: torch.Tensor,
    local_mask_latent: torch.Tensor,
    local_window_bbox: torch.Tensor,
    local_prompt_embeds: torch.Tensor,
    local_neg_embeds: torch.Tensor,
    timesteps_ref: torch.Tensor,
    total_steps: int,
    bank_cache: RefBankCache,
    use_bank_features: bool,
    bank_injection_mode: str,
    disable_patch_token: bool,
    guidance_scale: float,
    eta: float,
    generator: torch.Generator,
    offsets_xy_local: List[Tuple[int, int]],
    canvas_lat_w: int,
    canvas_lat_h: int,
    patch_lat: int,
    dist_world_size: int,
    progress_desc: Optional[str] = None,
    blend_mode: str = "uniform",
) -> torch.Tensor:
    """MultiDiffusion local sampling with per-step overlap blending via all_reduce(value/count)."""
    if blend_mode not in ("uniform", "center_weighted"):
        raise ValueError(f"Unknown blend_mode={blend_mode!r}. expected 'uniform' or 'center_weighted'")

    valid_modes = {"cond_only", "both", "split_uc_cond"}
    mode = str(bank_injection_mode).strip().lower()
    if mode not in valid_modes:
        raise ValueError(f"Unknown bank injection mode: {bank_injection_mode}. expected one of {sorted(valid_modes)}")

    device = local_latents_init.device
    dtype = local_latents_init.dtype

    scheduler.set_timesteps(int(total_steps), device=device)
    timesteps = timesteps_ref.to(device=device)
    if bool(use_bank_features) and len(timesteps) != len(bank_cache.cond_per_step):
        raise RuntimeError(
            f"Timestep/bank cache mismatch: timesteps={len(timesteps)} bank_cache={len(bank_cache.cond_per_step)}"
        )
    if bool(use_bank_features) and float(guidance_scale) != 1.0 and mode == "split_uc_cond":
        if bank_cache.uncond_per_step is None:
            raise RuntimeError("split_uc_cond requires uncond global U-Net bank cache, but cache is missing.")
        if len(timesteps) != len(bank_cache.uncond_per_step):
            raise RuntimeError(
                "Timestep/uncond-bank cache mismatch: "
                f"timesteps={len(timesteps)} uncond_bank_cache={len(bank_cache.uncond_per_step)}"
            )

    local_latents = local_latents_init
    bsz = int(local_latents.shape[0])

    use_center_weight = (blend_mode == "center_weighted")
    if use_center_weight:
        cw_map = _make_center_weight_map(int(patch_lat), device=device, dtype=dtype)  # [1,1,pl,pl]

    ch = int(local_latents.shape[1])
    value = torch.zeros((1, ch, int(canvas_lat_h), int(canvas_lat_w)), device=device, dtype=dtype)
    count = torch.zeros((1, 1, int(canvas_lat_h), int(canvas_lat_w)), device=device, dtype=dtype)

    pos_null = (
        patch_unifusion.null_positive_feature.view(1, 1, -1)
        .expand(max(bsz, 1), 1, -1)
        .to(device=device, dtype=dtype)
    )
    extra_step_kwargs = {"eta": float(eta), "generator": generator}

    step_iter = tqdm(timesteps, desc=progress_desc, dynamic_ncols=True) if progress_desc else timesteps
    for step_idx, t in enumerate(step_iter):
        if bsz > 0 and bool(use_bank_features):
            cond_bank_step = _repeat_bank_tokens(
                bank_cache.cond_per_step[step_idx],
                batch_size=bsz,
                device=device,
                dtype=dtype,
            )
            inject_mask: Optional[torch.Tensor] = None
            bank_step_to_inject: Dict[int, torch.Tensor] = cond_bank_step

            if float(guidance_scale) != 1.0:
                if mode == "cond_only":
                    inject_mask = torch.zeros((2 * bsz,), device=device, dtype=torch.bool)
                    inject_mask[bsz:] = True
                elif mode == "both":
                    inject_mask = None
                elif mode == "split_uc_cond":
                    assert bank_cache.uncond_per_step is not None
                    uncond_bank_step = _repeat_bank_tokens(
                        bank_cache.uncond_per_step[step_idx],
                        batch_size=bsz,
                        device=device,
                        dtype=dtype,
                    )
                    cond_keys = set(cond_bank_step.keys())
                    uncond_keys = set(uncond_bank_step.keys())
                    if cond_keys != uncond_keys:
                        raise RuntimeError(
                            "split_uc_cond key mismatch between cond/uncond banks at step "
                            f"{step_idx}: cond={sorted(cond_keys)} uncond={sorted(uncond_keys)}"
                        )
                    bank_step_to_inject = {
                        k: torch.cat([uncond_bank_step[k], cond_bank_step[k]], dim=0) for k in sorted(cond_keys)
                    }
                    inject_mask = None

            main_fuser_injector.update(bank_step_to_inject)
            main_fuser_injector.set_inject_mask(inject_mask)
        else:
            main_fuser_injector.clear()

        if bsz > 0:
            if float(guidance_scale) != 1.0:
                latent_model_input = torch.cat([local_latents] * 2, dim=0)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                mask_latent_cfg = torch.cat([local_mask_latent] * 2, dim=0)
                masked_image_latents_cfg = torch.cat([local_masked_latents] * 2, dim=0)
                local_model_input = torch.cat([latent_model_input, mask_latent_cfg, masked_image_latents_cfg], dim=1)

                local_embeds_cfg = torch.cat([local_neg_embeds, local_prompt_embeds], dim=0)
                bbox_cfg = torch.cat([local_window_bbox, local_window_bbox], dim=0)
                patch_masks_cfg = torch.ones((2 * bsz, 1), device=device, dtype=dtype)
                if bool(disable_patch_token):
                    patch_masks_cfg.zero_()
                elif mode == "cond_only":
                    patch_masks_cfg[:bsz] = 0
                pos_null_cfg = torch.cat([pos_null[:bsz], pos_null[:bsz]], dim=0)

                patch_kwargs_cfg = {
                    "instdiff": {
                        "boxes": bbox_cfg,
                        "positive_embeddings": pos_null_cfg,
                        "masks": patch_masks_cfg,
                    }
                }

                local_noise_pred_2b = main_unet(
                    local_model_input,
                    t,
                    encoder_hidden_states=local_embeds_cfg,
                    cross_attention_kwargs=patch_kwargs_cfg,
                ).sample

                noise_u, noise_c = local_noise_pred_2b.chunk(2, dim=0)
                local_noise_pred = noise_u + float(guidance_scale) * (noise_c - noise_u)
            else:
                latent_model_input = scheduler.scale_model_input(local_latents, t)
                local_model_input = torch.cat([latent_model_input, local_mask_latent, local_masked_latents], dim=1)

                patch_masks = (
                    torch.zeros((bsz, 1), device=device, dtype=dtype)
                    if bool(disable_patch_token)
                    else torch.ones((bsz, 1), device=device, dtype=dtype)
                )
                patch_kwargs = {
                    "instdiff": {
                        "boxes": local_window_bbox,
                        "positive_embeddings": pos_null[:bsz],
                        "masks": patch_masks,
                    }
                }

                local_noise_pred = main_unet(
                    local_model_input,
                    t,
                    encoder_hidden_states=local_prompt_embeds,
                    cross_attention_kwargs=patch_kwargs,
                ).sample

            local_latents_next = scheduler.step(local_noise_pred, t, local_latents, **extra_step_kwargs).prev_sample
        else:
            local_latents_next = local_latents

        # MultiDiffusion blending (mask-aware, out-only) via global all_reduce:
        value.zero_()
        count.zero_()
        for i, (x0, y0) in enumerate(offsets_xy_local):
            m_ch = local_mask_latent[i].to(dtype=dtype)  # [1,pl,pl] in {0,1}
            m = m_ch[0]  # [pl,pl]
            w = m if not use_center_weight else (cw_map[0, 0] * m)
            value[0, :, y0 : y0 + int(patch_lat), x0 : x0 + int(patch_lat)] += local_latents_next[i] * w
            count[0, 0, y0 : y0 + int(patch_lat), x0 : x0 + int(patch_lat)] += w

        if int(dist_world_size) > 1:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)

        avg_out = value / count.clamp_min(torch.finfo(count.dtype).tiny)
        fused: List[torch.Tensor] = []
        for i, (x0, y0) in enumerate(offsets_xy_local):
            m_ch = local_mask_latent[i].to(dtype=dtype)  # [1,pl,pl]
            patch_avg = avg_out[0, :, y0 : y0 + int(patch_lat), x0 : x0 + int(patch_lat)]
            patch_next = local_latents_next[i]
            fused.append(patch_next * (1.0 - m_ch) + patch_avg * m_ch)

        local_latents = torch.stack(fused, dim=0) if fused else local_latents
        main_fuser_injector.clear()

    return local_latents


def run_full_model_parallel(
    *,
    pipe: StableDiffusionINSTDIFFInpaintPipelineBBoxOnly,
    vae: AutoencoderKL,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    scheduler_local: DDIMScheduler,
    main_unet: UNet2DConditionModel,
    patch_unifusion: torch.nn.Module,
    global_writer: DiffusersGlobalUNetFuserWriter,
    main_fuser_injector: DiffusersUNetFuserBankInjector,
    center_img: np.ndarray,
    sample_id: str = "",
    prompt: str,
    local_prompt: str,
    negative_prompt: str,
    base_seed: int,
    patch_size: int,
    overlap_ratio_x: float,
    overlap_ratio_y: float,
    order: List[str],
    steps: int,
    guidance_scale: float,
    eta: float,
    layout: GlobalLayoutCondition,
    layout_max_instances: int,
    aam_iters: int,
    aam_lr: float,
    aam_denoising_steps: int,
    aam_stop_loss: float,
    aam_res: int,
    aam_smooth: bool,
    aam_tau_out: float,
    aam_lambda_tau: float,
    neg_aam_uncond: bool,
    neg_aam_iters: int,
    neg_aam_lr: float,
    neg_aam_denoising_steps: int,
    neg_aam_stop_loss: float,
    neg_aam_tau_src: float,
    neg_aam_lambda_tau: float,
    neg_aam_cfg_base: str,
    global_mask_dilate_px: int,
    use_bank_features: bool,
    use_forward_diffusion: bool,
    progressive_generation: bool,
    global_cfg_enabled: bool,
    bank_injection_mode: str,
    forward_strength: float,
    local_multidiffusion: bool = False,
    local_md_blend_mode: str = "uniform",
    disable_patch_token: bool = False,
    final_composite_mode: str = "first_wins",
    final_blend_feather_radius: int = 2,
    final_blend_feather_strength: float = 0.35,
    sample_tqdm_desc: Optional[str] = None,
    orig_img: Optional[np.ndarray] = None,
    center_bbox: Optional[Tuple[int, int, int, int]] = None,
    debug_sample_dir: Optional[Path] = None,
    disable_global_layout_adapter: bool = False,
    empty_attn_monitor: Optional[AAMEmptyAttnMonitor] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if int(patch_size) != 512:
        raise ValueError("This runner expects patch_size=512.")
    compose_mode = str(final_composite_mode).strip().lower()
    if compose_mode not in {"first_wins", "blend"}:
        raise ValueError(
            f"Unknown final_composite_mode: {final_composite_mode}. expected one of ['first_wins', 'blend']"
        )
    blend_radius = max(0, int(final_blend_feather_radius))
    blend_strength = float(np.clip(float(final_blend_feather_strength), 0.0, 1.0))

    center_h, center_w = int(center_img.shape[0]), int(center_img.shape[1])
    windows, canvas_w, canvas_h, cx, cy = plan_windows_band_ratio(
        patch_size=patch_size,
        overlap_ratio_x=overlap_ratio_x,
        overlap_ratio_y=overlap_ratio_y,
        center_w=center_w,
        center_h=center_h,
        order=order,
    )

    init_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    composer = PriorityComposer(canvas_h, canvas_w)
    init_canvas[cy : cy + center_h, cx : cx + center_w] = center_img
    composer.mark_center(cx, cy, center_h, center_w)
    final_canvas = init_canvas.copy()

    debug_global_dir: Optional[Path] = None
    debug_local_dir: Optional[Path] = None
    if debug_sample_dir is not None:
        debug_global_dir = Path(debug_sample_dir) / "global"
        debug_local_dir = Path(debug_sample_dir) / "local"
        debug_global_dir.mkdir(parents=True, exist_ok=True)
        debug_local_dir.mkdir(parents=True, exist_ok=True)

    orig_u8: Optional[np.ndarray] = None
    if orig_img is not None:
        orig_u8 = np.asarray(orig_img, dtype=np.uint8)
    gt_canvas = _build_gt_canvas(
        orig_u8=orig_u8,
        center_bbox=center_bbox,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        cx=cx,
        cy=cy,
    )
    debug_stage_idx = 0

    if not str(prompt).strip():
        prompt = " "
    if not str(local_prompt).strip():
        local_prompt = " "

    device = next(main_unet.parameters()).device
    weight_dtype = next(main_unet.parameters()).dtype
    dist_world_size = int(dist.get_world_size()) if dist.is_initialized() else 1
    dist_rank = int(dist.get_rank()) if dist.is_initialized() else 0

    global_prompt_embeds_1 = _encode_prompts(tokenizer, text_encoder, [prompt], device=device, dtype=weight_dtype)
    local_prompt_embeds_1 = _encode_prompts(
        tokenizer,
        text_encoder,
        [local_prompt],
        device=device,
        dtype=weight_dtype,
    )
    local_neg_embeds_1 = _encode_prompts(
        tokenizer,
        text_encoder,
        [negative_prompt],
        device=device,
        dtype=weight_dtype,
    )
    global_guidance_scale = float(guidance_scale) if bool(global_cfg_enabled) else 1.0
    inject_mode = str(bank_injection_mode).strip().lower()
    if bool(use_bank_features) and inject_mode == "split_uc_cond":
        if float(guidance_scale) != 1.0 and float(global_guidance_scale) == 1.0:
            raise ValueError(
                "split_uc_cond requires global U-Net CFG ON when local CFG is enabled. "
                "Enable the internal global CFG path before using split_uc_cond."
            )
    capture_global_uncond_bank = (
        bool(use_bank_features)
        and inject_mode == "split_uc_cond"
        and float(global_guidance_scale) != 1.0
    )

    if bool(progressive_generation):
        if dist_world_size > 1:
            raise NotImplementedError("progressive_generation is not supported in multi-GPU mode.")
        last_global_lb: Optional[torch.Tensor] = None
        patch_iter = tqdm(order, desc=sample_tqdm_desc, dynamic_ncols=True) if sample_tqdm_desc else order
        for patch_idx, key in enumerate(patch_iter, start=1):
            local_keys, local_rgbs_u8, local_masks_u8 = _build_local_inputs_for_keys(
                canvas=final_canvas,
                composer=composer,
                windows=windows,
                keys=[key],
                patch_size=patch_size,
            )
            if len(local_keys) == 0:
                continue

            seed_base = int(base_seed) + 2000 + int(patch_idx) * 100
            (
                global_gen_lb,
                global_gen_canvas,
                lb_scale,
                lb_pad,
                timesteps,
                bank_cache,
                global_masked_u8,
                global_layout_boxes,
            ) = _run_global_stage(
                pipe=pipe,
                global_writer=global_writer,
                sample_id=str(sample_id),
                canvas_u8=final_canvas,
                composer=composer,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                cx=cx,
                cy=cy,
                patch_size=patch_size,
                prompt=prompt,
                global_guidance_scale=float(global_guidance_scale),
                eta=float(eta),
                steps=int(steps),
                global_seed=seed_base + 1,
                layout=layout,
                layout_max_instances=int(layout_max_instances),
                use_bank_features=bool(use_bank_features),
                capture_global_uncond_bank=bool(capture_global_uncond_bank),
                aam_iters=int(aam_iters),
                aam_lr=float(aam_lr),
                aam_denoising_steps=int(aam_denoising_steps),
                aam_stop_loss=float(aam_stop_loss),
                aam_res=int(aam_res),
                aam_smooth=bool(aam_smooth),
                aam_tau_out=float(aam_tau_out),
                aam_lambda_tau=float(aam_lambda_tau),
                neg_aam_uncond=bool(neg_aam_uncond),
                neg_aam_iters=int(neg_aam_iters),
                neg_aam_lr=float(neg_aam_lr),
                neg_aam_denoising_steps=int(neg_aam_denoising_steps),
                neg_aam_stop_loss=float(neg_aam_stop_loss),
                neg_aam_tau_src=float(neg_aam_tau_src),
                neg_aam_lambda_tau=float(neg_aam_lambda_tau),
                neg_aam_cfg_base=str(neg_aam_cfg_base),
                global_mask_dilate_px=int(global_mask_dilate_px),
                disable_global_layout_adapter=bool(disable_global_layout_adapter),
                empty_attn_monitor=empty_attn_monitor,
            )
            last_global_lb = global_gen_lb
            debug_stage_idx += 1

            global_gt_lb_u8 = _letterbox_u8_rgb(gt_canvas, patch_size) if gt_canvas is not None else None
            if debug_global_dir is not None:
                _save_global_debug_views(
                    out_dir=debug_global_dir,
                    stage_idx=debug_stage_idx,
                    masked_input_u8=global_masked_u8,
                    ground_truth_u8=global_gt_lb_u8,
                    layout_boxes_norm=global_layout_boxes,
                    generated_u8=to_image_uint8(global_gen_lb.unsqueeze(0))[0],
                )

            local_timesteps, local_bank_cache = _select_local_schedule_and_banks(
                timesteps=timesteps,
                bank_cache=bank_cache,
                use_forward_diffusion=bool(use_forward_diffusion),
                forward_strength=float(forward_strength),
            )

            local_masked_latents, local_mask_latent = _prepare_local_condition_latents(
                vae=vae,
                local_rgbs_u8=local_rgbs_u8,
                local_masks_u8=local_masks_u8,
                device=device,
                dtype=weight_dtype,
            )
            local_window_bbox = _build_local_window_bbox_batch(
                local_keys=local_keys,
                windows=windows,
                patch_size=patch_size,
                lb_scale=lb_scale,
                lb_pad=lb_pad,
                device=device,
                dtype=weight_dtype,
            )

            local_latents_init = _prepare_local_init_latents(
                vae=vae,
                scheduler_local=scheduler_local,
                global_gen_canvas=global_gen_canvas,
                windows=windows,
                local_keys=local_keys,
                patch_size=patch_size,
                local_masked_latents=local_masked_latents,
                local_timesteps=local_timesteps,
                use_forward_diffusion=bool(use_forward_diffusion),
                prefuse_to_canvas=bool(local_multidiffusion),
                seed=seed_base + 2,
                device=device,
                dtype=weight_dtype,
            )

            local_debug_tags: List[str] = []
            if debug_local_dir is not None:
                for i, local_key in enumerate(local_keys, start=1):
                    win = windows[local_key]
                    local_gt_u8 = (
                        gt_canvas[win.y : win.y + patch_size, win.x : win.x + patch_size].copy()
                        if gt_canvas is not None
                        else local_rgbs_u8[i - 1].copy()
                    )
                    tag = _save_local_debug_input_views(
                        out_dir=debug_local_dir,
                        stage_idx=debug_stage_idx,
                        local_idx=i,
                        local_key=local_key,
                        masked_input_u8=_compose_masked_input_u8(local_rgbs_u8[i - 1], local_masks_u8[i - 1]),
                        ground_truth_u8=local_gt_u8,
                    )
                    local_debug_tags.append(tag)

            local_prompt_embeds = local_prompt_embeds_1.expand(local_latents_init.shape[0], -1, -1).contiguous()
            local_neg_embeds = local_neg_embeds_1.expand(local_latents_init.shape[0], -1, -1).contiguous()

            gen_local_step = torch.Generator(device=device).manual_seed(seed_base + 3)
            local_latents = (
                _sample_local_multidiffusion(
                    main_unet=main_unet,
                    patch_unifusion=patch_unifusion,
                    main_fuser_injector=main_fuser_injector,
                    scheduler=scheduler_local,
                    local_latents_init=local_latents_init,
                    local_masked_latents=local_masked_latents,
                    local_mask_latent=local_mask_latent,
                    local_window_bbox=local_window_bbox,
                    local_prompt_embeds=local_prompt_embeds,
                    local_neg_embeds=local_neg_embeds,
                    timesteps_ref=local_timesteps,
                    total_steps=int(steps),
                    bank_cache=local_bank_cache,
                    use_bank_features=bool(use_bank_features),
                    bank_injection_mode=inject_mode,
                    disable_patch_token=bool(disable_patch_token),
                    guidance_scale=float(guidance_scale),
                    eta=float(eta),
                    generator=gen_local_step,
                    local_keys=local_keys,
                    windows=windows,
                    patch_size=int(patch_size),
                    progress_desc=None,
                    blend_mode=local_md_blend_mode,
                )
                if bool(local_multidiffusion) and len(local_keys) > 1
                else _sample_local_parallel(
                    main_unet=main_unet,
                    patch_unifusion=patch_unifusion,
                    main_fuser_injector=main_fuser_injector,
                    scheduler=scheduler_local,
                    local_latents_init=local_latents_init,
                    local_masked_latents=local_masked_latents,
                    local_mask_latent=local_mask_latent,
                    local_window_bbox=local_window_bbox,
                    local_prompt_embeds=local_prompt_embeds,
                    local_neg_embeds=local_neg_embeds,
                    timesteps_ref=local_timesteps,
                    total_steps=int(steps),
                    bank_cache=local_bank_cache,
                    use_bank_features=bool(use_bank_features),
                    bank_injection_mode=inject_mode,
                    disable_patch_token=bool(disable_patch_token),
                    guidance_scale=float(guidance_scale),
                    eta=float(eta),
                    generator=gen_local_step,
                    progress_desc=None,
                )
            )

            local_out_u8 = to_image_uint8(_decode_vae(vae, local_latents))
            for i, local_key in enumerate(local_keys, start=1):
                if debug_local_dir is not None and len(local_debug_tags) >= i:
                    _save_local_debug_output_view(
                        out_dir=debug_local_dir,
                        tag=local_debug_tags[i - 1],
                        generated_u8=local_out_u8[i - 1],
                    )
                win = windows[local_key]
                composer.composite(
                    canvas=final_canvas,
                    out_patch=local_out_u8[i - 1],
                    mask_fill=local_masks_u8[i - 1],
                    x=win.x,
                    y=win.y,
                    pid=patch_idx,
                    mode=compose_mode,
                    blend_feather_radius=blend_radius,
                    blend_feather_strength=blend_strength,
                )

        if last_global_lb is None:
            (
                last_global_lb,
                _global_gen_canvas,
                _lb_scale,
                _lb_pad,
                _timesteps,
                _bank_cache,
                global_masked_u8,
                global_layout_boxes,
            ) = _run_global_stage(
                pipe=pipe,
                global_writer=global_writer,
                sample_id=str(sample_id),
                canvas_u8=final_canvas,
                composer=composer,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                cx=cx,
                cy=cy,
                patch_size=patch_size,
                prompt=prompt,
                global_guidance_scale=float(global_guidance_scale),
                eta=float(eta),
                steps=int(steps),
                global_seed=int(base_seed) + 2101,
                layout=layout,
                layout_max_instances=int(layout_max_instances),
                use_bank_features=bool(use_bank_features),
                capture_global_uncond_bank=bool(capture_global_uncond_bank),
                aam_iters=int(aam_iters),
                aam_lr=float(aam_lr),
                aam_denoising_steps=int(aam_denoising_steps),
                aam_stop_loss=float(aam_stop_loss),
                aam_res=int(aam_res),
                aam_smooth=bool(aam_smooth),
                aam_tau_out=float(aam_tau_out),
                aam_lambda_tau=float(aam_lambda_tau),
                neg_aam_uncond=bool(neg_aam_uncond),
                neg_aam_iters=int(neg_aam_iters),
                neg_aam_lr=float(neg_aam_lr),
                neg_aam_denoising_steps=int(neg_aam_denoising_steps),
                neg_aam_stop_loss=float(neg_aam_stop_loss),
                neg_aam_tau_src=float(neg_aam_tau_src),
                neg_aam_lambda_tau=float(neg_aam_lambda_tau),
                neg_aam_cfg_base=str(neg_aam_cfg_base),
                global_mask_dilate_px=int(global_mask_dilate_px),
                disable_global_layout_adapter=bool(disable_global_layout_adapter),
                empty_attn_monitor=empty_attn_monitor,
            )
            debug_stage_idx += 1
            global_gt_lb_u8 = _letterbox_u8_rgb(gt_canvas, patch_size) if gt_canvas is not None else None
            if debug_global_dir is not None:
                _save_global_debug_views(
                    out_dir=debug_global_dir,
                    stage_idx=debug_stage_idx,
                    masked_input_u8=global_masked_u8,
                    ground_truth_u8=global_gt_lb_u8,
                    layout_boxes_norm=global_layout_boxes,
                    generated_u8=to_image_uint8(last_global_lb.unsqueeze(0))[0],
                )

        global_u8 = to_image_uint8(last_global_lb.unsqueeze(0))[0]
        return final_canvas, init_canvas, global_u8, composer.filled_by.copy()

    (
        global_gen_lb,
        global_gen_canvas,
        lb_scale,
        lb_pad,
        timesteps,
        bank_cache,
        global_masked_u8,
        global_layout_boxes,
    ) = _run_global_stage(
        pipe=pipe,
        global_writer=global_writer,
        sample_id=str(sample_id),
        canvas_u8=init_canvas,
        composer=composer,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        cx=cx,
        cy=cy,
        patch_size=patch_size,
        prompt=prompt,
        global_guidance_scale=float(global_guidance_scale),
        eta=float(eta),
        steps=int(steps),
        global_seed=int(base_seed),
        layout=layout,
        layout_max_instances=int(layout_max_instances),
        use_bank_features=bool(use_bank_features),
        capture_global_uncond_bank=bool(capture_global_uncond_bank),
        aam_iters=int(aam_iters),
        aam_lr=float(aam_lr),
        aam_denoising_steps=int(aam_denoising_steps),
        aam_stop_loss=float(aam_stop_loss),
        aam_res=int(aam_res),
        aam_smooth=bool(aam_smooth),
        aam_tau_out=float(aam_tau_out),
        aam_lambda_tau=float(aam_lambda_tau),
        neg_aam_uncond=bool(neg_aam_uncond),
        neg_aam_iters=int(neg_aam_iters),
        neg_aam_lr=float(neg_aam_lr),
        neg_aam_denoising_steps=int(neg_aam_denoising_steps),
        neg_aam_stop_loss=float(neg_aam_stop_loss),
        neg_aam_tau_src=float(neg_aam_tau_src),
        neg_aam_lambda_tau=float(neg_aam_lambda_tau),
        neg_aam_cfg_base=str(neg_aam_cfg_base),
        global_mask_dilate_px=int(global_mask_dilate_px),
        disable_global_layout_adapter=bool(disable_global_layout_adapter),
        empty_attn_monitor=empty_attn_monitor,
    )
    debug_stage_idx += 1
    global_gt_lb_u8 = _letterbox_u8_rgb(gt_canvas, patch_size) if gt_canvas is not None else None
    if debug_global_dir is not None:
        _save_global_debug_views(
            out_dir=debug_global_dir,
            stage_idx=debug_stage_idx,
            masked_input_u8=global_masked_u8,
            ground_truth_u8=global_gt_lb_u8,
            layout_boxes_norm=global_layout_boxes,
            generated_u8=to_image_uint8(global_gen_lb.unsqueeze(0))[0],
        )

    local_timesteps, local_bank_cache = _select_local_schedule_and_banks(
        timesteps=timesteps,
        bank_cache=bank_cache,
        use_forward_diffusion=bool(use_forward_diffusion),
        forward_strength=float(forward_strength),
    )

    local_keys, local_rgbs_u8, local_masks_u8 = _build_local_inputs_for_keys(
        canvas=init_canvas,
        composer=composer,
        windows=windows,
        keys=order,
        patch_size=patch_size,
    )
    if len(local_keys) == 0:
        global_u8 = to_image_uint8(global_gen_lb.unsqueeze(0))[0]
        return final_canvas, init_canvas, global_u8, composer.filled_by.copy()

    local_debug_tags: List[str] = []
    if debug_local_dir is not None:
        for i, local_key in enumerate(local_keys, start=1):
            win = windows[local_key]
            local_gt_u8 = (
                gt_canvas[win.y : win.y + patch_size, win.x : win.x + patch_size].copy()
                if gt_canvas is not None
                else local_rgbs_u8[i - 1].copy()
            )
            tag = _save_local_debug_input_views(
                out_dir=debug_local_dir,
                stage_idx=debug_stage_idx,
                local_idx=i,
                local_key=local_key,
                masked_input_u8=_compose_masked_input_u8(local_rgbs_u8[i - 1], local_masks_u8[i - 1]),
                ground_truth_u8=local_gt_u8,
            )
            local_debug_tags.append(tag)

    if dist_world_size > 1 and bool(local_multidiffusion):
        offsets_all, canvas_lat_w, canvas_lat_h, patch_lat = _compute_local_latent_canvas_geometry(
            local_keys=local_keys,
            windows=windows,
            patch_size=int(patch_size),
            downscale=8,
        )
        sizes, starts = _split_shards(len(local_keys), dist_world_size)
        shard_start = int(starts[dist_rank])
        shard_size = int(sizes[dist_rank])
        shard_end = shard_start + shard_size

        shard_keys = local_keys[shard_start:shard_end]
        shard_rgbs_u8 = local_rgbs_u8[shard_start:shard_end]
        shard_masks_u8 = local_masks_u8[shard_start:shard_end]
        shard_offsets_xy = offsets_all[shard_start:shard_end]

        shard_masked_latents, shard_mask_latent = _prepare_local_condition_latents(
            vae=vae,
            local_rgbs_u8=shard_rgbs_u8,
            local_masks_u8=shard_masks_u8,
            device=device,
            dtype=weight_dtype,
        )
        shard_window_bbox = _build_local_window_bbox_batch(
            local_keys=shard_keys,
            windows=windows,
            patch_size=patch_size,
            lb_scale=lb_scale,
            lb_pad=lb_pad,
            device=device,
            dtype=weight_dtype,
        )
        shard_latents_init = _prepare_local_init_latents_mgpu(
            vae=vae,
            scheduler_local=scheduler_local,
            global_gen_canvas=global_gen_canvas,
            windows=windows,
            local_keys=shard_keys,
            patch_size=patch_size,
            local_masked_latents=shard_masked_latents,
            local_timesteps=local_timesteps,
            use_forward_diffusion=bool(use_forward_diffusion),
            seed=int(base_seed) + 202,
            device=device,
            dtype=weight_dtype,
            offsets_xy_local=shard_offsets_xy,
            canvas_lat_w=int(canvas_lat_w),
            canvas_lat_h=int(canvas_lat_h),
            patch_lat=int(patch_lat),
        )

        shard_prompt_embeds = local_prompt_embeds_1.expand(int(shard_latents_init.shape[0]), -1, -1).contiguous()
        shard_neg_embeds = local_neg_embeds_1.expand(int(shard_latents_init.shape[0]), -1, -1).contiguous()

        gen_local_step = torch.Generator(device=device).manual_seed(int(base_seed) + 303)
        shard_latents = _sample_local_multidiffusion_allreduce(
            main_unet=main_unet,
            patch_unifusion=patch_unifusion,
            main_fuser_injector=main_fuser_injector,
            scheduler=scheduler_local,
            local_latents_init=shard_latents_init,
            local_masked_latents=shard_masked_latents,
            local_mask_latent=shard_mask_latent,
            local_window_bbox=shard_window_bbox,
            local_prompt_embeds=shard_prompt_embeds,
            local_neg_embeds=shard_neg_embeds,
            timesteps_ref=local_timesteps,
            total_steps=int(steps),
            bank_cache=local_bank_cache,
            use_bank_features=bool(use_bank_features),
            bank_injection_mode=inject_mode,
            disable_patch_token=bool(disable_patch_token),
            guidance_scale=float(guidance_scale),
            eta=float(eta),
            generator=gen_local_step,
            offsets_xy_local=shard_offsets_xy,
            canvas_lat_w=int(canvas_lat_w),
            canvas_lat_h=int(canvas_lat_h),
            patch_lat=int(patch_lat),
            dist_world_size=int(dist_world_size),
            progress_desc=sample_tqdm_desc if dist_rank == 0 else None,
            blend_mode=local_md_blend_mode,
        )

        max_shard = int(max(sizes))
        if int(shard_latents.shape[0]) < max_shard:
            pad = torch.zeros(
                (max_shard - int(shard_latents.shape[0]),) + tuple(shard_latents.shape[1:]),
                device=device,
                dtype=shard_latents.dtype,
            )
            shard_send = torch.cat([shard_latents, pad], dim=0)
        else:
            shard_send = shard_latents

        gathered: List[torch.Tensor] = [torch.empty_like(shard_send) for _ in range(dist_world_size)]
        dist.all_gather(gathered, shard_send)

        if dist_rank == 0:
            all_latents_parts: List[torch.Tensor] = []
            for r in range(dist_world_size):
                sz = int(sizes[r])
                if sz <= 0:
                    continue
                all_latents_parts.append(gathered[r][:sz])
            all_latents = (
                torch.cat(all_latents_parts, dim=0)
                if all_latents_parts
                else torch.zeros((0, 4, int(patch_lat), int(patch_lat)), device=device, dtype=weight_dtype)
            )

            local_out_u8 = to_image_uint8(_decode_vae(vae, all_latents))
            for i, key in enumerate(local_keys, start=1):
                if debug_local_dir is not None and len(local_debug_tags) >= i:
                    _save_local_debug_output_view(
                        out_dir=debug_local_dir,
                        tag=local_debug_tags[i - 1],
                        generated_u8=local_out_u8[i - 1],
                    )
                win = windows[key]
                composer.composite(
                    canvas=final_canvas,
                    out_patch=local_out_u8[i - 1],
                    mask_fill=local_masks_u8[i - 1],
                    x=win.x,
                    y=win.y,
                    pid=i,
                    mode=compose_mode,
                    blend_feather_radius=blend_radius,
                    blend_feather_strength=blend_strength,
                )
    else:
        local_masked_latents, local_mask_latent = _prepare_local_condition_latents(
            vae=vae,
            local_rgbs_u8=local_rgbs_u8,
            local_masks_u8=local_masks_u8,
            device=device,
            dtype=weight_dtype,
        )
        local_window_bbox = _build_local_window_bbox_batch(
            local_keys=local_keys,
            windows=windows,
            patch_size=patch_size,
            lb_scale=lb_scale,
            lb_pad=lb_pad,
            device=device,
            dtype=weight_dtype,
        )
        local_latents_init = _prepare_local_init_latents(
            vae=vae,
            scheduler_local=scheduler_local,
            global_gen_canvas=global_gen_canvas,
            windows=windows,
            local_keys=local_keys,
            patch_size=patch_size,
            local_masked_latents=local_masked_latents,
            local_timesteps=local_timesteps,
            use_forward_diffusion=bool(use_forward_diffusion),
            prefuse_to_canvas=bool(local_multidiffusion),
            seed=int(base_seed) + 202,
            device=device,
            dtype=weight_dtype,
        )
        local_prompt_embeds = local_prompt_embeds_1.expand(local_latents_init.shape[0], -1, -1).contiguous()
        local_neg_embeds = local_neg_embeds_1.expand(local_latents_init.shape[0], -1, -1).contiguous()
        gen_local_step = torch.Generator(device=device).manual_seed(int(base_seed) + 303)
        local_latents = (
            _sample_local_multidiffusion(
                main_unet=main_unet,
                patch_unifusion=patch_unifusion,
                main_fuser_injector=main_fuser_injector,
                scheduler=scheduler_local,
                local_latents_init=local_latents_init,
                local_masked_latents=local_masked_latents,
                local_mask_latent=local_mask_latent,
                local_window_bbox=local_window_bbox,
                local_prompt_embeds=local_prompt_embeds,
                local_neg_embeds=local_neg_embeds,
                timesteps_ref=local_timesteps,
                total_steps=int(steps),
                bank_cache=local_bank_cache,
                use_bank_features=bool(use_bank_features),
                bank_injection_mode=inject_mode,
                disable_patch_token=bool(disable_patch_token),
                guidance_scale=float(guidance_scale),
                eta=float(eta),
                generator=gen_local_step,
                local_keys=local_keys,
                windows=windows,
                patch_size=int(patch_size),
                progress_desc=sample_tqdm_desc,
                blend_mode=local_md_blend_mode,
            )
            if bool(local_multidiffusion) and len(local_keys) > 1
            else _sample_local_parallel(
                main_unet=main_unet,
                patch_unifusion=patch_unifusion,
                main_fuser_injector=main_fuser_injector,
                scheduler=scheduler_local,
                local_latents_init=local_latents_init,
                local_masked_latents=local_masked_latents,
                local_mask_latent=local_mask_latent,
                local_window_bbox=local_window_bbox,
                local_prompt_embeds=local_prompt_embeds,
                local_neg_embeds=local_neg_embeds,
                timesteps_ref=local_timesteps,
                total_steps=int(steps),
                bank_cache=local_bank_cache,
                use_bank_features=bool(use_bank_features),
                bank_injection_mode=inject_mode,
                disable_patch_token=bool(disable_patch_token),
                guidance_scale=float(guidance_scale),
                eta=float(eta),
                generator=gen_local_step,
                progress_desc=sample_tqdm_desc,
            )
        )

        local_out_u8 = to_image_uint8(_decode_vae(vae, local_latents))
        for i, key in enumerate(local_keys, start=1):
            if debug_local_dir is not None and len(local_debug_tags) >= i:
                _save_local_debug_output_view(
                    out_dir=debug_local_dir,
                    tag=local_debug_tags[i - 1],
                    generated_u8=local_out_u8[i - 1],
                )
            win = windows[key]
            composer.composite(
                canvas=final_canvas,
                out_patch=local_out_u8[i - 1],
                mask_fill=local_masks_u8[i - 1],
                x=win.x,
                y=win.y,
                pid=i,
                mode=compose_mode,
                blend_feather_radius=blend_radius,
                blend_feather_strength=blend_strength,
            )

    global_u8 = to_image_uint8(global_gen_lb.unsqueeze(0))[0]
    return final_canvas, init_canvas, global_u8, composer.filled_by.copy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BlueOut full-model parallel outpainting (global U-Net + main U-Net)")

    # Model / checkpoints
    p.add_argument(
        "--global_ckpt",
        type=str,
        default="checkpoints/stage1_global_unet.pt",
        help="Stage 1 global U-Net checkpoint.",
    )
    p.add_argument(
        "--main_ckpt",
        type=str,
        default="checkpoints/stage2_main_adapters.pt",
    )
    p.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-inpainting")
    p.add_argument("--instdiff_model", type=str, default="kyeongry/instancediffusion_sd15")
    p.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--global_token_scale", type=float, default=1.0)

    # Data
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_samples", type=int, default=-1, help="<=0: process all samples")
    p.add_argument("--images_root", type=str, default="datasets/images/iconart")
    p.add_argument("--captions_file", type=str, default="datasets/caption/iconart/blip2_prompts.json")
    p.add_argument("--annotations_root", type=str, default="datasets/annotations/iconart")
    p.add_argument("--r", type=float, default=0.333)
    p.add_argument("--target_long_edge", type=int, default=1408)

    # Outpainting layout
    p.add_argument("--patch_size", type=int, default=512)
    p.add_argument("--order", type=str, default="N,E,S,W,NE,NW,SE,SW")
    p.add_argument("--layout_max_instances", type=int, default=30)

    # Sampling
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--guidance_scale", type=float, default=3.0)
    p.add_argument(
        "--forward_strength",
        type=float,
        default=1.0,
        help="Forward diffusion strength for local init when forward diffusion is ON. (0,1], default=1.0",
    )
    p.add_argument(
        "--local_init_pure_gaussian",
        action="store_true",
        help="Disable forward diffusion for local init; start local patch denoising from pure Gaussian noise.",
    )
    p.add_argument(
        "--disable_global_feature_injection",
        action="store_true",
        help="Disable Stage 1 global U-Net feature injection in the local stage.",
    )
    p.add_argument(
        "--disable_patch_token",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stage-2 ablation: use null local patch token (mask=0) while keeping global feature injection path.",
    )
    p.add_argument(
        "--disable_global_layout_adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable the Stage 1 global U-Net layout adapter/fuser during global-stage denoising.",
    )
    p.add_argument(
        "--progressive_generation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable ProOut-style progressive generation: update the canvas patch-by-patch "
            "and rerun Stage-1 blueprint generation before each next patch. "
            "Single-GPU only (set NPROC_PER_NODE=1)."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--local_prompt",
        type=str,
        default="harmonized painting, high resolution, best quality, high quality, harmonized simple background",
        help="Main UNet local conditioning prompt (cond branch text for local stage).",
    )
    p.add_argument(
        "--negative_prompt",
        type=str,
        default="ugly, nsfw, worst quality, watermark, signature, logo",
    )
    # Global-stage AAM (AlignNoise-style)
    p.add_argument("--aam_iters", type=int, default=10, help="AAM optimization iterations for the global stage.")
    p.add_argument("--aam_lr", type=float, default=1e-2, help="AAM Adam lr for (mu, log_var).")
    p.add_argument("--aam_denoising_steps", type=int, default=1, help="Timesteps per AAM iter used for loss.")
    p.add_argument("--aam_stop_loss", type=float, default=0.55, help="Early-stop threshold for AAM loss.")
    p.add_argument("--aam_res", type=int, default=16, help="Attention resolution for AAM map aggregation.")
    p.add_argument(
        "--aam_smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable AlignNoise-style Gaussian smoothing in AAM.",
    )
    p.add_argument("--aam_tau_out", type=float, default=0.15)
    p.add_argument("--aam_lambda_tau", type=float, default=1.0)
    p.add_argument(
        "--neg_aam_uncond",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable negative-AAM on the CFG unconditional branch with step-0 split latent sampling.",
    )
    p.add_argument("--neg_aam_iters", type=int, default=10, help="Negative-AAM optimization iterations for uncond branch.")
    p.add_argument("--neg_aam_lr", type=float, default=1e-2, help="Negative-AAM Adam lr for (mu, log_var).")
    p.add_argument("--neg_aam_denoising_steps", type=int, default=1, help="Timesteps per negative-AAM iter used for loss.")
    p.add_argument("--neg_aam_stop_loss", type=float, default=0.55, help="Early-stop threshold for negative-AAM loss.")
    p.add_argument("--neg_aam_tau_src", type=float, default=0.15, help="Hinge lower-bound target for AS_src in negative-AAM.")
    p.add_argument("--neg_aam_lambda_tau", type=float, default=1.0, help="Hinge weight for negative-AAM.")
    p.add_argument(
        "--neg_aam_cfg_base",
        type=str,
        default="uncond",
        choices=["uncond", "cond"],
        help="CFG combination base for step-0 split sampling when --neg_aam_uncond is enabled.",
    )
    p.add_argument(
        "--global_mask_dilate_px",
        type=int,
        default=1,
        help="Dilate global outpaint mask by N pixels after thresholding. default=1",
    )
    p.add_argument(
        "--main_query_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restrict global-bank attention queries to main U-Net tokens. Default is False.",
    )
    p.add_argument(
        "--final_composite_mode",
        type=str,
        default="first_wins",
        choices=["first_wins", "blend"],
        help="Final patch composition mode. default=first_wins (legacy).",
    )
    p.add_argument(
        "--final_blend_feather_radius",
        type=int,
        default=2,
        help="Only used when final_composite_mode=blend.",
    )
    p.add_argument(
        "--final_blend_feather_strength",
        type=float,
        default=0.35,
        help="Only used when final_composite_mode=blend. Range [0,1].",
    )
    # Output
    p.add_argument("--outdir", type=str, default="results/ours")
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Resume into the latest existing session directory that matches the current session type "
            "under --outdir. If none exists, start a new session."
        ),
    )
    p.add_argument(
        "--resume_session_dir",
        type=str,
        default="",
        help="Resume into an explicit existing session directory path.",
    )
    p.add_argument(
        "--resume_skip_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When resuming, skip samples whose output image already exists.",
    )
    p.add_argument("--save_ext", type=str, default="png", choices=["png", "jpg", "jpeg", "webp"])
    p.add_argument(
        "--save_comparison",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save side-by-side images (init canvas vs final).",
    )
    p.add_argument(
        "--save_global",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save Stage-1 global output image (letterboxed 512x512).",
    )
    p.add_argument(
        "--save_debug_io",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save global/local debug IO (masked input, GT, layout condition, generated patches).",
    )
    p.add_argument(
        "--force_save_comparison",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force-save comparison images for auditability regardless of --save_comparison.",
    )
    p.add_argument(
        "--save_center_crop_to_orig",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save final output center-cropped (or center-padded) to original image size.",
    )

    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _find_latest_matching_session(base_outdir: Path, session_type: str) -> Optional[Path]:
    if not base_outdir.exists():
        return None

    suffix = f"_{session_type}"
    matches: List[Path] = [p for p in base_outdir.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    if not matches:
        return None

    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _init_distributed() -> Tuple[bool, int, int, int]:
    """Initialize torch.distributed from torchrun env vars (RANK/WORLD_SIZE/LOCAL_RANK)."""
    if not dist.is_available():
        return False, 0, 1, 0

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def main() -> None:
    args = parse_args()
    # Keep audit-comparison as default behavior, but allow explicit opt-out.
    if bool(getattr(args, "force_save_comparison", True)):
        args.save_comparison = True

    # Canonical release configuration.
    disable_global_layout_adapter = bool(args.disable_global_layout_adapter)
    disable_global_feature_injection = bool(args.disable_global_feature_injection)
    use_bank_features = not bool(disable_global_feature_injection)
    use_forward_diffusion = not bool(args.local_init_pure_gaussian)
    progressive_generation = bool(args.progressive_generation)
    global_cfg_enabled = True
    bank_injection_mode = "cond_only"
    main_query_only = bool(args.main_query_only)
    local_multidiffusion = True
    local_md_blend_mode = "uniform"
    final_composite_mode = str(args.final_composite_mode).strip().lower()

    if int(args.batch_size) != 1:
        raise ValueError("This runner expects --batch_size=1.")
    if int(args.patch_size) != 512:
        raise ValueError("--patch_size must be 512 for SD v1.5 inpainting.")
    if bool(use_forward_diffusion) and not (0.0 < float(args.forward_strength) <= 1.0):
        raise ValueError(f"--forward_strength must be in (0, 1], got {args.forward_strength}")

    dist_enabled, dist_rank, dist_world_size, dist_local_rank = _init_distributed()
    is_main_process = (dist_rank == 0)
    if bool(progressive_generation) and int(dist_world_size) > 1:
        raise ValueError(
            "--progressive_generation currently supports single-GPU only. "
            "Set NPROC_PER_NODE=1 and use a single CUDA_VISIBLE_DEVICES id."
        )

    create_dirs = ["outputs"]
    if bool(args.save_comparison):
        create_dirs.append("comparisons")
    if bool(args.save_global):
        create_dirs.append("conditioning")

    if main_query_only and bank_injection_mode == "both" and final_composite_mode == "first_wins":
        variant_tag = "blueout_main_query_only"
    elif (not main_query_only) and bank_injection_mode == "both" and final_composite_mode == "first_wins":
        variant_tag = "blueout_both_branches"
    elif (not main_query_only) and bank_injection_mode == "cond_only" and final_composite_mode == "first_wins":
        variant_tag = "blueout"
    elif (not main_query_only) and bank_injection_mode == "both" and final_composite_mode == "blend":
        variant_tag = "blueout_blend"
    else:
        variant_tag = (
            f"blueout_custom_{bank_injection_mode}_{final_composite_mode}"
            f"{'_mq' if main_query_only else ''}"
        )
    session_type = f"{variant_tag}_gs{args.guidance_scale:g}_{args.steps}step"
    if bool(progressive_generation):
        session_type += "_progressive"
    if bool(args.neg_aam_uncond):
        session_type += "_aam_negative"
    if not bool(use_forward_diffusion):
        session_type += "_initgauss"
    elif abs(float(args.forward_strength) - 1.0) > 1e-8:
        session_type += f"_fs{str(args.forward_strength).replace('.', 'p')}"
    if bool(disable_global_feature_injection):
        session_type += "_local_no_global"
    if bool(disable_global_layout_adapter):
        session_type += "_global_layout_off"
    if bool(args.disable_patch_token):
        session_type += "_ptnull"

    session_dir: Path
    outputs_dir: Path
    comparisons_dir: Path
    conditioning_dir: Path
    explicit_resume_dir = str(args.resume_session_dir).strip()
    resume_enabled = bool(args.resume) or bool(explicit_resume_dir)
    if is_main_process:
        resume_target: Optional[Path] = None
        if explicit_resume_dir:
            candidate = Path(explicit_resume_dir).expanduser()
            if not candidate.is_absolute():
                candidate = (Path.cwd() / candidate).resolve()
            if not candidate.exists() or not candidate.is_dir():
                raise FileNotFoundError(f"--resume_session_dir does not exist or is not a directory: {candidate}")
            resume_target = candidate
            print(f"[Info] resume enabled: explicit session dir -> {resume_target}")
        elif bool(args.resume):
            latest = _find_latest_matching_session(Path(args.outdir), session_type)
            if latest is not None:
                resume_target = latest
                print(f"[Info] resume enabled: latest matching session -> {resume_target}")
            else:
                print("[Info] resume requested but no matching prior session found; creating a new session.")

        if resume_target is None:
            session_dir, _, outputs_dir, comparisons_dir, conditioning_dir, _ = create_output_structure(
                args.outdir,
                type=session_type,
                create=create_dirs,
            )
        else:
            session_dir = resume_target
            outputs_dir = session_dir / "outputs"
            comparisons_dir = session_dir / "comparisons"
            conditioning_dir = session_dir / "conditioning"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            if bool(args.save_comparison):
                comparisons_dir.mkdir(parents=True, exist_ok=True)
            if bool(args.save_global):
                conditioning_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Info] Session dir: {session_dir}")
            print(f"[Info] - Outputs: {outputs_dir}")
            if bool(args.save_comparison):
                print(f"[Info] - Comparisons: {comparisons_dir}")
            if bool(args.save_global):
                print(f"[Info] - Conditioning: {conditioning_dir}")
    else:
        # Workers participate in sampling, but don't write outputs.
        session_dir = Path(args.outdir)
        outputs_dir = session_dir
        comparisons_dir = session_dir
        conditioning_dir = session_dir

    if dist_enabled:
        shared_session: List[str] = [str(session_dir) if is_main_process else ""]
        dist.broadcast_object_list(shared_session, src=0)
        session_dir = Path(shared_session[0])
        outputs_dir = session_dir / "outputs"
        comparisons_dir = session_dir / "comparisons"
        conditioning_dir = session_dir / "conditioning"

    debug_io_root: Optional[Path] = None
    if is_main_process and bool(args.save_debug_io):
        debug_io_root = session_dir / "debug_io"
        debug_io_root.mkdir(parents=True, exist_ok=True)
        print(f"[Info] - Debug IO: {debug_io_root}")

    empty_attn_monitor: Optional[AAMEmptyAttnMonitor] = None
    if is_main_process and bool(disable_global_layout_adapter):
        empty_attn_monitor = AAMEmptyAttnMonitor(rank=int(dist_rank), verbose=True)

    device = torch.device("cuda", dist_local_rank) if torch.cuda.is_available() else torch.device("cpu")
    dtype = _resolve_torch_dtype(args.precision)
    set_seed(int(args.seed))

    if is_main_process:
        forward_strength_log = (
            f"{float(args.forward_strength):.3f}" if bool(use_forward_diffusion) else "N/A(local_init_pure_gaussian)"
        )
        print(f"[Info] device={device}, dtype={dtype}")
        print(f"[Info] loading components from base model: {args.base_model}")
        print(
            "[Info] config="
            "BlueOut (multi-gpu) "
            f"(world={dist_world_size}, "
            f"bank={use_bank_features}, "
            f"forward={use_forward_diffusion}, "
            f"progressive={progressive_generation}, "
            f"local_md={local_multidiffusion}, "
            f"md_blend={local_md_blend_mode}, "
            f"global_cfg={global_cfg_enabled}, "
            f"bank_mode={bank_injection_mode}, "
            f"main_query_only={main_query_only}, "
            f"global_layout_adapter={not bool(disable_global_layout_adapter)}, "
            f"global_feature_injection={not bool(disable_global_feature_injection)}, "
            f"disable_patch_token={bool(args.disable_patch_token)}, "
            f"final_composite={final_composite_mode}, "
            f"final_blend_radius={int(args.final_blend_feather_radius)}, "
            f"final_blend_strength={float(args.final_blend_feather_strength):.3f}, "
            f"global_mask_dilate_px={max(0, int(args.global_mask_dilate_px))}, "
            f"neg_aam_uncond={bool(args.neg_aam_uncond)}, "
            f"neg_aam_cfg_base={str(args.neg_aam_cfg_base)}, "
            f"forward_strength={forward_strength_log})"
        )

    vae = AutoencoderKL.from_pretrained(str(args.base_model), subfolder="vae", torch_dtype=dtype).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(str(args.base_model), subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(str(args.base_model), subfolder="text_encoder", torch_dtype=dtype).to(device)
    scheduler_global = DDIMScheduler.from_pretrained(str(args.base_model), subfolder="scheduler")
    scheduler_local = DDIMScheduler.from_pretrained(str(args.base_model), subfolder="scheduler")

    if is_main_process:
        print(f"[Info] loading Stage 1 global U-Net checkpoint: {args.global_ckpt}")
    global_unet = _load_global_unet_from_ckpt(
        str(args.global_ckpt),
        instdiff_model=str(args.instdiff_model),
        torch_dtype=dtype,
    ).to(device)

    if is_main_process:
        print("[Info] building main UNet (no ScaleU) and loading stage2 adapters")
    main_unet, patch_unifusion = build_main_unet_no_scaleu(
        base_model=str(args.base_model),
        instdiff_model=str(args.instdiff_model),
        torch_dtype=dtype,
    )
    main_unet = main_unet.to(device)
    patch_unifusion = patch_unifusion.to(device)
    _load_main_adapters_from_ckpt(main_unet, patch_unifusion, str(args.main_ckpt))

    global_fuser_enabled = not bool(disable_global_layout_adapter)
    global_fuser_scale = 1.0 if global_fuser_enabled else 0.0
    n_global = _set_instdiff_fuser_state(
        global_unet,
        enabled=bool(global_fuser_enabled),
        scale=float(global_fuser_scale),
    )
    main_fuser_enabled = not bool(disable_global_feature_injection)
    main_fuser_scale = 1.0 if main_fuser_enabled else 0.0
    n_main = _set_instdiff_fuser_state(main_unet, enabled=bool(main_fuser_enabled), scale=float(main_fuser_scale))
    if is_main_process:
        print(
            "[Info] global U-Net fuser (layout adapter) state="
            f"enabled={bool(global_fuser_enabled)}, scale={float(global_fuser_scale):.1f}"
        )
        print(
            "[Info] local main-fuser state="
            f"enabled={bool(main_fuser_enabled)}, scale={float(main_fuser_scale):.1f}"
        )
    if n_global == 0 or n_main == 0:
        print(
            "[Warn] InstanceDiffusion fuser modules not found "
            f"(global={n_global}, main={n_main}); outputs will not match BlueOut."
        )

    _freeze_(vae)
    _freeze_(text_encoder)
    _freeze_(global_unet)
    _freeze_(main_unet)
    _freeze_(patch_unifusion)

    # Create pipeline from loaded components (shares the same model objects; no extra memory).
    # pipe.unet IS global_unet, so the writer attached below works through the pipeline.
    global_pipe = StableDiffusionINSTDIFFInpaintPipelineBBoxOnly(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=global_unet,
        scheduler=scheduler_global,
        safety_checker=None,
        feature_extractor=None,
    )
    # Pipeline is already on the correct device via shared components.

    global_writer = DiffusersGlobalUNetFuserWriter(
        global_unet,
        detach_global_tokens=True,
        main_query_only=bool(main_query_only),
    )
    main_fuser_injector = DiffusersUNetFuserBankInjector(
        main_unet,
        global_token_scale=float(args.global_token_scale),
        main_query_only=bool(main_query_only),
    )
    if global_writer.num_blocks != main_fuser_injector.num_blocks:
        raise RuntimeError(
            f"Writer/Injector block mismatch: writer={global_writer.num_blocks} injector={main_fuser_injector.num_blocks}"
        )

    data_loader, dataset = create_progressive_condition_dataloader(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        shuffle=False,
        images_root=str(args.images_root),
        captions_file=str(args.captions_file),
        r=float(args.r),
        target_long_edge=int(args.target_long_edge),
    )

    total_len = len(dataset)
    if args.num_samples is not None and int(args.num_samples) > 0:
        n_to_run = min(int(args.num_samples), total_len)
        if is_main_process:
            print(f"[Info] processing first {n_to_run} / {total_len} samples")
    else:
        n_to_run = total_len
        if is_main_process:
            print(f"[Info] processing all samples: {total_len}")

    order = parse_order(args.order)
    all_times: List[float] = []
    skipped_existing = 0

    try:
        for iteration, batch in enumerate(data_loader):
            if iteration >= n_to_run:
                break

            t0 = time.time()

            orig_pil: Image.Image = batch["orig"]
            crop_pil: Image.Image = batch["crop"]
            prompt: str = str(batch["txt"])
            name: str = str(batch["data_keys"])
            output_stem = _sanitize_path_stem(name)
            out_path = outputs_dir / f"{output_stem}.{args.save_ext}"
            center_bbox = batch.get("center_bbox", None)
            center_bbox_tuple: Optional[Tuple[int, int, int, int]] = None
            if isinstance(center_bbox, (tuple, list)) and len(center_bbox) == 4:
                center_bbox_tuple = tuple(int(x) for x in center_bbox)

            should_skip = bool(resume_enabled) and bool(args.resume_skip_existing)
            if should_skip and dist_enabled:
                shared_skip: List[bool] = [bool(out_path.exists()) if is_main_process else False]
                dist.broadcast_object_list(shared_skip, src=0)
                should_skip = bool(shared_skip[0])
            elif should_skip and not dist_enabled:
                should_skip = bool(out_path.exists())

            if should_skip:
                if is_main_process:
                    skipped_existing += 1
                    tqdm.write(
                        f"[SKIP] {iteration+1}/{n_to_run} {output_stem}: output already exists "
                        f"({out_path.name})"
                    )
                continue

            orig_np = np.asarray(orig_pil.convert("RGB"), dtype=np.uint8)
            center_crop_np = np.asarray(crop_pil.convert("RGB"), dtype=np.uint8)
            patch = int(args.patch_size)
            orig_w, orig_h = orig_pil.size
            center_w = int(center_crop_np.shape[1])
            center_h = int(center_crop_np.shape[0])

            ovx_target = (3 * patch - orig_w) / 2.0
            ovy_target = (3 * patch - orig_h) / 2.0

            fallback_ratio = 0.875
            if orig_w < patch:
                ovx_target = float(patch) * float(fallback_ratio)
            if orig_h < patch:
                ovy_target = float(patch) * float(fallback_ratio)

            ovx_px = _snap_overlap_to_grid(
                target_px=float(ovx_target),
                patch_size=patch,
                center_size=center_w,
                grid=8,
                axis="x",
            )
            ovy_px = _snap_overlap_to_grid(
                target_px=float(ovy_target),
                patch_size=patch,
                center_size=center_h,
                grid=8,
                axis="y",
            )

            overlap_ratio_x = ovx_px / float(patch)
            overlap_ratio_y = ovy_px / float(patch)

            base_seed = int(args.seed) + int(iteration)

            if not (isinstance(center_bbox, (tuple, list)) and len(center_bbox) == 4):
                raise RuntimeError("Batch missing `center_bbox` required for layout alignment.")

            left, top, _cw, _ch = [int(x) for x in center_bbox]
            texts, boxes_norm, embeds = _load_iconart_layout_json(Path(args.annotations_root), str(name))
            layout = GlobalLayoutCondition(
                texts=texts,
                boxes_norm=boxes_norm,
                positive_embeddings=embeds,
                orig_w=int(orig_w),
                orig_h=int(orig_h),
                crop_left=int(left),
                crop_top=int(top),
            )

            sample_debug_dir: Optional[Path] = None
            if debug_io_root is not None:
                sample_debug_dir = debug_io_root / f"{iteration + 1:05d}_{_sanitize_path_stem(name)}"
                sample_debug_dir.mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                canvas, init_canvas, global_u8, filled_by = run_full_model_parallel(
                    pipe=global_pipe,
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    scheduler_local=scheduler_local,
                    main_unet=main_unet,
                    patch_unifusion=patch_unifusion,
                    global_writer=global_writer,
                    main_fuser_injector=main_fuser_injector,
                    center_img=center_crop_np,
                    sample_id=output_stem,
                    prompt=prompt,
                    local_prompt=str(args.local_prompt),
                    negative_prompt=str(args.negative_prompt),
                    base_seed=base_seed,
                    patch_size=patch,
                    overlap_ratio_x=overlap_ratio_x,
                    overlap_ratio_y=overlap_ratio_y,
                    order=order,
                    steps=int(args.steps),
                    guidance_scale=float(args.guidance_scale),
                    eta=float(args.eta),
                    layout=layout,
                    layout_max_instances=int(args.layout_max_instances),
                    aam_iters=int(args.aam_iters),
                    aam_lr=float(args.aam_lr),
                    aam_denoising_steps=int(args.aam_denoising_steps),
                    aam_stop_loss=float(args.aam_stop_loss),
                    aam_res=int(args.aam_res),
                    aam_smooth=bool(args.aam_smooth),
                    aam_tau_out=float(args.aam_tau_out),
                    aam_lambda_tau=float(args.aam_lambda_tau),
                    neg_aam_uncond=bool(args.neg_aam_uncond),
                    neg_aam_iters=int(args.neg_aam_iters),
                    neg_aam_lr=float(args.neg_aam_lr),
                    neg_aam_denoising_steps=int(args.neg_aam_denoising_steps),
                    neg_aam_stop_loss=float(args.neg_aam_stop_loss),
                    neg_aam_tau_src=float(args.neg_aam_tau_src),
                    neg_aam_lambda_tau=float(args.neg_aam_lambda_tau),
                    neg_aam_cfg_base=str(args.neg_aam_cfg_base),
                    global_mask_dilate_px=max(0, int(args.global_mask_dilate_px)),
                    use_bank_features=bool(use_bank_features),
                    use_forward_diffusion=bool(use_forward_diffusion),
                    progressive_generation=bool(progressive_generation),
                    global_cfg_enabled=bool(global_cfg_enabled),
                    bank_injection_mode=str(bank_injection_mode),
                    forward_strength=float(args.forward_strength),
                    local_multidiffusion=bool(local_multidiffusion),
                    local_md_blend_mode=str(local_md_blend_mode),
                    disable_patch_token=bool(args.disable_patch_token),
                    final_composite_mode=str(final_composite_mode),
                    final_blend_feather_radius=int(args.final_blend_feather_radius),
                    final_blend_feather_strength=float(args.final_blend_feather_strength),
                    sample_tqdm_desc=(f"Sampling sample {iteration + 1}/{n_to_run}" if is_main_process else None),
                    orig_img=orig_np,
                    center_bbox=center_bbox_tuple,
                    debug_sample_dir=sample_debug_dir,
                    disable_global_layout_adapter=bool(disable_global_layout_adapter),
                    empty_attn_monitor=empty_attn_monitor,
                )
            if is_main_process:
                canvas_to_save = canvas
                if bool(args.save_center_crop_to_orig):
                    canvas_to_save = _center_crop_or_pad_u8(canvas, target_w=int(orig_w), target_h=int(orig_h))
                canvas_pil = Image.fromarray(np.ascontiguousarray(canvas_to_save), mode="RGB")
                ext = str(args.save_ext).lower()
                if ext in {"jpg", "jpeg"}:
                    canvas_pil.convert("RGB").save(out_path, format="JPEG")
                else:
                    canvas_pil.save(out_path)

                if bool(args.save_comparison):
                    try:
                        init_pil = Image.fromarray(np.ascontiguousarray(init_canvas), mode="RGB")
                        side = stack_side_by_side_centered(init_pil, canvas_pil, bg=(24, 24, 24), gap=24)
                        side.save(comparisons_dir / f"{output_stem}_comparison.png")
                    except Exception:
                        pass

                if bool(args.save_global):
                    try:
                        Image.fromarray(np.ascontiguousarray(global_u8), mode="RGB").save(
                            conditioning_dir / f"{output_stem}_global_stage1.png"
                        )
                    except Exception:
                        pass

                dt = time.time() - t0
                all_times.append(dt)
                tqdm.write(f"[OK] {iteration+1}/{n_to_run} {output_stem}: saved {out_path.name} ({dt:.2f}s)")
    finally:
        try:
            global_writer.remove()
        except Exception:
            pass
        try:
            main_fuser_injector.remove()
        except Exception:
            pass
        if is_main_process and empty_attn_monitor is not None:
            try:
                empty_attn_monitor.write_json(session_dir / "aam_empty_attn_summary.json")
            except Exception as exc:
                print(f"[Warn] failed to write AAM empty-attn summary: {exc}")

    if is_main_process:
        processed_count = int(len(all_times))
        total_time = float(sum(all_times))
        empty_payload = empty_attn_monitor.build_payload() if empty_attn_monitor is not None else None
        print("\n[Final Results]")
        print(f"Total target samples: {n_to_run}")
        print(f"Processed (new): {processed_count}")
        if bool(resume_enabled) and bool(args.resume_skip_existing):
            print(f"Skipped (existing): {int(skipped_existing)}")
        if processed_count > 0:
            print(f"Total time: {total_time:.2f}s, avg: {total_time/processed_count:.2f}s/sample")
        if empty_payload is not None:
            print(
                "AAM empty-attn: "
                f"{int(empty_payload['event_count'])} events "
                f"across {int(empty_payload['sample_count'])} samples"
            )
            print(f"AAM empty-attn summary: {session_dir / 'aam_empty_attn_summary.json'}")
        print(f"Session dir: {session_dir}")

    if dist_enabled:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
