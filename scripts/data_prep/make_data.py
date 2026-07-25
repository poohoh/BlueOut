import os, argparse, json, torch, torchvision
import sys
import subprocess
import tempfile
import math
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

_this_dir = os.path.abspath(os.path.dirname(__file__))
_groundingdino_dir = os.path.join(_this_dir, "GroundingDINO")
# Ensure we import the vendored GroundingDINO copy bundled with this tool,
# even if another version is installed in the environment.
if os.path.isdir(_groundingdino_dir) and _groundingdino_dir not in sys.path:
    sys.path.insert(0, _groundingdino_dir)

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# from segment_anything import build_sam, build_sam_hq, SamPredictor

try:
    from ram.models import ram
    from ram import inference_ram
except ModuleNotFoundError:
    # Allow using the vendored submodule at Tools/Grounded-Segment-Anything/recognize-anything
    _this_dir = os.path.abspath(os.path.dirname(__file__))
    _ra_dir = os.path.join(_this_dir, 'recognize-anything')
    if os.path.isdir(_ra_dir) and _ra_dir not in sys.path:
        sys.path.insert(0, _ra_dir)
        from ram.models import ram
        from ram import inference_ram
    else:
        raise
import torchvision.transforms as TS

import pycocotools.mask as mask_util
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def atomic_save_json(file_path: str, obj: Any) -> None:
    """Atomically save JSON object to file using temporary file."""
    tmp_path = file_path + '.tmp'
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, file_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def atomic_save_image(file_path: str, image: Image.Image) -> None:
    """Atomically save PIL Image to file using temporary file."""
    tmp_path = file_path + '.tmp'
    try:
        # Extract format from original file extension
        _, ext = os.path.splitext(file_path)
        if ext.lower() == '.png':
            image.save(tmp_path, format='PNG')
        elif ext.lower() in ['.jpg', '.jpeg']:
            image.save(tmp_path, format='JPEG')
        else:
            image.save(tmp_path, format='PNG')  # Default to PNG
        os.replace(tmp_path, file_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def require_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required but not available.")
    return "cuda"

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    model = model.to(device)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

DEFAULT_URLS: Dict[str, str] = {
    # GroundingDINO config and weights (official)
    "config": "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "grounded": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    # SAM ViT-H (official Meta checkpoint)
    "sam": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    # SAM-HQ ViT-H (official authors' HF hub)
    "sam_hq": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth",
    # RAM (official authors' HF hub). Can override via RAM_WEIGHTS_URL.
    "ram": os.environ.get(
        "RAM_WEIGHTS_URL",
        "https://huggingface.co/xinyu1205/recognize_anything_model/resolve/main/ram_swin_large_14m.pth",
    ),
}

def _maybe_download(url: str, dst_path: str, desc: str) -> None:
    if not url:
        raise FileNotFoundError(
            f"Missing {desc} at {dst_path} and no download URL provided. "
            f"Set an env var (e.g., RAM_WEIGHTS_URL) or place the file manually."
        )
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # Use tqdm progress bar with urllib reporthook
    tqdm.write(f"Downloading {desc} from {url} -> {dst_path}")
    
    def _hook(count: int, block_size: int, total_size: int):
        if pbar.total != total_size and total_size is not None:
            pbar.total = total_size
        pbar.update(count * block_size - pbar.n)

    with tqdm(desc=f"Downloading {desc}", unit='B', unit_scale=True, unit_divisor=1024) as pbar:
        urllib.request.urlretrieve(url, dst_path, reporthook=_hook)
    
    tqdm.write(f"✓ Downloaded {desc} -> {dst_path}")

# NOTE: We intentionally avoid repo-relative resolution here to ensure
# deterministic behavior regardless of current working directory.

def ensure_checkpoints(paths: Dict[str, str]) -> Dict[str, str]:
    # Ensure config
    if not os.path.exists(paths["config"]):
        _maybe_download(DEFAULT_URLS["config"], paths["config"], "GroundingDINO config")
    # Ensure GroundingDINO weights
    if not os.path.exists(paths["grounded"]):
        _maybe_download(DEFAULT_URLS["grounded"], paths["grounded"], "GroundingDINO weights")
    # Ensure SAM weights (if using SAM-HQ and sam_hq path is set, leave to caller)
    if not os.path.exists(paths["sam"]):
        _maybe_download(DEFAULT_URLS["sam"], paths["sam"], "SAM ViT-H weights")
    # Ensure SAM-HQ weights (optional, only if target path provided)
    if paths.get("sam_hq") and not os.path.exists(paths["sam_hq"]):
        _maybe_download(DEFAULT_URLS["sam_hq"], paths["sam_hq"], "SAM-HQ ViT-H weights")
    # Ensure RAM weights
    if not os.path.exists(paths["ram"]):
        ram_url = paths.get("ram_url") or DEFAULT_URLS["ram"]
        _maybe_download(ram_url, paths["ram"], "RAM weights")
    return paths

def apply_img_transform(image_pil):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image


def _nms_pure_torch(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """CPU-only NMS fallback for environments without torchvision C++ ops."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    x1, y1, x2, y2 = boxes.unbind(dim=1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break

        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])

        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        union = areas[i] + areas[rest] - inter
        iou = inter / union.clamp(min=1e-6)

        rest = rest[iou <= iou_threshold]
        order = rest

    if not keep:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    return torch.stack(keep)


def nms_with_safe_fallback(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    try:
        return torchvision.ops.nms(boxes, scores, iou_threshold)
    except RuntimeError as exc:
        if "Couldn't load custom C++ ops" not in str(exc):
            raise
        return _nms_pure_torch(boxes, scores, iou_threshold)


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    # get phrase
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases

def random_color():
    return np.random.randint(0, 256, size=3)

def save_colormask(rles, savepath):
    
    H, W = rles[0]['size']
    
    color_map = np.zeros((H, W, 3), dtype=np.uint8)
    
    for rle_obj in rles:
        obj_mask = mask_util.decode(rle_obj)
        color = random_color()
        color_map[obj_mask == 1] = color 
        
    im = Image.fromarray(color_map)
    atomic_save_image(savepath, im)


def iou_xyxy(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    xa1, ya1, xa2, ya2 = box_a.tolist()
    xb1, yb1, xb2, yb2 = box_b.tolist()

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


TaskKey = Tuple[str, Optional[str]]
TaskSpec = Dict[str, Any]


def collect_dataset_tasks(
    images_root: str,
    image_extensions: Sequence[str],
    only_main_set: Optional[Set[str]],
) -> List[Tuple[str, Optional[str], List[str]]]:
    tasks: List[Tuple[str, Optional[str], List[str]]] = []

    main_categories = sorted(os.listdir(images_root))
    for main_category in main_categories:
        if only_main_set and main_category not in only_main_set:
            continue

        main_category_path = os.path.join(images_root, main_category)
        if not os.path.isdir(main_category_path):
            continue

        entries = sorted(os.listdir(main_category_path))
        sub_category_dirs = [
            entry for entry in entries
            if os.path.isdir(os.path.join(main_category_path, entry))
        ]
        root_level_images = [
            entry for entry in entries
            if os.path.isfile(os.path.join(main_category_path, entry))
            and entry.lower().endswith(image_extensions)
        ]

        for sub_category in sub_category_dirs:
            sub_dir_path = os.path.join(main_category_path, sub_category)
            try:
                sub_files = [
                    f for f in sorted(os.listdir(sub_dir_path))
                    if os.path.isfile(os.path.join(sub_dir_path, f))
                    and f.lower().endswith(image_extensions)
                ]
            except FileNotFoundError:
                sub_files = []
            if sub_files:
                tasks.append((main_category, sub_category, sub_files))

        if root_level_images:
            tasks.append((main_category, None, sorted(root_level_images)))

    return tasks


def print_summary(images_root: str, annotations_root: str, seg_root: str, bbox_root: str) -> None:
    total_categories = len(
        [d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))]
    )
    total_subcategories = 0
    total_images_processed = 0
    total_annotations = 0

    for main_cat in os.listdir(annotations_root):
        main_ann_path = os.path.join(annotations_root, main_cat)
        if not os.path.isdir(main_ann_path):
            continue

        root_ann_files = [f for f in os.listdir(main_ann_path) if f.endswith('.json')]
        if root_ann_files:
            total_subcategories += 1
            total_images_processed += len(root_ann_files)
            for ann_file in root_ann_files:
                ann_path = os.path.join(main_ann_path, ann_file)
                with open(ann_path, 'r') as f:
                    ann_data = json.load(f)
                    total_annotations += len(ann_data.get('objects', []))

        for sub_cat in os.listdir(main_ann_path):
            sub_ann_path = os.path.join(main_ann_path, sub_cat)
            if os.path.isdir(sub_ann_path):
                total_subcategories += 1
                ann_files = [f for f in os.listdir(sub_ann_path) if f.endswith('.json')]
                total_images_processed += len(ann_files)
                for ann_file in ann_files:
                    ann_path = os.path.join(sub_ann_path, ann_file)
                    with open(ann_path, 'r') as f:
                        ann_data = json.load(f)
                        total_annotations += len(ann_data.get('objects', []))

    print(f"\n🎉 Processing Complete!")
    print(f"  📁 Main Categories: {total_categories}")
    print(f"  📂 Sub Categories: {total_subcategories}")
    print(f"  🖼️  Images Processed: {total_images_processed}")
    print(f"  📎 Total Objects Detected: {total_annotations}")
    print(f"  📄 Annotations saved to: {annotations_root}")
    # print(f"  🗺️  Segmentations saved to: {seg_root}")  # Disabled - no segmentation
    print(f"  🟥 BBox visualizations saved to: {bbox_root}")
    if total_images_processed > 0:
        avg_objects = total_annotations / total_images_processed
        print(f"  📈 Average objects per image: {avg_objects:.1f}")

def main(
    config,
    task_list: Optional[List[TaskSpec]] = None,
    skip_summary: bool = False,
):

    # Set up directory paths for mirrored structure (absolute, fixed)
    datasets_root = config.get('datasets_root', 'datasets')
    images_root = os.path.join(datasets_root, 'images')
    caption_root = os.path.join(datasets_root, 'caption')
    annotations_root = os.path.join(datasets_root, 'annotations')
    seg_root = os.path.join(datasets_root, 'seg')
    bbox_root = os.path.join(datasets_root, config.get('bbox_dir_name', 'bbox_vis'))

    # Create base directories
    os.makedirs(annotations_root, exist_ok=True)
    # os.makedirs(seg_root, exist_ok=True)  # Disabled - no segmentation
    os.makedirs(bbox_root, exist_ok=True)
    print(f"Created base directories: {annotations_root}, {bbox_root} (seg disabled)")
    
    # Enforce CUDA usage and ensure checkpoints exist (download if missing)
    device = require_cuda()
    ensure_checkpoints(config['path'])

    ### load model ###
    model = load_model(config['path']['config'], config['path']['grounded'], device=device)
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = TS.Compose([TS.Resize((384, 384)), TS.ToTensor(), normalize])
    print("GroundingDINO setup done")
    
    ram_model = ram(pretrained=config['path']['ram'], image_size=384, vit='swin_l')
    ram_model.eval()
    ram_model = ram_model.to(device)
    print("Recognize Anything setup done")
    
    
    ### sam initialize ###
    # if config['use_sam_hq']:
    #     predictor = SamPredictor(build_sam_hq(checkpoint=config['path']['sam_hq']).to(device))
    # else:
    #     predictor = SamPredictor(build_sam(checkpoint=config['path']['sam']).to(device))
    # print("Sam setup done")
    
    ### load BLIP and CLIP model (unused) ###
    # Intentionally not used; kept as reference if needed later.
    

    image_extensions = ('.png', '.jpg', '.jpeg')

    # Restrict to specific main categories if provided
    only_main: List[str] = config.get('only_main') or []
    only_main_set = set(m.strip() for m in only_main)

    if task_list is None:
        task_list = config.get('task_list') or []

    task_assignments: Dict[TaskKey, List[Optional[List[str]]]] = {}
    if task_list:
        for task in task_list:
            key = (task['main'], task.get('sub'))
            files = task.get('files')
            if isinstance(files, list):
                task_assignments.setdefault(key, []).append(files)
            else:
                task_assignments.setdefault(key, []).append(None)

    task_whitelist: Optional[Set[TaskKey]] = set(task_assignments.keys()) if task_assignments else None
    allowed_main: Optional[Set[str]] = {main for main, _ in task_whitelist} if task_whitelist else None

    # Process datasets with 2-level structure: main_category/sub_category
    main_categories = sorted(os.listdir(images_root))
    if only_main_set:
        main_categories = [m for m in main_categories if m in only_main_set]
        print(f"Restricting to main categories: {sorted(list(only_main_set))}")
    if allowed_main is not None:
        main_categories = [m for m in main_categories if m in allowed_main]

    for main_category in tqdm(main_categories, desc="📁 Processing main categories", position=0):
        main_category_path = os.path.join(images_root, main_category)
        if not os.path.isdir(main_category_path):
            continue

        # Create mirror directories for main_category
        os.makedirs(os.path.join(annotations_root, main_category), exist_ok=True)
        # os.makedirs(os.path.join(seg_root, main_category), exist_ok=True)  # Disabled - no segmentation

        entries = sorted(os.listdir(main_category_path))
        sub_category_dirs = [e for e in entries if os.path.isdir(os.path.join(main_category_path, e))]
        root_level_images = [
            e for e in entries
            if os.path.isfile(os.path.join(main_category_path, e)) and
               e.lower().endswith(image_extensions)
        ]
        targets = [(sc, os.path.join(main_category_path, sc), None) for sc in sub_category_dirs]
        if root_level_images:
            targets.append((None, main_category_path, root_level_images))

        for sub_category, sub_category_path, preset_image_files in tqdm(
            targets, desc=f"📂 Processing {main_category}", position=1, leave=False
        ):
            task_key: TaskKey = (main_category, sub_category)
            assigned_overrides = task_assignments.get(task_key) if task_assignments else None
            if task_whitelist is not None and not assigned_overrides:
                continue

            overrides = assigned_overrides or [None]
            if sub_category is None:            # IconArt
                ann_dir = os.path.join(annotations_root, main_category)
                # seg_dir = os.path.join(seg_root, main_category)  # Disabled - no segmentation
                bbox_dir = os.path.join(bbox_root, main_category)
                caption_file = os.path.join(caption_root, main_category, 'blip2_prompts.json')
                category_label = main_category
            else:
                ann_dir = os.path.join(annotations_root, main_category, sub_category)
                # seg_dir = os.path.join(seg_root, main_category, sub_category)  # Disabled - no segmentation
                bbox_dir = os.path.join(bbox_root, main_category, sub_category)
                # laion-high-resolution은 특별한 파일명 구조 사용
                if main_category == 'laion-high-resolution':
                    caption_file = os.path.join(caption_root, main_category, f'{sub_category}_blip2_prompts.json')
                else:
                    caption_file = os.path.join(caption_root, main_category, sub_category, 'blip2_prompts.json')
                category_label = f"{main_category}/{sub_category}"

            os.makedirs(ann_dir, exist_ok=True)
            # os.makedirs(seg_dir, exist_ok=True)  # Disabled - no segmentation
            os.makedirs(bbox_dir, exist_ok=True)

            # Load captions for this category from JSON file (shared per override)
            captions = {}
            if os.path.exists(caption_file):
                print(f"Loading captions from {caption_file}")
                with open(caption_file, 'r', encoding='utf-8') as f:
                    first_char = f.read(1)
                    f.seek(0)
                    if first_char == '[':
                        try:
                            arr = json.load(f)
                        except json.JSONDecodeError as e:
                            raise ValueError(f"Invalid JSON array in {caption_file}: {e}")
                        for item in arr:
                            if isinstance(item, dict) and 'file' in item and 'prompt' in item:
                                captions[item['file']] = item['prompt']
                    else:
                        for line in f:
                            if line.strip():
                                data = json.loads(line.strip())
                                captions[data['file']] = data['prompt']
                print(f"Loaded {len(captions)} captions for {category_label}")
            else:
                tqdm.write(f"[SKIP] Missing caption file: {caption_file}")
                continue

            total_processed = 0
            total_skipped = 0

            for files_override in overrides:
                if files_override is not None:
                    image_files = files_override
                elif preset_image_files is not None:
                    image_files = preset_image_files
                else:
                    image_files = [
                        f for f in os.listdir(sub_category_path)
                        if f.lower().endswith(image_extensions)
                        and os.path.isfile(os.path.join(sub_category_path, f))
                    ]

                if not image_files:
                    tqdm.write(f"[SKIP] No images found in {category_label}")
                    continue

                processed_count = 0
                skipped_count = 0

                for filename in tqdm(image_files,
                                   desc=f"🖼️  {category_label}",
                                   position=2,
                                   leave=False,
                                   unit="img"):

                    image_name = os.path.splitext(filename)[0]
                    image_path = os.path.join(sub_category_path, filename)

                    # Check if already processed (resume functionality) - seg disabled
                    ann_file_path = os.path.join(ann_dir, f'{image_name}.json')
                    # seg_viz_path = os.path.join(seg_dir, f'{image_name}.png')  # Disabled
                    bbox_path = os.path.join(bbox_dir, f'{image_name}.png')

                    if config.get('resume', True) and all(os.path.exists(p) for p in [ann_file_path, bbox_path]):
                        continue  # Skip already processed file (seg disabled)

                    # Get caption from loaded JSON, fallback to default
                    global_caption = captions.get(filename, f"Image from {category_label}")

                    # Prepare image
                    image_pil = Image.open(image_path).convert("RGB")
                    image = apply_img_transform(image_pil)

                    # Run RAM
                    raw_image = image_pil.resize((384, 384))
                    raw_image = transform(raw_image).unsqueeze(0).to(device)
                    res = inference_ram(raw_image, ram_model)
                    tags = res[0].replace(' |', ',')

                    # Run GroundingDINO
                    boxes_filt, scores, pred_phrases = get_grounding_output(
                        model, image, tags,
                        config['threshold']['box'], config['threshold']['text'],
                        device=device
                    )

                    if len(boxes_filt) == 0:
                        tqdm.write(f"    ⚠️  No objects detected: {filename}")
                    image = np.array(image_pil)
                    # predictor.set_image(image)

                    size = image_pil.size
                    H, W = size[1], size[0]
                    pixel_boxes: List[torch.Tensor] = []
                    for i in range(boxes_filt.size(0)):
                        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                        boxes_filt[i][2:] += boxes_filt[i][:2]
                        pixel_boxes.append(boxes_filt[i].clone())
                    boxes_filt = boxes_filt.cpu()
                    pixel_boxes = [b.cpu() for b in pixel_boxes]

                    nms_idx = nms_with_safe_fallback(
                        boxes_filt, scores, config['threshold']['iou']
                    ).cpu().numpy().tolist()
                    boxes_filt = boxes_filt[nms_idx]
                    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
                    scores = scores[nms_idx]
                    if pixel_boxes:
                        pixel_boxes = [pixel_boxes[idx] for idx in nms_idx]
                    else:
                        pixel_boxes = []

                    overlap_prune = 0.45
                    if overlap_prune and overlap_prune > 0 and len(boxes_filt) > 1:
                        keep_indices: List[int] = []
                        for idx_keep, box in enumerate(boxes_filt):
                            if all(iou_xyxy(box, boxes_filt[kept_idx]) < overlap_prune for kept_idx in keep_indices):
                                keep_indices.append(idx_keep)
                        if keep_indices and len(keep_indices) < len(boxes_filt):
                            boxes_filt = boxes_filt[keep_indices]
                            pred_phrases = [pred_phrases[idx] for idx in keep_indices]
                            scores = scores[keep_indices]
                            pixel_boxes = [pixel_boxes[idx] for idx in keep_indices] if pixel_boxes else []

                    # current_box_count = len(boxes_filt)  # Disabled - not used without SAM

                    # Handle empty detections safely - SAM disabled
                    # if current_box_count == 0:
                    #     masks = torch.empty((0, H, W), dtype=torch.bool, device=device)
                    # else:
                    #     transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
                    #
                    #     try:
                    #         masks, _, _ = predictor.predict_torch(
                    #             point_coords=None,
                    #             point_labels=None,
                    #             boxes=transformed_boxes.to(device),
                    #             multimask_output=False,
                    #         )
                    #     except RuntimeError as e:
                    #         tqdm.write(f"[WARN] SAM predict failed for {category_label}/{filename}: {e}")
                    #         continue
                    
                    rel_image_path = f"{category_label}/{filename}"

                    ann = dict(
                        image_name = filename,
                        image_path = rel_image_path,
                        main_category = main_category,
                        sub_category = sub_category or "",
                        caption = global_caption,
                        objects= [],
                        boxes = [],
                        size = [H, W],
                        segmentation = []
                    )
                    # rles = []  # Disabled - no segmentation

                    value = 0
                    bbox_draw_entries = []
                    for idx, (label, box) in enumerate(zip(pred_phrases, boxes_filt)):
                        value += 1

                        try:
                            name, logit_part = label.rsplit('(', 1)
                            logit = logit_part[:-1]
                        except ValueError:
                            name = label
                            logit = "1.0"

                        try:
                            logit_value = float(logit)
                        except ValueError:
                            logit_value = 1.0

                        # mask = masks[value-1].cpu().numpy()[0] == True
                        # rle = mask_util.encode(np.array(mask[...,None], order="F", dtype="uint8"))[0]
                        # rle['counts'] = rle['counts'].decode('ascii')

                        pixel_box = pixel_boxes[idx] if idx < len(pixel_boxes) else box
                        x1, y1, x2, y2 = [int(round(x)) for x in pixel_box.tolist()]

                        box_xywh = [int(x) for x in box.numpy().tolist()]
                        box_xywh[2] = box_xywh[2] - box_xywh[0]
                        box_xywh[3] = box_xywh[3] - box_xywh[1]

                        box[0], box[2] = box[0]/W, box[2]/W
                        box[1], box[3] = box[1]/H, box[3]/H

                        ann['objects'].append(name)
                        ann['boxes'].append(box.tolist())

                        # coco_rle = {
                        #     'size': rle['size'],
                        #     'counts': rle['counts']
                        # }
                        # ann['segmentation'].append(coco_rle)
                        # rles.append(rle)

                        bbox_draw_entries.append({
                            'coords': (x1, y1, x2, y2),
                            'label': name.strip(),
                            'score': logit_value,
                        })

                    # ann_file_path already calculated above for resume check
                    atomic_save_json(ann_file_path, ann)

                    # Segmentation disabled - using bbox detection count instead
                    # if len(rles) > 0:
                    #     # seg_viz_path already calculated above for resume check
                    #     save_colormask(rles, savepath=seg_viz_path)
                    #     processed_count += 1
                    # else:
                    #     skipped_count += 1

                    if len(boxes_filt) > 0:
                        processed_count += 1
                    else:
                        skipped_count += 1

                    if bbox_draw_entries:
                        # bbox_path already calculated above for resume check
                        bbox_image = image_pil.copy()
                        draw = ImageDraw.Draw(bbox_image)
                        for entry in bbox_draw_entries:
                            x1, y1, x2, y2 = entry['coords']
                            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
                            label_text = entry['label']
                            if entry['score'] is not None:
                                label_text = f"{label_text} ({entry['score']:.2f})" if label_text else f"score {entry['score']:.2f}"
                            if label_text:
                                text_origin = (x1 + 4, y1 + 4)
                                draw.text(text_origin, label_text, fill=(255, 255, 255))
                        atomic_save_image(bbox_path, bbox_image)

                total_processed += processed_count
                total_skipped += skipped_count

            if total_processed > 0 or total_skipped > 0:
                tqdm.write(f"  ✓ {category_label}: {total_processed} processed, {total_skipped} skipped")

    if not skip_summary:
        print_summary(images_root, annotations_root, seg_root, bbox_root)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='processed_dataset', help='Name for the processing run')
    parser.add_argument('--ram_url', type=str, default='', help='URL to RAM weights (used if missing)')
    parser.add_argument('--only_main', nargs='*', default=None, help='Process only these main categories (e.g., humanart iconart wikiart)')
    parser.add_argument('--auto_mgpu', action='store_true', help='Automatically use all visible GPUs by spawning per-GPU workers')
    parser.add_argument('--no-resume', action='store_true', help='Disable resume functionality (reprocess all files)')
    parser.add_argument('--bbox_dir_name', type=str, default='bbox_vis', help='Directory name under the dataset root for bbox visualizations')
    parser.add_argument('--datasets_root', type=str, default='datasets', help='Dataset root containing images/, caption/, annotations/ outputs')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--tasks_file', type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    
    config = dict(
        path = dict(
            # All checkpoints/configs resolve under checkpoints/
            config='checkpoints/GroundingDINO/config/GroundingDINO_SwinT_OGC.py',
            ram='checkpoints/RAM/ram_swin_large_14m.pth',
            grounded='checkpoints/GroundingDINO/groundingdino_swint_ogc.pth',
            sam='checkpoints/SAM/sam_vit_h_4b8939.pth',
            sam_hq='checkpoints/SAM/sam_hq_vit_h.pth',
            ram_url=args.ram_url,
        ),
        threshold = dict(
            box = 0.25,
            text = 0.2,
            iou = 0.5,
        ),
        use_sam_hq = False,
        name = args.dataset_name,
        only_main = args.only_main,
        resume = not args.no_resume,
        bbox_dir_name = args.bbox_dir_name,
        datasets_root = args.datasets_root,
    )
    
    # Multi-GPU orchestrator (parent process)
    if (not args.worker) and args.auto_mgpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        ensure_checkpoints(config['path'])

        datasets_root = args.datasets_root
        images_root = os.path.join(datasets_root, 'images')
        annotations_root = os.path.join(datasets_root, 'annotations')
        seg_root = os.path.join(datasets_root, 'seg')
        bbox_root = os.path.join(datasets_root, args.bbox_dir_name)
        os.makedirs(annotations_root, exist_ok=True)
        # os.makedirs(seg_root, exist_ok=True)  # Disabled - no segmentation
        os.makedirs(bbox_root, exist_ok=True)

        image_extensions = ('.png', '.jpg', '.jpeg')
        only_main_list = args.only_main or []
        only_main_set = set(m.strip() for m in only_main_list) if only_main_list else None
        raw_tasks = collect_dataset_tasks(images_root, image_extensions, only_main_set)

        if not raw_tasks:
            print("No dataset tasks found; exiting.")
            print_summary(images_root, annotations_root, seg_root, bbox_root)
            sys.exit(0)

        ngpus = torch.cuda.device_count()
        if ngpus <= 0:
            print("CUDA devices not available; falling back to single-process execution.")
            main(config)
            sys.exit(0)

        expanded_tasks: List[Dict[str, Any]]
        if len(raw_tasks) >= ngpus or ngpus <= 1:
            expanded_tasks = [
                {'main': main, 'sub': sub, 'files': files}
                for main, sub, files in raw_tasks if files
            ]
        else:
            expanded_tasks = []
            base_chunks = max(1, ngpus // len(raw_tasks))
            remainder = ngpus % len(raw_tasks)
            for idx_task, (main, sub, files) in enumerate(raw_tasks):
                if not files:
                    continue
                num_chunks = base_chunks + (1 if idx_task < remainder else 0)
                num_chunks = max(1, min(len(files), num_chunks))
                chunk_size = max(1, math.ceil(len(files) / num_chunks))
                for start in range(0, len(files), chunk_size):
                    chunk_files = files[start:start + chunk_size]
                    if chunk_files:
                        expanded_tasks.append({'main': main, 'sub': sub, 'files': chunk_files})

        if not expanded_tasks:
            print("No dataset tasks found after expansion; exiting.")
            print_summary(images_root, annotations_root, seg_root, bbox_root)
            sys.exit(0)

        n_workers = min(ngpus, len(expanded_tasks))
        task_chunks: List[List[Dict[str, Any]]] = [[] for _ in range(n_workers)]
        for idx, task in enumerate(expanded_tasks):
            worker_idx = idx % n_workers
            task_chunks[worker_idx].append(task)

        visible_env = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible_env:
            device_pool = [d.strip() for d in visible_env.split(',') if d.strip()]
        else:
            device_pool = [str(i) for i in range(torch.cuda.device_count())]
        if not device_pool:
            device_pool = [str(i) for i in range(n_workers)]

        procs = []
        for idx, chunk in enumerate(task_chunks):
            if not chunk:
                continue
            tmp = tempfile.NamedTemporaryFile('w', delete=False, suffix='.json')
            json.dump(chunk, tmp)
            tmp.flush()
            tmp.close()

            env = os.environ.copy()
            device_id = device_pool[idx] if idx < len(device_pool) else str(idx)
            env['CUDA_VISIBLE_DEVICES'] = device_id
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                '--worker',
                '--dataset_name',
                args.dataset_name,
                '--tasks_file',
                tmp.name,
            ]
            if args.ram_url:
                cmd += ['--ram_url', args.ram_url]
            if only_main_list:
                cmd += ['--only_main'] + only_main_list
            if args.no_resume:
                cmd += ['--no-resume']
            if args.bbox_dir_name:
                cmd += ['--bbox_dir_name', args.bbox_dir_name]
            if args.datasets_root:
                cmd += ['--datasets_root', args.datasets_root]
            p = subprocess.Popen(cmd, env=env)
            procs.append((idx, p, tmp.name))

        ret = 0
        for idx, proc, _ in procs:
            code = proc.wait()
            if code != 0:
                ret = code
                print(f"[Worker GPU {idx}] failed with code {code}")

        for _, _, tmp_path in procs:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        print_summary(images_root, annotations_root, seg_root, bbox_root)
        sys.exit(ret)

    if args.worker:
        task_list: Optional[List[Dict[str, Optional[str]]]] = None
        if args.tasks_file:
            with open(args.tasks_file, 'r', encoding='utf-8') as f:
                task_list = json.load(f)
        if not task_list:
            print("[Worker] No tasks assigned; exiting.")
            sys.exit(0)
        config['task_list'] = task_list
        main(config, task_list=task_list, skip_summary=True)
    else:
        main(config)
