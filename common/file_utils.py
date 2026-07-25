from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Iterable

def create_output_structure(
    base_dir: str,
    type: str = "progressive_overlap",
    create: Iterable[str] = ("inputs", "outputs", "comparisons", "conditioning", "masks"),
) -> Tuple[Path, Path, Path, Path, Path, Path]:
    """
    Create output structure.

    base_dir/
        YYYY-MM-DD_HH-MM-SS_<type>/
            inputs/         (optional)
            outputs/        (optional)
            comparisons/    (optional)
            conditioning/   (optional)
            masks/          (optional)
    
    return: session_dir, inputs_dir, outputs_dir, comparisons_dir, conditioning_dir, masks_dir
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = Path(base_dir) / f"{timestamp}_{type}"

    inputs_dir = session_dir / "inputs"
    outputs_dir = session_dir / "outputs"
    comparisons_dir = session_dir / "comparisons"
    conditioning_dir = session_dir / "conditioning"
    masks_dir = session_dir / "masks"

    create_set = set(create or ())
    to_create = []
    if "inputs" in create_set:
        to_create.append(inputs_dir)
    if "outputs" in create_set:
        to_create.append(outputs_dir)
    if "comparisons" in create_set:
        to_create.append(comparisons_dir)
    if "conditioning" in create_set:
        to_create.append(conditioning_dir)
    if "masks" in create_set:
        to_create.append(masks_dir)

    for dir_path in to_create:
        dir_path.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Session dir: {session_dir}")
    if "inputs" in create_set:
        print(f"[Info] - Inputs: {inputs_dir}")
    if "outputs" in create_set:
        print(f"[Info] - Outputs: {outputs_dir}")
    if "comparisons" in create_set:
        print(f"[Info] - Comparisons: {comparisons_dir}")
    if "conditioning" in create_set:
        print(f"[Info] - Conditioning: {conditioning_dir}")

    return session_dir, inputs_dir, outputs_dir, comparisons_dir, conditioning_dir, masks_dir

def stack_side_by_side_centered(left: Image.Image, right: Image.Image, bg=(24,24,24), gap=24) -> Image.Image:
    """
    Stack two images side by side.
    """
    H = max(left.height, right.height)
    W = left.width + gap + right.width
    out = Image.new('RGB', (W, H), bg)
    out.paste(left.convert('RGB'), (0, (H - left.height) // 2))
    out.paste(right.convert('RGB'), (left.width + gap, (H - right.height) // 2))

    return out

def parse_order(s: str) -> List[str]:
    if not s:
        return ["N", "E", "S", "W", "NE", "NW", "SE", "SW"]
    parts = [p.strip().upper() for p in s.split(',') if p.strip()]

    return parts
