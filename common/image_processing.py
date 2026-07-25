import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from typing import Tuple

def ensure_uint8(arr: np.ndarray) -> np.ndarray:
    """ Convert numpy array to uint8 format. """
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def pil_to_tensor_normalized(image: Image.Image) -> torch.Tensor:
    """ Convert PIL Image to normalized torch tensor [-1,1] """
    import torchvision.transforms as T

    # process based on image mode
    if image.mode == 'L':           # grayscale
        mean, std = [0.5], [0.5]
    else:                           # RGB
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std)  # [0,1] -> [-1,1]
    ])

    return transform(image)

def torch_letterbox(tensor: torch.Tensor, target_size: int = 512, mode: str = 'bilinear') -> Tuple[torch.Tensor, float, Tuple[int, int]]:
    """
    Torch letterbox

    Args:
        tensor: mask [H,W], CHW image, or HWC image
        target_size: target size
        mode: 'bilinear' for RGB, 'nearest' for mask
    
    Returns:
        letterboxed, scale, pad
    """
    if tensor.ndim == 2:
        H, W = tensor.shape
        expanded_t = tensor.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        input_format = '2D'
    elif tensor.ndim == 3 and tensor.shape[0] <= 4:  # RGB, CHW
        _, H, W = tensor.shape
        expanded_t = tensor.unsqueeze(0)  # [1,C,H,W]
        input_format = 'CHW'
    elif tensor.ndim == 3:  # HWC
        H, W, _ = tensor.shape
        expanded_t = tensor.permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
        input_format = 'HWC'
    else:
        raise ValueError(f'Unsupported tensor shape: {tensor.shape}. Expected [H,W], [H,W,C], or [C,H,W]')

    # calculate scale and new size
    scale = min(target_size / W, target_size / H)
    new_w = int(W * scale)
    new_h = int(H * scale)

    # resize maintaining aspect ratio
    resized = torch.nn.functional.interpolate(
        expanded_t, size=(new_h, new_w), mode=mode, align_corners=False if mode == 'bilinear' else None
    )

    # calculate padding
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    pad_right = target_size - new_w - pad_left
    pad_bottom = target_size - new_h - pad_top

    # apply padding
    padded = torch.nn.functional.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)

    # return same format as input
    if input_format == '2D':
        return padded.squeeze(0).squeeze(0), scale, (pad_left, pad_top)
    elif input_format == 'CHW':
        return padded.squeeze(0), scale, (pad_left, pad_top)
    else:  # HWC
        return padded.squeeze(0).permute(1, 2, 0), scale, (pad_left, pad_top)

def torch_unletter(
    letterboxed_tensor: torch.Tensor,
    scale: float,
    pad: Tuple[int, int],
    orig_size: Tuple[int, int],
    mode: str = 'bilinear'
) -> torch.Tensor:
    """
    Reverse torch_letterbox. Remove padding and scale back.

    'bilinear' for RGB, 'nearest' for masks
    """
    pad_left, pad_top = pad
    orig_h, orig_w = orig_size

    # calculate scaled but not padded
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # detect input format
    if letterboxed_tensor.ndim == 2:
        tensor_4d = letterboxed_tensor.unsqueeze(0).unsqueeze(0)
        input_format = '2D'
    elif letterboxed_tensor.ndim == 3 and letterboxed_tensor.shape[0] <= 4:  # CHW
        tensor_4d = letterboxed_tensor.unsqueeze(0)  # (1, C, H, W)
        input_format = 'CHW'
    elif letterboxed_tensor.ndim == 3:  # HWC
        tensor_4d = letterboxed_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        input_format = 'HWC'
    else:
        raise ValueError(f"Unsupported tensor shape: {letterboxed_tensor.shape}")
    
    # remove padding
    if new_h <= 0 or new_w <= 0:
        raise ValueError(f"Invalid scaled size: {new_h}x{new_w}. Check scale={scale} and orig_size={orig_size}")
    
    cropped = tensor_4d[:, :, pad_top:pad_top + new_h, pad_left:pad_left + new_w]

    # scale back
    if new_h == orig_h and new_w == orig_w:
        restored = cropped
    else:
        restored = torch.nn.functional.interpolate(
            cropped, size=(orig_h, orig_w), mode=mode, align_corners=False if mode=='bilinear' else None
        )
    
    # return in same format
    if input_format == '2D':
        return restored.squeeze(0).squeeze(0)  # (H, W)
    elif input_format == 'CHW':
        return restored.squeeze(0)  # (C, H, W)
    else:
        return restored.squeeze(0).permute(1, 2, 0)  # (H, W, C)

def to_image_uint8(x: torch.Tensor) -> np.ndarray:
    """ Convert torch tensor to uint8 numpy array """
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = (x.clamp(-1, 1) + 1.0) / 2.0   # [-1, 1] -> [0, 1]
    x = (x * 255.0).round().clamp(0, 255).to(torch.uint8)
    x = x.permute(0, 2, 3, 1).cpu().numpy()   # BHWC

    return x