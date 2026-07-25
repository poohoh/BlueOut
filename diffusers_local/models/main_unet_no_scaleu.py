from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from diffusers import UNet2DConditionModel

from .bbox_only_projection import INSTDIFFTextBoundingboxProjectionBBoxOnly


def _extract_bbox_only_position_net_state_dict(position_net: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract bbox-only UniFusion weights from an InstanceDiffusion position_net.

    The full InstanceDiffusion projection has multiple branches in `linears_list`:
      - idx 0: bbox (xyxy) branch  ✅ we keep this
      - idx 1+: point/scribble/polygon/seg branches ❌ we discard
    """
    sd = position_net.state_dict()

    keep = {
        "null_positive_feature",
        "null_position_feature",
    }
    bbox_sd = {k: v for k, v in sd.items() if (k.startswith("linears_list.0.") or k in keep)}
    if not bbox_sd:
        raise ValueError(
            "Failed to extract bbox-only position_net weights from InstanceDiffusion position_net. "
            f"Got {len(sd)} keys but extracted 0. position_net={type(position_net)}"
        )
    return bbox_sd


def remove_scaleu_(unet: torch.nn.Module) -> List[str]:
    """
    Remove ScaleU parameters (scaleu_b_*, scaleu_s_*) from an InstanceDiffusion UNet.

    Returns the list of removed parameter names.
    """
    removed: List[str] = []

    # ScaleU is registered at the top level via `register_parameter("scaleu_b_{i}", ...)`.
    for name in list(getattr(unet, "_parameters", {}).keys()):
        if name.startswith("scaleu_b_") or name.startswith("scaleu_s_"):
            # delattr removes it from both __dict__ and _parameters
            if hasattr(unet, name):
                delattr(unet, name)
            removed.append(name)

    return removed


def build_main_unet_no_scaleu(
    *,
    base_model: str,
    instdiff_model: str,
    torch_dtype: torch.dtype,
) -> Tuple[UNet2DConditionModel, INSTDIFFTextBoundingboxProjectionBBoxOnly]:
    """
    Build a main UNet that matches the global U-Net (InstanceDiffusion UNet) structure but without ScaleU.

    - UNet backbone: InstanceDiffusion UNet (so fusers exist in every transformer block)
    - conv_in: converted to 9ch inpaint input
    - backbone weights: loaded from Stable Diffusion inpainting UNet
    - position_net: replaced by bbox-only UniFusion and initialized from InstanceDiffusion pretrained weights
    - ScaleU params: removed from the module tree
    """
    # 1) Start from InstanceDiffusion UNet (has fuser modules + UniFusion branch weights)
    unet = UNet2DConditionModel.from_pretrained(
        instdiff_model,
        subfolder="unet",
        torch_dtype=torch_dtype,
    )

    # 2) Initialize bbox-only UniFusion from InstanceDiffusion pretrained position_net
    if not hasattr(unet, "position_net") or unet.position_net is None:
        raise RuntimeError(
            "InstanceDiffusion UNet is missing position_net; cannot initialize UniFusion from pretrained weights."
        )
    patch_unifusion = INSTDIFFTextBoundingboxProjectionBBoxOnly(
        positive_len=768,
        out_dim=768,
    ).to(dtype=torch_dtype)
    patch_unifusion.load_state_dict(_extract_bbox_only_position_net_state_dict(unet.position_net), strict=True)

    # Replace position_net with bbox-only module (no seg/points branches)
    unet.position_net = patch_unifusion

    # 3) Convert conv_in to 9ch (inpaint-style)
    old_conv_in = unet.conv_in
    unet.conv_in = torch.nn.Conv2d(
        9,
        old_conv_in.out_channels,
        kernel_size=old_conv_in.kernel_size,
        stride=old_conv_in.stride,
        padding=old_conv_in.padding,
    ).to(dtype=torch_dtype)
    unet.config["in_channels"] = 9

    # 4) Load backbone weights from SD inpainting UNet (keeps ID fusers + UniFusion intact)
    base_unet = UNet2DConditionModel.from_pretrained(
        base_model,
        subfolder="unet",
        torch_dtype=torch_dtype,
    )
    base_sd = base_unet.state_dict()

    allowed = set(unet.state_dict().keys())
    filtered = {k: v for k, v in base_sd.items() if k in allowed}
    unet.load_state_dict(filtered, strict=False)
    del base_unet

    # 5) Remove ScaleU (main UNet should not use it)
    remove_scaleu_(unet)

    return unet, patch_unifusion
