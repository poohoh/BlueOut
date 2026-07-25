"""
Convert ReferenceNet (bbox-only) checkpoint to diffusers format.

This script reuses the existing convert_instdiff_to_diffusers.py logic
and adapts it for bbox-only ReferenceNet with 9-channel input.

Usage:
    python convert_refnet_bbox_only_to_diffusers.py \
        --checkpoint_path /path/to/referencenet-epochXXXX.pt \
        --output_path /path/to/output_dir
"""

import argparse
import os
import sys

import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

# Import existing conversion function
from convert_instdiff_to_diffusers import convert_instdiff_unet_checkpoint


def convert_checkpoint(
    checkpoint_path: str,
    output_path: str,
    base_model: str = "runwayml/stable-diffusion-inpainting",
    instdiff_model: str = "kyeongry/instancediffusion_sd15",
):
    """
    Convert ReferenceNet checkpoint to diffusers pipeline.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        output_path: Directory to save converted model
        base_model: Base SD Inpainting model for VAE/text_encoder/scheduler
        instdiff_model: InstanceDiffusion model for UNet structure (has fuser)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Remove 'inner.' prefix
    clean_checkpoint = {}
    for key, value in checkpoint.items():
        new_key = key[len("inner."):] if key.startswith("inner.") else key
        clean_checkpoint[new_key] = value

    # Load base components
    print(f"Loading VAE/text_encoder/scheduler from {base_model}...")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
    scheduler = DDIMScheduler.from_pretrained(base_model, subfolder="scheduler")

    # Load UNet from InstanceDiffusion (has fuser)
    print(f"Loading UNet from {instdiff_model}...")
    unet = UNet2DConditionModel.from_pretrained(instdiff_model, subfolder="unet")

    # Use existing conversion function
    # Wrap checkpoint in expected format (convert_instdiff_unet_checkpoint expects {"model": state_dict})
    print("Converting UNet weights using existing conversion logic...")
    config = {"layers_per_block": 2}
    wrapped_checkpoint = {"model": clean_checkpoint}
    converted_state_dict = convert_instdiff_unet_checkpoint(wrapped_checkpoint, config)

    # Modify conv_in for 9 channels
    print("Modifying conv_in for 9 channels (inpainting)...")
    old_conv_in = unet.conv_in
    new_conv_in = torch.nn.Conv2d(
        9, old_conv_in.out_channels,
        kernel_size=old_conv_in.kernel_size,
        stride=old_conv_in.stride,
        padding=old_conv_in.padding,
    )

    # Load conv_in weights from checkpoint
    if "conv_in.weight" in converted_state_dict:
        ckpt_weight = converted_state_dict["conv_in.weight"]
        if ckpt_weight.shape[1] == 9:
            # Checkpoint already has 9 channels
            new_conv_in.weight.data = ckpt_weight
        else:
            # Initialize: use checkpoint for first channels, zero for rest
            new_conv_in.weight.data.zero_()
            new_conv_in.weight.data[:, :ckpt_weight.shape[1]] = ckpt_weight
        new_conv_in.bias.data = converted_state_dict["conv_in.bias"]
        # Remove from dict to avoid loading again
        del converted_state_dict["conv_in.weight"]
        del converted_state_dict["conv_in.bias"]

    unet.conv_in = new_conv_in
    unet.config["in_channels"] = 9

    # Replace position_net with bbox-only version
    print("Setting up bbox-only position_net...")
    unet._set_pos_net_if_use_instdiff_bbox_only(cross_attention_dim=768)

    # Load converted weights
    print("Loading converted weights into UNet...")
    missing_keys, unexpected_keys = unet.load_state_dict(converted_state_dict, strict=False)

    print(f"Loaded keys: {len(converted_state_dict)}")
    print(f"Missing keys: {len(missing_keys)}")
    if missing_keys and len(missing_keys) < 20:
        for k in missing_keys:
            print(f"  - {k}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    if unexpected_keys and len(unexpected_keys) < 20:
        for k in unexpected_keys:
            print(f"  - {k}")

    # Save pipeline
    print(f"Saving to {output_path}...")
    os.makedirs(output_path, exist_ok=True)

    from diffusers import StableDiffusionINSTDIFFInpaintPipeline

    pipeline = StableDiffusionINSTDIFFInpaintPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    )
    pipeline.save_pretrained(output_path)

    print("Done!")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert ReferenceNet bbox-only checkpoint to diffusers")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to ReferenceNet .pt checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory for converted model",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
        help="Base SD Inpainting model ID",
    )
    parser.add_argument(
        "--instdiff_model",
        type=str,
        default="kyeongry/instancediffusion_sd15",
        help="InstanceDiffusion model ID for UNet structure",
    )
    args = parser.parse_args()

    convert_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        base_model=args.base_model,
        instdiff_model=args.instdiff_model,
    )


if __name__ == "__main__":
    main()
