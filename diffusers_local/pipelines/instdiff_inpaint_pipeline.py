"""
BBox-only InstanceDiffusion Inpainting Pipeline.

This module provides utilities to load and use the global U-Net bbox-only
model with diffusers infrastructure.
"""

from typing import Optional, Union
import os
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    assign_to_checkpoint,
    renew_attention_paths,
    renew_resnet_paths,
    shave_segments,
)
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file

from ..models import INSTDIFFTextBoundingboxProjectionBBoxOnly
from .instdiff_inpaint_pipeline_base import StableDiffusionINSTDIFFInpaintPipeline


def convert_ldm_unet_checkpoint(checkpoint: dict, config: dict) -> dict:
    """
    Convert ldm-style UNet checkpoint to diffusers format.

    This is a simplified version of convert_instdiff_unet_checkpoint that works
    with global U-Net checkpoints (which already have clean keys without a 'model.' prefix).
    """
    unet_state_dict = checkpoint
    keys = list(unet_state_dict.keys())

    new_checkpoint = {}

    # Time embedding
    if "time_embed.0.weight" in unet_state_dict:
        new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict["time_embed.0.weight"]
        new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict["time_embed.0.bias"]
        new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict["time_embed.2.weight"]
        new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict["time_embed.2.bias"]

    # Input/output convs
    if "input_blocks.0.0.weight" in unet_state_dict:
        new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
        new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

    if "out.0.weight" in unet_state_dict:
        new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
        new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
        new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
        new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

    # Count blocks
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})

    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }

    layers_per_block = config.get("layers_per_block", 2)

    # Process input blocks
    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (layers_per_block + 1)
        layer_in_block_id = (i - 1) % (layers_per_block + 1)

        resnets = [
            key for key in input_blocks[i]
            if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict[
                f"input_blocks.{i}.0.op.weight"
            ]
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict[
                f"input_blocks.{i}.0.op.bias"
            ]

        paths = renew_resnet_paths(resnets)
        meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
        assign_to_checkpoint(paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)

        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
            assign_to_checkpoint(paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)

    # Process middle blocks
    if num_middle_blocks >= 3:
        resnet_0 = middle_blocks[0]
        attentions = middle_blocks[1]
        resnet_1 = middle_blocks[2]

        resnet_0_paths = renew_resnet_paths(resnet_0)
        assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

        resnet_1_paths = renew_resnet_paths(resnet_1)
        assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

        attentions_paths = renew_attention_paths(attentions)
        meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
        assign_to_checkpoint(attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)

    # Process output blocks
    for i in range(num_output_blocks):
        block_id = i // (layers_per_block + 1)
        layer_in_block_id = i % (layers_per_block + 1)
        output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]

        if len(output_block_list) > 1:
            resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
            attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

            paths = renew_resnet_paths(resnets)
            meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}
            assign_to_checkpoint(paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)

            output_block_list = {k: sorted(v) for k, v in output_block_list.items()}
            if ["conv.bias", "conv.weight"] in output_block_list.values():
                index = list(output_block_list.values()).index(["conv.bias", "conv.weight"])
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.weight"
                ]
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.bias"
                ]

                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {"old": f"output_blocks.{i}.1", "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}"}
                assign_to_checkpoint(paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])
                new_checkpoint[new_path] = unet_state_dict[old_path]

    # Copy position_net and scaleu weights
    for key in keys:
        if "position_net" in key or "scaleu" in key:
            new_checkpoint[key] = unet_state_dict[key]

    return new_checkpoint


class StableDiffusionINSTDIFFInpaintPipelineBBoxOnly(StableDiffusionINSTDIFFInpaintPipeline):
    """
    BBox-only version of StableDiffusionINSTDIFFInpaintPipeline.

    This class provides a convenient way to load global U-Net bbox-only models
    that use 9-channel input and simplified position_net.
    """

    @classmethod
    def from_global_unet_checkpoint(
        cls,
        checkpoint_path: str,
        base_model: str = "runwayml/stable-diffusion-inpainting",
        instdiff_model: str = "kyeongry/instancediffusion_sd15",
        torch_dtype: torch.dtype = torch.float16,
        device: Union[str, torch.device] = "cuda",
    ) -> "StableDiffusionINSTDIFFInpaintPipelineBBoxOnly":
        """
        Load the pipeline from a global U-Net bbox-only checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file or converted model directory
            base_model: Base SD Inpainting model for VAE/text_encoder/scheduler
            instdiff_model: InstanceDiffusion model for UNet structure (has fuser)
            torch_dtype: Model dtype
            device: Device to load model on

        Returns:
            Configured pipeline ready for inference
        """
        # Load UNet from InstanceDiffusion (has fuser structure)
        print(f"Loading UNet from {instdiff_model}...")
        unet = UNet2DConditionModel.from_pretrained(
            instdiff_model,
            subfolder="unet",
            torch_dtype=torch_dtype,
        )

        # Modify conv_in for 9 channels
        print("Modifying conv_in for 9 channels...")
        old_conv_in = unet.conv_in
        new_conv_in = torch.nn.Conv2d(
            9, old_conv_in.out_channels,
            kernel_size=old_conv_in.kernel_size,
            stride=old_conv_in.stride,
            padding=old_conv_in.padding,
        ).to(dtype=torch_dtype)
        unet.conv_in = new_conv_in
        unet.config["in_channels"] = 9

        # Replace position_net with bbox-only version
        print("Setting bbox-only position_net...")
        unet.position_net = INSTDIFFTextBoundingboxProjectionBBoxOnly(
            positive_len=768,
            out_dim=768,
        ).to(dtype=torch_dtype)

        # Load checkpoint weights
        print(f"Loading weights from {checkpoint_path}...")

        if checkpoint_path.endswith(".pt") or checkpoint_path.endswith(".bin"):
            # Load raw .pt checkpoint and convert
            raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Remove 'inner.' prefix if present
            clean_checkpoint = {}
            for key, value in raw_checkpoint.items():
                new_key = key[len("inner."):] if key.startswith("inner.") else key
                clean_checkpoint[new_key] = value

            # Convert ldm keys to diffusers keys
            print("Converting checkpoint keys from ldm to diffusers format...")
            config = {"layers_per_block": 2}
            converted_state_dict = convert_ldm_unet_checkpoint(clean_checkpoint, config)

            # Handle conv_in weights for 9 channels
            if "conv_in.weight" in converted_state_dict:
                ckpt_weight = converted_state_dict["conv_in.weight"]
                if ckpt_weight.shape[1] == 9:
                    new_conv_in.weight.data = ckpt_weight.to(dtype=torch_dtype)
                else:
                    new_conv_in.weight.data.zero_()
                    new_conv_in.weight.data[:, :ckpt_weight.shape[1]] = ckpt_weight.to(dtype=torch_dtype)
                new_conv_in.bias.data = converted_state_dict["conv_in.bias"].to(dtype=torch_dtype)
                del converted_state_dict["conv_in.weight"]
                del converted_state_dict["conv_in.bias"]

            state_dict = converted_state_dict

        elif checkpoint_path.endswith(".safetensors"):
            state_dict = load_file(checkpoint_path)
        else:
            # Assume it's a directory with safetensors
            safetensors_path = os.path.join(checkpoint_path, "unet", "diffusion_pytorch_model.safetensors")
            if os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path)
            else:
                bin_path = os.path.join(checkpoint_path, "unet", "diffusion_pytorch_model.bin")
                state_dict = torch.load(bin_path, map_location="cpu")

        # Filter keys to match bbox-only structure and load
        bbox_only_keys = set(unet.state_dict().keys())
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in bbox_only_keys}

        missing, unexpected = unet.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded {len(filtered_state_dict)} keys, missing: {len(missing)}, unexpected: {len(unexpected)}")

        if missing and len(missing) < 20:
            print("Missing keys:")
            for k in missing:
                print(f"  - {k}")

        # Load other components from base model
        print(f"Loading VAE/text_encoder/scheduler from {base_model}...")
        vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=torch_dtype)
        tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=torch_dtype)
        scheduler = DDIMScheduler.from_pretrained(base_model, subfolder="scheduler")

        # Create pipeline
        pipe = cls(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
        )

        pipe = pipe.to(device)
        print("Pipeline ready!")

        return pipe

    @classmethod
    def from_converted_model(
        cls,
        model_path: str,
        instdiff_model: str = "kyeongry/instancediffusion_sd15",
        torch_dtype: torch.dtype = torch.float16,
        device: Union[str, torch.device] = "cuda",
    ) -> "StableDiffusionINSTDIFFInpaintPipelineBBoxOnly":
        """
        Load pipeline from a previously converted model directory.

        This method handles the position_net replacement automatically.

        Args:
            model_path: Path to converted model directory (output of conversion script)
            instdiff_model: InstanceDiffusion model for UNet structure
            torch_dtype: Model dtype
            device: Device to load model on

        Returns:
            Configured pipeline ready for inference
        """
        return cls.from_global_unet_checkpoint(
            checkpoint_path=model_path,
            base_model=model_path,  # Use converted model's VAE/scheduler
            instdiff_model=instdiff_model,
            torch_dtype=torch_dtype,
            device=device,
        )
