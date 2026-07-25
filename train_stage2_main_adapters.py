#!/usr/bin/env python
"""
Stage-2 main-only training (Diffusers) — global U-Net feature injection.

Implements the direction in `tmp_main_only_training_direction.md`:
- Stage 1 global U-Net (InstanceDiffusion UNet) writes a feature bank under `torch.no_grad()`
- Main UNet injects the bank via per-block gated fusers (post-attn1)
- Loss is computed on main UNet output only
- Only adapter modules are trained:
  - patch_unifusion (bbox -> patch token), initialized from InstanceDiffusion pretrained position_net (bbox branch)
  - main_fusers (per-block GatedSelfAttentionDense), initialized from InstanceDiffusion pretrained fusers (alphas reset)
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore[assignment]

try:
    import numpy as np  # type: ignore
    import torchvision  # type: ignore
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional dependencies for image logging
    np = None  # type: ignore[assignment]
    torchvision = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]


def _maybe_add_local_diffusers_to_path() -> None:
    """
    Optional convenience: if `diffusers` is not installed, try the repo-local fork.
    This does not install dependencies; it only adjusts `sys.path`.
    """
    try:
        import diffusers  # noqa: F401
    except Exception:
        import sys

        local_src = Path(__file__).resolve().parent / "third_party" / "diffusers" / "src"
        if local_src.is_dir():
            sys.path.insert(0, str(local_src))


_maybe_add_local_diffusers_to_path()

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTextModel, CLIPTokenizer, get_scheduler

from data.dataloader_train_global import create_progressive_condition_dataloader
from diffusers_local.models import INSTDIFFTextBoundingboxProjectionBBoxOnly, build_main_unet_no_scaleu
from diffusers_local.pipelines.instdiff_inpaint_pipeline import convert_ldm_unet_checkpoint
from diffusers_local.training import (
    DiffusersGlobalUNetFuserWriter,
    DiffusersUNetFuserBankInjector,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Models / checkpoints
    p.add_argument(
        "--global_checkpoint",
        type=str,
        default="",
        help="Path to stage1_epochXXXX.pt. If empty, uses --global_run_dir + --global_epoch.",
    )
    p.add_argument(
        "--global_run_dir",
        type=str,
        default="",
        help="Stage 1 run directory that contains ckpt/stage1_epochXXXX.pt.",
    )
    p.add_argument(
        "--global_epoch",
        type=int,
        default=60,
        help="1-based exported checkpoint number to load from --global_run_dir (default: 60 -> stage1_epoch0060.pt).",
    )
    p.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-inpainting")
    p.add_argument("--instdiff_model", type=str, default="kyeongry/instancediffusion_sd15")

    # Data
    p.add_argument("--images_root", type=str, default="datasets/images")
    p.add_argument("--annotations_root", type=str, default="datasets/annotations")
    p.add_argument("--include_datasets", type=str, default="", help="Comma-separated dataset names (optional).")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="DataLoader prefetch_factor (only used when num_workers>0).",
    )
    p.add_argument(
        "--persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DataLoader persistent_workers (only used when num_workers>0).",
    )
    p.add_argument("--n_max_instances", type=int, default=30)
    p.add_argument("--embedding_key", type=str, default="text_embedding_before")
    p.add_argument("--use_instance_attn_mask", action="store_true")

    # Training
    p.add_argument(
        "--max_epochs",
        type=int,
        default=0,
        help="If >0, train for this many epochs. If 0, train indefinitely (until interrupted), unless --max_train_steps is set.",
    )
    p.add_argument("--max_train_steps", type=int, default=0, help="If >0, overrides epoch-based steps.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=6)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=5000)
    p.add_argument("--gradient_clip_norm", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    # Global feature injection controls
    p.add_argument("--global_token_scale", type=float, default=1.0)
    p.add_argument(
        "--p_cfg_drop",
        type=float,
        default=0.1,
        help="CFG-style text dropout prob. Dropped samples use empty local prompt (text-only CFG); global U-Net features stay enabled.",
    )

    # Logging / saving
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--save_every_n_epochs", type=int, default=1)
    p.add_argument("--log_every_n_steps", type=int, default=20)
    p.add_argument(
        "--no_epoch_image_logging",
        action="store_true",
        default=False,
        help="Disable epoch-end image logging.",
    )
    p.add_argument(
        "--no_batch_image_logging",
        action="store_true",
        default=False,
        help="Disable periodic batch image logging.",
    )
    p.add_argument(
        "--batch_image_log_frequency",
        type=int,
        default=2000,
        help="Log images every N batches within each epoch (default: 2000).",
    )
    p.add_argument(
        "--max_log_images",
        type=int,
        default=10,
        help="Max images to save per image-log event (default: 10).",
    )
    p.add_argument(
        "--ddim_steps",
        type=int,
        default=30,
        help="DDIM steps for image logging sampling (default: 30).",
    )
    p.add_argument(
        "--unconditional_guidance_label",
        type=str,
        default="ugly, nsfw, worst quality, watermark, signature, logo",
        help="Negative prompt used for classifier-free guidance during image logging.",
    )
    p.add_argument(
        "--unconditional_guidance_scale",
        type=float,
        default=2.5,
        help="CFG scale used during image logging sampling (default: 2.5).",
    )
    p.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="DDIM eta for image logging sampling (default: 0.0).",
    )
    p.add_argument(
        "--image_log_seed",
        type=int,
        default=-1,
        help="If >=0, seed a dedicated generator for image logging (default: -1 => use global RNG).",
    )

    # wandb (match train_stage1_global_unet.py flags)
    p.add_argument("--wandb_project", type=str, default="outpainting-stage2", help="Wandb project name")
    p.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (team/username)")
    p.add_argument("--wandb_offline", action="store_true", default=False, help="Run wandb in offline mode")

    # Resume
    p.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Path to main_only_epochXXXX.pt checkpoint to resume training from.",
    )

    return p.parse_args()


def _resolve_global_checkpoint(args: argparse.Namespace) -> str:
    ckpt = (getattr(args, "global_checkpoint", "") or "").strip()
    if ckpt:
        return ckpt

    run_dir = (getattr(args, "global_run_dir", "") or "").strip()
    if not run_dir:
        raise SystemExit(
            "Missing Stage 1 global U-Net checkpoint. Set --global_checkpoint explicitly, "
            "or provide --global_run_dir (and optionally --global_epoch; default=60)."
        )

    epoch = int(getattr(args, "global_epoch", 60))
    ckpt = Path(run_dir) / "ckpt" / f"stage1_epoch{epoch:04d}.pt"
    if not ckpt.is_file():
        raise SystemExit(f"Stage 1 global U-Net checkpoint not found: {ckpt} (run_dir={run_dir} epoch={epoch})")
    return str(ckpt)


def _maybe_init_wandb(
    args: argparse.Namespace,
    accelerator: Accelerator,
    num_batches_per_epoch: int,
    num_update_steps_per_epoch: int,
    max_epochs: int,
    max_train_steps: int,
):
    if not accelerator.is_main_process:
        return None
    if str(os.environ.get("WANDB_DISABLED", "")).lower() in {"1", "true", "yes"}:
        return None
    try:
        import wandb  # type: ignore
    except Exception as e:
        print(f"[wandb][warn] wandb not available ({e}); continuing without wandb.")
        return None

    # Safer defaults for multiprocess launchers.
    os.environ.setdefault("WANDB_START_METHOD", "thread")
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    run_name = Path(args.output_dir).name
    init_kwargs = {
        "project": args.wandb_project,
        "name": run_name,
        "dir": str(Path(args.output_dir).resolve()),
        "config": {
            "output_dir": args.output_dir,
            "global_checkpoint": args.global_checkpoint,
            "base_model": args.base_model,
            "instdiff_model": args.instdiff_model,
            "images_root": args.images_root,
            "annotations_root": args.annotations_root,
            "include_datasets": args.include_datasets,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "persistent_workers": args.persistent_workers,
            "n_max_instances": args.n_max_instances,
            "embedding_key": args.embedding_key,
            "use_instance_attn_mask": args.use_instance_attn_mask,
            "mixed_precision": args.mixed_precision,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "gradient_clip_norm": args.gradient_clip_norm,
            "seed": args.seed,
            "global_token_scale": args.global_token_scale,
            "p_cfg_drop": args.p_cfg_drop,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_processes": accelerator.num_processes,
            "batches_per_epoch": num_batches_per_epoch,
            "updates_per_epoch": num_update_steps_per_epoch,
            "max_epochs": None if max_epochs == 0 else max_epochs,
            "max_train_steps": None if max_train_steps == 0 else max_train_steps,
        },
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity

    try:
        run = wandb.init(**init_kwargs)
        print(f"[wandb] initialized: project={args.wandb_project}, run={run_name}, url={run.get_url()}")
        return run
    except Exception as e:
        print(f"[wandb][error] init failed: {e}")
        return None


def _as_list_csv(s: str) -> Optional[List[str]]:
    s = (s or "").strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _to_bchw(x_bhwc: torch.Tensor) -> torch.Tensor:
    return x_bhwc.permute(0, 3, 1, 2).contiguous()


def _grad_norm_and_stats(params) -> Tuple[float, float, bool]:
    """Compute gradient norm, nonzero ratio, and NaN flag for a list of parameters."""
    params = list(params)
    params_with_grad = [p for p in params if p.grad is not None]
    total_params = sum(1 for p in params if p.requires_grad)
    if not params_with_grad:
        return 0.0, 0.0, False
    grads = [p.grad for p in params_with_grad]
    total_norm_sq = torch.zeros((), device=grads[0].device, dtype=torch.float32)
    for g in grads:
        total_norm_sq = total_norm_sq + g.detach().float().pow(2).sum()
    total_norm = torch.sqrt(total_norm_sq)
    has_nan = any(torch.isnan(g).any() or torch.isinf(g).any() for g in grads)
    ratio = len(params_with_grad) / max(1, total_params)
    return float(total_norm.item()), float(ratio), bool(has_nan)


def _epoch_offset() -> int:
    # Mirror PL callbacks behavior (useful when doing weights-only resumes).
    try:
        return int(os.environ.get("EPOCH_OFFSET", "0"))
    except Exception:
        return 0


def _mask_to_vis(mask_bchw: torch.Tensor) -> torch.Tensor:
    """
    Convert {0,1} masks to a 3ch tensor in [-1,1] for visualization.
    """
    m = mask_bchw[:, :1]
    m = (m * 2.0) - 1.0
    return m.repeat(1, 3, 1, 1)


def _save_image_grid(
    *,
    root_dir: Path,
    group: str,
    split: str,
    epoch: int,
    batch_idx: int,
    global_step: int,
    images: Dict[str, torch.Tensor],
    include_index_in_filename: bool,
) -> None:
    if np is None or torchvision is None or Image is None:
        return

    effective_epoch = int(epoch) + int(_epoch_offset())
    epoch_dir = root_dir / "image_log" / group / f"epoch_{effective_epoch + 1:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    for idx, key in enumerate(sorted(images.keys()), start=1):
        x = images[key].detach().float().cpu()
        grid = torchvision.utils.make_grid(x, nrow=4)
        grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1
        grid = grid.clamp(0.0, 1.0)
        grid = grid.permute(1, 2, 0).numpy()
        grid = (grid * 255).astype(np.uint8)

        if include_index_in_filename:
            filename = f"b-{batch_idx:06d}_{idx:02d}_{key}_gs-{global_step:06d}_e-{effective_epoch:06d}.png"
        else:
            filename = f"b-{batch_idx:06d}_{key}_gs-{global_step:06d}_e-{effective_epoch:06d}.png"

        Image.fromarray(grid).save(str(epoch_dir / filename))

    # Best-effort prompt dump (one file per image-log event)
    if split:
        prompts_path = epoch_dir / (
            f"b-{batch_idx:06d}_prompts_gs-{global_step:06d}_e-{effective_epoch:06d}.txt"
        )
        try:
            with open(prompts_path, "w", encoding="utf-8") as f:
                f.write(split.strip() + "\n")
        except Exception:
            pass


@torch.no_grad()
def _decode_vae(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    scale = getattr(vae.config, "scaling_factor", None)
    if scale is None:
        raise AttributeError("VAE is missing config.scaling_factor; do not hardcode latent scaling.")
    x = vae.decode(latents / float(scale)).sample
    return x.clamp(-1.0, 1.0)


@torch.no_grad()
def _log_global_local_images(
    *,
    args: argparse.Namespace,
    root_dir: Path,
    device: torch.device,
    weight_dtype: torch.dtype,
    accelerator: Accelerator,
    vae: AutoencoderKL,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    noise_scheduler: DDIMScheduler,
    global_unet: UNet2DConditionModel,
    main_unet: UNet2DConditionModel,
    global_writer: DiffusersGlobalUNetFuserWriter,
    main_fuser_injector: DiffusersUNetFuserBankInjector,
    patch_unifusion: torch.nn.Module,
    generator: Optional[torch.Generator],
    batch: Dict,
    epoch: int,
    batch_idx: int,
    global_step: int,
    include_index_in_filename: bool,
) -> None:
    if np is None or torchvision is None or Image is None:
        return

    max_images = int(getattr(args, "max_log_images", 10))
    ddim_steps = int(getattr(args, "ddim_steps", 30))
    guidance_scale = float(getattr(args, "unconditional_guidance_scale", 2.5))
    negative_prompt = str(getattr(args, "unconditional_guidance_label", "") or "")
    eta = float(getattr(args, "eta", 0.0))

    try:
      with accelerator.autocast():
        # Slice a small batch for logging
        image = _to_bchw(batch["image"].to(device, dtype=weight_dtype))[:max_images]
        mask = _to_bchw(batch["mask"].to(device, dtype=weight_dtype))[:max_images]
        masked_image = _to_bchw(batch["masked_image"].to(device, dtype=weight_dtype))[:max_images]

        global_image = _to_bchw(batch["global_image"].to(device, dtype=weight_dtype))[:max_images]
        global_image_full = _to_bchw(batch["global_image_full"].to(device, dtype=weight_dtype))[:max_images]
        global_mask = _to_bchw(batch["global_mask"].to(device, dtype=weight_dtype))[:max_images]

        ref_window_bbox = batch["ref_window_bbox"].to(device, dtype=weight_dtype)[:max_images]
        ref_boxes = batch["ref_boxes"].to(device, dtype=weight_dtype)[:max_images]
        ref_masks = batch["ref_masks"].to(device, dtype=weight_dtype)[:max_images]
        ref_pos = batch["ref_positive_embeddings"].to(device, dtype=weight_dtype)[:max_images]

        local_prompts: List[str] = list(batch["txt"])[: int(image.shape[0])]
        global_prompts: List[str] = list(batch["global_prompt"])[: int(image.shape[0])]

        # Text embeddings
        local_prompt_embeds = _encode_prompts(tokenizer, text_encoder, local_prompts, device=device, dtype=weight_dtype)
        global_prompt_embeds = _encode_prompts(tokenizer, text_encoder, global_prompts, device=device, dtype=weight_dtype)

        negative_prompts = [negative_prompt] * int(image.shape[0])
        local_neg_embeds = _encode_prompts(tokenizer, text_encoder, negative_prompts, device=device, dtype=weight_dtype)
        global_neg_embeds = _encode_prompts(tokenizer, text_encoder, negative_prompts, device=device, dtype=weight_dtype)

        # Prepare inpaint conditioning latents
        local_masked_latents = _encode_vae(vae, masked_image)
        local_mask_latent = F.interpolate(mask[:, :1], size=local_masked_latents.shape[-2:], mode="nearest")

        global_masked_latents = _encode_vae(vae, global_image)
        global_mask_latent = F.interpolate(global_mask[:, :1], size=global_masked_latents.shape[-2:], mode="nearest")

        # Init latents (noise)
        init_sigma = float(getattr(noise_scheduler, "init_noise_sigma", 1.0))
        local_latents = randn_tensor(
            local_masked_latents.shape,
            generator=generator,
            device=device,
            dtype=local_masked_latents.dtype,
        )
        local_latents = local_latents * init_sigma
        global_latents = randn_tensor(
            global_masked_latents.shape,
            generator=generator,
            device=device,
            dtype=global_masked_latents.dtype,
        )
        global_latents = global_latents * init_sigma

        B = int(local_latents.shape[0])

        # Sampling loop
        noise_scheduler.set_timesteps(ddim_steps, device=device)
        extra_step_kwargs = {"eta": eta}
        if generator is not None:
            extra_step_kwargs["generator"] = generator

        sampling_pbar = tqdm(noise_scheduler.timesteps, desc="[image-log] sampling", leave=False) if tqdm else None
        for t in (sampling_pbar if sampling_pbar else noise_scheduler.timesteps):
            # --- Global U-Net ---
            bank_tokens_by_block: List[torch.Tensor] = []
            if guidance_scale != 1.0:
                # Batched CFG (match diffusers): [UC|COND] in one forward
                latent_model_input = torch.cat([global_latents] * 2, dim=0)
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                mask_latent_cfg = torch.cat([global_mask_latent] * 2, dim=0)
                masked_image_latents_cfg = torch.cat([global_masked_latents] * 2, dim=0)
                global_model_input = torch.cat([latent_model_input, mask_latent_cfg, masked_image_latents_cfg], dim=1)

                encoder_hidden_states_cfg = torch.cat([global_neg_embeds, global_prompt_embeds], dim=0)

                boxes_cfg = torch.cat([ref_boxes, ref_boxes], dim=0)
                pos_cfg = torch.cat([ref_pos, ref_pos], dim=0)
                masks_cfg = torch.cat([ref_masks, ref_masks], dim=0)
                # Diffusers Instdiff behavior: disable grounding on unconditional branch
                masks_cfg[:B] = 0
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

                global_noise_pred_uncond, global_noise_pred_cond = global_noise_pred_2b.chunk(2, dim=0)
                global_noise_pred = global_noise_pred_uncond + guidance_scale * (
                    global_noise_pred_cond - global_noise_pred_uncond
                )

                for i in range(global_writer.num_blocks):
                    bank_list = global_writer.bank.get(i, [])
                    if not bank_list:
                        raise RuntimeError(f"[image-log] Global U-Net bank missing for block {i}.")
                    tokens = torch.cat(bank_list, dim=1) if len(bank_list) > 1 else bank_list[0]
                    # Use COND half only for ref injection into main UNet
                    bank_tokens_by_block.append(tokens[B:])
                global_writer.clear()
            else:
                latent_model_input = noise_scheduler.scale_model_input(global_latents, t)
                global_model_input = torch.cat([latent_model_input, global_mask_latent, global_masked_latents], dim=1)

                cross_attention_kwargs = {
                    "instdiff": {
                        "boxes": ref_boxes,
                        "positive_embeddings": ref_pos,
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

                for i in range(global_writer.num_blocks):
                    bank_list = global_writer.bank.get(i, [])
                    if not bank_list:
                        raise RuntimeError(f"[image-log] Global U-Net bank missing for block {i}.")
                    bank_tokens_by_block.append(torch.cat(bank_list, dim=1) if len(bank_list) > 1 else bank_list[0])
                global_writer.clear()

            global_latents = noise_scheduler.step(global_noise_pred, t, global_latents, **extra_step_kwargs).prev_sample
            # Diffusers default for 9ch inpaint UNets: do NOT blend latents with the init image here.
            # The UNet is conditioned via concatenated (mask, masked_image_latents).

            # --- Local (Main UNet w/ post-attn fuser injection) ---
            main_fuser_injector.update({i: bank_tokens_by_block[i] for i in range(len(bank_tokens_by_block))})

            if guidance_scale != 1.0:
                # Disable fuser injection for UC branch (match diffusers grounding semantics)
                inject_mask = torch.zeros((2 * B,), device=device, dtype=torch.bool)
                inject_mask[B:] = True
                main_fuser_injector.set_inject_mask(inject_mask)

                latent_model_input = torch.cat([local_latents] * 2, dim=0)
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                mask_latent_cfg = torch.cat([local_mask_latent] * 2, dim=0)
                masked_image_latents_cfg = torch.cat([local_masked_latents] * 2, dim=0)
                local_model_input = torch.cat([latent_model_input, mask_latent_cfg, masked_image_latents_cfg], dim=1)

                local_embeds_cfg = torch.cat([local_neg_embeds, local_prompt_embeds], dim=0)

                bbox_cfg = torch.cat([ref_window_bbox, ref_window_bbox], dim=0)
                patch_masks_cfg = torch.ones((2 * B, 1), device=device, dtype=weight_dtype)
                patch_masks_cfg[:B] = 0
                pos_null_cfg = (
                    patch_unifusion.null_positive_feature.view(1, 1, -1)
                    .expand(2 * B, 1, -1)
                    .to(device=device, dtype=weight_dtype)
                )
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
                local_noise_pred_uncond, local_noise_pred_cond = local_noise_pred_2b.chunk(2, dim=0)
                local_noise_pred = local_noise_pred_uncond + guidance_scale * (
                    local_noise_pred_cond - local_noise_pred_uncond
                )
            else:
                main_fuser_injector.set_inject_mask(None)
                latent_model_input = noise_scheduler.scale_model_input(local_latents, t)
                local_model_input = torch.cat([latent_model_input, local_mask_latent, local_masked_latents], dim=1)

                patch_masks = torch.ones((B, 1), device=device, dtype=weight_dtype)
                pos_null = (
                    patch_unifusion.null_positive_feature.view(1, 1, -1)
                    .expand(B, 1, -1)
                    .to(device=device, dtype=weight_dtype)
                )
                patch_kwargs = {
                    "instdiff": {
                        "boxes": ref_window_bbox,
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

            main_fuser_injector.clear()
            local_latents = noise_scheduler.step(local_noise_pred, t, local_latents, **extra_step_kwargs).prev_sample
            # Same as above: no explicit latents blending for 9ch inpaint UNets.

        # Decode samples
        global_sample = _decode_vae(vae, global_latents)
        local_sample = _decode_vae(vae, local_latents)

        # Save global/local separately
        global_images = {
            "global_input": global_image.detach().float(),
            "global_mask": _mask_to_vis(global_mask.detach().float()),
            "global_gt": global_image_full.detach().float(),
            "global_sample": global_sample.detach().float(),
        }
        local_images = {
            "local_masked_input": masked_image.detach().float(),
            "local_mask": _mask_to_vis(mask.detach().float()),
            "local_gt": image.detach().float(),
            "local_sample": local_sample.detach().float(),
        }

        global_prompts_dump = "\n".join([p.strip() for p in global_prompts[: int(image.shape[0])]])
        local_prompts_dump = "\n".join([p.strip() for p in local_prompts[: int(image.shape[0])]])

        _save_image_grid(
            root_dir=root_dir,
            group="global",
            split=global_prompts_dump,
            epoch=epoch,
            batch_idx=batch_idx,
            global_step=global_step,
            images=global_images,
            include_index_in_filename=include_index_in_filename,
        )
        _save_image_grid(
            root_dir=root_dir,
            group="local",
            split=local_prompts_dump,
            epoch=epoch,
            batch_idx=batch_idx,
            global_step=global_step,
            images=local_images,
            include_index_in_filename=include_index_in_filename,
        )
    finally:
        # Clean up hook state (training loop will clear again, but keep it tidy)
        global_writer.clear()
        main_fuser_injector.clear()


@torch.no_grad()
def _encode_prompts(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompts: List[str],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tokens = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    enc = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    return enc.last_hidden_state.to(dtype=dtype)


@torch.no_grad()
def _encode_vae(vae: AutoencoderKL, images_bchw: torch.Tensor) -> torch.Tensor:
    latents = vae.encode(images_bchw).latent_dist.sample()
    scale = getattr(vae.config, "scaling_factor", None)
    if scale is None:
        raise AttributeError("VAE is missing config.scaling_factor; do not hardcode latent scaling.")
    return latents * float(scale)


def _load_global_unet_from_ckpt(
    global_checkpoint: str,
    *,
    instdiff_model: str,
    torch_dtype: torch.dtype,
) -> UNet2DConditionModel:
    # UNet structure from InstanceDiffusion (has fuser modules)
    unet = UNet2DConditionModel.from_pretrained(
        instdiff_model,
        subfolder="unet",
        torch_dtype=torch_dtype,
    )

    # Modify conv_in to 9 channels (inpaint-style)
    old_conv_in = unet.conv_in
    unet.conv_in = torch.nn.Conv2d(
        9,
        old_conv_in.out_channels,
        kernel_size=old_conv_in.kernel_size,
        stride=old_conv_in.stride,
        padding=old_conv_in.padding,
    ).to(dtype=torch_dtype)
    unet.config["in_channels"] = 9

    # Ensure bbox-only position_net compatible with our Stage 1 global U-Net checkpoints
    unet.position_net = INSTDIFFTextBoundingboxProjectionBBoxOnly(
        positive_len=768,
        out_dim=768,
    ).to(dtype=torch_dtype)

    # Load raw checkpoint and convert LDM -> diffusers keys
    raw = torch.load(global_checkpoint, map_location="cpu")
    clean: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if k.startswith("inner."):
            clean[k[len("inner.") :]] = v
        else:
            clean[k] = v

    converted = convert_ldm_unet_checkpoint(clean, {"layers_per_block": 2})

    # Fill conv_in from checkpoint (may be 4ch or 9ch). This is required.
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

    # Filter to keys that exist in the instantiated bbox-only UNet
    allowed = set(unet.state_dict().keys())
    filtered = {k: v for k, v in converted.items() if k in allowed}

    # Ensure conv_in is not reported as missing by load_state_dict
    filtered["conv_in.weight"] = unet.conv_in.weight.detach().clone()
    filtered["conv_in.bias"] = unet.conv_in.bias.detach().clone()

    # Required checkpoint coverage: position_net + fusers must be fully populated.
    required_model_keys = {k for k in allowed if k.startswith("position_net.") or ".fuser." in k}
    missing_required = sorted(required_model_keys - set(filtered.keys()))
    if missing_required:
        preview = "\n".join(f"  - {k}" for k in missing_required[:20])
        raise RuntimeError(
            "Converted checkpoint is missing required Stage 1 global U-Net keys (position_net / fuser). "
            f"missing_required={len(missing_required)}\n{preview}"
        )

    missing, unexpected = unet.load_state_dict(filtered, strict=False)
    # Do not enforce strict loading here; stage-2 training only requires
    # strict init for the new gating modules. Still, surface suspicious loads.
    if len(missing) > 0:
        print(f"[GlobalUNet][warn] missing keys: {len(missing)} (unexpected: {len(unexpected)})")
        if len(missing) < 20:
            for k in missing:
                print(f"  - {k}")
    return unet


def _collect_transformer_blocks(unet: torch.nn.Module) -> List[torch.nn.Module]:
    blocks: List[torch.nn.Module] = []
    for module in unet.modules():
        tb = getattr(module, "transformer_blocks", None)
        if tb is None:
            continue
        for block in list(tb):
            blocks.append(block)
    return blocks


def _collect_fusers(unet: torch.nn.Module) -> List[torch.nn.Module]:
    fusers: List[torch.nn.Module] = []
    for block in _collect_transformer_blocks(unet):
        fuser = getattr(block, "fuser", None)
        if fuser is not None:
            fusers.append(fuser)
    return fusers


def _freeze_(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def _has_nonzero_grad(params) -> bool:
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        if g.is_sparse:
            g = g.coalesce().values()
        # avoid tiny-type underflow artifacts by checking max in fp32
        if torch.isfinite(g).any() and g.float().abs().max().item() > 0.0:
            return True
    return False


def _main() -> None:
    args = _parse_args()
    args.global_checkpoint = _resolve_global_checkpoint(args)

    accelerator = Accelerator(
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "ckpt"
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Snapshot args for reproducibility (best-effort)
        try:
            import json

            with open(output_dir / "args.json", "w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    set_seed(args.seed)

    # Dataloader (reused as-is from data/dataloader_train_global.py)
    train_loader, _ = create_progressive_condition_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        images_root=args.images_root,
        annotations_root=args.annotations_root,
        include_datasets=_as_list_csv(args.include_datasets),
        n_max_instances=args.n_max_instances,
        embedding_key=args.embedding_key,
        use_instance_attn_mask=args.use_instance_attn_mask,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )

    # Precision policy:
    # - weight_dtype: activation / IO dtype used inside autocast regions
    # - model_dtype: parameter storage dtype
    #
    # Keep model params in fp32 for mixed precision. Autocast handles lower-precision
    # compute while optimizer-owned trainable parameters retain fp32 storage.
    weight_dtype = torch.float32
    model_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Base SD inpaint components
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae", torch_dtype=model_dtype)
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder", torch_dtype=model_dtype)
    noise_scheduler = DDIMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    main_unet, patch_unifusion = build_main_unet_no_scaleu(
        base_model=args.base_model,
        instdiff_model=args.instdiff_model,
        torch_dtype=model_dtype,
    )

    # Stage 1 global U-Net (InstanceDiffusion) from LDM checkpoint
    global_unet = _load_global_unet_from_ckpt(
        args.global_checkpoint,
        instdiff_model=args.instdiff_model,
        torch_dtype=model_dtype,
    )

    # Move frozen models to device
    device = accelerator.device
    vae.to(device)
    text_encoder.to(device)
    main_unet.to(device)
    global_unet.to(device)

    # Freeze everything except gating modules
    _freeze_(vae)
    _freeze_(text_encoder)
    _freeze_(main_unet)
    _freeze_(global_unet)
    vae.eval()
    text_encoder.eval()
    main_unet.eval()
    global_unet.eval()

    # Attach patch tokenizer to main UNet so `cross_attention_kwargs["instdiff"]` triggers post-attn fusers.
    main_unet.position_net = patch_unifusion

    # Collect per-block fusers from the main UNet (InstanceDiffusion fuser modules)
    main_fusers: List[torch.nn.Module] = []
    main_fusers = _collect_fusers(main_unet)
    if not main_fusers:
        raise RuntimeError(
            "No fuser modules found in main_unet; ensure InstanceDiffusion UNet was used for main (instdiff_model)."
        )
    # Start from "no effect" while keeping weights initialized (same as InstanceDiffusion identity start).
    for f in main_fusers:
        with torch.no_grad():
            for name in ("alpha_attn", "alpha_dense"):
                p = getattr(f, name, None)
                if isinstance(p, torch.nn.Parameter):
                    p.zero_()

    # Only adapters trainable
    trainable_params = list(patch_unifusion.parameters()) + [p for f in main_fusers for p in f.parameters()]
    for p in trainable_params:
        p.requires_grad = True

    non_fp32_trainable = [p.dtype for p in trainable_params if p.dtype != torch.float32]
    if non_fp32_trainable:
        raise RuntimeError(
            "Mixed-precision training requires fp32 trainable parameter storage; "
            f"found dtypes: {sorted({str(dtype) for dtype in non_fp32_trainable})}"
        )

    # Sanity checks
    if accelerator.is_main_process:
        assert sum(p.requires_grad for p in global_unet.parameters()) == 0
        trainable_main = [p for p in main_unet.parameters() if p.requires_grad]
        assert trainable_main and all(p.requires_grad for p in trainable_main)
        assert len(trainable_main) == len(trainable_params)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    main_unet, optimizer, train_loader = accelerator.prepare(main_unet, optimizer, train_loader)

    # NOTE: train_loader is now potentially sharded/wrapped by Accelerate. Compute steps after prepare.
    num_batches_per_epoch = len(train_loader)
    num_update_steps_per_epoch = math.ceil(num_batches_per_epoch / args.gradient_accumulation_steps)

    # Training length (0 means "no limit" for that dimension).
    max_train_steps = int(args.max_train_steps) if args.max_train_steps and args.max_train_steps > 0 else 0
    max_epochs = int(args.max_epochs) if args.max_epochs and args.max_epochs > 0 else 0

    if max_train_steps > 0:
        # Derive a finite epoch count for checkpoint cadence / logging.
        max_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    elif max_epochs > 0:
        max_train_steps = max_epochs * num_update_steps_per_epoch

    # constant_with_warmup becomes constant after warmup; for indefinite training we only need warmup length.
    num_training_steps_for_sched = max_train_steps if max_train_steps > 0 else max(int(args.warmup_steps), 0) + 1
    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps_for_sched,
    )

    if accelerator.is_main_process:
        eff_batch = int(args.batch_size) * int(accelerator.num_processes) * int(args.gradient_accumulation_steps)
        print(
            "[precision] "
            f"mixed_precision={args.mixed_precision} "
            f"model_dtype={model_dtype} "
            f"autocast_io_dtype={weight_dtype}"
        )
        print(
            "[steps] "
            f"batches/epoch={num_batches_per_epoch} "
            f"updates/epoch={num_update_steps_per_epoch} "
            f"accum={args.gradient_accumulation_steps} "
            f"num_processes={accelerator.num_processes} "
            f"effective_batch_per_update={eff_batch} "
            f"max_epochs={'∞' if max_epochs == 0 else max_epochs} "
            f"max_train_steps={'∞' if max_train_steps == 0 else max_train_steps}"
        )

    wandb_run = _maybe_init_wandb(
        args=args,
        accelerator=accelerator,
        num_batches_per_epoch=num_batches_per_epoch,
        num_update_steps_per_epoch=num_update_steps_per_epoch,
        max_epochs=max_epochs,
        max_train_steps=max_train_steps,
    )

    # Global U-Net bank write + main fuser injection hooks
    global_writer = DiffusersGlobalUNetFuserWriter(global_unet, detach_global_tokens=True)
    main_fuser_injector = DiffusersUNetFuserBankInjector(
        accelerator.unwrap_model(main_unet),
        global_token_scale=args.global_token_scale,
    )
    assert global_writer.num_blocks == main_fuser_injector.num_blocks, (
        f"Writer/Injector block mismatch: writer={global_writer.num_blocks} injector={main_fuser_injector.num_blocks}"
    )

    image_log_generator: Optional[torch.Generator] = None
    if accelerator.is_main_process and int(getattr(args, "image_log_seed", -1)) >= 0:
        image_log_generator = torch.Generator(device=device).manual_seed(int(args.image_log_seed))

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    resume_path = (args.resume_from or "").strip()
    if resume_path:
        if accelerator.is_main_process:
            print(f"[Resume] Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")

        # Load model weights
        patch_unifusion.load_state_dict(ckpt["patch_unifusion"])
        main_fusers_sd = ckpt.get("main_fusers", None)
        if main_fusers_sd is None:
            raise KeyError("Checkpoint is missing required key: main_fusers")
        if len(main_fusers_sd) != len(main_fusers):
            raise RuntimeError(
                f"main_fusers count mismatch in checkpoint: ckpt={len(main_fusers_sd)} expected={len(main_fusers)}"
            )
        for i, sd in enumerate(main_fusers_sd):
            main_fusers[i].load_state_dict(sd)

        # Load optimizer and lr_scheduler state
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

        # Restore training state
        start_epoch = ckpt["epoch"] + 1  # Resume from next epoch
        global_step = ckpt["global_step"]

        if accelerator.is_main_process:
            print(f"[Resume] Restored epoch={ckpt['epoch']}, global_step={global_step}")
            print(f"[Resume] Starting from epoch {start_epoch}")

        del ckpt

    progress_bar = None
    if tqdm is not None:
        # If training is indefinite (no max_train_steps), show an epoch-scoped progress bar instead.
        if max_train_steps > 0:
            progress_bar = tqdm(
                total=max_train_steps,
                initial=global_step,
                desc="Steps",
                disable=not accelerator.is_main_process,
                dynamic_ncols=True,
                mininterval=1.0,
            )
        else:
            progress_bar = tqdm(
                total=num_update_steps_per_epoch,
                initial=0,
                desc="Epoch Steps",
                disable=not accelerator.is_main_process,
                dynamic_ncols=True,
                mininterval=1.0,
            )

    did_grad_check = False
    epoch_iter = range(start_epoch, max_epochs) if max_epochs > 0 else itertools.count(start_epoch)
    for epoch in epoch_iter:
        patch_unifusion.train(True)
        for f in main_fusers:
            f.train(True)
        epoch_step = 0
        if progress_bar is not None and max_train_steps == 0:
            progress_bar.reset(total=num_update_steps_per_epoch)
            progress_bar.set_description(f"Epoch {epoch + 1:03d}")

        last_batch: Optional[Dict] = None
        last_batch_idx = 0
        for batch_idx, batch in enumerate(train_loader):
            with accelerator.accumulate(main_unet):
                # Unpack and move tensors
                image = _to_bchw(batch["image"].to(device, dtype=weight_dtype))  # [-1,1] BCHW
                mask = _to_bchw(batch["mask"].to(device, dtype=weight_dtype))  # {0,1} BCHW (1=inpaint)
                masked_image = _to_bchw(batch["masked_image"].to(device, dtype=weight_dtype))  # [-1,1] BCHW

                global_image = _to_bchw(batch["global_image"].to(device, dtype=weight_dtype))
                global_image_full = _to_bchw(batch["global_image_full"].to(device, dtype=weight_dtype))
                global_mask = _to_bchw(batch["global_mask"].to(device, dtype=weight_dtype))

                ref_window_bbox = batch["ref_window_bbox"].to(device, dtype=weight_dtype)  # [B,1,4]

                ref_boxes = batch["ref_boxes"].to(device, dtype=weight_dtype)  # [B,N,4]
                ref_masks = batch["ref_masks"].to(device, dtype=weight_dtype)  # [B,N]
                ref_pos = batch["ref_positive_embeddings"].to(device, dtype=weight_dtype)  # [B,N,768]

                B = int(image.shape[0])

                # Text-only dropout for CFG readiness (per-sample)
                if args.p_cfg_drop > 0.0:
                    is_uncond_text = torch.rand(B, device=device) < float(args.p_cfg_drop)
                else:
                    is_uncond_text = torch.zeros(B, device=device, dtype=torch.bool)

                local_prompts: List[str] = list(batch["txt"])
                for i, drop in enumerate(is_uncond_text.tolist()):
                    if drop:
                        local_prompts[i] = ""
                global_prompts: List[str] = list(batch["global_prompt"])

                with accelerator.autocast():
                    # Encode prompts (frozen)
                    prompt_embeds = _encode_prompts(
                        tokenizer, text_encoder, local_prompts, device=device, dtype=weight_dtype
                    )
                    # Encode images to latents (frozen)
                    latents = _encode_vae(vae, image)
                    masked_latents = _encode_vae(vae, masked_image)

                    # Downsample masks to latent resolution
                    mask_latent = F.interpolate(mask[:, :1], size=latents.shape[-2:], mode="nearest")

                    # Timesteps (same t for both branches)
                    t = torch.randint(
                        0,
                        int(noise_scheduler.config.num_train_timesteps),
                        (B,),
                        device=device,
                        dtype=torch.long,
                    )

                    # Main noisy latents
                    noise = torch.randn_like(latents)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, t)
                    main_model_input = torch.cat([noisy_latents, mask_latent, masked_latents], dim=1)

                    # ---- Global U-Net bank write (no grad) ----
                    global_writer.clear()
                    main_fuser_injector.clear()

                    # Encode ref prompts (frozen)
                    ref_prompt_embeds = _encode_prompts(
                        tokenizer,
                        text_encoder,
                        global_prompts,
                        device=device,
                        dtype=weight_dtype,
                    )

                    # Encode ref images (frozen)
                    ref_image_latent = _encode_vae(vae, global_image)
                    ref_source_latent = _encode_vae(vae, global_image_full)
                    ref_mask_latent = F.interpolate(global_mask[:, :1], size=ref_image_latent.shape[-2:], mode="nearest")

                    with torch.no_grad():
                        ref_noise = torch.randn_like(ref_source_latent)
                        ref_noisy = noise_scheduler.add_noise(ref_source_latent, ref_noise, t)
                        ref_model_input = torch.cat([ref_noisy, ref_mask_latent, ref_image_latent], dim=1)

                        cross_attention_kwargs = {
                            "instdiff": {
                                "boxes": ref_boxes,
                                "positive_embeddings": ref_pos,
                                "masks": ref_masks,
                            }
                        }

                        _ = global_unet(
                            ref_model_input,
                            t,
                            encoder_hidden_states=ref_prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                        ).sample

                    # Pack writer bank -> tuple per block (concat lists if any)
                    bank_tokens_by_block: List[torch.Tensor] = []
                    for i in range(global_writer.num_blocks):
                        bank_list = global_writer.bank.get(i, [])
                        if not bank_list:
                            raise RuntimeError(f"Global U-Net bank missing for block {i}.")
                        bank_tokens_by_block.append(torch.cat(bank_list, dim=1) if len(bank_list) > 1 else bank_list[0])

                    # Feed global U-Net banks to main fusers (concat inside fuser; no linear on bank tokens).
                    main_fuser_injector.update({i: bank_tokens_by_block[i] for i in range(len(bank_tokens_by_block))})
                    # Free ungated writer banks early to reduce peak VRAM.
                    global_writer.clear()

                    # ---- Main UNet forward (grad-enabled; weights frozen) ----
                    patch_masks = torch.ones((B, 1), device=device, dtype=weight_dtype)
                    pos_null = (
                        patch_unifusion.null_positive_feature.view(1, 1, -1)
                        .expand(B, 1, -1)
                        .to(device=device, dtype=weight_dtype)
                    )
                    patch_kwargs = {
                        "instdiff": {
                            "boxes": ref_window_bbox,
                            "positive_embeddings": pos_null,
                            "masks": patch_masks,
                        }
                    }
                    noise_pred = main_unet(
                        main_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=patch_kwargs,
                    ).sample
                    main_fuser_injector.clear()

                    # Loss: main-only diffusion loss
                    pred_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")
                    if pred_type == "epsilon":
                        target = noise
                    elif pred_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, t)
                    elif pred_type == "sample":
                        target = latents
                    else:
                        raise ValueError(f"Unsupported prediction_type: {pred_type}")

                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                # One-time gradient flow sanity check (expect alpha grads first; other weights may be 0 at step 0).
                if accelerator.sync_gradients and (not did_grad_check):
                    alpha_params: List[torch.nn.Parameter] = []
                    for f in main_fusers:
                        for name in ("alpha_attn", "alpha_dense"):
                            p = getattr(f, name, None)
                            if isinstance(p, torch.nn.Parameter):
                                alpha_params.append(p)
                    ok_alpha = _has_nonzero_grad(alpha_params)

                    fail = torch.tensor([int(not ok_alpha)], device=device, dtype=torch.int)
                    fail_all = accelerator.gather(fail)
                    if fail_all.max().item() > 0:
                        if accelerator.is_main_process:
                            print("[grad-check][fail] nonzero grads required for main fuser alphas")
                        raise RuntimeError("Gradient flow check failed for main fuser adapters.")
                    did_grad_check = True

                if accelerator.sync_gradients:
                    if args.gradient_clip_norm and args.gradient_clip_norm > 0.0:
                        accelerator.clip_grad_norm_(trainable_params, args.gradient_clip_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            if accelerator.is_main_process:
                last_batch = batch
                last_batch_idx = int(batch_idx)

            # Periodic batch image logging (match stage-1: every 2000 batches + first batch)
            if (
                accelerator.is_main_process
                and (not bool(args.no_batch_image_logging))
                and int(args.batch_image_log_frequency) > 0
                and (int(batch_idx) % int(args.batch_image_log_frequency) == 0)
            ):
                try:
                    _log_global_local_images(
                        args=args,
                        root_dir=output_dir,
                        device=device,
                        weight_dtype=weight_dtype,
                        accelerator=accelerator,
                        vae=vae,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        noise_scheduler=noise_scheduler,
                        global_unet=global_unet,
                        main_unet=main_unet,
                        global_writer=global_writer,
                        main_fuser_injector=main_fuser_injector,
                        patch_unifusion=patch_unifusion,
                        generator=image_log_generator,
                        batch=batch,
                        epoch=int(epoch),
                        batch_idx=int(batch_idx),
                        global_step=int(global_step),
                        include_index_in_filename=False,
                    )
                except Exception as e:
                    print(f"[image-log][warn] batch logging failed: {e}")

            # Logging (main process only)
            if accelerator.sync_gradients:
                lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, "get_last_lr") else [args.lr]
                lr0 = float(lr[0]) if isinstance(lr, (list, tuple)) else float(lr)

                if progress_bar is not None:
                    progress_bar.update(1)
                    if global_step % args.log_every_n_steps == 0:
                        progress_bar.set_postfix(
                            step=f"{global_step:07d}",
                            ep_step=f"{epoch_step:06d}",
                            loss=f"{loss.item():.6f}",
                            lr=f"{lr0:.2e}",
                        )
                else:
                    if accelerator.is_main_process and (global_step % args.log_every_n_steps == 0):
                        print(
                            f"[epoch {epoch + 1:03d} ep_step {epoch_step:06d} step {global_step:07d}] "
                            f"loss={loss.item():.6f} lr={lr0:.2e}"
                        )
                if wandb_run is not None and accelerator.is_main_process and (global_step % args.log_every_n_steps == 0):
                    try:
                        import wandb  # type: ignore

                        log_dict = {
                            "train/loss": float(loss.item()),
                            "train/lr": float(lr0),
                            "train/epoch": int(epoch + 1),
                            "train/epoch_step": int(epoch_step),
                        }

                        # Gradient norms for trainable modules (match stage-1 logging)
                        try:
                            # patch_unifusion gradient norm
                            uni_norm, uni_ratio, uni_nan = _grad_norm_and_stats(patch_unifusion.parameters())
                            log_dict["train/grad_norm_patch_unifusion"] = uni_norm
                            log_dict["train/grad_nonzero_ratio_patch_unifusion"] = uni_ratio

                            # main_fusers gradient norm
                            fuser_params = [p for f in main_fusers for p in f.parameters()]
                            fuser_norm, fuser_ratio, fuser_nan = _grad_norm_and_stats(fuser_params)
                            log_dict["train/grad_norm_main_fusers"] = fuser_norm
                            log_dict["train/grad_nonzero_ratio_main_fusers"] = fuser_ratio

                            # Combined NaN flag
                            log_dict["train/has_nan_grad"] = float(uni_nan or fuser_nan)
                        except Exception:
                            pass

                        wandb.log(log_dict, step=int(global_step))
                    except Exception as e:
                        print(f"[wandb][warn] log failed: {e}")
                epoch_step += 1
                global_step += 1
                if max_train_steps > 0 and global_step >= max_train_steps:
                    break

        # Epoch-end image logging (match stage-1 EpochImageLogger)
        if accelerator.is_main_process and (not bool(args.no_epoch_image_logging)) and last_batch is not None:
            try:
                _log_global_local_images(
                    args=args,
                    root_dir=output_dir,
                    device=device,
                    weight_dtype=weight_dtype,
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    noise_scheduler=noise_scheduler,
                    global_unet=global_unet,
                    main_unet=main_unet,
                    global_writer=global_writer,
                    main_fuser_injector=main_fuser_injector,
                    patch_unifusion=patch_unifusion,
                    generator=image_log_generator,
                    batch=last_batch,
                    epoch=int(epoch),
                    batch_idx=int(last_batch_idx),
                    global_step=int(global_step),
                    include_index_in_filename=True,
                )
            except Exception as e:
                print(f"[image-log][warn] epoch logging failed: {e}")

        # Save checkpoint every N epochs
        if accelerator.is_main_process and ((epoch + 1) % int(args.save_every_n_epochs) == 0):
            ckpt = {
                "patch_unifusion": patch_unifusion.state_dict(),
                "main_fusers": [f.state_dict() for f in main_fusers],
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "args": vars(args),
            }

            out_path = ckpt_dir / f"main_only_epoch{epoch + 1:04d}.pt"
            torch.save(ckpt, out_path)
            print(f"[save] {out_path}")

        if max_train_steps > 0 and global_step >= max_train_steps:
            break

    if progress_bar is not None:
        progress_bar.close()
    if wandb_run is not None and accelerator.is_main_process:
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    _main()
