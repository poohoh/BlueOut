"""
Training script: Stage 1 global U-Net for progressive outpainting
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
from datetime import datetime
import shutil

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from ldm.modules.controlnet.logger import EpochImageLogger, ImageLogger

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

# DataLoader (ProOut progressive conditioning for training) — use multi-dataset loader
from data.dataloader_train_global import create_progressive_condition_dataloader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage 1 global U-Net (progressive overlap)")

    # model/config
    p.add_argument("--config", type=str, default="configs/outpainting/stage1_global_unet.yaml",
                   help="YAML config for Stage 1 global U-Net training")
    p.add_argument("--ckpt", type=str, default="checkpoints/pretrained/inpainting/sd-v1-5-inpainting.ckpt",
                   help="Base SD v1.5 inpainting checkpoint to initialize from")
    p.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"],
                   help="Training precision")

    # data
    p.add_argument("--data_root", type=str, default="datasets", help="dataset root")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size")
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    p.add_argument("--image_size", type=int, default=512, help="Training image size (square)")
    p.add_argument("--use_2d_only", action="store_true", help="Filter to 2D categories only")
    # Default: instance attention masks OFF. Enable with --use_instance_attn_mask.
    p.add_argument('--use_instance_attn_mask', dest='use_instance_attn_mask', action='store_true',
                   help='Enable per-instance attention masks for InstanceDiffusion grounding')
    p.set_defaults(use_instance_attn_mask=False)
    p.add_argument(
        "--bbox_drop_p",
        type=float,
        default=0.1,
        help="CFG-style random drop probability for bbox grounding (default: 0.1).",
    )
    
    # dataloader memory/speed knobs
    p.add_argument("--prefetch_factor", type=int, default=1,
                   help="DataLoader prefetch_factor (effective only when num_workers>0)")
    p.add_argument("--persistent_workers", action="store_true", default=False,
                   help="Enable persistent_workers when using multiple workers (NOT recommended for memory stability)")
    p.add_argument("--no_pin_memory", action="store_false", dest="pin_memory", default=True,
                   help="Disable DataLoader pin_memory to reduce RAM usage")

    # training
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_epochs", type=int, default=100, help="Number of training epochs")
    p.add_argument("--accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--limit_train_batches", type=float, default=1.0,
                   help="Limit training batches (float in (0,1] or int)")
    p.add_argument("--outdir", type=str, default="results/train", help="Base training output dir")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to a full-state PL checkpoint to resume from (e.g., last.ckpt)")
    # Lightweight resume: module-wise weight loading (Stage 1 global U-Net is required; extras optional).
    p.add_argument("--resume_global", type=str, default=None,
                   help="Path to stage1_epochXXXX.pt")
    p.add_argument("--resume_extras", type=str, default=None,
                   help="(Optional) Path to unet_extras-epochXXXX.pt")
    p.add_argument("--resume_warmup_steps", type=int, default=500,
                   help="Warmup steps to apply when resuming from weights-only checkpoints (default: 500)")

    # runtime
    p.add_argument("--devices", type=int, default=3, help="Number of GPUs/CPUs to use")
    p.add_argument("--num_nodes", type=int, default=1)
    p.add_argument("--strategy", type=str, default="auto", help="PL strategy: auto|ddp|ddp_find_unused_parameters_false|deepspeed|fsdp")
    p.add_argument("--log_every_n_steps", type=int, default=50, help="PL Trainer logging interval in steps")
    # For Stage 1 global U-Net training, disable expensive image logging by default.
    p.add_argument("--no_epoch_image_logging", action="store_true", default=False,
                   help="Disable per-epoch image logging to reduce overhead")
    p.add_argument("--no_batch_image_logging", action="store_true", default=False,
                   help="Disable per-N-batch image logging to reduce overhead")
    p.add_argument("--light_logging", action="store_true", default=False,
                   help="Skip extra per-step grad/memory/time logs inside the model")
    # interrupt behavior
    p.add_argument("--fast_interrupt", action="store_true", default=False,
                   help="Exit immediately on Ctrl+C without Lightning's graceful shutdown (opt-in; prevents RAM spikes)")

    # wandb
    p.add_argument("--wandb_project", type=str, default="outpainting-global-unet", help="Wandb project name")
    p.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (team/username)")
    p.add_argument("--wandb_offline", action="store_true", default=False,
                   help="Run wandb in offline mode")

    # overrides
    g = p.add_argument_group("overrides")
    g.add_argument("--freeze_unet", action="store_true", default=False,
                   help="Override config: set model.params.train_unet=False")
    g.add_argument("--train_unet", action="store_true", default=False,
                   help="Override config: set model.params.train_unet=True")

    return p.parse_args()


def map_precision(p: str) -> str:
    # Map to PL precision strings for best compatibility across versions
    if p == "fp16":
        return "16-mixed"
    if p == "bf16":
        return "bf16-mixed"
    return "32-true"


def build_model(
    cfg_path: str,
    ckpt_path: str,
    overrides: dict | None = None,
    skip_base_ckpt_init: bool = False,
) -> pl.LightningModule:
    cfg = OmegaConf.load(cfg_path)
    # Apply simple param overrides before instantiation
    if overrides:
        params = cfg.get('model', {}).get('params', {})
        # known explicit toggle
        if 'train_unet' in overrides:
            params['train_unet'] = bool(overrides['train_unet'])
        # apply any additional overrides (e.g., light_logging)
        for k, v in overrides.items():
            if k == 'train_unet':
                continue
            params[k] = v
        cfg.model.params = params
    model = instantiate_from_config(cfg.model)

    # set learning rate
    model.learning_rate = float(cfg.model.base_learning_rate)

    # Initialize from SD v1.5 inpainting weights (UNet + VAE + CLIP)
    if ckpt_path and Path(ckpt_path).exists():
        print(f"[Init] Loading base checkpoint: {ckpt_path}")
        # Use DDPM's loader to handle nested keys gracefully.
        # Explicitly ignore any accidental global-model keys if present in the checkpoint.
        ignore_keys = [
            'global_model.',
            'global_model',
        ]
        model.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
    elif skip_base_ckpt_init:
        print("[Init] Skipping base checkpoint init because full-state resume will restore model weights")
    else:
        raise ValueError("[Init] Base checkpoint not found or not provided; training from config init")

    # Set up grounding_tokenizer_input for CFG training (10% random drop)
    print("[Init] Setting up grounding_tokenizer_input for CFG training")
    grounding_tokenizer_input = instantiate_from_config(cfg.grounding_tokenizer_input)
    model.global_model.inner.grounding_tokenizer_input = grounding_tokenizer_input
    print("[Init] grounding_tokenizer_input attached to global_model.inner")

    return model


def main():
    # PL-DDP training (no accelerate)
    args = parse_args()

    pl.seed_everything(args.seed)

    precision = map_precision(args.precision)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}, precision: {precision}")

    # Enable Tensor Core/TF32 matmuls for Ampere+ GPUs (A6000) to speed up training.
    # Safe for mixed precision; numerics remain stable for diffusion.
    try:
        if torch.cuda.is_available():
            # Prefer PyTorch 2.x API
            try:
                torch.set_float32_matmul_precision('high')  # enables TF32 matmul paths
                print("[Perf] set_float32_matmul_precision('high') enabled")
            except Exception as e:
                print(f"[Perf][warn] set_float32_matmul_precision not available: {e}")
            # Backends toggles for maximal coverage
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("[Perf] TF32 allowed for CUDA matmul and cuDNN")
    except Exception as e:
        print(f"[Perf][warn] TF32 setup skipped: {e}")

    # Data
    train_loader, _ = create_progressive_condition_dataloader(
        dataset_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        image_size=args.image_size,
        use_2d_only=args.use_2d_only,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        use_instance_attn_mask=args.use_instance_attn_mask,
    )

    # Model
    # Build model with optional overrides
    overrides = {}
    args.resume_from = (
        str(args.resume_from).strip() if args.resume_from is not None else None
    ) or None
    requested_full_resume = args.resume_from is not None
    requested_light_resume = (args.resume_global is not None) or (args.resume_extras is not None)
    if requested_full_resume and requested_light_resume:
        raise ValueError("--resume_from cannot be combined with --resume_global/--resume_extras")
    if requested_full_resume and not Path(args.resume_from).is_file():
        raise FileNotFoundError(f"[Resume] Full-state checkpoint not found: {args.resume_from}")
    if args.freeze_unet and args.train_unet:
        raise ValueError("--freeze_unet and --train_unet are mutually exclusive")
    if args.freeze_unet:
        overrides['train_unet'] = False
    elif args.train_unet:
        overrides['train_unet'] = True
    # pass light_logging to model to reduce per-step metric logging
    if args.light_logging:
        overrides['light_logging'] = True
    # When resuming from weights-only snapshots, start with a short warmup to avoid optimizer shock.
    if requested_light_resume:
        overrides['warmup_steps'] = int(max(0, args.resume_warmup_steps))

    model = build_model(
        args.config,
        args.ckpt,
        overrides=overrides if overrides else None,
        skip_base_ckpt_init=requested_full_resume,
    )

    # Enable CFG-style random drop for bbox-only grounding (InstanceDiffusion behavior).
    # This mirrors previous_works/InstanceDiffusion where the UNet randomly replaces
    # grounding_input with a null input with p=0.1 during training.
    try:
        p_drop = float(args.bbox_drop_p)
        if p_drop < 0.0 or p_drop > 1.0:
            raise ValueError(f"--bbox_drop_p must be in [0,1], got {p_drop}")
        model.global_model.inner.grounding_dropout_p = p_drop
        print(f"[Init] global_model.inner.grounding_dropout_p set to {p_drop:.3f}")
    except Exception as e:
        print(f"[Init][warn] Failed to set grounding_dropout_p: {e}")

    # Show trainable parameter counts for quick verification
    def _count_params(module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable
    try:
        if hasattr(model, "model") and model.model is not None:
            unet_total, unet_train = _count_params(model.model)
            print(f"[Params] UNet total={unet_total:,} trainable={unet_train:,}")
        else:
            print("[Params] Main UNet removed in Stage 1 global U-Net training")

        global_total, global_train = _count_params(model.global_model)
        print(f"[Params] Stage1GlobalUNet total={global_total:,} trainable={global_train:,}")
        # bank-gating fusers live outside both UNet and the Stage 1 global U-Net as separate modules
        if hasattr(model, "bank_gating_fusers"):
            bank_total, bank_train = _count_params(model.bank_gating_fusers)
            print(f"[Params] BankGatingFusers total={bank_total:,} trainable={bank_train:,}")
        # Optional: patch-window UniFusion tokenizer (also separate from main/ref UNet)
        if hasattr(model, "patch_unifusion"):
            patch_total, patch_train = _count_params(model.patch_unifusion)
            print(f"[Params] PatchUniFusion total={patch_total:,} trainable={patch_train:,}")
    except Exception as e:
        print(f"[Params][warn] Failed to count parameters: {e}")

    # Session directory: results/train/<RUN_NAME or timestamp>/
    # - Use RUN_NAME env to keep a single folder across ranks/processes
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = os.environ.get("RUN_NAME", "").strip() or timestamp
    session_dir = Path(args.outdir) / run_id
    ckpt_dir = session_dir / "ckpt"
    log_dir = session_dir / "log"
    
    # Create directories only on rank 0
    rank_str = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    is_main = str(rank_str) == "0"
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Info] Training session: {session_dir}")
        print(f"[Info] - ckpt: {ckpt_dir}")
        print(f"[Info] - log:  {log_dir}")

        # Copy config file to session directory
        config_src = Path(args.config)
        if not config_src.exists():
            raise FileNotFoundError(f"Config file not found: {config_src}")
        config_dst = session_dir / config_src.name
        try:
            # Avoid SameFileError when resuming with a config already inside the session dir.
            if config_src.resolve() == config_dst.resolve():
                print(f"[Info] Config already in session dir: {config_dst}")
            else:
                shutil.copy2(str(config_src), str(config_dst))
                print(f"[Info] Config copied to: {config_dst}")
        except Exception as e:
            raise RuntimeError(f"[Info] Failed to copy config to session dir: {e}") from e

    # Enable cuDNN benchmark for fixed-size inputs (speeds up Conv algorithm selection)
    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
            print("[Perf] cuDNN benchmark enabled")
        except Exception as e:
            print(f"[Perf][warn] cuDNN benchmark setup skipped: {e}")

    # ModelCheckpoint: save only the latest checkpoint every epoch (no top-k)
    model_checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_on_exception=False,
        save_last=True,
        save_top_k=0,
        # Save full training state so last.ckpt can be used for true Trainer resume.
        save_weights_only=False,
    )

    # Export Stage 1 global U-Net weights periodically (for inference convenience)
    from pytorch_lightning.callbacks import Callback

    class ExportGlobalUNet(Callback):
        def __init__(self, out_dir: Path, every_n_epochs: int = 1, is_main: bool = True):
            self.out_dir = Path(out_dir)
            self.every_n_epochs = every_n_epochs
            self.is_main = is_main
            if self.is_main:
                self.out_dir.mkdir(parents=True, exist_ok=True)

        def on_train_epoch_end(self, trainer, pl_module):
            # Export checkpoints with 1-based epoch numbering so filenames match
            # image_log/.../epoch_XXXX and human-facing epoch counts.
            epoch0 = trainer.current_epoch
            # Allow an epoch offset (e.g., when resuming from a previous run)
            # so that new checkpoints do not overwrite earlier epoch files.
            try:
                epoch_offset = int(os.environ.get("EPOCH_OFFSET", "0"))
            except Exception as e:
                epoch_offset = 0
                print(f"[Export][warn] Failed to parse EPOCH_OFFSET env var, using 0: {e}")
            save_epoch = epoch0 + epoch_offset + 1
            if (epoch0 + 1) % self.every_n_epochs != 0:
                return
            if trainer.is_global_zero:
                # Save Stage 1 global U-Net
                try:
                    if hasattr(pl_module, "global_model") and pl_module.global_model is not None:
                        global_state = pl_module.global_model.state_dict()
                    else:
                        global_state = {}
                    global_path = self.out_dir / f"stage1_epoch{save_epoch:04d}.pt"
                    torch.save(global_state, str(global_path))
                    print(f"[Export] Saved Stage 1 global U-Net weights at {global_path}")
                except Exception as e:
                    print(f"[Export][warn] Failed to save Stage 1 global U-Net: {e}")

                # Save UNet only if it's trainable (restore original behavior)
                try:
                    train_unet = getattr(pl_module, 'train_unet', True)
                    if train_unet:
                        # pl_module.model is DiffusionWrapper; export its inner diffusion_model
                        unet_module = getattr(pl_module, 'model', None)
                        if unet_module is not None and hasattr(unet_module, 'diffusion_model'):
                            unet_state = unet_module.diffusion_model.state_dict()
                        else:
                            # fallback to whole wrapper
                            unet_state = unet_module.state_dict() if unet_module is not None else {}
                        unet_path = self.out_dir / f"unet-epoch{save_epoch:04d}.pt"
                        torch.save(unet_state, str(unet_path))
                        print(f"[Export] Saved UNet weights at {unet_path}")
                except Exception as e:
                    print(f"[Export][warn] Failed to save UNet: {e}")

                # Save main-UNet extras (patch UniFusion + bank-gating fusers) even if UNet is frozen
                try:
                    extras = {}
                    if hasattr(pl_module, 'patch_unifusion') and pl_module.patch_unifusion is not None:
                        extras['patch_unifusion'] = pl_module.patch_unifusion.state_dict()
                    if hasattr(pl_module, 'bank_gating_fusers') and pl_module.bank_gating_fusers is not None:
                        extras['bank_gating_fusers'] = pl_module.bank_gating_fusers.state_dict()
                    if extras:
                        extras_path = self.out_dir / f"unet_extras-epoch{save_epoch:04d}.pt"
                        torch.save(extras, str(extras_path))
                        print(f"[Export] Saved UNet extras at {extras_path}")
                except Exception as e:
                    print(f"[Export][warn] Failed to save UNet extras: {e}")

    # Always export module weights every epoch.
    # `--no_epoch_image_logging` should only affect image logging, not checkpoint/weight export.
    export_callbacks = [ExportGlobalUNet(out_dir=ckpt_dir, every_n_epochs=1, is_main=is_main)]
    # LR monitor only on main rank (requires a logger)
    lr_monitor = LearningRateMonitor(logging_interval="step") if is_main else None

    # Create logger only on main rank to avoid multi-run folder creation
    if is_main:
        # Set wandb mode (online/offline)
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"

        logger_kwargs = {
            "project": args.wandb_project,
            "name": run_id,
            "save_dir": str(log_dir),
            "log_model": False,  # Don't auto-upload checkpoints to save bandwidth
        }
        if args.wandb_entity:
            logger_kwargs["entity"] = args.wandb_entity

        logger = WandbLogger(**logger_kwargs)

        # Log training configuration to wandb
        logger.experiment.config.update({
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "learning_rate": model.learning_rate,
            "weight_decay": model.weight_decay,
            "gradient_clip_val": model.gradient_clip_val,
            "image_size": args.image_size,
            "precision": args.precision,
            "accum": args.accum,
            "devices": args.devices,
            "num_workers": args.num_workers,
            "config_file": args.config,
            "base_checkpoint": args.ckpt,
            "train_unet": getattr(model, 'train_unet', False),
            "use_2d_only": args.use_2d_only,
            "use_instance_attn_mask": args.use_instance_attn_mask,
            "seed": args.seed,
        })
    else:
        logger = False

    # PNG-only epoch image logger (5 samples per epoch, grouped by epoch)
    # Reduce image logging to save memory (sampling costs VRAM)
    epoch_img_logger = EpochImageLogger(max_images=10, disabled=bool(args.no_epoch_image_logging), log_images_kwargs={
        # generate samples once per epoch alongside inputs
        'sample': True,
        'ddim_steps': 30,
        # mimic inference CFG/uncond handling: provide a negative prompt as uc label
        'unconditional_guidance_label': "ugly, nsfw, worst quality, watermark, signature, logo",
        'unconditional_guidance_scale': 2.5,
        # keep only the most relevant keys to reduce confusion
        # Stage 1: keep global U-Net-related outputs and layouts.
        'return_keys': [
            'ref_hint_image',
            'ref_global_mask',
            'global_gt',
            'ref_samples',
            'layout_blank',
            'layout_on_global_gt',
            'layout_on_ref_samples',
        ]
    }, save_text_prompts=True, prompts_key='global_prompt', prompts_filename='global_prompts.txt')
    if not args.no_epoch_image_logging:
        export_callbacks.append(epoch_img_logger)

    # Also log periodically per N batches to save more samples during training
    # Save every 2000 batches to limit IO overhead
    batch_img_logger = ImageLogger(
        disabled=bool(args.no_batch_image_logging),
        batch_frequency=2000,   # log every 2k iterations
        max_images=10,
        log_first_step=True,    # only once at batch_idx==0 due to log_on_batch_idx=True
        log_on_batch_idx=True,  # avoid repeated logging while global_step==0 with grad accumulation
        log_images_kwargs={
            'sample': True,
            'ddim_steps': 30,
            'unconditional_guidance_scale': 2.5,
            'unconditional_guidance_label': "ugly, nsfw, worst quality, watermark, signature, logo",
            # Stage 1: keep global U-Net-related outputs and layouts.
            'return_keys': [
                'ref_hint_image',
                'ref_global_mask',
                'global_gt',
                'ref_samples',
                'layout_blank',
                'layout_on_global_gt',
                'layout_on_ref_samples',
            ]
        }
    )
    if not args.no_batch_image_logging:
        export_callbacks.append(batch_img_logger)


    # Strategy selection: enforce DDP on multi-GPU. 모든 파라미터가 매 iteration에 사용되므로 False로 설정
    if torch.cuda.is_available() and int(args.devices) > 1:
        from pytorch_lightning.strategies import DDPStrategy
        chosen_strategy = DDPStrategy(find_unused_parameters=False)
    else:
        # Single GPU/CPU: always use 'auto' to avoid passing None
        chosen_strategy = args.strategy if args.strategy not in (None, "", "auto") else "auto"

    # Trainer
    _callbacks = [c for c in [lr_monitor, model_checkpoint, *export_callbacks] if c is not None]

    # --- Lightweight resume of trainable modules (Stage 1 global U-Net + UNet extras) ---
    # Load these BEFORE constructing the Trainer to avoid PL trying to restore states.
    def _maybe_resume_trainables():
        # Determine if user requested lightweight resume
        requested = (args.resume_global is not None) or (args.resume_extras is not None)
        if not requested:
            return False
        
        # Stage 1 global U-Net weights are required for a meaningful weights-only resume.
        if not args.resume_global:
            raise ValueError("--resume_global is required for weights-only resume.")
        global_path = Path(args.resume_global)
        extras_path = Path(args.resume_extras) if args.resume_extras else None

        # Only global rank 0 performs file I/O before DDP construction.
        # Other ranks rely on DDP's initial parameter broadcast to sync weights.
        if is_main:
            if not global_path.is_file():
                raise FileNotFoundError(f"[Resume] Stage 1 global U-Net weights not found: {global_path}")

            global_sd = torch.load(str(global_path), map_location="cpu")
            model.global_model.load_state_dict(global_sd, strict=True)
            print(f"[Resume] Successfully loaded Stage 1 global U-Net from {global_path.name}")

            # Optional: load extra trainable modules if provided and present in the model.
            if extras_path is not None:
                if not extras_path.is_file():
                    raise FileNotFoundError(f"[Resume] UNet extras weights not found: {extras_path}")
                extras_sd = torch.load(str(extras_path), map_location="cpu")
                if not isinstance(extras_sd, dict):
                    raise ValueError(f"[Resume] Unexpected extras checkpoint format: {extras_path}")

                required = ['patch_unifusion', 'bank_gating_fusers']
                missing = [k for k in required if k not in extras_sd]
                if missing:
                    raise KeyError(f"[Resume] Missing required keys in extras checkpoint {extras_path.name}: {missing}")

                has_patch = hasattr(model, 'patch_unifusion') and model.patch_unifusion is not None
                has_bank = hasattr(model, 'bank_gating_fusers') and model.bank_gating_fusers is not None
                if not (has_patch and has_bank):
                    print(
                        "[Resume][warn] --resume_extras provided but model does not expose "
                        "patch_unifusion/bank_gating_fusers; skipping extras load."
                    )
                else:
                    model.patch_unifusion.load_state_dict(extras_sd['patch_unifusion'], strict=True)
                    model.bank_gating_fusers.load_state_dict(extras_sd['bank_gating_fusers'], strict=True)
                    print(f"[Resume] Successfully loaded UNet extras from {extras_path.name}")
        else:
            print("[Resume] Non-zero rank: waiting for DDP broadcast of parameters from rank0.")

        return True

    used_light_resume = _maybe_resume_trainables()

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=precision,
        accumulate_grad_batches=args.accum,
        strategy=chosen_strategy,
        # Gradient clipping for training stability (global norm)
        gradient_clip_val=model.gradient_clip_val,
        gradient_clip_algorithm="norm",
        callbacks=_callbacks,
        logger=logger,
        # Keep Lightning checkpointing enabled (ModelCheckpoint active)
        enable_checkpointing=True,
        enable_model_summary=False,  # PL model summary misreports under 16‑mixed; reduce confusion
        limit_train_batches=args.limit_train_batches,
        log_every_n_steps=int(max(1, args.log_every_n_steps)),
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )

    # Optional: Replace Lightning's SIGINT/SIGTERM handlers to avoid graceful teardown
    # which can allocate memory and freeze on some systems.
    if args.fast_interrupt or os.environ.get("FAST_INTERRUPT", "0") == "1":
        try:
            import signal
            def _fast_exit(signum, frame):
                try:
                    print("[Interrupt] Fast exit requested — skipping PL teardown/checkpoint.")
                except Exception:
                    pass
                # Immediate exit avoids dataloader worker joins and RAM spikes
                os._exit(130)
            signal.signal(signal.SIGINT, _fast_exit)
            signal.signal(signal.SIGTERM, _fast_exit)
            if is_main:
                print("[Interrupt] Fast interrupt mode enabled (Ctrl+C exits immediately).")
        except Exception as e:
            if is_main:
                print(f"[Interrupt][warn] Failed to install fast interrupt handler: {e}")

    # Fit — full-state resume uses ckpt_path, while module-wise resume is applied above.
    trainer.fit(model, train_dataloaders=train_loader, ckpt_path=args.resume_from)


if __name__ == "__main__":
    main()
