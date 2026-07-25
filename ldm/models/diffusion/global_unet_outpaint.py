"""
Stage 1 global U-Net + Inpainting
LatentInpaintDiffusion + Stage 1 global U-Net attention fusion
"""

from contextlib import nullcontext
from typing import Dict, Optional, Tuple
import time

import torch
from einops import rearrange
from torch.nn import functional as F
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from ldm.models.diffusion.ddpm import LatentInpaintDiffusion
from ldm.models.diffusion.ddim_global import DDIMGlobalSampler
from ldm.util import instantiate_from_config
from ldm.modules.global_unet.global_attention import (
    GlobalAttentionControlLDM,
)
from ldm.modules.global_unet.id_global_attention import (
    GlobalAttentionControlIDWriter,
)
from InstanceDiffusion.ldm.modules.attention import GatedSelfAttentionDense
from InstanceDiffusion.ldm.modules.diffusionmodules.text_grounding_net import UniFusion
from ldm.modules.attention import SpatialTransformer as LDM_SpatialTransformer
from ldm.modules.attention import BasicTransformerBlock as LDM_BTBlock


class GlobalUNetOutpaintDiffusion(LatentInpaintDiffusion):
    """
    LatentInpaintDiffusion + Stage 1 global U-Net

    - conserves inpainting functions (pass mask, masked_image as c_concat)
    - adds Stage 1 global U-Net pass and attention-bank fusion into the base UNet
    """

    # Visualization color palette for bounding boxes
    _VIZ_COLORS = [
        (255, 64, 64), (64, 255, 64), (64, 128, 255), (255, 192, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (0, 192, 255)
    ]

    def __init__(
        self,
        global_stage_config: Dict,
        global_key: str = "global_image",  # default batch key for the global image in batch
        # Injection & training controls
        global_token_scale: float = 1.0,     # global feature strength. 1.0이면 원본 그대로
        save_attention_maps: bool = False,
        # Optimizer controls for global U-Net
        train_global_unet: bool = True,
        train_unet: bool = True,                # 메인 UNet도 기본 학습 (AnimateAnyone 스타일)
        global_lr_scale: float = 1.0,
        # Additional supervised loss on Stage 1 global U-Net (predict ref-noise from noisy ref source)
        global_noise_loss_weight: float = 1.0,
        # Pretrained weights for global U-Net (ID 전용 권장)
        global_pretrained_ckpt: Optional[str] = None,
        # InstanceDiffusion module-only checkpoint (fuser/UniFusion/ScaleU). Required for ID.
        global_modules_ckpt: Optional[str] = None,
        # DC-AE integration (global branch only)
        global_use_dc_ae: bool = False,
        global_dc_ae_repo: Optional[str] = None,
        global_dc_ae_subfolder: Optional[str] = None,
        # When loading DC-AE from a single-file checkpoint, use a diffusers config
        # repo to supply the architecture. Default is SANA f32c32 v1.0 diffusers.
        global_dc_ae_config_repo: Optional[str] = None,
        global_scale_factor: Optional[float] = None,
        # Logging controls
        light_logging: bool = False,
        # LR scheduler (InstanceDiffusion-style warmup)
        scheduler_type: Optional[str] = None,   # 'constant' or 'cosine'
        warmup_steps: int = 0,
        total_training_steps: Optional[int] = None,
        # Training parameters (NOT passed to parent DDPM)
        gradient_clip_val: float = 0.0,
        weight_decay: float = 0.0,
        *args,
        **kwargs,
    ):
        # Store training parameters before calling super().__init__()
        # These are used by the training script but not passed to DDPM
        self.gradient_clip_val = float(gradient_clip_val)
        self.weight_decay = float(weight_decay)

        super().__init__(*args, **kwargs)

        # Build Stage 1 global U-Net UNet from config (must be InstanceDiffusion wrapper)
        self.global_model = instantiate_from_config(global_stage_config)
        self.global_key = global_key

        # Attention controllers (lazily initialized on first forward)
        self._global_writer: Optional[GlobalAttentionControlIDWriter] = None
        self._global_reader: Optional[GlobalAttentionControlLDM] = None
        self._global_cfg_enabled: bool = False

        # Controls
        self.global_token_scale = float(global_token_scale)
        self.save_attention_maps = bool(save_attention_maps)
        self.train_global_unet = bool(train_global_unet)
        self.train_unet = bool(train_unet)
        self.global_lr_scale = float(global_lr_scale)
        self.global_noise_loss_weight = float(global_noise_loss_weight)
        self.light_logging = bool(light_logging)
        self.global_pretrained_ckpt = global_pretrained_ckpt
        self.global_modules_ckpt = global_modules_ckpt
        # DC-AE toggles and buffers
        self.global_use_dc_ae = bool(global_use_dc_ae)
        self.global_dc_ae_repo = global_dc_ae_repo
        self.global_dc_ae_subfolder = global_dc_ae_subfolder
        # Keep explicit attribute so YAML `model.params.global_dc_ae_config_repo` is honored.
        # If None, DCAEWrapper will fall back to
        # "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers" internally.
        self.global_dc_ae_config_repo = global_dc_ae_config_repo
        if global_scale_factor is None:
            self.register_buffer('global_scale_factor', torch.tensor(1.0, dtype=torch.float32))
            self._global_scale_calibrated = False
        else:
            self.register_buffer('global_scale_factor', torch.tensor(float(global_scale_factor), dtype=torch.float32))
            self._global_scale_calibrated = True
        self.global_ae = None  # lazy-loaded DC-AE

        # LR scheduler knobs (match InstanceDiffusion semantics)
        self.scheduler_type = scheduler_type
        self.warmup_steps = int(warmup_steps or 0)
        self.total_training_steps = int(total_training_steps) if total_training_steps else None

        # Enforce InstanceDiffusion-only and no-random-init policy for Stage 1 global U-Net
        if not getattr(self.global_model, 'is_instance_diffusion', False):
            raise ValueError(
                "GlobalUNetOutpaintDiffusion now supports only InstanceDiffusion global U-Nets. "
                "Ensure 'global_stage_config' uses IDGlobalUNetWrapper."
            )
        if not self.global_pretrained_ckpt or str(self.global_pretrained_ckpt).strip() == "":
            raise ValueError(
                "GlobalUNetOutpaintDiffusion requires 'global_pretrained_ckpt' to be set. "
                "Provide a valid checkpoint path in config (model.params.global_pretrained_ckpt) "
                "or via your training/inference script."
            )
        # Prepare strict loading plan for Stage 1 global U-Net (InstanceDiffusion wrapper):
        #  - Step 1: load base UNet weights from SD inpainting ckpt into ref.inner
        #  - Step 2: load ID-specific modules (fuser/UniFusion/ScaleU) from `global_modules_ckpt`
        #  - Verify coverage: any expected key not loaded -> raise error
        ref_inner = getattr(self.global_model, 'inner', self.global_model)

        # Classify keys by role so we can verify coverage
        def _classify_global_unet_keys(state_dict_keys: list[str]) -> tuple[set[str], set[str]]:
            mod_keys: set[str] = set()
            base_keys: set[str] = set()
            for k in state_dict_keys:
                if k.startswith('position_net.') or k.startswith('scaleu_b_') or k.startswith('scaleu_s_') or ('.fuser.' in k):
                    mod_keys.add(k)
                else:
                    base_keys.add(k)
            return mod_keys, base_keys

        ref_state = ref_inner.state_dict()
        mod_expected, base_expected = _classify_global_unet_keys(list(ref_state.keys()))

        # Step 1: load base UNet weights from SD v1.5 inpainting
        self._load_global_base_from_ckpt(
            self.global_pretrained_ckpt,
            expected_keys=base_expected
        )

        # Step 2: load ID module-only checkpoint
        if not self.global_modules_ckpt or str(self.global_modules_ckpt).strip() == "":
            raise ValueError(
                "GlobalUNetOutpaintDiffusion requires 'global_modules_ckpt' to be set. "
                "Expected the InstanceDiffusion module-only checkpoint (fuser/UniFusion/ScaleU)."
            )
        self._load_global_modules_from_ckpt(
            self.global_modules_ckpt,
            expected_keys=mod_expected
        )

        print("[GlobalUNet] All parameters loaded and verified successfully!")

        # Reset ID fuser gating scalars so global features start from zero strength.
        # InstanceDiffusion ckpt may carry non-zero alpha_*; we explicitly zero them
        # to let inpainting-UNet coupling grow during fine-tuning.
        zeroed = 0
        for m in ref_inner.modules():
            if isinstance(m, GatedSelfAttentionDense):
                with torch.no_grad():
                    if hasattr(m, "alpha_attn"):
                        m.alpha_attn.zero_()
                    if hasattr(m, "alpha_dense"):
                        m.alpha_dense.zero_()
                zeroed += 1
        if zeroed == 0:
            raise RuntimeError("[GlobalUNet] No GatedSelfAttentionDense modules found to zero alphas.")
        print(f"[GlobalUNet] Zero-initialized alpha_attn/alpha_dense for {zeroed} ID fusers")

        # Freeze Stage 1 global U-Net base UNet weights while keeping ID modules trainable.
        # - base_expected: standard UNet weights (convs, resblocks, etc.)
        # - mod_expected:  ID-specific modules (UniFusion/ScaleU/fusers) that we want to train.
        try:
            frozen_params = 0
            total_params = 0
            trainable_id_params = 0
            fuser_params = 0
            unifusion_params = 0
            scaleu_params = 0

            for name, p in ref_inner.named_parameters():
                total_params += p.numel()
                if name in base_expected:
                    p.requires_grad = False
                    frozen_params += p.numel()
                elif name in mod_expected:
                    # Ensure ID modules (UniFusion/ScaleU/fusers/ScaleU) remain trainable
                    p.requires_grad = True
                    trainable_id_params += p.numel()
                    if ".fuser." in name:
                        fuser_params += p.numel()
                    elif name.startswith("position_net."):
                        unifusion_params += p.numel()
                    elif name.startswith("scaleu_b_") or name.startswith("scaleu_s_"):
                        scaleu_params += p.numel()

            # Sanity checks: base keys must be frozen, module keys must be trainable.
            base_still_trainable = [
                n for n, p in ref_inner.named_parameters()
                if n in base_expected and p.requires_grad
            ]
            mod_not_trainable = [
                n for n, p in ref_inner.named_parameters()
                if n in mod_expected and not p.requires_grad
            ]
            if base_still_trainable:
                raise ValueError(
                    "[GlobalUNet] Some base-UNet parameters in Stage 1 global U-Net are still trainable: "
                    + ", ".join(base_still_trainable[:10])
                )
            if mod_not_trainable:
                raise ValueError(
                    "[GlobalUNet] Some ID-module parameters in Stage 1 global U-Net are not trainable: "
                    + ", ".join(mod_not_trainable[:10])
                )

            # Enforce presence of key ID modules: fusers, UniFusion, ScaleU.
            if fuser_params == 0:
                raise RuntimeError(
                    "[GlobalUNet] No trainable fuser parameters found in Stage 1 global U-Net. "
                    "Check that InstanceDiffusion fusers are present and correctly classified in _classify_global_unet_keys."
                )
            if unifusion_params == 0:
                raise RuntimeError(
                    "[GlobalUNet] No trainable UniFusion (position_net) parameters found in Stage 1 global U-Net. "
                    "Check that position_net.* keys are present and treated as ID modules."
                )
            if scaleu_params == 0:
                raise RuntimeError(
                    "[GlobalUNet] No trainable ScaleU parameters found in Stage 1 global U-Net. "
                    "Check that scaleu_b_*/scaleu_s_* keys are present and treated as ID modules."
                )

            print(
                "[GlobalUNet] Frozen base UNet params in Stage 1 global U-Net: "
                f"{frozen_params:,}/{total_params:,} | "
                f"trainable ID params={trainable_id_params:,} "
                f"(fuser={fuser_params:,}, UniFusion={unifusion_params:,}, ScaleU={scaleu_params:,})"
            )
        except Exception as e:
            raise ValueError(f"Failed to freeze Stage 1 global U-Net base UNet parameters: {e}")

        # Track classifier-free guidance usage (set via helper before sampling)
        self._global_cfg_active = False

        # --- Sampling-time helpers for global-branch denoising (used by DDIMGlobalSampler) ---
        # When enabled, the sampler can override the ref noisy latent and read back
        # the last (ref_noisy, ref_pred) to perform a DDIM-style update identical to
        # main UNet. These are only populated in no_grad() contexts.
        self._global_branch_denoising_enabled: bool = False
        self._global_noisy_override: Optional[torch.Tensor] = None
        self._last_global_noisy: Optional[torch.Tensor] = None
        self._last_global_pred: Optional[torch.Tensor] = None

        # Detect whether LDM UNet blocks use gradient checkpointing.
        # If not, we can safely clear the reader bank at the end of forward
        # to avoid retaining ref tokens across steps (helps prevent OOM).
        try:
            from ldm.modules.attention import BasicTransformerBlock as LDM_BTBlock  # type: ignore
            uses_ckpt = False
            for m in self.model.diffusion_model.modules():
                if isinstance(m, LDM_BTBlock) and getattr(m, 'checkpoint', False):
                    uses_ckpt = True
                    break
            self._ldm_blocks_checkpointed = uses_ckpt
        except Exception:
            self._ldm_blocks_checkpointed = True  # conservative default

        # Freeze or unfreeze main UNet according to flag
        if not self.train_unet:
            try:
                for p in self.model.parameters():
                    p.requires_grad = False
            except Exception:
                raise ValueError("UNet should not be trainable")
        else:
            # Ensure main UNet is trainable (in case of previous runs that froze it)
            try:
                for p in self.model.parameters():
                    p.requires_grad = True
                # keep module in train mode; EMA/sampling scopes will switch as needed
                self.model.train(True)
            except Exception:
                raise ValueError("UNet should be trainable")

        # ---- Initialize patch-window tokenizer & bank-gating fusers (strict, from pretrained) ----
        self._init_patch_gating_modules()

        # Ensure all non-UNet auxiliary modules stay trainable (InstanceDiffusion extras):
        # - Stage 1 global U-Net ID modules (already enforced above via mod_expected)
        # - patch_unifusion (separate UniFusion for patch window)
        # - bank_gating_fusers (gates global banks into main UNet)
        try:
            if not hasattr(self, "patch_unifusion"):
                raise RuntimeError(
                    "[Init] patch_unifusion not found. This module is required for patch-window tokenization."
                )
            if not hasattr(self, "bank_gating_fusers"):
                raise RuntimeError(
                    "[Init] bank_gating_fusers not found. These modules are required for global token injection."
                )

            for p in self.patch_unifusion.parameters():
                p.requires_grad = True
            for p in self.bank_gating_fusers.parameters():
                p.requires_grad = True
        except Exception as e:
            raise ValueError(f"Failed to enforce trainable flags for auxiliary modules: {e}")

    def set_global_cfg_mode(self, enabled: bool) -> None:
        """Enable or disable CFG-specific handling for Stage 1 global U-Net."""
        self._global_cfg_active = bool(enabled)

    # --------- APIs used by the sampler for global-branch denoising ---------
    def enable_global_branch_denoising(self, enabled: bool = True) -> None:
        self._global_branch_denoising_enabled = bool(enabled)

    def is_global_branch_denoising_enabled(self) -> bool:
        return bool(self._global_branch_denoising_enabled)

    def set_global_noisy_override(self, z: Optional[torch.Tensor]) -> None:
        """Provide external ref noisy latent (x_t) for the next apply_model() call.
        Shape must match the COND-half batch size when CFG is active, otherwise full batch.
        After being consumed once, the override is cleared.
        """
        self._global_noisy_override = z

    def pop_last_global_state(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return and clear the cached (ref_noisy_used, ref_pred) from the last apply_model call.
        Only set during sampling (no_grad). Returns (None, None) if unavailable.
        """
        ref_noisy, ref_pred = self._last_global_noisy, self._last_global_pred
        self._last_global_noisy, self._last_global_pred = None, None
        return ref_noisy, ref_pred

    def _load_global_base_from_ckpt(self, path: str, expected_keys: set[str]):
        """Load pretrained weights into the global model and verify coverage.

        Args:
            path: Path to checkpoint file
            expected_keys: Set of keys that must be loaded

        Raises:
            RuntimeError: If expected keys are not fully loaded
        """
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception as exc:
            raise FileNotFoundError(f"Failed to load global_pretrained_ckpt: {path} ({exc})") from exc

        if not hasattr(self.global_model, 'inner'):
            raise ValueError("IDGlobalUNetWrapper with attribute 'inner' is required for InstanceDiffusion.")
        ref = self.global_model.inner

        # Extract state dict
        if isinstance(ckpt, dict) and 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            sd = ckpt['state_dict']
        elif isinstance(ckpt, dict):
            sd = ckpt
        else:
            raise ValueError("global_pretrained_ckpt does not contain a valid state dict")

        # Remap prefixes to match ID-UNet structure
        def _remap_prefixes(state_dict):
            mapped = {}
            for k, v in state_dict.items():
                if k.startswith('model.diffusion_model.'):
                    mapped[k[len('model.diffusion_model.'):]] = v
                elif k.startswith('module.diffusion_model.'):
                    mapped[k[len('module.diffusion_model.'):]] = v
                elif k.startswith('diffusion_model.'):
                    mapped[k[len('diffusion_model.'):]] = v
                elif k.startswith('inner.'):
                    mapped[k[len('inner.'):]] = v
                else:
                    mapped[k] = v
            return mapped

        sd_mapped = _remap_prefixes(sd)

        # Filter by shape match
        ref_state = ref.state_dict()
        sd_filtered = {}
        for k, v in sd_mapped.items():
            if k in ref_state and isinstance(v, torch.Tensor) and isinstance(ref_state[k], torch.Tensor):
                if v.shape == ref_state[k].shape:
                    sd_filtered[k] = v

        # Basic sanity check
        if len(sd_filtered) == 0:
            sample_keys = list(sd.keys())[:5]
            raise RuntimeError(
                f"Loading global_pretrained_ckpt failed: no matching parameters after prefix/shape filter. "
                f"Ensure the checkpoint contains diffusion_model.* weights. Example keys: {sample_keys}"
            )

        # Strict verification: ALL expected keys must be loaded
        loaded_keys = set(sd_filtered.keys())
        missing = expected_keys.difference(loaded_keys)
        if missing:
            sample = sorted(list(missing))[:20]
            raise RuntimeError(
                f"[GlobalUNet] Failed to load {len(missing)} base UNet keys from {path}:\n" +
                "\n".join([f"  · {k}" for k in sample])
            )

        # Load into model
        ref.load_state_dict(sd_filtered, strict=False)
        print(f"[GlobalUNet] Loaded base UNet from {path}: {len(sd_filtered)} keys")

    def _load_global_modules_from_ckpt(self, path: str, expected_keys: set[str]):
        """Load InstanceDiffusion module-only checkpoint into ref.inner and verify coverage.

        Args:
            path: Path to InstanceDiffusion modules checkpoint
            expected_keys: Set of ID module keys that must be loaded

        Raises:
            RuntimeError: If expected keys are not fully loaded
        """
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception as exc:
            raise FileNotFoundError(f"Failed to load global_modules_ckpt: {path} ({exc})") from exc

        if not hasattr(self.global_model, 'inner'):
            raise ValueError("IDGlobalUNetWrapper with attribute 'inner' is required for InstanceDiffusion.")
        ref = self.global_model.inner
        ref_state = ref.state_dict()

        # Extract state dict
        if isinstance(ckpt, dict) and 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            sd = ckpt['state_dict']
        elif isinstance(ckpt, dict):
            sd = ckpt
        else:
            raise ValueError("global_modules_ckpt does not contain a valid state dict")

        # Strip DDP/Wrapper prefixes
        def _strip_prefix(k: str) -> str:
            if k.startswith('module.'):
                k = k[len('module.'):]
            return k

        sd_stripped = {_strip_prefix(k): v for k, v in sd.items()}

        # Filter: only keep expected keys with matching shapes
        sd_filtered = {}
        for k in expected_keys:
            if k in sd_stripped and k in ref_state:
                v = sd_stripped[k]
                if isinstance(v, torch.Tensor) and isinstance(ref_state[k], torch.Tensor) and v.shape == ref_state[k].shape:
                    sd_filtered[k] = v

        # Strict verification: check all expected keys are loaded
        loaded_keys = set(sd_filtered.keys())
        missing = expected_keys.difference(loaded_keys)
        if missing:
            sample = sorted(list(missing))[:20]
            raise RuntimeError(
                f"[GlobalUNet] Failed to load {len(missing)} ID module keys from {path}:\n" +
                "\n".join([f"  · {k}" for k in sample])
            )

        # Load into model
        ref.load_state_dict(sd_filtered, strict=False)
        print(f"[GlobalUNet] Loaded ID modules from {path}: {len(sd_filtered)} keys")

    @torch.no_grad()
    def get_input(
        self,
        batch: Dict,
        k,
        cond_key: Optional[str] = None,
        bs: Optional[int] = None,
        return_first_stage_outputs: bool = False,
    ):
        """
        get_input(): preprocessing batch
            - VAE -> latent z
            - mask/masked image concat
            - CLIP embedding -> c_crossattn
            - ...

        Extend parent get_input by adding 'ref_image_latent' encoded from the global global image.
        Uses self.global_key (default 'global_image').
        """
        # Parent assembles z, c_cat (mask+masked latent), and cross-attn text
        z, all_conds, x, xrec, xc = super().get_input(
            batch,
            self.first_stage_key,
            return_first_stage_outputs=True,
            bs=bs,
        )

        # Find global image & global mask
        if self.global_key not in batch:
            raise ValueError(
                f"GlobalUNetOutpaintDiffusion requires a global image in batch under key '{self.global_key}'"
            )
        if "global_mask" not in batch:
            raise ValueError("GlobalUNetOutpaintDiffusion requires 'global_mask' in batch for global-U-Net concat input.")
        # Global denoising source is the letterboxed 512x512 global image.
        ref_key = self.global_key
        # Canonical names: ref_image_* for global image (augmented), ref_mask_* for global mask
        ref_image_bhwc = batch[ref_key]
        ref_mask_bhwc = batch["global_mask"]
        # Source for ref denoising: prefer full global image if provided
        ref_source_bhwc = batch.get("global_image_full", ref_image_bhwc)
        if bs is not None:
            ref_image_bhwc = ref_image_bhwc[:bs]
            ref_mask_bhwc = ref_mask_bhwc[:bs]
            ref_source_bhwc = ref_source_bhwc[:bs]

        # Convert BHWC -> BCHW
        if ref_image_bhwc.dim() == 4 and ref_image_bhwc.shape[1] not in (1, 3, 4):
            ref_image_bchw = rearrange(ref_image_bhwc, "b h w c -> b c h w").contiguous().float()
        else:
            ref_image_bchw = ref_image_bhwc.contiguous().float()
        # Global mask to BCHW (0/1) and resize later to latent spatial size
        if ref_mask_bhwc.dim() == 4 and ref_mask_bhwc.shape[1] not in (1, 3, 4):
            ref_mask_bchw = rearrange(ref_mask_bhwc, "b h w c -> b c h w").contiguous().float()
        else:
            ref_mask_bchw = ref_mask_bhwc.contiguous().float()

        # Convert ref source BHWC -> BCHW if needed
        if ref_source_bhwc.dim() == 4 and ref_source_bhwc.shape[1] not in (1, 3, 4):
            ref_source_bchw = rearrange(ref_source_bhwc, "b h w c -> b c h w").contiguous().float()
        else:
            ref_source_bchw = ref_source_bhwc.contiguous().float()

        ref_image_bchw = ref_image_bchw.to(self.device, non_blocking=True)
        ref_mask_bchw = ref_mask_bchw.to(self.device, non_blocking=True)
        ref_source_bchw = ref_source_bchw.to(self.device, non_blocking=True)

        # Optionally use DC‑AE latents for the global branch (32ch@16x16). If enabled,
        # skip computing VAE ref latents to avoid redundant encodes.
        if getattr(self, 'global_use_dc_ae', False):
            # Lazy-init DC-AE wrapper
            if getattr(self, 'global_ae', None) is None:
                try:
                    from ldm.modules.autoencoder.dc_ae_wrapper import DCAEWrapper
                except Exception as e:
                    raise ImportError("DC-AE wrapper missing. Install efficientvit and ensure wrapper is available.") from e
                if not getattr(self, 'global_dc_ae_repo', None):
                    raise ValueError("global_use_dc_ae=True requires 'global_dc_ae_repo' in config.")
                self.global_ae = DCAEWrapper(
                    repo_or_path=self.global_dc_ae_repo,
                    subfolder=getattr(self, 'global_dc_ae_subfolder', None),
                    scale_factor=float(self.global_scale_factor.item()) if hasattr(self, 'global_scale_factor') else None,
                    backend='diffusers',
                    config_repo_or_path=getattr(self, 'global_dc_ae_config_repo', None),
                    force_fp32=True,
                    device=self.device,
                )
                # Ensure wrapper buffers (e.g., scale_factor) live on the same device as the module for DDP buffer sync
                try:
                    self.global_ae.to(self.device)
                except Exception:
                    pass
                try:
                    cfg_repo = getattr(self, 'global_dc_ae_config_repo', None)
                    cfg_repo = cfg_repo or "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers"
                    print(f"[DC-AE] Using repo='{self.global_dc_ae_repo}', config='{cfg_repo}', scale_factor={float(self.global_scale_factor.item()):.5f}")
                except Exception:
                    pass

            # Compute ref latents using DC‑AE (autocast honored in wrapper)
            ref_image_latent_dc = self.global_ae.encode(ref_image_bchw)
            # Update/propagate calibrated scale
            try:
                if hasattr(self.global_ae, 'scale_factor') and hasattr(self, 'global_scale_factor'):
                    self.global_scale_factor.data[...] = float(self.global_ae.scale_factor.item())
                    self._global_scale_calibrated = True
            except Exception:
                pass
            # Mask at latent resolution (16x16)
            ref_mask_1ch = ref_mask_bchw[:, :1] if ref_mask_bchw.shape[1] != 1 else ref_mask_bchw
            ref_mask_latent_dc = torch.nn.functional.interpolate(
                ref_mask_1ch, size=ref_image_latent_dc.shape[-2:], mode="nearest"
            )
            # Overwrite conditioning tensors for DC-AE path
            all_conds["ref_image_latent"] = [ref_image_latent_dc]
            all_conds["ref_mask_latent"] = [ref_mask_latent_dc]
            all_conds["ref_input"] = [torch.cat([ref_mask_latent_dc, ref_image_latent_dc], dim=1)]
            # Overwrite ref source latent (for ref_noisy construction) — use full global image if available
            all_conds["ref_denoise_latent"] = [self.global_ae.encode(ref_source_bchw)]
        else:
            # VAE latent path for global branch (4ch@64x64)
            # Encode global image -> 4ch latent
            ref_image_latent = self.get_first_stage_encoding(self.encode_first_stage(ref_image_bchw))
            # Resize mask to latent spatial size -> 1ch
            if ref_mask_bchw.shape[1] != 1:
                ref_mask_bchw = ref_mask_bchw[:, :1]
            ref_mask_latent = torch.nn.functional.interpolate(
                ref_mask_bchw, size=ref_image_latent.shape[-2:], mode="nearest"
            )
            # For Stage 1 global U-Net input: align channel order with main UNet c_concat (mask then latent)
            # ref_input (5ch, VAE)
            all_conds["ref_image_latent"] = [ref_image_latent]
            all_conds["ref_mask_latent"] = [ref_mask_latent]
            all_conds["ref_input"] = [torch.cat([ref_mask_latent, ref_image_latent], dim=1)]
            # Cache global-source latent for ref_noisy construction in apply_model
            ref_source_latent = self.get_first_stage_encoding(self.encode_first_stage(ref_source_bchw))
            all_conds["ref_denoise_latent"] = [ref_source_latent]

        # InstanceDiffusion: grounding 입력 필수 (항상 강제)
        # Keys: ref_boxes[B,N,4], ref_masks[B,N or B,N,1], ref_positive_embeddings[B,N,768]
        if not isinstance(batch, dict):
            raise ValueError("Batch must be a dict and contain grounding keys for InstanceDiffusion.")
        missing = [k for k in ("ref_boxes", "ref_masks", "ref_positive_embeddings") if k not in batch]
        if missing:
            raise ValueError(
                "Missing required grounding keys in batch for InstanceDiffusion: " + ", ".join(missing)
            )
        maybe_bs = bs if bs is not None else ref_image_bchw.shape[0]
        def _slice_if_needed(x):
            return x[:maybe_bs] if (hasattr(x, 'shape') and x.shape[0] >= maybe_bs) else x

        boxes = _slice_if_needed(batch["ref_boxes"])  # type: ignore[index]
        masks = _slice_if_needed(batch["ref_masks"])  # type: ignore[index]
        pos = _slice_if_needed(batch["ref_positive_embeddings"])  # type: ignore[index]
        attm = _slice_if_needed(batch.get("ref_att_masks")) if isinstance(batch, dict) else None
        # Store to cond; shape/content validation will happen in _build_grounding_from_cond
        all_conds['ref_boxes'] = [boxes]
        all_conds['ref_masks'] = [masks]
        all_conds['ref_positive_embeddings'] = [pos]
        if attm is not None:
            all_conds['ref_att_masks'] = [attm]

        # Encode global prompt for Stage 1 global U-Net context
        if 'global_prompt' not in batch:
            raise ValueError("GlobalUNetOutpaintDiffusion requires 'global_prompt' in batch for global context.")
        global_prompts = batch['global_prompt']

        if bs is not None and isinstance(global_prompts, (list, tuple)):
            global_prompts = list(global_prompts)[:bs]
        ref_c_txt = self.cond_stage_model.encode(global_prompts)

        all_conds['ref_c_txt'] = [ref_c_txt]

        # Pass patch window bbox (normalized xyxy on letterboxed 512) if present
        try:
            if 'ref_window_bbox' in batch:
                wb = _slice_if_needed(batch['ref_window_bbox'])  # [B,1,4]
                all_conds['ref_window_bbox'] = [wb]
        except Exception:
            pass

        # Initialize grounding_tokenizer_input for CFG random drop (bbox-only)
        # Prepare a minimal batch dict that GroundingNetInput expects
        if hasattr(self.global_model, 'inner') and hasattr(self.global_model.inner, 'grounding_tokenizer_input'):
            gti = self.global_model.inner.grounding_tokenizer_input
            if gti is not None:
                # Original (full modalities) — kept for reference:
                # gti_batch = {
                #     'boxes': boxes.cpu() if boxes.is_cuda else boxes,
                #     'masks': masks.cpu() if masks.is_cuda else masks,
                #     'text_embeddings': pos.cpu() if pos.is_cuda else pos,
                #     'scribbles': torch.zeros((boxes.shape[0], boxes.shape[1], 40), dtype=boxes.dtype),
                #     'polygons': torch.zeros((boxes.shape[0], boxes.shape[1], 512), dtype=boxes.dtype),
                #     'segs': torch.zeros((boxes.shape[0], 30, 64, 64), dtype=boxes.dtype),
                #     'points': torch.zeros((boxes.shape[0], boxes.shape[1], 2), dtype=boxes.dtype),
                # }
                # BBOX-ONLY: use minimal keys
                gti_batch = {
                    'boxes': boxes,
                    'masks': masks,
                    'text_embeddings': pos,
                }
                if attm is not None:
                    gti_batch['att_masks'] = attm
                # Call prepare to set gti.set = True and cache basic shape info
                gti.prepare(gti_batch, return_att_masks=(attm is not None))

        if return_first_stage_outputs:
            return z, all_conds, x, xrec, xc
        return z, all_conds

    # applying one step
    def apply_model(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        cond: Dict,
        *args,
        return_global_prediction: bool = False,
        **kwargs,
    ):
        """
        Process inpainting condition and inject Stage 1 global U-Net features via attention.
        """
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        # cross-attn (text), local prompt for unet (this is used in controlnet based mechanism)
        cond_txt_unet = torch.cat(cond["c_crossattn"], 1) if "c_crossattn" in cond else None

        # inpainting concat -> x_in (9ch: 4 + 4 + 1)
        if "c_concat" not in cond or cond["c_concat"] is None:
            raise ValueError("GlobalUNetOutpaintDiffusion requires 'c_concat' (mask/masked latent) in conditioning.")
        x_in = torch.cat([x_noisy] + cond["c_concat"], dim=1)

        # Global input is required (no fallback):
        # - VAE mode:  [mask(1) | image_latent(4)]  -> 5ch
        # - DC-AE mode: [mask(1) | image_latent(32)] -> 33ch
        # Naming convention:
        # - *_full: concatenated batch across cond-list; if CFG on and B=2N, layout is [UC | COND]
        # - *_run: subset actually used to run the global pass; if CFG on, this is the COND half only (N)
        if "ref_input" not in cond or cond["ref_input"] is None:
            raise ValueError("GlobalUNetOutpaintDiffusion requires 'ref_input' in conditioning.")
        ref_input_full = torch.cat(cond["ref_input"], dim=0)  # B,Cref,H',W'

        batch = x_in.shape[0]
        ref_input_batch = ref_input_full.shape[0]

        if batch != ref_input_batch:
            raise ValueError(
                f"Main/ref batch differ: main={batch}, ref={ref_input_batch}. Ensure conditioning pairs match."
            )

        do_cfg = False
        ref_input_run = ref_input_full
        t_run = t

        # Global context (global prompt) for GlobalUNet; UNet uses local cond_txt_unet (full batch)
        if 'ref_c_txt' not in cond or cond['ref_c_txt'] is None:
            raise ValueError("GlobalUNetOutpaintDiffusion requires 'ref_c_txt' (global prompt embedding) in conditioning.")
        ref_ctx_full = torch.cat(cond['ref_c_txt'], 1)
        ctx_ref_run = ref_ctx_full
        uc_size = 0

        if self._global_cfg_active and batch % 2 == 0:
            uc_size = batch // 2
            # Even if upstream passed non-zero uc latent, treat as zero by construction
            do_cfg = True
            ref_input_run = ref_input_full[uc_size:]
            if t.shape[0] == batch:
                t_run = t[uc_size:]
            if ref_ctx_full is not None and ref_ctx_full.dim() >= 1 and ref_ctx_full.shape[0] == batch:
                ctx_ref_run = ref_ctx_full[uc_size:]
            # ref_input coverage already includes mask (channel dim)

        # Lazily install attention controllers
        if self._global_writer is None:
            self._global_writer = GlobalAttentionControlIDWriter(           # hook
                getattr(self.global_model, 'inner', self.global_model),
                batch_size=ref_input_run.shape[0],
                global_token_scale=self.global_token_scale,
                save_attention_maps=self.save_attention_maps,
                # Ensure global tokens keep gradient for global-model training
                detach_global_tokens=False,
            )
        # Recreate writer if critical settings changed (batch, detach flag, scales)
        elif (
            getattr(self._global_writer, "batch_size", None) != ref_input_run.shape[0]
            or getattr(self._global_writer, "detach_global_tokens", None) is True
            or getattr(self._global_writer, "global_token_scale", None) != float(self.global_token_scale)
            or getattr(self._global_writer, "save_attention_maps", None) != bool(self.save_attention_maps)
        ):
            try:
                self._global_writer.remove()
            except Exception:
                pass
            self._global_writer = GlobalAttentionControlIDWriter(
                getattr(self.global_model, 'inner', self.global_model),
                batch_size=ref_input_run.shape[0],
                global_token_scale=self.global_token_scale,
                save_attention_maps=self.save_attention_maps,
                detach_global_tokens=False,
            )

        if self._global_reader is None:
            self._global_reader = GlobalAttentionControlLDM(            # hook
                diffusion_model,
                mode="read",
                batch_size=batch,
                global_token_scale=self.global_token_scale,
                save_attention_maps=self.save_attention_maps,
                # Reader does not cache tokens; flag unused but set False for clarity
                detach_global_tokens=False,
            )
        elif (
            getattr(self._global_reader, "batch_size", None) != batch
            or getattr(self._global_reader, "detach_global_tokens", None) is True
            or getattr(self._global_reader, "global_token_scale", None) != float(self.global_token_scale)
            or getattr(self._global_reader, "save_attention_maps", None) != bool(self.save_attention_maps)
        ):
            try:
                self._global_reader.remove()
            except Exception:
                pass
            self._global_reader = GlobalAttentionControlLDM(
                diffusion_model,
                mode="read",
                batch_size=batch,
                global_token_scale=self.global_token_scale,
                save_attention_maps=self.save_attention_maps,
                detach_global_tokens=False,
            )

        # Sanity: block counts must match
        if self._global_writer.num_blocks != self._global_reader.num_blocks:
            raise RuntimeError(
                f"Global writer/reader block count mismatch: writer={self._global_writer.num_blocks} "
                f"reader={self._global_reader.num_blocks}. Ensure UNet configs match."
            )

        # Clear previous banks and run global pass synchronized with current t
        self._global_writer.clear()

        # Build GlobalUNet input to match order: [ref_noisy | ref_mask_latent(1) | ref_image_latent]
        # - VAE mode:  4 + 1 + 4 = 9ch
        # - DC-AE mode: 32 + 1 + 32 = 65ch
        # - ref_noisy is q_sample(ref_denoise_latent_run, t_run)
        if "ref_denoise_latent" not in cond or cond["ref_denoise_latent"] is None:
            raise ValueError("GlobalUNetOutpaintDiffusion requires 'ref_denoise_latent' in conditioning.")
        ref_denoise_latent_full = torch.cat(cond["ref_denoise_latent"], dim=0)
        if ref_denoise_latent_full.shape[0] != ref_input_full.shape[0]:
            raise ValueError("'ref_denoise_latent' batch does not match 'ref_input' batch size.")
        ref_denoise_latent_run = ref_denoise_latent_full[uc_size:] if do_cfg else ref_denoise_latent_full

        # Build/refine ref noisy input (x_t). If sampler provided an override, use it; otherwise sample anew.
        if self._global_branch_denoising_enabled and (self._global_noisy_override is not None):
            # Expect override only for the COND half when CFG is active
            if self._global_noisy_override.shape[0] != ref_input_run.shape[0]:
                raise ValueError(
                    "ref_noisy_override batch mismatch. "
                    f"got={self._global_noisy_override.shape[0]} expected={ref_input_run.shape[0]}"
                )
            ref_noisy = self._global_noisy_override
            # one-shot consumption
            self._global_noisy_override = None
            # Not generating a new noise tensor here; the sampler controls the chain
            ref_noise = None
        else:
            # Training vs Sampling:
            # - Training (grad enabled): use q_sample(x_start=ref_denoise_latent_run, t)
            # - Sampling/Inference (no grad): start from pure Gaussian z_t ~ N(0, I)
            if not torch.is_grad_enabled():
                ref_noise = None
                ref_noisy = torch.randn_like(ref_denoise_latent_run)
            else:
                ref_noise = torch.randn_like(ref_denoise_latent_run)
                ref_noisy = self.q_sample(x_start=ref_denoise_latent_run, t=t_run, noise=ref_noise)
        x_ref_in = torch.cat([ref_noisy, ref_input_run], dim=1)

        # Run writer on the selected subset with required grounding (ID only)
        grounding = self._build_grounding_from_cond(cond, start_index=uc_size if do_cfg else 0)
        if not hasattr(self.global_model, 'forward_with_grounding'):
            raise ValueError("IDGlobalUNetWrapper with forward_with_grounding is required.")
        ref_pred = self.global_model.forward_with_grounding(
            x=x_ref_in,
            timesteps=t_run,
            context=ctx_ref_run,
            grounding_input=grounding,
        )
        # Share cached tokens to reader
        self._global_reader.update(self._global_writer)

        # ---- Bank gating with patch-window token (before CFG expansion) ----
        # Build patch token for the same run subset as writer (COND half when CFG active)
        bbox_full_list = cond.get('ref_window_bbox', None)
        if not bbox_full_list:
            raise ValueError("ref_window_bbox missing in cond; ensure dataloader/get_input provide it.")
        bbox_full = torch.cat(bbox_full_list, dim=0)  # [B,1,4]
        bbox_run = bbox_full[uc_size:] if do_cfg else bbox_full
        patch_token_768 = self._make_patch_token(bbox_run)  # [B_run,1,768]

        # Gate each bank tensor per block with its corresponding fuser
        if not hasattr(self, 'bank_gating_fusers'):
            raise RuntimeError("Bank-gating fusers not initialized.")
        if len(self.bank_gating_fusers) != self._global_reader.num_blocks:
            raise RuntimeError(
                f"Bank-gating fusers count mismatch: fusers={len(self.bank_gating_fusers)} "
                f"reader_blocks={self._global_reader.num_blocks}"
            )
        for idx, bank_list in list(self._global_reader.bank.items()):
            fuser = self.bank_gating_fusers[idx]
            new_list = []
            for tokens in bank_list:
                if tokens.shape[0] != patch_token_768.shape[0]:
                    raise RuntimeError(
                        f"Gating batch mismatch at block {idx}: tokens.B={tokens.shape[0]} vs patch.B={patch_token_768.shape[0]}"
                    )
                objs = patch_token_768.to(device=tokens.device, dtype=tokens.dtype)
                gated = fuser(tokens, objs)
                new_list.append(gated)
            self._global_reader.bank[idx] = new_list

        # If CFG: expand each cached token bank from [B, N, C] to [2B, N, C] as [zeros, real]
        # BANK를 업데이트 — 실패 시 침묵하지 말고 명확히 에러를 발생시켜 디버깅 용이성 확보
        if do_cfg:
            expected_cond_b = ref_input_run.shape[0]
            expected_full_b = batch
            for idx, bank_list in list(self._global_reader.bank.items()):
                expanded_list = []
                for tokens in bank_list:
                    # tokens: [B_cond, N, C] captured from conditional half
                    if tokens.shape[0] != expected_cond_b:
                        raise RuntimeError(
                            "CFG token-bank expansion failed: unexpected cond-batch size. "
                            f"block={idx}, tokens.shape[0]={tokens.shape[0]}, expected_cond={expected_cond_b}, "
                            f"expected_full={expected_full_b}. Ensure writer runs on COND half only and "
                            "batch pairing [UC|COND] is correct."
                        )
                    zeros = torch.zeros_like(tokens)
                    expanded_list.append(torch.cat([zeros, tokens], dim=0))
                self._global_reader.bank[idx] = expanded_list
        self._global_writer.clear()

        # Call main UNet (reader hooks will inject global tokens)
        eps = diffusion_model(x=x_in, timesteps=t, context=cond_txt_unet)

        # Clear banks depending on whether LDM blocks use checkpointing.
        # - When checkpointing is ON: keep until on_after_backward() (recompute needs them).
        # - When checkpointing is OFF: safe to clear now to lower live set size.
        if not getattr(self, '_ldm_blocks_checkpointed', True):
            try:
                if hasattr(self, "_global_reader") and (self._global_reader is not None):
                    self._global_reader.clear()
            except Exception:
                pass
        # Cache ref inputs/outputs for sampler-driven reference denoising (sampling only, i.e., no_grad)
        if not torch.is_grad_enabled():
            try:
                self._last_global_noisy = ref_noisy.detach()
                self._last_global_pred = ref_pred.detach()
            except Exception:
                self._last_global_noisy, self._last_global_pred = None, None

        if return_global_prediction:
            # Return both main UNet eps and Stage 1 global U-Net outputs for supervised loss
            # Note: ref_pred/ref_noise are on the COND subset when CFG is enabled.
            # Training path does not enable CFG for the global branch, so sizes match.
            return eps, {"ref_pred": ref_pred, "ref_noise": ref_noise}
        return eps

    # Override: add supervised noise-prediction loss on Stage 1 global U-Net
    def p_losses(self, x_start, cond, t, noise: Optional[torch.Tensor] = None):
        # ---- main UNet path (identical to parent) ----
        noise = noise if noise is not None else torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        out = self.apply_model(x_noisy, t, cond, return_global_prediction=True)
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            model_output = out[0]
            ref_dict = out[1]
        else:
            # Fallback: behave like parent if no ref outputs were returned
            model_output = out
            ref_dict = None

        loss_dict: Dict[str, torch.Tensor] = {}
        prefix = 'train' if self.training else 'val'

        # Parameterization target
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        # Main UNet loss (matches parent implementation)
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_simple_mean = loss_simple.mean()
        loss_dict.update({f'{prefix}/loss_simple': loss_simple_mean.detach()})

        logvar_t = self.logvar[t].to(self.device)
        loss_main = (loss_simple / torch.exp(logvar_t) + logvar_t)
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss_main.mean().detach()})
            loss_dict.update({'logvar': self.logvar.data.mean()})
        loss_main = self.l_simple_weight * loss_main.mean()

        # VLB term (same as parent)
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb.detach()})

        total_loss = loss_main + (self.original_elbo_weight * loss_vlb)
        # Monitor timestep sampling
        try:
            loss_dict.update({f'{prefix}/timestep_mean': t.float().mean().detach()})
        except Exception:
            pass

        # ---- Stage 1 global U-Net supervised noise loss (new) ----
        if ref_dict is None or ("ref_pred" not in ref_dict or "ref_noise" not in ref_dict):
            raise RuntimeError("Stage 1 global U-Net supervised loss requires ref_pred and ref_noise.")

        ref_pred = ref_dict["ref_pred"]
        ref_noise = ref_dict["ref_noise"]

        # Simple L2 on eps prediction per-sample
        ref_loss_simple = self.get_loss(ref_pred, ref_noise, mean=False).mean([1, 2, 3])
        ref_loss_simple_mean = ref_loss_simple.mean()

        # Apply the same logvar weighting scheme as main loss
        ref_loss_weighted = (ref_loss_simple / torch.exp(logvar_t) + logvar_t)
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/ref_loss_gamma': ref_loss_weighted.mean().detach()})
        
        # Match main: l_simple_weight and batch mean
        ref_loss_main = self.l_simple_weight * ref_loss_weighted.mean()

        # VLB-style auxiliary term mirroring main path
        ref_loss_vlb = self.get_loss(ref_pred, ref_noise, mean=False).mean(dim=(1, 2, 3))
        ref_loss_vlb = (self.lvlb_weights[t] * ref_loss_vlb).mean()
        loss_dict.update({f'{prefix}/ref_loss_vlb': ref_loss_vlb.detach()})

        # Combine like main: main-term + original_elbo_weight*vlb
        ref_total = ref_loss_main + (self.original_elbo_weight * ref_loss_vlb)

        # Logging: keep simple and total (detach for memory efficiency)
        loss_dict.update({f'{prefix}/ref_loss_simple': ref_loss_simple_mean.detach()})
        loss_dict.update({f'{prefix}/ref_loss': ref_total.detach()})

        # Ratio ref to main (simple losses) - already detached via no_grad
        with torch.no_grad():
            ratio = ref_loss_simple_mean / (loss_simple_mean + 1e-12)
        loss_dict.update({f'{prefix}/ref_to_main_simple': ratio})

        # Add weighted ref loss into total
        if self.global_noise_loss_weight > 0:
            total_loss = total_loss + self.global_noise_loss_weight * ref_total

        # Final combined (detach for logging, but return original for backward)
        loss_dict.update({f'{prefix}/loss': total_loss.detach()})
        return total_loss, loss_dict

    def _build_grounding_from_cond(self, cond: Dict[str, torch.Tensor], start_index: int = 0) -> Dict[str, torch.Tensor]:
        """
        Build InstanceDiffusion grounding_input dict from cond entries.
        Slices from `start_index:` along batch dim to support CFG cond-only path.
        Falls back to zeros when keys are missing.
        """
        device = self.device
        # Require mandatory keys
        for k in ("ref_boxes", "ref_masks", "ref_positive_embeddings"):
            if k not in cond or not cond[k] or not isinstance(cond[k][0], torch.Tensor):
                raise ValueError(f"Missing required grounding key: {k}")

        # Helper to take first tensor from cond list, then slice (cond only from CFG)
        def _slice(name: str) -> torch.Tensor:
            t = cond[name][0]
            if start_index > 0 and t.shape[0] >= start_index:
                t = t[start_index:]
            return t.to(device, non_blocking=True)

        boxes = _slice('ref_boxes')
        masks = _slice('ref_masks')
        pos = _slice('ref_positive_embeddings')

        # Shape validation
        if boxes.dim() != 3 or boxes.shape[-1] != 4:
            raise ValueError(f"ref_boxes must be [B,N,4], got {tuple(boxes.shape)}")
        
        # Accept [B,N] (ID style); if [B,N,1], squeeze to [B,N]
        if masks.dim() == 3 and masks.shape[-1] == 1:
            masks = masks.squeeze(-1)
        if masks.dim() != 2:
            raise ValueError(f"ref_masks must be [B,N], got {tuple(masks.shape)}")
        if pos.dim() != 3 or pos.shape[-1] != 768:
            raise ValueError(f"ref_positive_embeddings must be [B,N,768], got {tuple(pos.shape)}")

        B, N, _ = pos.shape
        def _zeros_like(shape):
            return torch.zeros((B, *shape), device=device)

        # Original grounding dict with all modalities (kept as comments):
        # grounding = {
        #     'boxes': boxes,
        #     'masks': masks,
        #     'positive_embeddings': pos,
        #     'scribbles': _zeros_like((N, 40)),   # 20 points * (x,y)
        #     'polygons': _zeros_like((N, 512)),   # 256 points * (x,y)
        #     'segs': torch.zeros((B, 30, 64, 64), device=device),
        #     'points': _zeros_like((N, 2)),
        # }
        # BBOX-ONLY grounding dict:
        grounding = {
            'boxes': boxes,
            'masks': masks,
            'positive_embeddings': pos,
        }

        # Optional instance attention masks: if present, use them; otherwise skip masking
        if 'ref_att_masks' in cond and cond['ref_att_masks']:
            attm = cond['ref_att_masks'][0]
            if start_index > 0 and hasattr(attm, 'shape') and attm.shape[0] >= start_index:
                attm = attm[start_index:]
            if not (attm.dim() == 4 and attm.shape[1:] == (N, 64, 64)):
                raise ValueError(f"ref_att_masks must be [B,N,64,64], got {tuple(attm.shape)}")
            grounding['att_masks'] = attm.to(device=device, dtype=torch.float32)
        return grounding

    # Debug utility to fetch last-step attention maps from reader
    def get_global_attention_maps(self) -> Optional[Dict[int, torch.Tensor]]:
        if self._global_reader is None:
            return None
        try:
            return self._global_reader.get_debug_maps()
        except Exception:
            return None

    @staticmethod
    def _draw_boxes_with_labels(draw, boxes, texts_list, img_width, img_height, colors):
        """
        Helper function to draw bounding boxes with text labels on a PIL ImageDraw object.

        Args:
            draw: PIL ImageDraw object
            boxes: numpy array of shape (N, 4) with normalized [x0, y0, x1, y1] coordinates
            texts_list: list of text labels (or None)
            img_width: image width in pixels
            img_height: image height in pixels
            colors: list of RGB tuples for box colors
        """
        n = int(boxes.shape[0])
        for j in range(n):
            x0, y0, x1, y1 = [float(v) for v in boxes[j]]
            # Clamp coordinates to [0, 1] and skip degenerate boxes
            x0, y0 = max(0.0, min(1.0, x0)), max(0.0, min(1.0, y0))
            x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
            if x1 <= x0 or y1 <= y0:
                continue

            # Convert to pixel coordinates
            X0, Y0 = int(round(x0 * img_width)), int(round(y0 * img_height))
            X1, Y1 = int(round(x1 * img_width)), int(round(y1 * img_height))
            color = colors[j % len(colors)]

            # Draw bounding box
            draw.rectangle([X0, Y0, X1, Y1], outline=color, width=2)

            # Draw text label if available
            label = None
            try:
                if texts_list is not None and j < len(texts_list):
                    label = str(texts_list[j])
            except Exception:
                label = None

            if label:
                bx0, by0 = X0 + 2, max(0, Y0 + 2)
                try:
                    # Try to get accurate text bounding box
                    l, t, r, b = draw.textbbox((bx0, by0), label)
                    bx1, by1 = r + 2, b + 2
                except Exception:
                    # Fallback to approximation
                    approx_w = 7 * len(label)
                    approx_h = 12
                    bx1, by1 = bx0 + approx_w + 4, by0 + approx_h + 2

                # Draw black background and white text
                draw.rectangle([bx0, by0, bx1, by1], fill=(0, 0, 0))
                draw.text((bx0 + 2, by0 + 1), label, fill=(255, 255, 255))

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N: int = 10,
        n_row: int = 5,
        sample: bool = True,
        ddim_steps: int = 50,
        ddim_eta: float = 0.0,
        return_keys=None,
        quantize_denoised: bool = True,
        inpaint: bool = True,
        plot_denoise_rows: bool = False,
        plot_progressive_rows: bool = True,
        plot_diffusion_rows: bool = True,
        unconditional_guidance_scale: float = 1.0,
        unconditional_guidance_label=None,
        use_ema_scope: bool = True,
        **kwargs,
    ):
        """
        Log images for training monitoring - includes inpainting and global inputs.
        Completely self-contained implementation (does NOT call super().log_images()).
        """
        use_ddim = ddim_steps is not None

        log = dict()
        # Get z, all conditionings and original gt for logging
        z, c, x, xrec, _ = self.get_input(batch, self.first_stage_key, bs=N, return_first_stage_outputs=True)

        # Extract different conditioning types
        c_concat = c.get("c_concat", [None])[0] if c.get("c_concat") else None
        c_crossattn = c["c_crossattn"][0][:N] if "c_crossattn" in c else None

        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)

        # Local GT (what we want to compare against)
        log["local_gt"] = x

        # Log inpainting control (mask + masked_image) - separated for better visualization
        if c_concat is not None:
            # Split c_concat: first 1 channel = mask, next 4 channels = masked_image latent
            mask_latent = c_concat[:N, :1]  # (N, 1, H//8, W//8)
            masked_image_latent = c_concat[:N, 1:5]  # (N, 4, H//8, W//8)

            # Decode masked image latent (local masked input in image space)
            log["local_masked_image"] = self.decode_first_stage(masked_image_latent)

            # Upsample mask to image size and normalize for visualization
            mask_img = torch.nn.functional.interpolate(
                mask_latent, size=(log["local_masked_image"].shape[-2:]), mode='nearest'
            )
            log["local_mask"] = mask_img * 2.0 - 1.0  # [-1, 1] range for display

        # Log Stage 1 global U-Net-specific inputs
        if "global_image" in batch:
            gi = batch["global_image"][:N]
            if gi.dim() == 4 and gi.shape[1] not in (1, 3, 4):  # BHWC -> BCHW
                gi = rearrange(gi, 'b h w c -> b c h w')
            log["ref_hint_image"] = gi.to(self.device).float()

        # Global inpainting mask (GlobalUNet 실제 입력)
        if "global_mask" in batch:
            gm = batch["global_mask"][:N]
            if gm.dim() == 4 and gm.shape[1] not in (1, 3, 4):  # BHWC -> BCHW
                gm = rearrange(gm, 'b h w c -> b c h w')
            gm = gm.to(self.device).float()
            log["ref_global_mask"] = (gm * 2.0 - 1.0)

        # Local patch location mask (global 상에서 현재 local patch 위치)
        if "window_mask" in batch:
            wm = batch["window_mask"][:N]
            if wm.dim() == 4 and wm.shape[1] not in (1, 3, 4):  # BHWC -> BCHW
                wm = rearrange(wm, 'b h w c -> b c h w')
            wm = wm.to(self.device).float()
            log["local_patch_mask"] = (wm * 2.0 - 1.0)

        # Log the original global image (letterboxed, no VAE encode-decode)
        if "global_image_full" in batch:
            gif = batch["global_image_full"][:N]
            if gif.dim() == 4 and gif.shape[1] not in (1, 3, 4):  # BHWC -> BCHW
                gif = rearrange(gif, 'b h w c -> b c h w')
            log["global_gt"] = gif.to(self.device).float()

        # Log text conditioning
        if c_crossattn is not None and self.cond_stage_key in batch:
            from ldm.util import log_txt_as_img
            log["conditioning"] = log_txt_as_img((512, 512),
                                                batch[self.cond_stage_key][:N], size=16)

        # Build three overlays per sample:
        #   1) ref_layout_blank: white bg, black letterbox, all boxes+texts (global coords)
        #   2) ref_layout_on_global_gt: draw on global_image_full (GT, letterboxed)
        #   3) ref_layout_on_ref_samples: draw on ref_samples (global coords like global GT)
        try:
            have_global = all(k in batch for k in ("global_image_full", "ref_boxes"))
            if have_global:
                import numpy as _np
                from PIL import Image as _Image, ImageDraw as _ImageDraw

                B = min(N, int(batch["global_image_full"].shape[0]))
                blanks, on_gt = [], []

                for i in range(B):
                    # Source global (HWC, [-1,1]) and uint8 copy for GT overlay
                    img_hwc = batch["global_image_full"][i].detach().float().cpu()
                    img_np = img_hwc.numpy()  # Convert once and reuse
                    img_np01 = _np.clip((img_np + 1.0) * 0.5, 0.0, 1.0)
                    img_u8 = (img_np01 * 255.0 + 0.5).astype(_np.uint8)
                    if img_u8.shape[-1] == 1:
                        img_u8 = _np.repeat(img_u8, 3, axis=-1)

                    H, W = img_u8.shape[0], img_u8.shape[1]

                    # Prepare canvases
                    pil_gt = _Image.fromarray(img_u8)
                    draw_gt = _ImageDraw.Draw(pil_gt)

                    # Blank layout: white bg, black letterbox inferred from img_hwc==0
                    blank = _np.ones((H, W, 3), dtype=_np.uint8) * 255
                    try:
                        pad_mask = (_np.abs(img_np) < 1e-6).all(axis=-1)
                        blank[pad_mask] = 0
                    except Exception:
                        pass
                    pil_blank = _Image.fromarray(blank)
                    draw_blank = _ImageDraw.Draw(pil_blank)

                    # Data for boxes and texts
                    boxes = batch["ref_boxes"][i].detach().float().cpu().numpy()  # (N,4) in [0,1]
                    texts_list = None
                    try:
                        if isinstance(batch.get('ref_texts'), list) and i < len(batch['ref_texts']):
                            texts_list = list(batch['ref_texts'][i])
                    except Exception:
                        texts_list = None

                    # Draw boxes on both canvases using helper method
                    self._draw_boxes_with_labels(draw_gt, boxes, texts_list, W, H, self._VIZ_COLORS)
                    self._draw_boxes_with_labels(draw_blank, boxes, texts_list, W, H, self._VIZ_COLORS)

                    # Convert back to [-1,1] BCHW (float16 for memory efficiency)
                    for target, acc in ((pil_blank, blanks), (pil_gt, on_gt)):
                        out_np = _np.array(target)
                        out_t = torch.from_numpy(out_np).permute(2, 0, 1).float() / 255.0
                        out_t = out_t * 2.0 - 1.0
                        out_t = out_t.half()  # Visualization only: float16 sufficient
                        acc.append(out_t)

                if blanks:
                    log["layout_blank"] = torch.stack(blanks, dim=0)
                if on_gt:
                    log["layout_on_global_gt"] = torch.stack(on_gt, dim=0)
        except Exception:
            # overlay is best-effort; ignore failures to avoid breaking training
            pass

        if sample:
            # Get sampling results via DDIMGlobalSampler (Ref branch denoising enabled)
            sampler = DDIMGlobalSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            _ref_last = {}
            def _ref_cb(pred_x0_ref, _i):
                _ref_last['pred_x0'] = pred_x0_ref
            samples, intermed = sampler.sample(
                S=ddim_steps,
                batch_size=N,
                shape=shape,
                conditioning=c,
                verbose=False,
                eta=ddim_eta,
                global_img_callback=_ref_cb,
            )
            x_samples = self.decode_first_stage(samples)
            log["local_samples"] = x_samples
            # Decode global branch last pred_x0 if available
            try:
                if 'pred_x0' in _ref_last and _ref_last['pred_x0'] is not None:
                    pred_x0_ref = _ref_last['pred_x0']
                    if getattr(self, 'global_use_dc_ae', False) and getattr(self, 'global_ae', None) is not None:
                        # DC-AE mode: decode with DC-AE decoder
                        try:
                            # Shape validation (to catch config errors early)
                            if pred_x0_ref.shape[1] != 32:
                                import warnings
                                warnings.warn(
                                    f"[DC-AE] Expected 32 channels but got {pred_x0_ref.shape[1]}. "
                                    f"Check Stage 1 global U-Net out_channels in config."
                                )
                            log["ref_samples"] = self.global_ae.decode(pred_x0_ref)
                        except Exception:
                            # Ref decode not critical for training
                            pass
                    else:
                        # VAE mode: decode with first-stage VAE
                        log["ref_samples"] = self.decode_first_stage(pred_x0_ref)
            except Exception:
                pass

            if plot_denoise_rows and isinstance(intermed, dict) and ('pred_x0' in intermed):
                try:
                    denoise_grid = self._get_denoise_row_from_list(intermed['pred_x0'])
                    log["denoise_row"] = denoise_grid
                except Exception:
                    pass

            # Also draw layout on the Stage 1 global U-Net-generated samples (global coords)
            try:
                if ("ref_samples" in log) and ("ref_boxes" in batch):
                    import numpy as _np
                    from PIL import Image as _Image, ImageDraw as _ImageDraw

                    gen = log["ref_samples"].detach().cpu()  # (B,3,H,W) in [-1,1]
                    B = min(gen.shape[0], N, int(batch["ref_boxes"].shape[0]))
                    on_ref = []
                    for i in range(B):
                        # Convert generated to uint8 HWC
                        g_chw = gen[i]
                        g_np = _np.clip((g_chw.numpy() + 1.0) * 0.5, 0.0, 1.0)
                        g_u8 = (g_np * 255.0 + 0.5).astype(_np.uint8)
                        g_u8 = _np.transpose(g_u8, (1, 2, 0))  # CHW->HWC
                        pil = _Image.fromarray(g_u8)
                        draw = _ImageDraw.Draw(pil)

                        Hs, Ws = pil.height, pil.width
                        boxes = batch["ref_boxes"][i].detach().float().cpu().numpy()
                        texts_list = None
                        try:
                            if isinstance(batch.get('ref_texts'), list) and i < len(batch['ref_texts']):
                                texts_list = list(batch['ref_texts'][i])
                        except Exception:
                            texts_list = None

                        # Draw boxes on generated samples using helper method
                        self._draw_boxes_with_labels(draw, boxes, texts_list, Ws, Hs, self._VIZ_COLORS)

                        out_np = _np.array(pil)
                        out_t = torch.from_numpy(out_np).permute(2, 0, 1).float() / 255.0
                        out_t = out_t * 2.0 - 1.0
                        out_t = out_t.half()  # Visualization only: float16 sufficient
                        on_ref.append(out_t)

                    if on_ref:
                        log["layout_on_ref_samples"] = torch.stack(on_ref, dim=0)
            except Exception:
                pass

        # CFG sampling if guidance scale > 1
        if unconditional_guidance_scale > 1.0:
            # Build unconditional text from label (default empty/negative prompt)
            uc_cross = self.get_unconditional_conditioning(N, unconditional_guidance_label)

            # For CFG: unconditional must match conditional keys for sampler compatibility
            # IMPORTANT: ddim.py iterates over ALL keys in c and expects them in unconditional too
            uc_full = {"c_concat": c.get("c_concat", []), "c_crossattn": [uc_cross]}

            # Include ALL global-related keys in uc to prevent KeyError in ddim sampler
            # Global latents (same for cond/uncond - we want same global image)
            if "ref_input" in c:
                uc_full["ref_input"] = c["ref_input"]
            if "ref_image_latent" in c:
                uc_full["ref_image_latent"] = c["ref_image_latent"]
            if "ref_mask_latent" in c:
                uc_full["ref_mask_latent"] = c["ref_mask_latent"]
            if "ref_denoise_latent" in c:
                uc_full["ref_denoise_latent"] = c["ref_denoise_latent"]

            # Global text context (use unconditional text)
            if "ref_c_txt" in c:
                uc_full["ref_c_txt"] = [uc_cross]

            # Include grounding keys (same for cond/uncond)
            for key in ("ref_boxes", "ref_masks", "ref_positive_embeddings", "ref_att_masks", "ref_window_bbox"):
                if key in c:
                    uc_full[key] = c[key]

            ema_scope = self.ema_scope if use_ema_scope else nullcontext
            with ema_scope("Sampling with classifier-free guidance"):
                # Enable CFG mode for Stage 1 global U-Net
                # # Enable CFG mode and global-branch denoising for Stage 1 global U-Net
                self.set_global_cfg_mode(True)
                try:
                    sampler_cfg = DDIMGlobalSampler(self)
                    shape = (self.channels, self.image_size, self.image_size)
                    _ref_last_cfg = {}
                    def _ref_cb_cfg(pred_x0_ref, _i):
                        _ref_last_cfg['pred_x0'] = pred_x0_ref
                    samples_cfg, _ = sampler_cfg.sample(
                        S=ddim_steps,
                        batch_size=N,
                        shape=shape,
                        conditioning=c,
                        verbose=False,
                        eta=ddim_eta,
                        unconditional_guidance_scale=unconditional_guidance_scale,
                        unconditional_conditioning=uc_full,
                        global_img_callback=_ref_cb_cfg,
                    )

                    x_samples_cfg = self.decode_first_stage(samples_cfg)
                    log[f"local_samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
                    # Note: Ref branch does not use CFG mixing; no separate CFG-tagged ref output is logged.
                finally:
                    # Disable CFG mode after sampling
                    # # Disable modes after sampling
                    # self.enable_global_branch_denoising(False)
                    self.set_global_cfg_mode(False)

        # Optional filtering of returned keys for logging simplicity
        if return_keys:
            try:
                keys = [k for k in return_keys if k in log]
                if keys:
                    return {k: log[k] for k in keys}
            except Exception:
                pass
        return log

    # Override optimizer to include global_model params (IDM-VTON style)
    def configure_optimizers(self):
        from torch.optim import AdamW
        lr = float(self.learning_rate)
        # Use weight decay from config (model.params.weight_decay)
        weight_decay = float(self.weight_decay)
        param_groups = []

        # main UNet
        if self.train_unet:
            print("!!!  Training main Unet  !!!")
            param_groups.append({"params": list(self.model.parameters()), "lr": lr})

        # # cond stage if trainable
        # if getattr(self, "cond_stage_trainable", False):
        #     param_groups.append({"params": list(self.cond_stage_model.parameters()), "lr": lr})

        # global U-Net — only optimize trainable (requires_grad=True) parameters.
        # Base UNet weights inside the Stage 1 global U-Net are frozen in __init__;
        # here we respect that by filtering parameters instead of passing all of them.
        if self.train_global_unet and (self.global_model is not None):
            ref_lr = lr * float(self.global_lr_scale)
            ref_params = [p for p in self.global_model.parameters() if p.requires_grad]
            if not ref_params:
                raise ValueError("global U-Net should exist and expose trainable parameters")
            param_groups.append({"params": ref_params, "lr": ref_lr})
        else:
            raise ValueError("global U-Net should exist")

        # bank-gating fusers + patch tokenizer (main path, separate group; follow UNet lr)
        extra_params = []

        # bank_gating_fusers must exist, have parameters, and all must be trainable.
        if hasattr(self, 'bank_gating_fusers') and isinstance(self.bank_gating_fusers, torch.nn.ModuleList):
            bank_params = list(self.bank_gating_fusers.parameters())
            if not bank_params:
                raise RuntimeError(
                    "[Opt] bank_gating_fusers exist but have no parameters. "
                    "Check _init_patch_gating_modules initialization."
                )
            non_trainable = sum(1 for p in bank_params if not p.requires_grad)
            if non_trainable > 0:
                raise RuntimeError(
                    f"[Opt] bank_gating_fusers have {non_trainable} frozen parameters. "
                    "All bank-gating fusers must be trainable (requires_grad=True)."
                )
            extra_params.extend(bank_params)
            print(
                f"[Opt] Adding {len(self.bank_gating_fusers)} bank-gating fusers "
                f"({sum(p.numel() for p in bank_params):,} params)"
            )
        else:
            raise RuntimeError(
                "[Opt] bank_gating_fusers not found. These are required for global token injection."
            )

        # patch_unifusion must exist, have parameters, and be trainable.
        if hasattr(self, 'patch_unifusion'):
            patch_params = list(self.patch_unifusion.parameters())
            if not patch_params:
                raise RuntimeError("[Opt] patch_unifusion has no parameters.")
            frozen_patch = sum(1 for p in patch_params if not p.requires_grad)
            if frozen_patch > 0:
                raise RuntimeError(
                    f"[Opt] patch_unifusion has {frozen_patch} frozen parameters. "
                    "All patch_unifusion parameters must be trainable (requires_grad=True)."
                )
            extra_params.extend(patch_params)
            print(
                f"[Opt] Adding patch_unifusion "
                f"({sum(p.numel() for p in patch_params):,} params)"
            )
        else:
            raise RuntimeError(
                "[Opt] patch_unifusion not found. This module is required for patch-window tokenization."
            )

        if not extra_params:
            raise RuntimeError(
                "[Opt] No extra params (bank_gating_fusers/patch_unifusion). "
                "These modules are required for training."
            )

        param_groups.append({"params": extra_params, "lr": lr})

        # Standard AdamW for Lightning gradient clipping compatibility (fused=True not supported).
        # Apply a modest weight decay to regularize all trainable modules.
        opt = AdamW(param_groups, lr=lr, weight_decay=weight_decay)

        # Preferred: Transformers warmup schedulers (InstanceDiffusion-style)
        if isinstance(getattr(self, 'scheduler_type', None), str):
            stype = str(self.scheduler_type).lower().strip()
            if stype in ("constant", "cosine"):
                ws = max(0, self.warmup_steps)
                if stype == "constant":
                    sched = get_constant_schedule_with_warmup(opt, num_warmup_steps=ws)
                else:  # cosine
                    if not self.total_training_steps or self.total_training_steps <= 0:
                        print("[LR][warn] total_training_steps not set for cosine; using constant warmup")
                        sched = get_constant_schedule_with_warmup(opt, num_warmup_steps=ws)
                    else:
                        sched = get_cosine_schedule_with_warmup(
                            opt, num_warmup_steps=ws, num_training_steps=self.total_training_steps
                        )
                return [opt], [{
                    'scheduler': sched,
                    'interval': 'step',
                    'frequency': 1,
                }]

        # # Legacy path: LambdaLR via scheduler_config
        # if getattr(self, "use_scheduler", False):
        #     from torch.optim.lr_scheduler import LambdaLR
        #     scheduler = instantiate_from_config(self.scheduler_config)
        #     scheduler = [{
        #         'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
        #         'interval': 'step',
        #         'frequency': 1
        #     }]
        #     return [opt], scheduler
        return opt

    @torch.no_grad()
    def get_unconditional_conditioning(self, batch_size, null_label=None):
        """Delegate to parent class to maintain signature compatibility."""
        return super().get_unconditional_conditioning(batch_size, null_label)

    # ==================== housekeeping ====================
    def on_after_backward(self):
        """
        Minimal housekeeping after backward.

        Notes for debugging (run these in your console when needed):
          - any(p.grad is not None for p in self.global_model.parameters())
          - all(p.grad is not None for p in self.global_model.inner.position_net.parameters())
          - any(p.grad is not None for p in self.model.parameters())
        """
        # If LDM UNet uses gradient checkpointing, keep clearing banks here
        # because backward recomputation needed them alive during backward.
        if getattr(self, '_ldm_blocks_checkpointed', True):
            try:
                if hasattr(self, "_global_reader") and (self._global_reader is not None):
                    self._global_reader.clear()
            except Exception:
                pass
            try:
                if hasattr(self, "_global_writer") and (self._global_writer is not None):
                    self._global_writer.clear()
            except Exception:
                pass
        # Gradient stats logging moved to on_before_optimizer_step to align with
        # optimizer stepping and log_every_n_steps. Nothing else to do here.
        return

    def on_before_optimizer_step(self, *args, **kwargs):
        """Log gradient statistics right before optimizer.step().
        This aligns logging with gradient accumulation and Trainer.log_every_n_steps.
        """
        if self.light_logging:
            return
        # Determine logging interval and gate by the step that will be committed
        try:
            log_interval = int(getattr(self.trainer, 'log_every_n_steps', 25))
        except Exception:
            log_interval = 25
        log_interval = max(1, log_interval)

        next_step = int(self.global_step) + 1
        if next_step % log_interval != 0:
            return

        try:
            def _grad_norm_and_stats(params):
                params = list(params)
                params_with_grad = [p for p in params if p.grad is not None and p.requires_grad]
                total_params = sum(1 for p in params if p.requires_grad)
                if not params_with_grad:
                    return 0.0, 0.0, False
                grads = [p.grad for p in params_with_grad]
                total_norm_sq = torch.zeros((), device=self.device, dtype=torch.float32)
                for g in grads:
                    total_norm_sq = total_norm_sq + g.detach().float().pow(2).sum()
                total_norm = torch.sqrt(total_norm_sq)
                has_nan = any(torch.isnan(g).any() or torch.isinf(g).any() for g in grads)
                ratio = len(params_with_grad) / max(1, total_params)

                return float(total_norm.item()), float(ratio), bool(has_nan)

            # UNet grads
            if hasattr(self, 'model'):
                unet_norm, unet_ratio, unet_has_nan = _grad_norm_and_stats(self.model.parameters())
                self.log('train/grad_norm_unet', unet_norm, on_step=True, on_epoch=False, prog_bar=False)
                self.log('train/grad_nonzero_ratio_unet', unet_ratio, on_step=True, on_epoch=False, prog_bar=False)
            else:
                unet_has_nan = False

            # Stage 1 global U-Net grads (all trainable params)
            if hasattr(self, 'global_model'):
                ref_norm, ref_ratio, ref_has_nan = _grad_norm_and_stats(self.global_model.parameters())
                self.log('train/grad_norm_ref', ref_norm, on_step=True, on_epoch=False, prog_bar=False)
                self.log('train/grad_nonzero_ratio_ref', ref_ratio, on_step=True, on_epoch=False, prog_bar=False)
            else:
                ref_has_nan = False

            # ---- Finer-grained grad norms for key submodules ----
            # 1) InstanceDiffusion UniFusion (position_net) inside Stage 1 global U-Net
            try:
                ref_inner = getattr(self.global_model, "inner", self.global_model)
                if hasattr(ref_inner, "position_net"):
                    uni_params = list(ref_inner.position_net.parameters())
                    if uni_params:
                        uni_norm, uni_ratio, _ = _grad_norm_and_stats(uni_params)
                        self.log('train/grad_norm_ref_unifusion', uni_norm, on_step=True, on_epoch=False, prog_bar=False)
                        self.log('train/grad_nonzero_ratio_ref_unifusion', uni_ratio, on_step=True, on_epoch=False, prog_bar=False)
            except Exception:
                pass

            # 2) ScaleU blocks inside Stage 1 global U-Net (scaleu_b_* / scaleu_s_*)
            try:
                ref_inner = getattr(self.global_model, "inner", self.global_model)
                scaleu_params = []
                for name, p in ref_inner.named_parameters():
                    if name.startswith("scaleu_b_") or name.startswith("scaleu_s_"):
                        scaleu_params.append(p)
                if scaleu_params:
                    scaleu_norm, scaleu_ratio, _ = _grad_norm_and_stats(scaleu_params)
                    self.log('train/grad_norm_ref_scaleu', scaleu_norm, on_step=True, on_epoch=False, prog_bar=False)
                    self.log('train/grad_nonzero_ratio_ref_scaleu', scaleu_ratio, on_step=True, on_epoch=False, prog_bar=False)
            except Exception:
                pass

            # 3) GatedSelfAttentionDense fusers inside Stage 1 global U-Net (ID UNet)
            try:
                ref_inner = getattr(self.global_model, "inner", self.global_model)
                gsa_params = []
                for m in ref_inner.modules():
                    if isinstance(m, GatedSelfAttentionDense):
                        gsa_params.extend(list(m.parameters()))
                if gsa_params:
                    gsa_norm, gsa_ratio, _ = _grad_norm_and_stats(gsa_params)
                    self.log('train/grad_norm_ref_fusers', gsa_norm, on_step=True, on_epoch=False, prog_bar=False)
                    self.log('train/grad_nonzero_ratio_ref_fusers', gsa_ratio, on_step=True, on_epoch=False, prog_bar=False)
            except Exception:
                pass

            # 4) Bank-side fusers (main UNet bridge)
            try:
                if hasattr(self, "bank_gating_fusers") and isinstance(self.bank_gating_fusers, torch.nn.ModuleList):
                    bank_params = list(self.bank_gating_fusers.parameters())
                    if bank_params:
                        bank_norm, bank_ratio, _ = _grad_norm_and_stats(bank_params)
                        self.log('train/grad_norm_bank_fusers', bank_norm, on_step=True, on_epoch=False, prog_bar=False)
                        self.log('train/grad_nonzero_ratio_bank_fusers', bank_ratio, on_step=True, on_epoch=False, prog_bar=False)
            except Exception:
                pass

            # 5) Patch UniFusion tokenizer (patch_unifusion)
            try:
                if hasattr(self, "patch_unifusion") and self.patch_unifusion is not None:
                    patch_params = list(self.patch_unifusion.parameters())
                    if patch_params:
                        patch_norm, patch_ratio, _ = _grad_norm_and_stats(patch_params)
                        self.log('train/grad_norm_patch_unifusion', patch_norm, on_step=True, on_epoch=False, prog_bar=False)
                        self.log('train/grad_nonzero_ratio_patch_unifusion', patch_ratio, on_step=True, on_epoch=False, prog_bar=False)
            except Exception:
                pass

            # Combined NaN flag
            self.log('train/has_nan_grad', float(unet_has_nan or ref_has_nan),
                    on_step=True, on_epoch=False, prog_bar=False)
        except Exception:
            pass
        return

    # ---------------------- Patch gating modules ----------------------
    def _init_patch_gating_modules(self) -> None:
        """Build patch-window tokenizer (as a separate UniFusion instance) and
        bank-gating fusers. All new modules are strictly initialized from the
        already-loaded Stage 1 global U-Net (ID) modules. No random init allowed.

        Raises:
            RuntimeError if any module cannot be initialized strictly from pretrained weights.
        """
        # 1) Build patch tokenizer as UniFusion (bbox-only)
        ref_inner = getattr(self.global_model, 'inner', None)
        if ref_inner is None or not hasattr(ref_inner, 'position_net'):
            raise RuntimeError("global model inner.position_net (UniFusion) not found for initialization.")
        position_net = ref_inner.position_net

        # Make a new UniFusion with identical config (bbox-only, separate tokenizer)
        self.patch_unifusion = UniFusion(
            in_dim=768,
            out_dim=768,
            mid_dim=3072,
            train_add_boxes=True,
            train_add_points=False,
            train_add_scribbles=False,
            train_add_masks=False,
            test_drop_boxes=False,
            test_drop_points=False,
            test_drop_scribbles=False,
            test_drop_masks=False,
            use_seperate_tokenizer=True,
        )
        # Strictly load entire state dict from the global UniFusion
        try:
            self.patch_unifusion.load_state_dict(position_net.state_dict(), strict=True)
        except Exception as e:
            raise RuntimeError("Failed to initialize patch UniFusion from global UniFusion weights (strict).") from e

        # 2) Build bank-gating fusers for main LDM UNet and strictly load from ref fusers
        id_fusers: list[torch.nn.Module] = []
        for module in ref_inner.modules():
            blocks = getattr(module, 'transformer_blocks', None)
            if blocks is None:
                continue
            for block in list(blocks):
                fuser = getattr(block, 'fuser', None)
                if fuser is not None and hasattr(fuser, 'state_dict'):
                    id_fusers.append(fuser)
        if not id_fusers:
            raise RuntimeError("No fusers found in global model; cannot initialize bank-gating fusers.")

        ldm_blocks: list[LDM_BTBlock] = []
        for module in self.model.diffusion_model.modules():
            if isinstance(module, LDM_SpatialTransformer):
                for block in getattr(module, 'transformer_blocks', []):
                    if isinstance(block, LDM_BTBlock):
                        ldm_blocks.append(block)
        if not ldm_blocks:
            raise RuntimeError("No LDM BasicTransformerBlocks found for bank-gating fusers.")
        if len(ldm_blocks) != len(id_fusers):
            raise RuntimeError(
                f"Block count mismatch between LDM({len(ldm_blocks)}) and ID fusers({len(id_fusers)})."
            )

        fusers: list[torch.nn.Module] = []
        for i, (ldm_blk, id_fuser) in enumerate(zip(ldm_blocks, id_fusers)):
            attn1 = getattr(ldm_blk, 'attn1', None)
            if attn1 is None:
                raise RuntimeError(f"LDM block {i} has no attn1; cannot derive head dims.")
            heads = int(getattr(attn1, 'heads', 0) or 0)
            try:
                inner_dim = attn1.to_q.weight.shape[0]
                if heads <= 0:
                    raise ValueError("heads must be > 0")
                d_head = inner_dim // heads
                query_dim = attn1.to_q.in_features
            except Exception as e:
                raise RuntimeError(f"Failed to infer dims for LDM block {i}") from e

            base_fuser = GatedSelfAttentionDense(
                query_dim=query_dim,
                context_dim=768,
                n_heads=heads,
                d_head=d_head,
                efficient_attention=True,
            )
            try:
                base_fuser.load_state_dict(id_fuser.state_dict(), strict=True)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize bank-gating fuser {i} from global fuser weights (strict)."
                ) from e

            # Zero-initialized projection wrapper: gradual activation from identity
            class _BankFuserWrapper(torch.nn.Module):
                def __init__(self, inner_fuser: GatedSelfAttentionDense, dim: int):
                    super().__init__()
                    self.inner_fuser = inner_fuser
                    self.proj_out = torch.nn.Linear(dim, dim)
                    with torch.no_grad():
                        self.proj_out.weight.zero_()
                        if self.proj_out.bias is not None:
                            self.proj_out.bias.zero_()

                def forward(self, x, objs, grounding_input=None, drop_box_mask=False):
                    x_fused = self.inner_fuser(
                        x, objs, grounding_input=grounding_input, drop_box_mask=drop_box_mask
                    )
                    return self.proj_out(x_fused)  # Zero-initialized projection

            wrapper = _BankFuserWrapper(base_fuser, query_dim)
            fusers.append(wrapper)

        self.bank_gating_fusers = torch.nn.ModuleList(fusers)

    def _make_patch_token(self, bbox_b14: torch.Tensor) -> torch.Tensor:
        """Build [B,1,768] patch token using the separate UniFusion instance.
        Use its own null_positive_feature as the positive embedding and mask=1
        so that boxes are respected and positive passes through unchanged.
        """
        if not hasattr(self, 'patch_unifusion'):
            raise RuntimeError("Patch UniFusion not initialized.")
        B = bbox_b14.shape[0]
        device = next(self.patch_unifusion.parameters()).device
        dtype = next(self.patch_unifusion.parameters()).dtype
        boxes = bbox_b14.to(device=device, dtype=dtype)
        masks = torch.ones(B, 1, device=device, dtype=dtype)
        pos_null = self.patch_unifusion.null_positive_feature.view(1, 1, -1).expand(B, 1, -1)
        pos_null = pos_null.to(device=device, dtype=dtype)
        
        # Ensure no random box-drop is applied on patch path: run in eval mode while keeping grads
        prev_mode = self.patch_unifusion.training
        try:
            self.patch_unifusion.train(False)
            objs, _ = self.patch_unifusion(boxes, masks, pos_null)
        finally:
            self.patch_unifusion.train(prev_mode)
        return objs  # [B,1,768]

    # Lightweight timing/memory logging
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        # Call parent hook first
        try:
            super().on_train_batch_start(batch, batch_idx, dataloader_idx)
        except Exception:
            pass
        # Start timing and memory tracking
        try:
            self._step_t0 = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)
        except Exception:
            pass

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Minimal hook after a training batch.

        Notes for debugging (run these ad-hoc in your console if needed):
          - any(p.grad is not None for p in self.global_model.parameters())
          - all(p.grad is not None for p in self.global_model.inner.position_net.parameters())
          - any(p.grad is not None for p in self.model.parameters())
          - {n: (p.grad.abs().max().item() if p.grad is not None else None)
               for n,p in self.global_model.named_parameters()}
        """
        # Call parent hook first
        try:
            super().on_train_batch_end(outputs, batch, batch_idx)
        except Exception:
            pass

        # Step timing and memory usage (optional)
        if not self.light_logging:
            try:
                if hasattr(self, '_step_t0'):
                    dt = time.time() - float(self._step_t0)
                    self.log('train/step_time_sec', dt, on_step=True, on_epoch=False, prog_bar=False)
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
                    peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                    self.log('train/mem_alloc_gb', alloc, on_step=True, on_epoch=False, prog_bar=False)
                    self.log('train/mem_max_alloc_gb', peak, on_step=True, on_epoch=False, prog_bar=False)
            except Exception:
                pass
        return
