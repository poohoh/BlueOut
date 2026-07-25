"""
ControlNet + Inpainting
LatentInpaintDiffusion + ControlNet
"""

import torch
from contextlib import nullcontext
from einops import rearrange

from ldm.models.diffusion.ddpm import LatentInpaintDiffusion
from ldm.util import instantiate_from_config


class ControlOutpaintDiffusion(LatentInpaintDiffusion):
    """
    LatentInpaintDiffusion + ControlNet
    - conserve inpainting functions (pass mask, masked_image as c_concat)
    - add control signal through ControlNet
    """

    def __init__(self, control_stage_config, control_key, only_mid_control=False, 
                 global_average_pooling=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.global_average_pooling = global_average_pooling
        self.control_scales = [1.0] * 13

    def get_input(self, batch, k, cond_key=None, bs=None, return_first_stage_outputs=False):
        """
        Unify conditioning assembly for ControlOutpaintDiffusion.
        """
        # original inpainting input
        if return_first_stage_outputs:
            z, all_conds, x, xrec, xc = super().get_input(batch, k, cond_key, bs, return_first_stage_outputs=True)
        else:
            z, all_conds = super().get_input(batch, k, cond_key, bs, return_first_stage_outputs=False)

        # Control hint: RGB+mask
        hint = batch[self.control_key]
        if bs is not None:
            hint = hint[:bs]

        # Safe format conversion: only rearrange if likely BHWC format
        if hint.dim() == 4 and hint.shape[1] not in (1, 3, 4):  # not already BCHW
            hint = rearrange(hint, 'b h w c -> b c h w')

        hint = hint.to(self.device).to(memory_format=torch.contiguous_format).float()

        # Add to conditioning dict
        all_conds[self.control_key] = [hint]

        # Global text for ControlNet must be provided explicitly
        if 'global_prompt' not in batch:
            raise ValueError("Missing 'global_prompt' in batch for global context (c_control_txt)")
        global_prompts = batch['global_prompt']

        # If bs is given, align prompts length
        if bs is not None and isinstance(global_prompts, (list, tuple)):
            global_prompts = list(global_prompts)[:bs]
        
        c_control_txt = self.cond_stage_model.encode(global_prompts)
        all_conds['c_control_txt'] = [c_control_txt]

        if return_first_stage_outputs:
            return z, all_conds, x, xrec, xc
        return z, all_conds

    # applying one step
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        """
        process inpainting condition + control signal
        """
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        # crossattn condition (text)
        # concat하는 이유: 단일 프롬프트여도 c_crossattn은 리스트. ex) [B, 77, 768]
        cond_txt_unet = torch.cat(cond['c_crossattn'], 1)

        # global prompt
        if 'c_control_txt' not in cond or cond['c_control_txt'] is None:
            raise ValueError("ControlOutpaintDiffusion requires 'c_control_txt' in conditioning (global context)")
        cond_txt_ctrl = torch.cat(cond['c_control_txt'], 1)

        # outpainting concat -> x_in (9ch)
        if 'c_concat' in cond and cond['c_concat'] is not None:
            x_in = torch.cat([x_noisy] + cond['c_concat'], dim=1)
        else:
            x_in = x_noisy
        
        # controlnet (hint has key c_control)
        # control hint가 있어야 함: 없으면 명시적으로 에러 발생
        if 'c_control' not in cond or cond['c_control'] is None:
            raise ValueError(
                "ControlOutpaintDiffusion requires 'c_control' in conditioning. "
                "Provide control hint (e.g., generated via hint_encoder)."
            )

        # Use pre-encoded hint from get_input()
        hint = torch.cat(cond['c_control'], dim=1)

        # control encoder input: global prompt, x_in: latent 9ch (same with unet), hint: global image + global mask
        control = self.control_model(x=x_in, hint=hint, timesteps=t, context=cond_txt_ctrl)
        control = [c * s for c, s in zip(control, self.control_scales)]

        # Apply global average pooling if enabled (for global composition guidance)
        if self.global_average_pooling:
            control = [torch.mean(c, dim=(2, 3), keepdim=True) for c in control]
        
        # call unet
        # context(text prompt): local prompt
        eps = diffusion_model(
            x=x_in, timesteps=t, context=cond_txt_unet,
            control=control, only_mid_control=self.only_mid_control
        )

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, batch_size, null_label=None):
        """
        Delegate to parent class to maintain signature compatibility.
        CFG c_control handling is done in sampler level.
        """
        return super().get_unconditional_conditioning(batch_size, null_label)
    
    def configure_optimizers(self):
        """
        Fine-tune ControlNet and hint encoder while freezing SD UNet.
        Always use a PyTorch LR scheduler: linear warmup -> constant.
        """
        lr = self.learning_rate

        # Freeze SD UNet parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Train ControlNet parameters only (UNet frozen)
        params = list(self.control_model.parameters())

        if self.learn_logvar:
            print(f"{self.__class__.__name__}: Learning logvar")
            params.append(self.logvar)

        opt = torch.optim.AdamW(params, lr=lr)

        # Scheduler disabled: use constant LR
        # warmup_steps = 2500
        # start_factor = 1e-6
        # def warmup_then_constant(step: int):
        #     if step < warmup_steps:
        #         return start_factor + (1.0 - start_factor) * (step / max(1, warmup_steps))
        #     return 1.0
        # sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=warmup_then_constant)
        # return [opt], [{
        #     'scheduler': sched,
        #     'interval': 'step',
        #     'frequency': 1
        # }]
        return [opt]
    
    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, 
                   return_keys=None, quantize_denoised=True, inpaint=True, plot_denoise_rows=False, 
                   plot_progressive_rows=True, plot_diffusion_rows=False, 
                   unconditional_guidance_scale=7.0, unconditional_guidance_label="",
                   use_ema_scope=True, **kwargs):
        """
        Log images for training monitoring - includes both inpainting and control inputs
        """
        use_ddim = ddim_steps is not None
        
        log = dict()
        # get z, all conditionings and original gt for logging
        z, c, x, xrec, _ = self.get_input(batch, self.first_stage_key, bs=N, return_first_stage_outputs=True)
        
        # Extract different conditioning types
        c_concat = c.get("c_concat", [None])[0] if c.get("c_concat") else None
        c_crossattn = c["c_crossattn"][0][:N] if "c_crossattn" in c else None  
        c_control = c.get("c_control", [None])[0] if c.get("c_control") else None
        
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)

        # Local GT (what we want to compare against)
        log["gt"] = x
        
        # Log inpainting control (mask + masked_image) - separated for better visualization
        if c_concat is not None:
            # Split c_concat: first 1 channel = mask, next 4 channels = masked_image latent
            mask_latent = c_concat[:N, :1]  # (N, 1, H//8, W//8)
            masked_image_latent = c_concat[:N, 1:5]  # (N, 4, H//8, W//8)
            
            # Decode masked image latent (local masked input in image space)
            log["masked_image"] = self.decode_first_stage(masked_image_latent)
            
            # Upsample mask to image size and normalize for visualization
            mask_img = torch.nn.functional.interpolate(
                mask_latent, size=(log["masked_image"].shape[-2:]), mode='nearest'
            )
            log["mask"] = mask_img * 2.0 - 1.0  # [-1, 1] range for display
            
            # Keep latent concat visualization (mask+masked_latent) for reference
            log["inpaint_control"] = c_concat[:N] * 2.0 - 1.0
            
        # Log ControlNet hint
        if c_control is not None:
            log["controlnet_hint"] = c_control[:N] * 2.0 - 1.0
            
        # Log text conditioning
        if c_crossattn is not None and self.cond_stage_key in batch:
            from ldm.util import log_txt_as_img
            log["conditioning"] = log_txt_as_img((512, 512), 
                                                batch[self.cond_stage_key][:N], size=16)

        # Also log provided global inputs if available
        if "global_image" in batch:
            gi = batch["global_image"][:N]
            if gi.dim() == 4 and gi.shape[1] not in (1, 3, 4):  # BHWC -> BCHW
                gi = rearrange(gi, 'b h w c -> b c h w')
            log["global_image"] = gi
        if "global_mask" in batch:
            gm = batch["global_mask"][:N]
            if gm.dim() == 4 and gm.shape[1] not in (1, 3, 4):  # BHWC -> BCHW
                gm = rearrange(gm, 'b h w c -> b c h w')
            log["global_mask"] = gm * 2.0 - 1.0
        
        if sample:
            # Get sampling results
            samples, z_denoise_row = self.sample_log(
                cond=c, batch_size=N, ddim=use_ddim,
                ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid
                
        # CFG sampling if guidance scale > 1
        if unconditional_guidance_scale > 1.0:
            # Follow inference behavior: build unconditional text from label (default empty/negative prompt)
            uc_cross = self.get_unconditional_conditioning(N, unconditional_guidance_label)
            
            # For CFG: unconditional must match conditional keys for sampler compatibility
            uc_full = {"c_concat": c.get("c_concat", []), "c_crossattn": [uc_cross]}
            # Include control-related keys in uc to prevent KeyError in DDIMSampler
            if "c_control" in c:
                uc_full["c_control"] = c["c_control"]  # same hint for cond/uncond
            if "c_control_txt" in c:
                # use unconditional text for control branch too (mirror inference)
                uc_full["c_control_txt"] = [uc_cross]
            
            ema_scope = self.ema_scope if use_ema_scope else nullcontext
            with ema_scope("Sampling with classifier-free guidance"):
                samples_cfg, _ = self.sample_log(
                    cond=c, batch_size=N, ddim=use_ddim,
                    ddim_steps=ddim_steps, eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=uc_full)
                    
                x_samples_cfg = self.decode_first_stage(samples_cfg)
                log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        # Optional filtering of returned keys for logging simplicity
        if return_keys:
            try:
                keys = [k for k in return_keys if k in log]
                if keys:
                    return {k: log[k] for k in keys}
            except Exception:
                pass
        return log

    def on_before_optimizer_step(self, *args, **kwargs):
        """Log gradient statistics right before optimizer.step().
        This aligns logging with gradient accumulation and Trainer.log_every_n_steps.
        """
        if bool(getattr(self, "light_logging", False)):
            return
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
                params_with_grad = [p for p in params if p.requires_grad and p.grad is not None]
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

            ctrl_norm, ctrl_ratio, ctrl_nan = _grad_norm_and_stats(self.control_model.parameters())
            self.log('train/grad_norm_controlnet', ctrl_norm, on_step=True, on_epoch=False, prog_bar=False)
            self.log('train/grad_nonzero_ratio_controlnet', ctrl_ratio, on_step=True, on_epoch=False, prog_bar=False)
            self.log('train/has_nan_grad', float(ctrl_nan), on_step=True, on_epoch=False, prog_bar=False)
        except Exception:
            pass
