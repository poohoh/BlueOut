"""DDIM sampler with Stage 1 global U-Net progressive denoising.

This is a copy of ldm.models.diffusion.ddim.DDIMSampler with a few additions:
- Keeps the main UNet sampling identical (supports CFG etc.)
- Also steps a Stage 1 global U-Net branch across timesteps to generate features
  that get injected via attention (no CFG on global branch itself).

The Stage 1 global U-Net integration is opt-in and requires the model to expose
lightweight sampling-time hooks (implemented in GlobalUNetOutpaintDiffusion):
  - set_global_cfg_mode(bool): whether to slice to COND half under CFG
  - enable_global_branch_denoising(bool): enable/disable global-branch stepping
  - set_global_noisy_override(tensor|None): provide x_t for global branch (COND half)
  - pop_last_global_state() -> (global_noisy_used, global_eps_pred)

If these hooks are missing, the sampler falls back to standard DDIM behavior.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
    extract_into_tensor,
)


class DDIMGlobalSampler(object):
    def __init__(self, model, schedule: str = "linear", device: torch.device | None = None, **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.device = device or getattr(model, "device", torch.device("cuda"))

        # Global-branch state (when enabled)
        self._global_z = None                # ref x_t (COND-half shape when CFG)
        self._global_pred_x0_last = None     # last pred_x0 for global branch (for optional callback)

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert (
            alphas_cumprod.shape[0] == self.ddpm_num_timesteps
        ), "alphas have to be defined for each timestep"
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev))

        # q(x_t | x_{t-1}) helpers
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod.cpu()))
        )
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu())))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1))
        )

        # DDIM parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer("ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        dynamic_threshold=None,
        ucg_schedule=None,
        global_img_callback=None,  # optional callback for pred_x0_ref per step
        **kwargs,
    ):
        # Basic sanity on conditioning batch size
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list):
                    ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {ctmp.shape[0]} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(
                        f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}"
                    )

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)

        # If the model supports CFG-mode selection for the Stage 1 global U-Net, set it here
        try:
            if hasattr(self.model, "set_global_cfg_mode"):
                # If CFG scale varies over schedule, choose per-step later; here seed with current value
                global_cfg = bool(unconditional_guidance_scale is not None and unconditional_guidance_scale > 1.0)
                self.model.set_global_cfg_mode(global_cfg)
        except Exception:
            pass

        # Enable global-branch denoising if supported; ensure cleanup afterwards
        global_sampling_enabled = False
        try:
            if hasattr(self.model, "enable_global_branch_denoising"):
                self.model.enable_global_branch_denoising(True)
                global_sampling_enabled = True
        except Exception:
            global_sampling_enabled = False

        device = self.model.betas.device
        b = batch_size
        if x_T is None:
            # Create 4D tensor (batch_size, C, H, W) like original DDIM
            C, H, W = shape
            size = (batch_size, C, H, W)
            img = torch.randn(size, device=device)
        else:
            img = x_T

        if not isinstance(eta, float):
            eta = eta.item()
        if eta < 0.0:
            eta = 0.0

        intermediates = {"x_inter": [img], "pred_x0": [img]}
        # reset global branch state per sampling run
        self._global_z = None
        self._global_pred_x0_last = None

        time_range = np.flip(self.ddim_timesteps)
        total_steps = self.ddim_timesteps.shape[0]
        print(f"Running DDIMGlobal Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="DDIMGlobal Sampler", total=total_steps)

        try:
            for i, step in enumerate(iterator):
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)

                if mask is not None:
                    assert x0 is not None
                    img_orig = self.model.q_sample(x0, ts)
                    img = img_orig * mask + (1.0 - mask) * img

                if ucg_schedule is not None:
                    assert len(ucg_schedule) == len(time_range)
                    unconditional_guidance_scale = ucg_schedule[i]
                    # keep Stage 1 global U-Net in CFG-mode whenever main CFG is on
                    try:
                        if hasattr(self.model, "set_global_cfg_mode"):
                            self.model.set_global_cfg_mode(
                                bool(unconditional_guidance_scale is not None and unconditional_guidance_scale > 1.0)
                            )
                    except Exception:
                        pass

                outs = self.p_sample_ddim(
                    img,
                    conditioning,
                    ts,
                    index=index,
                    use_original_steps=False,
                    quantize_denoised=quantize_x0,
                    temperature=temperature,
                    noise_dropout=noise_dropout,
                    score_corrector=score_corrector,
                    corrector_kwargs=corrector_kwargs,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    dynamic_threshold=dynamic_threshold,
                    global_img_callback=global_img_callback,
                )
                img, pred_x0 = outs

                if callback:
                    callback(i)
                if img_callback:
                    img_callback(pred_x0, i)

                if index % log_every_t == 0 or index == total_steps - 1:
                    intermediates["x_inter"].append(img)
                    intermediates["pred_x0"].append(pred_x0)
        finally:
            # best-effort cleanup
            try:
                if global_sampling_enabled and hasattr(self.model, "enable_global_branch_denoising"):
                    self.model.enable_global_branch_denoising(False)
            except Exception:
                pass
            try:
                if hasattr(self.model, "set_global_cfg_mode"):
                    self.model.set_global_cfg_mode(False)
            except Exception:
                pass

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        c,
        t,
        index,
        repeat_noise: bool = False,
        use_original_steps: bool = False,
        quantize_denoised: bool = False,
        temperature: float = 1.0,
        noise_dropout: float = 0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale: float = 1.0,
        unconditional_conditioning=None,
        dynamic_threshold=None,
        global_img_callback=None,
    ):
        b, *_, device = *x.shape, x.device

        # If model supports providing global x_t, supply the previous step's x_t
        try:
            if (
                hasattr(self.model, "enable_global_branch_denoising")
                and hasattr(self.model, "set_global_noisy_override")
            ):
                # None on first step -> model will construct q_sample(source, t)
                self.model.set_global_noisy_override(self._global_z)
        except Exception:
            pass

        # ----- main UNet forward (identical to base sampler) -----
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            model_output = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = {}
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([unconditional_conditioning[k][i], c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([unconditional_conditioning[k], c[k]])
            elif isinstance(c, list):
                c_in = []
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", "not implemented"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # pred x0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        # Keep parity with original DDIM: not implemented here
        if dynamic_threshold is not None:
            raise NotImplementedError()

        # Direction to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        # ----- Global branch update (no CFG on ref) -----
        try:
            if hasattr(self.model, "pop_last_global_state"):
                ref_noisy_used, ref_eps = self.model.pop_last_global_state()
            else:
                # TODO: 여기는 없으면 에러를 내는 게 낫지 않은지 (안그러면 샘플링 단계에서 매 스텝마다 다른 노이즈가 더해질 것이므로)
                ref_noisy_used, ref_eps = None, None
        except Exception:
            # TODO: 여기도 안되면 에러를 내는 게 낫지 않은지
            ref_noisy_used, ref_eps = None, None

        if (ref_noisy_used is not None) and (ref_eps is not None):
            # global branch is trained/returned in eps parameterization
            pred_x0_ref = (ref_noisy_used - sqrt_one_minus_at * ref_eps) / a_t.sqrt()

            # Optional quantization to VQ codebook for global branch as well
            if quantize_denoised:
                pred_x0_ref, _, *_ = self.model.first_stage_model.quantize(pred_x0_ref)
            
            # Parity with original DDIM: no dynamic thresholding
            dir_xt_ref = (1.0 - a_prev - sigma_t**2).sqrt() * ref_eps
            noise_ref = sigma_t * noise_like(ref_noisy_used.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.0:
                noise_ref = torch.nn.functional.dropout(noise_ref, p=noise_dropout)
            x_prev_ref = a_prev.sqrt() * pred_x0_ref + dir_xt_ref + noise_ref

            # keep for the next step and optional callback
            self._global_z = x_prev_ref.detach()
            self._global_pred_x0_last = pred_x0_ref.detach()

            if global_img_callback is not None:
                try:
                    global_img_callback(self._global_pred_x0_last, int(index))
                except Exception:
                    pass

        return x_prev, pred_x0
