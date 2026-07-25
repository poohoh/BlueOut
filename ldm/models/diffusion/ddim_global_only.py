"""Stage 1 DDIM sampler with InstanceDiffusion-style CFG.

This sampler is intended for Stage 1 models where `model.apply_model(x, t, cond)`
returns an eps/noise prediction produced by the Stage 1 global U-Net.

CFG matches InstanceDiffusion:
  eps = eps_uncond + scale * (eps_cond - eps_uncond)

Because our conditioning is a dict (layout + ref inputs), the unconditional pass
is built by copying required non-layout keys (e.g. `ref_input`, `ref_denoise_latent`)
and nulling layout keys by default (`ref_boxes`, `ref_masks`, `ref_positive_embeddings`).
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
)


class DDIMGlobalOnlySampler(object):
    def __init__(self, model, schedule: str = "linear", device: torch.device | None = None, **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.device = device or getattr(model, "device", torch.device("cuda"))

        self._global_pred_x0_last = None  # last pred_x0 for optional callback

    def _build_unconditional_conditioning(self, conditioning, unconditional_conditioning):
        """
        Build an unconditional conditioning dict in the style of InstanceDiffusion CFG:
          - unconditional text context
          - null/empty layout (boxes/masks/positive_embeddings)

        This is intentionally conservative:
          - required non-layout keys (e.g. ref_input/ref_denoise_latent) are copied from `conditioning`
          - layout keys are zeroed unless explicitly provided in `unconditional_conditioning`
          - optional attention-mask keys are dropped by default (to avoid masking everything)
        """
        if conditioning is None:
            return unconditional_conditioning

        if not isinstance(conditioning, dict):
            # fall back to vanilla DDIM semantics for non-dict conditionings
            return unconditional_conditioning

        if unconditional_conditioning is None:
            return None

        if not isinstance(unconditional_conditioning, dict):
            # Allow passing only the unconditional text embedding tensor/list; convert to dict.
            if isinstance(unconditional_conditioning, (list, tuple)) and unconditional_conditioning and isinstance(
                unconditional_conditioning[0], torch.Tensor
            ):
                unconditional_conditioning = {"ref_c_txt": list(unconditional_conditioning)}
            else:
                unconditional_conditioning = {"ref_c_txt": [unconditional_conditioning]}

        # Start from a shallow copy of the conditional dict so required keys are present.
        uc = dict(conditioning)

        # --- Text context ---
        # Prefer explicit ref_c_txt, fall back to c_crossattn if caller reused SD-style naming.
        if "ref_c_txt" in unconditional_conditioning and unconditional_conditioning["ref_c_txt"] is not None:
            txt = unconditional_conditioning["ref_c_txt"]
            if isinstance(txt, torch.Tensor):
                txt = [txt]
            uc["ref_c_txt"] = txt
        elif "c_crossattn" in unconditional_conditioning and unconditional_conditioning["c_crossattn"] is not None:
            txt = unconditional_conditioning["c_crossattn"]
            if isinstance(txt, torch.Tensor):
                txt = [txt]
            uc["ref_c_txt"] = txt

        # --- Layout / grounding (null by default) ---
        layout_keys = ("ref_boxes", "ref_masks", "ref_positive_embeddings")
        for k in layout_keys:
            if k in unconditional_conditioning and unconditional_conditioning[k] is not None:
                uc[k] = unconditional_conditioning[k]
                continue
            if k in conditioning and conditioning[k]:
                base = conditioning[k][0]
                if isinstance(base, torch.Tensor):
                    uc[k] = [torch.zeros_like(base)]

        # Optional: do NOT carry over instance attention masks for unconditional by default.
        # (InstanceDiffusion unconditional pass omits grounding_input entirely.)
        if "ref_att_masks" in unconditional_conditioning:
            uc["ref_att_masks"] = unconditional_conditioning["ref_att_masks"]
        else:
            uc["ref_att_masks"] = []

        # Keep visual_valid_mask as-is (it encodes letterbox validity / padding, not a layout condition).
        # If the caller provided an override, respect it.
        if "visual_valid_mask" in unconditional_conditioning:
            uc["visual_valid_mask"] = unconditional_conditioning["visual_valid_mask"]

        return uc

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

        # NOTE(Stage 1 CFG):
        # We implement CFG in this sampler via two separate model passes (cond/uncond),
        # matching InstanceDiffusion inference behavior. Therefore we must NOT enable
        # model-side "CFG slicing" helpers (which assume [uc|cond] concatenated batches).
        if hasattr(self.model, "set_global_cfg_mode"):
            self.model.set_global_cfg_mode(False)

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
        self._global_pred_x0_last = None

        time_range = np.flip(self.ddim_timesteps)
        total_steps = self.ddim_timesteps.shape[0]
        print(f"Running DDIMGlobalOnly Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="DDIMGlobalOnly Sampler", total=total_steps)

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
                    # see NOTE(Stage 1 CFG) above: keep model-side CFG slicing disabled
                    if hasattr(self.model, "set_global_cfg_mode"):
                        self.model.set_global_cfg_mode(False)

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
            if hasattr(self.model, "set_global_cfg_mode"):
                self.model.set_global_cfg_mode(False)

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

        # CFG scale sanity: must be a finite scalar.
        if isinstance(unconditional_guidance_scale, torch.Tensor):
            if unconditional_guidance_scale.numel() != 1:
                raise ValueError(
                    f"unconditional_guidance_scale must be a scalar, got shape={tuple(unconditional_guidance_scale.shape)}"
                )
            unconditional_guidance_scale = float(unconditional_guidance_scale.item())
        else:
            unconditional_guidance_scale = float(unconditional_guidance_scale)
        if not np.isfinite(unconditional_guidance_scale):
            raise ValueError(f"unconditional_guidance_scale must be finite, got {unconditional_guidance_scale}")
        if unconditional_guidance_scale < 0.0:
            raise ValueError(f"unconditional_guidance_scale must be >= 0, got {unconditional_guidance_scale}")
        if unconditional_guidance_scale != 1.0 and unconditional_conditioning is None:
            raise ValueError(
                "CFG requested (unconditional_guidance_scale != 1) but unconditional_conditioning is None."
            )

        # ----- Stage 1 forward with CFG (InstanceDiffusion-style) -----
        eps_cond = self.model.apply_model(x, t, c, return_global_prediction=False)
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            model_output = eps_cond
        else:
            uc = self._build_unconditional_conditioning(c, unconditional_conditioning)
            eps_uncond = self.model.apply_model(x, t, uc, return_global_prediction=False)
            model_output = eps_uncond + float(unconditional_guidance_scale) * (eps_cond - eps_uncond)

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

        # Track last pred_x0 for optional callback compatibility
        self._global_pred_x0_last = pred_x0.detach()
        if global_img_callback is not None:
            global_img_callback(self._global_pred_x0_last, int(index))

        return x_prev, pred_x0
