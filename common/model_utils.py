import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
from .image_processing import pil_to_tensor_normalized, torch_letterbox

def load_model_from_config(
    config_path: str,
    ckpt_path: str,
    device: torch.device,
    precision: str = "fp16",
    controlnet_path: str = None,
    init_controlnet_from_unet: bool = True,
):
    """
    Load model from config and checkpoint.

    Default behavior (when controlnet_path is None):
    - Initialize ControlNet the same way as at training start by copying
      compatible weights from UNet to ControlNet (time_embed, input_blocks,
      middle_block.*), while zero-conv layers remain zero-initialized.

    If controlnet_path is provided:
    - Load the ControlNet state dict from that file and apply it, leaving the
      SD (UNet/VAE/CLIP) weights from `ckpt_path` unchanged.
    """
    print(f"[Info] Loading config: {config_path}")
    cfg = OmegaConf.load(config_path)

    print(f"[Info] Instantiating model...")
    model = instantiate_from_config(cfg.model)

    print(f"[Info] Loading checkpoint...")
    state_dict = torch.load(ckpt_path, map_location="cpu")

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"[Info] Model loaded. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

    # Optionally initialize or load ControlNet weights
    try:
        has_control = hasattr(model, "control_model") and model.control_model is not None
    except Exception:
        has_control = False

    if has_control:
        if controlnet_path:
            # Load trained ControlNet weights
            print(f"[Info] Loading trained ControlNet weights: {controlnet_path}")
            cn_state = torch.load(controlnet_path, map_location="cpu")
            # Accept either a raw state_dict or wrapped dict
            if isinstance(cn_state, dict) and "state_dict" in cn_state and all(
                isinstance(k, str) for k in cn_state["state_dict"].keys()
            ):
                cn_state = cn_state["state_dict"]

            # torch.compile() can prepend "_orig_mod." to parameter names when exporting state_dict.
            # Strip this prefix automatically so exported ControlNet checkpoints can be reused for inference.
            if isinstance(cn_state, dict) and cn_state and all(isinstance(k, str) for k in cn_state.keys()):
                orig_mod_prefix = "_orig_mod."
                if all(k.startswith(orig_mod_prefix) for k in cn_state.keys()):
                    cn_state = {k[len(orig_mod_prefix):]: v for k, v in cn_state.items()}
                    print("[Info] Normalized ControlNet checkpoint keys by stripping '_orig_mod.' prefix.")

            try:
                model.control_model.load_state_dict(cn_state, strict=True)
                print("[Info] ControlNet weights loaded successfully (strict=True).")
            except RuntimeError as e:
                raise RuntimeError(
                    "ControlNet checkpoint load failed with strict=True. "
                    "Checkpoint does not match model.control_model exactly."
                ) from e
        elif init_controlnet_from_unet:
            # Initialize ControlNet from UNet like training startup
            try:
                unet = model.model.diffusion_model  # ControlledUnetModel (UnetModel-compatible)
                control = model.control_model       # ControlNet
                src_sd = unet.state_dict()
                dst_sd = control.state_dict()
                allowed_prefixes = ("time_embed", "input_blocks", "middle_block.")

                # Validate presence and shapes for the overlapping subsets
                expected = [k for k in dst_sd if k.startswith(allowed_prefixes)]
                missing_src = [k for k in expected if k not in src_sd]
                shape_mismatch = [k for k in expected if k in src_sd and dst_sd[k].shape != src_sd[k].shape]
                if missing_src:
                    print(f"[Warn] UNet->ControlNet init: missing {len(missing_src)} UNet keys; proceeding with available overlaps")
                if shape_mismatch:
                    print(f"[Warn] UNet->ControlNet init: {len(shape_mismatch)} shape mismatches; skipping those keys")

                copied = 0
                for k, v in src_sd.items():
                    if k.startswith(allowed_prefixes) and k in dst_sd and dst_sd[k].shape == v.shape:
                        dst_sd[k] = v.clone()
                        copied += 1
                control.load_state_dict(dst_sd, strict=False)
                print(f"[Info] Initialized ControlNet from UNet ({copied} tensors copied); zero-convs remain zero.")
            except Exception as e:
                print(f"[Warn] Failed to initialize ControlNet from UNet: {e}")
    model.eval().to(device)

    # reference/main weight copy disabled

    if precision == "fp16":
        model = model.half()
    elif precision == "bf16":
        model = model.to(torch.bfloat16)
    
    return model

def prepare_conditioning(
    model,
    x_img: torch.Tensor,   # (B, 3, H, W), [-1,1]
    x_mask: torch.Tensor,  # (B, 1, H, W), {0, 1}, 1 to fill
    prompts: List[str],
    negative_prompts: List[str] = None,
) -> Tuple[Dict, Dict]:
    """
    Build conditioning following gradio/inpainting.py logic
    """
    B, _, H, W = x_img.shape

    # create batch dict
    masked_image = x_img * (1.0 - x_mask)

    batch = {
        "image": x_img,
        "txt": prompts,
        "mask": x_mask,
        "masked_image": masked_image
    }

    # global text conditioning
    c = model.cond_stage_model.encode(prompts)

    # concatenated conditioning
    c_cat = list()
    for ck in model.concat_keys:  # masked_image, mask
        cc = batch[ck].float()
        if ck != model.masked_image_key:  # mask
            # downsample mask to latent size
            bchw = [B, 4, H//8, W//8]
            cc = torch.nn.functional.interpolate(cc, size=bchw[-2:], mode='nearest')
        else:
            # encode masked image to latents
            cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
        c_cat.append(cc)
    c_cat = torch.cat(c_cat, dim=1)

    # unconditional conditioning
    if negative_prompts is None:
        negative_prompts = [""] * B
    uc_cross = model.get_unconditional_conditioning(B, negative_prompts[0] if negative_prompts else "")

    # build final conditioning
    cond = {"c_concat": [c_cat], "c_crossattn": [c]}
    uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

    return cond, uc_full

def run_outpaint(
    local_rgb: torch.Tensor,
    local_mask: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    """
    Produce outpainted local patch.
    
    expected kwargs: model, sampler, steps, cfg, eta, seed, prompt, negative_prompt
    CHW normalized tensor inputs
    """
    model = kwargs["model"]
    sampler = kwargs["sampler"]
    steps = int(kwargs.get("steps", 30))
    cfg = float(kwargs.get("cfg", 7.0))
    eta = float(kwargs.get("eta", 0.0))
    seed = int(kwargs.get("seed", 42))
    prompt = kwargs.get("prompt", "")
    nprompt = kwargs.get("negative_prompt", "")

    device = local_rgb.device
    _, H, W = local_rgb.shape  # CHW
    
    # local patches are always 512x512 in progressive outpainting
    assert H == 512 and W == 512, f"Expected 512x512 local patch, got {H}x{W}"
    
    rgb_t = local_rgb.unsqueeze(0)  # CHW -> 1CHW
    mask_t = local_mask.unsqueeze(0).unsqueeze(0)  # HW -> 11HW
    
    # conditioning
    c, uc = prepare_conditioning(model=model,
        x_img=rgb_t,
        x_mask=mask_t,
        prompts=[prompt],
        negative_prompts=[nprompt] if nprompt else None,
    )

    # latent shape
    z_shape = (4, 512//8, 512//8)

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # Toggle global U-Net CFG handling to mirror IDM-VTON.
    cfg_enabled = cfg > 1.0
    cfg_toggle = hasattr(model, "set_global_cfg_mode")
    if cfg_toggle:
        model.set_global_cfg_mode(cfg_enabled)

    try:
        z_samples, _ = sampler.sample(
            S=steps,
            conditioning=c,
            batch_size=1,
            shape=z_shape,
            verbose=False,
            unconditional_guidance_scale=cfg,
            unconditional_conditioning=uc,
            eta=eta,
            x_T=None,
            generator=g,
        )
    finally:
        if cfg_toggle:
            model.set_global_cfg_mode(False)

    out = model.decode_first_stage(z_samples)  # (1,3,512,512), [-1,1]
    out = out.clamp(-1,1)
    out = out.squeeze(0)   # (3, 512, 512)
    
    return out

def prepare_conditioning_control(
    model,
    x_img: torch.Tensor,   # (B, 3, H, W), [-1,1]
    x_mask: torch.Tensor,  # (B, 1, H, W), {0, 1}, 1 to fill
    global_img: torch.Tensor,
    global_mask: torch.Tensor,
    prompts: List[str],                 # local prompt(s) for UNet
    negative_prompts: List[str] = None,
    control_prompts: List[str] = None,  # optional global prompt(s) for ControlNet
) -> Tuple[Dict, Dict]:
    """
    x_img: local image
    x_mask: local mask
    global_img: global canvas image
    global_mask: global mask

    Build conditioning following gradio/inpainting.py logic

    add 'c_control' condition for controlnet input (global image + global mask)
    """
    B, _, H, W = x_img.shape

    # create batch dict
    masked_image = x_img * (1.0 - x_mask)

    batch = {
        "image": x_img,
        "txt": prompts,
        "mask": x_mask,
        "masked_image": masked_image
    }

    # text conditioning (UNet: local)
    c_local = model.cond_stage_model.encode(prompts)

    # (optional) local prompt experiments removed; local prompts are provided per patch at inference

    # text conditioning for ControlNet (global prompt)
    c_control_txt = model.cond_stage_model.encode(control_prompts)

    # concatenated conditioning
    c_cat = list()
    for ck in model.concat_keys:  # masked_image, mask
        cc = batch[ck].float()
        if ck != model.masked_image_key:  # mask
            # downsample mask to latent size
            bchw = [B, 4, H//8, W//8]
            cc = torch.nn.functional.interpolate(cc, size=bchw[-2:], mode='nearest')
        else:
            # encode masked image to latents
            cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
        c_cat.append(cc)
    c_cat = torch.cat(c_cat, dim=1)

    # c_control (input for controlnet): global image + global mask
    c_control = torch.cat([global_img, global_mask], dim=1)

    # unconditional conditioning
    if negative_prompts is None:
        negative_prompts = [""] * B
    uc_cross = model.get_unconditional_conditioning(B, negative_prompts[0] if negative_prompts else "")

    # build final conditioning
    cond = {"c_concat": [c_cat], "c_crossattn": [c_local], "c_control": [c_control]}
    if c_control_txt is not None:
        cond["c_control_txt"] = [c_control_txt]
    uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross], "c_control": [c_control]}
    if c_control_txt is not None:
        uc_full["c_control_txt"] = [uc_cross]

    return cond, uc_full

def run_outpaint_control(
    local_rgb: torch.Tensor,
    local_mask: torch.Tensor,
    global_rgb: torch.Tensor,
    global_mask: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    """
    Produce outpainted local patch.
    
    expected kwargs: model, sampler, steps, cfg, eta, seed, prompt, negative_prompt
    CHW normalized tensor inputs
    """
    model = kwargs["model"]
    sampler = kwargs["sampler"]
    steps = kwargs["steps"]
    cfg = float(kwargs["cfg"])
    eta = float(kwargs["eta"])
    seed = int(kwargs["seed"])
    prompt = kwargs["prompt"]
    nprompt = kwargs["negative_prompt"]
    # optional split prompts
    local_prompt = kwargs.get("local_prompt")
    global_prompt = kwargs.get("global_prompt")

    assert local_prompt is not None, "Local prompt is None"
    assert global_prompt is not None, "Global prompt is None"

    device = local_rgb.device
    _, H, W = global_rgb.shape  # CHW
    target_size = 512

    # local patches are 512x512
    _, h, w = local_rgb.shape
    assert h == 512 and w == 512, f"Expected 512x512 local patch, got {h}x{w}"

    # apply letterbox to non-512x512 global image
    if H != 512 or W != 512:
        global_rgb_512, scale_rgb, (pad_left_rgb, pad_top_rgb) = torch_letterbox(global_rgb, target_size=target_size, mode='bilinear')
        global_mask_512, scale_m, (pad_left_m, pad_top_m) = torch_letterbox(global_mask, target_size=target_size, mode='nearest')

        # scale and pad must be same between rgb and mask
        assert scale_rgb==scale_m and pad_left_rgb==pad_left_m and pad_top_rgb==pad_top_m, "Scale and pad must be same between global rgb and mask."

        print(f"[Letterbox] Global image: {H}x{W} -> {target_size}x{target_size}, scale={scale_rgb:.3f}, pad=({pad_left_rgb}, {pad_top_rgb})")

        global_rgb_t = global_rgb_512.unsqueeze(0)  # CHW -> 1CHW
        global_mask_t = global_mask_512.unsqueeze(0).unsqueeze(0)  # HW -> 11HW
    else:
        # standard path
        global_rgb_t = global_rgb.unsqueeze(0)  # CHW -> 1CHW
        global_mask_t = global_mask.unsqueeze(0).unsqueeze(0)  # HW -> 11HW
    
    # local patches
    local_rgb_t = local_rgb.unsqueeze(0)  # CHW -> 1CHW
    local_mask_t = local_mask.unsqueeze(0).unsqueeze(0)  # HW -> 11HW
    
    # conditioning
    c, uc = prepare_conditioning_control(
        model=model,
        x_img=local_rgb_t,
        x_mask=local_mask_t,
        global_img=global_rgb_t,
        global_mask=global_mask_t,
        prompts=[local_prompt],
        control_prompts=[global_prompt],
        negative_prompts=[nprompt] if nprompt else None,
    )

    # latent shape
    z_shape = (4, target_size//8, target_size//8)

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    z_samples, _ = sampler.sample(
        S=steps,
        conditioning=c,
        batch_size=1,
        shape=z_shape,
        verbose=False,
        unconditional_guidance_scale=cfg,
        unconditional_conditioning=uc,
        eta=eta,
        x_T=None,
        generator=g,
    )

    out = model.decode_first_stage(z_samples)  # (1,3,512,512), [-1,1]
    out = out.clamp(-1,1)
    out = out.squeeze(0)   # (3, 512, 512)
    
    return out


# ===================== Global U-Net helpers =====================
def prepare_conditioning_global(
    model,
    x_img: torch.Tensor,   # (B, 3, H, W), [-1,1]
    x_mask: torch.Tensor,  # (B, 1, H, W), {0, 1}
    global_img: torch.Tensor, # (B, 3, H, W), [-1,1]
    prompts: List[str],
    negative_prompts: List[str] = None,
) -> Tuple[Dict, Dict]:
    """
    Conditioning for GlobalUNetOutpaintDiffusion:
    - c_concat: [masked_image(latent), mask(downsampled)]
    - c_crossattn: text embeddings
    - ref_latent: VAE-encoded global image latent (4ch)
    - uc: unconditional for CFG
    """
    B, _, H, W = x_img.shape
    device = x_img.device

    masked_image = x_img * (1.0 - x_mask)

    # text conditioning
    c_text = model.cond_stage_model.encode(prompts)

    # concat conditioning
    c_cat = []
    # encode masked image to latents
    masked_latent = model.get_first_stage_encoding(model.encode_first_stage(masked_image))
    c_cat.append(masked_latent)
    # downsample mask to latent size
    mask_ds = torch.nn.functional.interpolate(x_mask.float(), size=masked_latent.shape[-2:], mode='nearest')
    c_cat.append(mask_ds)
    c_cat = torch.cat(c_cat, dim=1)

    # global latent
    ref_latent = model.get_first_stage_encoding(model.encode_first_stage(global_img))

    # unconditional (IDM-VTON style): provide zero reference for uc branch
    if negative_prompts is None:
        negative_prompts = [""] * B
    uc_text = model.get_unconditional_conditioning(B, negative_prompts[0])

    cond = {"c_concat": [c_cat], "c_crossattn": [c_text], "ref_latent": [ref_latent]}

    # zero global latent for uc so sampler concatenation results in [0_global | global]
    zero_ref = torch.zeros_like(ref_latent)
    uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_text], "ref_latent": [zero_ref]}
    return cond, uc_full


def run_outpaint_global(
    local_rgb: torch.Tensor,
    local_mask: torch.Tensor,
    global_rgb: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """
    Produce outpainted local patch using global U-Net-based diffusion.

    expected kwargs: model, sampler, steps, cfg, eta, seed, prompt, negative_prompt
    """
    model = kwargs["model"]
    sampler = kwargs["sampler"]
    steps = int(kwargs.get("steps", 30))
    cfg = float(kwargs.get("cfg", 7.0))
    eta = float(kwargs.get("eta", 0.0))
    seed = int(kwargs.get("seed", 42))
    prompt = kwargs.get("prompt", "")
    nprompt = kwargs.get("negative_prompt", "")

    device = local_rgb.device
    _, h, w = local_rgb.shape
    assert h == 512 and w == 512, f"Expected 512x512 local patch, got {h}x{w}"

    # batchify
    local_rgb_t = local_rgb.unsqueeze(0)
    local_mask_t = local_mask.unsqueeze(0).unsqueeze(0)
    global_rgb_t = global_rgb.unsqueeze(0)

    # conditioning
    c, uc = prepare_conditioning_global(
        model=model,
        x_img=local_rgb_t,
        x_mask=local_mask_t,
        global_img=global_rgb_t,
        prompts=[prompt],
        negative_prompts=[nprompt] if nprompt else None,
    )

    z_shape = (4, h // 8, w // 8)
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    cfg_toggle = hasattr(model, "set_global_cfg_mode")
    if cfg_toggle:
        model.set_global_cfg_mode(cfg > 1.0)

    try:
        z_samples, _ = sampler.sample(
            S=steps,
            conditioning=c,
            batch_size=1,
            shape=z_shape,
            verbose=False,
            unconditional_guidance_scale=cfg,
            unconditional_conditioning=uc,
            eta=eta,
            x_T=None,
            generator=g,
        )
    finally:
        if cfg_toggle:
            model.set_global_cfg_mode(False)

    out = model.decode_first_stage(z_samples).clamp(-1, 1).squeeze(0)
    return out
