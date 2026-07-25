import torch
import torch.nn as nn


class DCAEWrapper(nn.Module):
    """
    Lightweight wrapper around DC‑AE for reference-branch use.

    Backend:
      - diffusers AutoencoderDC (diffusers-formatted DC‑AE repositories)

    Behavior:
      - Lazily loads model on first encode/decode.
      - encode()/decode() run in FP32 by default (autocast off) for stability.
      - scale_factor is kept as a buffer; auto-read from diffusers config if present,
        otherwise use the provided value (e.g., 0.41407 for SANA f32c32).
    """

    def __init__(
        self,
        repo_or_path: str,
        subfolder: str | None = None,
        scale_factor: float | None = None,
        local_files_only: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        force_fp32: bool = True,
        backend: str = "diffusers",  # default: diffusers-only
        config_repo_or_path: str | None = None,  # used for diffusers from_single_file fallback
    ) -> None:
        super().__init__()
        self.repo_or_path = repo_or_path
        self.subfolder = subfolder
        self.local_files_only = bool(local_files_only)
        self._model = None  # lazy
        self._decode_available = True
        self.register_buffer(
            "scale_factor",
            torch.tensor(scale_factor if scale_factor is not None else 1.0, dtype=torch.float32),
            persistent=True,
        )
        self._calibrated = bool(scale_factor is not None)
        # Heuristic: if no scale provided and repo hints f32c32 SANA, prefill 0.41407 for visibility
        if not self._calibrated:
            repo_str = f"{self.repo_or_path}|{self.subfolder or ''}"
            if isinstance(repo_str, str) and 'f32c32' in repo_str:
                self.scale_factor.data[...] = 0.41407
                # keep _calibrated = False so that config can still override later if present
        self._device_hint = device
        self._dtype_hint = dtype
        self.force_fp32 = bool(force_fp32)
        self.backend_pref = str(backend).lower()
        self._backend_used = None  # set after _lazy_load
        self.config_repo_or_path = config_repo_or_path

    # ------------------------ internals ------------------------
    def _lazy_load(self):
        if self._model is not None:
            return
        from diffusers import AutoencoderDC
        model = None
        load_exc = None

        # Prefer single-file path whenever a config repo is provided to avoid noisy fallbacks
        prefer_single = bool(self.config_repo_or_path)

        if not prefer_single:
            # 1) Try diffusers repo directly (quiet: disable low_cpu_mem_usage warnings)
            try:
                model = AutoencoderDC.from_pretrained(
                    self.repo_or_path,
                    subfolder=self.subfolder,
                    torch_dtype=torch.float32 if self.force_fp32 else (self._dtype_hint or torch.float32),
                    local_files_only=self.local_files_only,
                    low_cpu_mem_usage=False,  # avoid accelerate warning log
                )
            except TypeError:
                # older diffusers without low_cpu_mem_usage kwarg
                try:
                    model = AutoencoderDC.from_pretrained(
                        self.repo_or_path,
                        subfolder=self.subfolder,
                        torch_dtype=torch.float32 if self.force_fp32 else (self._dtype_hint or torch.float32),
                        local_files_only=self.local_files_only,
                    )
                except Exception as e:
                    load_exc = e
                    model = None
            except Exception as e:
                load_exc = e
                model = None

        # 2) Single-file checkpoint (original format) with diffusers config (preferred path)
        if model is None:
            # Determine checkpoint path: use explicit file or fetch from HF Hub
            ckpt_path = self.repo_or_path
            if isinstance(self.repo_or_path, str) and not (
                self.repo_or_path.endswith('.safetensors') or self.repo_or_path.endswith('.bin')
            ):
                from huggingface_hub import hf_hub_download
                # Try safetensors first (to avoid pickle warning), then common .bin filenames
                tried = []
                for fname in ('model.safetensors', 'pytorch_model.bin', 'model.bin'):
                    try:
                        ckpt_path = hf_hub_download(
                            repo_id=self.repo_or_path,
                            filename=fname,
                            subfolder=self.subfolder if self.subfolder else None,
                            local_files_only=self.local_files_only,
                        )
                        break
                    except Exception as e_dl:
                        tried.append((fname, repr(e_dl)))
                        ckpt_path = None
                if ckpt_path is None:
                    detail = ", ".join([f"{n}: {e}" for n, e in tried])
                    raise RuntimeError(
                        f"Failed to download DC-AE checkpoint from '{self.repo_or_path}' (tried {detail})"
                    )

            # Choose a config repo for architecture (defaults to SANA f32c32 1.0 diffusers)
            cfg_repo = self.config_repo_or_path or "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers"
            try:
                try:
                    model = AutoencoderDC.from_single_file(
                        ckpt_path,
                        config=cfg_repo,
                        torch_dtype=torch.float32 if self.force_fp32 else (self._dtype_hint or torch.float32),
                        local_files_only=self.local_files_only,
                        low_cpu_mem_usage=False,  # avoid accelerate warning log
                    )
                except TypeError:
                    model = AutoencoderDC.from_single_file(
                        ckpt_path,
                        config=cfg_repo,
                        torch_dtype=torch.float32 if self.force_fp32 else (self._dtype_hint or torch.float32),
                        local_files_only=self.local_files_only,
                    )
            except Exception as e2:
                hint = (
                    "\nTried diffusers repo and single-file fallback. "
                    "If this is an original (efficientvit) repo, either provide a diffusers-formatted repo "
                    "or specify a direct checkpoint path plus a compatible diffusers config."
                )
                raise RuntimeError(
                    f"Failed to load DC-AE for '{self.repo_or_path}'. from_pretrained error={load_exc!r}; "
                    f"from_single_file error={e2!r}{hint}"
                ) from e2

        # scaling factor from config if available; otherwise apply conservative fallback for SANA f32c32
        if not self._calibrated:
            cfg_scale = None
            try:
                if hasattr(model, 'config') and hasattr(model.config, 'scaling_factor'):
                    cfg_scale = float(model.config.scaling_factor)
            except Exception:
                cfg_scale = None
            if cfg_scale is not None and cfg_scale > 0 and abs(cfg_scale - 1.0) > 1e-6:
                self.scale_factor.data[...] = cfg_scale
                self._calibrated = True
            else:
                # Heuristic fallback: SANA f32c32 commonly uses 0.41407
                repo_str = f"{self.repo_or_path}|{self.config_repo_or_path or ''}"
                if isinstance(repo_str, str) and 'f32c32' in repo_str:
                    self.scale_factor.data[...] = 0.41407
                    self._calibrated = True

        self._model, self._backend_used = model, 'diffusers'

        # Move to requested device if provided
        if self._device_hint is not None:
            self._model.to(self._device_hint)

        # DC-AE weights are inference-time; keep in eval and frozen
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False

        # Detect decode availability (some minimal exports may not include it)
        self._decode_available = hasattr(self._model, "decode")

    # ------------------------ API ------------------------
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._lazy_load()
        # Expect BCHW images in [-1,1] or [0,1]; keep caller responsible for range.
        # Force compute in FP32 when requested; return in original dtype for pipeline compatibility.
        orig_dtype = x.dtype
        # choose autocast device type based on input device
        device_type = 'cuda' if x.is_cuda else 'cpu'
        use_autocast = (not self.force_fp32)
        with torch.autocast(device_type=device_type, enabled=use_autocast):
            x32 = x.to(dtype=torch.float32)
            z = self._model.encode(x32)
        # diffusers may return objects, extract tensor
        if not torch.is_tensor(z):
            if hasattr(z, 'latent'):
                z = z.latent
            elif hasattr(z, 'latents'):
                z = z.latents
            elif isinstance(z, (tuple, list)):
                z = z[0]

        z_scaled = self.scale_factor * z
        # Return in original dtype to reduce downstream casts/memory
        return z_scaled.to(dtype=orig_dtype)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self._lazy_load()
        if not self._decode_available:
            raise RuntimeError("DC-AE decode() is not available in this build.")
        # Invert scaling and force compute in FP32 for stability; keep output in FP32 for logging quality
        device_type = 'cuda' if z.is_cuda else 'cpu'
        use_autocast = (not self.force_fp32)
        with torch.autocast(device_type=device_type, enabled=use_autocast):
            z32 = (z / (self.scale_factor + 1e-12)).to(dtype=torch.float32)
            img = self._model.decode(z32)
        # diffusers may return DecoderOutput-like object; extract tensor
        if not torch.is_tensor(img):
            if hasattr(img, 'sample'):
                img = img.sample
            elif hasattr(img, 'samples'):
                img = img.samples
            elif isinstance(img, (tuple, list)) and len(img) > 0:
                img = img[0]
        return img
