from __future__ import annotations

from typing import Dict, Optional, Any

import torch
from ldm.util import instantiate_from_config


class IDGlobalUNetWrapper(torch.nn.Module):
    """
    Thin wrapper around InstanceDiffusion UNet so it can be used as a
    Stage 1 global U-Net drop‑in.

    - Keeps the call signature: forward(x, timesteps, context)
    - Exposes forward_with_grounding(...) to pass UniFusion inputs explicitly
    - Holds the inner ID UNet at `self.inner`

    Grounding input contract (UniFusion / GroundingNetInput style):
      boxes:  Float[B,N,4]
      masks:  Float[B,N,1] with 1.0 for valid tokens, 0.0 for padded
      positive_embeddings: Float[B,N,768] per‑instance text feature
      scribbles|polygons|segs|points: may be provided; otherwise zeros

    If grounding is not provided, zeros are used so the wrapper is safe
    to call in existing pipelines until the dataset supplies real inputs.
    """

    is_instance_diffusion: bool = True

    def __init__(self, unet_config: Dict[str, Any], grounding_tokenizer: Optional[Dict[str, Any]] = None):
        super().__init__()
        # Expect an ID UNet config (InstanceDiffusion.*.openaimodel.UNetModel)
        # The UNet itself expects a `grounding_tokenizer` submodule (UniFusion)
        # via `instantiate_from_config` in its __init__.
        if grounding_tokenizer is not None:
            # Inject tokenizer config into unet_config params if not present
            unet_config = dict(unet_config)  # shallow copy
            params = dict(unet_config.get("params", {}))
            params.setdefault("grounding_tokenizer", grounding_tokenizer)
            unet_config["params"] = params

        self.inner: torch.nn.Module = instantiate_from_config(unet_config)

    # --------------------------- helpers ---------------------------
    @staticmethod
    def _zeros_like(batch: int, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros((batch, *shape), device=device, dtype=dtype)

    def _make_null_grounding(self, b: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
        # Minimal null with N=1
        N = 1
        return {
            "boxes": torch.zeros(b, N, 4, device=device, dtype=dtype),
            "masks": torch.zeros(b, N, 1, device=device, dtype=dtype),
            "positive_embeddings": torch.zeros(b, N, 768, device=device, dtype=dtype),
            # BBOX-ONLY: unused modalities commented out
            # "scribbles": torch.zeros(b, N, 20, 2, device=device, dtype=dtype),
            # "polygons": torch.zeros(b, N, 256, 2, device=device, dtype=dtype),
            # "segs": torch.zeros(b, 1, 64, 64, device=device, dtype=dtype),
            # "points": torch.zeros(b, N, 2, device=device, dtype=dtype),
        }

    def forward_with_grounding(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor],
        grounding_input: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Require explicit grounding_input — upstream builds zeros when needed
        if grounding_input is None:
            raise ValueError(
                "IDGlobalUNetWrapper.forward_with_grounding requires 'grounding_input'. "
                "Provide boxes/masks/positive_embeddings (zeros allowed)."
            )

        # Build ID input dict
        model_input = {
            "x": x,
            "timesteps": timesteps,
            "context": context,
            "grounding_input": grounding_input,
        }
        return self.inner(model_input)

    # Default LDM‑style call (uses null grounding)
    def forward(self, x, timesteps, context=None):  # noqa: D401
        raise RuntimeError(
            "IDGlobalUNetWrapper.forward called without grounding_input. "
            "Use forward_with_grounding(x, t, context, grounding_input=...) with boxes/masks/positive_embeddings."
        )
