from typing import Dict, List, Optional

import torch

from ldm.modules.attention import SpatialTransformer, BasicTransformerBlock


class GlobalAttentionControlLDM:
    """
    Lightweight attention wrapper for CompVis/LDM UNet blocks.

    Injection style: concat + self-attention (AnimateAnyone/IDM-VTON style)
    - write mode: cache self-attention inputs (normalized hidden states) from the
      global U-Net at every BasicTransformerBlock.attn1.
    - read mode: in the main UNet, concatenate [current_hidden | cached_global_hidden]
      along the sequence dimension and run attn1 in self-attention mode. Only the
      first slice corresponding to current_hidden is returned, matching LDM's
      BasicTransformerBlock residual expectation.

    Notes
    - Targets CompVis/LDM BasicTransformerBlock (ldm.modules.attention).
    - Works for 2D UNet; no temporal duplication is performed here.
    - Optional CFG handling: if do_classifier_free_guidance=True and batch is doubled
      (uncond first, cond second), unconditional half attends only to itself (i.e.,
      self-attention over current tokens without global tokens).
    """

    def __init__(
        self,
        unet: torch.nn.Module,
        mode: str = "write",
        batch_size: int = 1,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        # Injection controls
        global_token_scale: float = 1.0,
        save_attention_maps: bool = False,
        detach_global_tokens: bool = True,
    ) -> None:
        assert mode in ("write", "read"), "mode must be 'write' or 'read'"
        self.unet = unet
        self.mode = mode
        self.batch_size = batch_size
        self.num_images_per_prompt = num_images_per_prompt
        self.device = device or torch.device("cpu")
        self.global_token_scale = float(global_token_scale)
        self.save_attention_maps = bool(save_attention_maps)
        self.detach_global_tokens = bool(detach_global_tokens)

        # Per-block token banks: {block_index: [Tensor, Tensor, ...]}
        self.bank: Dict[int, List[torch.Tensor]] = {}

        # Keep references to wrapped blocks and original forwards for clean restore
        self.blocks: List[BasicTransformerBlock] = []
        self._orig_forwards: Dict[int, torch.nn.Module] = {}

        self._register()

        # Debug attention maps storage: {block_idx: Tensor[B, N_x, N_ref]}
        self.debug_attn_maps: Dict[int, torch.Tensor] = {}

    # ---------------------------- public API -----------------------------
    def update(self, other: "GlobalAttentionControlLDM") -> None:
        """Copy cached global features from another controller (writer → reader)."""
        self.bank = {k: list(v) for k, v in other.bank.items()}

    def clear(self) -> None:
        for k in list(self.bank.keys()):
            self.bank[k].clear()
        self.bank.clear()

    def set_mode(self, mode: str) -> None:
        assert mode in ("write", "read")
        self.mode = mode

    def remove(self) -> None:
        """Restore original attn1.forward for all wrapped blocks."""
        for idx, block in enumerate(self.blocks):
            attn1 = getattr(block, "attn1", None)
            orig = getattr(attn1, "_global_attn1_orig", None)
            if attn1 is not None and orig is not None:
                attn1.forward = orig  # type: ignore[attr-defined]
                delattr(attn1, "_global_attn1_orig")
        self.blocks.clear()
        self._orig_forwards.clear()
        self.clear()
        self.debug_attn_maps.clear()

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def get_debug_maps(self) -> Dict[int, torch.Tensor]:
        return dict(self.debug_attn_maps)

    # --------------------------- internal logic --------------------------
    def _register(self) -> None:
        """Find all BasicTransformerBlock under SpatialTransformer and wrap attn1."""
        block_index = 0
        for module in self.unet.modules():
            if isinstance(module, SpatialTransformer):
                # Each SpatialTransformer contains a list of BasicTransformerBlock
                for block in getattr(module, "transformer_blocks", []):
                    if not isinstance(block, BasicTransformerBlock):
                        continue
                    self._wrap_block_attn1(block, block_index)
                    self.blocks.append(block)
                    block_index += 1

    def _wrap_block_attn1(self, block: BasicTransformerBlock, block_index: int) -> None:
        attn1 = getattr(block, "attn1", None)
        if attn1 is None:
            return
        orig_forward = attn1.forward
        def wrapped_forward(
            x: torch.FloatTensor, context: Optional[torch.FloatTensor] = None, mask=None
        ) -> torch.FloatTensor:
            # x is the normalized hidden states at this block [B, N, C]
            if self.mode == "write":
                # Cache global tokens
                token = x.detach() if self.detach_global_tokens else x
                self.bank.setdefault(block_index, []).append(token)
                
                # Preserve original behavior in writer mode
                return orig_forward(x, context=context, mask=mask)

            if self.mode == "read":
                bank_list = self.bank.get(block_index, [])
                if not bank_list:
                    # No global tokens; fall back to original
                    return orig_forward(x, context=context, mask=mask)

                # Concatenate current tokens with cached global tokens along sequence dim
                # 여러 토큰들이 생성될 때를 가정. 현재 프레임워크에서는 그냥 bank_list[0]만 사용된다고 볼 수 있음.
                ref_tokens = (
                    torch.cat(bank_list, dim=1) if len(bank_list) > 1 else bank_list[0]
                )
                # Keep default autograd behavior from caller. Do not force-enable grads here
                # to avoid accidentally creating graphs during inference paths.
                ref_tokens = ref_tokens.to(device=x.device, dtype=x.dtype, non_blocking=True)

                # Apply token-level scaling to global part
                # global token의 영향을 조절
                if self.global_token_scale != 1.0:
                    ref_tokens = ref_tokens * self.global_token_scale

                # Self-attention over [x | ref_tokens]
                x_cat = torch.cat([x, ref_tokens], dim=1)
                out_cat = orig_forward(x_cat, context=None, mask=mask)

                # Keep only the slice corresponding to original x (match residual add upstream)
                # x.shape[1]은 기존 spatial 크기이므로, ref 부분은 버리고 기존 토큰만큼만 가져감.
                out = out_cat[:, : x.shape[1], :]

                # Optional: save attention maps (only supported for non-xformers CrossAttention)
                if self.save_attention_maps:
                    attn1 = getattr(block, "attn1", None)
                    try:
                        from ldm.modules.attention import CrossAttention
                    except Exception:
                        CrossAttention = None
                    if CrossAttention is not None and isinstance(attn1, CrossAttention):
                        # Recompute attention probs for debug from x_cat
                        h = attn1.heads
                        q = attn1.to_q(x_cat)
                        k = attn1.to_k(x_cat)
                        # reshape to (B*h, N, d)
                        q = q.view(q.shape[0], q.shape[1], h, -1).permute(0,2,1,3).reshape(-1, q.shape[1], q.shape[2]//h)
                        k = k.view(k.shape[0], k.shape[1], h, -1).permute(0,2,1,3).reshape(-1, k.shape[1], k.shape[2]//h)
                        # only queries from original x tokens
                        N_x = x.shape[1]
                        N_total = x_cat.shape[1]
                        N_ref = N_total - N_x
                        q_x = q[:, :N_x, :]
                        # scaled dot-product attention scores
                        sim = torch.einsum('bij,bkj->bik', q_x, k) * attn1.scale
                        attn = torch.softmax(sim, dim=-1)
                        # take columns corresponding to ref tokens and average heads
                        attn = attn.view(x.shape[0], h, N_x, N_total)
                        attn_ref = attn[:, :, :, N_x:]
                        attn_ref = attn_ref.mean(dim=1)  # (B, N_x, N_ref)
                        self.debug_attn_maps[block_index] = attn_ref.detach().cpu()
                return out

            # default fall-through
            return orig_forward(x, context=context, mask=mask)

        # bind
        attn1._global_attn1_orig = orig_forward  # type: ignore[attr-defined]
        attn1.forward = wrapped_forward  # type: ignore[assignment]
        self._orig_forwards[block_index] = orig_forward
