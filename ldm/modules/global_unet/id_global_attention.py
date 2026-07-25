from __future__ import annotations

from typing import Dict, List, Optional

import torch


class GlobalAttentionControlIDWriter:
    """
    Writer controller for InstanceDiffusion UNet blocks.

    Caches tokens immediately AFTER the GatedSelfAttentionDense (fuser) in
    each BasicTransformerBlock so that fused global tokens (with UniFusion
    drop/mask/null handling applied) are stored. Reader side remains the
    standard LDM controller that concatenates cached tokens and runs attn1.

    Implementation notes
    - We wrap `block.fuser.forward` for every InstanceDiffusion
      BasicTransformerBlock found under its SpatialTransformer.
    - The wrapped fuser returns x_fused; we push that `x_fused` into the bank.
    - Does not modify reader behavior; only provides `bank` and bookkeeping.
    - Token scaling is intentionally NOT applied here (handled in reader).
    """

    def __init__(
        self,
        unet: torch.nn.Module,
        batch_size: int = 1,
        global_token_scale: float = 1.0,
        save_attention_maps: bool = False,
        detach_global_tokens: bool = False,
    ) -> None:
        self.unet = unet
        self.batch_size = int(batch_size)
        # Kept for API compatibility; scaling is applied on the reader side only.
        self.global_token_scale = float(global_token_scale)
        self.save_attention_maps = bool(save_attention_maps)
        self.detach_global_tokens = bool(detach_global_tokens)

        self.bank: Dict[int, List[torch.Tensor]] = {}
        self.blocks: List[torch.nn.Module] = []
        self._orig_fuser_forwards: Dict[int, torch.nn.Module] = {}
        self._register()

    # ---------------------------- public API -----------------------------
    def update(self, other: "GlobalAttentionControlIDWriter") -> None:
        self.bank = {k: list(v) for k, v in other.bank.items()}

    def clear(self) -> None:
        for k in list(self.bank.keys()):
            self.bank[k].clear()
        self.bank.clear()

    def remove(self) -> None:
        for idx, block in enumerate(self.blocks):
            fuser = getattr(block, "fuser", None)
            orig = getattr(fuser, "_global_fuser_orig", None)
            if fuser is not None and orig is not None:
                fuser.forward = orig  # type: ignore[attr-defined]
                delattr(fuser, "_global_fuser_orig")
        self.blocks.clear()
        self._orig_fuser_forwards.clear()
        self.clear()

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def get_debug_maps(self):  # for API parity only
        return {}

    # --------------------------- internal logic --------------------------
    def _register(self) -> None:
        # Duck-typing traversal: find modules that expose `transformer_blocks`,
        # and wrap any sub-block that has a `.fuser` module with a forward.
        block_index = 0
        for module in self.unet.modules():
            blocks = getattr(module, "transformer_blocks", None)
            if blocks is None:
                continue
            for block in list(blocks):
                if hasattr(block, "fuser") and hasattr(block.fuser, "forward"):
                    self._wrap_block_fuser(block, block_index)
                    self.blocks.append(block)
                    block_index += 1

    def _wrap_block_fuser(self, block: torch.nn.Module, block_index: int) -> None:
        fuser = getattr(block, "fuser", None)
        if fuser is None:
            return
        orig_forward = fuser.forward

        def wrapped_forward(x, objs, grounding_input=None, drop_box_mask=False):
            x_fused = orig_forward(x, objs, grounding_input=grounding_input, drop_box_mask=drop_box_mask)
            token = x_fused.detach() if self.detach_global_tokens else x_fused
            # global tokens into the main UNet attention.
            self.bank.setdefault(block_index, []).append(token)
            return x_fused

        fuser._global_fuser_orig = orig_forward  # type: ignore[attr-defined]
        fuser.forward = wrapped_forward  # type: ignore[assignment]
        self._orig_fuser_forwards[block_index] = orig_forward
