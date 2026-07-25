from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


TensorBank = Dict[int, List[torch.Tensor]]


def _run_fuser_main_query_only(
    *,
    fuser: torch.nn.Module,
    hidden_states: torch.Tensor,
    kv_visual_tokens: torch.Tensor,
    objs: torch.Tensor,
    block_index: int,
    kwargs: Dict[str, object],
) -> torch.Tensor:
    if kwargs:
        raise TypeError(
            "main_query_only mode does not support forwarding custom fuser kwargs. "
            f"got keys={sorted(kwargs.keys())} at block_index={block_index}."
        )

    required = ("enabled", "linear", "norm1", "norm2", "attn", "ff", "alpha_attn", "alpha_dense", "scale")
    missing = [name for name in required if not hasattr(fuser, name)]
    if missing:
        raise TypeError(
            "main_query_only requires a diffusers-style GatedSelfAttentionDense-compatible fuser. "
            f"Missing attrs={missing} at block_index={block_index}."
        )

    if not bool(getattr(fuser, "enabled")):
        return hidden_states

    objs_qdim = fuser.linear(objs)
    kv_tokens = torch.cat([kv_visual_tokens, objs_qdim], dim=1)

    q_tokens = fuser.norm1(hidden_states)
    kv_tokens = fuser.norm1(kv_tokens)
    attn_out = fuser.attn(q_tokens, encoder_hidden_states=kv_tokens)

    scale = fuser.scale
    x = hidden_states + scale * fuser.alpha_attn.tanh() * attn_out
    x = x + scale * fuser.alpha_dense.tanh() * fuser.ff(fuser.norm2(x))
    return x


class DiffusersGlobalUNetFuserWriter:
    """
    GlobalUNet writer for diffusers-style InstanceDiffusion UNets.

    - Wraps every transformer block's `fuser.forward` and caches its output tokens.
    - The cached tokens are later consumed by a reader hook on the main UNet.
    """

    def __init__(
        self,
        unet: torch.nn.Module,
        *,
        detach_global_tokens: bool = True,
        main_query_only: bool = False,
    ) -> None:
        self.unet = unet
        self.detach_global_tokens = bool(detach_global_tokens)
        self.main_query_only = bool(main_query_only)

        self.bank: TensorBank = {}
        self._wrapped: List[Tuple[int, torch.nn.Module]] = []

        self._register()

    @property
    def num_blocks(self) -> int:
        return len(self._wrapped)

    def clear(self) -> None:
        self.bank.clear()

    def remove(self) -> None:
        for _, fuser in self._wrapped:
            orig = getattr(fuser, "_global_unet_writer_orig_forward", None)
            if orig is not None:
                fuser.forward = orig  # type: ignore[assignment]
                delattr(fuser, "_global_unet_writer_orig_forward")
        self._wrapped.clear()
        self.clear()

    def _register(self) -> None:
        block_index = 0
        for module in self.unet.modules():
            blocks = getattr(module, "transformer_blocks", None)
            if blocks is None:
                continue
            for block in list(blocks):
                fuser = getattr(block, "fuser", None)
                if fuser is None or not hasattr(fuser, "forward"):
                    continue
                self._wrap_fuser(fuser, block_index)
                self._wrapped.append((block_index, fuser))
                block_index += 1

    def _wrap_fuser(self, fuser: torch.nn.Module, block_index: int) -> None:
        orig_forward = fuser.forward

        def wrapped_forward(x, objs, *args, **kwargs):
            if not self.main_query_only:
                out = orig_forward(x, objs, *args, **kwargs)
            else:
                if args:
                    raise TypeError(
                        "Unexpected positional args to fuser.forward in main_query_only mode; "
                        "update writer wrapper if needed."
                    )
                out = _run_fuser_main_query_only(
                    fuser=fuser,
                    hidden_states=x,
                    kv_visual_tokens=x,
                    objs=objs,
                    block_index=block_index,
                    kwargs=kwargs,
                )
            token = out.detach() if self.detach_global_tokens else out
            self.bank.setdefault(block_index, []).append(token)
            return out

        fuser._global_unet_writer_orig_forward = orig_forward  # type: ignore[attr-defined]
        fuser.forward = wrapped_forward  # type: ignore[assignment]


class DiffusersGlobalUNetCrossAttnWriter:
    """
    GlobalUNet writer variant for diffusers-style InstanceDiffusion UNets.

    - Wraps every transformer block's `norm3.forward`.
    - Caches the incoming tokens to `norm3`, which correspond to hidden states
      immediately AFTER cross-attention residual addition and BEFORE the FFN.
    - Keeps the same public API as the fuser writer so training scripts can
      swap extraction points without touching the injector path.
    """

    def __init__(
        self,
        unet: torch.nn.Module,
        *,
        detach_global_tokens: bool = True,
        main_query_only: bool = False,
    ) -> None:
        self.unet = unet
        self.detach_global_tokens = bool(detach_global_tokens)
        # Kept for API parity with DiffusersGlobalUNetFuserWriter.
        self.main_query_only = bool(main_query_only)

        self.bank: TensorBank = {}
        self._wrapped: List[Tuple[int, torch.nn.Module]] = []

        self._register()

    @property
    def num_blocks(self) -> int:
        return len(self._wrapped)

    def clear(self) -> None:
        self.bank.clear()

    def remove(self) -> None:
        for _, norm3 in self._wrapped:
            orig = getattr(norm3, "_global_unet_crossattn_writer_orig_forward", None)
            if orig is not None:
                norm3.forward = orig  # type: ignore[assignment]
                delattr(norm3, "_global_unet_crossattn_writer_orig_forward")
        self._wrapped.clear()
        self.clear()

    def _register(self) -> None:
        block_index = 0
        for module in self.unet.modules():
            blocks = getattr(module, "transformer_blocks", None)
            if blocks is None:
                continue
            for block in list(blocks):
                norm3 = getattr(block, "norm3", None)
                if norm3 is None or not hasattr(norm3, "forward"):
                    continue
                self._wrap_norm3(norm3, block_index)
                self._wrapped.append((block_index, norm3))
                block_index += 1

    def _wrap_norm3(self, norm3: torch.nn.Module, block_index: int) -> None:
        orig_forward = norm3.forward

        def wrapped_forward(hidden_states, *args, **kwargs):
            token = hidden_states.detach() if self.detach_global_tokens else hidden_states
            self.bank.setdefault(block_index, []).append(token)
            return orig_forward(hidden_states, *args, **kwargs)

        norm3._global_unet_crossattn_writer_orig_forward = orig_forward  # type: ignore[attr-defined]
        norm3.forward = wrapped_forward  # type: ignore[assignment]


class DiffusersUNetAttnReader:
    """
    Main UNet reader hook for diffusers-style UNets.

    Injection style: concat + self-attn token mixing.
    - Wraps every transformer block's `attn1.forward` (self-attn).
    - In read mode, concatenates `[current_tokens | ref_tokens]` for attention,
      then returns only the slice corresponding to `current_tokens`.
    """

    def __init__(
        self,
        unet: torch.nn.Module,
        *,
        global_token_scale: float = 1.0,
    ) -> None:
        self.unet = unet
        self.global_token_scale = float(global_token_scale)

        self.bank: TensorBank = {}
        # Optional per-sample mask that indicates whether global tokens should be injected.
        # Shape: [B] (bool). If None, injection applies to the full batch.
        self.inject_mask: Optional[torch.Tensor] = None
        self._wrapped: List[Tuple[int, torch.nn.Module]] = []

        self._register()

    @property
    def num_blocks(self) -> int:
        return len(self._wrapped)

    def update(self, bank: TensorBank) -> None:
        self.bank = {k: list(v) for k, v in bank.items()}

    def clear(self) -> None:
        self.bank.clear()
        self.inject_mask = None

    def set_inject_mask(self, inject_mask: Optional[torch.Tensor]) -> None:
        self.inject_mask = inject_mask

    def remove(self) -> None:
        for _, attn1 in self._wrapped:
            orig = getattr(attn1, "_global_unet_reader_orig_forward", None)
            if orig is not None:
                attn1.forward = orig  # type: ignore[assignment]
                delattr(attn1, "_global_unet_reader_orig_forward")
        self._wrapped.clear()
        self.clear()

    def _register(self) -> None:
        block_index = 0
        for module in self.unet.modules():
            blocks = getattr(module, "transformer_blocks", None)
            if blocks is None:
                continue
            for block in list(blocks):
                attn1 = getattr(block, "attn1", None)
                if attn1 is None or not hasattr(attn1, "forward"):
                    continue
                self._wrap_attn1(attn1, block_index)
                self._wrapped.append((block_index, attn1))
                block_index += 1

    def _wrap_attn1(self, attn1: torch.nn.Module, block_index: int) -> None:
        orig_forward = attn1.forward

        def wrapped_forward(
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **cross_attention_kwargs,
        ) -> torch.Tensor:
            # Only inject into self-attn. If encoder_hidden_states is set, this is cross-attn.
            if encoder_hidden_states is not None:
                return orig_forward(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )

            bank_list = self.bank.get(block_index, [])
            if not bank_list:
                return orig_forward(
                    hidden_states,
                    encoder_hidden_states=None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )

            ref_tokens = torch.cat(bank_list, dim=1) if len(bank_list) > 1 else bank_list[0]

            inject_mask = self.inject_mask
            if inject_mask is None:
                # CFG convenience: if main batch is doubled [UC|COND], duplicate ref tokens for both branches
                # (text-only CFG). For "ref-guidance" behavior (no ref in UC), pass an explicit inject_mask.
                if ref_tokens.shape[0] != hidden_states.shape[0]:
                    if hidden_states.shape[0] == 2 * ref_tokens.shape[0]:
                        ref_tokens = torch.cat([ref_tokens, ref_tokens], dim=0)
                    else:
                        raise RuntimeError(
                            "Global token batch mismatch for attn injection: "
                            f"hidden_states.B={hidden_states.shape[0]} vs ref_tokens.B={ref_tokens.shape[0]} "
                            f"(block_index={block_index})."
                        )

                ref_tokens = ref_tokens.to(device=hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
                if self.global_token_scale != 1.0:
                    ref_tokens = ref_tokens * self.global_token_scale

                x_cat = torch.cat([hidden_states, ref_tokens], dim=1)
                out_cat = orig_forward(
                    x_cat,
                    encoder_hidden_states=None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )

                # Keep only original token slice to match residual expectations upstream.
                return out_cat[:, : hidden_states.shape[1], :]

            # Per-sample masking: inject only for selected samples (e.g., CFG dropout during training).
            if inject_mask.ndim != 1 or inject_mask.shape[0] != hidden_states.shape[0]:
                raise ValueError(
                    "inject_mask must be 1D with shape [B]. "
                    f"got shape={tuple(inject_mask.shape)} expected B={hidden_states.shape[0]}."
                )
            inject_mask = inject_mask.to(device=hidden_states.device).to(dtype=torch.bool)

            if inject_mask.all():
                # Fast-path: inject for all samples
                if ref_tokens.shape[0] != hidden_states.shape[0]:
                    raise RuntimeError(
                        "inject_mask is all-True but ref token batch does not match. "
                        f"hidden_states.B={hidden_states.shape[0]} ref_tokens.B={ref_tokens.shape[0]} "
                        f"(block_index={block_index})."
                    )
                ref_tokens = ref_tokens.to(device=hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
                if self.global_token_scale != 1.0:
                    ref_tokens = ref_tokens * self.global_token_scale
                x_cat = torch.cat([hidden_states, ref_tokens], dim=1)
                out_cat = orig_forward(
                    x_cat,
                    encoder_hidden_states=None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
                return out_cat[:, : hidden_states.shape[1], :]

            if (~inject_mask).all():
                # Fast-path: inject for none
                return orig_forward(
                    hidden_states,
                    encoder_hidden_states=None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )

            if attention_mask is not None:
                # For our training usage this is always None. Extending attention masks to
                # account for concatenated global tokens is non-trivial, so fail loudly.
                raise NotImplementedError("inject_mask with non-None attention_mask is not supported.")

            idx_cond = torch.where(inject_mask)[0]
            idx_uncond = torch.where(~inject_mask)[0]

            out = torch.empty_like(hidden_states)

            # Unconditional samples: no ref tokens concatenated
            if idx_uncond.numel() > 0:
                out[idx_uncond] = orig_forward(
                    hidden_states[idx_uncond],
                    encoder_hidden_states=None,
                    attention_mask=None,
                    **cross_attention_kwargs,
                )

            # Conditional samples: inject
            if ref_tokens.shape[0] == hidden_states.shape[0]:
                ref_tokens_cond = ref_tokens[idx_cond]
            else:
                if ref_tokens.shape[0] != idx_cond.numel():
                    raise RuntimeError(
                        "Global token batch mismatch for masked injection: "
                        f"ref_tokens.B={ref_tokens.shape[0]} expected={idx_cond.numel()} "
                        f"(block_index={block_index})."
                    )
                ref_tokens_cond = ref_tokens

            ref_tokens_cond = ref_tokens_cond.to(
                device=hidden_states.device, dtype=hidden_states.dtype, non_blocking=True
            )
            if self.global_token_scale != 1.0:
                ref_tokens_cond = ref_tokens_cond * self.global_token_scale

            x_cat = torch.cat([hidden_states[idx_cond], ref_tokens_cond], dim=1)
            out_cat = orig_forward(
                x_cat,
                encoder_hidden_states=None,
                attention_mask=None,
                **cross_attention_kwargs,
            )
            out[idx_cond] = out_cat[:, : hidden_states.shape[1], :]
            return out

        attn1._global_unet_reader_orig_forward = orig_forward  # type: ignore[attr-defined]
        attn1.forward = wrapped_forward  # type: ignore[assignment]


class DiffusersUNetFuserBankInjector:
    """
    Main UNet fuser hook that concatenates GlobalUNet bank tokens.

    Injection style: concat inside the fuser's self-attn (post-attn1).
    - Wraps every transformer block's `fuser.forward`.
    - Calls the original fuser on `[hidden_states | ref_tokens]` (same channel dim),
      while passing `objs` (e.g., patch token) unchanged so the fuser's internal
      `linear(context_dim->query_dim)` handles it.
    - Returns only the slice corresponding to the original `hidden_states` length.
    - Optional mode: build attention queries from the main `hidden_states` only,
      while keys/values still see `[hidden_states | ref_tokens | objs]`.
    """

    def __init__(
        self,
        unet: torch.nn.Module,
        *,
        global_token_scale: float = 1.0,
        main_query_only: bool = False,
    ) -> None:
        self.unet = unet
        self.global_token_scale = float(global_token_scale)
        self.main_query_only = bool(main_query_only)

        # Per-block global bank tokens: idx -> [B, N_bank, C_block]
        self.bank: Dict[int, torch.Tensor] = {}
        # Optional per-sample mask to enable/disable fuser injection.
        # Shape: [B] (bool). If None, injection applies to the full batch.
        self.inject_mask: Optional[torch.Tensor] = None
        self._wrapped: List[Tuple[int, torch.nn.Module]] = []

        self._register()

    @property
    def num_blocks(self) -> int:
        return len(self._wrapped)

    def update(self, bank: Dict[int, torch.Tensor]) -> None:
        self.bank = {int(k): v for k, v in bank.items()}

    def clear(self) -> None:
        self.bank.clear()
        self.inject_mask = None

    def set_inject_mask(self, inject_mask: Optional[torch.Tensor]) -> None:
        self.inject_mask = inject_mask

    def remove(self) -> None:
        for _, fuser in self._wrapped:
            orig = getattr(fuser, "_global_unet_fuser_injector_orig_forward", None)
            if orig is not None:
                fuser.forward = orig  # type: ignore[assignment]
                delattr(fuser, "_global_unet_fuser_injector_orig_forward")
        self._wrapped.clear()
        self.clear()

    def _register(self) -> None:
        block_index = 0
        for module in self.unet.modules():
            blocks = getattr(module, "transformer_blocks", None)
            if blocks is None:
                continue
            for block in list(blocks):
                fuser = getattr(block, "fuser", None)
                if fuser is None or not hasattr(fuser, "forward"):
                    continue
                self._wrap_fuser(fuser, block_index)
                self._wrapped.append((block_index, fuser))
                block_index += 1

    def _wrap_fuser(self, fuser: torch.nn.Module, block_index: int) -> None:
        orig_forward = fuser.forward

        def _run_fuser_with_bank(
            hidden_states: torch.Tensor,
            objs: torch.Tensor,
            *,
            bank_tokens: Optional[torch.Tensor],
            **kwargs,
        ) -> torch.Tensor:
            if bank_tokens is None:
                return orig_forward(hidden_states, objs, **kwargs)

            if bank_tokens.shape[-1] != hidden_states.shape[-1]:
                raise RuntimeError(
                    "Global bank channel mismatch for fuser injection: "
                    f"hidden.C={hidden_states.shape[-1]} bank.C={bank_tokens.shape[-1]} "
                    f"(block_index={block_index})."
                )

            ref_tokens = bank_tokens.to(
                device=hidden_states.device, dtype=hidden_states.dtype, non_blocking=True
            )
            if self.global_token_scale != 1.0:
                ref_tokens = ref_tokens * self.global_token_scale

            x_cat = torch.cat([hidden_states, ref_tokens], dim=1)
            if not self.main_query_only:
                out_cat = orig_forward(x_cat, objs, **kwargs)
                return out_cat[:, : hidden_states.shape[1], :]

            return _run_fuser_main_query_only(
                fuser=fuser,
                hidden_states=hidden_states,
                kv_visual_tokens=x_cat,
                objs=objs,
                block_index=block_index,
                kwargs=kwargs,
            )

        def wrapped_forward(hidden_states: torch.Tensor, objs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            # Keep kwargs so we don't break custom fuser variants.
            if args:
                raise TypeError("Unexpected positional args to fuser.forward; update injector wrapper if needed.")

            bank_tokens = self.bank.get(block_index, None)
            inject_mask = self.inject_mask

            if inject_mask is None:
                # CFG convenience: if main batch is doubled [UC|COND], duplicate global bank tokens for both branches.
                if bank_tokens is not None and bank_tokens.shape[0] != hidden_states.shape[0]:
                    if hidden_states.shape[0] == 2 * bank_tokens.shape[0]:
                        bank_tokens = torch.cat([bank_tokens, bank_tokens], dim=0)
                    else:
                        raise RuntimeError(
                            "Global bank batch mismatch for fuser injection: "
                            f"hidden.B={hidden_states.shape[0]} bank.B={bank_tokens.shape[0]} "
                            f"(block_index={block_index})."
                        )
                return _run_fuser_with_bank(hidden_states, objs, bank_tokens=bank_tokens, **kwargs)

            if inject_mask.ndim != 1 or inject_mask.shape[0] != hidden_states.shape[0]:
                raise ValueError(
                    "inject_mask must be 1D with shape [B]. "
                    f"got shape={tuple(inject_mask.shape)} expected B={hidden_states.shape[0]}."
                )
            inject_mask = inject_mask.to(device=hidden_states.device).to(dtype=torch.bool)

            if inject_mask.all():
                if bank_tokens is not None and bank_tokens.shape[0] != hidden_states.shape[0]:
                    raise RuntimeError(
                        "inject_mask is all-True but global bank batch does not match. "
                        f"hidden.B={hidden_states.shape[0]} bank.B={bank_tokens.shape[0]} "
                        f"(block_index={block_index})."
                    )
                return _run_fuser_with_bank(hidden_states, objs, bank_tokens=bank_tokens, **kwargs)

            if (~inject_mask).all():
                # Skip fuser entirely
                return hidden_states

            idx_cond = torch.where(inject_mask)[0]
            idx_uncond = torch.where(~inject_mask)[0]

            out = torch.empty_like(hidden_states)
            out[idx_uncond] = hidden_states[idx_uncond]

            bank_tokens_cond: Optional[torch.Tensor] = None
            if bank_tokens is not None:
                if bank_tokens.shape[0] == hidden_states.shape[0]:
                    bank_tokens_cond = bank_tokens[idx_cond]
                else:
                    if bank_tokens.shape[0] != idx_cond.numel():
                        raise RuntimeError(
                            "Global bank batch mismatch for masked fuser injection: "
                            f"bank.B={bank_tokens.shape[0]} expected={idx_cond.numel()} "
                            f"(block_index={block_index})."
                        )
                    bank_tokens_cond = bank_tokens

            out[idx_cond] = _run_fuser_with_bank(
                hidden_states[idx_cond],
                objs[idx_cond],
                bank_tokens=bank_tokens_cond,
                **kwargs,
            )
            return out

        fuser._global_unet_fuser_injector_orig_forward = orig_forward  # type: ignore[attr-defined]
        fuser.forward = wrapped_forward  # type: ignore[assignment]


class ZeroInitBankFuserWrapper(torch.nn.Module):
    """
    Wrapper around a per-block fuser module that starts from "no effect".

    Implementation matches the LDM Stage 1 global U-Net behavior:
    - inner fuser performs gated fusion
    - output is passed through a zero-initialized projection
    """

    def __init__(self, inner_fuser: torch.nn.Module, dim: int) -> None:
        super().__init__()
        self.inner_fuser = inner_fuser
        self.proj_out = torch.nn.Linear(int(dim), int(dim))
        with torch.no_grad():
            self.proj_out.weight.zero_()
            if self.proj_out.bias is not None:
                self.proj_out.bias.zero_()

    def forward(self, x: torch.Tensor, objs: torch.Tensor) -> torch.Tensor:
        x_fused = self.inner_fuser(x, objs)
        return self.proj_out(x_fused)


def make_patch_token_(patch_unifusion: torch.nn.Module, bbox_b14: torch.Tensor) -> torch.Tensor:
    """
    Build [B,1,768] patch token from bbox [B,1,4] using a UniFusion-style position net.

    Uses `null_positive_feature` as the positive embedding and mask=1 so that the
    bbox position is preserved.
    """
    if not hasattr(patch_unifusion, "null_positive_feature"):
        raise AttributeError("patch_unifusion must define 'null_positive_feature' for patch tokenization.")

    B = int(bbox_b14.shape[0])
    device = next(patch_unifusion.parameters()).device
    dtype = next(patch_unifusion.parameters()).dtype

    boxes = bbox_b14.to(device=device, dtype=dtype)
    masks = torch.ones(B, 1, device=device, dtype=dtype)
    pos_null = patch_unifusion.null_positive_feature.view(1, 1, -1).expand(B, 1, -1).to(device=device, dtype=dtype)

    prev_training = patch_unifusion.training
    try:
        # Disable any internal random drop while preserving gradients.
        patch_unifusion.train(False)
        out = patch_unifusion(boxes, masks, pos_null)
    finally:
        patch_unifusion.train(prev_training)

    # UniFusion-like modules may return (objs, drop_mask)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def apply_bank_gating_(
    bank: TensorBank,
    bank_gating_fusers: torch.nn.ModuleList,
    patch_token_768: torch.Tensor,
) -> None:
    """
    In-place gating of bank tokens per block.

    bank[idx] is a list of tensors [B, N, C] captured at that block.
    """
    for idx, bank_list in list(bank.items()):
        if idx < 0 or idx >= len(bank_gating_fusers):
            raise RuntimeError(f"bank_gating_fusers index out of range: idx={idx} len={len(bank_gating_fusers)}")
        fuser = bank_gating_fusers[idx]
        new_list: List[torch.Tensor] = []
        for tokens in bank_list:
            if tokens.shape[0] != patch_token_768.shape[0]:
                raise RuntimeError(
                    f"Gating batch mismatch at block {idx}: tokens.B={tokens.shape[0]} vs patch.B={patch_token_768.shape[0]}"
                )
            objs = patch_token_768.to(device=tokens.device, dtype=tokens.dtype, non_blocking=True)
            new_list.append(fuser(tokens, objs))
        bank[idx] = new_list


class MainOnlyGatingModules(torch.nn.Module):
    """
    Trainable modules for stage-2 main-only training.

    - `patch_unifusion`: bbox -> patch token
    - `bank_gating_fusers`: per-block fusers to gate GlobalUNet banks using patch token
    """

    def __init__(self, patch_unifusion: torch.nn.Module, bank_gating_fusers: torch.nn.ModuleList) -> None:
        super().__init__()
        self.patch_unifusion = patch_unifusion
        self.bank_gating_fusers = bank_gating_fusers

    def forward(self, bank_tokens_by_block: Tuple[torch.Tensor, ...], bbox_b14: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        patch_token = make_patch_token_(self.patch_unifusion, bbox_b14)
        out: List[torch.Tensor] = []
        for idx, tokens in enumerate(bank_tokens_by_block):
            if idx >= len(self.bank_gating_fusers):
                raise RuntimeError(
                    f"bank_gating_fusers shorter than bank tokens: idx={idx} len={len(self.bank_gating_fusers)}"
                )
            out.append(self.bank_gating_fusers[idx](tokens, patch_token))
        return tuple(out)
