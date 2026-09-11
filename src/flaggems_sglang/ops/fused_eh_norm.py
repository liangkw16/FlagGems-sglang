# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Triton port of SGLang 8014d9d fused_eh_norm (EAGLE eh-norm + concat).

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_eh_norm(
    embeds_ptr,
    prev_ptr,
    enorm_ptr,
    hnorm_ptr,
    out_ptr,
    es0,
    es1,
    ps0,
    ps1,
    ws0,
    hs0,
    os0,
    hidden,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < hidden

    e = tl.load(embeds_ptr + row * es0 + cols * es1, mask=mask, other=0.0).to(
        tl.float32
    )
    ew = tl.load(enorm_ptr + cols * ws0, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(e * e, axis=0) / hidden
    e_out = e * tl.rsqrt(ms + eps) * ew
    tl.store(
        out_ptr + row * os0 + cols,
        e_out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )

    h = tl.load(prev_ptr + row * ps0 + cols * ps1, mask=mask, other=0.0).to(
        tl.float32
    )
    hw = tl.load(hnorm_ptr + cols * hs0, mask=mask, other=0.0).to(tl.float32)
    ms_h = tl.sum(h * h, axis=0) / hidden
    h_out = h * tl.rsqrt(ms_h + eps) * hw
    tl.store(
        out_ptr + row * os0 + hidden + cols,
        h_out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def fused_eh_norm(
    inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, eps
):
    assert inputs_embeds.ndim == 2
    tokens, hidden = inputs_embeds.shape
    assert previous_hidden.shape == (tokens, hidden)
    assert enorm_weight.numel() == hidden
    assert hnorm_weight.numel() == hidden
    out = torch.empty(
        (tokens, hidden * 2),
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
    )
    if tokens and hidden:
        _fused_eh_norm[(tokens,)](
            inputs_embeds,
            previous_hidden,
            enorm_weight,
            hnorm_weight,
            out,
            inputs_embeds.stride(0),
            inputs_embeds.stride(1),
            previous_hidden.stride(0),
            previous_hidden.stride(1),
            enorm_weight.stride(0),
            hnorm_weight.stride(0),
            out.stride(0),
            hidden,
            eps,
            BLOCK=triton.next_power_of_2(hidden),
            num_warps=8,
            num_stages=1,
        )
    return out


__all__ = ["fused_eh_norm"]
