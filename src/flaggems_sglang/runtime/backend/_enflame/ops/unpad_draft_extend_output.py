# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Enflame vendor for unpad_draft_extend_output, e19 ruleset round.
# The e13-era form ran a (bs, tiles) grid at the empirical 24-SIP
# width with runtime strides and full int64 addressing; per the
# official gcu300 codegen rules (max_grid_size=(12,1,1),
# num_warps=2, compile-time stride divisibility for the DMA path,
# enable_i64=False) this round: one flat 1D work-item axis over
# (segment, tile) pairs grid-strided across at most 12 programs,
# every stride (including the strided len/cum tensors the harness
# exercises) baked as constexpr, int32
# offsets throughout, BLOCK 16384 kept from the width ladder peak
# (18.7 @4096 -> 33.6 @8192 -> 55.1 @16384).

import torch
import triton
import triton.language as tl

_MAX_CTAS = 12
_NUM_WARPS = 2
_BLOCK = 16384


@triton.jit
def _unpad(
    raw_out,
    lens,
    cum,
    out,
    items,
    tiles,
    SPAN: tl.constexpr,
    TPB: tl.constexpr,
    LS: tl.constexpr,
    CS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for item in range(tl.program_id(0), items, tl.num_programs(0)):
        seg = item // tiles
        tile = item % tiles
        n = tl.load(lens + seg * LS)
        beg = tl.load(cum + seg * CS)
        src = seg * TPB * SPAN
        dst = beg * SPAN
        elems = n * SPAN
        for base in range(tile * BLOCK, elems, tiles * BLOCK):
            offs = base + tl.arange(0, BLOCK)
            m = offs < elems
            v = tl.load(raw_out + src + offs, m, other=0)
            tl.store(out + dst + offs, v, m)


def unpad_draft_extend_output(raw_out, cu_seqlens_q, seq_lens_q, sum_seq_lens_q):
    assert raw_out.ndim == 4
    bs, token_per_batch, heads, dim = raw_out.shape
    assert seq_lens_q.shape == (bs,) and cu_seqlens_q.shape == (bs + 1,)
    assert seq_lens_q.dtype == cu_seqlens_q.dtype == torch.int32
    assert raw_out.dtype in (torch.float16, torch.bfloat16)
    assert raw_out.is_contiguous()
    out = torch.empty(
        (sum_seq_lens_q, heads, dim),
        dtype=raw_out.dtype,
        device=raw_out.device,
    )
    span = heads * dim
    if bs and token_per_batch and out.numel():
        assert raw_out.numel() < 2**31  # int32 element offsets
        spanchunks = max(1, (token_per_batch * span + _BLOCK - 1) // _BLOCK)
        items = bs * spanchunks
        _unpad[(min(items, _MAX_CTAS),)](
            raw_out,
            seq_lens_q,
            cu_seqlens_q,
            out,
            items,
            spanchunks,
            SPAN=span,
            TPB=token_per_batch,
            LS=seq_lens_q.stride(0),
            CS=cu_seqlens_q.stride(0),
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
            num_stages=4,
        )
    return out


__all__ = ["unpad_draft_extend_output"]
