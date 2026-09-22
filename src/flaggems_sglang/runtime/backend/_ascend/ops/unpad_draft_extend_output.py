# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Ascend vendor for unpad_draft_extend_output, e20 ruleset round
# (width ladder banked 442 @16384 at e17). The triton-ascend
# performance guidelines: the vector compare unit has no int path
# (int32/int64 compares degrade to scalar) and vector ADD has no
# int64 - so every offset here is int32, the tail mask runs in fp32,
# and the masked load drops its `other` fill (the undefined lanes are
# never stored - the pre-fill otherwise serializes the MTE2 move).
# num_warps=16 per the official >=4096-element tile rung; 2D grid
# (bs, tiles) with the in-kernel tile stride retained so an
# understated tile count still covers every element.

import torch
import triton
import triton.language as tl

_BLOCK = 16384
_NUM_WARPS = 16


@triton.jit
def _unpad(
    raw_out, lens, cum, out, span, tpb, lstride, cstride,
    BLOCK: tl.constexpr,
):
    seg = tl.program_id(0)
    tile = tl.program_id(1)
    n = tl.load(lens + seg * lstride)
    beg = tl.load(cum + seg * cstride)
    src = seg * tpb * span
    dst = beg * span
    elems = n * span
    for base in range(tile * BLOCK, elems, tl.num_programs(1) * BLOCK):
        offs = base + tl.arange(0, BLOCK)
        m = offs.to(tl.float32) < elems.to(tl.float32)
        v = tl.load(raw_out + src + offs, m)
        tl.store(out + dst + offs, v, m)


def unpad_draft_extend_output(raw_out, cu_seqlens_q, seq_lens_q, sum_seq_lens_q):
    assert raw_out.ndim == 4
    bs, token_per_batch, heads, dim = raw_out.shape
    assert seq_lens_q.shape == (bs,) and cu_seqlens_q.shape == (bs + 1,)
    assert seq_lens_q.dtype == cu_seqlens_q.dtype == torch.int32
    assert raw_out.dtype in (torch.float16, torch.bfloat16)
    assert raw_out.is_contiguous()
    assert raw_out.numel() < 2**31  # int32 element offsets
    out = torch.empty(
        (sum_seq_lens_q, heads, dim),
        dtype=raw_out.dtype,
        device=raw_out.device,
    )
    span = heads * dim
    if bs and token_per_batch and out.numel():
        tiles = min(max(1, (token_per_batch * span + _BLOCK - 1) // _BLOCK), 255)
        _unpad[(bs, tiles)](
            raw_out,
            seq_lens_q,
            cu_seqlens_q,
            out,
            span,
            token_per_batch,
            seq_lens_q.stride(0),
            cu_seqlens_q.stride(0),
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
    return out


__all__ = ["unpad_draft_extend_output"]
