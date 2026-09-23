# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Ascend e23: keep e22's proven (bs, tiles) grid and int64 offsets.
# Full tiles copy without a mask; only the final partial tile uses the
# e22 masked path. e21r combined this idea with persistent rotation,
# fp32 tail arithmetic and warps16, then failed correctness on Ascend.
# This isolates the full-tile path without those changes.

import torch
import triton
import triton.language as tl


@triton.jit
def _unpad(
    raw_out,
    lens,
    cum,
    out,
    span,
    tpb,
    lstride,
    cstride,
    BLOCK: tl.constexpr,
):
    seg = tl.program_id(0)
    tile = tl.program_id(1)
    n = tl.load(lens + seg.to(tl.int64) * lstride)
    beg = tl.load(cum + seg.to(tl.int64) * cstride)
    src = seg.to(tl.int64) * tpb * span
    dst = beg.to(tl.int64) * span
    elems = n.to(tl.int64) * span
    for base in range(
        tile.to(tl.int64) * BLOCK,
        elems,
        tl.num_programs(1).to(tl.int64) * BLOCK,
    ):
        offs = base + tl.arange(0, BLOCK)
        if base + BLOCK <= elems:
            v = tl.load(raw_out + src + offs)
            tl.store(out + dst + offs, v)
        else:
            m = offs < elems
            v = tl.load(raw_out + src + offs, m)
            tl.store(out + dst + offs, v, m)


def unpad_draft_extend_output(
    raw_out, cu_seqlens_q, seq_lens_q, sum_seq_lens_q
):
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
        tiles = min(max(1, (token_per_batch * span + 16383) // 16384), 255)
        _unpad[(bs, tiles)](
            raw_out,
            seq_lens_q,
            cu_seqlens_q,
            out,
            span,
            token_per_batch,
            seq_lens_q.stride(0),
            cu_seqlens_q.stride(0),
            BLOCK=16384,
        )
    return out


__all__ = ["unpad_draft_extend_output"]
