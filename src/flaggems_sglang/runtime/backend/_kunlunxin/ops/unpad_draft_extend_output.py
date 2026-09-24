# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Kunlunxin vendor for unpad_draft_extend_output: batch-segment copy at
# BLOCK 16384 (wide direction - 1024 crashed to 6.1 while the 4096 generic
# reads 16.6-17.8 vs the 26-39 field band; 16384 read 34.6, 32768 fell
# to 27.3 - peak at 16384).

import torch
import triton
import triton.language as tl


@triton.jit
def _unpad(
    raw_out, lens, cum, out, span, tpb, lstride, cstride,
    BLOCK: tl.constexpr,
):
    seg = tl.program_id(0)
    tile = tl.program_id(1)
    n = tl.load(lens + seg.to(tl.int64) * lstride)
    beg = tl.load(cum + seg.to(tl.int64) * cstride)
    # int64 only for the scalar segment bases; the lane pipeline is
    # int32 (in-segment offsets are bounded by token_per_batch*span,
    # guarded by the wrapper) - the i64 lane vector is slow ALU and, on
    # XPU at extreme widths, numerically unsafe
    src = seg.to(tl.int64) * tpb * span
    dst = (beg.to(tl.int64) * span)
    elems32 = n * span
    for base in range(tile * BLOCK, elems32, tl.num_programs(1) * BLOCK):
        offs = base + tl.arange(0, BLOCK)
        m = offs < elems32
        v = tl.load(raw_out + src + offs, m, other=0)
        tl.store(out + dst + offs, v, m)


@triton.jit
def _unpad_wide(
    raw_out, lens, cum, out, span, tpb, lstride, cstride,
    BLOCK: tl.constexpr,
):
    # i64 pipeline for segments reaching the int32 lane domain; the
    # wrapper dispatches on token_per_batch*span
    seg = tl.program_id(0)
    tile = tl.program_id(1)
    n = tl.load(lens + seg.to(tl.int64) * lstride)
    beg = tl.load(cum + seg.to(tl.int64) * cstride)
    src = seg.to(tl.int64) * tpb * span
    dst = (beg.to(tl.int64) * span)
    elems = n.to(tl.int64) * span
    for base in range(
        tile.to(tl.int64) * BLOCK, elems, tl.num_programs(1).to(tl.int64) * BLOCK
    ):
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
        tiles = min(max(1, (token_per_batch * span + 16383) // 16384), 255)
        # headroom: the scf.for induction variable can step past elems
        # by up to nprog*BLOCK (<= 255*BLOCK) before exiting, so the
        # int32 pipeline needs that margin below 2^31
        headroom = 2**31 - 256 * 16384
        kernel = (
            _unpad if token_per_batch * span < headroom else _unpad_wide
        )
        kernel[(bs, tiles)](
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
