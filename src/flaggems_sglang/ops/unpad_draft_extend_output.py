# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Inverse of pad_draft_extend_query: gather accepted rows into a ragged
# [total, H, D] tensor. E6 restructures to per-batch contiguous segment
# copies: batch b contributes seq_lens_q[b] * H * D contiguous elements
# from raw_out row b (the padded layout keeps each batch's accepted prefix
# contiguous), which removes the whole searchsorted row-mapping staging
# and the per-row gather; the kernel uses only scalar segment starts and
# contiguous vector offsets (no tensor-index gather anywhere).

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

    def tiles_for(units, block):
        return min(max(1, (token_per_batch * units + block - 1) // block), 255)


    span = heads * dim
    if bs and token_per_batch and out.numel():
        if span % 2 == 0:
            # uint32 datapath: same bytes, half the lanes per tile, so
            # the compiler issues 32-bit accesses instead of 16-bit
            # (codex-ask axis; element fallback keeps odd spans exact).
            _unpad[(bs, tiles_for(span // 2, 2048))](
                raw_out.view(torch.uint32),
                seq_lens_q,
                cu_seqlens_q,
                out.view(torch.uint32),
                span // 2,
                token_per_batch,
                seq_lens_q.stride(0),
                cu_seqlens_q.stride(0),
                BLOCK=2048,
            )
        else:
            _unpad[(bs, tiles_for(span, 4096))](
                raw_out,
                seq_lens_q,
                cu_seqlens_q,
                out,
                span,
                token_per_batch,
                seq_lens_q.stride(0),
                cu_seqlens_q.stride(0),
                BLOCK=4096,
            )
    return out


__all__ = ["unpad_draft_extend_output"]
