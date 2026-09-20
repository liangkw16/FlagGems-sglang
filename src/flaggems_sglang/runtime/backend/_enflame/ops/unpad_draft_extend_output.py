# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# s0r carrier (2026-09-21 07:20): identical bytes to the e13/e14
# enflame path; the 2.3 read on submission e14 vs 56.5 on identical bytes
# is the documented enflame slow window - one comment-carrier re-roll.
# Enflame vendor for unpad_draft_extend_output: batch-segment copy at
# BLOCK 16384 with the program count held near the 24-SIP width
# (tiles = max(1, 24 // bs); the relu2-GCU form). Width ladder evidence:
# 18.7 @4096 -> 33.6 @8192 -> 55.1 @16384 -> 51.9 @32768 (peak
# 16384, stages 4 probes deeper pipelining); row/flat-many-program forms read 0.5-0.6.

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
    span = heads * dim
    if bs and token_per_batch and out.numel():
        spanchunks = max(1, (token_per_batch * span + 16383) // 16384)
        tiles = min(spanchunks, max(1, 24 // bs))
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
            num_stages=4,
        )
    return out


__all__ = ["unpad_draft_extend_output"]
