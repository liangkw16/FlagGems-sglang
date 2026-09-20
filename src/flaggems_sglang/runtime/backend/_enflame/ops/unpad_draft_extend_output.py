# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Enflame vendor for unpad_draft_extend_output: fixed-P output partition
# (codex-ask structural axis) - each of the <=24 programs owns a BLOCK-
# aligned contiguous slab of the OUTPUT and walks segments with scalar
# upper-bound steps, so per-program work no longer depends on which
# segment the tiles land in. BLOCK 16384 stays at the measured peak.

import torch
import triton
import triton.language as tl


@triton.jit
def _unpad_part(
    raw_out, cum, out, span, tpb, total, bs, cstride,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    nprog = tl.num_programs(0).to(tl.int64)
    C = ((total + nprog * BLOCK - 1) // (nprog * BLOCK)) * BLOCK
    lo = pid * C
    hi = tl.minimum(lo + C, total)
    if lo < hi:
        seg = tl.zeros((), tl.int64)
        pos = lo
        while pos < hi:
            # scalar upper bound: skip (possibly empty) segments ending
            # at or before pos; pos < total keeps seg+1 within cum
            while tl.load(cum + (seg + 1) * cstride).to(tl.int64) * span <= pos:
                seg += 1
            seg_start = tl.load(cum + seg * cstride).to(tl.int64) * span
            seg_end = tl.load(cum + (seg + 1) * cstride).to(tl.int64) * span
            end = tl.minimum(seg_end, hi)
            src = seg * tpb * span + (pos - seg_start)
            while pos < end:
                offs = pos + tl.arange(0, BLOCK)
                m = offs < end
                v = tl.load(raw_out + src + (offs - pos), m, other=0)
                tl.store(out + offs, v, m)
                pos += BLOCK


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
    if bs and token_per_batch and out.numel():
        _unpad_part[(min(24, triton.cdiv(out.numel(), 16384)),)](
            raw_out,
            cu_seqlens_q,
            out,
            heads * dim,
            token_per_batch,
            out.numel(),
            bs,
            cu_seqlens_q.stride(0),
            BLOCK=16384,
        )
    return out


__all__ = ["unpad_draft_extend_output"]
