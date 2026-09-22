# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Ascend vendor for unpad_draft_extend_output, e21 candidate: the e20
# ruleset bytes (int32 flat-span addressing since the vector ADD unit
# has no int64 path, fp32 tail mask since the vector compare unit has
# no int path - gated to the sub-2^24 domain by a scalar branch, no
# masked-load `other` pre-fill, num_warps=16 at the >=4096 tile rung)
# re-launched as the official capped grid-stride persistent kernel:
# grid=(min(bs, CAP),) with CAP=64 pre-registered (the T77-e5 verified
# shape; a 32/40/48 sweep is the only follow-up if this lands positive
# but under the gate). Each program rotates over segments
# p, p+num_programs, ... (vector_operator.md: "keep the launch close
# to the number of physical Vector Cores and let each program process
# multiple tiles in an inner loop"; "GPU-style small tiles with very
# large grids often cause repeated dispatch overhead on NPUs"), reads
# the segment's lens/cum pair exactly once per visit, and the inner
# block loop walks only the real accepted length elems - the padded
# tail beyond n tokens spawns zero idle tiles, unlike the (bs, tiles)
# grid where tiles is derived from the padded tpb*span.

import torch
import triton
import triton.language as tl

_BLOCK = 16384
_NUM_WARPS = 16
_MAX_PROGRAMS = 64  # pre-registered CAP; T77-e5 capped-request form


@triton.jit
def _unpad(
    raw_out, lens, cum, out, span, tpb, lstride, cstride, bs,
    BLOCK: tl.constexpr,
):
    for seg in range(tl.program_id(0), bs, tl.num_programs(0)):
        n = tl.load(lens + seg * lstride)
        beg = tl.load(cum + seg * cstride)
        src = seg * tpb * span
        dst = beg * span
        elems = n * span
        # fp32 compares lose precision past 2**24 elements (the boundary
        # offset rounds up to elems and the last element goes unwritten -
        # a codex-review find), so the vector-mask fast path is gated to
        # the sub-2^24 domain by a scalar branch; larger inputs keep the
        # exact integer mask
        if elems < 16777216:
            for base in range(0, elems, BLOCK):
                offs = base + tl.arange(0, BLOCK)
                m = offs.to(tl.float32) < elems.to(tl.float32)
                v = tl.load(raw_out + src + offs, m)
                tl.store(out + dst + offs, v, m)
        else:
            for base in range(0, elems, BLOCK):
                offs = base + tl.arange(0, BLOCK)
                m = offs < elems
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
        _unpad[(min(bs, _MAX_PROGRAMS),)](
            raw_out,
            seq_lens_q,
            cu_seqlens_q,
            out,
            span,
            token_per_batch,
            seq_lens_q.stride(0),
            cu_seqlens_q.stride(0),
            bs,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
    return out


__all__ = ["unpad_draft_extend_output"]
