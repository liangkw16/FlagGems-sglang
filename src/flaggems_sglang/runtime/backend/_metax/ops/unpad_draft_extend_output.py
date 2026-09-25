# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Metax vendor for unpad_draft_extend_output: batch-segment copy at
# BLOCK 8192 with default warps (warps8 read 213.3 vs 242.3 default;
# width saturated at 249.6; stages 4 probes deeper pipelining on muxi).
#
# e27 load-unmask (store-only mask): the muxi width/parameter axes are
# exhausted (BLOCK 8192 saturated, warps8 negative, stages4 and u32
# no-ops), so the residual gap to the 391-392 leader band sits in the
# masked-access form. Full tiles (base+BLOCK<=elems) load WITHOUT a
# mask - the mask was all-true there, and every lane stays inside the
# segment's own slot - removing the per-lane predication and the
# other=0 prefill that dominate small copies (platform case 3 is only
# 164352 elements). The store ALWAYS keeps the mask m: only the load
# side is de-masked (e23 dropped both sides on _ascend and lost -28.8%
# on huawei; this form is confined to _amd/_metax). The partial tail
# tile keeps a masked load - an unguarded tail read could cross the
# raw_out allocation end by up to BLOCK-1 elements on the last
# segment - without the other= prefill: masked-out lanes hold undef
# values that the store mask keeps out of memory (e22 semantics,
# byte-identical output).

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
        if base + BLOCK <= elems:
            # e27 full tile: mask was all-true; drop the predication
            v = tl.load(raw_out + src + offs)
        else:
            # partial tail: masked read (an unguarded one could cross
            # the raw_out end by BLOCK-1 on the last segment); no
            # other= prefill - the store mask keeps undef lanes out
            v = tl.load(raw_out + src + offs, m)
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
        tiles = min(max(1, (token_per_batch * span + 8191) // 8192), 255)
        _unpad[(bs, tiles)](
            raw_out,
            seq_lens_q,
            cu_seqlens_q,
            out,
            span,
            token_per_batch,
            seq_lens_q.stride(0),
            cu_seqlens_q.stride(0),
            BLOCK=8192,
        )
    return out


__all__ = ["unpad_draft_extend_output"]
