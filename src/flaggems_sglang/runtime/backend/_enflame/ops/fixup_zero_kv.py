# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Enflame vendor for fixup_zero_kv: the e8 in-place core (zero-KV rows
# only, healthy segments exit immediately) on the official gcu300
# geometry - the e9 port kept the generic's batch*ot grid and only
# pinned num_warps, which is not the 12-CTA clamp the GCU codegen
# documents (max_grid_size=(12,1,1)). This round flattens the
# (segment, tile) work items and grid-strides them across at most 12
# programs with num_warps=2 and wide flat stores; e15 doubles the
# store width to BLOCK_V=2048 (the gcu300 tile guidance scales with
# the element width and the e12 read 102.6 with 1024-wide stores -
# the width ladder is live: 102.6 @1024 -> 113.9 @2048, e16 takes
# the 4096 rung toward the 193 field band).

import torch
import triton
import triton.language as tl

_MAX_CTAS = 12
_NUM_WARPS = 2


@triton.jit(do_not_specialize=["batch", "os0", "ls0"])
def _fixup_zero_kv(
    out,
    lse,
    lens,
    cum,
    batch,
    os0,
    ls0,
    HV: tl.constexpr,
    NH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # e19 segment-band scheduling (the codex-ask round's main
    # candidate): each program owns every num_programs-th segment,
    # reads that segment's metadata once, and sweeps all of its tiles
    # - no per-item lens loads and no empty advisory tiles, at the
    # cost of serializing a lone long zero segment into one program
    for seg in range(tl.program_id(0), batch, tl.num_programs(0)):
        if tl.load(lens + seg) == 0:
            beg = tl.load(cum + seg).to(tl.int64)
            end = tl.load(cum + seg + 1).to(tl.int64)
            v = tl.arange(0, BLOCK_V).to(tl.int64)
            h = tl.arange(0, BLOCK_H).to(tl.int64)
            hm = h < NH
            zeros = tl.zeros((BLOCK_T, BLOCK_V), dtype=out.dtype.element_ty)
            ninf = tl.full(
                (BLOCK_T, BLOCK_H), float("-inf"), dtype=tl.float32
            )
            for tile in range(0, tl.cdiv(end - beg, BLOCK_T)):
                t = (
                    beg
                    + tile * BLOCK_T
                    + tl.arange(0, BLOCK_T).to(tl.int64)
                )
                tm = t < end
                for v0 in tl.static_range(0, HV, BLOCK_V):
                    vv = v0 + v[None, :]
                    m = tm[:, None] & (vv < HV)
                    tl.store(out + t[:, None] * os0 + vv, zeros, m)
                tl.store(
                    lse + t[:, None] * ls0 + h[None, :],
                    ninf,
                    tm[:, None] & hm[None, :],
                )


def fixup_zero_kv(out, lse, kv_lens, cum_seq_lens, max_seq_len):
    assert out.ndim == 3 and lse.ndim == 2
    total_tokens, num_heads, v_head_dim = out.shape
    assert lse.shape == (total_tokens, num_heads)
    assert out.dtype in (torch.float16, torch.bfloat16)
    assert lse.dtype == torch.float32
    assert out.stride(2) == 1 and out.stride(1) == v_head_dim
    assert lse.stride(1) == 1
    batch = kv_lens.numel()
    assert cum_seq_lens.numel() == batch + 1
    assert kv_lens.dtype == cum_seq_lens.dtype == torch.int32
    if batch and total_tokens:
        hv, nh = num_heads * v_head_dim, num_heads
        block_t = 8
        _fixup_zero_kv[(min(batch, _MAX_CTAS),)](
            out,
            lse,
            kv_lens,
            cum_seq_lens,
            batch,
            out.stride(0),
            lse.stride(0),
            HV=hv,
            NH=nh,
            num_warps=_NUM_WARPS,
            BLOCK_T=block_t,
            BLOCK_V=4096,
            BLOCK_H=triton.next_power_of_2(max(1, nh)),
        )
    return out, lse


__all__ = ["fixup_zero_kv"]
