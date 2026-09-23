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


@triton.jit
def _fixup_zero_kv(
    out,
    lse,
    lens,
    cum,
    batch,
    ot,
    items,
    os0,
    ls0,
    HV: tl.constexpr,
    NH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    INDEX_T: tl.constexpr,
):
    for item in range(tl.program_id(0), items, tl.num_programs(0)):
        seg = item // ot
        zero = tl.load(lens + seg) == 0
        if zero:
            beg = tl.load(cum + seg).to(INDEX_T)
            end = tl.load(cum + seg + 1).to(INDEX_T)
            v = tl.arange(0, BLOCK_V).to(INDEX_T)
            h = tl.arange(0, BLOCK_H).to(INDEX_T)
            hm = h < NH
            zeros = tl.zeros((BLOCK_T, BLOCK_V), dtype=out.dtype.element_ty)
            ninf = tl.full((BLOCK_T, BLOCK_H), float("-inf"), dtype=tl.float32)
            # max_seq_len only sizes the virtual grid. If it
            # underestimates a zero-KV segment, every slot continues
            # through that segment rather than dropping later rows.
            for tile in range(item % ot, tl.cdiv(end - beg, BLOCK_T), ot):
                t = beg + tile * BLOCK_T + tl.arange(0, BLOCK_T).to(INDEX_T)
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
        span = max_seq_len if isinstance(max_seq_len, int) else total_tokens
        ot = max(1, triton.cdiv(min(span, total_tokens), block_t))
        items = batch * ot
        # ponytail: use i32 only while every row offset fits; oversized
        # tensors keep the existing i64 path.
        max_offset = max(
            (total_tokens - 1) * out.stride(0) + hv - 1,
            (total_tokens - 1) * lse.stride(0) + nh - 1,
        )
        index_t = tl.int32 if max_offset <= 2**31 - 1 else tl.int64
        _fixup_zero_kv[(min(items, _MAX_CTAS),)](
            out,
            lse,
            kv_lens,
            cum_seq_lens,
            batch,
            ot,
            items,
            out.stride(0),
            lse.stride(0),
            HV=hv,
            NH=nh,
            num_warps=_NUM_WARPS,
            BLOCK_T=block_t,
            BLOCK_V=4096,
            BLOCK_H=triton.next_power_of_2(max(1, nh)),
            INDEX_T=index_t,
        )
    return out, lse


__all__ = ["fixup_zero_kv"]
