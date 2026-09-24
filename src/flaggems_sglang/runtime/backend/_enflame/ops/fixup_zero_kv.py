# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Enflame vendor for fixup_zero_kv, e22 flat-span round: a zero-KV
# segment's out rows are one CONTIGUOUS flat span (beg*HV .. end*HV)
# and its lse rows another, so the fill streams each span as pure 1D
# wide stores - the maximal DMA shape - instead of the e8-e16
# [BLOCK_T, BLOCK_V] 2D tiles that interleave the out and lse streams
# per tile. All addressing is int32 with an explicit wrapper domain
# branch (enable_i64=False emulates int64; 2^31-scale inputs take the
# _fixup_zero_kv_i64 fallback = the e16 tile form, both paths Triton).
# Launch keeps the gcu300 12-CTA cap, warps 2 pinned (the e26 unpin
# lost 68.1 vs 109.5), num_stages 3.

import torch
import triton
import triton.language as tl

_MAX_CTAS = 12
_BLOCK_F = 16384


@triton.jit(do_not_specialize=["batch", "ot2"])
def _fixup_zero_kv_flat(
    out,
    lse,
    lens,
    cum,
    batch,
    ot2,
    HV: tl.constexpr,
    NH: tl.constexpr,
    GR: tl.constexpr,
    BLOCK_F: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # item = (segment, row group): the lens check runs once per GR
    # rows exactly like the parent tile form (the per-flat-chunk item
    # form paid one lens load per chunk - a 16x check inflation on
    # healthy spans); the group loop strides by ot2 so an understated
    # host estimate still covers every group, and each group's out/lse
    # extents stream as flat contiguous stores (the DMA-pure shape).
    for item in range(tl.program_id(0), batch * ot2, tl.num_programs(0)):
        seg = item // ot2
        if tl.load(lens + seg) == 0:
            beg = tl.load(cum + seg)
            end = tl.load(cum + seg + 1)
            rows = end - beg
            groups = tl.cdiv(rows, GR)
            zeros = tl.zeros((BLOCK_F,), dtype=out.dtype.element_ty)
            ninf = tl.full((BLOCK_L,), float("-inf"), dtype=tl.float32)
            for g in range(item % ot2, groups, ot2):
                r0 = g * GR
                rr = tl.minimum(GR, rows - r0)
                base = (beg + r0) * HV
                spano = rr * HV
                for c0 in range(0, spano, BLOCK_F):
                    offs = c0 + tl.arange(0, BLOCK_F)
                    tl.store(out + base + offs, zeros, offs < spano)
                basel = (beg + r0) * NH
                spanl = rr * NH
                for c0 in range(0, spanl, BLOCK_L):
                    offs = c0 + tl.arange(0, BLOCK_L)
                    tl.store(lse + basel + offs, ninf, offs < spanl)


@triton.jit(do_not_specialize=["items", "ot"])
def _fixup_zero_kv_i64(
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
):
    for item in range(tl.program_id(0), items, tl.num_programs(0)):
        seg = item // ot
        zero = tl.load(lens + seg) == 0
        if zero:
            beg = tl.load(cum + seg).to(tl.int64)
            end = tl.load(cum + seg + 1).to(tl.int64)
            v = tl.arange(0, BLOCK_V).to(tl.int64)
            h = tl.arange(0, BLOCK_H).to(tl.int64)
            hm = h < NH
            zeros = tl.zeros((BLOCK_T, BLOCK_V), dtype=out.dtype.element_ty)
            ninf = tl.full((BLOCK_T, BLOCK_H), float("-inf"), dtype=tl.float32)
            for tile in range(item % ot, tl.cdiv(end - beg, BLOCK_T), ot):
                t = beg + tile * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
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
        # explicit domain branch (survives python -O): the flat int32
        # fast path covers every realistic KV cache shape; 2^31-scale
        # inputs take the int64 tile fallback - both paths Triton
        # flat fast path gates: dense row pitch on both outputs (a
        # strided view would place beg*HV at the wrong physical row -
        # the tile fallback carries the real stride(0)), the int32
        # element domain, and an int32-safe item bound batch*ot2
        ot2 = max(1, triton.cdiv(min(span, total_tokens), 8))
        if (
            out.stride(0) == hv
            and lse.stride(0) == nh
            and total_tokens * hv < 2**31
            and total_tokens * nh < 2**31
            and batch * ot2 < 2**31
        ):
            _fixup_zero_kv_flat[(min(batch * ot2, _MAX_CTAS),)](
                out,
                lse,
                kv_lens,
                cum_seq_lens,
                batch,
                ot2,
                HV=hv,
                NH=nh,
                GR=8,
                BLOCK_F=_BLOCK_F,
                BLOCK_L=triton.next_power_of_2(max(1, nh)),
                num_warps=2,
                num_stages=3,
            )
            return out, lse
        ot = max(1, triton.cdiv(min(span, total_tokens), block_t))
        items = batch * ot
        _fixup_zero_kv_i64[(min(items, _MAX_CTAS),)](
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
            num_warps=2,
            BLOCK_T=block_t,
            BLOCK_V=4096,
            BLOCK_H=triton.next_power_of_2(max(1, nh)),
        )
    return out, lse


__all__ = ["fixup_zero_kv"]
