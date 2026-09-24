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


@triton.jit(do_not_specialize=["batch", "oct", "lct"])
def _fixup_zero_kv_flat(
    out,
    lse,
    lens,
    cum,
    batch,
    oct,
    lct,
    HV: tl.constexpr,
    NH: tl.constexpr,
    BLOCK_F: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # out pass: each item is one (segment, flat chunk) pair; the pass
    # bound is batch * oct (its own chunk pitch - sharing the other
    # pass's bound would push seg past batch); oct is the host's
    # chunks-per-span estimate and the chunk loop strides by it, so an
    # understated estimate still covers every chunk
    for item in range(tl.program_id(0), batch * oct, tl.num_programs(0)):
        seg = item // oct
        if tl.load(lens + seg) == 0:
            beg = tl.load(cum + seg)
            end = tl.load(cum + seg + 1)
            base = beg * HV
            span = (end - beg) * HV
            zeros = tl.zeros((BLOCK_F,), dtype=out.dtype.element_ty)
            for c in range(item % oct, tl.cdiv(span, BLOCK_F), oct):
                offs = c * BLOCK_F + tl.arange(0, BLOCK_F)
                tl.store(out + base + offs, zeros, offs < span)
    # lse pass: flat span over tokens*NH with its own batch * lct
    # bound; lct is the host's per-span chunk estimate with the same
    # stride-cover property
    for item in range(tl.program_id(0), batch * lct, tl.num_programs(0)):
        seg = item // lct
        if tl.load(lens + seg) == 0:
            beg = tl.load(cum + seg)
            end = tl.load(cum + seg + 1)
            base = beg * NH
            span = (end - beg) * NH
            ninf = tl.full((BLOCK_L,), float("-inf"), dtype=tl.float32)
            for c in range(item % lct, tl.cdiv(span, BLOCK_L), lct):
                offs = c * BLOCK_L + tl.arange(0, BLOCK_L)
                tl.store(lse + base + offs, ninf, offs < span)


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
        if total_tokens * hv < 2**31 and total_tokens * nh < 2**31:
            fs = min(span, total_tokens)
            oct_ = max(1, triton.cdiv(fs * hv, _BLOCK_F))
            lct = max(1, triton.cdiv(fs * nh, triton.next_power_of_2(max(1, nh))))
            _fixup_zero_kv_flat[
                (min(batch * max(oct_, lct), _MAX_CTAS),)
            ](
                out,
                lse,
                kv_lens,
                cum_seq_lens,
                batch,
                oct_,
                lct,
                HV=hv,
                NH=nh,
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
