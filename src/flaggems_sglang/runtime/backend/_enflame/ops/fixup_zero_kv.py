# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Enflame vendor for fixup_zero_kv: the e8 in-place core (zero-KV rows
# only, healthy segments exit immediately) on the official gcu300
# geometry - the e9 port kept the generic's batch*ot grid and only
# pinned num_warps, which is not the 12-CTA clamp the GCU codegen
# documents (max_grid_size=(12,1,1)). This round flattens the
# (segment, tile) work items and grid-strides them across at most 12
# programs with num_warps=2 and wide flat stores; the e26 unpin read
# 68.1 vs 109.5 - the pin is load-bearing on this store shape - so the
# warp count stays pinned; e15 doubles the
# store width to BLOCK_V=2048 (the gcu300 tile guidance scales with
# the element width and the e12 read 102.6 with 1024-wide stores -
# the width ladder is live: 102.6 @1024 -> 113.9 @2048, e16 takes
# the 4096 rung toward the 193 field band).
# E30 (2026-09-25): single-variable addressing-domain candidate. The
# chip ruleset pins GCU `enable_i64=False` (int64 addressing arithmetic
# is soft-emulated; all-int32 is the documented form), and the whole
# offset chain here is i64 (beg/end/arange/t*os0). This round keeps the
# [BLOCK_T=8, BLOCK_V=4096] tile, the item mapping, warps2 and the
# 12-CTA launch byte-for-byte and only swaps that chain for int32 on
# shapes whose every computed offset fits signed int32; the wrapper
# picks the kernel by host-side shape/stride arithmetic (no device
# probe, no try/except, no cache) and out-of-domain shapes keep the
# original i64 kernel bytes. Unlike the e22r2 flat-span rewrite (four
# variables at once, -10%), this changes exactly one variable.

import torch
import triton
import triton.language as tl

_MAX_CTAS = 12
_BLOCK_T = 8
_BLOCK_V = 4096
_INT32_LIMIT = 2**31


@triton.jit
def _fixup_zero_kv32(
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
    # The e30 twin of _fixup_zero_kv with the addressing chain in
    # int32: beg/end load as the cum dtype's native i32, the arange
    # lanes stay i32, and t*os0 is an i32 multiply. The wrapper only
    # selects this kernel when every lane offset (masked lanes
    # included - they still compute their addresses) stays inside
    # signed int32.
    for item in range(tl.program_id(0), items, tl.num_programs(0)):
        seg = item // ot
        zero = tl.load(lens + seg) == 0
        if zero:
            beg = tl.load(cum + seg)
            end = tl.load(cum + seg + 1)
            v = tl.arange(0, BLOCK_V)
            h = tl.arange(0, BLOCK_H)
            hm = h < NH
            zeros = tl.zeros((BLOCK_T, BLOCK_V), dtype=out.dtype.element_ty)
            ninf = tl.full((BLOCK_T, BLOCK_H), float("-inf"), dtype=tl.float32)
            # max_seq_len only sizes the virtual grid. If it
            # underestimates a zero-KV segment, every slot continues
            # through that segment rather than dropping later rows.
            for tile in range(item % ot, tl.cdiv(end - beg, BLOCK_T), ot):
                t = beg + tile * BLOCK_T + tl.arange(0, BLOCK_T)
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
            # max_seq_len only sizes the virtual grid. If it
            # underestimates a zero-KV segment, every slot continues
            # through that segment rather than dropping later rows.
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


def _fits_int32(total_tokens, os0, ls0, hv, block_h, items):
    # Host-side addressing-domain predicate (pure shape/stride
    # arithmetic, no device read). It bounds every value the kernel
    # computes in i32, not just the stored offsets: the ragged token
    # tail runs t lanes up to end + BLOCK_T - 1, the folded value tail
    # runs v lanes up to HV + BLOCK_V - 1 past the row (masked lanes
    # still compute their addresses), and the tile-loop bound
    # tl.cdiv(end - beg, BLOCK_T) computes end - beg + BLOCK_T - 1 -
    # an intermediate, not an offset. The stride products alone cannot
    # bound the token count: expand()-built broadcast views carry
    # stride(0)==0 on both streams (the wrapper's stride(1)==1
    # assertions then force NH=1), and review r2 showed
    # _fits_int32(2**31-1, 0, 0, 1, 1, 1) was True while the wrapped
    # cdiv turned the loop bound negative, silently skipping every
    # write. The token clause bounds both the t lanes and - given
    # cum[-1] <= total_tokens, the contract every kernel's OOB safety
    # already assumes - the cdiv intermediate. Stride-based products
    # stay so a padded-row view cannot smuggle a huge os0 into the
    # i32 kernel.
    t_hi = total_tokens + _BLOCK_T
    return (
        t_hi < _INT32_LIMIT
        and items < _INT32_LIMIT
        and t_hi * os0 + hv + _BLOCK_V < _INT32_LIMIT
        and t_hi * ls0 + block_h < _INT32_LIMIT
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
        block_h = triton.next_power_of_2(max(1, nh))
        span = max_seq_len if isinstance(max_seq_len, int) else total_tokens
        ot = max(1, triton.cdiv(min(span, total_tokens), _BLOCK_T))
        items = batch * ot
        kern = (
            _fixup_zero_kv32
            if _fits_int32(
                total_tokens,
                out.stride(0),
                lse.stride(0),
                hv,
                block_h,
                items,
            )
            else _fixup_zero_kv
        )
        kern[(min(items, _MAX_CTAS),)](
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
            BLOCK_T=_BLOCK_T,
            BLOCK_V=_BLOCK_V,
            BLOCK_H=block_h,
        )
    return out, lse


__all__ = ["fixup_zero_kv"]
