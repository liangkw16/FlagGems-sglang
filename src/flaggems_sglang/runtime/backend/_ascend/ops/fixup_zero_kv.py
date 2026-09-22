# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Ascend vendor for fixup_zero_kv, e-round 1: the T92-e20 ruleset
# migrated onto the e8 in-place core (zero-KV rows only, healthy
# segments exit after one scalar lens load). The triton-ascend
# performance guidelines: the vector compare unit has no int path
# (int32/int64 compares degrade to scalar) and vector ADD has no
# int64 - so every offset here is int32 and the tail masks run in
# fp32, replacing the generic's int64 arange arithmetic and integer
# vector compares. Grid is (batch, min(tiles, 255)): axis 0 pins the
# segment so its metadata is read exactly once per program (the e17
# flat-span falsification was the 1D item tiling that reloaded
# lens/cum per item and cut lse into 2048 blocks - here both out and
# lse are single flat 1D streams of 16384-wide blocks per segment),
# and the tile cap bounds the program count the way CANN's
# per-program cost punished the uncapped per-request grid on T77.
# This op is pure store (constants 0 / -inf), so the T92 masked-load
# other/MTE2 serialization pitfall does not apply. num_warps=16 per
# the official >=4096-element tile rung.

import torch
import triton
import triton.language as tl

_BLOCK = 16384
_NUM_WARPS = 16
_MAX_TILES = 255


@triton.jit
def _fixup_zero_kv(
    out,
    lse,
    lens,
    cum,
    hv,
    nh,
    lstride,
    cstride,
    BLOCK: tl.constexpr,
):
    # One metadata read per program: healthy segments pay a single
    # scalar lens load and exit; the tile loop strides by the grid's
    # axis-1 size so an understated host tile estimate (advisory
    # max_seq_len) still covers every element.
    seg = tl.program_id(0)
    tile = tl.program_id(1)
    if tl.load(lens + seg * lstride) == 0:
        beg = tl.load(cum + seg * cstride)
        end = tl.load(cum + (seg + 1) * cstride)
        rows = end - beg
        zeros = tl.zeros((BLOCK,), dtype=out.dtype.element_ty)
        ninf = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
        oe = rows * hv
        le = rows * nh
        # fp32 compares lose precision past 2**24 elements (the
        # boundary offset rounds up and the last element goes
        # unwritten - the T92-e20 codex-review find), so the
        # vector-mask fast path is gated to the sub-2**24 domain by a
        # scalar branch; larger spans keep the exact integer mask
        if oe < 16777216:
            for base in range(
                tile * BLOCK, oe, tl.num_programs(1) * BLOCK
            ):
                offs = base + tl.arange(0, BLOCK)
                m = offs.to(tl.float32) < oe.to(tl.float32)
                tl.store(out + beg * hv + offs, zeros, m)
        else:
            for base in range(
                tile * BLOCK, oe, tl.num_programs(1) * BLOCK
            ):
                offs = base + tl.arange(0, BLOCK)
                m = offs < oe
                tl.store(out + beg * hv + offs, zeros, m)
        if le < 16777216:
            for base in range(
                tile * BLOCK, le, tl.num_programs(1) * BLOCK
            ):
                offs = base + tl.arange(0, BLOCK)
                m = offs.to(tl.float32) < le.to(tl.float32)
                tl.store(lse + beg * nh + offs, ninf, m)
        else:
            for base in range(
                tile * BLOCK, le, tl.num_programs(1) * BLOCK
            ):
                offs = base + tl.arange(0, BLOCK)
                m = offs < le
                tl.store(lse + beg * nh + offs, ninf, m)


def fixup_zero_kv(out, lse, kv_lens, cum_seq_lens, max_seq_len):
    assert out.ndim == 3 and lse.ndim == 2
    total_tokens, num_heads, v_head_dim = out.shape
    assert lse.shape == (total_tokens, num_heads)
    assert out.dtype in (torch.float16, torch.bfloat16)
    assert lse.dtype == torch.float32
    # the flat-span streams address a whole segment as one contiguous
    # 1D range, so both buffers must be row-contiguous end to end (the
    # generic contract only pins contiguity inside a token row)
    assert out.is_contiguous() and lse.is_contiguous()
    # int32 element offsets
    assert out.numel() < 2**31 and lse.numel() < 2**31
    batch = kv_lens.numel()
    assert cum_seq_lens.numel() == batch + 1
    assert kv_lens.dtype == cum_seq_lens.dtype == torch.int32
    if batch and total_tokens:
        hv, nh = num_heads * v_head_dim, num_heads
        # max_seq_len only sizes the launch (advisory); the in-kernel
        # tile stride keeps coverage if it understates the real spans
        span = max_seq_len if isinstance(max_seq_len, int) else total_tokens
        rows = min(span, total_tokens)
        tiles = min(
            max(1, (rows * hv + _BLOCK - 1) // _BLOCK), _MAX_TILES
        )
        _fixup_zero_kv[(batch, tiles)](
            out,
            lse,
            kv_lens,
            cum_seq_lens,
            hv,
            nh,
            kv_lens.stride(0),
            cum_seq_lens.stride(0),
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
    return out, lse


__all__ = ["fixup_zero_kv"]
