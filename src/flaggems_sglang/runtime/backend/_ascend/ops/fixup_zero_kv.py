# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Ascend vendor for fixup_zero_kv, e-round 2: the UB-safe redesign of
# the 48555b55 e-round-1 vendor. The e1 kernel kept the right ruleset
# (int32 addressing everywhere, healthy segments exit after one scalar
# lens load, per-segment flat 1D out/lse streams) but its four
# fp32/int-gated branch bodies jointly overflowed the 192KB Ascend UB
# exactly the way T84 e15/e16 and T92 e20 died on BiShengHIR ("ub
# overflow ... multi-buffer"): at BLOCK=16384 a single masked branch
# holds zeros 32KB + offs 64KB + mask 64KB. This redesign takes the
# T40-e16 softcap_inplace_logits form (platform-validated on huawei):
# every non-final block stores with no mask at all - the hot path
# pays zero compares - and only the final block of each stream masks,
# with an exact int32 vector compare. Int32 compares are exact at any
# span, so the e1 fp32-domain gate (<2**24, fp32 tail masks losing
# precision past 2**24) and its four-way branch duplication are gone
# entirely; the degraded scalar fallback that the Ascend ruleset
# charges for integer vector CMP is confined to one block per stream.
# BLOCK 16384->4096 and num_warps 16->8 (one rung below the >=4096
# ladder entry on purpose, halving per-program lane pressure) drop a
# branch body to zeros 8KB + offs 16KB + int mask, leaving headroom
# even if BiShengHIR still allocates the out/lse masked/unmasked
# bodies jointly. Grid stays (batch, tiles) - axis 0 pins the segment
# so its metadata is read exactly once per program (the e17 flat-span
# falsification was the 1D item tiling that reloaded lens/cum per
# item) - but Ascend flattens a 2D grid onto a single <=65535-program
# axis, so the wrapper now caps the product, not just axis 1:
# tiles = min(cdiv(rows*hv, BLOCK), 255, 65535 // batch), and the
# in-kernel block stride keeps coverage whatever tile count the caps
# pick (a batch beyond the axis-0 limit cannot launch on any 2D grid
# shape; the generic 1D launch hits the same wall at batch*ot). This
# op is pure store (constants 0 / -inf), so the T92 masked-load
# other/MTE2 serialization pitfall does not apply.

import torch
import triton
import triton.language as tl

_BLOCK = 4096
_NUM_WARPS = 8
_MAX_TILES = 255
_MAX_GRID = 65535


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
    # scalar lens load and exit; the block loops stride by the grid's
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
        nsub_o = tl.cdiv(oe, BLOCK)
        nsub_l = tl.cdiv(le, BLOCK)
        # T40-e16 form: whole-block unmasked stores on every
        # non-final block, one int-masked store on the final block
        # per stream. The "last" test is a scalar compare (scalar CMP
        # has a full-rate path on Ascend; only vector integer CMP
        # degrades), and the tail mask is an exact int32 compare at
        # any span size - no fp32 mask domain anywhere.
        for i in range(tile, nsub_o, tl.num_programs(1)):
            offs = i * BLOCK + tl.arange(0, BLOCK)
            if i == nsub_o - 1:
                tl.store(out + beg * hv + offs, zeros, offs < oe)
            else:
                tl.store(out + beg * hv + offs, zeros)
        for i in range(tile, nsub_l, tl.num_programs(1)):
            offs = i * BLOCK + tl.arange(0, BLOCK)
            if i == nsub_l - 1:
                tl.store(lse + beg * nh + offs, ninf, offs < le)
            else:
                tl.store(lse + beg * nh + offs, ninf)


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
            max(1, (rows * hv + _BLOCK - 1) // _BLOCK),
            _MAX_TILES,
            # Ascend flattens the 2D grid onto one <=65535-program
            # axis: cap the product, not just axis 1. The floor at 1
            # keeps every segment reachable; a batch alone above the
            # axis-0 limit is unlaunchable on any grid shape.
            max(1, _MAX_GRID // batch),
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
