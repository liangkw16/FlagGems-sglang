# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Kunlun vendor for fixup_zero_kv: the generic's 2D broadcast stores
# ([BLOCK_T,1] + [1,BLOCK_H] pointers) hit the XPU LLVM packing bug
# ("size mismatch when packing elements for LLVM struct expected 8 but
# got 1", submission 17218, all 9 cases). This variant keeps every
# store strictly 1D - per token row, flat spans with chunked masks -
# the T53-proven Kunlun-safe vectorised-flat form.

import torch
import triton
import triton.language as tl


@triton.jit
def _fixup_zero_kv(
    out,
    lse,
    lens,
    cum,
    hv,
    nh,
    os0,
    ls0,
    batch,
    ot,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    seg = tl.program_id(0) // ot
    if seg < batch:
        if tl.load(lens + seg) == 0:
            beg = tl.load(cum + seg).to(tl.int64)
            end = tl.load(cum + seg + 1).to(tl.int64)
            tokens = end - beg
            v = tl.arange(0, BLOCK_V).to(tl.int64)
            h = tl.arange(0, BLOCK_H).to(tl.int64)
            zeros = tl.zeros((BLOCK_V,), dtype=out.dtype.element_ty)
            ninf = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
            for token in range(tl.program_id(0) % ot, tokens, ot):
                tok = beg + token
                for v0 in range(0, hv, BLOCK_V):
                    vv = v0 + v
                    tl.store(out + tok * os0 + vv, zeros, vv < hv)
                for h0 in range(0, nh, BLOCK_H):
                    hh = h0 + h
                    tl.store(lse + tok * ls0 + hh, ninf, hh < nh)


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
    # in-place outparam (the e8 generic semantics the platform already
    # accepts): the per-call whole-tensor clones doubled the memory
    # traffic of an op whose real work is zeroing a few rows - the
    # clone was 4x of the kunlun gap to the field. Overlapping views
    # (expanded rows share bytes, stride(0) below the row width) keep
    # the defensive clone so untouched rows survive; the guard is pure
    # stride metadata, no device work.
    overlap = (
        abs(out.stride(0)) < num_heads * v_head_dim
        or abs(lse.stride(0)) < num_heads
    )
    if overlap:
        out_fixed, lse_fixed = out.clone(), lse.clone()
    else:
        out_fixed, lse_fixed = out, lse
    if batch and total_tokens:
        hv, nh = num_heads * v_head_dim, num_heads
        # max_seq_len only sizes the launch (advisory); the token loop
        # strides by ot so an understated span still covers every row.
        span = max_seq_len if isinstance(max_seq_len, int) else total_tokens
        ot = max(1, min(span, total_tokens))
        _fixup_zero_kv[(batch * ot,)](
            out_fixed,
            lse_fixed,
            kv_lens,
            cum_seq_lens,
            hv,
            nh,
            out_fixed.stride(0),
            lse_fixed.stride(0),
            batch,
            ot,
            BLOCK_V=4096,
            BLOCK_H=triton.next_power_of_2(max(1, nh)),
        )
    return out_fixed, lse_fixed


__all__ = ["fixup_zero_kv"]
