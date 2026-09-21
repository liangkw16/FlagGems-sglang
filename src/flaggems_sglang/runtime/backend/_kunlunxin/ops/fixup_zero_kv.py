# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# KunlunXin vendor for fixup_zero_kv: the e8 in-place core (zero-KV
# writes only, healthy segments exit immediately) in the XPU-safe flat
# form. The e9 port kept the generic's (BLOCK_T, BLOCK_H) broadcast
# lse store and hit the XPU LLVM packing bug ("size mismatch when
# packing elements for LLVM struct expected 8 but got 1" - the same
# family that sank e1/e2); every store here is a flat 1D span: the out
# row as BLOCK_V chunks of the contiguous head*dim vector, the lse row
# as one contiguous BLOCK_H span.

import torch
import triton
import triton.language as tl


@triton.jit
def _fixup_zero_kv(
    out,
    lse,
    lens,
    cum,
    batch,
    ot,
    os0,
    ls0,
    HV: tl.constexpr,
    NH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    seg = tl.program_id(0)
    if seg < batch:
        zero = tl.load(lens + seg) == 0
        if zero:
            beg = tl.load(cum + seg).to(tl.int64)
            end = tl.load(cum + seg + 1).to(tl.int64)
            v = tl.arange(0, BLOCK_V).to(tl.int64)
            h = tl.arange(0, BLOCK_H).to(tl.int64)
            zeros = tl.zeros((BLOCK_V,), dtype=out.dtype.element_ty)
            ninf = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
            hm = h < NH
            for t in range(beg, end):
                base = t * os0
                for v0 in tl.static_range(0, HV, BLOCK_V):
                    vv = v0 + v
                    m = vv < HV
                    tl.store(out + base + vv, zeros, m)
                tl.store(lse + t * ls0 + h, ninf, hm)


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
        _fixup_zero_kv[(batch,)](
            out,
            lse,
            kv_lens,
            cum_seq_lens,
            batch,
            1,
            out.stride(0),
            lse.stride(0),
            HV=hv,
            NH=nh,
            BLOCK_T=1,
            BLOCK_V=1024,
            BLOCK_H=triton.next_power_of_2(max(1, nh)),
        )
    return out, lse


__all__ = ["fixup_zero_kv"]
