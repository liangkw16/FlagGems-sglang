# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 99 ltx2_split_rotary: split-half (rotate-half) rotary embedding
# with per-head cos/sin tables. x is [B, T, H*D] bf16; cos/sin are
# [B, H, T, half] and read through their own strides (no permute copy).
# The reference's deliberate intermediate rounding is reproduced
# exactly: each product rounds to bf16 first, then the add/sub runs in
# fp32 (one final rounding at the store).

import torch
import triton
import triton.language as tl


@triton.jit
def _ltx2_split_rotary_kernel(
    x,
    cos,
    sin,
    out,
    T,
    inner,
    half,
    x_s0,
    x_s1,
    c_s0,
    c_s1,
    c_s2,
    c_s3,
    o_s0,
    o_s1,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    n_heads = inner // HEAD_DIM
    bt = pid // n_heads
    h = pid - bt * n_heads
    t = bt % T
    b = bt // T
    row = b * x_s0 + t * x_s1
    crow = b * c_s0 + h * c_s1 + t * c_s2
    d = tl.arange(0, BLOCK_H).to(tl.int64)
    m = d < half
    x1 = tl.load(x + row + h * HEAD_DIM + d, mask=m, other=0).to(tl.float32)
    x2 = tl.load(
        x + row + h * HEAD_DIM + half + d, mask=m, other=0
    ).to(tl.float32)
    c = tl.load(cos + crow + d * c_s3, mask=m, other=0).to(tl.float32)
    s = tl.load(sin + crow + d * c_s3, mask=m, other=0).to(tl.float32)
    o1 = (x1 * c).to(out.dtype.element_ty).to(tl.float32) - x2 * s
    o2 = (x2 * c).to(out.dtype.element_ty).to(tl.float32) + x1 * s
    tl.store(
        out + row + h * HEAD_DIM + d,
        o1.to(out.dtype.element_ty),
        mask=m,
    )
    tl.store(
        out + row + h * HEAD_DIM + half + d,
        o2.to(out.dtype.element_ty),
        mask=m,
    )


def ltx2_split_rotary(x, cos, sin):
    assert x.dim() == 3
    batch, seq_len, inner = x.shape
    _, num_heads, _, half = cos.shape
    head_dim = half * 2
    assert inner == num_heads * head_dim
    out = torch.empty_like(x)
    if out.numel():
        grid = (batch * seq_len * num_heads,)
        _ltx2_split_rotary_kernel[grid](
            x,
            cos,
            sin,
            out,
            seq_len,
            inner,
            half,
            x.stride(0),
            x.stride(1),
            cos.stride(0),
            cos.stride(1),
            cos.stride(2),
            cos.stride(3),
            out.stride(0),
            out.stride(1),
            HEAD_DIM=head_dim,
            BLOCK_H=max(16, triton.next_power_of_2(half)),
            num_warps=4,
        )
    return out


__all__ = ["ltx2_split_rotary"]
