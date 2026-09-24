# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 94 concat_mla_absorb_q: cat the NoPE and RoPE halves of absorbed
# Q along the last dim into one contiguous row. The two sources carry
# their own row strides, so both base pointers are computed from the
# runtime strides and each half streams with its own masked column run;
# a single kernel writes the full destination row in one pass.

import torch
import triton
import triton.language as tl


@triton.jit
def _concat_mla_absorb_q_kernel(
    a,
    b,
    out,
    n_rows,
    d1,
    a_s0,
    a_s1,
    a_s2,
    b_s0,
    b_s1,
    b_s2,
    a_last,
    b_last,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid.to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
    rmask = rows < n_rows
    i0 = rows // d1
    i1 = rows - i0 * d1
    a_base = i0 * a_s0 + i1 * a_s1
    b_base = i0 * b_s0 + i1 * b_s1
    o_base = rows * (a_last + b_last)
    cols = tl.arange(0, BLOCK_C).to(tl.int64)
    for c0 in range(0, a_last, BLOCK_C):
        cc = c0 + cols
        m = rmask[:, None] & (cc[None, :] < a_last)
        value = tl.load(a + a_base[:, None] + cc[None, :] * a_s2, mask=m, other=0)
        tl.store(out + o_base[:, None] + cc[None, :], value, mask=m)
    for c0 in range(0, b_last, BLOCK_C):
        cc = c0 + cols
        m = rmask[:, None] & (cc[None, :] < b_last)
        value = tl.load(b + b_base[:, None] + cc[None, :] * b_s2, mask=m, other=0)
        tl.store(out + o_base[:, None] + a_last + cc[None, :], value, mask=m)


def concat_mla_absorb_q(a, b):
    assert a.dtype == b.dtype
    assert a.dim() == 3 and b.dim() == 3
    assert a.shape[:-1] == b.shape[:-1]
    a_last = a.shape[-1]
    b_last = b.shape[-1]
    d0, d1 = a.shape[0], a.shape[1]
    n_rows = d0 * d1
    out = torch.empty(
        (d0, d1, a_last + b_last), dtype=a.dtype, device=a.device
    )
    if n_rows and (a_last + b_last):
        _concat_mla_absorb_q_kernel[(triton.cdiv(n_rows, 4),)](
            a,
            b,
            out,
            n_rows,
            d1,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            a_last,
            b_last,
            BLOCK_R=4,
            BLOCK_C=512,
            num_warps=8,
        )
    return out


__all__ = ["concat_mla_absorb_q"]
