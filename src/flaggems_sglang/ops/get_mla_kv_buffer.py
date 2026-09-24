# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 98 get_mla_kv_buffer: gather paged MLA KV rows by loc and split
# each row into the NoPE and RoPE halves, storing into fresh tensors in
# the target dtypes (implicit conversion at store, RTNE like torch .to).
# One kernel: loc is loaded once per row tile and both halves stream
# from the same gathered row; all index arithmetic is i64.

import torch
import triton
import triton.language as tl


@triton.jit
def _get_mla_kv_buffer_kernel(
    kv_buffer,
    loc,
    nope_out,
    rope_out,
    n_rows,
    kv_stride,
    nope_dim,
    rope_dim,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid.to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
    rmask = rows < n_rows
    idx = tl.load(loc + rows, mask=rmask, other=0).to(tl.int64)
    base = idx * kv_stride

    cols = tl.arange(0, BLOCK_C).to(tl.int64)
    for c0 in range(0, nope_dim, BLOCK_C):
        cc = c0 + cols
        m = rmask[:, None] & (cc[None, :] < nope_dim)
        value = tl.load(kv_buffer + base[:, None] + cc[None, :], mask=m, other=0)
        dst = rows[:, None] * nope_dim + cc[None, :]
        tl.store(nope_out + dst, value.to(nope_out.dtype.element_ty), mask=m)
    for c0 in range(0, rope_dim, BLOCK_C):
        cc = c0 + cols
        m = rmask[:, None] & (cc[None, :] < rope_dim)
        value = tl.load(
            kv_buffer + base[:, None] + nope_dim + cc[None, :], mask=m, other=0
        )
        dst = rows[:, None] * rope_dim + cc[None, :]
        tl.store(rope_out + dst, value.to(rope_out.dtype.element_ty), mask=m)


def get_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope):
    n = loc.shape[0]
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = kv_buffer.shape[-1] - nope_dim
    nope = torch.empty(
        (n, nope_dim), dtype=cache_k_nope.dtype, device=kv_buffer.device
    )
    rope = torch.empty(
        (n, rope_dim), dtype=cache_k_rope.dtype, device=kv_buffer.device
    )
    if n and nope_dim and rope_dim:
        _get_mla_kv_buffer_kernel[(triton.cdiv(n, 8),)](
            kv_buffer,
            loc,
            nope,
            rope,
            n,
            kv_buffer.stride(0),
            nope_dim,
            rope_dim,
            BLOCK_R=8,
            BLOCK_C=512,
            num_warps=8,
        )
    return nope, rope


__all__ = ["get_mla_kv_buffer"]
