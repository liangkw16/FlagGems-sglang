# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 105 set_mla_kv_buffer: scatter-write mirror of T98 - each source
# row i concatenates its NoPE and RoPE halves into one contiguous row
# written to slot loc[i] of the cloned buffer. One 2D row-tile program
# (the T98 generic form, platform-validated 7/7 on the same row width)
# with i64 loc addressing; untouched slots keep the cloned base bytes.

import torch
import triton
import triton.language as tl


@triton.jit
def _set_mla_kv_buffer_kernel(
    kv_buffer,
    loc,
    nope,
    rope,
    n_rows,
    loc_s0,
    kv_s0,
    kv_s1,
    nope_dim,
    rope_dim,
    ns0,
    ns1,
    rs0,
    rs1,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid.to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
    rmask = rows < n_rows
    idx = tl.load(loc + rows * loc_s0, mask=rmask, other=0).to(tl.int64)
    base = idx * kv_s0
    cols = tl.arange(0, BLOCK_C).to(tl.int64)
    for c0 in range(0, nope_dim, BLOCK_C):
        cc = c0 + cols
        m = rmask[:, None] & (cc[None, :] < nope_dim)
        v = tl.load(
            nope + rows[:, None] * ns0 + cc[None, :] * ns1, mask=m, other=0
        )
        tl.store(kv_buffer + base[:, None] + cc[None, :] * kv_s1, v, mask=m)
    for c0 in range(0, rope_dim, BLOCK_C):
        cc = c0 + cols
        m = rmask[:, None] & (cc[None, :] < rope_dim)
        v = tl.load(
            rope + rows[:, None] * rs0 + cc[None, :] * rs1, mask=m, other=0
        )
        tl.store(kv_buffer + base[:, None] + (nope_dim + cc[None, :]) * kv_s1, v, mask=m)


def set_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope):
    n = loc.shape[0]
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    out = kv_buffer.clone(memory_format=torch.contiguous_format)
    if n and (nope_dim or rope_dim):
        _set_mla_kv_buffer_kernel[(triton.cdiv(n, 8),)](
            out,
            loc,
            cache_k_nope,
            cache_k_rope,
            n,
            loc.stride(0),
            out.stride(0),
            out.stride(1),
            nope_dim,
            rope_dim,
            cache_k_nope.stride(0),
            cache_k_nope.stride(1),
            cache_k_rope.stride(0),
            cache_k_rope.stride(1),
            BLOCK_R=8,
            BLOCK_C=512,
            num_warps=8,
        )
    return out


__all__ = ["set_mla_kv_buffer"]
