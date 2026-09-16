# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 8014d9d: kernels/ops/moe/gate_topk.py.

import torch
import triton
import triton.language as tl


@triton.jit
def get_topmask_and_fullmask(x):
    tl.static_assert(
        x.dtype.is_int_unsigned(),
        "floating-point value must be passed as bits",
    )
    tm: tl.constexpr = 1 << (-1 + x.dtype.primitive_bitwidth)
    fm: tl.constexpr = (1 << x.dtype.primitive_bitwidth) - 1
    tm_arr = tl.full(x.shape, tm, dtype=x.dtype)
    fm_arr = tl.full(x.shape, fm, dtype=x.dtype)
    return tm_arr, fm_arr


@triton.jit
def fpval_to_key(x, is_nan):
    tm, fm = get_topmask_and_fullmask(x)
    # Numeric ties include -0 and +0. Keep one sorting key for both;
    # selected values are reloaded from the original input below.
    x = tl.where((x & (tm - 1)) == 0, tl.full(x.shape, 0, x.dtype), x)
    key = x ^ tl.where((x & tm) != 0, fm, tm)
    # All NaN signs/payloads sort above +inf and tie by column index.
    return tl.where(is_nan, fm, key)


@triton.jit
def _streaming_topk_kernel(
    x_ptr,
    stride_xm,
    values_ptr,
    indices_ptr,
    M,
    N,
    m_blocks,
    N_PAD: tl.constexpr,
    K: tl.constexpr,
    K_POW2: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    for pid in range(tl.program_id(0), m_blocks, tl.num_programs(0)):
        offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offs_m < M

        # Sorting uses an unsigned dtype with the value packed into the upper
        # bits and the column index into the lower bits.
        x_nbits: tl.constexpr = x_ptr.dtype.element_ty.primitive_bitwidth
        x_utype: tl.constexpr = tl.dtype(f"uint{x_nbits}")
        # The largest tag is N_PAD (column zero), not N_PAD - 1.
        INDEX_BITS: tl.constexpr = 16 if N_PAD < 65536 else 32
        if x_nbits + INDEX_BITS <= 32:
            y_nbits: tl.constexpr = 32
        else:
            y_nbits: tl.constexpr = 64
        tl.static_assert(x_nbits + INDEX_BITS <= y_nbits)
        x_ultype: tl.constexpr = tl.dtype(f"uint{y_nbits}")

        # Iterate in reverse column order so only the 1st iter needs a mask.
        num_iters: tl.constexpr = N_PAD // BLOCK_SIZE_N - 1
        offs_x_n = num_iters * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offs_x_n < N

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_x_n[None, :]
        x = tl.load(
            x_ptrs,
            mask=(mask_m[:, None] & mask_n[None, :]),
            other=float("-inf"),
        )
        x = fpval_to_key(x.to(x_utype, bitcast=True), x != x)
        x = (x.to(x_ultype) << INDEX_BITS) | (N_PAD - offs_x_n)[None, :].to(
            x_ultype
        )
        acc = tl.topk(x, K_POW2, dim=1)

        for _i in (tl.static_range if num_iters <= 4 else range)(num_iters):
            acc = tl.bitonic_merge(acc)  # sorted ascending for the merge
            x_ptrs -= BLOCK_SIZE_N
            offs_x_n -= BLOCK_SIZE_N
            x = tl.load(x_ptrs, mask=mask_m[:, None], other=float("-inf"))
            x = fpval_to_key(x.to(x_utype, bitcast=True), x != x)
            x = (x.to(x_ultype) << INDEX_BITS) | (N_PAD - offs_x_n)[
                None, :
            ].to(x_ultype)
            acc = tl.maximum(acc, tl.topk(x, K_POW2, dim=1))

        offs_k = tl.arange(0, K_POW2)
        mask_k = offs_k < K
        acc = tl.sort(acc, dim=1, descending=True)
        acc = tl.where(mask_k[None, :], acc, 0)
        y_indices_raw = (acc & ((1 << INDEX_BITS) - 1)).to(tl.uint32)
        y_indices = N_PAD - y_indices_raw
        offs_mk = offs_m[:, None].to(tl.int64) * K + offs_k[None, :]
        mask_mk = mask_m[:, None] & mask_k[None, :]
        # Preserve the selected input bits, including a tied negative zero.
        y_values = tl.load(
            x_ptr
            + offs_m[:, None].to(tl.int64) * stride_xm
            + y_indices.to(tl.int64),
            mask=mask_mk,
            other=0,
        )
        tl.store(values_ptr + offs_mk, y_values, mask=mask_mk)
        tl.store(indices_ptr + offs_mk, y_indices, mask=mask_mk)


def gate_topk(x, k):
    assert x.is_contiguous() and x.ndim == 2
    assert x.numel() <= 2**31
    assert 1 <= k <= 32
    n_rows, n_cols = x.shape
    assert k <= n_cols
    values = torch.empty((n_rows, k), dtype=x.dtype, device=x.device)
    indices = torch.empty((n_rows, k), dtype=torch.int32, device=x.device)
    if n_rows and n_cols:
        block_m = 32
        m_blocks = triton.cdiv(n_rows, block_m)
        _streaming_topk_kernel[(min(m_blocks, 65535),)](
            x,
            x.stride(0),
            values,
            indices,
            n_rows,
            n_cols,
            m_blocks,
            N_PAD=triton.cdiv(n_cols, 32) * 32,
            K=k,
            K_POW2=triton.next_power_of_2(k),
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=32,
        )
    return values, indices


__all__ = ["gate_topk"]
