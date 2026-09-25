# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 95 create_chunked_prefix_cache_kv_indices: chunked-prefill
# variant of the FlashInfer kv-indices builder. Request i copies its
# token-pool window req_to_token[pool_i, start_i : start_i+n_i] into
# out[cu_i : cu_i+n_i]; the untouched part of the destination keeps the
# base bytes. s0 cloned the whole base buffer (a full extra pass whose
# bytes are ~99% overwritten under the canonical packed layout the
# reference uses); e2 replaces the clone with a fill kernel that copies
# base bytes ONLY where no window lands - membership decided by a
# fixed-step binary search (searchsorted-right) over the non-decreasing
# cu array, so gapped layouts stay correct too. Two launches total,
# same as clone+window, but a third less traffic.

import torch
import triton
import triton.language as tl


@triton.jit
def _create_chunked_prefix_cache_kv_indices_kernel(
    req_to_token,
    req_pool_indices,
    chunk_start_idx,
    chunk_seq_lens,
    chunk_cu_seq_lens,
    out,
    pool_s0,
    start_s0,
    len_s0,
    cu_s0,
    r2t_s0,
    r2t_s1,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    pool = tl.load(req_pool_indices + row * pool_s0).to(tl.int64)
    start = tl.load(chunk_start_idx + row * start_s0).to(tl.int64)
    length = tl.load(chunk_seq_lens + row * len_s0).to(tl.int64)
    beg = tl.load(chunk_cu_seq_lens + row * cu_s0).to(tl.int64)
    cols = tl.arange(0, BLOCK_C).to(tl.int64)
    for c0 in range(0, length, BLOCK_C):
        cc = c0 + cols
        m = cc < length
        value = tl.load(
            req_to_token + pool * r2t_s0 + (start + cc) * r2t_s1, mask=m, other=0
        )
        tl.store(out + beg + cc, value, mask=m)


@triton.jit
def _fill_non_window_kernel(
    chunk_cu_seq_lens,
    chunk_seq_lens,
    chunk_kv_indices,
    out,
    total,
    n_req,
    cu_s0,
    len_s0,
    BLOCK: tl.constexpr,
    STEPS: tl.constexpr,
):
    # copy base bytes into every destination element not covered by any
    # window [cu_i, cu_i + n_i). cu is non-decreasing (cumsum layout),
    # so the covering window, if one exists, is index (count of cu<=e)
    # minus one - found by a fixed-step searchsorted-right loop over
    # ceil(log2(n_req+1)) iterations that all lanes run to convergence.
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < total
    big = 9223372036854775807
    lo = tl.zeros((BLOCK,), dtype=tl.int64)
    hi = tl.full((BLOCK,), n_req, dtype=tl.int64)
    for _ in range(0, STEPS):
        mid = (lo + hi) // 2
        valid = m & (mid < n_req) & (lo < hi)
        cuv = tl.load(
            chunk_cu_seq_lens + mid * cu_s0,
            mask=valid,
            other=big,
        ).to(tl.int64)
        le = cuv <= offs
        lo = tl.where(valid & le, mid + 1, lo)
        hi = tl.where(valid & (~le), mid, hi)
    j = lo - 1
    has = lo > 0
    cu_start = tl.load(
        chunk_cu_seq_lens + j * cu_s0, mask=m & has, other=0
    ).to(tl.int64)
    nv = tl.load(
        chunk_seq_lens + j * len_s0, mask=m & has, other=0
    ).to(tl.int64)
    covered = has & (offs >= cu_start) & (offs < cu_start + nv)
    value = tl.load(chunk_kv_indices + offs, mask=m & (~covered), other=0)
    tl.store(out + offs, value, mask=m & (~covered))


def create_chunked_prefix_cache_kv_indices(
    req_to_token,
    req_pool_indices,
    chunk_start_idx,
    chunk_seq_lens,
    chunk_cu_seq_lens,
    chunk_kv_indices,
):
    n_req = req_pool_indices.shape[0]
    total = chunk_kv_indices.numel()
    out = torch.empty_like(chunk_kv_indices)
    if n_req and total:
        _create_chunked_prefix_cache_kv_indices_kernel[(n_req,)](
            req_to_token,
            req_pool_indices,
            chunk_start_idx,
            chunk_seq_lens,
            chunk_cu_seq_lens,
            out,
            req_pool_indices.stride(0),
            chunk_start_idx.stride(0) if chunk_start_idx.dim() else 1,
            chunk_seq_lens.stride(0) if chunk_seq_lens.dim() else 1,
            chunk_cu_seq_lens.stride(0) if chunk_cu_seq_lens.dim() else 1,
            req_to_token.stride(0),
            req_to_token.stride(1),
            BLOCK_C=1024,
            num_warps=4,
        )
        steps = max(1, (n_req + 1).bit_length())
        _fill_non_window_kernel[(triton.cdiv(total, 4096),)](
            chunk_cu_seq_lens,
            chunk_seq_lens,
            chunk_kv_indices,
            out,
            total,
            n_req,
            chunk_cu_seq_lens.stride(0) if chunk_cu_seq_lens.dim() else 1,
            chunk_seq_lens.stride(0) if chunk_seq_lens.dim() else 1,
            BLOCK=4096,
            STEPS=steps,
            num_warps=4,
        )
    elif total:
        out.copy_(chunk_kv_indices)
    return out


__all__ = ["create_chunked_prefix_cache_kv_indices"]
