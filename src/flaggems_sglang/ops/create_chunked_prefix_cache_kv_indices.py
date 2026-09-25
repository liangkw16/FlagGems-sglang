# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 95 create_chunked_prefix_cache_kv_indices: chunked-prefill
# variant of the FlashInfer kv-indices builder. Request i copies its
# token-pool window req_to_token[pool_i, start_i : start_i+n_i] into
# out[cu_i : cu_i+n_i]; the untouched tail of the cloned destination is
# preserved. One program per request with a dynamic masked column run;
# every index computation is i64.

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


def create_chunked_prefix_cache_kv_indices(
    req_to_token,
    req_pool_indices,
    chunk_start_idx,
    chunk_seq_lens,
    chunk_cu_seq_lens,
    chunk_kv_indices,
):
    out = chunk_kv_indices.clone()
    n_req = req_pool_indices.shape[0]
    if n_req:
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
    return out


__all__ = ["create_chunked_prefix_cache_kv_indices"]
