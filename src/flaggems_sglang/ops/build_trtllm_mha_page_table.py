# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Address formula from SGLang 8014d9d: ops/kvcache/trtllm_mha_page_table.py.

import torch
import triton
import triton.language as tl


@triton.jit
def _build_page_table(
    pool,
    requests,
    lengths,
    old,
    out,
    total,
    columns,
    ps0,
    ps1,
    rs,
    ls,
    os0,
    os1,
    PAGE_SIZE: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for block in range(
        tl.program_id(0), tl.cdiv(total, BLOCK), tl.num_programs(0)
    ):
        i = block * BLOCK + tl.arange(0, BLOCK)
        row = (i // columns).to(tl.int64)
        page = (i % columns).to(tl.int64)
        valid = i < total
        length = tl.load(lengths + row * ls, valid, other=0).to(tl.int64)
        active = valid & (page < (length + PAGE_SIZE - 1) // PAGE_SIZE)
        request = tl.load(requests + row * rs, active, other=0).to(tl.int64)
        slot = tl.load(
            pool + request * ps0 + page * PAGE_SIZE * ps1, active, other=0
        )
        previous = tl.load(
            old + row * os0 + page * os1, valid & ~active, other=0
        )
        # Positive divisors of 4096 are powers of two; signed shift is floor division.
        value = tl.where(active, slot >> SHIFT, previous)
        tl.store(out + i, value, valid)


def build_trtllm_mha_page_table(
    req_to_token, req_pool_indices, cache_seqlens, page_table, page_size
):
    assert req_to_token.ndim == page_table.ndim == 2
    assert req_pool_indices.ndim == cache_seqlens.ndim == 1
    assert (
        req_pool_indices.numel()
        == cache_seqlens.numel()
        == page_table.shape[0]
    )
    assert page_table.dtype == torch.int32
    assert (
        isinstance(page_size, int) and page_size > 0 and 4096 % page_size == 0
    )
    for tensor in (req_to_token, req_pool_indices, cache_seqlens):
        assert tensor.dtype in (torch.int32, torch.int64)
    out = torch.empty(
        page_table.shape, dtype=page_table.dtype, device=page_table.device
    )
    n = out.numel()
    if n:
        _build_page_table[(min(triton.cdiv(n, 256), 65535),)](
            req_to_token,
            req_pool_indices,
            cache_seqlens,
            page_table,
            out,
            n,
            out.shape[1],
            *req_to_token.stride(),
            req_pool_indices.stride(0),
            cache_seqlens.stride(0),
            *page_table.stride(),
            PAGE_SIZE=page_size,
            SHIFT=page_size.bit_length() - 1,
            BLOCK=256,
        )
    return out


__all__ = ["build_trtllm_mha_page_table"]
