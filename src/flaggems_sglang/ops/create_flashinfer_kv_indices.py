# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 8014d9d: kernels/ops/kvcache/kv_indices.py.

import torch
import triton
import triton.language as tl


@triton.jit
def _create_kv_indices(
    pool,
    requests,
    lengths,
    indptr,
    starts,
    out,
    batch,
    ps0,
    ps1,
    rs,
    ls,
    ips,
    ss,
    os,
    HAS_START: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for row in range(tl.program_id(0), batch, tl.num_programs(0)):
        row_offset = row.to(tl.int64)
        request = tl.load(requests + row_offset * rs).to(tl.int64)
        length = tl.load(lengths + row_offset * ls).to(tl.int64)
        begin = tl.load(indptr + row_offset * ips).to(tl.int64)
        start = tl.full((), 0, tl.int64)
        if HAS_START:
            start = tl.load(starts + row_offset * ss).to(tl.int64)
        for tile in range(
            tl.program_id(1), tl.cdiv(length, BLOCK), tl.num_programs(1)
        ):
            i = tile * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
            value = tl.load(
                pool + request * ps0 + (start + i) * ps1, i < length, other=0
            )
            tl.store(out + (begin + i) * os, value, i < length)


def create_flashinfer_kv_indices(
    req_to_token,
    req_pool_indices,
    page_kernel_lens,
    kv_indptr,
    kv_start_idx,
    kv_indices,
):
    assert req_to_token.ndim == 2 and kv_indices.ndim == 1
    batch = req_pool_indices.numel()
    assert (
        req_pool_indices.ndim == page_kernel_lens.ndim == kv_indptr.ndim == 1
    )
    assert page_kernel_lens.numel() == batch and kv_indptr.numel() == batch + 1
    for tensor in (
        req_to_token,
        req_pool_indices,
        page_kernel_lens,
        kv_indptr,
        kv_indices,
    ):
        assert tensor.dtype in (torch.int32, torch.int64)
    if kv_start_idx is not None:
        assert kv_start_idx.ndim == 1 and kv_start_idx.numel() == batch
        assert kv_start_idx.dtype in (torch.int32, torch.int64)
    out = kv_indices.clone()
    if batch and out.numel():
        splits = min(
            max(1, triton.cdiv(req_to_token.shape[1], 512)),
            max(1, 128 // batch),
            32,
        )
        _create_kv_indices[(min(batch, 65535), splits)](
            req_to_token,
            req_pool_indices,
            page_kernel_lens,
            kv_indptr,
            kv_start_idx,
            out,
            batch,
            *req_to_token.stride(),
            req_pool_indices.stride(0),
            page_kernel_lens.stride(0),
            kv_indptr.stride(0),
            kv_start_idx.stride(0) if kv_start_idx is not None else 0,
            out.stride(0),
            HAS_START=kv_start_idx is not None,
            BLOCK=512,
        )
    return out


__all__ = ["create_flashinfer_kv_indices"]
