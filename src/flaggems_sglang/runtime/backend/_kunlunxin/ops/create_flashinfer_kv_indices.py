# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 8014d9d kernels/ops/kvcache/kv_indices.py.

# Kunlunxin vendor: the e5 no-clone kernel in this chip's proven
# geometry (BLOCK=512, splits capped at 32); the clone-free generic gained
# the big GPUs +36-57% and kunlunxin reads 2.78 vs the leader's 4.16.

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
    old,
    out,
    batch,
    out_numel,
    ps0,
    ps1,
    rs,
    ls,
    ips,
    ss,
    os,
    olds,
    HAS_START: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for row in range(tl.program_id(0), batch, tl.num_programs(0)):
        row_offset = row.to(tl.int64)
        request = tl.load(requests + row_offset * rs).to(tl.int64)
        length = tl.load(lengths + row_offset * ls).to(tl.int64)
        begin = tl.load(indptr + row_offset * ips).to(tl.int64)
        next_begin = tl.load(indptr + (row_offset + 1) * ips).to(tl.int64)
        start = tl.full((), 0, tl.int64)
        if HAS_START:
            start = tl.load(starts + row_offset * ss).to(tl.int64)
        # The output is allocated empty, so this kernel also restores the
        # regions the reference leaves untouched: the head before row 0,
        # the inter-row gaps and the tail past the last row (one region
        # per neighbouring program, zero iterations on gapless shapes).
        # Scalar selects stay arithmetic - integer tl.where has no passing
        # GCU300 precedent - and the added loads clamp their masked-lane
        # addresses arithmetically for the Ascend 507035 discipline.
        first = (row == 0).to(tl.int64)
        last = (row_offset == batch - 1).to(tl.int64)
        head_len = begin * first
        gap_lo = begin + length
        gap_hi = next_begin + (out_numel - next_begin) * last
        for tile in range(
            tl.program_id(1), tl.cdiv(head_len, BLOCK), tl.num_programs(1)
        ):
            i = tile * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
            m = i < head_len
            value = tl.load(old + (i * m).to(tl.int64) * olds, m, other=0)
            tl.store(out + i * os, value, m)
        gap = gap_hi - gap_lo
        for tile in range(
            tl.program_id(1), tl.cdiv(gap, BLOCK), tl.num_programs(1)
        ):
            i = tile * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
            m = i < gap
            off = gap_lo + i
            value = tl.load(old + (off * m).to(tl.int64) * olds, m, other=0)
            tl.store(out + off * os, value, m)
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
    # The wrapper clone is dropped (see the vendor header): the kernel
    # restores the untouched regions itself, free on gapless shapes.
    out = torch.empty_like(kv_indices)
    if not (batch and out.numel()):
        out.copy_(kv_indices)
        return out
    # Target ~512 cooperating programs (the upstream SGLang AMD
    # parallelization shape for long contexts): more token blocks per
    # row and idle blocks fall through their zero-iteration loops.
    # The 255 cap keeps grid.y under the Enflame hardware limit that
    # batch<=2 x wide-context shapes would otherwise cross.
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
        kv_indices,
        out,
        batch,
        out.numel(),
        *req_to_token.stride(),
        req_pool_indices.stride(0),
        page_kernel_lens.stride(0),
        kv_indptr.stride(0),
        kv_start_idx.stride(0) if kv_start_idx is not None else 0,
        out.stride(0),
        kv_indices.stride(0),
        HAS_START=kv_start_idx is not None,
        BLOCK=512,
    )
    return out


__all__ = ["create_flashinfer_kv_indices"]
