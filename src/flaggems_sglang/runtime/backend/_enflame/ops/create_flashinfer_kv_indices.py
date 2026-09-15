# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 8014d9d kernels/ops/kvcache/kv_indices.py.

# Enflame vendor: the e5 no-clone kernel in this chip's proven geometry
# (BLOCK=512, splits capped at 32) - the gap-copy arithmetic uses only
# GCU-proven primitives (arithmetic selects, clamped masked loads,
# linear stores) and the clone removal gained the big GPUs +36-57%;
# enflame sits 18.3 vs the leader's 39.

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
        # Scalar selects stay pure integer arithmetic (no bool->i64
        # casts, no integer tl.where, no i64 vector multiplies - the
        # clamp form e6 used is the T67-e1 GCU compile suspect), and the
        # gap loads use plain masked offsets, the proven GCU form.
        head_len = begin * (1 - tl.minimum(row, 1))
        gap_lo = begin + length
        last = 1 - tl.minimum(batch - 1 - row, 1)
        gap_hi = next_begin + (out_numel - next_begin) * last
        # E11 single-variable axis: scalar base pre-offset. Every vector
        # address below is now base_ptr + i * stride with the scalar part
        # folded into the pointer itself (the OffsetAnalysis-friendly
        # linear form from the _kunlunxin softmax precedent) instead of
        # (scalar + i) * stride inside the lane expression. BLOCK, splits
        # and the load/store primitives stay byte-equivalent to e8.
        for tile in range(
            tl.program_id(1), tl.cdiv(head_len, BLOCK), tl.num_programs(1)
        ):
            i = tile * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
            m = i < head_len
            value = tl.load(old + i * olds, m, other=0)
            tl.store(out + i * os, value, m)
        gap = gap_hi - gap_lo
        old_gap = old + gap_lo * olds
        out_gap = out + gap_lo * os
        for tile in range(
            tl.program_id(1), tl.cdiv(gap, BLOCK), tl.num_programs(1)
        ):
            i = tile * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
            m = i < gap
            value = tl.load(old_gap + i * olds, m, other=0)
            tl.store(out_gap + i * os, value, m)
        pool_src = pool + request * ps0 + start * ps1
        out_row = out + begin * os
        for tile in range(
            tl.program_id(1), tl.cdiv(length, BLOCK), tl.num_programs(1)
        ):
            i = tile * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
            value = tl.load(pool_src + i * ps1, i < length, other=0)
            tl.store(out_row + i * os, value, i < length)


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
