# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Ascend vendor: persistent launch kept close to the physical Vector
# Core count (Ascend vector_operator.md: GPU-style huge grids pay
# repeated dispatch overhead on NPUs - decode_attention template).
# Kernel bytes identical to the generic (row/tile loops stride
# in-kernel on both axes); only the launch shape changes: the grid
# product stays at the vector-core count (axis1 capped at 8).

import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

_DEFAULT_VECTOR_CORES = 40


def _get_num_vector_cores(device_index):
    properties = driver.active.utils.get_device_properties(device_index)
    return int(properties.get("num_vectorcore", _DEFAULT_VECTOR_CORES))


def _worker_count(tensor):
    index = getattr(getattr(tensor, "device", None), "index", None)
    if index is None:
        return _DEFAULT_VECTOR_CORES
    return _get_num_vector_cores(index)


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
        for tile in range(tl.program_id(1), tl.cdiv(gap, BLOCK), tl.num_programs(1)):
            i = tile * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
            m = i < gap
            off = gap_lo + i
            value = tl.load(old + (off * m).to(tl.int64) * olds, m, other=0)
            tl.store(out + off * os, value, m)
        for tile in range(tl.program_id(1), tl.cdiv(length, BLOCK), tl.num_programs(1)):
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
    assert req_pool_indices.ndim == page_kernel_lens.ndim == kv_indptr.ndim == 1
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
    # E5 drops the wrapper clone (a full read+write of the output on top
    # of the kernel's own traffic). The kernel now restores the untouched
    # regions itself, which is free on gapless production shapes. The
    # per-chip leaderboard reopening evidence: the leader sits 1.5-1.7x
    # ahead on every bandwidth-bound chip while the proxy shapes that
    # closed this axis in round 2 were launch-bound.
    out = torch.empty_like(kv_indices)
    if not (batch and out.numel()):
        out.copy_(kv_indices)
        return out
    # Target ~512 cooperating programs (the upstream SGLang AMD
    # parallelization shape for long contexts): more token blocks per
    # row and idle blocks fall through their zero-iteration loops.
    # The 255 cap keeps grid.y under the Enflame hardware limit that
    # batch<=2 x wide-context shapes would otherwise cross.
    workers = _worker_count(out)
    splits = min(
        max(1, triton.cdiv(req_to_token.shape[1], 256)),
        max(1, 512 // batch),
        8,
    )
    grid_x = min(batch, max(1, workers // splits))
    _create_kv_indices[(grid_x, splits)](
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
        BLOCK=256,
    )
    return out


__all__ = ["create_flashinfer_kv_indices"]
