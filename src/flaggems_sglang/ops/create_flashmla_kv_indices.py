# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 92d831d kernels/ops/kvcache/kv_indices.py
# (create_flashmla_kv_indices_triton).

import torch
import triton
import triton.language as tl


@triton.jit
def _create_flashmla_kv_indices(
    pool,
    requests,
    lengths,
    starts,
    old,
    out,
    batch,
    width,
    ps0,
    ps1,
    rs,
    ls,
    ss,
    os0,
    os1,
    olds0,
    olds1,
    HAS_START: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # One entry per page instead of per token: gather the token slot at
    # every page boundary and fold it to a page id. The output is
    # allocated empty, so each row's program also restores the untouched
    # page columns past its valid prefix (the reference clones the base
    # tensor and only overwrites out[i, :num_pages]). i64 stays in the
    # addressing arithmetic only; scalar selects stay arithmetic (no
    # bool->i64 casts, no integer tl.where) and masked lanes use plain
    # masked offsets - the GCU-proven forms from the T63 family.
    for row in range(tl.program_id(0), batch, tl.num_programs(0)):
        row64 = row.to(tl.int64)
        request = tl.load(requests + row64 * rs).to(tl.int64)
        length = tl.load(lengths + row64 * ls).to(tl.int64)
        start = tl.full((), 0, tl.int64)
        if HAS_START:
            start = tl.load(starts + row64 * ss).to(tl.int64)
        num_pages = (length + PAGE_SIZE - 1) // PAGE_SIZE
        for tile in range(
            tl.program_id(1), tl.cdiv(num_pages, BLOCK_P), tl.num_programs(1)
        ):
            p = tile * BLOCK_P + tl.arange(0, BLOCK_P).to(tl.int64)
            m = p < num_pages
            slot = tl.load(
                pool + request * ps0 + (start + p * PAGE_SIZE) * ps1,
            m,
            other=0,
            )
            tl.store(out + row64 * os0 + p * os1, slot // PAGE_SIZE, m)
        tail = width - num_pages
        for tile in range(
            tl.program_id(1), tl.cdiv(tail, BLOCK_P), tl.num_programs(1)
        ):
            j = tile * BLOCK_P + tl.arange(0, BLOCK_P).to(tl.int64)
            m = j < tail
            off = num_pages + j
            value = tl.load(old + row64 * olds0 + off * olds1, m, other=0)
            tl.store(out + row64 * os0 + off * os1, value, m)


def create_flashmla_kv_indices(
    req_to_token,
    req_pool_indices,
    page_kernel_lens,
    kv_start_idx,
    kv_indices,
    page_size,
):
    assert req_to_token.ndim == 2 and kv_indices.ndim == 2
    batch = req_pool_indices.numel()
    width = kv_indices.shape[1]
    assert req_pool_indices.ndim == page_kernel_lens.ndim == 1
    assert page_kernel_lens.numel() == batch
    for tensor in (
        req_to_token,
        req_pool_indices,
        page_kernel_lens,
        kv_indices,
    ):
        assert tensor.dtype in (torch.int32, torch.int64)
    if kv_start_idx is not None:
        assert kv_start_idx.ndim == 1 and kv_start_idx.numel() == batch
        assert kv_start_idx.dtype in (torch.int32, torch.int64)
    assert isinstance(page_size, int) and page_size >= 1
    # slot ids are token-table entries (non-negative in the harness's
    # domain); Triton integer division truncates toward zero, which
    # equals the reference's floor division there and differs only for
    # negative slots, which the production table never contains.
    out = torch.empty_like(kv_indices)
    if not (batch and width and out.numel()):
        out.copy_(kv_indices)
        return out
    # Row-parallel grid with per-row page-tile splits; grid.y keeps the
    # Enflame 255 cap and grid.x a conservative bound (rows grid-stride
    # beyond it), so the single generic launch stays inside every
    # supported chip's launch envelope.
    splits = min(
        max(1, triton.cdiv(width, 256)),
        max(1, 512 // batch),
        255,
    )
    _create_flashmla_kv_indices[(min(batch, 2048), splits)](
        req_to_token,
        req_pool_indices,
        page_kernel_lens,
        kv_start_idx,
        kv_indices,
        out,
        batch,
        width,
        *req_to_token.stride(),
        req_pool_indices.stride(0),
        page_kernel_lens.stride(0),
        kv_start_idx.stride(0) if kv_start_idx is not None else 0,
        *out.stride(),
        *kv_indices.stride(),
        HAS_START=kv_start_idx is not None,
        PAGE_SIZE=page_size,
        BLOCK_P=256,
    )
    return out


__all__ = ["create_flashmla_kv_indices"]
