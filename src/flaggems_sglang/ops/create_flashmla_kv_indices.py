# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 92d831d kernels/ops/kvcache/kv_indices.py
# (create_flashmla_kv_indices_triton).
#
# E4 reorganises the work: one program owns exactly one output page-
# block of one row (static grid axis 1 = cdiv(width, BLOCK_P)); every
# output element has a single writer - lanes below the row's page count
# gather and fold the token slot, lanes above copy the preserved tail
# from the input tensor. The dynamic tile loops of the earlier form
# (each program striding over scattered valid tiles plus a separate
# tail loop) disappear; this targets the Ascend control-flow overhead
# behind the unexplained 93.7 vs 182.6 huawei gap. i64 stays in
# addressing only; scalar selects stay arithmetic; masked lanes use
# plain masked offsets (GCU-proven forms from the T63 family).

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
    for row in range(tl.program_id(0), batch, tl.num_programs(0)):
        row64 = row.to(tl.int64)
        request = tl.load(requests + row64 * rs).to(tl.int64)
        length = tl.load(lengths + row64 * ls).to(tl.int64)
        start = tl.full((), 0, tl.int64)
        if HAS_START:
            start = tl.load(starts + row64 * ss).to(tl.int64)
        num_pages = (length + PAGE_SIZE - 1) // PAGE_SIZE
        p = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P).to(tl.int64)
        m_valid = p < num_pages
        slot = tl.load(
            pool + request * ps0 + (start + p * PAGE_SIZE) * ps1,
            m_valid,
            other=0,
        )
        tl.store(out + row64 * os0 + p * os1, slot // PAGE_SIZE, m_valid)
        m_tail = (p >= num_pages) & (p < width)
        value = tl.load(old + row64 * olds0 + p * olds1, m_tail, other=0)
        tl.store(out + row64 * os0 + p * os1, value, m_tail)


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
    # equals the reference's floor division there.
    out = torch.empty_like(kv_indices)
    if not (batch and width and out.numel()):
        out.copy_(kv_indices)
        return out
    # Static page-block grid: one tile per program (grid.y keeps the
    # Enflame 255 cap - wider tables fall back to the dynamic vendor
    # forms); rows grid-stride on axis 0.
    tiles = triton.cdiv(width, 256)
    # The static per-tile grid must cover the whole table; grid.y is
    # capped at 255 on Enflame-class devices.
    assert tiles <= 255
    _create_flashmla_kv_indices[(min(batch, 2048), tiles)](
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
