# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# e6c structure: int64-free meta/draft kept, argmax extraction swapped
# to the where+min form that the enflame-passed router generic uses
# (return_indices lowering was the last unproven op on this path after
# six structure failures)
# varied only the argmax kernels while the meta kernel (int64 load/add/store)
# and draft kernel (mixed int64/int32 where) stayed byte-identical and every
# attempt hit Pipeline run failed. This variant keeps all compute in
# int32/fp: int64 outputs are written through torch int32 views as lo/hi
# pairs, and the draft write is a pure copy plus an int32 column scatter
# (no int64 ops, no mixed-dtype where, no int64 conversions).

import torch
import triton
import triton.language as tl

_BLOCK_V = 1024
_BLOCK_C = 1024
_BLOCK_FLAT = 1024
_BLOCK_R = 128
_MAX_GRID = 65535


@triton.jit
def _draft_topk1_scan_kernel(
    logits_ptr,
    chunk_values_ptr,
    chunk_indices_ptr,
    n_rows,
    vocab_size,
    n_chunks,
    BLOCK_V: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n_tiles = n_rows * n_chunks
    n_programs = tl.num_programs(0)
    for tile in tl.range(tl.program_id(0), n_tiles, n_programs):
        row = tile // n_chunks
        chunk = tile - row * n_chunks
        offsets = chunk * BLOCK_V + tl.arange(0, BLOCK_V)
        mask = offsets < vocab_size
        values = tl.load(
            logits_ptr + row * vocab_size + offsets,
            mask=mask,
            other=-float("inf"),
        )
        chunk_max = tl.max(values, axis=0)
        chunk_idx = tl.min(
            tl.where(values == chunk_max, offsets, vocab_size), axis=0
        )
        tl.store(chunk_values_ptr + tile, chunk_max)
        tl.store(chunk_indices_ptr + tile, chunk_idx.to(tl.int32))


@triton.jit
def _draft_topk1_finalize_kernel(
    chunk_values_ptr,
    chunk_indices_ptr,
    out_index_i32_ptr,
    n_rows,
    n_chunks,
    BLOCK_C: tl.constexpr,
):
    n_programs = tl.num_programs(0)
    for row in tl.range(tl.program_id(0), n_rows, n_programs):
        base = row * n_chunks
        offsets = tl.arange(0, BLOCK_C)
        mask = offsets < n_chunks
        values = tl.load(
            chunk_values_ptr + base + offsets,
            mask=mask,
            other=-float("inf"),
        )
        best = tl.max(values, axis=0)
        chunk_sel = tl.min(tl.where(values == best, offsets, n_chunks), axis=0)
        idxs = tl.load(chunk_indices_ptr + base + offsets, mask=mask, other=0)
        top_index = tl.sum(tl.where(offsets == chunk_sel, idxs, 0), axis=0)
        pair = tl.arange(0, 2)
        lohi = tl.where(pair == 0, top_index, 0)
        tl.store(out_index_i32_ptr + row * 2 + pair, lohi)


@triton.jit
def _draft_topk1_meta_kernel(
    positions_i32_ptr,
    out_positions_i32_ptr,
    ones_ptr,
    n_rows,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    for start in range(pid * BLOCK, n_rows, step):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < n_rows
        lo = tl.load(positions_i32_ptr + 2 * offsets, mask=mask, other=0)
        hi = tl.load(positions_i32_ptr + 2 * offsets + 1, mask=mask, other=0)
        carry = (lo == -1).to(tl.int32)
        tl.store(out_positions_i32_ptr + 2 * offsets, lo + 1, mask=mask)
        tl.store(
            out_positions_i32_ptr + 2 * offsets + 1, hi + carry, mask=mask
        )
        tl.store(ones_ptr + offsets, 1.0, mask=mask)


@triton.jit
def _draft_topk1_copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    for start in range(pid * BLOCK, n_elements, step):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        value = tl.load(src_ptr + offsets, mask=mask, other=0)
        tl.store(dst_ptr + offsets, value, mask=mask)


@triton.jit
def _draft_topk1_scatter_kernel(
    out_i32_ptr,
    topk_index_i32_ptr,
    n_rows,
    draft_width,
    column,
    WIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    if WIDE == 1:
        row_stride = 2 * draft_width
        col0 = 2 * column
        width = 2
    else:
        row_stride = draft_width
        col0 = column
        width = 1
    for start in range(pid * BLOCK, n_rows, step):
        rows = start + tl.arange(0, BLOCK)
        mask = rows < n_rows
        pair = tl.arange(0, 2)
        idx = tl.load(topk_index_i32_ptr + 2 * rows, mask=mask, other=0)
        dst = out_i32_ptr + rows[:, None] * row_stride + col0 + pair[None, :]
        vals = tl.where(pair[None, :] == 0, idx[:, None], 0)
        tl.store(dst, vals, mask=mask[:, None] & (pair[None, :] < width))


@triton.jit
def _draft_topk1_patch_kernel_fallback(
    src_ptr,
    dst_ptr,
    topk_index_i32_ptr,
    n_elements,
    draft_width,
    column,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    for start in range(pid * BLOCK, n_elements, step):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        row = offsets // draft_width
        col = offsets - row * draft_width
        value = tl.load(src_ptr + offsets, mask=mask, other=0)
        topk = tl.load(topk_index_i32_ptr + 2 * row, mask=mask, other=0)
        topk = topk.to(dst_ptr.dtype.element_ty)
        value = tl.where(col == column, topk, value)
        tl.store(dst_ptr + offsets, value, mask=mask)


def draft_topk1(
    next_token_logits, positions, draft_tokens=None, draft_token_column=0
):
    logits = next_token_logits.contiguous()
    positions = positions.contiguous()
    n_rows = logits.shape[0]
    vocab_size = logits.shape[-1]

    topk_index = torch.empty(
        (n_rows, 1), dtype=torch.int64, device=logits.device
    )
    topk_p = torch.empty(
        (n_rows, 1), dtype=torch.float32, device=logits.device
    )
    out_positions = torch.empty_like(positions)
    topk_index_i32 = topk_index.view(torch.int32)
    positions_i32 = positions.view(torch.int32)
    out_positions_i32 = out_positions.view(torch.int32)
    if n_rows > 0:
        n_chunks = triton.cdiv(vocab_size, _BLOCK_V)
        chunk_values = torch.empty(
            (n_rows, n_chunks),
            dtype=torch.float32,
            device=logits.device,
        )
        chunk_indices = torch.empty(
            (n_rows, n_chunks), dtype=torch.int32, device=logits.device
        )
        n_tiles = n_rows * n_chunks
        _draft_topk1_scan_kernel[(min(n_tiles, _MAX_GRID),)](
            logits,
            chunk_values,
            chunk_indices,
            n_rows,
            vocab_size,
            n_chunks,
            BLOCK_V=_BLOCK_V,
            BLOCK_C=_BLOCK_C,
        )
        block_c = 1
        while block_c < n_chunks:
            block_c *= 2
        _draft_topk1_finalize_kernel[(min(n_rows, _MAX_GRID),)](
            chunk_values,
            chunk_indices,
            topk_index_i32,
            n_rows,
            n_chunks,
            BLOCK_C=max(block_c, 1),
        )
        grid_flat = (min(triton.cdiv(n_rows, _BLOCK_FLAT), _MAX_GRID),)
        _draft_topk1_meta_kernel[grid_flat](
            positions_i32,
            out_positions_i32,
            topk_p,
            n_rows,
            BLOCK=_BLOCK_FLAT,
        )

    out_draft_tokens = None
    if draft_tokens is not None:
        draft = draft_tokens.contiguous()
        out_draft_tokens = torch.empty_like(draft)
        n_elements = draft.numel()
        if n_elements > 0:
            grid = (min(triton.cdiv(n_elements, _BLOCK_FLAT), _MAX_GRID),)
            _draft_topk1_copy_kernel[grid](
                draft,
                out_draft_tokens,
                n_elements,
                BLOCK=_BLOCK_FLAT,
            )
            if draft.dtype in (torch.int32, torch.int64) and n_rows > 0:
                wide = 1 if draft.dtype == torch.int64 else 0
                grid_r = (min(triton.cdiv(n_rows, _BLOCK_R), _MAX_GRID),)
                _draft_topk1_scatter_kernel[grid_r](
                    out_draft_tokens.view(torch.int32),
                    topk_index_i32,
                    n_rows,
                    draft.shape[-1],
                    int(draft_token_column),
                    WIDE=wide,
                    BLOCK=_BLOCK_R,
                )
            elif draft.dtype not in (torch.int32, torch.int64):
                _draft_topk1_patch_kernel_fallback[grid](
                    draft,
                    out_draft_tokens,
                    topk_index_i32,
                    n_elements,
                    draft.shape[-1],
                    int(draft_token_column),
                    BLOCK=_BLOCK_FLAT,
                )
    return topk_p, topk_index, out_positions, out_draft_tokens


__all__ = ["draft_topk1"]
