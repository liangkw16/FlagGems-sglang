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

import torch
import triton
import triton.language as tl

_BLOCK_V = 1024
_BLOCK_C = 1024
_BLOCK_FLAT = 1024
_MAX_GRID = 65535


@triton.jit
def _draft_topk1_scan_kernel(
    logits_ptr,
    out_index_ptr,
    n_rows,
    vocab_size,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    best_val = tl.full((1,), -float("inf"), tl.float32)
    best_idx = tl.zeros((1,), dtype=tl.int32)
    for v_start in tl.range(0, vocab_size, BLOCK_V):
        offsets = v_start + tl.arange(0, BLOCK_V)
        mask = offsets < vocab_size
        values = tl.load(
            logits_ptr + row * vocab_size + offsets,
            mask=mask,
            other=-float("inf"),
        )
        indices = offsets.to(tl.int32)
        v_pair = tl.reshape(values, (512, 2))
        i_pair = tl.reshape(indices, (512, 2))
        va, vb = tl.split(v_pair)
        ia, ib = tl.split(i_pair)
        take = va >= vb
        values = tl.where(take, va, vb)
        indices = tl.where(take, ia, ib)
        v_pair = tl.reshape(values, (256, 2))
        i_pair = tl.reshape(indices, (256, 2))
        va, vb = tl.split(v_pair)
        ia, ib = tl.split(i_pair)
        take = va >= vb
        values = tl.where(take, va, vb)
        indices = tl.where(take, ia, ib)
        v_pair = tl.reshape(values, (128, 2))
        i_pair = tl.reshape(indices, (128, 2))
        va, vb = tl.split(v_pair)
        ia, ib = tl.split(i_pair)
        take = va >= vb
        values = tl.where(take, va, vb)
        indices = tl.where(take, ia, ib)
        v_pair = tl.reshape(values, (64, 2))
        i_pair = tl.reshape(indices, (64, 2))
        va, vb = tl.split(v_pair)
        ia, ib = tl.split(i_pair)
        take = va >= vb
        values = tl.where(take, va, vb)
        indices = tl.where(take, ia, ib)
        v_pair = tl.reshape(values, (32, 2))
        i_pair = tl.reshape(indices, (32, 2))
        va, vb = tl.split(v_pair)
        ia, ib = tl.split(i_pair)
        take = va >= vb
        values = tl.where(take, va, vb)
        indices = tl.where(take, ia, ib)
        v_pair = tl.reshape(values, (16, 2))
        i_pair = tl.reshape(indices, (16, 2))
        va, vb = tl.split(v_pair)
        ia, ib = tl.split(i_pair)
        take = va >= vb
        values = tl.where(take, va, vb)
        indices = tl.where(take, ia, ib)
        v_pair = tl.reshape(values, (8, 2))
        i_pair = tl.reshape(indices, (8, 2))
        va, vb = tl.split(v_pair)
        ia, ib = tl.split(i_pair)
        take = va >= vb
        values = tl.where(take, va, vb)
        indices = tl.where(take, ia, ib)
        v_pair = tl.reshape(values, (4, 2))
        i_pair = tl.reshape(indices, (4, 2))
        va, vb = tl.split(v_pair)
        ia, ib = tl.split(i_pair)
        take = va >= vb
        values = tl.where(take, va, vb)
        indices = tl.where(take, ia, ib)
        v_pair = tl.reshape(values, (2, 2))
        i_pair = tl.reshape(indices, (2, 2))
        va, vb = tl.split(v_pair)
        ia, ib = tl.split(i_pair)
        take = va >= vb
        values = tl.where(take, va, vb)
        indices = tl.where(take, ia, ib)
        v_pair = tl.reshape(values, (1, 2))
        i_pair = tl.reshape(indices, (1, 2))
        va, vb = tl.split(v_pair)
        ia, ib = tl.split(i_pair)
        take = va >= vb
        values = tl.where(take, va, vb)
        indices = tl.where(take, ia, ib)
        better = values > best_val
        best_val = tl.where(better, values, best_val)
        best_idx = tl.where(better, indices, best_idx)
    lane = tl.arange(0, 1)
    tl.store(out_index_ptr + row + lane, best_idx.to(tl.int64))


@triton.jit
def _draft_topk1_meta_kernel(
    positions_ptr,
    out_positions_ptr,
    ones_ptr,
    n_rows,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    for start in range(pid * BLOCK, n_rows, step):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < n_rows
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0)
        tl.store(out_positions_ptr + offsets, pos + 1, mask=mask)
        tl.store(ones_ptr + offsets, 1.0, mask=mask)


@triton.jit
def _draft_topk1_draft_kernel(
    src_ptr,
    dst_ptr,
    topk_index_ptr,
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
        topk = tl.load(topk_index_ptr + row, mask=mask, other=0)
        value = tl.where(col == column, topk, value)
        tl.store(
            dst_ptr + offsets, value.to(dst_ptr.dtype.element_ty), mask=mask
        )


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
    if n_rows > 0:
        assert n_rows <= _MAX_GRID
        _draft_topk1_scan_kernel[(n_rows,)](
            logits,
            topk_index,
            n_rows,
            vocab_size,
            BLOCK_V=_BLOCK_V,
        )
        grid_flat = (min(triton.cdiv(n_rows, _BLOCK_FLAT), _MAX_GRID),)
        _draft_topk1_meta_kernel[grid_flat](
            positions,
            out_positions,
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
            _draft_topk1_draft_kernel[grid](
                draft,
                out_draft_tokens,
                topk_index,
                n_elements,
                draft.shape[-1],
                int(draft_token_column),
                BLOCK=_BLOCK_FLAT,
            )
    return topk_p, topk_index, out_positions, out_draft_tokens


__all__ = ["draft_topk1"]
