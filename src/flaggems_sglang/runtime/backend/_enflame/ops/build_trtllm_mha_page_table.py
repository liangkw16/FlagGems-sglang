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

# Enflame vendor: every case of submission 12896 died in make_gcuir
# ("Pipeline run failed: PassManager execution failed") before numeric
# comparison. Differential TTIR analysis against the batch-5 kernels that
# passed Enflame (create_flashinfer_kv_indices, deepep_permute) leaves
# four suspects this kernel was alone in using: vector loads of int64
# elements (the proven killer of clamp_position's int64 case -- the same
# backend loads i64 scalars happily), the ~ complement (arith.xori),
# tl.where over two loads (arith.select on ints), and the int64->int32
# value cast (arith.trunci). This vendor removes all four:
#   * an int64 req_to_token is viewed to little-endian int32 words in the
#     wrapper (zero-copy reinterpret; gathered slot values are token
#     indices below the tensor's own numel bound, and the reference's
#     .to(int32) makes int32 the semantic result width either way), so
#     every data load/store stays int32;
#   * the select becomes two masked stores (compute / copy-old);
#   * the else-mask is spelled `page >= n_pages` instead of `~active`.
# What remains is only the Enflame-proven set: scalar loads with .to(
# tl.int64) (kv_indices), scalar int64 ceil-div (kv_indices cdiv),
# vector arithmetic shift `slot >> SHIFT` and `&` masks
# (apply_token_bitmask 2.85x on GCU), vector int32 comparisons, and
# int64 address offsets. Grid capped at 24 per the 24-SIP vendor guidance;
# num_warps left to the backend default.

import torch
import triton
import triton.language as tl

_MAX_GRID = 24
_BLOCK = 256


@triton.jit
def _build_page_table(
    pool,
    requests,
    lengths,
    old,
    out,
    tasks,
    tiles,
    columns,
    ps0,
    ps1,
    rs,
    ls,
    os0,
    os1,
    WORD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for task in range(tl.program_id(0), tasks, tl.num_programs(0)):
        row = (task // tiles).to(tl.int64)
        page = (task % tiles) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        length = tl.load(lengths + row * ls).to(tl.int64)
        n_pages = (length + PAGE_SIZE - 1) // PAGE_SIZE
        request = tl.load(requests + row * rs).to(tl.int64)
        valid = page < columns
        # n_pages can exceed columns (cache_seqlens beyond the table width),
        # so every load/store mask must be bounded by valid exactly like
        # the generic's single masked store.
        active = valid & (page < n_pages)
        inactive = valid & (page >= n_pages)
        # WORD==2 addresses the low word of each int64 slot (little-endian);
        # WORD==1 is a native int32 pool. Positive divisors of 4096 are
        # powers of two, so the signed shift is floor division for the
        # negative sentinel values the tests feed on purpose.
        slot = tl.load(
            pool + (request * ps0 + page * PAGE_SIZE * ps1) * WORD,
            active,
            other=0,
        )
        previous = tl.load(old + row * os0 + page * os1, inactive, other=0)
        tl.store(out + row * columns + page, slot >> SHIFT, active)
        tl.store(out + row * columns + page, previous, inactive)


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
    # Strides passed to the kernel are always in pool ELEMENT units of the
    # original dtype; WORD converts them to int32-word units inside the
    # kernel, matching how torch lays out the narrowed view.
    pool_elements = req_to_token
    if req_to_token.dtype == torch.int64 and pool_elements.stride(-1) != 1:
        pool_elements = pool_elements.contiguous()
    pool = (
        pool_elements.view(torch.int32)
        if req_to_token.dtype == torch.int64
        else pool_elements
    )
    word = 2 if req_to_token.dtype == torch.int64 else 1
    ps0, ps1 = pool_elements.stride()
    out = torch.empty(
        page_table.shape, dtype=page_table.dtype, device=page_table.device
    )
    n = out.numel()
    if n:
        tiles = triton.cdiv(out.shape[1], _BLOCK)
        tasks = out.shape[0] * tiles
        _build_page_table[(min(tasks, _MAX_GRID),)](
            pool,
            req_pool_indices,
            cache_seqlens,
            page_table,
            out,
            tasks,
            tiles,
            out.shape[1],
            ps0,
            ps1,
            req_pool_indices.stride(0),
            cache_seqlens.stride(0),
            page_table.stride(0),
            page_table.stride(1),
            WORD=word,
            PAGE_SIZE=page_size,
            SHIFT=page_size.bit_length() - 1,
            BLOCK=_BLOCK,
        )
    return out


__all__ = ["build_trtllm_mha_page_table"]
