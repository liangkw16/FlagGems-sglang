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

# Ascend vendor: the generic passes Huawei at only 8.7-10.0x while the
# leader reads 24.8 -- the single chip carrying the whole remaining task
# gap. The hygon round proved the direct 2D (rows, page blocks) grid
# with no flattened task, no div/mod and no grid-stride loop on this
# platform (+22% on Hygon), and the upstream SGLang kernel uses exactly
# this shape, so this vendor mirrors the platform-proven hygon bytes
# with the Ascend-appropriate defaults: one program per (row, page
# block), scalar row metadata, one gather plus one masked store per
# lane, no num_warps pin. BLOCK stays 256 (a 1KB int32 tile, far below
# the 1572864-bit UB budget). E6 additionally clamps every memory op's
# address into the row width: e5's first touch died with the 507035
# aclnnInplaceCopy stream-sync timeout, the error family triton-ascend
# issues #16275/#1490 attribute to masked-lane out-of-bounds addresses
# being evaluated for real, and the e5 kernel addressed page_table and
# req_to_token with unclamped out-of-range lanes.

import torch
import triton
import triton.language as tl

_BLOCK = 256
_MAX_TILES = 65535


@triton.jit
def _build_page_table(
    pool,
    requests,
    lengths,
    old,
    out,
    columns,
    ps0,
    ps1,
    rs,
    ls,
    ps0o,
    ps1o,
    ps0l,
    ps1l,
    PAGE_SIZE: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    page = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    # Ascend evaluates masked-lane addresses for real (triton-ascend issues
    # #1490/#16275 both surface as 507035 MTE illegal GM address), so every
    # memory op addresses through a page clamped into the row width; the
    # out-of-range tail is never stored, the clamp only keeps its address
    # inside the buffers.
    page_rd = tl.where(page < columns, page, 0)
    length = tl.load(lengths + row * ls).to(tl.int64)
    n_pages = (length + PAGE_SIZE - 1) // PAGE_SIZE
    request = tl.load(requests + row * rs).to(tl.int64)
    # n_pages can exceed columns (cache_seqlens beyond the table width),
    # so the gather mask stays bounded by the row width.
    active = (page < columns) & (page < n_pages)
    inactive = (page < columns) & (page >= n_pages)
    # Positive divisors of 4096 are powers of two; signed shift is floor
    # division for the negative sentinel values the tests feed on purpose.
    slot = tl.load(
        pool + request * ps0 + page_rd * PAGE_SIZE * ps1, active, other=0
    )
    previous = tl.load(old + row * ps0l + page_rd * ps1l, inactive, other=0)
    tl.store(out + row * ps0o + page_rd * ps1o, slot >> SHIFT, active)
    tl.store(out + row * ps0o + page_rd * ps1o, previous, inactive)


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
        tiles = triton.cdiv(out.shape[1], _BLOCK)
        _build_page_table[(out.shape[0], min(tiles, _MAX_TILES))](
            req_to_token,
            req_pool_indices,
            cache_seqlens,
            page_table,
            out,
            out.shape[1],
            *req_to_token.stride(),
            req_pool_indices.stride(0),
            cache_seqlens.stride(0),
            *out.stride(),
            *page_table.stride(),
            PAGE_SIZE=page_size,
            SHIFT=page_size.bit_length() - 1,
            BLOCK=_BLOCK,
        )
    return out


__all__ = ["build_trtllm_mha_page_table"]
