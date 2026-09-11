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

# Enflame vendor, round 6. Five platform rounds established that on this
# GCU stack a per-lane 1D scatter through a loaded index is unusable:
# extending the loaded index to int64 fails make_gcuir (rounds 1/3), and
# both the raw i32 addptr (round 2) and the i32-times-runtime-stride
# form (round 5, re-rolled as e4r after an evaluator crash) compile but
# scatter to wrong addresses. FlagGems' own enflame gcu300 index_put
# scatters successfully with a different shape: a 2D row-segment store
# where each loaded index bases one contiguous run of elements
# (`cur_index * stride` in pure int32, no extsi, the rank-collapsed case
# using an arange(0, 1) column). Round 6 adopts exactly that shape: each
# loaded source id bases a one-element segment, the store stays a 2D
# [BLOCK, 1] tile with i32 index arithmetic throughout, and the load
# keeps the proven int64 range math on the little-endian lo-word view.
# reorder_ids is a permutation of range(num_toks) with
# 0 <= num_toks <= 2**31 by contract, so the lo word is exactly the
# value (a zero-copy reinterpret, losslessness proven from the contract)
# and every destination is written exactly once. Grid capped at 24 per
# the 24-SIP vendor guidance; num_warps left to the backend default.

import torch
import triton
import triton.language as tl

_MAX_GRID = 24
_BLOCK = 256


@triton.jit(do_not_specialize=["os"])
def _compute_src2dst(ids, out, n, stride, os, BLOCK: tl.constexpr):
    for block in range(
        tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)
    ):
        dst = block * BLOCK + tl.arange(0, BLOCK)
        src = tl.load(ids + dst.to(tl.int64) * stride, dst < n, other=0)
        # 2D row-segment scatter: one contiguous one-element segment per
        # loaded index, i32 index arithmetic only (the FlagGems enflame
        # index_put rank-collapsed shape).
        col = tl.arange(0, 1)
        tl.store(
            out + src[:, None] * os + col[None, :],
            dst[:, None] + col[None, :],
            (dst < n)[:, None] & (col < 1)[None, :],
        )


def compute_src2dst(reorder_ids, num_toks):
    assert reorder_ids.ndim == 1 and reorder_ids.numel() == num_toks
    assert reorder_ids.dtype in (torch.int32, torch.int64)
    assert 0 <= num_toks <= 2**31
    out = torch.empty(num_toks, dtype=torch.int32, device=reorder_ids.device)
    if reorder_ids.dtype == torch.int64:
        words = reorder_ids
        if words.stride(-1) != 1:
            words = words.contiguous()
        ids = words.view(torch.int32)[0::2]
    else:
        ids = reorder_ids
    if num_toks:
        _compute_src2dst[(min(triton.cdiv(num_toks, _BLOCK), _MAX_GRID),)](
            ids, out, num_toks, ids.stride(0), out.stride(0), BLOCK=_BLOCK
        )
    return out


__all__ = ["compute_src2dst"]
