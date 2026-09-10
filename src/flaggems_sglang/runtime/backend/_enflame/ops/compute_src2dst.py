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

# Enflame vendor: the generic kernel dies in make_gcuir ("Pipeline run
# failed: PassManager execution failed") on every case of submission
# 12900. The task contract fixes reorder_ids to int64 (argsort output),
# so the generic's only specialization vector-loads int64 elements -- the
# same construct that singles out the failing int64 case of clamp_position
# (12898) while every kernel that passed Enflame in this batch (kv_indices,
# deepep_permute) loads int64 scalars only. reorder_ids is a permutation of
# range(num_toks) and the contract bounds 0 <= num_toks <= 2**31, so the
# little-endian low int32 word of each element is exactly its value: the
# wrapper views the tensor to int32 words (a zero-copy reinterpret with a
# proof of losslessness from the contract, not data narrowing) and the
# kernel scatters in pure int32. Every kernel construct is Enflame-proven:
# the load address form matches clamp_position's passing int32 cases, the
# store index is a loaded value times a runtime stride (decode_attention
# gathers through loaded int32 pages the same way), and int64 survives
# only in address offsets. Grid capped at 24 per the 24-SIP vendor
# guidance; num_warps left to the backend default.

import torch
import triton
import triton.language as tl

_MAX_GRID = 24
_BLOCK = 256


@triton.jit
def _compute_src2dst(ids, out, n, stride, os, BLOCK: tl.constexpr):
    for block in range(
        tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)
    ):
        dst = block * BLOCK + tl.arange(0, BLOCK)
        src = tl.load(ids + dst.to(tl.int64) * stride, dst < n, other=0)
        tl.store(out + src.to(tl.int64) * os, dst, dst < n)


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
