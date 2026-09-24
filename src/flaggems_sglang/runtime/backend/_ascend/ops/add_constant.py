# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Ascend vendor for Task 93 add_constant.

Our generic flat map read 0.131x on the platform's Ascend evaluator
while the leaderboard leader holds 0.98x, and the platform-validated
PR#70 Ascend elementwise recipe (T29-CAND line) prescribes the
opposite launch shape: a persistent grid-stride pass with tile areas
capped around 1024 elements and num_warps=4, instead of one thin
program per 1024-element tile. This vendor follows that recipe: grid
is clamped to a small persistent count, each program strides over
consecutive tiles, CONSTANT stays a JIT fold. Not compiled on Ascend
hardware locally; the platform evaluator is the arbiter.
"""

import torch
import triton
import triton.language as tl

_TILE = 1024
_PERSISTENT = 512


@triton.jit
def _add_constant_persistent(
    src,
    out,
    numel,
    CONSTANT: tl.constexpr,
    TILE: tl.constexpr,
):
    for base in range(
        tl.program_id(0) * TILE, numel, tl.num_programs(0) * TILE
    ):
        offs = base + tl.arange(0, TILE)
        m = offs < numel
        value = tl.load(src + offs, mask=m, other=0)
        tl.store(out + offs, value + CONSTANT, mask=m)


def add_constant(src, constant):
    assert src.dtype == torch.int32
    assert src.dim() == 1 and src.is_contiguous()
    numel = src.numel()
    out = torch.empty_like(src)
    if numel:
        grid = (min(triton.cdiv(numel, _TILE), _PERSISTENT),)
        _add_constant_persistent[grid](
            src, out, numel, constant, TILE=_TILE, num_warps=4
        )
    return out


__all__ = ["add_constant"]
