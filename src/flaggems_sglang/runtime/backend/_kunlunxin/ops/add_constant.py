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

"""Kunlunxin vendor for Task 93 add_constant.

Our generic 1024-lane flat map read 0.442x on kunlunxin; this batch's
platform data (T94 e1-e5 ladder, T96/T98 vendors) shows the XPU
backend wants wide flat vectors, and PR#56's platform-validated
kunlunxin int32 elementwise kernel runs BLOCK=16384. This vendor is
the same flat map at 16384 lanes. Not compiled on kunlunxin hardware
locally; the platform evaluator is the arbiter.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 16384


@triton.jit
def _add_constant_wide(
    src,
    out,
    numel,
    CONSTANT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < numel
    value = tl.load(src + offs, mask=m, other=0)
    tl.store(out + offs, value + CONSTANT, mask=m)


def add_constant(src, constant):
    assert src.dtype == torch.int32
    assert src.dim() == 1 and src.is_contiguous()
    numel = src.numel()
    out = torch.empty_like(src)
    if numel:
        _add_constant_wide[(triton.cdiv(numel, _BLOCK),)](
            src, out, numel, constant, BLOCK=_BLOCK, num_warps=8
        )
    return out


__all__ = ["add_constant"]
