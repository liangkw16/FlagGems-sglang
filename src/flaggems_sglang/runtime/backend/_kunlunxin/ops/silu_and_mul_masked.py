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

_BLOCK_SIZE = 1024
_MAX_GRID = 65535


@triton.jit
def _silu_and_mul_masked_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    half_width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK_SIZE
    for start in range(pid * BLOCK_SIZE, n_elements, step):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        rows = offsets // half_width
        cols = offsets - rows * half_width
        input_base = rows * (2 * half_width) + cols
        gate = tl.load(input_ptr + input_base, mask=mask, other=0.0).to(
            tl.float32
        )
        up = tl.load(
            input_ptr + input_base + half_width,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        output = (gate / (1.0 + tl.exp(-gate))) * up
        tl.store(
            output_ptr + offsets,
            output.to(output_ptr.dtype.element_ty),
            mask=mask,
        )


def silu_and_mul_masked(input, masked_m):
    input = input.contiguous()
    experts, tokens, width = input.shape
    half_width = width // 2
    output = torch.empty(
        (experts, tokens, half_width),
        dtype=torch.bfloat16,
        device=input.device,
    )
    if experts == 0 or tokens == 0 or half_width == 0:
        return output

    n_elements = experts * tokens * half_width
    grid = (min(triton.cdiv(n_elements, _BLOCK_SIZE), _MAX_GRID),)
    _silu_and_mul_masked_kernel[grid](
        input,
        output,
        n_elements,
        half_width,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return output


__all__ = ["silu_and_mul_masked"]
