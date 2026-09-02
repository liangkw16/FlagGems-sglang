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
def _interleaved_rope_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    dim,
    bound_height,
    bound_width,
    stream_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK_SIZE
    for start in range(pid * BLOCK_SIZE, n_elements, step):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        s = offsets // dim
        d = offsets - s * dim
        from_height = (d % 3 == 1) & (d < bound_height)
        from_width = (d % 3 == 2) & (d < bound_width)
        stream = tl.where(from_height, 1, tl.where(from_width, 2, 0))
        value = tl.load(
            x_ptr + stream * stream_stride + s * dim + d, mask=mask, other=0
        )
        tl.store(output_ptr + offsets, value, mask=mask)


def interleaved_rope(x, mrope_section):
    x = x.contiguous()
    seq_len = x.shape[1]
    dim = x.shape[2]
    output = torch.empty((seq_len, dim), dtype=x.dtype, device=x.device)
    n_elements = seq_len * dim
    if n_elements == 0:
        return output
    bound_height = int(mrope_section[1]) * 3
    bound_width = int(mrope_section[2]) * 3
    grid = (min(triton.cdiv(n_elements, _BLOCK_SIZE), _MAX_GRID),)
    _interleaved_rope_kernel[grid](
        x,
        output,
        n_elements,
        dim,
        bound_height,
        bound_width,
        n_elements,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return output


__all__ = ["interleaved_rope"]

# water-sample carrier r1 (2026-09-02): bytes identical to s0-99c154e team best
