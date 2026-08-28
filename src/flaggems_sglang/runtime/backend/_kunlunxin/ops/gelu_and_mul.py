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

_BLOCK_SIZE = 2048
_MAX_GRID = 65535


@triton.jit
def _gelu_and_mul_kernel(
    x_ptr,
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
        row = offsets // half_width
        col = offsets - row * half_width
        base = row * (2 * half_width) + col
        gate = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(x_ptr + base + half_width, mask=mask, other=0.0).to(
            tl.float32
        )
        scaled = gate * 0.7071067811865476
        abs_scaled = tl.abs(scaled)
        t = 1.0 / (1.0 + 0.3275911 * abs_scaled)
        poly = t * (
            0.254829592
            + t
            * (
                -0.284496736
                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))
            )
        )
        erf_abs = 1.0 - poly * tl.exp(-abs_scaled * abs_scaled)
        erf_scaled = tl.where(scaled < 0.0, -erf_abs, erf_abs)
        gelu = gate * 0.5 * (1.0 + erf_scaled)
        tl.store(
            output_ptr + offsets,
            (gelu * up).to(output_ptr.dtype.element_ty),
            mask=mask,
        )


def gelu_and_mul(hidden_states):
    x = hidden_states.contiguous()
    last_dim = x.shape[-1]
    half_width = last_dim // 2
    output = torch.empty(
        x.shape[:-1] + (half_width,), dtype=x.dtype, device=x.device
    )
    if last_dim == 0:
        return output
    rows = x.numel() // last_dim
    n_elements = rows * half_width
    if n_elements == 0:
        return output
    grid = (min(triton.cdiv(n_elements, _BLOCK_SIZE), _MAX_GRID),)
    _gelu_and_mul_kernel[grid](
        x,
        output,
        n_elements,
        half_width,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return output


__all__ = ["gelu_and_mul"]
