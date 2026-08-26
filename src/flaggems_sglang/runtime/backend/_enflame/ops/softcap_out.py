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

try:
    from triton.language.extra.gcu import libdevice as tl_extra_shim
except ImportError:
    from triton.language.extra import libdevice as tl_extra_shim


@triton.jit
def _softcap_out_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    softcap_const,
    BLOCK_SIZE: tl.constexpr,
    CAP_RECIPROCAL_OVERFLOWS: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    for block_idx in range(pid, num_blocks, grid_size):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scaled = x / softcap_const
        output = tl_extra_shim.tanh(scaled)
        output *= softcap_const
        if CAP_RECIPROCAL_OVERFLOWS:
            output = tl.where(x == 0.0, x / (x - x), output)
        tl.store(output_ptr + offsets, output, mask=mask)


def softcap_out(x, softcap_const):
    if isinstance(softcap_const, torch.Tensor):
        if softcap_const.numel() != 1:
            raise ValueError("softcap_const must contain one value")
        softcap_const = softcap_const.item()
    softcap_const = float(softcap_const)
    cap_reciprocal_overflows = (
        0.0 < abs(softcap_const) <= float.fromhex("0x1p-128")
    )
    x = x.contiguous()
    output = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    n_elements = x.numel()
    if n_elements == 0:
        return output
    grid = (min(triton.cdiv(n_elements, 32768), 12),)
    _softcap_out_kernel[grid](
        x,
        output,
        n_elements,
        softcap_const,
        BLOCK_SIZE=32768,
        CAP_RECIPROCAL_OVERFLOWS=cap_reciprocal_overflows,
    )
    return output


__all__ = ["softcap_out"]
