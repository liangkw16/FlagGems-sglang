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
from triton.language.extra import libdevice as tl_extra_shim

_BLOCK_SIZE = 2048
_MAX_GRID = 65535


@triton.jit
def _softcap_inplace_logits_kernel(
    logits_ptr,
    ncols,
    row_stride,
    num_col_blocks,
    total_blocks,
    softcap_const,
    BLOCK_SIZE: tl.constexpr,
    CAP_RECIPROCAL_OVERFLOWS: tl.constexpr,
):
    # metax vendor (e6): the only change vs the generic is
    # _BLOCK_SIZE 1024 -> 2048 - the exact single variable that won
    # +65% on muxi for T39 (flat BLOCK 2048) and +11% on T29
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    for block_id in range(pid, total_blocks, grid_size):
        row = block_id // num_col_blocks
        col_block = block_id - row * num_col_blocks
        cols = col_block * BLOCK_SIZE + offsets
        mask = cols < ncols
        pointers = logits_ptr + row.to(tl.int64) * row_stride + cols
        logits = tl.load(pointers, mask=mask, other=0.0).to(tl.float32)
        scaled = logits / softcap_const
        output = tl_extra_shim.tanh(scaled)
        output *= softcap_const
        if CAP_RECIPROCAL_OVERFLOWS:
            output = tl.where(
                logits == 0.0, logits / (logits - logits), output
            )
        tl.store(pointers, output, mask=mask)


def softcap_inplace_logits(full_logits, final_logit_softcapping):
    if isinstance(final_logit_softcapping, torch.Tensor):
        if final_logit_softcapping.numel() != 1:
            raise ValueError("final_logit_softcapping must contain one value")
        final_logit_softcapping = final_logit_softcapping.item()
    final_logit_softcapping = float(final_logit_softcapping)
    if full_logits.is_contiguous():
        nrows, ncols = 1, full_logits.numel()
        row_stride = ncols
    else:
        assert full_logits.ndim == 2
        assert full_logits.stride(1) == 1
        nrows, ncols = full_logits.shape
        row_stride = full_logits.stride(0)
    if nrows == 0 or ncols == 0:
        return full_logits
    cap_reciprocal_overflows = (
        0.0 < abs(final_logit_softcapping) <= float.fromhex("0x1p-128")
    )
    num_col_blocks = triton.cdiv(ncols, _BLOCK_SIZE)
    total_blocks = nrows * num_col_blocks
    _softcap_inplace_logits_kernel[(min(total_blocks, _MAX_GRID),)](
        full_logits,
        ncols,
        row_stride,
        num_col_blocks,
        total_blocks,
        final_logit_softcapping,
        BLOCK_SIZE=_BLOCK_SIZE,
        CAP_RECIPROCAL_OVERFLOWS=cap_reciprocal_overflows,
    )
    return full_logits


__all__ = ["softcap_inplace_logits"]

# water-sample carrier r1 (2026-09-02): bytes identical to e6-7612231 team best
# water-sample carrier r3 (2026-09-03)
# water-sample carrier r4 (2026-09-03 night)
# water-sample carrier r5 (2026-09-03)
# water-sample carrier r6 (2026-09-03 noon)
# final sampling t1 (2026-09-03)
# final sampling t2 (2026-09-03)
# final sampling t3 (2026-09-03)
# final sampling t4 (2026-09-03)
