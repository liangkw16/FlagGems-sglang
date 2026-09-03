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
        scaled_sq = scaled * scaled
        near_zero = scaled * (
            1.0 + scaled_sq * (-1.0 / 3.0 + scaled_sq * (2.0 / 15.0))
        )
        saturated = 2.0 / (1.0 + tl.exp(-2.0 * scaled)) - 1.0
        output = softcap_const * tl.where(
            tl.abs(scaled) < 0.25, near_zero, saturated
        )
        if CAP_RECIPROCAL_OVERFLOWS:
            output = tl.where(
                logits == 0.0, logits / (logits - logits), output
            )
        tl.store(pointers, output, mask=mask)


@triton.jit
def _softcap_rows_npu_kernel(
    logits_ptr,
    ncols,
    n_rows,
    softcap_const,
    CAP_RECIPROCAL_OVERFLOWS: tl.constexpr,
    SUB: tl.constexpr,
):
    row = tl.program_id(0)
    base = row.to(tl.int64) * ncols
    nsub = tl.cdiv(ncols, SUB)
    for sub in range(0, nsub):
        offs = sub * SUB + tl.arange(0, SUB)
        last = sub == nsub - 1
        if last:
            mask = offs < ncols
            pointers = logits_ptr + base + offs
            logits = tl.load(pointers, mask=mask, other=0.0).to(tl.float32)
            scaled = logits / softcap_const
            output = softcap_const * (
                2.0 / (1.0 + tl.exp(-2.0 * scaled)) - 1.0
            )
            if CAP_RECIPROCAL_OVERFLOWS:
                output = tl.where(
                    logits == 0.0, logits / (logits - logits), output
                )
            tl.store(pointers, output, mask=mask)
        else:
            pointers = logits_ptr + base + offs
            logits = tl.load(pointers).to(tl.float32)
            scaled = logits / softcap_const
            output = softcap_const * (
                2.0 / (1.0 + tl.exp(-2.0 * scaled)) - 1.0
            )
            if CAP_RECIPROCAL_OVERFLOWS:
                output = tl.where(
                    logits == 0.0, logits / (logits - logits), output
                )
            tl.store(pointers, output)


def softcap_inplace_logits(full_logits, final_logit_softcapping):
    if isinstance(final_logit_softcapping, torch.Tensor):
        if final_logit_softcapping.numel() != 1:
            raise ValueError("final_logit_softcapping must contain one value")
        final_logit_softcapping = final_logit_softcapping.item()
    final_logit_softcapping = float(final_logit_softcapping)
    if full_logits.is_contiguous() and full_logits.ndim == 2:
        nrows, ncols = full_logits.shape
        if nrows == 0 or ncols == 0:
            return full_logits
        cap_ro = (
            0.0 < abs(final_logit_softcapping) <= float.fromhex("0x1p-128")
        )
        sub = 4096 if ncols > 4096 else triton.next_power_of_2(ncols)
        _softcap_rows_npu_kernel[(min(nrows, 65535),)](
            full_logits,
            ncols,
            nrows,
            final_logit_softcapping,
            CAP_RECIPROCAL_OVERFLOWS=cap_ro,
            SUB=sub,
        )
        return full_logits
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
    num_col_blocks = triton.cdiv(ncols, 512)
    total_blocks = nrows * num_col_blocks
    _softcap_inplace_logits_kernel[(min(total_blocks, 48),)](
        full_logits,
        ncols,
        row_stride,
        num_col_blocks,
        total_blocks,
        final_logit_softcapping,
        BLOCK_SIZE=512,
        CAP_RECIPROCAL_OVERFLOWS=cap_reciprocal_overflows,
    )
    return full_logits


__all__ = ["softcap_inplace_logits"]


# e16: row-structured NPU kernel (hot path no CMP, SUB subtiling, per-row program)
