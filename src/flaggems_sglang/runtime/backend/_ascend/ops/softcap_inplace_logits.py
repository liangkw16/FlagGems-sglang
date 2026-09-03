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

import os

os.environ.setdefault("TRITON_ALL_BLOCKS_PARALLEL", "1")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

try:
    from flaggems_sglang.runtime import torch_device_fn  # noqa: E402
    from flaggems_sglang.utils.libentry import libentry  # noqa: E402

    _HAS_LIBENTRY = True
except Exception:
    torch_device_fn = None
    libentry = None
    _HAS_LIBENTRY = False

# e15: official Ascend idiom package (triton-ascend NPU performance
# guidelines as _ascend starting hypotheses): dynamic BLOCK 32768 with
# in-kernel 4096 sub-tiling (multi-buffer pipelining), tail branch so
# the hot path carries no per-element CMP (int32 vector CMP degrades to
# scalar on the NPU), care_padding=False loads, libentry launch inside
# the torch_device_fn device context when the repo package is importable.
_SUB = 4096
_MAX_GRID = 65535


def _libentry():
    return libentry() if _HAS_LIBENTRY else (lambda fn: fn)


@_libentry()
@triton.jit
def _softcap_inplace_logits_npu_kernel(
    logits_ptr,
    n_elements,
    softcap_const,
    BLOCK: tl.constexpr,
    SUB: tl.constexpr,
    CAP_RECIPROCAL_OVERFLOWS: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK
    last = pid == tl.num_programs(0) - 1
    if last:
        for sub in range(0, BLOCK, SUB):
            offs = sub + tl.arange(0, SUB)
            mask = (base + offs) < n_elements
            pointers = logits_ptr + base + offs
            logits = tl.load(pointers, mask=mask, care_padding=False).to(
                tl.float32
            )
            output = softcap_const * (
                2.0 / (1.0 + tl.exp(-2.0 * (logits / softcap_const))) - 1.0
            )
            if CAP_RECIPROCAL_OVERFLOWS:
                output = tl.where(
                    logits == 0.0, logits / (logits - logits), output
                )
            tl.store(pointers, output.to(pointers.dtype.element_ty), mask=mask)
    else:
        for sub in range(0, BLOCK, SUB):
            offs = sub + tl.arange(0, SUB)
            pointers = logits_ptr + base + offs
            logits = tl.load(pointers, care_padding=False).to(tl.float32)
            output = softcap_const * (
                2.0 / (1.0 + tl.exp(-2.0 * (logits / softcap_const))) - 1.0
            )
            if CAP_RECIPROCAL_OVERFLOWS:
                output = tl.where(
                    logits == 0.0, logits / (logits - logits), output
                )
            tl.store(pointers, output.to(pointers.dtype.element_ty))


@_libentry()
@triton.jit
def _softcap_inplace_logits_cuda_kernel(
    logits_ptr,
    n_elements,
    softcap_const,
    BLOCK: tl.constexpr,
    SUB: tl.constexpr,
    CAP_RECIPROCAL_OVERFLOWS: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK
    last = pid == tl.num_programs(0) - 1
    if last:
        for sub in range(0, BLOCK, SUB):
            offs = sub + tl.arange(0, SUB)
            mask = (base + offs) < n_elements
            pointers = logits_ptr + base + offs
            logits = tl.load(pointers, mask=mask, other=0.0).to(tl.float32)
            output = softcap_const * (
                2.0 / (1.0 + tl.exp(-2.0 * (logits / softcap_const))) - 1.0
            )
            if CAP_RECIPROCAL_OVERFLOWS:
                output = tl.where(
                    logits == 0.0, logits / (logits - logits), output
                )
            tl.store(pointers, output.to(pointers.dtype.element_ty), mask=mask)
    else:
        for sub in range(0, BLOCK, SUB):
            offs = sub + tl.arange(0, SUB)
            pointers = logits_ptr + base + offs
            logits = tl.load(pointers).to(tl.float32)
            output = softcap_const * (
                2.0 / (1.0 + tl.exp(-2.0 * (logits / softcap_const))) - 1.0
            )
            if CAP_RECIPROCAL_OVERFLOWS:
                output = tl.where(
                    logits == 0.0, logits / (logits - logits), output
                )
            tl.store(pointers, output.to(pointers.dtype.element_ty))


@triton.jit
def _softcap_inplace_logits_strided_kernel(
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
        output = softcap_const * (2.0 / (1.0 + tl.exp(-2.0 * scaled)) - 1.0)
        if CAP_RECIPROCAL_OVERFLOWS:
            output = tl.where(
                logits == 0.0, logits / (logits - logits), output
            )
        tl.store(pointers, output, mask=mask)


def softcap_inplace_logits(full_logits, final_logit_softcapping):
    if isinstance(final_logit_softcapping, torch.Tensor):
        if final_logit_softcapping.numel() != 1:
            raise ValueError("final_logit_softcapping must contain one value")
        final_logit_softcapping = float(final_logit_softcapping)
    final_logit_softcapping = float(final_logit_softcapping)
    cap_reciprocal_overflows = (
        0.0 < abs(final_logit_softcapping) <= float.fromhex("0x1p-128")
    )
    if full_logits.is_contiguous():
        n = full_logits.numel()
        if n == 0:
            return full_logits
        block = max(32768, triton.next_power_of_2(triton.cdiv(n, _MAX_GRID)))
        grid = triton.cdiv(n, block)
        is_npu = full_logits.device.type == "npu"
        kernel = (
            _softcap_inplace_logits_npu_kernel
            if is_npu
            else _softcap_inplace_logits_cuda_kernel
        )
        if _HAS_LIBENTRY:
            with torch_device_fn.device(full_logits.device):
                kernel[(grid,)](
                    full_logits,
                    n,
                    final_logit_softcapping,
                    BLOCK=block,
                    SUB=_SUB,
                )
        else:
            kernel[(grid,)](
                full_logits,
                n,
                final_logit_softcapping,
                BLOCK=block,
                SUB=_SUB,
            )
        return full_logits
    assert full_logits.ndim == 2
    assert full_logits.stride(1) == 1
    nrows, ncols = full_logits.shape
    if nrows == 0 or ncols == 0:
        return full_logits
    num_col_blocks = triton.cdiv(ncols, 512)
    total_blocks = nrows * num_col_blocks
    _softcap_inplace_logits_strided_kernel[(min(total_blocks, _MAX_GRID),)](
        full_logits,
        ncols,
        full_logits.stride(0),
        num_col_blocks,
        total_blocks,
        final_logit_softcapping,
        BLOCK_SIZE=512,
        CAP_RECIPROCAL_OVERFLOWS=cap_reciprocal_overflows,
    )
    return full_logits


__all__ = ["softcap_inplace_logits"]
