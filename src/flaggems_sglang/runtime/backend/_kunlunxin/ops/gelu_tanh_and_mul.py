# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Kunlunxin vendor, e4: keeps this op's proven BLOCK_COL=1024 (4096 once
# crashed the reading to 0.25) and drops the runtime grid-stride loop -
# the exact single change that took T75's kunlun reading from 0.263 to
# 0.899 on the platform. One program per (row, column tile), launched
# as a flat grid; the math, fp32 compute and the output cast stay
# byte-identical to the generic.

import torch
import triton
import triton.language as tl

_BLOCK_COL = 1024


@triton.jit
def _gelu_tanh_and_mul_kernel(
    x_ptr,
    output_ptr,
    rows,
    half_width,
    BLOCK_COL: tl.constexpr,
):
    block_id = tl.program_id(0)
    num_col_blocks = tl.cdiv(half_width, BLOCK_COL)
    row_id = block_id // num_col_blocks
    col_block = block_id - row_id * num_col_blocks
    col_offsets = col_block * BLOCK_COL + tl.arange(0, BLOCK_COL)
    col_mask = col_offsets < half_width
    row_base = row_id.to(tl.int64) * (2 * half_width)
    gate = tl.load(
        x_ptr + row_base + col_offsets,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        x_ptr + row_base + half_width + col_offsets,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)
    # tanh via the exp identity (tl.exp is the only transcendental
    # verified on all eight chips; tl.math/libdevice tanh is not).
    # Saturates safely: inner -> +inf gives tanh -> 1, -inf -> -1.
    inner = 0.7978845608028654 * gate * (1.0 + 0.044715 * gate * gate)
    tanh_inner = 2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0
    gelu = 0.5 * gate * (1.0 + tanh_inner)
    tl.store(
        output_ptr + row_id.to(tl.int64) * half_width + col_offsets,
        (gelu * up).to(output_ptr.dtype.element_ty),
        mask=col_mask,
    )


def gelu_tanh_and_mul(input):
    x = input.contiguous()
    last_dim = x.shape[-1]
    assert last_dim % 2 == 0
    half_width = last_dim // 2
    output = torch.empty(
        x.shape[:-1] + (half_width,), dtype=x.dtype, device=x.device
    )
    rows = output.numel() // half_width if half_width else 0
    if rows and half_width:
        num_col_blocks = triton.cdiv(half_width, _BLOCK_COL)
        _gelu_tanh_and_mul_kernel[(rows * num_col_blocks,)](
            x,
            output,
            rows,
            half_width,
            BLOCK_COL=_BLOCK_COL,
        )
    return output


__all__ = ["gelu_tanh_and_mul"]
