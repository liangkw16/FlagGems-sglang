# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Enflame vendor, e6: the recipe (cap-24 grid-stride) with BLOCK raised
# to 8192 - the width axis was monotonically positive on this chip
# (1024->4096 +71% here in e3) and 4096->8192 just paid +23% on the
# same-family T75 task. One isolated width probe, everything else
# byte-identical to the e4 carrier.

import torch
import triton
import triton.language as tl

_BLOCK_COL = 8192
_MAX_PROGS = 24
_MAX_GRID = 65535


@triton.jit
def _gelu_tanh_and_mul_kernel(
    x_ptr,
    output_ptr,
    rows,
    half_width,
    BLOCK_COL: tl.constexpr,
):
    pid = tl.program_id(0)
    num_col_blocks = tl.cdiv(half_width, BLOCK_COL)
    total_blocks = rows * num_col_blocks
    grid_size = tl.num_programs(0)
    for block_id in range(pid, total_blocks, grid_size):
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


@triton.jit
def _gelu_tanh_grouped4(
    x_ptr, output_ptr, rows, half_width, BLOCK: tl.constexpr
):
    cols = tl.arange(0, BLOCK)
    for group in range(tl.program_id(0), tl.cdiv(rows, 4), tl.num_programs(0)):
        row = group.to(tl.int64) * 4 + tl.arange(0, 4)
        mask = (row[:, None] < rows) & (cols[None, :] < half_width)
        row_base = row[:, None] * (2 * half_width)
        gate = tl.load(x_ptr + row_base + cols[None, :], mask, other=0.0).to(
            tl.float32
        )
        up = tl.load(
            x_ptr + row_base + half_width + cols[None, :], mask, other=0.0
        ).to(tl.float32)
        inner = 0.7978845608028654 * gate * (1.0 + 0.044715 * gate * gate)
        tanh_inner = 2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0
        gelu = 0.5 * gate * (1.0 + tanh_inner)
        tl.store(
            output_ptr + row[:, None] * half_width + cols[None, :],
            (gelu * up).to(output_ptr.dtype.element_ty),
            mask,
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
    if rows >= 4 and 0 < half_width <= 1024:
        _gelu_tanh_grouped4[(min(triton.cdiv(rows, 4), _MAX_PROGS),)](
            x,
            output,
            rows,
            half_width,
            BLOCK=triton.next_power_of_2(half_width),
            num_warps=4,
        )
    elif rows and half_width:
        _gelu_tanh_and_mul_kernel[
            (min(rows * triton.cdiv(half_width, _BLOCK_COL), _MAX_PROGS),)
        ](
            x,
            output,
            rows,
            half_width,
            BLOCK_COL=_BLOCK_COL,
            num_warps=4,
        )
    return output


__all__ = ["gelu_tanh_and_mul"]
