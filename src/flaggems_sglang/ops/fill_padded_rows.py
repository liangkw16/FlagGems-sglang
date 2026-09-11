# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 8014d9d: kernels/ops/moe/fill_padded_rows.py.

import torch
import triton
import triton.language as tl


@triton.jit
def _fill_padded_rows(
    out_ptr,
    num_non_padded_ptr,
    n_cols,
    fill_value,
    stride_row,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    n_valid = tl.load(num_non_padded_ptr).to(tl.int64)
    if row >= n_valid:
        cols = tl.arange(0, BLOCK_COLS).to(tl.int64)
        mask = cols < n_cols
        fill = tl.full(
            (BLOCK_COLS,), fill_value, dtype=out_ptr.dtype.element_ty
        )
        tl.store(out_ptr + row * stride_row + cols, fill, mask=mask)


def fill_padded_rows(x, num_token_non_padded, fill_value):
    assert x.ndim == 2 and x.stride(1) == 1
    assert num_token_non_padded.numel() == 1
    assert not num_token_non_padded.dtype.is_floating_point
    assert num_token_non_padded.device == x.device
    if isinstance(fill_value, torch.Tensor):
        fill_value = fill_value.item()
    out = x.clone()
    n_rows, n_cols = out.shape
    if n_rows and n_cols:
        _fill_padded_rows[(n_rows,)](
            out,
            num_token_non_padded,
            n_cols,
            fill_value,
            out.stride(0),
            BLOCK_COLS=triton.next_power_of_2(n_cols),
        )
    return out


__all__ = ["fill_padded_rows"]
