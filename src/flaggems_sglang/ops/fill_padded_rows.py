# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 8014d9d: kernels/ops/moe/fill_padded_rows.py.

import torch
import triton
import triton.language as tl


@triton.jit
def _fill_padded_rows(
    x_ptr,
    out_ptr,
    num_non_padded_ptr,
    n_cols,
    fill_value,
    stride_x,
    stride_out,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col_block = tl.program_id(1)
    n_valid = tl.load(num_non_padded_ptr).to(tl.int64)
    cols = (col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)).to(tl.int64)
    mask = cols < n_cols
    # The copy load stays outside the runtime branch: GCU300 has only ever
    # legalized this team's masked vector loads at top level (deepep_permute
    # form), while stores under scalar branches are proven (fill S0,
    # deepep_permute). Pad rows mask the load out entirely.
    value = tl.load(
        x_ptr + row * stride_x + cols, mask=mask & (row < n_valid), other=0
    )
    if row < n_valid:
        tl.store(out_ptr + row * stride_out + cols, value, mask=mask)
    else:
        fill = tl.full(
            (BLOCK_COLS,), fill_value, dtype=out_ptr.dtype.element_ty
        )
        tl.store(out_ptr + row * stride_out + cols, fill, mask=mask)


def fill_padded_rows(x, num_token_non_padded, fill_value):
    assert x.ndim == 2 and x.stride(1) == 1
    assert num_token_non_padded.numel() == 1
    assert not num_token_non_padded.dtype.is_floating_point
    assert num_token_non_padded.device == x.device
    if isinstance(fill_value, torch.Tensor):
        fill_value = fill_value.item()
    n_rows, n_cols = x.shape
    # One kernel writes every output element exactly once: valid rows are
    # copied from x and padded rows are filled in place, instead of cloning
    # the whole tensor and overwriting the padding a second time. E3 tiles
    # the columns (one program per (row, col-block), cap 1024 lanes) - the
    # per-chip leaderboard shows the leaders 3-4x ahead exactly on the wide-
    # shape chips (tianshu/huawei/enflame) while narrow shapes were at
    # parity, and e2's whole-row form ran 84s on tianshu.
    out = torch.empty((n_rows, n_cols), dtype=x.dtype, device=x.device)
    if n_rows and n_cols:
        block = max(min(triton.next_power_of_2(n_cols), 1024), 16)
        _fill_padded_rows[(n_rows, triton.cdiv(n_cols, block))](
            x,
            out,
            num_token_non_padded,
            n_cols,
            fill_value,
            x.stride(0),
            out.stride(0),
            BLOCK_COLS=block,
        )
    return out


__all__ = ["fill_padded_rows"]
