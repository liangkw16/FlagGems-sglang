# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Ascend vendor: persistent launch kept close to the physical Vector
# Core count (Ascend vector_operator.md: GPU-style huge grids pay
# repeated dispatch overhead on NPUs - decode_attention template).
# The generic runs one program per row; this vendor strides rows in
# an inner loop and hoists the device-side row count out of it. The
# per-row math and the top-level masked load are identical.

import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

_DEFAULT_VECTOR_CORES = 40


def _get_num_vector_cores(device_index):
    properties = driver.active.utils.get_device_properties(device_index)
    return int(properties.get("num_vectorcore", _DEFAULT_VECTOR_CORES))


def _worker_count(tensor):
    index = getattr(getattr(tensor, "device", None), "index", None)
    if index is None:
        return _DEFAULT_VECTOR_CORES
    return _get_num_vector_cores(index)


@triton.jit
def _fill_padded_rows_persistent(
    x_ptr,
    out_ptr,
    num_non_padded_ptr,
    n_rows,
    n_cols,
    fill_value,
    stride_x,
    stride_out,
    BLOCK_COLS: tl.constexpr,
):
    count = tl.load(num_non_padded_ptr)
    n_valid = tl.minimum(count, n_rows).to(tl.int64)
    # Match Python slice start, including negative and out-of-range counts.
    if n_valid < 0:
        n_valid = tl.maximum(n_valid + n_rows, 0)
    cols = tl.arange(0, BLOCK_COLS).to(tl.int64)
    mask = cols < n_cols
    for row in range(tl.program_id(0), n_rows, tl.num_programs(0)):
        row64 = row.to(tl.int64)
        # The copy load stays outside the runtime branch (GCU proven
        # form, see generic); pad rows mask the load out entirely.
        value = tl.load(
            x_ptr + row64 * stride_x + cols,
            mask=mask & (row64 < n_valid),
            other=0,
        )
        if row64 < n_valid:
            tl.store(out_ptr + row64 * stride_out + cols, value, mask=mask)
        else:
            fill = tl.full((BLOCK_COLS,), fill_value, dtype=out_ptr.dtype.element_ty)
            tl.store(out_ptr + row64 * stride_out + cols, fill, mask=mask)


def fill_padded_rows(x, num_token_non_padded, fill_value):
    assert x.ndim == 2 and x.stride(1) == 1
    assert num_token_non_padded.numel() == 1
    assert not num_token_non_padded.dtype.is_floating_point
    assert num_token_non_padded.device == x.device
    if isinstance(fill_value, torch.Tensor):
        fill_value = fill_value.item()
    n_rows, n_cols = x.shape
    out = torch.empty((n_rows, n_cols), dtype=x.dtype, device=x.device)
    if n_rows and n_cols:
        _fill_padded_rows_persistent[(min(n_rows, _worker_count(x)),)](
            x,
            out,
            num_token_non_padded,
            n_rows,
            n_cols,
            fill_value,
            x.stride(0),
            out.stride(0),
            BLOCK_COLS=triton.next_power_of_2(n_cols),
        )
    return out


__all__ = ["fill_padded_rows"]
