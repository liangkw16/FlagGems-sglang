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

# E10 keeps the two-stage 16x16 math/workspace from E7, but removes every
# device-side loop. The host expands N slices and batch chunks; each kernel
# launch uses a direct (P tile, batch, head) grid. E4/E5 proved that direct
# scheduling alone cannot compile a full [P, dstate] matrix on Kunlun.

import torch
import triton
import triton.language as tl

_MAX_GRID = 65535
_N_SLICE = 16


@triton.jit(do_not_specialize=["output_start", "slice_start"])
def _ssu_state_kernel(
    state_ptr,
    new_state_ptr,
    x_ptr,
    dt_ptr,
    a_ptr,
    b_ptr,
    dt_bias_ptr,
    output_start,
    slice_start,
    num_heads,
    dim,
    dstate,
    num_groups,
    HAS_DT_BIAS: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    N_SLICE: tl.constexpr,
    isCloseCoreTiling: tl.constexpr,
    isCloseVectorization: tl.constexpr,
    isCloseUnrollControl: tl.constexpr,
):
    out_idx = output_start + tl.program_id(0)
    p_idx = out_idx % dim
    row = out_idx // dim
    h = row % num_heads
    b = row // num_heads
    n_off = tl.arange(0, N_SLICE)
    n_idx = slice_start + n_off
    n_mask = n_idx < dstate
    ratio = num_heads // num_groups
    g = h // ratio
    dt_val = tl.load(dt_ptr + out_idx).to(tl.float32)
    if HAS_DT_BIAS:
        dt_val += tl.load(dt_bias_ptr + h * dim + p_idx).to(tl.float32)
    if DT_SOFTPLUS:
        dt_val = tl.maximum(dt_val, 0.0) + tl.log(
            1.0 + tl.exp(-tl.abs(dt_val))
        )
    x_val = tl.load(x_ptr + out_idx).to(tl.float32)
    # A is [nheads, dim, dstate] (full layout, E3 contract).
    a_val = tl.load(
        a_ptr + (h * dim + p_idx) * dstate + n_idx,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)
    b_val = tl.load(
        b_ptr + (b * num_groups + g) * dstate + n_idx,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)
    s_base = out_idx * dstate + n_idx
    s_val = tl.load(state_ptr + s_base, mask=n_mask, other=0.0).to(tl.float32)
    d_a = tl.exp(dt_val * a_val)
    new_s = s_val * d_a + (dt_val * x_val) * b_val
    state_ty = new_state_ptr.dtype.element_ty
    tl.store(new_state_ptr + s_base, new_s.to(state_ty), mask=n_mask)


@triton.jit(do_not_specialize=["output_start", "slice_index", "slice_start"])
def _ssu_partial_kernel(
    state_ptr,
    x_ptr,
    dt_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    dt_bias_ptr,
    partial_y_ptr,
    output_start,
    slice_index,
    slice_start,
    num_heads,
    dim,
    dstate,
    num_groups,
    num_slices,
    HAS_DT_BIAS: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    N_SLICE: tl.constexpr,
    isCloseCoreTiling: tl.constexpr,
    isCloseVectorization: tl.constexpr,
    isCloseUnrollControl: tl.constexpr,
):
    out_idx = output_start + tl.program_id(0)
    p_idx = out_idx % dim
    row = out_idx // dim
    h = row % num_heads
    b = row // num_heads
    n_off = tl.arange(0, N_SLICE)
    n_idx = slice_start + n_off
    n_mask = n_idx < dstate
    ratio = num_heads // num_groups
    g = h // ratio
    dt_val = tl.load(dt_ptr + out_idx).to(tl.float32)
    if HAS_DT_BIAS:
        dt_val += tl.load(dt_bias_ptr + h * dim + p_idx).to(tl.float32)
    if DT_SOFTPLUS:
        dt_val = tl.maximum(dt_val, 0.0) + tl.log(
            1.0 + tl.exp(-tl.abs(dt_val))
        )
    x_val = tl.load(x_ptr + out_idx).to(tl.float32)
    a_val = tl.load(
        a_ptr + (h * dim + p_idx) * dstate + n_idx,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)
    bc_base = (b * num_groups + g) * dstate + n_idx
    b_val = tl.load(b_ptr + bc_base, mask=n_mask, other=0.0).to(tl.float32)
    c_val = tl.load(c_ptr + bc_base, mask=n_mask, other=0.0).to(tl.float32)
    s_val = tl.load(
        state_ptr + out_idx * dstate + n_idx,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)
    new_s = s_val * tl.exp(dt_val * a_val) + (dt_val * x_val) * b_val
    part = tl.sum(tl.where(n_mask, new_s * c_val, 0.0), axis=0)
    tl.store(partial_y_ptr + out_idx * num_slices + slice_index, part)


@triton.jit(do_not_specialize=["output_start"])
def _ssu_stage2_kernel(
    partial_y_ptr,
    x_ptr,
    d_ptr,
    z_ptr,
    y_ptr,
    output_start,
    num_heads,
    dim,
    num_slices,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    N_SLICE_POW2: tl.constexpr,
    isCloseCoreTiling: tl.constexpr,
    isCloseVectorization: tl.constexpr,
):
    out_idx = output_start + tl.program_id(0)
    p_idx = out_idx % dim
    row = out_idx // dim
    h = row % num_heads
    sl_off = tl.arange(0, N_SLICE_POW2)
    sl_mask = sl_off < num_slices
    y_ty = y_ptr.dtype.element_ty
    parts = tl.load(
        partial_y_ptr + out_idx * num_slices + sl_off,
        mask=sl_mask,
        other=0.0,
    )
    y_val = tl.sum(parts, axis=0)
    if HAS_D:
        d_val = tl.load(d_ptr + h * dim + p_idx).to(tl.float32)
        x_val = tl.load(x_ptr + out_idx).to(tl.float32)
        y_val += d_val * x_val
    if HAS_Z:
        z_val = tl.load(z_ptr + out_idx).to(tl.float32)
        y_val *= z_val * tl.sigmoid(z_val)
    tl.store(y_ptr + out_idx, y_val.to(y_ty))


def selective_state_update(
    state,
    x,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
):
    state = state.contiguous()
    x = x.contiguous()
    dt = dt.contiguous()
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    if z is not None:
        z = z.contiguous()
    if D is not None:
        D = D.contiguous()
    if dt_bias is not None:
        dt_bias = dt_bias.contiguous()

    batch, nheads, dim, dstate = state.shape
    num_groups = B.shape[1]
    y = torch.empty_like(x)
    new_state = state.clone()
    total_rows = batch * nheads
    if total_rows * dim == 0 or dstate == 0:
        return y, new_state

    num_slices = triton.cdiv(dstate, _N_SLICE)
    partial_y = torch.empty(
        (total_rows * dim * num_slices,),
        dtype=torch.float32,
        device=x.device,
    )
    total_outputs = total_rows * dim
    for slice_start in range(0, dstate, _N_SLICE):
        slice_index = slice_start // _N_SLICE
        for output_start in range(0, total_outputs, _MAX_GRID):
            output_count = min(_MAX_GRID, total_outputs - output_start)
            _ssu_partial_kernel[(output_count,)](
                state,
                x,
                dt,
                A,
                B,
                C,
                dt_bias if dt_bias is not None else x,
                partial_y,
                output_start,
                slice_index,
                slice_start,
                nheads,
                dim,
                dstate,
                num_groups,
                num_slices,
                HAS_DT_BIAS=dt_bias is not None,
                DT_SOFTPLUS=bool(dt_softplus),
                N_SLICE=_N_SLICE,
                isCloseCoreTiling=True,
                isCloseVectorization=True,
                isCloseUnrollControl=True,
            )
        for output_start in range(0, total_outputs, _MAX_GRID):
            output_count = min(_MAX_GRID, total_outputs - output_start)
            _ssu_state_kernel[(output_count,)](
                state,
                new_state,
                x,
                dt,
                A,
                B,
                dt_bias if dt_bias is not None else x,
                output_start,
                slice_start,
                nheads,
                dim,
                dstate,
                num_groups,
                HAS_DT_BIAS=dt_bias is not None,
                DT_SOFTPLUS=bool(dt_softplus),
                N_SLICE=_N_SLICE,
                isCloseCoreTiling=True,
                isCloseVectorization=True,
                isCloseUnrollControl=True,
            )
    for output_start in range(0, total_outputs, _MAX_GRID):
        output_count = min(_MAX_GRID, total_outputs - output_start)
        _ssu_stage2_kernel[(output_count,)](
            partial_y,
            x,
            D if D is not None else x,
            z if z is not None else x,
            y,
            output_start,
            nheads,
            dim,
            num_slices,
            HAS_D=D is not None,
            HAS_Z=z is not None,
            N_SLICE_POW2=triton.next_power_of_2(max(num_slices, 2)),
            isCloseCoreTiling=True,
            isCloseVectorization=True,
        )
    return y, new_state


__all__ = ["selective_state_update"]
