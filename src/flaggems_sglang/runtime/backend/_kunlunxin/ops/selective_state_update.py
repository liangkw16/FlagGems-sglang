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
_BLOCK_P = 8
_N_SLICE = 16


@triton.jit(do_not_specialize=["batch_start", "slice_index", "slice_start"])
def _ssu_stage1_kernel(
    state_ptr,
    x_ptr,
    dt_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    dt_bias_ptr,
    partial_y_ptr,
    batch_start,
    slice_index,
    slice_start,
    num_heads,
    dim,
    dstate,
    num_groups,
    num_slices,
    HAS_DT_BIAS: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    N_SLICE: tl.constexpr,
    isCloseCoreTiling: tl.constexpr,
    isCloseVectorization: tl.constexpr,
    isCloseUnrollControl: tl.constexpr,
):
    p_tile = tl.program_id(0)
    b = batch_start + tl.program_id(1)
    h = tl.program_id(2)
    p_off = tl.arange(0, BLOCK_P)
    n_off = tl.arange(0, N_SLICE)
    p_idx = p_tile * BLOCK_P + p_off
    n_idx = slice_start + n_off
    p_mask = p_idx < dim
    n_mask = n_idx < dstate
    pn_mask = p_mask[:, None] & n_mask[None, :]
    row = b * num_heads + h
    ratio = num_heads // num_groups
    g = h // ratio
    dt_val = tl.load(dt_ptr + row * dim + p_idx, mask=p_mask, other=0.0).to(
        tl.float32
    )
    if HAS_DT_BIAS:
        dt_val += tl.load(
            dt_bias_ptr + h * dim + p_idx, mask=p_mask, other=0.0
        ).to(tl.float32)
    if DT_SOFTPLUS:
        dt_val = tl.maximum(dt_val, 0.0) + tl.log(
            1.0 + tl.exp(-tl.abs(dt_val))
        )
    x_val = tl.load(x_ptr + row * dim + p_idx, mask=p_mask, other=0.0).to(
        tl.float32
    )
    # A is [nheads, dim, dstate] (full layout, E3 contract).
    a_val = tl.load(
        a_ptr + (h * dim + p_idx)[:, None] * dstate + n_idx[None, :],
        mask=pn_mask,
        other=0.0,
    ).to(tl.float32)
    b_val = tl.load(
        b_ptr + (b * num_groups + g) * dstate + n_idx,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)
    c_val = tl.load(
        c_ptr + (b * num_groups + g) * dstate + n_idx,
        mask=n_mask,
        other=0.0,
    ).to(tl.float32)
    s_base = (row * dim + p_idx)[:, None] * dstate + n_idx[None, :]
    s_val = tl.load(state_ptr + s_base, mask=pn_mask, other=0.0).to(tl.float32)
    d_a = tl.exp(dt_val[:, None] * a_val)
    new_s = s_val * d_a + (dt_val * x_val)[:, None] * b_val[None, :]
    state_ty = state_ptr.dtype.element_ty
    tl.store(state_ptr + s_base, new_s.to(state_ty), mask=pn_mask)
    part = tl.sum(
        tl.where(n_mask[None, :], new_s, 0.0) * c_val[None, :], axis=1
    )
    tl.store(
        partial_y_ptr + (row * dim + p_idx) * num_slices + slice_index,
        part,
        mask=p_mask,
    )


@triton.jit(do_not_specialize=["batch_start"])
def _ssu_stage2_kernel(
    partial_y_ptr,
    x_ptr,
    d_ptr,
    z_ptr,
    y_ptr,
    batch_start,
    num_heads,
    dim,
    num_slices,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    BLOCK_P: tl.constexpr,
    N_SLICE_POW2: tl.constexpr,
    isCloseVectorization: tl.constexpr,
):
    p_tile = tl.program_id(0)
    b = batch_start + tl.program_id(1)
    h = tl.program_id(2)
    p_off = tl.arange(0, BLOCK_P)
    sl_off = tl.arange(0, N_SLICE_POW2)
    sl_mask = sl_off < num_slices
    y_ty = y_ptr.dtype.element_ty
    row = b * num_heads + h
    p_idx = p_tile * BLOCK_P + p_off
    p_mask = p_idx < dim
    parts = tl.load(
        partial_y_ptr
        + (row * dim + p_idx)[:, None] * num_slices
        + sl_off[None, :],
        mask=p_mask[:, None] & sl_mask[None, :],
        other=0.0,
    )
    y_val = tl.sum(parts, axis=1)
    if HAS_D:
        d_val = tl.load(d_ptr + h * dim + p_idx, mask=p_mask, other=0.0).to(
            tl.float32
        )
        x_val = tl.load(x_ptr + row * dim + p_idx, mask=p_mask, other=0.0).to(
            tl.float32
        )
        y_val += d_val * x_val
    if HAS_Z:
        z_val = tl.load(z_ptr + row * dim + p_idx, mask=p_mask, other=0.0).to(
            tl.float32
        )
        y_val *= z_val * tl.sigmoid(z_val)
    tl.store(y_ptr + row * dim + p_idx, y_val.to(y_ty), mask=p_mask)


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
    tiles_per_head = triton.cdiv(dim, _BLOCK_P)
    programs_per_batch = tiles_per_head * nheads
    assert (
        programs_per_batch <= _MAX_GRID
    ), "sample exceeds physical grid limit"
    batch_chunk = max(1, _MAX_GRID // programs_per_batch)
    for slice_start in range(0, dstate, _N_SLICE):
        slice_index = slice_start // _N_SLICE
        for batch_start in range(0, batch, batch_chunk):
            batch_count = min(batch_chunk, batch - batch_start)
            _ssu_stage1_kernel[(tiles_per_head, batch_count, nheads)](
                new_state,
                x,
                dt,
                A,
                B,
                C,
                dt_bias if dt_bias is not None else x,
                partial_y,
                batch_start,
                slice_index,
                slice_start,
                nheads,
                dim,
                dstate,
                num_groups,
                num_slices,
                HAS_DT_BIAS=dt_bias is not None,
                DT_SOFTPLUS=bool(dt_softplus),
                BLOCK_P=_BLOCK_P,
                N_SLICE=_N_SLICE,
                isCloseCoreTiling=True,
                isCloseVectorization=True,
                isCloseUnrollControl=True,
            )
    for batch_start in range(0, batch, batch_chunk):
        batch_count = min(batch_chunk, batch - batch_start)
        _ssu_stage2_kernel[(tiles_per_head, batch_count, nheads)](
            partial_y,
            x,
            D if D is not None else x,
            z if z is not None else x,
            y,
            batch_start,
            nheads,
            dim,
            num_slices,
            HAS_D=D is not None,
            HAS_Z=z is not None,
            BLOCK_P=_BLOCK_P,
            N_SLICE_POW2=triton.next_power_of_2(max(num_slices, 2)),
            isCloseVectorization=True,
        )
    return y, new_state


__all__ = ["selective_state_update"]
