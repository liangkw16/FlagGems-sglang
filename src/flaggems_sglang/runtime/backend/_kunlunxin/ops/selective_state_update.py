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

_MAX_GRID = 65535
_BLOCK_P = 4


@triton.jit
def _selective_state_update_kernel(
    state_ptr,
    x_ptr,
    dt_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    z_ptr,
    dt_bias_ptr,
    y_ptr,
    total_tiles,
    num_heads,
    dim,
    dstate,
    num_groups,
    tiles_per_head,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    DSTATE: tl.constexpr,
):
    # one program per [b, h, p-tile]; tile shape [BLOCK_P, DSTATE];
    # flat 1D capped grid with block-level div/mod, int32 offsets
    pid = tl.program_id(0)
    grid_stride = tl.num_programs(0)
    p_off = tl.arange(0, BLOCK_P)
    n_off = tl.arange(0, DSTATE)
    for tile_id in range(pid, total_tiles, grid_stride):
        tile_in_head = tile_id % tiles_per_head
        bh = tile_id // tiles_per_head
        h = bh % num_heads
        b = bh // num_heads
        p_idx = tile_in_head * BLOCK_P + p_off
        p_mask = p_idx < dim
        n_mask = n_off < dstate
        pn_mask = p_mask[:, None] & n_mask[None, :]

        row = b * num_heads + h
        dt_val = tl.load(
            dt_ptr + row * dim + p_idx, mask=p_mask, other=0.0
        ).to(tl.float32)
        if HAS_DT_BIAS:
            dt_val += tl.load(
                dt_bias_ptr + h * dim + p_idx, mask=p_mask, other=0.0
            ).to(tl.float32)
        if DT_SOFTPLUS:
            # overflow-safe softplus: max(t,0) + log1p(exp(-|t|))
            dt_val = tl.maximum(dt_val, 0.0) + tl.log(
                1.0 + tl.exp(-tl.abs(dt_val))
            )

        x_val = tl.load(x_ptr + row * dim + p_idx, mask=p_mask, other=0.0).to(
            tl.float32
        )

        ratio = num_heads // num_groups
        g = h // ratio
        a_val = tl.load(
            a_ptr + (h * dim + p_idx)[:, None] * dstate + n_off[None, :],
            mask=pn_mask,
            other=0.0,
        ).to(tl.float32)
        b_val = tl.load(
            b_ptr + (b * num_groups + g) * dstate + n_off,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        c_val = tl.load(
            c_ptr + (b * num_groups + g) * dstate + n_off,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)

        # A is [H, P, N] in the executable scoring contract.
        d_a = tl.exp(dt_val[:, None] * a_val)
        s_base = (row * dim + p_idx)[:, None] * dstate + n_off[None, :]
        s_val = tl.load(state_ptr + s_base, mask=pn_mask, other=0.0).to(
            tl.float32
        )
        new_s = s_val * d_a + (dt_val * x_val)[:, None] * b_val[None, :]
        state_ty = state_ptr.dtype.element_ty
        tl.store(state_ptr + s_base, new_s.to(state_ty), mask=pn_mask)

        y_val = tl.sum(
            tl.where(n_mask[None, :], new_s, 0.0) * c_val[None, :], axis=1
        )
        if HAS_D:
            d_val = tl.load(
                d_ptr + h * dim + p_idx, mask=p_mask, other=0.0
            ).to(tl.float32)
            y_val += d_val * x_val
        if HAS_Z:
            z_val = tl.load(
                z_ptr + row * dim + p_idx, mask=p_mask, other=0.0
            ).to(tl.float32)
            y_val *= z_val * tl.sigmoid(z_val)
        y_ty = y_ptr.dtype.element_ty
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
    # the platform harness may pass vllm-style 1-D [nheads] variants of
    # A / D / dt_bias; normalize to the 2-D layouts the kernel indexes
    if A.dim() == 1:
        A = A.unsqueeze(1).expand(nheads, dstate).contiguous()
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(1).expand(nheads, dim).contiguous()
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(1).expand(nheads, dim).contiguous()
    y = torch.empty_like(x)
    new_state = state.clone()
    if batch * nheads * dim == 0 or dstate == 0:
        return y, new_state

    tiles_per_head = triton.cdiv(dim, _BLOCK_P)
    total_tiles = batch * nheads * tiles_per_head
    grid = (min(total_tiles, _MAX_GRID),)
    _selective_state_update_kernel[grid](
        new_state,
        x,
        dt,
        A,
        B,
        C,
        D if D is not None else x,
        z if z is not None else x,
        dt_bias if dt_bias is not None else x,
        y,
        total_tiles,
        nheads,
        dim,
        dstate,
        num_groups,
        tiles_per_head,
        HAS_D=D is not None,
        HAS_Z=z is not None,
        HAS_DT_BIAS=dt_bias is not None,
        DT_SOFTPLUS=bool(dt_softplus),
        BLOCK_P=_BLOCK_P,
        DSTATE=triton.next_power_of_2(dstate),
    )
    return y, new_state


__all__ = ["selective_state_update"]
