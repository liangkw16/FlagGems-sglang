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
def _ssu_fused_kernel(
    state_ptr,
    new_state_ptr,
    x_ptr,
    dt_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    z_ptr,
    dt_bias_ptr,
    y_ptr,
    num_heads,
    dim,
    dstate,
    num_groups,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    N_BLOCK: tl.constexpr,
    isCloseCoreTiling: tl.constexpr,
    isCloseVectorization: tl.constexpr,
    isCloseUnrollControl: tl.constexpr,
):
    out_idx = tl.program_id(0)
    p_idx = out_idx % dim
    row = out_idx // dim
    h = row % num_heads
    b = row // num_heads
    ratio = num_heads // num_groups
    g = h // ratio

    n_off = tl.arange(0, N_BLOCK)
    n_mask = n_off < dstate
    dt_val = tl.load(dt_ptr + out_idx).to(tl.float32)
    if HAS_DT_BIAS:
        dt_val += tl.load(dt_bias_ptr + h * dim + p_idx).to(tl.float32)
    if DT_SOFTPLUS:
        dt_val = tl.maximum(dt_val, 0.0) + tl.log(
            1.0 + tl.exp(-tl.abs(dt_val))
        )
    x_val = tl.load(x_ptr + out_idx).to(tl.float32)

    state_base = out_idx * dstate
    a_base = (h * dim + p_idx) * dstate
    bc_base = (b * num_groups + g) * dstate
    state_val = tl.load(
        state_ptr + state_base + n_off, mask=n_mask, other=0.0
    ).to(tl.float32)
    a_val = tl.load(a_ptr + a_base + n_off, mask=n_mask, other=0.0).to(
        tl.float32
    )
    new_s = state_val * tl.exp(dt_val * a_val)
    b_val = tl.load(b_ptr + bc_base + n_off, mask=n_mask, other=0.0).to(
        tl.float32
    )
    new_s += (dt_val * x_val) * b_val
    c_val = tl.load(c_ptr + bc_base + n_off, mask=n_mask, other=0.0).to(
        tl.float32
    )
    y_val = tl.sum(new_s * c_val, axis=0)

    state_ty = new_state_ptr.dtype.element_ty
    tl.store(
        new_state_ptr + state_base + n_off,
        new_s.to(state_ty),
        mask=n_mask,
    )
    if HAS_D:
        d_val = tl.load(d_ptr + h * dim + p_idx).to(tl.float32)
        y_val += d_val * x_val
    if HAS_Z:
        z_val = tl.load(z_ptr + out_idx).to(tl.float32)
        y_val *= z_val * tl.sigmoid(z_val)
    y_ty = y_ptr.dtype.element_ty
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
    new_state = torch.empty_like(state)
    total_outputs = batch * nheads * dim
    if total_outputs == 0 or dstate == 0:
        return y, new_state

    _ssu_fused_kernel[(total_outputs,)](
        state,
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
        nheads,
        dim,
        dstate,
        num_groups,
        HAS_D=D is not None,
        HAS_Z=z is not None,
        HAS_DT_BIAS=dt_bias is not None,
        DT_SOFTPLUS=bool(dt_softplus),
        N_BLOCK=triton.next_power_of_2(dstate),
        isCloseCoreTiling=True,
        isCloseVectorization=True,
        isCloseUnrollControl=True,
    )
    return y, new_state


__all__ = ["selective_state_update"]
