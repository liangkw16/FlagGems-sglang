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

_BLOCK_SIZE = 2048
_MAX_GRID = 65535


@triton.jit
def _act_and_mul_flat_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    half_width,
    swiglu_limit,
    BLOCK_SIZE: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    ACT_IS_GELU: tl.constexpr,
):
    # kunlunxin vendor: flat output-element grid-stride over
    # BLOCK_SIZE lanes (BLOCK is the only axis that ever moved
    # kunlunxin - T29/T39 dual proof, 2048 peak); activation math
    # identical to the generic kernel including the input-dtype
    # multiply after the fp32 activation is cast back.
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK_SIZE
    for start in range(pid * BLOCK_SIZE, n_elements, step):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        rows = offsets // half_width
        cols = offsets - rows * half_width
        input_base = rows.to(tl.int64) * (2 * half_width) + cols
        gate = tl.load(x_ptr + input_base, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(x_ptr + input_base + half_width, mask=mask, other=0.0).to(
            tl.float32
        )
        if HAS_LIMIT:
            gate = tl.minimum(gate, swiglu_limit)
            up = tl.minimum(tl.maximum(up, -swiglu_limit), swiglu_limit)
        if ACT_IS_GELU:
            scaled = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
            exp_neg = tl.exp(-2.0 * tl.abs(scaled))
            ratio = (1.0 - exp_neg) / (1.0 + exp_neg)
            tanh_scaled = tl.where(scaled < 0.0, -ratio, ratio)
            act = gate * 0.5 * (1.0 + tanh_scaled)
        else:
            act = gate / (1.0 + tl.exp(-gate))
        elem_ty = output_ptr.dtype.element_ty
        tl.store(
            output_ptr + offsets,
            act.to(elem_ty) * up.to(elem_ty),
            mask=mask,
        )


def act_and_mul(gateup_output, activation="silu", swiglu_limit=None):
    if activation not in ("silu", "gelu"):
        raise ValueError(f"Unsupported activation: {activation}")
    x = gateup_output.contiguous()
    last_dim = x.shape[-1]
    half_width = last_dim // 2
    output = torch.empty(x.shape[:-1] + (half_width,), dtype=x.dtype, device=x.device)
    rows = x.numel() // last_dim if last_dim else 0
    n_elements = rows * half_width
    if n_elements == 0:
        return output
    has_limit = swiglu_limit is not None
    limit = float(swiglu_limit) if has_limit else 0.0
    grid = (min(triton.cdiv(n_elements, _BLOCK_SIZE), _MAX_GRID),)
    _act_and_mul_flat_kernel[grid](
        x,
        output,
        n_elements,
        half_width,
        limit,
        BLOCK_SIZE=_BLOCK_SIZE,
        HAS_LIMIT=has_limit,
        ACT_IS_GELU=(activation == "gelu"),
    )
    return output


__all__ = ["act_and_mul"]
