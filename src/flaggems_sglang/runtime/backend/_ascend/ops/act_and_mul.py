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

# Ascend vendor: one program per row with in-row sub-blocks where only
# the tail block carries a mask - int32 vector compares degenerate to
# scalar code on Ascend (T40 platform evidence, +130% there), so the
# hot path stays comparison-free.

import torch
import triton
import triton.language as tl

_BLOCK_INNER = 1024
_MAX_GRID = 65535


@triton.jit
def _act_and_mul_row_kernel(
    x_ptr,
    output_ptr,
    rows,
    half_width,
    swiglu_limit,
    BLOCK_INNER: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    ACT_IS_GELU: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    num_full = half_width // BLOCK_INNER
    for row in range(pid, rows, grid_size):
        row_base = row.to(tl.int64) * (2 * half_width)
        out_base = row.to(tl.int64) * half_width
        for cb in range(0, num_full):
            cols = cb * BLOCK_INNER + tl.arange(0, BLOCK_INNER)
            gate = tl.load(x_ptr + row_base + cols).to(tl.float32)
            up = tl.load(x_ptr + row_base + half_width + cols).to(tl.float32)
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
                output_ptr + out_base + cols,
                act.to(elem_ty) * up.to(elem_ty),
            )
        cols = num_full * BLOCK_INNER + tl.arange(0, BLOCK_INNER)
        mask = cols < half_width
        gate = tl.load(x_ptr + row_base + cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(x_ptr + row_base + half_width + cols, mask=mask, other=0.0).to(
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
            output_ptr + out_base + cols,
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
    if rows * half_width == 0:
        return output
    has_limit = swiglu_limit is not None
    limit = float(swiglu_limit) if has_limit else 0.0
    grid = (min(rows, _MAX_GRID),)
    _act_and_mul_row_kernel[grid](
        x,
        output,
        rows,
        half_width,
        limit,
        BLOCK_INNER=_BLOCK_INNER,
        HAS_LIMIT=has_limit,
        ACT_IS_GELU=(activation == "gelu"),
    )
    return output


__all__ = ["act_and_mul"]
