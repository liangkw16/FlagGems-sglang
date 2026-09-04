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

# Kunlunxin vendor: vectorized elementwise over flattened [B*H] index
# (the generic scalar-loop form measured 0.0625x on this chip).

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _gdn_gating_vec_kernel(
    a_log_ptr,
    a_ptr,
    b_ptr,
    dt_bias_ptr,
    g_ptr,
    beta_out_ptr,
    total,
    num_heads,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    for start in range(pid * BLOCK, total, grid_size * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < total
        bh = offs // num_heads
        h = offs - bh * num_heads

        a_val = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b_val = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        a_log = tl.load(a_log_ptr + h, mask=mask, other=0.0).to(tl.float32)
        dt_b = tl.load(dt_bias_ptr + h, mask=mask, other=0.0).to(tl.float32)

        x = a_val + dt_b
        bx = beta * x
        softplus_x = tl.where(
            bx <= threshold,
            tl.maximum(bx, 0.0) / beta + tl.log(1.0 + tl.exp(-tl.abs(bx))) / beta,
            x,
        )
        g = -tl.exp(a_log) * softplus_x
        beta_output = 1.0 / (1.0 + tl.exp(-b_val))

        tl.store(g_ptr + offs, g, mask=mask)
        tl.store(beta_out_ptr + offs, beta_output, mask=mask)


def fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    batch, num_heads = a.shape
    device = a.device
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=device)
    beta_output = torch.empty(1, batch, num_heads, dtype=torch.float32, device=device)
    a = a.contiguous().view(-1)
    b = b.contiguous().view(-1)
    total = batch * num_heads
    if total == 0:
        return g, beta_output
    grid = (min(triton.cdiv(total, _BLOCK), 65535),)
    _gdn_gating_vec_kernel[grid](
        A_log,
        a,
        b,
        dt_bias,
        g,
        beta_output,
        total,
        num_heads,
        beta=float(beta),
        threshold=float(threshold),
        BLOCK=_BLOCK,
        num_warps=4,
        num_stages=1,
    )
    return g, beta_output


__all__ = ["fused_gdn_gating"]
