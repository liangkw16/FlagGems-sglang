# r1 water re-roll carrier of e4 sub-110xx second-identity bytes.
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
def _gdn_gating_kernel(
    a_log_ptr,
    a_ptr,
    b_ptr,
    dt_bias_ptr,
    g_ptr,
    beta_out_ptr,
    batch,
    num_heads,
    a_stride_batch,
    a_stride_head,
    b_stride_batch,
    b_stride_head,
    beta: tl.constexpr,
    threshold: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    total = batch * num_heads
    for idx in range(pid, total, grid_size):
        bh = idx // num_heads
        h = idx - bh * num_heads

        a_val = tl.load(a_ptr + bh * a_stride_batch + h * a_stride_head).to(tl.float32)
        b_val = tl.load(b_ptr + bh * b_stride_batch + h * b_stride_head).to(tl.float32)
        a_log = tl.load(a_log_ptr + h).to(tl.float32)
        dt_b = tl.load(dt_bias_ptr + h).to(tl.float32)

        x = a_val + dt_b
        # softplus with threshold: beta*x <= threshold -> softplus, else x
        if beta * x <= threshold:
            softplus_x = tl.log(1.0 + tl.exp(beta * x)) / beta
        else:
            softplus_x = x

        g = -tl.exp(a_log) * softplus_x
        beta_output = 1.0 / (1.0 + tl.exp(-b_val))

        tl.store(g_ptr + idx, g)
        tl.store(beta_out_ptr + idx, beta_output)


def fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    batch, num_heads = a.shape
    device = a.device
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=device)
    beta_output = torch.empty(1, batch, num_heads, dtype=torch.float32, device=device)
    if batch * num_heads == 0:
        return g, beta_output
    total = batch * num_heads
    grid = (min(total, 1024),)
    _gdn_gating_kernel[grid](
        A_log,
        a,
        b,
        dt_bias,
        g,
        beta_output,
        batch,
        num_heads,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        beta=float(beta),
        threshold=float(threshold),
        num_warps=4,
        num_stages=1,
    )
    return g, beta_output


__all__ = ["fused_gdn_gating"]
