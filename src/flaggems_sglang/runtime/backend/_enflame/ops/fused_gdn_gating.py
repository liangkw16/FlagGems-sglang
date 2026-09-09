# Enflame vendor: one program per batch row with the full head vector
# (T51-proven enflame fix: no grid-stride loop, no runtime branch).
#
# e6: launch parameters unpinned -- leaving num_warps to the GCU backend
# default has won twice on this chip (T19-E5, T51-E5, +38% each), and
# batch-4 ops without an explicit num_warps carry a median enflame
# speedup of 2.08 vs 0.73 for pinned ones.
#
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
def _gdn_gating_row_kernel(
    a_log_ptr,
    a_ptr,
    b_ptr,
    dt_bias_ptr,
    g_ptr,
    beta_out_ptr,
    num_heads,
    a_stride_batch,
    a_stride_head,
    b_stride_batch,
    b_stride_head,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    H_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, H_PAD)
    mask = offs < num_heads

    a_val = tl.load(
        a_ptr + row * a_stride_batch + offs * a_stride_head,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    b_val = tl.load(
        b_ptr + row * b_stride_batch + offs * b_stride_head,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    a_log = tl.load(a_log_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dt_b = tl.load(dt_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    x = a_val + dt_b
    # softplus with threshold in E1's platform-validated math; the
    # select form keeps the kernel branch-free (enflame rejects
    # runtime control flow).
    e = tl.exp(beta * x)
    softplus_x = tl.where(beta * x <= threshold, tl.log(1.0 + e) / beta, x)

    g = -tl.exp(a_log) * softplus_x
    beta_output = 1.0 / (1.0 + tl.exp(-b_val))

    tl.store(g_ptr + row * num_heads + offs, g, mask=mask)
    tl.store(beta_out_ptr + row * num_heads + offs, beta_output, mask=mask)


def fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    batch, num_heads = a.shape
    device = a.device
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=device)
    beta_output = torch.empty(
        1, batch, num_heads, dtype=torch.float32, device=device
    )
    if batch * num_heads == 0:
        return g, beta_output
    h_pad = max(triton.next_power_of_2(num_heads), 2)
    _gdn_gating_row_kernel[(batch,)](
        A_log,
        a,
        b,
        dt_bias,
        g,
        beta_output,
        num_heads,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        beta=float(beta),
        threshold=float(threshold),
        H_PAD=h_pad,
    )
    return g, beta_output


__all__ = ["fused_gdn_gating"]
