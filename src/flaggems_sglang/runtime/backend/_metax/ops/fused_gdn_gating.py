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

# Metax vendor (e4): the Kunlunxin-proven flat vectorization over the
# [B*H] index with a 2048-lane block (the scalar generic form measured
# 1.31-1.34x on this chip). Unlike the Kunlunxin wrapper this kernel
# keeps a/b row/col strides explicit instead of flattening a contiguous
# copy (SGLang #22312: non-contiguous gating inputs must not be read
# through an assumed-contiguous layout).

import torch
import triton
import triton.language as tl

_BLOCK = 2048


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
    a_stride_batch,
    a_stride_head,
    b_stride_batch,
    b_stride_head,
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

        a_val = tl.load(
            a_ptr + bh * a_stride_batch + h * a_stride_head,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        b_val = tl.load(
            b_ptr + bh * b_stride_batch + h * b_stride_head,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        a_log = tl.load(a_log_ptr + h, mask=mask, other=0.0).to(tl.float32)
        dt_b = tl.load(dt_bias_ptr + h, mask=mask, other=0.0).to(tl.float32)

        x = a_val + dt_b
        e = tl.exp(beta * x)
        softplus_x = tl.where(beta * x <= threshold, tl.log(1.0 + e) / beta, x)
        g = -tl.exp(a_log) * softplus_x
        beta_output = 1.0 / (1.0 + tl.exp(-b_val))

        tl.store(g_ptr + offs, g, mask=mask)
        tl.store(beta_out_ptr + offs, beta_output, mask=mask)


def fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    batch, num_heads = a.shape
    device = a.device
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=device)
    beta_output = torch.empty(
        1, batch, num_heads, dtype=torch.float32, device=device
    )
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
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        beta=float(beta),
        threshold=float(threshold),
        BLOCK=_BLOCK,
        num_warps=4,
        num_stages=1,
    )
    return g, beta_output


__all__ = ["fused_gdn_gating"]
