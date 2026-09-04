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

# Ascend: GQA-shared dot removes the ratio-fold redundant dot work
# (vllm-project/vllm-ascend#7576): k tiles are loaded coalesced via
# make_block_ptr as [BT, K] and dotted against tl.trans(b_k); the dot
# and the strict lower-triangular mask are computed ONCE per k-group
# and shared by every head in the group (HPG = H // Hg), which removes
# the ratio-fold redundant dot work of the generic kernel; per head
# only beta scaling, the task's safe-exp decay and the strided store
# remain.

import torch
import triton
import triton.language as tl


@triton.jit
def _kkt_vllmstyle_kernel(
    k_ptr,
    beta_ptr,
    g_ptr,
    output_ptr,
    seqlen,
    k_size,
    nheads,
    hpg,
    K_POW2: tl.constexpr,
    HAS_G: tl.constexpr,
    BT: tl.constexpr,
    USE_INPUT_DTYPE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)
    num_k_heads = nheads // hpg

    o_t = tl.arange(0, BT)
    o_t_fp32 = o_t.to(tl.float32)
    lower_tri = (o_t_fp32[:, None] > o_t_fp32[None, :]).to(tl.float32)

    k_head_base = k_ptr + (pid_b * seqlen * num_k_heads) * k_size
    for i_kg in range(0, num_k_heads):
        p_k = tl.make_block_ptr(
            k_head_base + i_kg * k_size,
            (seqlen, k_size),
            (num_k_heads * k_size, 1),
            (pid_t * BT, 0),
            (BT, K_POW2),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if not USE_INPUT_DTYPE:
            b_k = b_k.to(tl.float32)
        base_lower = tl.dot(b_k, tl.trans(b_k), input_precision="ieee") * lower_tri
        for i_h_local in range(0, hpg):
            i_h = i_kg * hpg + i_h_local
            beta_base = beta_ptr + pid_b * seqlen * nheads + i_h
            beta_i = tl.load(
                beta_base + (pid_t * BT + o_t) * nheads,
                mask=pid_t * BT + o_t < seqlen,
                other=0.0,
            ).to(tl.float32)
            res = base_lower * beta_i[:, None]
            if HAS_G:
                g_base = g_ptr + pid_b * seqlen * nheads + i_h
                g_i = tl.load(
                    g_base + (pid_t * BT + o_t) * nheads,
                    mask=pid_t * BT + o_t < seqlen,
                    other=0.0,
                ).to(tl.float32)
                g_diff = g_i[:, None] - g_i[None, :]
                res = res * tl.where(g_diff <= 0.0, tl.exp(g_diff), 0.0)
            p_a = tl.make_block_ptr(
                output_ptr + (pid_b * seqlen * nheads + i_h) * BT,
                (seqlen, BT),
                (nheads * BT, 1),
                (pid_t * BT, 0),
                (BT, BT),
                (1, 0),
            )
            tl.store(
                p_a,
                res.to(output_ptr.dtype.element_ty),
                boundary_check=(0, 1),
            )


def chunk_scaled_dot_kkt(k, beta, g_cumsum=None, chunk_size=64):
    batch, seqlen, num_k_heads, k_size = k.shape
    num_heads = beta.shape[-1]
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if seqlen % chunk_size:
        raise ValueError("seqlen must be divisible by chunk_size")
    if num_heads % num_k_heads:
        raise ValueError("num_heads must be divisible by num_k_heads")
    hpg = num_heads // num_k_heads
    nchunks = seqlen // chunk_size
    output = torch.empty(
        (batch, seqlen, num_heads, chunk_size),
        dtype=torch.float32,
        device=k.device,
    )
    if output.numel() == 0:
        return output

    if g_cumsum is None:
        g_cumsum = beta
    grid = (nchunks, batch)
    _kkt_vllmstyle_kernel[grid](
        k,
        beta,
        g_cumsum,
        output,
        seqlen,
        k_size,
        num_heads,
        hpg,
        K_POW2=triton.next_power_of_2(k_size),
        HAS_G=g_cumsum is not beta,
        BT=chunk_size,
        USE_INPUT_DTYPE=k.dtype in (torch.float16, torch.bfloat16),
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["chunk_scaled_dot_kkt"]
