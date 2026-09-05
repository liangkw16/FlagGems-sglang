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
def _extend_attn_kernel(
    q_ptr,
    kext_ptr,
    vext_ptr,
    kb_ptr,
    vb_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    o_ptr,
    head_size,
    scale,
    H_KV: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_id = tl.program_id(0)
    q_head = tl.program_id(1)
    m_block = tl.program_id(2)

    kv_head = q_head // GROUP_SIZE

    q_start = tl.load(qo_indptr_ptr + batch_id)
    q_end = tl.load(qo_indptr_ptr + batch_id + 1)
    extend_len = q_end - q_start
    if m_block * BLOCK_M >= extend_len:
        return

    kv_start = tl.load(kv_indptr_ptr + batch_id)
    kv_end = tl.load(kv_indptr_ptr + batch_id + 1)
    prefix_len = kv_end - kv_start
    total_len = prefix_len + extend_len

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < extend_len
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_size

    q = tl.load(
        q_ptr
        + (q_start + offs_m[:, None]) * (H_KV * GROUP_SIZE * head_size)
        + q_head * head_size
        + offs_d[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    m_i = tl.full((BLOCK_M,), -1e30, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for n_start in range(0, total_len, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_valid = offs_n < total_len
        is_prefix = offs_n < prefix_len
        ext_pos = offs_n - prefix_len
        ext_valid = (ext_pos >= 0) & (ext_pos < extend_len)

        prefix_idx = tl.load(
            kv_indices_ptr + kv_start + offs_n,
            mask=is_prefix & n_valid,
            other=0,
        )

        # K: from buffer (prefix) or extend tensor
        k_buf = tl.load(
            kb_ptr
            + prefix_idx[:, None] * (H_KV * head_size)
            + kv_head * head_size
            + offs_d[None, :],
            mask=is_prefix[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        k_ext = tl.load(
            kext_ptr
            + (q_start + ext_pos)[:, None] * (H_KV * head_size)
            + kv_head * head_size
            + offs_d[None, :],
            mask=ext_valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        k = tl.where(is_prefix[:, None], k_buf, k_ext)

        # Scores [M, N]
        qk = tl.dot(q, tl.trans(k), input_precision="ieee") * scale

        # Causal: key_pos <= prefix_len + query_offset
        causal_ok = offs_n[None, :] <= (prefix_len + offs_m[:, None])
        qk = tl.where(causal_ok & n_valid[None, :] & m_mask[:, None], qk, -1e30)

        # Online softmax
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        p = tl.where(n_valid[None, :] & causal_ok, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # V: from buffer or extend
        v_buf = tl.load(
            vb_ptr
            + prefix_idx[:, None] * (H_KV * head_size)
            + kv_head * head_size
            + offs_d[None, :],
            mask=is_prefix[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v_ext = tl.load(
            vext_ptr
            + (q_start + ext_pos)[:, None] * (H_KV * head_size)
            + kv_head * head_size
            + offs_d[None, :],
            mask=ext_valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v = tl.where(is_prefix[:, None], v_buf, v_ext)

        acc += tl.dot(p.to(tl.float32), v, input_precision="ieee")

    # Final: divide by l_i (outside the loop!)
    acc = acc / l_i[:, None]

    tl.store(
        o_ptr
        + (q_start + offs_m[:, None]) * (H_KV * GROUP_SIZE * head_size)
        + q_head * head_size
        + offs_d[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=m_mask[:, None] & d_mask[None, :],
    )


def extend_attention(
    q_extend,
    k_extend,
    v_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    max_len_extend,
):
    E, H_Q, D = q_extend.shape
    H_KV = k_extend.shape[1]
    group_size = H_Q // H_KV
    scale = 1.0 / D**0.5

    o = torch.empty(E, H_Q, D, dtype=torch.float32, device=q_extend.device)
    if E == 0 or H_Q == 0:
        return o

    B = qo_indptr.size(0) - 1
    if B == 0:
        return o

    q_extend = q_extend.contiguous()
    k_extend = k_extend.contiguous()
    v_extend = v_extend.contiguous()
    k_buffer = k_buffer.contiguous()
    v_buffer = v_buffer.contiguous()
    qo_indptr = qo_indptr.contiguous()
    kv_indptr = kv_indptr.contiguous()
    kv_indices = kv_indices.contiguous()

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = max(triton.next_power_of_2(D), 16)

    grid = (B, H_Q, triton.cdiv(max_len_extend, BLOCK_M))
    _extend_attn_kernel[grid](
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        o,
        D,
        float(scale),
        H_KV=H_KV,
        GROUP_SIZE=group_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=1,
    )
    return o


__all__ = ["extend_attention"]
