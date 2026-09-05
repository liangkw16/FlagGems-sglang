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

# Ascend vendor: two-pass attention (first pass computes all scores
# and the max/sum, second pass accumulates V). Numerically matches
# the reference's direct softmax more closely than online softmax.

import torch
import triton
import triton.language as tl


@triton.jit
def _extend_attn_2pass(
    q_ptr,
    kext_ptr,
    vext_ptr,
    kb_ptr,
    vb_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    batch_ids_ptr,
    o_ptr,
    scale,
    H_Q,
    H_KV,
    GROUP_SIZE,
    head_size,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    global_q_idx = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // GROUP_SIZE

    b = tl.load(batch_ids_ptr + global_q_idx)
    q_start = tl.load(qo_indptr_ptr + b)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)
    prefix_len = kv_end - kv_start
    q_offset = global_q_idx - q_start
    total_len = prefix_len + tl.load(qo_indptr_ptr + b + 1) - q_start

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_size

    q = tl.load(
        q_ptr + global_q_idx * (H_Q * head_size) + q_head * head_size + offs_d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    # Pass 1: find max score
    m_val = -1e30
    for n_start in range(0, total_len, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_valid = offs_n < total_len
        is_prefix = offs_n < prefix_len
        ext_pos = offs_n - prefix_len
        ext_valid = (ext_pos >= 0) & (ext_pos < total_len - prefix_len)

        prefix_idx = tl.load(
            kv_indices_ptr + kv_start + offs_n,
            mask=is_prefix & n_valid,
            other=0,
        )
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

        score = tl.sum(k * q[None, :], axis=1) * scale
        causal_ok = offs_n <= (prefix_len + q_offset)
        score = tl.where(causal_ok & n_valid, score, -1e30)
        m_val = tl.maximum(m_val, tl.max(score, axis=0))

    # Pass 2: compute sum of exp and accumulate V
    l_val = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for n_start in range(0, total_len, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_valid = offs_n < total_len
        is_prefix = offs_n < prefix_len
        ext_pos = offs_n - prefix_len
        ext_valid = (ext_pos >= 0) & (ext_pos < total_len - prefix_len)

        prefix_idx = tl.load(
            kv_indices_ptr + kv_start + offs_n,
            mask=is_prefix & n_valid,
            other=0,
        )
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

        score = tl.sum(k * q[None, :], axis=1) * scale
        causal_ok = offs_n <= (prefix_len + q_offset)
        p = tl.exp(score - m_val)
        p = tl.where(causal_ok & n_valid, p, 0.0)
        l_val += tl.sum(p, axis=0)

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

        acc += tl.sum(p[:, None] * v, axis=0)

    result = acc / l_val
    tl.store(
        o_ptr + global_q_idx * (H_Q * head_size) + q_head * head_size + offs_d,
        result.to(o_ptr.dtype.element_ty),
        mask=d_mask,
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

    # Force fp32 for Ascend numerical stability
    q_extend = q_extend.contiguous().float()
    k_extend = k_extend.contiguous().float()
    v_extend = v_extend.contiguous().float()
    k_buffer = k_buffer.contiguous().float()
    v_buffer = v_buffer.contiguous().float()

    # Precompute batch assignment
    batch_ids = torch.zeros(E, dtype=torch.int32, device=q_extend.device)
    for b in range(B):
        qs = int(qo_indptr[b].item())
        qe = int(qo_indptr[b + 1].item())
        batch_ids[qs:qe] = b

    BLOCK_N = 64
    BLOCK_D = max(triton.next_power_of_2(D), 16)
    grid = (E, H_Q)
    _extend_attn_2pass[grid](
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        batch_ids,
        o,
        float(scale),
        H_Q,
        H_KV,
        group_size,
        D,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=1,
    )
    return o


__all__ = ["extend_attention"]
