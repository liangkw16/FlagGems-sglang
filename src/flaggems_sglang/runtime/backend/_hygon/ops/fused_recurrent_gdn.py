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

try:
    from flaggems_sglang.utils.triton_lang_helper import tl_extra_shim
except ImportError:
    from triton.language.extra import libdevice as tl_extra_shim


# Hygon E10 candidate: fixed serial accumulation order over k = 0..63 with
# multiply and add rounded separately.
@triton.jit(do_not_specialize=["sequence_length"])
def _fused_recurrent_gdn_k64_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    initial_state_ptr,
    state_ptr,
    outer_ptr,
    output_ptr,
    scale,
    sequence_length,
    value_heads,
    value_dim,
    head_group_ratio,
    stride_q_batch,
    stride_q_time,
    stride_q_head,
    stride_q_dim,
    stride_k_batch,
    stride_k_time,
    stride_k_head,
    stride_k_dim,
    stride_v_batch,
    stride_v_time,
    stride_v_head,
    stride_v_dim,
    stride_g_batch,
    stride_g_time,
    stride_g_head,
    stride_beta_batch,
    stride_beta_time,
    stride_beta_head,
    stride_beta_value,
    stride_initial_batch,
    stride_initial_head,
    stride_initial_value,
    stride_initial_key,
    stride_output_batch,
    stride_output_time,
    stride_output_head,
    stride_output_value,
    HAS_INITIAL_STATE: tl.constexpr,
    BETA_IS_VECTOR: tl.constexpr,
):
    program = tl.program_id(0)
    value_offset = program % value_dim
    batch_head = program // value_dim
    batch = batch_head // value_heads
    value_head = batch_head % value_heads
    query_head = value_head // head_group_ratio
    state_base = program * 64

    key_offsets = tl.arange(0, 64)
    state = tl.zeros((64,), dtype=tl.float32)
    if HAS_INITIAL_STATE:
        initial_offsets = (
            batch * stride_initial_batch
            + value_head * stride_initial_head
            + value_offset * stride_initial_value
            + key_offsets * stride_initial_key
        )
        state += tl.load(initial_state_ptr + initial_offsets).to(tl.float32)
    tl.store(state_ptr + state_base + key_offsets, state)

    for timestep in range(0, sequence_length):
        gate_offset = (
            batch * stride_g_batch
            + timestep * stride_g_time
            + value_head * stride_g_head
        )
        decay = tl_extra_shim.exp(tl.load(g_ptr + gate_offset).to(tl.float32))
        key_base = (
            batch * stride_k_batch
            + timestep * stride_k_time
            + query_head * stride_k_head
        )

        accumulation = 0.0
        for key_offset in tl.static_range(0, 64):
            lane_0 = tl.load(state_ptr + state_base + key_offset) * decay
            tl.store(state_ptr + state_base + key_offset, lane_0)
            key_0 = tl.load(k_ptr + key_base + key_offset * stride_k_dim).to(
                tl.float32
            )
            product = lane_0 * key_0
            accumulation = accumulation + product
        prediction = accumulation

        value_address = (
            batch * stride_v_batch
            + timestep * stride_v_time
            + value_head * stride_v_head
            + value_offset * stride_v_dim
        )
        value = tl.load(v_ptr + value_address).to(tl.float32)
        beta_address = (
            batch * stride_beta_batch
            + timestep * stride_beta_time
            + value_head * stride_beta_head
        )
        if BETA_IS_VECTOR:
            beta_address += value_offset * stride_beta_value
        beta_value = tl.load(beta_ptr + beta_address).to(tl.float32)
        correction = (value - prediction) * beta_value

        query_base = (
            batch * stride_q_batch
            + timestep * stride_q_time
            + query_head * stride_q_head
        )
        for key_offset in tl.static_range(0, 64):
            key_0 = tl.load(k_ptr + key_base + key_offset * stride_k_dim).to(
                tl.float32
            )
            tl.store(
                outer_ptr + state_base + key_offset,
                correction * key_0,
            )
        accumulation = 0.0
        for key_offset in tl.static_range(0, 64):
            lane_0 = tl.load(state_ptr + state_base + key_offset) + tl.load(
                outer_ptr + state_base + key_offset
            )
            tl.store(state_ptr + state_base + key_offset, lane_0)
            query_0 = (
                tl.load(q_ptr + query_base + key_offset * stride_q_dim).to(
                    tl.float32
                )
                * scale
            )
            product = lane_0 * query_0
            accumulation = accumulation + product
        result = accumulation
        output_address = (
            batch * stride_output_batch
            + timestep * stride_output_time
            + value_head * stride_output_head
            + value_offset * stride_output_value
        )
        tl.store(output_ptr + output_address, result)


@triton.jit(do_not_specialize=["sequence_length"])
def _fused_recurrent_gdn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    initial_state_ptr,
    output_ptr,
    final_state_ptr,
    scale,
    sequence_length,
    value_heads,
    key_dim,
    value_dim,
    head_group_ratio,
    stride_q_batch,
    stride_q_time,
    stride_q_head,
    stride_q_dim,
    stride_k_batch,
    stride_k_time,
    stride_k_head,
    stride_k_dim,
    stride_v_batch,
    stride_v_time,
    stride_v_head,
    stride_v_dim,
    stride_g_batch,
    stride_g_time,
    stride_g_head,
    stride_beta_batch,
    stride_beta_time,
    stride_beta_head,
    stride_beta_value,
    stride_initial_batch,
    stride_initial_head,
    stride_initial_value,
    stride_initial_key,
    stride_output_batch,
    stride_output_time,
    stride_output_head,
    stride_output_value,
    stride_final_batch,
    stride_final_head,
    stride_final_value,
    stride_final_key,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    HAS_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    BETA_IS_VECTOR: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
):
    stride_q_batch = tl.cast(stride_q_batch, tl.int64)
    stride_q_time = tl.cast(stride_q_time, tl.int64)
    stride_q_head = tl.cast(stride_q_head, tl.int64)
    stride_q_dim = tl.cast(stride_q_dim, tl.int64)
    stride_k_batch = tl.cast(stride_k_batch, tl.int64)
    stride_k_time = tl.cast(stride_k_time, tl.int64)
    stride_k_head = tl.cast(stride_k_head, tl.int64)
    stride_k_dim = tl.cast(stride_k_dim, tl.int64)
    stride_v_batch = tl.cast(stride_v_batch, tl.int64)
    stride_v_time = tl.cast(stride_v_time, tl.int64)
    stride_v_head = tl.cast(stride_v_head, tl.int64)
    stride_v_dim = tl.cast(stride_v_dim, tl.int64)
    stride_g_batch = tl.cast(stride_g_batch, tl.int64)
    stride_g_time = tl.cast(stride_g_time, tl.int64)
    stride_g_head = tl.cast(stride_g_head, tl.int64)
    stride_beta_batch = tl.cast(stride_beta_batch, tl.int64)
    stride_beta_time = tl.cast(stride_beta_time, tl.int64)
    stride_beta_head = tl.cast(stride_beta_head, tl.int64)
    stride_beta_value = tl.cast(stride_beta_value, tl.int64)
    stride_initial_batch = tl.cast(stride_initial_batch, tl.int64)
    stride_initial_head = tl.cast(stride_initial_head, tl.int64)
    stride_initial_value = tl.cast(stride_initial_value, tl.int64)
    stride_initial_key = tl.cast(stride_initial_key, tl.int64)
    stride_output_batch = tl.cast(stride_output_batch, tl.int64)
    stride_output_time = tl.cast(stride_output_time, tl.int64)
    stride_output_head = tl.cast(stride_output_head, tl.int64)
    stride_output_value = tl.cast(stride_output_value, tl.int64)
    stride_final_batch = tl.cast(stride_final_batch, tl.int64)
    stride_final_head = tl.cast(stride_final_head, tl.int64)
    stride_final_value = tl.cast(stride_final_value, tl.int64)
    stride_final_key = tl.cast(stride_final_key, tl.int64)

    value_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // value_heads
    value_head = batch_head % value_heads
    query_head = value_head // head_group_ratio

    key_offset = tl.arange(0, BLOCK_K)
    value_offset = value_tile * BLOCK_V + tl.arange(0, BLOCK_V)
    key_mask = key_offset < key_dim
    value_mask = value_offset < value_dim
    state_mask = value_mask[:, None] & key_mask[None, :]

    state = tl.zeros((BLOCK_V, BLOCK_K), dtype=tl.float32)
    if HAS_INITIAL_STATE:
        initial_offsets = (
            batch * stride_initial_batch
            + value_head * stride_initial_head
            + value_offset[:, None] * stride_initial_value
            + key_offset[None, :] * stride_initial_key
        )
        state += tl.load(
            initial_state_ptr + initial_offsets,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)

    for timestep in range(0, sequence_length):
        query_offsets = (
            batch * stride_q_batch
            + timestep * stride_q_time
            + query_head * stride_q_head
            + key_offset * stride_q_dim
        )
        key_offsets = (
            batch * stride_k_batch
            + timestep * stride_k_time
            + query_head * stride_k_head
            + key_offset * stride_k_dim
        )
        query = tl.load(q_ptr + query_offsets, mask=key_mask, other=0.0).to(
            tl.float32
        )
        key = tl.load(k_ptr + key_offsets, mask=key_mask, other=0.0).to(
            tl.float32
        )
        if USE_QK_L2NORM:
            query *= tl.rsqrt(tl.sum(query * query, axis=0) + 1e-6)
            key *= tl.rsqrt(tl.sum(key * key, axis=0) + 1e-6)
        query *= scale

        gate_offset = (
            batch * stride_g_batch
            + timestep * stride_g_time
            + value_head * stride_g_head
        )
        gate = tl.load(g_ptr + gate_offset).to(tl.float32)
        state *= tl.exp(gate)

        value_offsets = (
            batch * stride_v_batch
            + timestep * stride_v_time
            + value_head * stride_v_head
            + value_offset * stride_v_dim
        )
        value = tl.load(v_ptr + value_offsets, mask=value_mask, other=0.0).to(
            tl.float32
        )
        prediction = tl.sum(state * key[None, :], axis=1)
        correction = value - prediction

        beta_offset = (
            batch * stride_beta_batch
            + timestep * stride_beta_time
            + value_head * stride_beta_head
        )
        if BETA_IS_VECTOR:
            beta_value = tl.load(
                beta_ptr + beta_offset + value_offset * stride_beta_value,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            beta_value = tl.load(beta_ptr + beta_offset).to(tl.float32)
        correction *= beta_value

        state += correction[:, None] * key[None, :]
        result = tl.sum(state * query[None, :], axis=1)
        output_offsets = (
            batch * stride_output_batch
            + timestep * stride_output_time
            + value_head * stride_output_head
            + value_offset * stride_output_value
        )
        tl.store(output_ptr + output_offsets, result, mask=value_mask)

    if STORE_FINAL_STATE:
        final_offsets = (
            batch * stride_final_batch
            + value_head * stride_final_head
            + value_offset[:, None] * stride_final_value
            + key_offset[None, :] * stride_final_key
        )
        tl.store(final_state_ptr + final_offsets, state, mask=state_mask)


def fused_recurrent_gdn(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    batch, sequence_length, query_heads, key_dim = q.shape
    value_heads = v.shape[2]
    value_dim = v.shape[-1]
    if query_heads <= 0 or value_heads % query_heads != 0:
        raise ValueError(
            "value heads must be divisible by positive query heads"
        )
    if key_dim <= 0 or value_dim <= 0:
        raise ValueError("key and value dimensions must be positive")

    output = torch.empty(
        (batch, sequence_length, value_heads, value_dim),
        dtype=v.dtype,
        device=q.device,
    )
    final_state = None
    if output_final_state:
        final_state = torch.empty(
            (batch, value_heads, value_dim, key_dim),
            dtype=torch.float32,
            device=q.device,
        )
    if batch == 0 or (sequence_length == 0 and final_state is None):
        return output, final_state

    beta_is_vector = beta.ndim == v.ndim
    beta_stride = (*beta.stride(),) if beta_is_vector else (*beta.stride(), 0)
    initial_stride = (
        initial_state.stride() if initial_state is not None else (0,) * 4
    )
    final_stride = (
        final_state.stride() if final_state is not None else (0,) * 4
    )
    if (
        key_dim == 64
        and q.dtype == torch.bfloat16
        and not use_qk_l2norm_in_kernel
    ):
        state = final_state
        if state is None:
            state = torch.empty(
                (batch, value_heads, value_dim, key_dim),
                dtype=torch.float32,
                device=q.device,
            )
        outer = torch.empty_like(state)
        _fused_recurrent_gdn_k64_kernel[(batch * value_heads * value_dim,)](
            q,
            k,
            v,
            g,
            beta,
            initial_state if initial_state is not None else q,
            state,
            outer,
            output,
            float(scale),
            sequence_length,
            value_heads,
            value_dim,
            value_heads // query_heads,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *g.stride(),
            *beta_stride,
            *initial_stride,
            *output.stride(),
            HAS_INITIAL_STATE=initial_state is not None,
            BETA_IS_VECTOR=beta_is_vector,
            num_warps=1,
            num_stages=1,
        )
        return output, final_state
    block_k = triton.next_power_of_2(key_dim)
    block_v = min(triton.next_power_of_2(value_dim), 8)
    _fused_recurrent_gdn_kernel[
        (triton.cdiv(value_dim, block_v), batch * value_heads)
    ](
        q,
        k,
        v,
        g,
        beta,
        initial_state if initial_state is not None else q,
        output,
        final_state if final_state is not None else q,
        float(scale),
        sequence_length,
        value_heads,
        key_dim,
        value_dim,
        value_heads // query_heads,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *g.stride(),
        *beta_stride,
        *initial_stride,
        *output.stride(),
        *final_stride,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        HAS_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        BETA_IS_VECTOR=beta_is_vector,
        USE_QK_L2NORM=bool(use_qk_l2norm_in_kernel),
        num_warps=(
            1
            if block_k == 128 and q.dtype in (torch.float16, torch.bfloat16)
            else 4
        ),
        num_stages=1,
    )
    return output, final_state


__all__ = ["fused_recurrent_gdn"]
