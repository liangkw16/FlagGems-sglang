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


# Ascend E9 candidate: fixed adjacent-pair reduction tree; elementwise products
# are rounded before any addition.
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

        for key_offset in tl.static_range(0, 64):
            decays = tl.load(state_ptr + state_base + key_offset) * decay
            tl.store(state_ptr + state_base + key_offset, decays)
        factor_0 = tl.load(state_ptr + state_base + 0) * tl.load(
            k_ptr + key_base + 0 * stride_k_dim
        ).to(tl.float32)
        factor_1 = tl.load(state_ptr + state_base + 1) * tl.load(
            k_ptr + key_base + 1 * stride_k_dim
        ).to(tl.float32)
        factor_2 = tl.load(state_ptr + state_base + 2) * tl.load(
            k_ptr + key_base + 2 * stride_k_dim
        ).to(tl.float32)
        factor_3 = tl.load(state_ptr + state_base + 3) * tl.load(
            k_ptr + key_base + 3 * stride_k_dim
        ).to(tl.float32)
        factor_4 = tl.load(state_ptr + state_base + 4) * tl.load(
            k_ptr + key_base + 4 * stride_k_dim
        ).to(tl.float32)
        factor_5 = tl.load(state_ptr + state_base + 5) * tl.load(
            k_ptr + key_base + 5 * stride_k_dim
        ).to(tl.float32)
        factor_6 = tl.load(state_ptr + state_base + 6) * tl.load(
            k_ptr + key_base + 6 * stride_k_dim
        ).to(tl.float32)
        factor_7 = tl.load(state_ptr + state_base + 7) * tl.load(
            k_ptr + key_base + 7 * stride_k_dim
        ).to(tl.float32)
        factor_8 = tl.load(state_ptr + state_base + 8) * tl.load(
            k_ptr + key_base + 8 * stride_k_dim
        ).to(tl.float32)
        factor_9 = tl.load(state_ptr + state_base + 9) * tl.load(
            k_ptr + key_base + 9 * stride_k_dim
        ).to(tl.float32)
        factor_10 = tl.load(state_ptr + state_base + 10) * tl.load(
            k_ptr + key_base + 10 * stride_k_dim
        ).to(tl.float32)
        factor_11 = tl.load(state_ptr + state_base + 11) * tl.load(
            k_ptr + key_base + 11 * stride_k_dim
        ).to(tl.float32)
        factor_12 = tl.load(state_ptr + state_base + 12) * tl.load(
            k_ptr + key_base + 12 * stride_k_dim
        ).to(tl.float32)
        factor_13 = tl.load(state_ptr + state_base + 13) * tl.load(
            k_ptr + key_base + 13 * stride_k_dim
        ).to(tl.float32)
        factor_14 = tl.load(state_ptr + state_base + 14) * tl.load(
            k_ptr + key_base + 14 * stride_k_dim
        ).to(tl.float32)
        factor_15 = tl.load(state_ptr + state_base + 15) * tl.load(
            k_ptr + key_base + 15 * stride_k_dim
        ).to(tl.float32)
        factor_16 = tl.load(state_ptr + state_base + 16) * tl.load(
            k_ptr + key_base + 16 * stride_k_dim
        ).to(tl.float32)
        factor_17 = tl.load(state_ptr + state_base + 17) * tl.load(
            k_ptr + key_base + 17 * stride_k_dim
        ).to(tl.float32)
        factor_18 = tl.load(state_ptr + state_base + 18) * tl.load(
            k_ptr + key_base + 18 * stride_k_dim
        ).to(tl.float32)
        factor_19 = tl.load(state_ptr + state_base + 19) * tl.load(
            k_ptr + key_base + 19 * stride_k_dim
        ).to(tl.float32)
        factor_20 = tl.load(state_ptr + state_base + 20) * tl.load(
            k_ptr + key_base + 20 * stride_k_dim
        ).to(tl.float32)
        factor_21 = tl.load(state_ptr + state_base + 21) * tl.load(
            k_ptr + key_base + 21 * stride_k_dim
        ).to(tl.float32)
        factor_22 = tl.load(state_ptr + state_base + 22) * tl.load(
            k_ptr + key_base + 22 * stride_k_dim
        ).to(tl.float32)
        factor_23 = tl.load(state_ptr + state_base + 23) * tl.load(
            k_ptr + key_base + 23 * stride_k_dim
        ).to(tl.float32)
        factor_24 = tl.load(state_ptr + state_base + 24) * tl.load(
            k_ptr + key_base + 24 * stride_k_dim
        ).to(tl.float32)
        factor_25 = tl.load(state_ptr + state_base + 25) * tl.load(
            k_ptr + key_base + 25 * stride_k_dim
        ).to(tl.float32)
        factor_26 = tl.load(state_ptr + state_base + 26) * tl.load(
            k_ptr + key_base + 26 * stride_k_dim
        ).to(tl.float32)
        factor_27 = tl.load(state_ptr + state_base + 27) * tl.load(
            k_ptr + key_base + 27 * stride_k_dim
        ).to(tl.float32)
        factor_28 = tl.load(state_ptr + state_base + 28) * tl.load(
            k_ptr + key_base + 28 * stride_k_dim
        ).to(tl.float32)
        factor_29 = tl.load(state_ptr + state_base + 29) * tl.load(
            k_ptr + key_base + 29 * stride_k_dim
        ).to(tl.float32)
        factor_30 = tl.load(state_ptr + state_base + 30) * tl.load(
            k_ptr + key_base + 30 * stride_k_dim
        ).to(tl.float32)
        factor_31 = tl.load(state_ptr + state_base + 31) * tl.load(
            k_ptr + key_base + 31 * stride_k_dim
        ).to(tl.float32)
        factor_32 = tl.load(state_ptr + state_base + 32) * tl.load(
            k_ptr + key_base + 32 * stride_k_dim
        ).to(tl.float32)
        factor_33 = tl.load(state_ptr + state_base + 33) * tl.load(
            k_ptr + key_base + 33 * stride_k_dim
        ).to(tl.float32)
        factor_34 = tl.load(state_ptr + state_base + 34) * tl.load(
            k_ptr + key_base + 34 * stride_k_dim
        ).to(tl.float32)
        factor_35 = tl.load(state_ptr + state_base + 35) * tl.load(
            k_ptr + key_base + 35 * stride_k_dim
        ).to(tl.float32)
        factor_36 = tl.load(state_ptr + state_base + 36) * tl.load(
            k_ptr + key_base + 36 * stride_k_dim
        ).to(tl.float32)
        factor_37 = tl.load(state_ptr + state_base + 37) * tl.load(
            k_ptr + key_base + 37 * stride_k_dim
        ).to(tl.float32)
        factor_38 = tl.load(state_ptr + state_base + 38) * tl.load(
            k_ptr + key_base + 38 * stride_k_dim
        ).to(tl.float32)
        factor_39 = tl.load(state_ptr + state_base + 39) * tl.load(
            k_ptr + key_base + 39 * stride_k_dim
        ).to(tl.float32)
        factor_40 = tl.load(state_ptr + state_base + 40) * tl.load(
            k_ptr + key_base + 40 * stride_k_dim
        ).to(tl.float32)
        factor_41 = tl.load(state_ptr + state_base + 41) * tl.load(
            k_ptr + key_base + 41 * stride_k_dim
        ).to(tl.float32)
        factor_42 = tl.load(state_ptr + state_base + 42) * tl.load(
            k_ptr + key_base + 42 * stride_k_dim
        ).to(tl.float32)
        factor_43 = tl.load(state_ptr + state_base + 43) * tl.load(
            k_ptr + key_base + 43 * stride_k_dim
        ).to(tl.float32)
        factor_44 = tl.load(state_ptr + state_base + 44) * tl.load(
            k_ptr + key_base + 44 * stride_k_dim
        ).to(tl.float32)
        factor_45 = tl.load(state_ptr + state_base + 45) * tl.load(
            k_ptr + key_base + 45 * stride_k_dim
        ).to(tl.float32)
        factor_46 = tl.load(state_ptr + state_base + 46) * tl.load(
            k_ptr + key_base + 46 * stride_k_dim
        ).to(tl.float32)
        factor_47 = tl.load(state_ptr + state_base + 47) * tl.load(
            k_ptr + key_base + 47 * stride_k_dim
        ).to(tl.float32)
        factor_48 = tl.load(state_ptr + state_base + 48) * tl.load(
            k_ptr + key_base + 48 * stride_k_dim
        ).to(tl.float32)
        factor_49 = tl.load(state_ptr + state_base + 49) * tl.load(
            k_ptr + key_base + 49 * stride_k_dim
        ).to(tl.float32)
        factor_50 = tl.load(state_ptr + state_base + 50) * tl.load(
            k_ptr + key_base + 50 * stride_k_dim
        ).to(tl.float32)
        factor_51 = tl.load(state_ptr + state_base + 51) * tl.load(
            k_ptr + key_base + 51 * stride_k_dim
        ).to(tl.float32)
        factor_52 = tl.load(state_ptr + state_base + 52) * tl.load(
            k_ptr + key_base + 52 * stride_k_dim
        ).to(tl.float32)
        factor_53 = tl.load(state_ptr + state_base + 53) * tl.load(
            k_ptr + key_base + 53 * stride_k_dim
        ).to(tl.float32)
        factor_54 = tl.load(state_ptr + state_base + 54) * tl.load(
            k_ptr + key_base + 54 * stride_k_dim
        ).to(tl.float32)
        factor_55 = tl.load(state_ptr + state_base + 55) * tl.load(
            k_ptr + key_base + 55 * stride_k_dim
        ).to(tl.float32)
        factor_56 = tl.load(state_ptr + state_base + 56) * tl.load(
            k_ptr + key_base + 56 * stride_k_dim
        ).to(tl.float32)
        factor_57 = tl.load(state_ptr + state_base + 57) * tl.load(
            k_ptr + key_base + 57 * stride_k_dim
        ).to(tl.float32)
        factor_58 = tl.load(state_ptr + state_base + 58) * tl.load(
            k_ptr + key_base + 58 * stride_k_dim
        ).to(tl.float32)
        factor_59 = tl.load(state_ptr + state_base + 59) * tl.load(
            k_ptr + key_base + 59 * stride_k_dim
        ).to(tl.float32)
        factor_60 = tl.load(state_ptr + state_base + 60) * tl.load(
            k_ptr + key_base + 60 * stride_k_dim
        ).to(tl.float32)
        factor_61 = tl.load(state_ptr + state_base + 61) * tl.load(
            k_ptr + key_base + 61 * stride_k_dim
        ).to(tl.float32)
        factor_62 = tl.load(state_ptr + state_base + 62) * tl.load(
            k_ptr + key_base + 62 * stride_k_dim
        ).to(tl.float32)
        factor_63 = tl.load(state_ptr + state_base + 63) * tl.load(
            k_ptr + key_base + 63 * stride_k_dim
        ).to(tl.float32)
        stage_0_0 = factor_0 + factor_1
        stage_0_1 = factor_2 + factor_3
        stage_0_2 = factor_4 + factor_5
        stage_0_3 = factor_6 + factor_7
        stage_0_4 = factor_8 + factor_9
        stage_0_5 = factor_10 + factor_11
        stage_0_6 = factor_12 + factor_13
        stage_0_7 = factor_14 + factor_15
        stage_0_8 = factor_16 + factor_17
        stage_0_9 = factor_18 + factor_19
        stage_0_10 = factor_20 + factor_21
        stage_0_11 = factor_22 + factor_23
        stage_0_12 = factor_24 + factor_25
        stage_0_13 = factor_26 + factor_27
        stage_0_14 = factor_28 + factor_29
        stage_0_15 = factor_30 + factor_31
        stage_0_16 = factor_32 + factor_33
        stage_0_17 = factor_34 + factor_35
        stage_0_18 = factor_36 + factor_37
        stage_0_19 = factor_38 + factor_39
        stage_0_20 = factor_40 + factor_41
        stage_0_21 = factor_42 + factor_43
        stage_0_22 = factor_44 + factor_45
        stage_0_23 = factor_46 + factor_47
        stage_0_24 = factor_48 + factor_49
        stage_0_25 = factor_50 + factor_51
        stage_0_26 = factor_52 + factor_53
        stage_0_27 = factor_54 + factor_55
        stage_0_28 = factor_56 + factor_57
        stage_0_29 = factor_58 + factor_59
        stage_0_30 = factor_60 + factor_61
        stage_0_31 = factor_62 + factor_63
        stage_1_0 = stage_0_0 + stage_0_1
        stage_1_1 = stage_0_2 + stage_0_3
        stage_1_2 = stage_0_4 + stage_0_5
        stage_1_3 = stage_0_6 + stage_0_7
        stage_1_4 = stage_0_8 + stage_0_9
        stage_1_5 = stage_0_10 + stage_0_11
        stage_1_6 = stage_0_12 + stage_0_13
        stage_1_7 = stage_0_14 + stage_0_15
        stage_1_8 = stage_0_16 + stage_0_17
        stage_1_9 = stage_0_18 + stage_0_19
        stage_1_10 = stage_0_20 + stage_0_21
        stage_1_11 = stage_0_22 + stage_0_23
        stage_1_12 = stage_0_24 + stage_0_25
        stage_1_13 = stage_0_26 + stage_0_27
        stage_1_14 = stage_0_28 + stage_0_29
        stage_1_15 = stage_0_30 + stage_0_31
        stage_2_0 = stage_1_0 + stage_1_1
        stage_2_1 = stage_1_2 + stage_1_3
        stage_2_2 = stage_1_4 + stage_1_5
        stage_2_3 = stage_1_6 + stage_1_7
        stage_2_4 = stage_1_8 + stage_1_9
        stage_2_5 = stage_1_10 + stage_1_11
        stage_2_6 = stage_1_12 + stage_1_13
        stage_2_7 = stage_1_14 + stage_1_15
        stage_3_0 = stage_2_0 + stage_2_1
        stage_3_1 = stage_2_2 + stage_2_3
        stage_3_2 = stage_2_4 + stage_2_5
        stage_3_3 = stage_2_6 + stage_2_7
        stage_4_0 = stage_3_0 + stage_3_1
        stage_4_1 = stage_3_2 + stage_3_3
        stage_5_0 = stage_4_0 + stage_4_1
        prediction = stage_5_0

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
            k_val = tl.load(k_ptr + key_base + key_offset * stride_k_dim).to(
                tl.float32
            )
            tl.store(
                outer_ptr + state_base + key_offset,
                correction * k_val,
            )
        for key_offset in tl.static_range(0, 64):
            merged = tl.load(state_ptr + state_base + key_offset) + tl.load(
                outer_ptr + state_base + key_offset
            )
            tl.store(state_ptr + state_base + key_offset, merged)
        factor_0 = tl.load(state_ptr + state_base + 0) * (
            tl.load(q_ptr + query_base + 0 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_1 = tl.load(state_ptr + state_base + 1) * (
            tl.load(q_ptr + query_base + 1 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_2 = tl.load(state_ptr + state_base + 2) * (
            tl.load(q_ptr + query_base + 2 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_3 = tl.load(state_ptr + state_base + 3) * (
            tl.load(q_ptr + query_base + 3 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_4 = tl.load(state_ptr + state_base + 4) * (
            tl.load(q_ptr + query_base + 4 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_5 = tl.load(state_ptr + state_base + 5) * (
            tl.load(q_ptr + query_base + 5 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_6 = tl.load(state_ptr + state_base + 6) * (
            tl.load(q_ptr + query_base + 6 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_7 = tl.load(state_ptr + state_base + 7) * (
            tl.load(q_ptr + query_base + 7 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_8 = tl.load(state_ptr + state_base + 8) * (
            tl.load(q_ptr + query_base + 8 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_9 = tl.load(state_ptr + state_base + 9) * (
            tl.load(q_ptr + query_base + 9 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_10 = tl.load(state_ptr + state_base + 10) * (
            tl.load(q_ptr + query_base + 10 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_11 = tl.load(state_ptr + state_base + 11) * (
            tl.load(q_ptr + query_base + 11 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_12 = tl.load(state_ptr + state_base + 12) * (
            tl.load(q_ptr + query_base + 12 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_13 = tl.load(state_ptr + state_base + 13) * (
            tl.load(q_ptr + query_base + 13 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_14 = tl.load(state_ptr + state_base + 14) * (
            tl.load(q_ptr + query_base + 14 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_15 = tl.load(state_ptr + state_base + 15) * (
            tl.load(q_ptr + query_base + 15 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_16 = tl.load(state_ptr + state_base + 16) * (
            tl.load(q_ptr + query_base + 16 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_17 = tl.load(state_ptr + state_base + 17) * (
            tl.load(q_ptr + query_base + 17 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_18 = tl.load(state_ptr + state_base + 18) * (
            tl.load(q_ptr + query_base + 18 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_19 = tl.load(state_ptr + state_base + 19) * (
            tl.load(q_ptr + query_base + 19 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_20 = tl.load(state_ptr + state_base + 20) * (
            tl.load(q_ptr + query_base + 20 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_21 = tl.load(state_ptr + state_base + 21) * (
            tl.load(q_ptr + query_base + 21 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_22 = tl.load(state_ptr + state_base + 22) * (
            tl.load(q_ptr + query_base + 22 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_23 = tl.load(state_ptr + state_base + 23) * (
            tl.load(q_ptr + query_base + 23 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_24 = tl.load(state_ptr + state_base + 24) * (
            tl.load(q_ptr + query_base + 24 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_25 = tl.load(state_ptr + state_base + 25) * (
            tl.load(q_ptr + query_base + 25 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_26 = tl.load(state_ptr + state_base + 26) * (
            tl.load(q_ptr + query_base + 26 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_27 = tl.load(state_ptr + state_base + 27) * (
            tl.load(q_ptr + query_base + 27 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_28 = tl.load(state_ptr + state_base + 28) * (
            tl.load(q_ptr + query_base + 28 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_29 = tl.load(state_ptr + state_base + 29) * (
            tl.load(q_ptr + query_base + 29 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_30 = tl.load(state_ptr + state_base + 30) * (
            tl.load(q_ptr + query_base + 30 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_31 = tl.load(state_ptr + state_base + 31) * (
            tl.load(q_ptr + query_base + 31 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_32 = tl.load(state_ptr + state_base + 32) * (
            tl.load(q_ptr + query_base + 32 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_33 = tl.load(state_ptr + state_base + 33) * (
            tl.load(q_ptr + query_base + 33 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_34 = tl.load(state_ptr + state_base + 34) * (
            tl.load(q_ptr + query_base + 34 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_35 = tl.load(state_ptr + state_base + 35) * (
            tl.load(q_ptr + query_base + 35 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_36 = tl.load(state_ptr + state_base + 36) * (
            tl.load(q_ptr + query_base + 36 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_37 = tl.load(state_ptr + state_base + 37) * (
            tl.load(q_ptr + query_base + 37 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_38 = tl.load(state_ptr + state_base + 38) * (
            tl.load(q_ptr + query_base + 38 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_39 = tl.load(state_ptr + state_base + 39) * (
            tl.load(q_ptr + query_base + 39 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_40 = tl.load(state_ptr + state_base + 40) * (
            tl.load(q_ptr + query_base + 40 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_41 = tl.load(state_ptr + state_base + 41) * (
            tl.load(q_ptr + query_base + 41 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_42 = tl.load(state_ptr + state_base + 42) * (
            tl.load(q_ptr + query_base + 42 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_43 = tl.load(state_ptr + state_base + 43) * (
            tl.load(q_ptr + query_base + 43 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_44 = tl.load(state_ptr + state_base + 44) * (
            tl.load(q_ptr + query_base + 44 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_45 = tl.load(state_ptr + state_base + 45) * (
            tl.load(q_ptr + query_base + 45 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_46 = tl.load(state_ptr + state_base + 46) * (
            tl.load(q_ptr + query_base + 46 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_47 = tl.load(state_ptr + state_base + 47) * (
            tl.load(q_ptr + query_base + 47 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_48 = tl.load(state_ptr + state_base + 48) * (
            tl.load(q_ptr + query_base + 48 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_49 = tl.load(state_ptr + state_base + 49) * (
            tl.load(q_ptr + query_base + 49 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_50 = tl.load(state_ptr + state_base + 50) * (
            tl.load(q_ptr + query_base + 50 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_51 = tl.load(state_ptr + state_base + 51) * (
            tl.load(q_ptr + query_base + 51 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_52 = tl.load(state_ptr + state_base + 52) * (
            tl.load(q_ptr + query_base + 52 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_53 = tl.load(state_ptr + state_base + 53) * (
            tl.load(q_ptr + query_base + 53 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_54 = tl.load(state_ptr + state_base + 54) * (
            tl.load(q_ptr + query_base + 54 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_55 = tl.load(state_ptr + state_base + 55) * (
            tl.load(q_ptr + query_base + 55 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_56 = tl.load(state_ptr + state_base + 56) * (
            tl.load(q_ptr + query_base + 56 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_57 = tl.load(state_ptr + state_base + 57) * (
            tl.load(q_ptr + query_base + 57 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_58 = tl.load(state_ptr + state_base + 58) * (
            tl.load(q_ptr + query_base + 58 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_59 = tl.load(state_ptr + state_base + 59) * (
            tl.load(q_ptr + query_base + 59 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_60 = tl.load(state_ptr + state_base + 60) * (
            tl.load(q_ptr + query_base + 60 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_61 = tl.load(state_ptr + state_base + 61) * (
            tl.load(q_ptr + query_base + 61 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_62 = tl.load(state_ptr + state_base + 62) * (
            tl.load(q_ptr + query_base + 62 * stride_q_dim).to(tl.float32)
            * scale
        )
        factor_63 = tl.load(state_ptr + state_base + 63) * (
            tl.load(q_ptr + query_base + 63 * stride_q_dim).to(tl.float32)
            * scale
        )
        stage_0_0 = factor_0 + factor_1
        stage_0_1 = factor_2 + factor_3
        stage_0_2 = factor_4 + factor_5
        stage_0_3 = factor_6 + factor_7
        stage_0_4 = factor_8 + factor_9
        stage_0_5 = factor_10 + factor_11
        stage_0_6 = factor_12 + factor_13
        stage_0_7 = factor_14 + factor_15
        stage_0_8 = factor_16 + factor_17
        stage_0_9 = factor_18 + factor_19
        stage_0_10 = factor_20 + factor_21
        stage_0_11 = factor_22 + factor_23
        stage_0_12 = factor_24 + factor_25
        stage_0_13 = factor_26 + factor_27
        stage_0_14 = factor_28 + factor_29
        stage_0_15 = factor_30 + factor_31
        stage_0_16 = factor_32 + factor_33
        stage_0_17 = factor_34 + factor_35
        stage_0_18 = factor_36 + factor_37
        stage_0_19 = factor_38 + factor_39
        stage_0_20 = factor_40 + factor_41
        stage_0_21 = factor_42 + factor_43
        stage_0_22 = factor_44 + factor_45
        stage_0_23 = factor_46 + factor_47
        stage_0_24 = factor_48 + factor_49
        stage_0_25 = factor_50 + factor_51
        stage_0_26 = factor_52 + factor_53
        stage_0_27 = factor_54 + factor_55
        stage_0_28 = factor_56 + factor_57
        stage_0_29 = factor_58 + factor_59
        stage_0_30 = factor_60 + factor_61
        stage_0_31 = factor_62 + factor_63
        stage_1_0 = stage_0_0 + stage_0_1
        stage_1_1 = stage_0_2 + stage_0_3
        stage_1_2 = stage_0_4 + stage_0_5
        stage_1_3 = stage_0_6 + stage_0_7
        stage_1_4 = stage_0_8 + stage_0_9
        stage_1_5 = stage_0_10 + stage_0_11
        stage_1_6 = stage_0_12 + stage_0_13
        stage_1_7 = stage_0_14 + stage_0_15
        stage_1_8 = stage_0_16 + stage_0_17
        stage_1_9 = stage_0_18 + stage_0_19
        stage_1_10 = stage_0_20 + stage_0_21
        stage_1_11 = stage_0_22 + stage_0_23
        stage_1_12 = stage_0_24 + stage_0_25
        stage_1_13 = stage_0_26 + stage_0_27
        stage_1_14 = stage_0_28 + stage_0_29
        stage_1_15 = stage_0_30 + stage_0_31
        stage_2_0 = stage_1_0 + stage_1_1
        stage_2_1 = stage_1_2 + stage_1_3
        stage_2_2 = stage_1_4 + stage_1_5
        stage_2_3 = stage_1_6 + stage_1_7
        stage_2_4 = stage_1_8 + stage_1_9
        stage_2_5 = stage_1_10 + stage_1_11
        stage_2_6 = stage_1_12 + stage_1_13
        stage_2_7 = stage_1_14 + stage_1_15
        stage_3_0 = stage_2_0 + stage_2_1
        stage_3_1 = stage_2_2 + stage_2_3
        stage_3_2 = stage_2_4 + stage_2_5
        stage_3_3 = stage_2_6 + stage_2_7
        stage_4_0 = stage_3_0 + stage_3_1
        stage_4_1 = stage_3_2 + stage_3_3
        stage_5_0 = stage_4_0 + stage_4_1
        result = stage_5_0
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
