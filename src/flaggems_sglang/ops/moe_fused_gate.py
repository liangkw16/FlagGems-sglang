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

_MAX_GRID = 65535


@triton.jit
def _moe_fused_gate_kernel(
    scores_ptr,
    bias_ptr,
    weights_ptr,
    indices_ptr,
    n_rows,
    n_experts,
    experts_per_group_rt,
    routed_scaling_factor,
    moe_softcapping,
    SCORING: tl.constexpr,
    K_ROUTED: tl.constexpr,
    TOPK: tl.constexpr,
    RENORM: tl.constexpr,
    SCALE_OUT: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    G_BLOCK: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    n_tiles = tl.cdiv(n_rows, 1)
    n_programs = tl.num_programs(0)
    experts = tl.arange(0, BLOCK_E)
    expert_mask = experts < n_experts
    for row in tl.range(tl.program_id(0), n_tiles, n_programs):
        scores = tl.load(
            scores_ptr + row * n_experts + experts,
            mask=expert_mask,
            other=0.0,
        ).to(tl.float32)
        bias = tl.load(bias_ptr + experts, mask=expert_mask, other=0.0).to(
            tl.float32
        )

        if SCORING == 0:  # sigmoid
            activated = tl.sigmoid(scores)
            biased = activated + bias
        elif SCORING == 1:  # sqrtsoftplus
            sp = tl.maximum(scores, 0.0) + tl.log(
                1.0 + tl.exp(-tl.abs(scores))
            )
            activated = tl.sqrt(sp)
            biased = activated + bias
        else:  # softmax with optional tanh softcap
            logits = scores
            if moe_softcapping != 0.0:
                scaled = logits / moe_softcapping
                scaled_sq = scaled * scaled
                near_zero = scaled * (
                    1.0 + scaled_sq * (-1.0 / 3.0 + scaled_sq * (2.0 / 15.0))
                )
                saturated = 2.0 / (1.0 + tl.exp(-2.0 * scaled)) - 1.0
                tanh_form = tl.where(
                    tl.abs(scaled) < 0.25, near_zero, saturated
                )
                logits = tanh_form * moe_softcapping
            biased = logits + bias
            best = tl.max(tl.where(expert_mask, biased, -float("inf")), axis=0)
            activated = tl.exp(biased - best)
            denom = tl.sum(tl.where(expert_mask, activated, 0.0), axis=0)
            activated = activated / denom

        biased = tl.where(expert_mask, biased, -float("inf"))

        if NUM_GROUPS > 1:
            # per-group top-2 via static_range over exact memory bounds
            # (reshape would misalign groups when epg is not pow2)
            g_idx = tl.arange(0, G_BLOCK)
            g_score = tl.full((G_BLOCK,), -float("inf"), dtype=tl.float32)
            for g in tl.static_range(NUM_GROUPS):
                lo = g * experts_per_group_rt
                hi = lo + experts_per_group_rt
                in_group = (experts >= lo) & (experts < hi)
                vals = tl.where(in_group, biased, -float("inf"))
                m1 = tl.max(vals, axis=0)
                m2 = tl.max(
                    tl.where(vals == m1, -float("inf"), vals),
                    axis=0,
                )
                g_score = tl.where(g_idx == g, m1 + m2, g_score)
            better = tl.sum(
                (g_score[None, :] > g_score[:, None]).to(tl.int32), axis=1
            )
            keep_flat = tl.zeros((BLOCK_E,), dtype=tl.int1)
            for g in tl.static_range(NUM_GROUPS):
                lo = g * experts_per_group_rt
                hi = lo + experts_per_group_rt
                in_group = (experts >= lo) & (experts < hi)
                keep_g = (
                    tl.sum(tl.where(g_idx == g, better, 0), axis=0)
                    < TOPK_GROUP
                )
                keep_flat |= in_group & keep_g
            biased = tl.where(keep_flat, biased, -float("inf"))

        selected = tl.zeros((BLOCK_E,), dtype=tl.int1)
        routed_sum = 0.0
        for slot in tl.static_range(K_ROUTED):
            cand = tl.where(selected, -float("inf"), biased)
            cand_value = tl.max(cand, axis=0)
            cand_index = tl.min(
                tl.where(cand == cand_value, experts, BLOCK_E), axis=0
            )
            picked = experts == cand_index
            selected |= picked
            act_val = tl.sum(tl.where(picked, activated, 0.0), axis=0)
            tl.store(weights_ptr + row * TOPK + slot, act_val)
            tl.store(indices_ptr + row * TOPK + slot, cand_index)
            routed_sum += act_val

        if TOPK > K_ROUTED:
            shared_weight = routed_sum / routed_scaling_factor
            for slot in tl.static_range(K_ROUTED, TOPK):
                tl.store(weights_ptr + row * TOPK + slot, shared_weight)
                tl.store(
                    indices_ptr + row * TOPK + slot,
                    n_experts + (slot - K_ROUTED),
                )

        if RENORM:
            norm = tl.where(routed_sum > 0.0, routed_sum, 1.0)
            for slot in tl.static_range(TOPK):
                w = tl.load(weights_ptr + row * TOPK + slot)
                tl.store(weights_ptr + row * TOPK + slot, w / norm)
        if SCALE_OUT:
            for slot in tl.static_range(TOPK):
                w = tl.load(weights_ptr + row * TOPK + slot)
                tl.store(
                    weights_ptr + row * TOPK + slot,
                    w * routed_scaling_factor,
                )


def moe_fused_gate(
    scores,
    bias,
    topk,
    scoring_func="sigmoid",
    num_fused_shared_experts=0,
    renormalize=True,
    routed_scaling_factor=1.0,
    apply_routed_scaling_factor_on_output=False,
    moe_softcapping=0.0,
    num_expert_group=1,
    topk_group=1,
):
    scores = scores.contiguous()
    bias = bias.contiguous()
    n_rows, n_experts = scores.shape
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0
    scoring = {
        "sigmoid": 0,
        "sqrtsoftplus": 1,
        "softmax": 2,
    }.get(scoring_func, 2)
    k_routed = topk - num_fused_shared_experts
    weights = torch.empty(
        (n_rows, topk), dtype=torch.float32, device=scores.device
    )
    indices = torch.empty(
        (n_rows, topk), dtype=torch.int32, device=scores.device
    )
    if n_rows == 0 or n_experts == 0:
        return weights, indices
    groups = num_expert_group
    experts_per_group = n_experts // groups if groups > 1 else 1
    block_e = triton.next_power_of_2(n_experts)
    g_block = triton.next_power_of_2(max(groups, 1))
    grid = (min(n_rows, _MAX_GRID),)
    _moe_fused_gate_kernel[grid](
        scores,
        bias,
        weights,
        indices,
        n_rows,
        n_experts,
        experts_per_group,
        float(routed_scaling_factor),
        float(moe_softcapping),
        SCORING=scoring,
        K_ROUTED=k_routed,
        TOPK=topk,
        RENORM=bool(renormalize),
        SCALE_OUT=bool(apply_routed_scaling_factor_on_output),
        NUM_GROUPS=groups,
        TOPK_GROUP=topk_group,
        G_BLOCK=g_block,
        BLOCK_E=block_e,
    )
    return weights, indices


__all__ = ["moe_fused_gate"]
