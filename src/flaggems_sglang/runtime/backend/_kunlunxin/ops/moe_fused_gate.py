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
def _stage1_selector(
    scores_ptr,
    bias_ptr,
    sel_ptr,
    n_experts,
    row_start,
    moe_softcapping,
    SCORING: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = row_start + tl.program_id(0)
    experts = tl.arange(0, BLOCK_N)
    mask = experts < n_experts

    scores = tl.load(
        scores_ptr + row * n_experts + experts, mask=mask, other=0.0
    ).to(tl.float32)
    bias = tl.load(bias_ptr + experts, mask=mask, other=0.0).to(tl.float32)

    if SCORING == 0:
        selector = tl.sigmoid(scores) + bias
    elif SCORING == 1:
        u = tl.exp(-tl.abs(scores))
        z = u / (2.0 + u)
        z2 = z * z
        softplus = tl.maximum(scores, 0.0) + 2.0 * z * (
            1.0
            + z2
            * (
                1.0 / 3
                + z2
                * (1.0 / 5 + z2 * (1.0 / 7 + z2 * (1.0 / 9 + z2 * (1.0 / 11))))
            )
        )
        selector = tl.sqrt(softplus) + bias
    else:
        logits = scores
        if moe_softcapping != 0.0:
            scaled = logits / moe_softcapping
            scaled_sq = scaled * scaled
            near_zero = scaled * (
                1.0 + scaled_sq * (-1.0 / 3.0 + scaled_sq * (2.0 / 15.0))
            )
            saturated = 2.0 / (1.0 + tl.exp(-2.0 * scaled)) - 1.0
            logits = (
                tl.where(tl.abs(scaled) < 0.25, near_zero, saturated)
                * moe_softcapping
            )
        selector = logits + bias

    tl.store(sel_ptr + row * n_experts + experts, selector, mask=mask)


@triton.jit
def _stage2_route(
    sel_ptr,
    indices_ptr,
    n_experts,
    experts_per_group,
    row_start,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_G: tl.constexpr,
    K_ROUTED: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
):
    row = row_start + tl.program_id(0)
    experts = tl.arange(0, BLOCK_N)
    expert_mask = experts < n_experts
    selector = tl.load(
        sel_ptr + row * n_experts + experts,
        mask=expert_mask,
        other=-float("inf"),
    )

    if NUM_GROUPS > 1:
        group_ids = tl.arange(0, BLOCK_G)
        group_scores = tl.full((BLOCK_G,), -float("inf"), dtype=tl.float32)
        for group in tl.static_range(NUM_GROUPS):
            lo = group * experts_per_group
            hi = lo + experts_per_group
            in_group = (experts >= lo) & (experts < hi) & expert_mask
            values = tl.where(in_group, selector, -float("inf"))
            first_value = tl.max(values, axis=0)
            first_index = tl.min(
                tl.where(values == first_value, experts, n_experts), axis=0
            )
            second_value = tl.max(
                tl.where(experts == first_index, -float("inf"), values),
                axis=0,
            )
            group_scores = tl.where(
                group_ids == group,
                first_value + second_value,
                group_scores,
            )

        selected_groups = tl.zeros((BLOCK_G,), dtype=tl.int1)
        for _ in tl.static_range(TOPK_GROUP):
            available = tl.where(selected_groups, -float("inf"), group_scores)
            best_value = tl.max(available, axis=0)
            best_group = tl.min(
                tl.where(available == best_value, group_ids, BLOCK_G),
                axis=0,
            )
            selected_groups |= group_ids == best_group

        keep = tl.zeros((BLOCK_N,), dtype=tl.int1)
        for group in tl.static_range(NUM_GROUPS):
            lo = group * experts_per_group
            hi = lo + experts_per_group
            in_group = (experts >= lo) & (experts < hi) & expert_mask
            group_is_selected = (
                tl.sum(
                    tl.where(
                        group_ids == group,
                        selected_groups.to(tl.int32),
                        0,
                    ),
                    axis=0,
                )
                > 0
            )
            keep |= in_group & group_is_selected
        selector = tl.where(keep, selector, -float("inf"))

    selected = tl.zeros((BLOCK_N,), dtype=tl.int1)
    slots = tl.arange(0, BLOCK_K)
    for slot in tl.static_range(K_ROUTED):
        available = tl.where(selected, -float("inf"), selector)
        best_value = tl.max(available, axis=0)
        best_index = tl.min(
            tl.where(available == best_value, experts, n_experts), axis=0
        )
        selected |= experts == best_index
        tl.store(
            indices_ptr + row * TOPK + slots,
            best_index,
            mask=slots == slot,
        )


@triton.jit
def _stage3_finalize(
    scores_ptr,
    sel_ptr,
    indices_ptr,
    weights_ptr,
    n_experts,
    row_start,
    routed_scaling_factor,
    SCORING: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_ROUTED: tl.constexpr,
    TOPK: tl.constexpr,
    RENORM: tl.constexpr,
    SCALE_OUT: tl.constexpr,
):
    row = row_start + tl.program_id(0)
    slots = tl.arange(0, BLOCK_K)
    routed_mask = slots < K_ROUTED
    topk_mask = slots < TOPK
    ids = tl.load(
        indices_ptr + row * TOPK + slots,
        mask=routed_mask,
        other=0,
    )

    if SCORING == 0:
        raw = tl.load(
            scores_ptr + row * n_experts + ids,
            mask=routed_mask,
            other=0.0,
        ).to(tl.float32)
        routed_weights = tl.where(routed_mask, tl.sigmoid(raw), 0.0)
    elif SCORING == 1:
        raw = tl.load(
            scores_ptr + row * n_experts + ids,
            mask=routed_mask,
            other=0.0,
        ).to(tl.float32)
        u = tl.exp(-tl.abs(raw))
        z = u / (2.0 + u)
        z2 = z * z
        softplus = tl.maximum(raw, 0.0) + 2.0 * z * (
            1.0
            + z2
            * (
                1.0 / 3
                + z2
                * (1.0 / 5 + z2 * (1.0 / 7 + z2 * (1.0 / 9 + z2 * (1.0 / 11))))
            )
        )
        routed_weights = tl.where(routed_mask, tl.sqrt(softplus), 0.0)
    else:
        experts = tl.arange(0, BLOCK_N)
        expert_mask = experts < n_experts
        selector = tl.load(
            sel_ptr + row * n_experts + experts,
            mask=expert_mask,
            other=-float("inf"),
        )
        row_max = tl.max(selector, axis=0)
        exponentials = tl.where(expert_mask, tl.exp(selector - row_max), 0.0)
        denominator = tl.sum(exponentials, axis=0)
        selected_values = tl.load(
            sel_ptr + row * n_experts + ids,
            mask=routed_mask,
            other=-float("inf"),
        )
        routed_weights = tl.where(
            routed_mask,
            tl.exp(selected_values - row_max) / denominator,
            0.0,
        )

    routed_sum = tl.sum(routed_weights, axis=0)
    shared_weight = routed_sum / routed_scaling_factor
    weights = tl.where(
        routed_mask,
        routed_weights,
        tl.where(topk_mask, shared_weight, 0.0),
    )
    if RENORM:
        weights /= tl.where(routed_sum > 0.0, routed_sum, 1.0)
    if SCALE_OUT:
        weights *= routed_scaling_factor

    tl.store(weights_ptr + row * TOPK + slots, weights, mask=topk_mask)
    shared_mask = topk_mask & ~routed_mask
    tl.store(
        indices_ptr + row * TOPK + slots,
        n_experts + slots - K_ROUTED,
        mask=shared_mask,
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

    block_n = triton.next_power_of_2(n_experts)
    block_k = triton.next_power_of_2(max(topk, 1))
    groups = num_expert_group
    block_g = triton.next_power_of_2(max(groups, 1))
    experts_per_group = n_experts // groups if groups > 1 else 1
    selector = torch.empty(
        (n_rows, n_experts), dtype=torch.float32, device=scores.device
    )

    row_start = 0
    while row_start < n_rows:
        row_count = min(n_rows - row_start, _MAX_GRID)
        grid = (row_count,)
        _stage1_selector[grid](
            scores,
            bias,
            selector,
            n_experts,
            row_start,
            float(moe_softcapping),
            SCORING=scoring,
            BLOCK_N=block_n,
            num_warps=4,
            num_stages=1,
        )
        _stage2_route[grid](
            selector,
            indices,
            n_experts,
            experts_per_group,
            row_start,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            BLOCK_G=block_g,
            K_ROUTED=k_routed,
            TOPK=topk,
            NUM_GROUPS=groups,
            TOPK_GROUP=topk_group,
            num_warps=4,
            num_stages=1,
        )
        _stage3_finalize[grid](
            scores,
            selector,
            indices,
            weights,
            n_experts,
            row_start,
            float(routed_scaling_factor),
            SCORING=scoring,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            K_ROUTED=k_routed,
            TOPK=topk,
            RENORM=bool(renormalize),
            SCALE_OUT=bool(apply_routed_scaling_factor_on_output),
            num_warps=4,
            num_stages=1,
        )
        row_start += row_count

    return weights, indices


__all__ = ["moe_fused_gate"]
