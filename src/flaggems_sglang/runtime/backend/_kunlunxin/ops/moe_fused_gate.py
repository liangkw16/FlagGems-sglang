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
def _group_score(
    sel_ptr,
    gs_ptr,
    n_experts,
    n_groups,
    experts_per_group,
    row_start,
    BLOCK_V: tl.constexpr,
):
    row = row_start + tl.program_id(0)
    for group in range(n_groups):
        lo = group * experts_per_group
        offs = tl.arange(0, BLOCK_V)
        in_group = offs < experts_per_group
        vals = tl.load(
            sel_ptr + row * n_experts + lo + offs,
            mask=in_group,
            other=-float("inf"),
        )
        first = tl.max(vals, axis=0)
        cnt_first = tl.sum(tl.where(vals == first, 1, 0), axis=0)
        smaller = tl.max(tl.where(vals < first, vals, -float("inf")), axis=0)
        second = tl.where(cnt_first >= 2, first, smaller)
        tl.store(gs_ptr + row * n_groups + group, first + second)


@triton.jit
def _group_select(
    gs_ptr,
    gk_ptr,
    n_groups,
    row_start,
    BLOCK_G: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
):
    row = row_start + tl.program_id(0)
    group_ids = tl.arange(0, BLOCK_G)
    group_mask = group_ids < n_groups
    vals = tl.load(
        gs_ptr + row * n_groups + group_ids,
        mask=group_mask,
        other=-float("inf"),
    )
    better = (vals[None, :] > vals[:, None]) | (
        (vals[None, :] == vals[:, None])
        & (group_ids[None, :] < group_ids[:, None])
    )
    rank = tl.sum(
        tl.where(group_mask[None, :], better.to(tl.int32), 0), axis=1
    )
    keep = group_mask & (rank < TOPK_GROUP)
    tl.store(
        gk_ptr + row * n_groups + group_ids,
        keep.to(tl.int32),
        mask=group_mask,
    )


@triton.jit
def _apply_group_mask(
    sel_ptr,
    eg_ptr,
    gk_ptr,
    selw_ptr,
    n_experts,
    n_groups,
    row_start,
    BLOCK_N: tl.constexpr,
):
    row = row_start + tl.program_id(0)
    experts = tl.arange(0, BLOCK_N)
    mask = experts < n_experts
    sel = tl.load(
        sel_ptr + row * n_experts + experts, mask=mask, other=-float("inf")
    )
    eg = tl.load(eg_ptr + experts, mask=mask, other=0)
    keep = tl.load(gk_ptr + row * n_groups + eg, mask=mask, other=0)
    selw = tl.where(keep > 0, sel, -float("inf"))
    tl.store(selw_ptr + row * n_experts + experts, selw, mask=mask)


@triton.jit
def _pick_slot(
    selw_ptr,
    selected_ptr,
    indices_ptr,
    n_experts,
    row_start,
    slot,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = row_start + tl.program_id(0)
    experts = tl.arange(0, BLOCK_N)
    mask = experts < n_experts
    sel = tl.load(
        selw_ptr + row * n_experts + experts, mask=mask, other=-float("inf")
    )
    taken = tl.load(
        selected_ptr + row * n_experts + experts, mask=mask, other=0
    )
    cand = tl.where(mask & (taken == 0), sel, -float("inf"))
    best_value = tl.max(cand, axis=0)
    best_index = tl.min(
        tl.where(cand == best_value, experts, n_experts), axis=0
    )
    tl.store(indices_ptr + row * TOPK + slot, best_index)
    tl.store(selected_ptr + row * n_experts + best_index, 1)


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
    selected = torch.zeros(
        (n_rows, n_experts), dtype=torch.int32, device=scores.device
    )
    if groups > 1:
        group_scores = torch.empty(
            (n_rows, groups), dtype=torch.float32, device=scores.device
        )
        group_keep = torch.empty(
            (n_rows, groups), dtype=torch.int32, device=scores.device
        )
        expert_group = torch.div(
            torch.arange(n_experts, device=scores.device, dtype=torch.int32),
            experts_per_group,
            rounding_mode="floor",
        ).clamp_(max=groups - 1)
        sel_work = torch.empty_like(selector)
    else:
        sel_work = selector

    block_v = triton.next_power_of_2(max(experts_per_group, 1))
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
        if groups > 1:
            _group_score[grid](
                selector,
                group_scores,
                n_experts,
                groups,
                experts_per_group,
                row_start,
                BLOCK_V=block_v,
                num_warps=4,
                num_stages=1,
            )
            _group_select[grid](
                group_scores,
                group_keep,
                groups,
                row_start,
                BLOCK_G=block_g,
                TOPK_GROUP=topk_group,
                num_warps=4,
                num_stages=1,
            )
            _apply_group_mask[grid](
                selector,
                expert_group,
                group_keep,
                sel_work,
                n_experts,
                groups,
                row_start,
                BLOCK_N=block_n,
                num_warps=4,
                num_stages=1,
            )
        for slot in range(k_routed):
            _pick_slot[grid](
                sel_work,
                selected,
                indices,
                n_experts,
                row_start,
                slot,
                TOPK=topk,
                BLOCK_N=block_n,
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
