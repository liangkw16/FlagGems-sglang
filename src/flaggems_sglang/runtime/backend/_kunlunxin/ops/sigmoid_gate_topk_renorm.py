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
def _stage1_select_kernel(
    logits_ptr,
    bias_ptr,
    ids_ptr,
    row_start,
    n_tokens,
    n_routed,
    n_shared,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row = row_start + pid
    if row >= n_tokens:
        return

    total_experts = n_routed + n_shared
    n_offs = tl.arange(0, BLOCK_N)
    routed_mask = n_offs < n_routed

    raw_routed = tl.load(
        logits_ptr + row.to(tl.int64) * total_experts + n_offs,
        mask=routed_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    bias = tl.load(
        bias_ptr + n_offs,
        mask=routed_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    sel_score = tl.where(
        routed_mask, tl.sigmoid(raw_routed) + bias, -float("inf")
    )

    slot_offs = tl.arange(0, BLOCK_K)

    selected_mask = tl.zeros((BLOCK_N,), dtype=tl.int1)

    for slot in tl.static_range(TOPK):
        cand = tl.where(selected_mask, -float("inf"), sel_score)
        best_val = tl.max(cand, axis=0)
        best_idx = tl.min(
            tl.where((cand == best_val) & routed_mask, n_offs, n_routed),
            axis=0,
        )
        picked = n_offs == best_idx
        selected_mask = selected_mask | picked
        store_mask = slot_offs == slot
        tl.store(
            ids_ptr + row.to(tl.int64) * TOPK + slot_offs,
            tl.broadcast_to(best_idx, (BLOCK_K,)),
            mask=store_mask,
        )


@triton.jit
def _stage2_normalize_kernel(
    logits_ptr,
    ids_ptr,
    routed_w_ptr,
    shared_w_ptr,
    row_start,
    n_tokens,
    n_routed,
    n_shared,
    route_scale,
    global_scale_ptr,
    global_scale_scalar,
    GLOBAL_SCALE_IS_TENSOR: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    row = row_start + pid
    if row >= n_tokens:
        return

    if GLOBAL_SCALE_IS_TENSOR:
        global_scale = tl.load(global_scale_ptr).to(tl.float32)
    else:
        global_scale = global_scale_scalar
    scale = route_scale * global_scale

    total_experts = n_routed + n_shared
    row_base = row.to(tl.int64) * total_experts

    slot_offs = tl.arange(0, BLOCK_K)
    slot_mask = slot_offs < TOPK

    expert_ids = tl.load(
        ids_ptr + row.to(tl.int64) * TOPK + slot_offs,
        mask=slot_mask,
        other=0,
    )

    routed_logits = tl.load(
        logits_ptr + row_base + expert_ids,
        mask=slot_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    routed_probs = tl.where(slot_mask, tl.sigmoid(routed_logits), 0.0)

    denom = tl.sum(routed_probs, axis=0)

    s_offs = tl.arange(0, BLOCK_S)
    s_mask = s_offs < n_shared

    shared_logits = tl.load(
        logits_ptr + row_base + n_routed + s_offs,
        mask=s_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    shared_probs = tl.where(s_mask, tl.sigmoid(shared_logits), 0.0)

    denom += tl.sum(shared_probs, axis=0)

    inv_denom = 1.0 / (denom + 1e-9)

    out_ty = routed_w_ptr.dtype.element_ty

    norm_routed = (routed_probs * inv_denom * scale).to(out_ty)
    tl.store(
        routed_w_ptr + row.to(tl.int64) * TOPK + slot_offs,
        norm_routed,
        mask=slot_mask,
    )

    if BLOCK_S > 0:
        norm_shared = (shared_probs * inv_denom * scale).to(out_ty)
        tl.store(
            shared_w_ptr + row.to(tl.int64) * n_shared + s_offs,
            norm_shared,
            mask=s_mask,
        )


def sigmoid_gate_topk_renorm(
    logits, k, n_shared_experts, route_scale, global_scale, bias
):
    logits = logits.contiguous()
    bias = bias.contiguous()
    n_tokens, total_experts = logits.shape
    n_routed = total_experts - n_shared_experts

    routed_weights = torch.empty(
        (n_tokens, k), dtype=logits.dtype, device=logits.device
    )
    indices = torch.empty(
        (n_tokens, k), dtype=torch.int32, device=logits.device
    )
    shared_weights = torch.empty(
        (n_tokens, n_shared_experts),
        dtype=logits.dtype,
        device=logits.device,
    )

    if n_tokens == 0 or k == 0:
        return routed_weights, indices, shared_weights

    global_scale_is_tensor = torch.is_tensor(global_scale)
    if global_scale_is_tensor:
        global_scale_ptr = global_scale
        global_scale_scalar = 0.0
    else:
        global_scale_ptr = logits
        global_scale_scalar = float(global_scale)

    BLOCK_N = triton.next_power_of_2(max(n_routed, 1))
    BLOCK_K = triton.next_power_of_2(max(k, 1))
    BLOCK_S = triton.next_power_of_2(max(n_shared_experts, 1))

    row_start = 0
    while row_start < n_tokens:
        chunk = min(n_tokens - row_start, _MAX_GRID)
        grid = (chunk,)

        _stage1_select_kernel[grid](
            logits,
            bias,
            indices,
            row_start,
            n_tokens,
            n_routed,
            n_shared_experts,
            TOPK=k,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=1,
        )

        _stage2_normalize_kernel[grid](
            logits,
            indices,
            routed_weights,
            shared_weights,
            row_start,
            n_tokens,
            n_routed,
            n_shared_experts,
            float(route_scale),
            global_scale_ptr,
            global_scale_scalar,
            GLOBAL_SCALE_IS_TENSOR=global_scale_is_tensor,
            TOPK=k,
            BLOCK_K=BLOCK_K,
            BLOCK_S=BLOCK_S,
            num_warps=4,
            num_stages=1,
        )

        row_start += chunk

    return routed_weights, indices, shared_weights


__all__ = ["sigmoid_gate_topk_renorm"]
