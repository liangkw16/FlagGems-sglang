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
# e1 re-carrier (held for a kunlun health window): S0 hit the 1830s
# validation-stage Segmentation fault (crash-family case 15) with the
# cleanest possible kernel (no dot, no libdevice transcendentals);
# bytes identical to 311570f otherwise. Fire on a T38 threshold-team
# count rise.


@triton.jit
def _sigmoid_gate_topk_renorm_kernel(
    logits_ptr,
    bias_ptr,
    routed_w_ptr,
    indices_ptr,
    shared_w_ptr,
    n_tokens,
    n_routed,
    n_shared,
    route_scale,
    global_scale,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    # one program per token row; two deterministic selection passes
    # (pass 1: indices + prob total; pass 2: normalized weights) to
    # avoid same-program store->load visibility hazards
    pid = tl.program_id(0)
    grid_stride = tl.num_programs(0)
    offs = tl.arange(0, BLOCK_E)
    row_mask = offs < (n_routed + n_shared)
    routed_mask = offs < n_routed
    shared_mask = row_mask & (offs >= n_routed)
    out_ty = routed_w_ptr.dtype.element_ty
    scale = route_scale * global_scale
    for row in range(pid, n_tokens, grid_stride):
        raw = tl.load(
            logits_ptr + row * (n_routed + n_shared) + offs,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        act = tl.sigmoid(raw)
        bias = tl.load(bias_ptr + offs, mask=routed_mask, other=0.0).to(
            tl.float32
        )
        sel = tl.where(routed_mask, act + bias, -float("inf"))

        total = tl.sum(tl.where(shared_mask, act, 0.0), axis=0)
        selected = tl.zeros((BLOCK_E,), dtype=tl.int1)
        for slot in tl.static_range(TOPK):
            cand = tl.where(selected, -float("inf"), sel)
            cand_value = tl.max(cand, axis=0)
            cand_index = tl.min(
                tl.where(cand == cand_value, offs, BLOCK_E), axis=0
            )
            picked = offs == cand_index
            selected |= picked
            prob = tl.sigmoid(tl.sum(tl.where(picked, raw, 0.0), axis=0))
            total += prob
            tl.store(indices_ptr + row * TOPK + slot, cand_index)

        inv_total = 1.0 / total
        selected = tl.zeros((BLOCK_E,), dtype=tl.int1)
        for slot in tl.static_range(TOPK):
            cand = tl.where(selected, -float("inf"), sel)
            cand_value = tl.max(cand, axis=0)
            cand_index = tl.min(
                tl.where(cand == cand_value, offs, BLOCK_E), axis=0
            )
            picked = offs == cand_index
            selected |= picked
            prob = tl.sigmoid(tl.sum(tl.where(picked, raw, 0.0), axis=0))
            tl.store(
                routed_w_ptr + row * TOPK + slot,
                (prob * inv_total * scale).to(out_ty),
            )
        tl.store(
            shared_w_ptr + row * n_shared + (offs - n_routed),
            (act * inv_total * scale).to(out_ty),
            mask=shared_mask,
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
    if torch.is_tensor(global_scale):
        global_scale = float(global_scale.reshape(-1)[0].item())
    grid = (min(n_tokens, _MAX_GRID),)
    _sigmoid_gate_topk_renorm_kernel[grid](
        logits,
        bias,
        routed_weights,
        indices,
        shared_weights,
        n_tokens,
        n_routed,
        n_shared_experts,
        float(route_scale),
        float(global_scale),
        TOPK=k,
        BLOCK_E=triton.next_power_of_2(max(total_experts, 2)),
    )
    return routed_weights, indices, shared_weights


__all__ = ["sigmoid_gate_topk_renorm"]
