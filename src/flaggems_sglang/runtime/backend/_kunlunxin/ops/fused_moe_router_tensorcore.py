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

_BLOCK_B = 32
_BLOCK_E = 32
_BLOCK_K = 64
_MAX_GRID = 65535


@triton.jit
def _router_gemm_kernel(
    x_ptr,
    w_ptr,
    logits_ptr,
    n_rows,
    n_experts,
    tiles_e,
    program_start,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = program_start + tl.program_id(0)
    tile_b = tile // tiles_e
    tile_e = tile - tile_b * tiles_e
    rows = tile_b * BLOCK_B + tl.arange(0, BLOCK_B)
    experts = tile_e * BLOCK_E + tl.arange(0, BLOCK_E)
    row_mask = rows < n_rows
    expert_mask = experts < n_experts

    acc = tl.zeros((BLOCK_B, BLOCK_E), dtype=tl.float32)
    for k_start in range(0, HIDDEN_DIM, BLOCK_K):
        ks = k_start + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + rows[:, None] * HIDDEN_DIM + ks[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            w_ptr + experts[None, :] * HIDDEN_DIM + ks[:, None],
            mask=expert_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(x, w, input_precision="ieee")

    tl.store(
        logits_ptr + rows[:, None] * n_experts + experts[None, :],
        acc,
        mask=row_mask[:, None] & expert_mask[None, :],
    )


@triton.jit
def _router_softmax_top2_kernel(
    logits_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    n_experts,
    softcap,
    row_start,
    HAS_BIAS: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    row = row_start + tl.program_id(0)
    experts = tl.arange(0, BLOCK_E)
    expert_mask = experts < n_experts
    logits = tl.load(
        logits_ptr + row * n_experts + experts,
        mask=expert_mask,
        other=-float("inf"),
    ).to(tl.float32)

    if HAS_SOFTCAP:
        scaled = logits / softcap
        scaled_sq = scaled * scaled
        near_zero = scaled * (
            1.0 + scaled_sq * (-1.0 / 3.0 + scaled_sq * (2.0 / 15.0))
        )
        saturated = 2.0 / (1.0 + tl.exp(-2.0 * scaled)) - 1.0
        logits = (
            tl.where(tl.abs(scaled) < 0.25, near_zero, saturated) * softcap
        )
    if HAS_BIAS:
        logits += tl.load(bias_ptr + experts, mask=expert_mask, other=0.0).to(
            tl.float32
        )
    logits = tl.where(expert_mask, logits, -float("inf"))

    best_value = tl.max(logits, axis=0)
    best_index = tl.min(
        tl.where(logits == best_value, experts, n_experts), axis=0
    )
    denom = tl.sum(tl.exp(logits - best_value), axis=0)

    tl.store(weights_ptr + row * TOPK, 1.0 / denom)
    tl.store(ids_ptr + row * TOPK, best_index.to(tl.int32))
    if TOPK == 2:
        candidates = tl.where(experts == best_index, -float("inf"), logits)
        second_value = tl.max(candidates, axis=0)
        second_index = tl.min(
            tl.where(candidates == second_value, experts, n_experts), axis=0
        )
        tl.store(
            weights_ptr + row * TOPK + 1,
            tl.exp(second_value - best_value) / denom,
        )
        tl.store(ids_ptr + row * TOPK + 1, second_index.to(tl.int32))


def _next_pow2(value):
    result = 1
    while result < value:
        result *= 2
    return result


def fused_moe_router_tensorcore(
    x, router_weight, topk, moe_softcapping, correction_bias=None
):
    x = x.contiguous()
    router_weight = router_weight.contiguous()
    bias = (
        correction_bias.contiguous()
        if isinstance(correction_bias, torch.Tensor)
        else correction_bias
    )
    n_rows = x.shape[0]
    hidden_dim = x.shape[-1]
    n_experts = router_weight.shape[0]
    assert topk in (1, 2), "tensorcore variant requires topk <= 2"
    assert hidden_dim % _BLOCK_K == 0, "hidden size must be a multiple of 64"
    assert n_experts <= 256, "expert count above 256 unsupported"

    topk_weights = torch.empty(
        (n_rows, topk), dtype=torch.float32, device=x.device
    )
    topk_ids = torch.empty((n_rows, topk), dtype=torch.int32, device=x.device)
    if n_rows == 0:
        return topk_weights, topk_ids

    logits = torch.empty(
        (n_rows, n_experts), dtype=torch.float32, device=x.device
    )
    tiles_e = triton.cdiv(n_experts, _BLOCK_E)
    total_programs = triton.cdiv(n_rows, _BLOCK_B) * tiles_e
    for program_start in range(0, total_programs, _MAX_GRID):
        program_count = min(_MAX_GRID, total_programs - program_start)
        _router_gemm_kernel[(program_count,)](
            x,
            router_weight,
            logits,
            n_rows,
            n_experts,
            tiles_e,
            program_start,
            HIDDEN_DIM=hidden_dim,
            BLOCK_B=_BLOCK_B,
            BLOCK_E=_BLOCK_E,
            BLOCK_K=_BLOCK_K,
            num_warps=4,
            num_stages=1,
        )

    bias_arg = bias if bias is not None else x
    block_e = _next_pow2(n_experts)
    for row_start in range(0, n_rows, _MAX_GRID):
        row_count = min(_MAX_GRID, n_rows - row_start)
        _router_softmax_top2_kernel[(row_count,)](
            logits,
            bias_arg,
            topk_weights,
            topk_ids,
            n_experts,
            float(moe_softcapping),
            row_start,
            HAS_BIAS=bias is not None,
            HAS_SOFTCAP=moe_softcapping != 0,
            TOPK=topk,
            BLOCK_E=block_e,
            num_warps=4,
            num_stages=1,
        )
    return topk_weights, topk_ids


__all__ = ["fused_moe_router_tensorcore"]

# water-sample carrier r1 (2026-09-02): bytes identical to e8-140a632 team best
# water-sample carrier r2 (2026-09-03)
