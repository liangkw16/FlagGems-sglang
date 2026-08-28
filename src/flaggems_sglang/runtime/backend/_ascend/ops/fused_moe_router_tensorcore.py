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

_BLOCK_B = 64
_BLOCK_E = 64
_BLOCK_K = 64
_ROWS_REDUCE = 8
_MAX_GRID = 65535


@triton.jit
def _router_gemm_splitk_kernel(
    x_ptr,
    w_ptr,
    partials_ptr,
    n_rows,
    n_experts,
    hidden_dim,
    k_chunk,
    tiles_e,
    n_splits,
    BLOCK_B: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n_tiles = tl.cdiv(n_rows, BLOCK_B) * tiles_e * n_splits
    n_programs = tl.num_programs(0)
    for tile in tl.range(tl.program_id(0), n_tiles, n_programs):
        split_id = tile % n_splits
        tile_rest = tile // n_splits
        tile_e = tile_rest % tiles_e
        tile_b = tile_rest // tiles_e
        rows = tile_b * BLOCK_B + tl.arange(0, BLOCK_B)
        experts = tile_e * BLOCK_E + tl.arange(0, BLOCK_E)
        row_mask = rows < n_rows
        expert_mask = experts < n_experts

        k_begin = split_id * k_chunk
        k_end = tl.minimum(k_begin + k_chunk, hidden_dim)

        acc = tl.zeros((BLOCK_B, BLOCK_E), dtype=tl.float32)
        for k_start in tl.range(k_begin, k_end, BLOCK_K):
            ks = k_start + tl.arange(0, BLOCK_K)
            k_mask = ks < k_end
            x_tile = tl.load(
                x_ptr + rows[:, None] * hidden_dim + ks[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            w_tile = tl.load(
                w_ptr + experts[:, None] * hidden_dim + ks[None, :],
                mask=expert_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(
                x_tile, tl.trans(w_tile), input_precision="ieee"
            )

        tl.store(
            partials_ptr
            + split_id * n_rows * n_experts
            + rows[:, None] * n_experts
            + experts[None, :],
            acc,
            mask=row_mask[:, None] & expert_mask[None, :],
        )


@triton.jit
def _router_softmax_top2_kernel(
    partials_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    n_rows,
    n_experts,
    n_splits,
    softcap,
    HAS_BIAS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    n_tiles = tl.cdiv(n_rows, BLOCK_R)
    n_programs = tl.num_programs(0)
    for tile in tl.range(tl.program_id(0), n_tiles, n_programs):
        rows = tile * BLOCK_R + tl.arange(0, BLOCK_R)
        experts = tl.arange(0, BLOCK_E)
        row_mask = rows < n_rows
        expert_mask = experts < n_experts

        logits = tl.zeros((BLOCK_R, BLOCK_E), dtype=tl.float32)
        for split_id in tl.range(0, n_splits, 1):
            part = tl.load(
                partials_ptr
                + split_id * n_rows * n_experts
                + rows[:, None] * n_experts
                + experts[None, :],
                mask=row_mask[:, None] & expert_mask[None, :],
                other=0.0,
            )
            logits += part
        logits = tl.where(expert_mask[None, :], logits, -float("inf"))

        scaled = logits / softcap
        scaled_sq = scaled * scaled
        near_zero = scaled * (
            1.0 + scaled_sq * (-1.0 / 3.0 + scaled_sq * (2.0 / 15.0))
        )
        saturated = 2.0 / (1.0 + tl.exp(-2.0 * scaled)) - 1.0
        tanh_form = tl.where(tl.abs(scaled) < 0.25, near_zero, saturated)
        logits = tl.where(softcap != 0.0, tanh_form * softcap, logits)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + experts, mask=expert_mask, other=0.0).to(
                tl.float32
            )
            logits += bias[None, :]

        best_value = tl.max(logits, axis=1)
        best_index = tl.min(
            tl.where(logits == best_value[:, None], experts, n_experts),
            axis=1,
        )
        denom = tl.sum(tl.exp(logits - best_value[:, None]), axis=1)

        tl.store(weights_ptr + rows * TOPK, 1.0 / denom, mask=row_mask)
        tl.store(ids_ptr + rows * TOPK, best_index.to(tl.int32), mask=row_mask)
        if TOPK == 2:
            cand = tl.where(
                experts[None, :] == best_index[:, None],
                -float("inf"),
                logits,
            )
            second_value = tl.max(cand, axis=1)
            second_index = tl.min(
                tl.where(cand == second_value[:, None], experts, n_experts),
                axis=1,
            )
            tl.store(
                weights_ptr + rows * TOPK + 1,
                tl.exp(second_value - best_value) / denom,
                mask=row_mask,
            )
            tl.store(
                ids_ptr + rows * TOPK + 1,
                second_index.to(tl.int32),
                mask=row_mask,
            )


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
    assert n_experts <= 256, "expert count above 256 unsupported"

    topk_weights = torch.empty(
        (n_rows, topk), dtype=torch.float32, device=x.device
    )
    topk_ids = torch.empty((n_rows, topk), dtype=torch.int32, device=x.device)
    if n_rows == 0:
        return topk_weights, topk_ids

    n_splits = max(1, min(triton.cdiv(hidden_dim, 512), 8))
    k_chunk = triton.cdiv(hidden_dim, n_splits)
    partials = torch.empty(
        (n_splits, n_rows, n_experts), dtype=torch.float32, device=x.device
    )
    tiles_e = triton.cdiv(n_experts, _BLOCK_E)
    n_tiles = triton.cdiv(n_rows, _BLOCK_B) * tiles_e * n_splits
    _router_gemm_splitk_kernel[(min(n_tiles, _MAX_GRID),)](
        x,
        router_weight,
        partials,
        n_rows,
        n_experts,
        hidden_dim,
        k_chunk,
        tiles_e,
        n_splits,
        BLOCK_B=_BLOCK_B,
        BLOCK_E=_BLOCK_E,
        BLOCK_K=_BLOCK_K,
        num_stages=2,
    )

    block_e = _next_pow2(n_experts)
    bias_arg = bias if bias is not None else x
    grid_reduce = (min(triton.cdiv(n_rows, _ROWS_REDUCE), _MAX_GRID),)
    _router_softmax_top2_kernel[grid_reduce](
        partials,
        bias_arg,
        topk_weights,
        topk_ids,
        n_rows,
        n_experts,
        n_splits,
        float(moe_softcapping),
        HAS_BIAS=bias is not None,
        TOPK=topk,
        BLOCK_E=block_e,
        BLOCK_R=_ROWS_REDUCE,
    )
    return topk_weights, topk_ids


__all__ = ["fused_moe_router_tensorcore"]
