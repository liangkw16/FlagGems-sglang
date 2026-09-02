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
_BLOCK = 512


@triton.jit
def _moe_fused_mul_sum_kernel(
    inputs_ptr,
    weights_ptr,
    ids_ptr,
    map_ptr,
    output_ptr,
    total_blocks,
    num_col_blocks,
    hidden_dim,
    scale,
    HAS_MAP: tl.constexpr,
    IS_EP: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # flat 1D grid over (token, hidden-block) tiles: one integer division
    # per block; int32 addressing throughout (shapes < 2^31)
    pid = tl.program_id(0)
    grid_stride = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    out_ty = output_ptr.dtype.element_ty
    for block_id in range(pid, total_blocks, grid_stride):
        token = block_id // num_col_blocks
        col_block = block_id - token * num_col_blocks
        h = col_block * BLOCK + offs
        hmask = h < hidden_dim
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        row_base = token * TOP_K
        for k in tl.static_range(TOP_K):
            weight = tl.load(weights_ptr + row_base + k).to(tl.float32) * scale
            if HAS_MAP or IS_EP:
                expert_id = tl.load(ids_ptr + row_base + k)
                if HAS_MAP:
                    mapped = tl.load(map_ptr + expert_id)
                    weight = tl.where(mapped >= 0, weight, 0.0)
                else:
                    weight = tl.where(expert_id >= 0, weight, 0.0)
            values = tl.load(
                inputs_ptr + (row_base + k) * hidden_dim + h,
                mask=hmask,
                other=0.0,
            ).to(tl.float32)
            acc += values * weight
        tl.store(
            output_ptr + token * hidden_dim + h, acc.to(out_ty), mask=hmask
        )


def moe_fused_mul_sum(
    inputs,
    topk_weights,
    topk_ids=None,
    expert_map=None,
    routed_scaling_factor=None,
    is_ep=False,
):
    inputs = inputs.contiguous()
    topk_weights = topk_weights.contiguous()
    if topk_ids is not None:
        topk_ids = topk_ids.contiguous()
    num_tokens, top_k, hidden_dim = inputs.shape
    output = torch.empty(
        (num_tokens, hidden_dim), dtype=inputs.dtype, device=inputs.device
    )
    if num_tokens == 0 or hidden_dim == 0 or top_k == 0:
        return output

    scale = (
        1.0 if routed_scaling_factor is None else float(routed_scaling_factor)
    )
    has_map = expert_map is not None
    use_ep = is_ep and topk_ids is not None and not has_map

    num_col_blocks = triton.cdiv(hidden_dim, _BLOCK)
    total_blocks = num_tokens * num_col_blocks
    grid = (min(total_blocks, _MAX_GRID),)
    _moe_fused_mul_sum_kernel[grid](
        inputs,
        topk_weights,
        topk_ids if topk_ids is not None else inputs,
        expert_map if has_map else inputs,
        output,
        total_blocks,
        num_col_blocks,
        hidden_dim,
        scale,
        HAS_MAP=has_map,
        IS_EP=use_ep,
        TOP_K=top_k,
        BLOCK=_BLOCK,
    )
    return output


__all__ = ["moe_fused_mul_sum"]

# water-sample carrier r1 (2026-09-02): bytes identical to s0-63e2550 team best
# water-sample carrier r2 (2026-09-03)
# non-enflame sampling carrier (2026-09-03, enflame service outage)
