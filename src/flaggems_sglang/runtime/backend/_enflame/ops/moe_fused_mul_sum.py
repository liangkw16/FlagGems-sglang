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
    top_k,
    HAS_MAP: tl.constexpr,
    IS_EP: tl.constexpr,
    TOP_K: tl.constexpr,
    K_TILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # enflame vendor (e2): [TOP_K, BLOCK] 2D tile loaded in one shot with a
    # parallel axis-0 weighted reduction - the T33 E1->E2 structural class
    # (sequential static_range loop -> parallel matrix), replacing the
    # generic per-k strided loads
    pid = tl.program_id(0)
    grid_stride = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    k_offs = tl.arange(0, K_TILE)
    kmask = k_offs < top_k
    out_ty = output_ptr.dtype.element_ty
    for block_id in range(pid, total_blocks, grid_stride):
        token = block_id // num_col_blocks
        col_block = block_id - token * num_col_blocks
        h = col_block * BLOCK + offs
        hmask = h < hidden_dim
        row_base = token * TOP_K
        w_vec = tl.load(
            weights_ptr + row_base + k_offs, mask=kmask, other=0.0
        ).to(tl.float32)
        if HAS_MAP or IS_EP:
            expert_ids = tl.load(
                ids_ptr + row_base + k_offs, mask=kmask, other=0
            )
            if HAS_MAP:
                mapped = tl.load(map_ptr + expert_ids, mask=kmask, other=0)
                w_vec = tl.where(mapped >= 0, w_vec, 0.0) * scale
            else:
                w_vec = tl.where(expert_ids >= 0, w_vec, 0.0) * scale
        else:
            w_vec = w_vec * scale
        tile = tl.load(
            inputs_ptr
            + (row_base + k_offs)[:, None] * hidden_dim
            + h[None, :],
            mask=kmask[:, None] & hmask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = tl.sum(tile * w_vec[:, None], axis=0)
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
        top_k,
        HAS_MAP=has_map,
        IS_EP=use_ep,
        TOP_K=top_k,
        K_TILE=triton.next_power_of_2(max(top_k, 2)),
        BLOCK=_BLOCK,
    )
    return output


__all__ = ["moe_fused_mul_sum"]
