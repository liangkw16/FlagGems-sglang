# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 109 zero_experts_identity - Kunlun XPU variant.
# The generic kernel addresses rows with a scalar program-id multiplied
# into every store (scalar/vector addptr mixing); on XPU that encoding
# miscompiles and leaves freshly-allocated output bytes unwritten, which
# the platform saw as 3e38 garbage. This variant flattens the output to
# one int32 element grid, derives row/col per lane, keeps every address
# expression a plain int32 vector, and gathers the zero-expert scale sum
# with a runtime top_k loop instead of a lane reduction.

import torch
import triton
import triton.language as tl


@triton.jit
def _zero_experts_flat_kernel(
    indices,
    scales,
    hidden,
    out,
    num_experts,
    top_k,
    hidden_dim,
    total,
    ns0,
    ss0,
    hs0,
    hs1,
    os0,
    os1,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < total
    row = offs // hidden_dim
    col = offs - row * hidden_dim
    zsum = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in range(0, top_k):
        idj = tl.load(indices + row * ns0 + j, mask=m, other=0)
        scj = tl.load(scales + row * ss0 + j, mask=m, other=0.0).to(tl.float32)
        zsum += tl.where(idj.to(tl.int32) >= num_experts, scj, 0.0)
    h = tl.load(hidden + row * hs0 + col * hs1, mask=m, other=0.0).to(tl.float32)
    tl.store(
        out + row * os0 + col * os1,
        (h * zsum).to(out.dtype.element_ty),
        mask=m,
    )


def zero_experts_identity(expert_indices, expert_scales, num_experts, zero_expert_type, hidden_states):
    num_tokens, hidden_dim = hidden_states.shape
    total = num_tokens * hidden_dim
    out = torch.empty(hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device)
    if total:
        _zero_experts_flat_kernel[(triton.cdiv(total, 1024),)](
            expert_indices,
            expert_scales,
            hidden_states,
            out,
            num_experts,
            expert_indices.shape[1],
            hidden_dim,
            total,
            expert_indices.stride(0),
            expert_scales.stride(0),
            hidden_states.stride(0),
            hidden_states.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK=1024,
            num_warps=4,
        )
    return out


__all__ = ["zero_experts_identity"]
