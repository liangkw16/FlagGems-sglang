# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 109 zero_experts_identity - Kunlun XPU variant.
# The generic kernel folds a scalar program-id into every store address
# (scalar/vector addptr mixing); on XPU that encoding miscompiles and
# leaves freshly-allocated output bytes unwritten, which the platform
# saw as 3e38 garbage. This variant keeps the generic one-program-per-
# token shape and speed but widens the row index into the offset vector
# itself, so every pointer expression is a plain int32 vector - no
# scalar lanes in any address, no div/mod per element.

import torch
import triton
import triton.language as tl


@triton.jit
def _zero_experts_kernel(
    indices,
    scales,
    hidden,
    out,
    num_experts,
    top_k,
    hidden_dim,
    ns0,
    ss0,
    hs0,
    hs1,
    os0,
    os1,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int32)
    ks = tl.arange(0, BLOCK_K)
    km = ks < top_k
    rk = tl.zeros((BLOCK_K,), dtype=tl.int32) + row
    idx = tl.load(indices + rk * ns0 + ks, mask=km, other=0)
    sc = tl.load(scales + rk * ss0 + ks, mask=km, other=0.0).to(tl.float32)
    ne = num_experts.to(tl.int32)
    zero_sum = tl.sum(tl.where(idx.to(tl.int32) >= ne, sc, 0.0), axis=0)
    cols = tl.arange(0, BLOCK_D).to(tl.int32)
    for c0 in range(0, hidden_dim, BLOCK_D):
        cc = c0 + cols
        m = cc < hidden_dim
        rv = tl.zeros((BLOCK_D,), dtype=tl.int32) + row
        h = tl.load(
            hidden + rv * hs0 + cc * hs1, mask=m, other=0.0
        ).to(tl.float32)
        tl.store(
            out + rv * os0 + cc * os1,
            (h * zero_sum).to(out.dtype.element_ty),
            mask=m,
        )


def zero_experts_identity(expert_indices, expert_scales, num_experts, zero_expert_type, hidden_states):
    num_tokens, hidden_dim = hidden_states.shape
    top_k = expert_indices.shape[1]
    out = torch.empty(hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device)
    if num_tokens and hidden_dim:
        _zero_experts_kernel[(num_tokens,)](
            expert_indices,
            expert_scales,
            hidden_states,
            out,
            num_experts,
            top_k,
            hidden_dim,
            expert_indices.stride(0),
            expert_scales.stride(0),
            hidden_states.stride(0),
            hidden_states.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_K=max(16, triton.next_power_of_2(top_k)),
            BLOCK_D=1024,
            num_warps=8,
        )
    return out


__all__ = ["zero_experts_identity"]
