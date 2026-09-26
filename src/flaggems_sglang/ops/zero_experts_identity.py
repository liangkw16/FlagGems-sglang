# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 109 zero_experts_identity: identity-expert reduction - each token
# scales its own hidden state by the sum of routing weights whose expert
# id routes to the no-op path (id >= num_experts). One program per
# token: a small lane vector masks and reduces the top_k scales, then
# the whole hidden row (a 256-multiple) is scaled in fp32 and stored in
# the input dtype.

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
    os0,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    ks = tl.arange(0, BLOCK_K)
    km = ks < top_k
    idx = tl.load(indices + row * ns0 + ks, mask=km, other=0)
    sc = tl.load(scales + row * ss0 + ks, mask=km, other=0.0).to(tl.float32)
    zero_sum = tl.sum(tl.where(idx >= num_experts, sc, 0.0), axis=0)
    cols = tl.arange(0, BLOCK_D).to(tl.int64)
    for c0 in range(0, hidden_dim, BLOCK_D):
        cc = c0 + cols
        m = cc < hidden_dim
        h = tl.load(hidden + row * hs0 + cc, mask=m, other=0.0).to(tl.float32)
        tl.store(out + row * os0 + cc, (h * zero_sum).to(out.dtype.element_ty), mask=m)


def zero_experts_identity(expert_indices, expert_scales, num_experts, zero_expert_type, hidden_states):
    num_tokens, hidden_dim = hidden_states.shape
    top_k = expert_indices.shape[1]
    out = torch.empty_like(hidden_states)
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
            out.stride(0),
            BLOCK_K=max(16, triton.next_power_of_2(top_k)),
            BLOCK_D=1024,
            num_warps=8,
        )
    return out


__all__ = ["zero_experts_identity"]
