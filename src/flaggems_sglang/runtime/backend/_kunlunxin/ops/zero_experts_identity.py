# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 109 zero_experts_identity - Kunlun XPU variant.
# Platform evidence (subs 21838/21854/21858): the per-token row layout
# (scalar or broadcast row folded into every store address) miscompiles
# on XPU and leaves the fresh output buffer unwritten (3e38 garbage),
# while the fully flat int32 element grid with div/mod row derivation is
# numerically correct but 0.0146x - every element re-gathered the k
# routing rows. This variant keeps the proven flat skeletons and splits
# the work in two kernels: kernel A reduces each token's zero-expert
# scale sum once into a tiny fp32 buffer, kernel B scales the hidden
# state with one temp load per element instead of 2k gathers.

import torch
import triton
import triton.language as tl


@triton.jit
def _zero_experts_rowsum_kernel(
    indices,
    scales,
    rowsum,
    num_experts,
    top_k,
    ns0,
    ss0,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    ks = tl.arange(0, BLOCK_K)
    km = ks < top_k
    offs = pid * top_k + ks
    row = offs // top_k
    slot = offs - row * top_k
    idx = tl.load(indices + row * ns0 + slot, mask=km, other=0)
    sc = tl.load(scales + row * ss0 + slot, mask=km, other=0.0).to(tl.float32)
    z = tl.where(idx.to(tl.int32) >= num_experts, sc, 0.0)
    s = tl.sum(z, axis=0)
    sv = s + tl.zeros((BLOCK_K,), dtype=tl.float32)
    tl.store(rowsum + offs, sv, mask=km & (slot == 0))


@triton.jit
def _zero_experts_scale_kernel(
    rowsum,
    hidden,
    out,
    top_k,
    hidden_dim,
    total,
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
    zs = tl.load(rowsum + row, mask=m, other=0.0)
    h = tl.load(hidden + row * hs0 + col * hs1, mask=m, other=0.0).to(tl.float32)
    tl.store(out + row * os0 + col * os1, (h * zs).to(out.dtype.element_ty), mask=m)


def zero_experts_identity(expert_indices, expert_scales, num_experts, zero_expert_type, hidden_states):
    num_tokens, hidden_dim = hidden_states.shape
    top_k = expert_indices.shape[1]
    out = torch.empty(hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device)
    total = num_tokens * hidden_dim
    if total:
        rowsum = torch.zeros(num_tokens, dtype=torch.float32, device=hidden_states.device)
        _zero_experts_rowsum_kernel[(num_tokens,)](
            expert_indices,
            expert_scales,
            rowsum,
            num_experts,
            top_k,
            expert_indices.stride(0),
            expert_scales.stride(0),
            BLOCK_K=max(16, triton.next_power_of_2(top_k)),
            num_warps=4,
        )
        _zero_experts_scale_kernel[(triton.cdiv(total, 1024),)](
            rowsum,
            hidden_states,
            out,
            top_k,
            hidden_dim,
            total,
            hidden_states.stride(0),
            hidden_states.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK=1024,
            num_warps=4,
        )
    return out


__all__ = ["zero_experts_identity"]
