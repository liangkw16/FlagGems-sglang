# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang fd32226 kernels/ops/diffusion/norm/group_norm_silu_triton.py.

# Kunlunxin vendor, E3: the S0 2D tile (8192-lane cap) and E1 (2048)
# both hit OutOfResources: uni_sram on real executions; E2's 1D form
# tripped the intermittent evaluator crash instead. E3 returns to the
# 2D tile shape but caps tiles at 512 lanes and pins num_warps=1 -
# between the two known-budget failure points.

import torch
import triton
import triton.language as tl


@triton.jit
def _group_norm_silu(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    G,
    spatial,
    group_channels,
    eps,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    g_idx = pid % G
    base = pid.to(tl.int64) * group_channels * spatial
    wbase = g_idx * group_channels
    cols = tl.arange(0, BLOCK_C)
    cm = cols < group_channels
    # Loop-carried scalars only (the tensor-carry scan is the proven
    # GCU300 PassManager poison); the [C, S] tile avoids per-element
    # runtime division for the channel lookup entirely.
    total = group_channels * spatial
    # Variance is computed about the mean (a third pass) because
    # E[x^2]-mean^2 cancels catastrophically for large-mean fp32 groups.
    sum_ = tl.zeros((), dtype=tl.float32)
    for s0 in range(0, spatial, BLOCK_S):
        rows = s0 + tl.arange(0, BLOCK_S)
        m = cm[:, None] & (rows[None, :] < spatial)
        v = tl.load(
            x_ptr
            + base
            + cols[:, None].to(tl.int64) * spatial
            + rows[None, :],
            mask=m,
            other=0.0,
        ).to(tl.float32)
        sum_ += tl.sum(v)
    mean = sum_ / total
    sumsq = tl.zeros((), dtype=tl.float32)
    for s0 in range(0, spatial, BLOCK_S):
        rows = s0 + tl.arange(0, BLOCK_S)
        m = cm[:, None] & (rows[None, :] < spatial)
        v = tl.load(
            x_ptr
            + base
            + cols[:, None].to(tl.int64) * spatial
            + rows[None, :],
            mask=m,
            other=0.0,
        ).to(tl.float32)
        # Arithmetic mask (not tl.where) zeroes the padded lanes before
        # the reduction - other=0.0 would contribute (0-mean)^2 here.
        d = (v - mean) * m.to(tl.float32)
        sumsq += tl.sum(d * d)
    var = sumsq / total
    inv = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + wbase + cols, mask=cm, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + wbase + cols, mask=cm, other=0.0).to(tl.float32)
    for s0 in range(0, spatial, BLOCK_S):
        rows = s0 + tl.arange(0, BLOCK_S)
        m = cm[:, None] & (rows[None, :] < spatial)
        v = tl.load(
            x_ptr
            + base
            + cols[:, None].to(tl.int64) * spatial
            + rows[None, :],
            mask=m,
            other=0.0,
        ).to(tl.float32)
        y = (v - mean) * inv * w[:, None] + b[:, None]
        y = y * (1.0 / (1.0 + tl.exp(-y)))
        tl.store(
            out_ptr
            + base
            + cols[:, None].to(tl.int64) * spatial
            + rows[None, :],
            y.to(out_ptr.dtype.element_ty),
            mask=m,
        )


def group_norm_silu(x, weight, bias, num_groups, eps):
    assert x.ndim >= 2
    assert x.shape[1] == weight.numel() == bias.numel()
    channels = x.shape[1]
    assert channels % num_groups == 0
    xc = x.contiguous()
    spatial = 1
    for s in xc.shape[2:]:
        spatial *= s
    group_channels = channels // num_groups
    out = torch.empty_like(xc)
    n_groups_total = xc.shape[0] * num_groups
    if n_groups_total and spatial:
        block_c = triton.next_power_of_2(group_channels)
        block_s = min(triton.next_power_of_2(spatial), max(32, 512 // block_c))
        _group_norm_silu[(min(n_groups_total, 65535),)](
            xc,
            weight,
            bias,
            out,
            num_groups,
            spatial,
            group_channels,
            eps,
            BLOCK_C=block_c,
            BLOCK_S=block_s,
            num_warps=1,
        )
    return out.reshape(x.shape)


__all__ = ["group_norm_silu"]
