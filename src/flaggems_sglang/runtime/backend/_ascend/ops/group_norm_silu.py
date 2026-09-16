# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang fd32226 kernels/ops/diffusion/norm/
# group_norm_silu_triton.py.

# Ascend vendor, e6: the flat skeleton that unlocked this op on Kunlunxin
# (E5, submission 14528), ported onto the Ascend rule set. One program
# owns one (n, group): a flat 1D reduce over the group's contiguous span
# with [BLOCK_HW] vector accumulators and a single tl.sum at the end,
# then a per-channel normalize whose GROUP_SIZE loop unrolls statically
# around scalar weight/bias loads and contiguous BLOCK_HW block stores -
# no 2D tile, no idx // spatial gather. Ascend-specific discipline:
# BLOCK_HW stays <= 1024 lanes (UB budget), the centered-variance mask
# is arithmetic (tl.where on the reduction is the known GCU poison and
# triton-ascend #1610 wants the reduction axis on the fastest lane dim,
# which a flat 1D span gives directly), silu runs in fp32 with the
# output cast at the store, and eps is do_not_specialize'd. The padded
# lanes of the centered pass are explicitly zeroed - a bare d * d would
# add mean^2 per padded lane whenever num_elements is not a multiple of
# BLOCK_HW. Variance stays centered about the mean on purpose.

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["eps"])
def _group_norm_silu_small(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    num_groups,
    group_channels,
    spatial,
    eps,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # Resident centered variance from FlagGems a7620cc1 ops/groupnorm.py.
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    rows = tl.arange(0, BLOCK_S)
    cm = cols < group_channels
    mask = cm[:, None] & (rows[None, :] < spatial)
    offsets = (
        pid.to(tl.int64) * group_channels * spatial
        + cols[:, None].to(tl.int64) * spatial
        + rows[None, :]
    )
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    total = group_channels * spatial
    mean = tl.sum(x) / total
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered) / total
    wbase = (pid % num_groups) * group_channels
    w = tl.load(w_ptr + wbase + cols, mask=cm, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + wbase + cols, mask=cm, other=0.0).to(tl.float32)
    y = centered * tl.rsqrt(var + eps) * w[:, None] + b[:, None]
    y = y * (1.0 / (1.0 + tl.exp(-y)))
    tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["eps"])
def _group_norm_silu_asc(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    num_groups,
    group_channels,
    spatial,
    eps,
    GROUP_SIZE: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    num_elements = group_channels * spatial
    base = pid * num_elements
    wbase = (pid % num_groups) * group_channels

    sum_acc = tl.zeros([BLOCK_HW], dtype=tl.float32)
    for off in range(0, num_elements, BLOCK_HW):
        idx = off + tl.arange(0, BLOCK_HW)
        m = idx < num_elements
        x = tl.load(x_ptr + base + idx, mask=m, other=0.0).to(tl.float32)
        sum_acc += x
    mean = tl.sum(sum_acc) / num_elements

    sq_acc = tl.zeros([BLOCK_HW], dtype=tl.float32)
    for off in range(0, num_elements, BLOCK_HW):
        idx = off + tl.arange(0, BLOCK_HW)
        m = idx < num_elements
        x = tl.load(x_ptr + base + idx, mask=m, other=0.0).to(tl.float32)
        d = x - mean
        sq_acc += tl.where(m, d * d, 0.0)
    var = tl.sum(sq_acc) / num_elements
    rstd = tl.rsqrt(var + eps)

    for c in range(0, GROUP_SIZE):
        w = tl.load(w_ptr + wbase + c).to(tl.float32)
        b = tl.load(b_ptr + wbase + c).to(tl.float32)
        cbase = base + c * spatial
        for off in range(0, spatial, BLOCK_HW):
            idx = off + tl.arange(0, BLOCK_HW)
            m = idx < spatial
            x = tl.load(x_ptr + cbase + idx, mask=m, other=0.0).to(tl.float32)
            y = (x - mean) * rstd * w + b
            y = y * (1.0 / (1.0 + tl.exp(-y)))
            y = y.to(out_ptr.dtype.element_ty)
            tl.store(out_ptr + cbase + idx, y, mask=m)


def group_norm_silu(x, weight, bias, num_groups, eps):
    assert x.ndim >= 2
    assert x.shape[1] == weight.numel() == bias.numel()
    channels = x.shape[1]
    assert channels % num_groups == 0
    xc = x.contiguous()
    w = weight.contiguous()
    b = bias.contiguous()
    spatial = 1
    for s in xc.shape[2:]:
        spatial *= s
    group_channels = channels // num_groups
    out = torch.empty_like(xc)
    n_groups_total = xc.shape[0] * num_groups
    if n_groups_total and spatial and group_channels:
        block_c = triton.next_power_of_2(group_channels)
        block_s = triton.next_power_of_2(spatial)
        if block_c * block_s <= 2048:
            _group_norm_silu_small[(n_groups_total,)](
                xc,
                w,
                b,
                out,
                num_groups,
                group_channels,
                spatial,
                eps,
                BLOCK_C=block_c,
                BLOCK_S=block_s,
            )
        else:
            block_hw = min(block_s, 1024)
            _group_norm_silu_asc[(n_groups_total,)](
                xc,
                w,
                b,
                out,
                num_groups,
                group_channels,
                spatial,
                eps,
                GROUP_SIZE=group_channels,
                BLOCK_HW=block_hw,
            )
    return out.reshape(x.shape)


__all__ = ["group_norm_silu"]
