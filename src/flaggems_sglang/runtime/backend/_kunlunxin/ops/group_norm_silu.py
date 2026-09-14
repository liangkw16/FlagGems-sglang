# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang fd32226 kernels/ops/diffusion/norm/
# group_norm_silu_triton.py.

# Kunlunxin vendor, E5. E1/E3/E4 proved that shrinking the [C, S] tile
# (8192/2048/512/64 lanes) never satisfied "uni_sram", and FlagTree #1126
# shows that label is a wrapper around ANY make_ttxir pass failure - so
# E5 abandons the 2D tile entirely and adopts the loop skeleton of
# FlagGems master _kunlunxin/ops/native_group_norm.py, the only form
# known to compile this op class on this stack: a FLAT 1D reduce over
# the group's contiguous span with [BLOCK_HW] vector accumulators and a
# single tl.sum at the end, then a per-channel normalize whose
# GROUP_SIZE loop unrolls statically around scalar weight/bias loads
# and contiguous BLOCK_HW stores (no idx // spatial gather anywhere).
# Two deliberate deviations from master: variance stays centered about
# the mean (E[x^2] - mean^2 cancels for large-mean groups - the
# generic's regression matrix pins this), and silu is fused into the
# normalize pass in fp32 with the output cast at the store, matching
# the reference contract exactly.

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["eps"])
def _group_norm_silu_kx(
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

    # Centered second pass: (x - mean)^2 accumulated in vector form, one
    # tl.sum at the end (E[x^2] - mean^2 is numerically unsafe here).
    sq_acc = tl.zeros([BLOCK_HW], dtype=tl.float32)
    for off in range(0, num_elements, BLOCK_HW):
        idx = off + tl.arange(0, BLOCK_HW)
        m = idx < num_elements
        x = tl.load(x_ptr + base + idx, mask=m, other=0.0).to(tl.float32)
        d = x - mean
        sq_acc += d * d
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
    if n_groups_total and spatial:
        # Master's proven ceiling: block_hw of 1024 lanes. The grid is
        # one program per (n, group) - same as generic and FlagGems
        # master; N * num_groups stays far below the 65535 grid limit
        # for every shape this operator serves.
        block_hw = min(triton.next_power_of_2(spatial), 1024)
        _group_norm_silu_kx[(n_groups_total,)](
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
