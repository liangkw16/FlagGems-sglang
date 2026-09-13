"""Kunlunxin vendor for group_norm_silu (1D spatial form)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _group_norm_silu_1d(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    G,
    spatial,
    group_channels,
    eps,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    g_idx = pid % G
    base = pid.to(tl.int64) * group_channels * spatial
    wbase = g_idx * group_channels
    total = group_channels * spatial
    sum_ = tl.zeros((), dtype=tl.float32)
    for s0 in range(0, spatial, BLOCK_S):
        rows = s0 + tl.arange(0, BLOCK_S)
        m = rows < spatial
        v = tl.load(x_ptr + base + rows, mask=m, other=0.0).to(tl.float32)
        sum_ += tl.sum(v)
    mean = sum_ / total
    sumsq = tl.zeros((), dtype=tl.float32)
    for s0 in range(0, spatial, BLOCK_S):
        rows = s0 + tl.arange(0, BLOCK_S)
        m = rows < spatial
        v = tl.load(x_ptr + base + rows, mask=m, other=0.0).to(tl.float32)
        d = (v - mean) * m.to(tl.float32)
        sumsq += tl.sum(d * d)
    var = sumsq / total
    inv = tl.rsqrt(var + eps)
    for c in range(0, group_channels):
        w = tl.load(w_ptr + wbase + c).to(tl.float32)
        b = tl.load(b_ptr + wbase + c).to(tl.float32)
        cbase = base + c.to(tl.int64) * spatial
        for s0 in range(0, spatial, BLOCK_S):
            rows = s0 + tl.arange(0, BLOCK_S)
            m = rows < spatial
            v = tl.load(x_ptr + cbase + rows, mask=m, other=0.0).to(tl.float32)
            y = (v - mean) * inv * w + b
            y = y * (1.0 / (1.0 + tl.exp(-y)))
            tl.store(
                out_ptr + cbase + rows,
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
        block_s = min(triton.next_power_of_2(spatial), 1024)
        _group_norm_silu_1d[(min(n_groups_total, 65535),)](
            xc,
            weight,
            bias,
            out,
            num_groups,
            spatial,
            group_channels,
            eps,
            BLOCK_S=block_s,
            num_warps=1,
        )
    return out.reshape(x.shape)


__all__ = ["group_norm_silu"]
