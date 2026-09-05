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

# Kunlunxin vendor: three-kernel split with 3D micro-programs and a
# rank-1 width reduction (Codex P1). The scalar FMA chain miscompiles
# deterministically on this backend (E4-E6 identical wrong values) and
# the E7 [W_PAD, BLOCK_D] 2D-tile form hits the uni_sram compile wall,
# so the conv kernel holds ONLY a [W_PAD] vector and one tl.sum; bias,
# SiLU and the output cast live in a separate flat kernel (T53
# vectorized-flat recipe) and the state copy is a plain flat copy.

import torch
import triton
import triton.language as tl

_MAX_GRID = 65535


@triton.jit(do_not_specialize=["M"])
def _ccu_conv_gemm_kernel(
    win_ptr,
    weight_ptr,
    pre_ptr,
    M,
    seqlen,
    dim,
    a_matrix_stride,
    BLOCK_N: tl.constexpr,
    WIDTH: tl.constexpr,
    W_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # E10 (Codex P2): the conv as a regular per-channel GEMM - windows
    # A[D, B*S, W] dotted against the broadcast weight vector. Both the
    # rank-1 tl.sum form (E8/E9) and every FMA-chain form miscompile on
    # kunlunxin, while this regular-GEMM family is proven correct there
    # (T45 e8/e9 platform evidence).
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = 1  # single N tile: only column 0 of C is meaningful
    tiles_per_matrix = num_pid_m * num_pid_n
    d = pid // tiles_per_matrix
    local = pid - d * tiles_per_matrix
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = local // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (local % group_size_m)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, W_PAD)
    a_ptrs = (
        win_ptr
        + d * a_matrix_stride
        + offs_m[:, None] * WIDTH
        + offs_k[None, :]
    )
    # B operand: the [W] weight vector broadcast across N columns
    # (stride 0 on the n axis keeps this a fully regular access).
    b_ptrs = weight_ptr + d.to(tl.int64) * WIDTH + offs_k[:, None] + 0 * offs_n[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, WIDTH, W_PAD):
        mask_k = offs_k < WIDTH - k
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & (offs_n[None, :] < BLOCK_N),
            other=0.0,
        )
        accumulator = tl.dot(a, b, acc=accumulator, input_precision="ieee")
        a_ptrs += W_PAD
        b_ptrs += W_PAD

    # Store only column 0 into pre[b, d, t] (row m encodes (b, t)).
    b_idx = offs_m // seqlen
    t_idx = offs_m % seqlen
    pre_off = (b_idx * dim + d) * seqlen + t_idx
    store_mask = (offs_m[:, None] < M) & (offs_n[None, :] == 0)
    tl.store(
        pre_ptr + (pre_off[:, None] + 0 * offs_n[None, :]),
        accumulator,
        mask=store_mask,
    )


@triton.jit
def _ccu_postprocess_kernel(
    pre_ptr,
    bias_ptr,
    out_ptr,
    total,
    seqlen,
    dim,
    HAS_BIAS: tl.constexpr,
    ACT_IS_SILU: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    for start in range(pid * BLOCK, total, step):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < total
        v = tl.load(pre_ptr + idx, mask=mask, other=0.0)
        if HAS_BIAS:
            d = (idx // seqlen) % dim
            v += tl.load(bias_ptr + d, mask=mask, other=0.0)
        if ACT_IS_SILU:
            # SiLU in the statement's exact form; stability rewrites
            # fail the checker at large negative inputs.
            v = v / (1.0 + tl.exp(-v))
        tl.store(out_ptr + idx, v.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _ccu_state_copy_kernel(
    xcat_ptr,
    new_state_ptr,
    total,
    seqlen,
    state_len,
    l_cat,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    for start in range(pid * BLOCK, total, step):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < total
        row = idx // state_len
        i = idx % state_len
        v = tl.load(
            xcat_ptr + row.to(tl.int64) * l_cat + seqlen + i, mask=mask, other=0.0
        )
        tl.store(
            new_state_ptr + idx,
            v.to(new_state_ptr.dtype.element_ty),
            mask=mask,
        )


def causal_conv1d_update(x, conv_state, weight, bias=None, activation="silu"):
    squeeze_out = x.dim() == 2
    if squeeze_out:
        x = x.unsqueeze(-1)
    orig_dtype = x.dtype
    # Concatenate state+x along time axis (data layout, not computation)
    x_cat = torch.cat([conv_state.float(), x.float()], dim=-1).contiguous()
    weight_f = weight.contiguous().float()  # [D, W]
    if bias is not None:
        bias = bias.contiguous().float()
    batch, dim, seqlen = x.shape
    state_len = conv_state.shape[-1]
    width = weight.shape[1]
    out = torch.empty(
        batch, dim, seqlen, dtype=torch.float32, device=x.device
    )
    new_state = torch.empty(
        conv_state.shape, dtype=conv_state.dtype, device=x.device
    )
    if batch * dim == 0:
        out = out.to(orig_dtype)
        if squeeze_out:
            out = out.squeeze(-1)
        return out, new_state

    w_pad = max(triton.next_power_of_2(width), 16)  # tl.dot needs K >= 16
    l_cat = state_len + seqlen
    pre = torch.empty(
        batch * dim * seqlen, dtype=torch.float32, device=x.device
    )

    # Kernel 1 (E10): windows [D, B*S, W] via unfold, one regular GEMM
    # per channel against the broadcast weight vector. Window for
    # output t starts at t + state_len + 1 - width in x_cat.
    win_start = state_len + 1 - width
    win = x_cat.unfold(-1, width, 1)[
        ..., win_start : win_start + seqlen, :
    ]
    a_mat = (
        win.permute(1, 0, 2, 3).contiguous().view(dim, batch * seqlen, width)
    )
    m_rows = batch * seqlen
    grid1 = (dim * triton.cdiv(m_rows, 32),)
    _ccu_conv_gemm_kernel[grid1](
        a_mat,
        weight_f,
        pre,
        m_rows,
        seqlen,
        dim,
        a_mat.stride(0),
        BLOCK_N=32,
        WIDTH=width,
        W_PAD=w_pad,
        BLOCK_M=32,
        GROUP_M=8,
        num_warps=4,
        num_stages=1,
    )

    # Kernel 2: flat bias + SiLU + dtype cast (out is contiguous).
    total = batch * dim * seqlen
    grid2 = (min(triton.cdiv(total, 1024), _MAX_GRID),)
    _ccu_postprocess_kernel[grid2](
        pre,
        bias if bias is not None else pre,
        out,
        total,
        seqlen,
        dim,
        HAS_BIAS=bias is not None,
        ACT_IS_SILU=(activation in ("silu", "swish")),
        BLOCK=1024,
        num_warps=4,
        num_stages=1,
    )

    # Kernel 3: flat state copy from the tail of x_cat.
    total_ns = batch * dim * state_len
    grid3 = (min(triton.cdiv(total_ns, 1024), _MAX_GRID),)
    _ccu_state_copy_kernel[grid3](
        x_cat,
        new_state,
        total_ns,
        seqlen,
        state_len,
        l_cat,
        BLOCK=1024,
        num_warps=4,
        num_stages=1,
    )

    out = out.to(orig_dtype)
    if squeeze_out:
        out = out.squeeze(-1)
    return out, new_state


__all__ = ["causal_conv1d_update"]
