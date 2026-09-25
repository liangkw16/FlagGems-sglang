# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 104 rmsnorm_hf: HuggingFace LlamaRMSNorm semantics - the
# normalized activation is rounded back to the input dtype BEFORE the
# weight multiply (unlike the all-fp32 fused_rmsnorm). Both kernels
# below reproduce that order exactly: fp32 mean-of-squares reduction,
# fp32 rsqrt scale, y rounded to the output dtype, then the weight
# multiply in fp32 with one final rounding at the store.
#
# Dispatch (host side):
# - hidden > 1024 and 256 | hidden -> exact-subblock two-pass kernel
#   with D_TILE = min(hidden & -hidden, 1024). Every sub-block covers
#   exact lanes, so no phase carries a bounds mask or a masked-load
#   `other` prefill (Ascend: int32 vector CMP degrades to scalar and
#   prefill serializes MTE2 - chip-rulesets). The fp32 live set drops
#   from a whole next_pow2(hidden) row (32KB per program at hidden>=
#   8192, beyond Metax max_tile_size=2048) to a single D_TILE chunk;
#   x is re-read in pass 2 instead of staying live across the
#   reduction (T51 fla_layernorm_gated E10 structure, Huawei +10%,
#   8/8 valid). Sub-block loops use tl.range (not static unrolling) so
#   Ascend UB holds one live tile per iteration.
# - otherwise -> the platform-proven whole-row masked kernel (S0 bytes).

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_hf_kernel(
    x,
    weight,
    out,
    x_s0,
    x_s1,
    w_s0,
    o_s0,
    hidden,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < hidden
    xv = tl.load(x + row * x_s0 + cols * x_s1, mask=m, other=0.0).to(
        tl.float32
    )
    mean_square = tl.sum(xv * xv, axis=0) / hidden
    y = (xv * tl.rsqrt(mean_square + eps)).to(out.dtype.element_ty)
    w = tl.load(weight + cols * w_s0, mask=m, other=0.0).to(tl.float32)
    tl.store(
        out + row * o_s0 + cols,
        (y.to(tl.float32) * w).to(out.dtype.element_ty),
        mask=m,
    )


@triton.jit
def _rmsnorm_hf_subblock_kernel(
    x,
    weight,
    out,
    x_s0,
    x_s1,
    w_s0,
    o_s0,
    hidden,
    eps,
    D_TILE: tl.constexpr,
    N_SUB: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    x_row = x + row * x_s0
    o_row = out + row * o_s0
    offs = tl.arange(0, D_TILE)
    # Pass 1: accumulate sum(x^2) over exact sub-blocks (no masks).
    sum_sq = 0.0
    for s in tl.range(0, N_SUB):
        xv = tl.load(x_row + (s * D_TILE + offs) * x_s1).to(tl.float32)
        sum_sq += tl.sum(xv * xv, axis=0)
    rstd = tl.rsqrt(sum_sq / hidden + eps)
    # Pass 2: re-read x, scale, round to the output dtype, then the
    # fp32 weight multiply - the same math order as the whole-row path.
    for s in tl.range(0, N_SUB):
        cols = s * D_TILE + offs
        xv = tl.load(x_row + cols * x_s1).to(tl.float32)
        y = (xv * rstd).to(out.dtype.element_ty)
        w = tl.load(weight + cols * w_s0).to(tl.float32)
        tl.store(
            o_row + cols,
            (y.to(tl.float32) * w).to(out.dtype.element_ty),
        )


def rmsnorm_hf(input, weight, eps):
    assert input.dim() == 2
    rows, hidden = input.shape
    # torch promotes weight * y per type rules: a wider weight dtype
    # widens the result beyond the input dtype
    out_dtype = torch.promote_types(weight.dtype, input.dtype)
    out = torch.empty(
        (rows, hidden), dtype=out_dtype, device=input.device
    )
    if rows and hidden:
        if hidden > 1024 and hidden % 256 == 0:
            d_tile = min(hidden & (-hidden), 1024)
            _rmsnorm_hf_subblock_kernel[(rows,)](
                input,
                weight,
                out,
                input.stride(0),
                input.stride(1),
                weight.stride(0),
                out.stride(0),
                hidden,
                eps,
                D_TILE=d_tile,
                N_SUB=hidden // d_tile,
                num_warps=4,
                num_stages=1,
            )
        else:
            block = max(16, triton.next_power_of_2(hidden))
            _rmsnorm_hf_kernel[(rows,)](
                input,
                weight,
                out,
                input.stride(0),
                input.stride(1),
                weight.stride(0),
                out.stride(0),
                hidden,
                eps,
                BLOCK=block,
                num_warps=8,
            )
    return out


__all__ = ["rmsnorm_hf"]
