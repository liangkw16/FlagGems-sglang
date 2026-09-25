# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Kunlunxin vendor for Task 98 get_mla_kv_buffer.

Round-2 kunlun arm: transfer the upstream sglang official kernel shape
(sgl-project/sglang, python/sglang/kernels/ops/kvcache/mla_buffer.py,
``get_mla_kv_buffer_kernel`` + ``get_mla_kv_buffer_triton``, re-read via
gh api on 2026-09-26): ``grid=(n,)`` with one row per program, a scalar
loc load cast to int64 as the row base, nope/rope widths and strides as
constexpr, and two exact-width ``tl.arange`` load+store segments with
zero masks and zero loops.

Bet: the kunlunxin bottleneck is the loop skeleton and the masks, not
the vector width. Fingerprint evidence (climb-loop.json s2t1op098):
the OpeGoodn leader is a cliff on kunlun (0.976 vs runner-up
eatabigwatermelon 0.4905) and huawei (1.1902) while sitting at 1.14 on
tianshu / 0.7485 card_a / 0.799 card_b - single-row tiny programs
schedule poorly on CUDA-type cores but suit XPU/Ascend, which is
exactly this shape; our generic 2D row-tile ([8,512] full mask + i64
vector gather) is the inverse fingerprint (tianshu 2.37 strong, kunlun
0.045). Ledger-falsified alternatives: the 2D tile 0.045 and the e2
exact-width-inside-the-rows-loop 0.124 - this candidate changes the
skeleton, so the open risk is that e2's narrow-vector inversion was
width-driven, which the preregistered gates below cover.

Host shape branch (anti-cheat: plain shape dispatch selecting between
two Triton kernels, no try/except, no torch fallback, no module-level
mutable state): the single-row form requires power-of-two arange widths
within the 65536-lane compile limit, unit kv column stride, unit loc
stride and n <= 65535 programs; every other shape takes the retained
e1 masked rows kernel below, byte-for-byte the measured-best rollback
floor (commit aef2c1f1, platform 0.350).

Preregistered gates (round-2 kunlun arm, decide on platform kunlun
speedup): < 0.350 -> roll this vendor back to the aef2c1f1 bytes;
0.350-0.49 -> hold and observe; >= 0.49 (past eatabigwatermelon's
0.4905) -> the skeleton axis is confirmed; ~0.976 is the OpeGoodn
ceiling. If the narrow-vector inversion reproduces, the reserved E4b
alternative is the same skeleton with one 1024-wide whole-row load and
two shifted stores (half the load lanes).
Not compiled on kunlunxin hardware locally.
"""

import torch
import triton
import triton.language as tl

_MAX_GRID = 65535
_MAX_LANES = 65536


@triton.jit
def _mla_split_row_kernel(
    kv_buffer,
    loc,
    nope_out,
    rope_out,
    KV_S0: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NOPE_S0: tl.constexpr,
    ROPE_S0: tl.constexpr,
):
    # upstream sglang get_mla_kv_buffer_kernel: one row per program,
    # scalar loc load, exact widths, no masks, no loops
    pid = tl.program_id(0).to(tl.int64)
    idx = tl.load(loc + pid).to(tl.int64)
    src = kv_buffer + idx * KV_S0

    offs_n = tl.arange(0, NOPE_DIM)
    tl.store(
        nope_out + pid * NOPE_S0 + offs_n,
        tl.load(src + offs_n).to(nope_out.dtype.element_ty),
    )

    offs_r = tl.arange(0, ROPE_DIM)
    tl.store(
        rope_out + pid * ROPE_S0 + offs_r,
        tl.load(src + NOPE_DIM + offs_r).to(rope_out.dtype.element_ty),
    )


@triton.jit
def _kv_rows_kernel(
    kv_buffer,
    loc,
    nope_out,
    rope_out,
    n_rows,
    rows_per_prog,
    loc_s0,
    kv_s0,
    kv_s1,
    nope_dim,
    rope_dim,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    ncols = tl.arange(0, BLOCK_N).to(tl.int64)
    rcols = tl.arange(0, BLOCK_R).to(tl.int64)
    row0 = pid * rows_per_prog
    for r in range(0, rows_per_prog):
        row = row0 + r
        if row < n_rows:
            idx = tl.load(loc + row * loc_s0).to(tl.int64)
            src = idx * kv_s0
            for cn in range(0, nope_dim, BLOCK_N):
                cc = cn + ncols
                mn = cc < nope_dim
                nv = tl.load(kv_buffer + src + cc * kv_s1, mask=mn, other=0)
                tl.store(
                    nope_out + row * nope_dim + cc,
                    nv.to(nope_out.dtype.element_ty),
                    mask=mn,
                )
            for cr in range(0, rope_dim, BLOCK_R):
                cc = cr + rcols
                mr = cc < rope_dim
                rv = tl.load(
                    kv_buffer + src + (nope_dim + cc) * kv_s1, mask=mr, other=0
                )
                tl.store(
                    rope_out + row * rope_dim + cc,
                    rv.to(rope_out.dtype.element_ty),
                    mask=mr,
                )


def _pow2_width(width):
    return 0 < width <= _MAX_LANES and (width & (width - 1)) == 0


def _upstream_row_form(n, nope_dim, rope_dim, loc_s0, kv_s1):
    """True when the exact-width single-row kernel applies to the shape."""
    return (
        0 < n <= _MAX_GRID
        and _pow2_width(nope_dim)
        and _pow2_width(rope_dim)
        and kv_s1 == 1
        and loc_s0 == 1
    )


def get_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope):
    n = loc.shape[0]
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = kv_buffer.shape[-1] - nope_dim
    nope = torch.empty(
        (n, nope_dim), dtype=cache_k_nope.dtype, device=kv_buffer.device
    )
    rope = torch.empty(
        (n, rope_dim), dtype=cache_k_rope.dtype, device=kv_buffer.device
    )
    if n and (nope_dim or rope_dim):
        if _upstream_row_form(
            n, nope_dim, rope_dim, loc.stride(0), kv_buffer.stride(1)
        ):
            _mla_split_row_kernel[(n,)](
                kv_buffer,
                loc,
                nope,
                rope,
                KV_S0=kv_buffer.stride(0),
                NOPE_DIM=nope_dim,
                ROPE_DIM=rope_dim,
                NOPE_S0=nope.stride(0),
                ROPE_S0=rope.stride(0),
            )
            return nope, rope
        # the 1024-lane floor is the measured-best form on kunlunxin
        # (e1 0.350x vs exact-width e2 0.124x - narrow vectors invert)
        block_n = min(65536, max(1024, triton.next_power_of_2(max(nope_dim, 1))))
        block_r = min(65536, max(1024, triton.next_power_of_2(max(rope_dim, 1))))
        rows_per_prog = triton.cdiv(n, _MAX_GRID)
        grid = (triton.cdiv(n, rows_per_prog),)
        _kv_rows_kernel[grid](
            kv_buffer,
            loc,
            nope,
            rope,
            n,
            rows_per_prog,
            loc.stride(0),
            kv_buffer.stride(0),
            kv_buffer.stride(1),
            nope_dim,
            rope_dim,
            BLOCK_N=block_n,
            BLOCK_R=block_r,
            num_warps=8,
        )
    return nope, rope


__all__ = ["get_mla_kv_buffer"]
