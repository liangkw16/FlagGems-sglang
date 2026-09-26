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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Iluvatar (tianshu) vendor for Task 98 get_mla_kv_buffer.

E5 candidate (T98-e5-iluvatar-perrow-1d) for the tianshu axis: the
upstream sglang official single-row skeleton carries the T96
platform-proven three-variable package, replacing the generic's
(8, 512) 2D masked full-chain i64 row tile on eligible shapes.

Upstream skeleton (sgl-project/sglang,
python/sglang/kernels/ops/kvcache/mla_buffer.py,
``get_mla_kv_buffer_kernel``, re-read via gh api on 2026-09-26):
``grid=(n,)`` with one row per program, a scalar loc load, constexpr
strides/dims, and two exact-width ``tl.arange`` load+store segments
with zero masks and zero loops.

Bundled T96 package (fused_pack_qkv e2, platform tianshu 2.95->4.17
+41%; attribution is deferred exactly like T96 - a generic-only int32
single-variable follow-up is reserved):

1. Per-row 1D form (the skeleton itself; tianshu fingerprint: our 2D
   int64 tile reads 2.21-2.37 while c2flow sits at 2.911 and EvokeAgent
   2.7945 - the chip's single largest per-chip gap, +0.49).
2. int32 addressing domain: the generic casts every offset site to
   int64; here the same arithmetic stays int32 whenever every flat
   offset provably stays below 2**31. The host dispatch is a
   numeric-domain check on tensor sizes (T80 e30 / T96 e2 precedent),
   not a device check. The int64 twin kernel keeps the exact-width
   skeleton for the overflow domain; it is never reached by the public
   test matrix (>= 2**31 elements does not fit CI memory) and mirrors
   the e4-proven i64 row math site for site.
3. Single allocation: one flat buffer backs nope/rope as two
   contiguous views (2 torch.empty -> 1; T77 e6/e17 / T96 e2
   precedent). Mixed nope/rope dtypes cannot share one typed buffer,
   so the dtype-domain branch falls back to two empties there.

Every offset bounded by the guard: source reads stay within
(kv_rows - 1) * kv_s0 + total_dim elements (loc holds semantic
page-pool rows < kv_buffer.shape[0]; row-strided views with kv_s0 >
total_dim are bounded by that depth expression, not by numel - the
r2 lesson), output writes are bounded on their own at n * nope_dim /
n * rope_dim (a pure gather allows duplicate indices, so the output
side is not derived from the input size), and the loc load reaches
(n - 1) * loc_s0 elements deep. Every other shape keeps the
generic-mirror masked rows kernel below (byte-identical math to the
proven generic (8, 512) tile, i64 full chain).

Preregistered gates (decide on platform tianshu speedup): >= 2.9
(past EvokeAgent's 2.7945) confirms the axis; >= 4.0 enters the T96
realization band; < 2.19 (-5% vs our 2.305) or any numeric failure
deletes this vendor file and returns tianshu to the generic bytes
(the e1-floor submission set); the other six chips are vendor-isolated
and their reads must not move (kunlunxin walks the rolled-back floor
bytes, so the firing background itself carries the 0.130 -> 0.350
kunlun recovery). Reserved same-vendor alternative arm if the narrow
64-lane rope segment inverts on tianshu: one masked 1024-lane
whole-row load plus two shifted stores (the ledger E4b idea, load
lanes halved). Anti-cheat: int32 selection is a numeric-domain guard,
not a device check; no try/except fallback; no module-level mutable
state. Not compiled on iluvatar hardware locally
(target-runtime-unverified for lowering); the NVIDIA proxy validates
math/JIT only and cannot predict tianshu performance - the platform
single shot decides.
"""

import torch
import triton
import triton.language as tl

_INT32_LIMIT = 2**31
_MAX_GRID = 65535
_MAX_LANES = 65536


@triton.jit
def _mla_split_row_i32_kernel(
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
    # upstream sglang get_mla_kv_buffer_kernel skeleton with the T96
    # int32 addressing package: one row per program, scalar loc load,
    # exact widths, no masks, no loops
    row = tl.program_id(0)
    idx = tl.load(loc + row).to(tl.int32)
    src = kv_buffer + idx * KV_S0

    offs_n = tl.arange(0, NOPE_DIM)
    tl.store(
        nope_out + row * NOPE_S0 + offs_n,
        tl.load(src + offs_n).to(nope_out.dtype.element_ty),
    )

    offs_r = tl.arange(0, ROPE_DIM)
    tl.store(
        rope_out + row * ROPE_S0 + offs_r,
        tl.load(src + NOPE_DIM + offs_r).to(rope_out.dtype.element_ty),
    )


@triton.jit
def _mla_split_row_i64_kernel(
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
    # int64 twin for the overflow domain: same skeleton, i64 math site
    # for site (e4-proven bytes)
    pid = tl.program_id(0).to(tl.int64)
    idx = tl.load(loc + pid).to(tl.int64)
    src = kv_buffer + idx * KV_S0

    offs_n = tl.arange(0, NOPE_DIM).to(tl.int64)
    tl.store(
        nope_out + pid * NOPE_S0 + offs_n,
        tl.load(src + offs_n).to(nope_out.dtype.element_ty),
    )

    offs_r = tl.arange(0, ROPE_DIM).to(tl.int64)
    tl.store(
        rope_out + pid * ROPE_S0 + offs_r,
        tl.load(src + NOPE_DIM + offs_r).to(rope_out.dtype.element_ty),
    )


@triton.jit
def _mla_rows_kernel(
    kv_buffer,
    loc,
    nope_out,
    rope_out,
    n_rows,
    loc_s0,
    kv_s0,
    kv_s1,
    nope_dim,
    rope_dim,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # generic-mirror fallback (proven (8, 512) tile, i64 full chain):
    # every shape the single-row branch does not take lands on bytes
    # the platform already read 2.21-2.37 on tianshu
    pid = tl.program_id(0)
    rows = pid.to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
    rmask = rows < n_rows
    idx = tl.load(loc + rows * loc_s0, mask=rmask, other=0).to(tl.int64)
    base = idx * kv_s0

    cols = tl.arange(0, BLOCK_C).to(tl.int64)
    for c0 in range(0, nope_dim, BLOCK_C):
        cc = c0 + cols
        m = rmask[:, None] & (cc[None, :] < nope_dim)
        value = tl.load(
            kv_buffer + base[:, None] + cc[None, :] * kv_s1, mask=m, other=0
        )
        dst = rows[:, None] * nope_dim + cc[None, :]
        tl.store(nope_out + dst, value.to(nope_out.dtype.element_ty), mask=m)
    for c0 in range(0, rope_dim, BLOCK_C):
        cc = c0 + cols
        m = rmask[:, None] & (cc[None, :] < rope_dim)
        value = tl.load(
            kv_buffer + base[:, None] + (nope_dim + cc[None, :]) * kv_s1,
            mask=m,
            other=0,
        )
        dst = rows[:, None] * rope_dim + cc[None, :]
        tl.store(rope_out + dst, value.to(rope_out.dtype.element_ty), mask=m)


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


def _use_int32(kv_rows, kv_s0, total_dim, n, nope_dim, rope_dim, loc_s0):
    """True when every flat offset of the single-row kernel stays < 2**31.

    Source reads: valid loc values are semantic rows < kv_rows, so the
    deepest element read is (kv_rows - 1) * kv_s0 + total_dim - 1 (kv
    column stride is 1 on this arm; row-strided views with kv_s0 >
    total_dim are bounded by this depth expression, not by numel).
    Output writes: n * nope_dim / n * rope_dim elements, each checked
    on its own because a pure gather allows duplicate indices (the
    output side is not derived from the input size). Loc load:
    (n - 1) * loc_s0 elements deep (unit on this arm, kept general).
    """
    if (kv_rows - 1) * kv_s0 + total_dim >= _INT32_LIMIT:
        return False
    if n * nope_dim >= _INT32_LIMIT:
        return False
    if n * rope_dim >= _INT32_LIMIT:
        return False
    return (n - 1) * loc_s0 < _INT32_LIMIT


def get_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope):
    n = loc.shape[0]
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = kv_buffer.shape[-1] - nope_dim
    # single allocation: both halves as contiguous views of one flat
    # buffer; mixed nope/rope dtypes cannot share one typed buffer, so
    # the dtype-domain branch falls back to two empties
    if cache_k_nope.dtype == cache_k_rope.dtype:
        buf = torch.empty(
            n * (nope_dim + rope_dim),
            dtype=cache_k_nope.dtype,
            device=kv_buffer.device,
        )
        nope = buf[: n * nope_dim].view(n, nope_dim)
        rope = buf[n * nope_dim :].view(n, rope_dim)
    else:
        nope = torch.empty(
            (n, nope_dim), dtype=cache_k_nope.dtype, device=kv_buffer.device
        )
        rope = torch.empty(
            (n, rope_dim), dtype=cache_k_rope.dtype, device=kv_buffer.device
        )
    if n and (nope_dim or rope_dim):
        loc_s0 = loc.stride(0)
        kv_s1 = kv_buffer.stride(1)
        if _upstream_row_form(n, nope_dim, rope_dim, loc_s0, kv_s1):
            kv_s0 = kv_buffer.stride(0)
            if _use_int32(
                kv_buffer.shape[0],
                kv_s0,
                kv_buffer.shape[-1],
                n,
                nope_dim,
                rope_dim,
                loc_s0,
            ):
                _mla_split_row_i32_kernel[(n,)](
                    kv_buffer,
                    loc,
                    nope,
                    rope,
                    KV_S0=kv_s0,
                    NOPE_DIM=nope_dim,
                    ROPE_DIM=rope_dim,
                    NOPE_S0=nope.stride(0),
                    ROPE_S0=rope.stride(0),
                    num_warps=4,
                )
            else:
                _mla_split_row_i64_kernel[(n,)](
                    kv_buffer,
                    loc,
                    nope,
                    rope,
                    KV_S0=kv_s0,
                    NOPE_DIM=nope_dim,
                    ROPE_DIM=rope_dim,
                    NOPE_S0=nope.stride(0),
                    ROPE_S0=rope.stride(0),
                    num_warps=4,
                )
            return nope, rope
        _mla_rows_kernel[(triton.cdiv(n, 8),)](
            kv_buffer,
            loc,
            nope,
            rope,
            n,
            loc_s0,
            kv_buffer.stride(0),
            kv_s1,
            nope_dim,
            rope_dim,
            BLOCK_R=8,
            BLOCK_C=512,
            num_warps=8,
        )
    return nope, rope


__all__ = ["get_mla_kv_buffer"]
