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

"""Iluvatar (tianshu) vendor for Task 96 fused_pack_qkv.

E2 candidate for the tianshu axis. Platform reads put our generic 2D
row-tile at 2.95x on tianshu while the other 13 teams sit at 4.6-6.1
(climb-loop s2t1op096, 2026-09-26); the s0->e1 byte reads differ by
0.2% for this chip, so the gap is structural, not water level. Three
bundled changes vs generic (attribution is deferred: a generic-only
int32 single-variable follow-up is reserved as E3):

1. Per-row 1D form from the upstream sglang prototype
   (``_fused_pack_qkv_kernel``): ``grid=(n_valid,)`` with one program
   per row, ``BLOCK_HD = next_pow2(row_elems)`` as a single masked 1D
   segment, and a scalar index load per program. In-task sibling
   evidence: the kunlunxin 2D tile read 0.010x while 1D row programs
   read 0.583x (57x); T51 E9 showed tianshu prefers per-row. The
   column loop below only fires for rows wider than the 65536-lane
   cap (test_row_elems_beyond_block_cap, H*D=70000).
2. int32 addressing domain. The generic casts all five offset sites
   (rows/idx/src/dst/cols) to int64; here the same arithmetic stays
   int32 whenever every flat offset provably stays below 2**31. The
   host dispatch below is a numeric-domain check on tensor sizes
   (T80 e30 precedent), not a device check. The int64 twin kernel is
   kept for the overflow domain; it is never reached by the public
   test matrix (>= 2**31 elements does not fit CI memory) and mirrors
   the generic's proven int64 math site for site.
3. Single allocation: one flat buffer backs q_out/k_out/v_out as
   three contiguous views (3 torch.empty -> 1; T77 e6/e17 precedent,
   tianshu +8.6%).

Every offset bounded by the guard: source reads idx * row_elems +
cc < q.numel() (indices are semantic B*S positions), output writes
row * row_elems + cc < n * row_elems <= q.numel(), and the indices
load reaches (n - 1) * idx_s0 elements deep (strided views included).
Not compiled on iluvatar hardware locally (target-runtime-unverified
for lowering); the NVIDIA proxy validates math/JIT only and cannot
predict tianshu performance - the platform single shot decides.
"""

import torch
import triton
import triton.language as tl

_INT32_LIMIT = 2**31
_MAX_LANES = 65536
_MIN_LANES = 128


@triton.jit
def _pack_row_i32_kernel(
    q,
    k,
    v,
    indices,
    q_out,
    k_out,
    v_out,
    row_elems,
    idx_s0,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.load(indices + row * idx_s0).to(tl.int32)
    src = idx * row_elems
    dst = row * row_elems
    cols = tl.arange(0, BLOCK_C)
    for c0 in range(0, row_elems, BLOCK_C):
        cc = c0 + cols
        m = cc < row_elems
        qv = tl.load(q + src + cc, mask=m, other=0)
        tl.store(q_out + dst + cc, qv, mask=m)
        kv = tl.load(k + src + cc, mask=m, other=0)
        tl.store(k_out + dst + cc, kv, mask=m)
        vv = tl.load(v + src + cc, mask=m, other=0)
        tl.store(v_out + dst + cc, vv, mask=m)


@triton.jit
def _pack_row_i64_kernel(
    q,
    k,
    v,
    indices,
    q_out,
    k_out,
    v_out,
    row_elems,
    idx_s0,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    idx = tl.load(indices + row * idx_s0).to(tl.int64)
    src = idx * row_elems
    dst = row * row_elems
    cols = tl.arange(0, BLOCK_C).to(tl.int64)
    for c0 in range(0, row_elems, BLOCK_C):
        cc = c0 + cols
        m = cc < row_elems
        qv = tl.load(q + src + cc, mask=m, other=0)
        tl.store(q_out + dst + cc, qv, mask=m)
        kv = tl.load(k + src + cc, mask=m, other=0)
        tl.store(k_out + dst + cc, kv, mask=m)
        vv = tl.load(v + src + cc, mask=m, other=0)
        tl.store(v_out + dst + cc, vv, mask=m)


def _use_int32(q_numel, n_rows, idx_s0):
    """True when every flat offset stays below 2**31.

    Bounds q/k/v source reads (idx * row_elems < q_numel for semantic
    B*S indices), the three output writes (n_rows * row_elems <=
    q_numel because a valid gather needs n_rows <= B*S) and the
    indices load ((n_rows - 1) * idx_s0 elements deep, strided views
    included).
    """
    if q_numel >= _INT32_LIMIT:
        return False
    return n_rows == 0 or (n_rows - 1) * idx_s0 < _INT32_LIMIT


def fused_pack_qkv(q, k, v, indices):
    assert q.shape == k.shape == v.shape
    assert q.dtype == k.dtype == v.dtype
    q = q if q.is_contiguous() else q.contiguous()
    k = k if k.is_contiguous() else k.contiguous()
    v = v if v.is_contiguous() else v.contiguous()
    n = indices.shape[0]
    row_elems = q.shape[-2] * q.shape[-1]
    idx_s0 = indices.stride(0) if indices.dim() else 1
    # single allocation: three contiguous views into one flat buffer
    buf = torch.empty(3 * n * row_elems, dtype=q.dtype, device=q.device)
    q_out = buf[: n * row_elems].view(n, q.shape[-2], q.shape[-1])
    k_out = buf[n * row_elems : 2 * n * row_elems].view(
        n, q.shape[-2], q.shape[-1]
    )
    v_out = buf[2 * n * row_elems :].view(n, q.shape[-2], q.shape[-1])
    if n:
        block_c = min(_MAX_LANES, max(_MIN_LANES, triton.next_power_of_2(row_elems)))
        grid = (n,)
        if _use_int32(q.numel(), n, idx_s0):
            _pack_row_i32_kernel[grid](
                q,
                k,
                v,
                indices,
                q_out,
                k_out,
                v_out,
                row_elems,
                idx_s0,
                BLOCK_C=block_c,
                num_warps=4,
            )
        else:
            _pack_row_i64_kernel[grid](
                q,
                k,
                v,
                indices,
                q_out,
                k_out,
                v_out,
                row_elems,
                idx_s0,
                BLOCK_C=block_c,
                num_warps=4,
            )
    return q_out, k_out, v_out


__all__ = ["fused_pack_qkv"]
