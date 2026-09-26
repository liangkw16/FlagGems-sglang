# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 96 fused_pack_qkv: gather the kept tokens of Q/K/V into three
# varlen tensors with one launch. E3 replaces the s0 2D row-tile
# (BLOCK_R=4 x BLOCK_C=1024 = 4096 elements per program, int64-only
# addressing, three allocations) with the per-row 1D form that the E2
# _iluvatar vendor already delivered on the platform (tianshu
# 2.95 -> 4.171, +41%), so the five chips still riding generic
# (muxi/huawei/card_a/card_b/haiguang) get the same structure:
#
# 1. Per-row 1D programs: one program owns ``rows_per_prog``
#    consecutive output rows; each row loads its scalar index once and
#    copies q/k/v as three masked 1D segments. The grid is sized by
#    row with a 65535-program cap (``rows_per_prog = cdiv(n, 65535)``)
#    to hold the Ascend coreDim limit, so the evaluation domain
#    (n <= B*S) rides the pure one-row-per-program shape while the
#    multi-row fallback stays reachable. All tensors inside the kernel
#    are 1D: the old form's 2D broadcast mask violated muxi's
#    "1D grid + in-kernel 2D tensor" TTGIR hazard and forced the
#    Ascend Vector CMP penalty (chip-rulesets.md:29/31/36).
# 2. BLOCK_C is capped at 2048 elements: muxi's max_tile_size is 2048
#    elements per program and the old 4096-element tile was 2x over
#    the limit (chip-rulesets.md:29). Rows wider than the cap fall
#    into the chunked column loop (regression: H*D=70000).
# 3. int32 addressing domain: the s0 kernel cast all five offset
#    sites (rows/idx/src/dst/cols) to int64. The i32 twin keeps the
#    same arithmetic in int32 whenever every flat offset provably
#    stays below 2**31 (numeric-domain host dispatch, not a device
#    check); the i64 twin mirrors the proven s0 int64 math site by
#    site for the overflow domain.
# 4. Single allocation: one flat buffer backs q_out/k_out/v_out as
#    three contiguous views (3 torch.empty -> 1).

import torch
import triton
import triton.language as tl

_INT32_LIMIT = 2**31
_MAX_LANES = 2048  # muxi max_tile_size, chip-rulesets.md:29
_MIN_LANES = 128
_MAX_GRID = 65535  # Ascend coreDim program cap


@triton.jit
def _pack_rows_i32_kernel(
    q,
    k,
    v,
    indices,
    q_out,
    k_out,
    v_out,
    n_rows,
    row_elems,
    rows_per_prog,
    idx_s0,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    row0 = pid * rows_per_prog
    cols = tl.arange(0, BLOCK_C)
    for r in range(0, rows_per_prog):
        row = row0 + r
        if row < n_rows:
            idx = tl.load(indices + row * idx_s0).to(tl.int32)
            src = idx * row_elems
            dst = row * row_elems
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
def _pack_rows_i64_kernel(
    q,
    k,
    v,
    indices,
    q_out,
    k_out,
    v_out,
    n_rows,
    row_elems,
    rows_per_prog,
    idx_s0,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row0 = pid * rows_per_prog
    cols = tl.arange(0, BLOCK_C).to(tl.int64)
    for r in range(0, rows_per_prog):
        row = row0 + r
        if row < n_rows:
            idx = tl.load(indices + row * idx_s0).to(tl.int64)
            src = idx * row_elems
            dst = row * row_elems
            for c0 in range(0, row_elems, BLOCK_C):
                cc = c0 + cols
                m = cc < row_elems
                qv = tl.load(q + src + cc, mask=m, other=0)
                tl.store(q_out + dst + cc, qv, mask=m)
                kv = tl.load(k + src + cc, mask=m, other=0)
                tl.store(k_out + dst + cc, kv, mask=m)
                vv = tl.load(v + src + cc, mask=m, other=0)
                tl.store(v_out + dst + cc, vv, mask=m)


def _use_int32(q_numel, n_rows, row_elems, idx_s0):
    """True when every flat offset stays below 2**31.

    Bounds q/k/v source reads (idx * row_elems < q_numel for semantic
    B*S index values), the three output writes (n_rows * row_elems
    elements total, checked on its own: a pure gather allows duplicate
    indices, so the output can be arbitrarily larger than the input)
    and the indices load ((n_rows - 1) * idx_s0 elements deep,
    strided views included).
    """
    if q_numel >= _INT32_LIMIT:
        return False
    if n_rows * row_elems >= _INT32_LIMIT:
        return False
    return n_rows == 0 or (n_rows - 1) * idx_s0 < _INT32_LIMIT


def fused_pack_qkv(q, k, v, indices):
    assert q.shape == k.shape == v.shape
    assert q.dtype == k.dtype == v.dtype
    # the gather treats each row as a contiguous H*D run; a non-contiguous
    # input is normalized with an explicit copy so the flat addressing
    # stays exact (reference's reshape has the same effect)
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
        rows_per_prog = triton.cdiv(n, _MAX_GRID)
        grid = (triton.cdiv(n, rows_per_prog),)
        if _use_int32(q.numel(), n, row_elems, idx_s0):
            _pack_rows_i32_kernel[grid](
                q,
                k,
                v,
                indices,
                q_out,
                k_out,
                v_out,
                n,
                row_elems,
                rows_per_prog,
                idx_s0,
                BLOCK_C=block_c,
                num_warps=4,
            )
        else:
            _pack_rows_i64_kernel[grid](
                q,
                k,
                v,
                indices,
                q_out,
                k_out,
                v_out,
                n,
                row_elems,
                rows_per_prog,
                idx_s0,
                BLOCK_C=block_c,
                num_warps=4,
            )
    return q_out, k_out, v_out


__all__ = ["fused_pack_qkv"]
