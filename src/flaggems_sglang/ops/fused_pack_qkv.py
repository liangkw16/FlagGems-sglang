# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 96 fused_pack_qkv: gather the kept tokens of Q/K/V into three
# varlen tensors with one launch. indices holds flat B*S positions and
# is loaded once per row tile; every row is a contiguous H*D run so the
# copy is a 2D tile with the index broadcast along columns. Index math
# is i64 (int32 platform indices times H*D can exceed 2**31).

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_pack_qkv_kernel(
    q,
    k,
    v,
    indices,
    q_out,
    k_out,
    v_out,
    n_rows,
    row_elems,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid.to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
    rmask = rows < n_rows
    idx = tl.load(indices + rows, mask=rmask, other=0).to(tl.int64)
    src = idx * row_elems
    dst = rows * row_elems
    cols = tl.arange(0, BLOCK_C).to(tl.int64)
    for c0 in range(0, row_elems, BLOCK_C):
        cc = c0 + cols
        m = rmask[:, None] & (cc[None, :] < row_elems)
        offs = cc[None, :]
        qv = tl.load(q + src[:, None] + offs, mask=m, other=0)
        tl.store(q_out + dst[:, None] + offs, qv, mask=m)
        kv = tl.load(k + src[:, None] + offs, mask=m, other=0)
        tl.store(k_out + dst[:, None] + offs, kv, mask=m)
        vv = tl.load(v + src[:, None] + offs, mask=m, other=0)
        tl.store(v_out + dst[:, None] + offs, vv, mask=m)


def fused_pack_qkv(q, k, v, indices):
    assert q.shape == k.shape == v.shape
    assert q.dtype == k.dtype == v.dtype
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    n = indices.shape[0]
    row_elems = q.shape[-2] * q.shape[-1]
    shape = (n, q.shape[-2], q.shape[-1])
    q_out = torch.empty(shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(shape, dtype=k.dtype, device=k.device)
    v_out = torch.empty(shape, dtype=v.dtype, device=v.device)
    if n:
        _fused_pack_qkv_kernel[(triton.cdiv(n, 4),)](
            q,
            k,
            v,
            indices,
            q_out,
            k_out,
            v_out,
            n,
            row_elems,
            BLOCK_R=4,
            BLOCK_C=1024,
            num_warps=4,
        )
    return q_out, k_out, v_out


__all__ = ["fused_pack_qkv"]
