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

"""Kunlunxin vendor for Task 96 fused_pack_qkv.

The generic 2D row-tile kernel read 0.010x on kunlunxin (every other
chip 0.94-3.3x) - the broadcast 2D tile lowers badly on the XPU
backend while 1D vector row copies do not (same batch's kv-indices
kernel reached 27x with that shape). One program owns a few
consecutive output rows; each row loads its scalar index once and
copies q/k/v as three masked 1D segments; the launch stays under the
65535 program cap. Not compiled on kunlunxin hardware locally.
"""

import torch
import triton
import triton.language as tl

_MAX_GRID = 65535


@triton.jit
def _pack_rows_kernel(
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
    cols = tl.arange(0, BLOCK_C).to(tl.int64)
    row0 = pid * rows_per_prog
    for r in range(0, rows_per_prog):
        row = row0 + r
        if row < n_rows:
            idx = tl.load(indices + row * idx_s0).to(tl.int64)
            src = idx * row_elems
            for c0 in range(0, row_elems, BLOCK_C):
                cc = c0 + cols
                mask = cc < row_elems
                qv = tl.load(q + src + cc, mask=mask, other=0)
                tl.store(q_out + row * row_elems + cc, qv, mask=mask)
                kv = tl.load(k + src + cc, mask=mask, other=0)
                tl.store(k_out + row * row_elems + cc, kv, mask=mask)
                vv = tl.load(v + src + cc, mask=mask, other=0)
                tl.store(v_out + row * row_elems + cc, vv, mask=mask)


def fused_pack_qkv(q, k, v, indices):
    assert q.shape == k.shape == v.shape
    assert q.dtype == k.dtype == v.dtype
    q = q if q.is_contiguous() else q.contiguous()
    k = k if k.is_contiguous() else k.contiguous()
    v = v if v.is_contiguous() else v.contiguous()
    n = indices.shape[0]
    row_elems = q.shape[-2] * q.shape[-1]
    shape = (n, q.shape[-2], q.shape[-1])
    q_out = torch.empty(shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(shape, dtype=k.dtype, device=k.device)
    v_out = torch.empty(shape, dtype=v.dtype, device=v.device)
    if n:
        block_c = min(65536, max(1024, triton.next_power_of_2(row_elems)))
        rows_per_prog = triton.cdiv(n, _MAX_GRID)
        grid = (triton.cdiv(n, rows_per_prog),)
        _pack_rows_kernel[grid](
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
            indices.stride(0) if indices.dim() else 1,
            BLOCK_C=block_c,
            num_warps=8,
        )
    return q_out, k_out, v_out


__all__ = ["fused_pack_qkv"]
