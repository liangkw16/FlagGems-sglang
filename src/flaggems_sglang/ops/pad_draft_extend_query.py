# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 100 pad_draft_extend_query: scatter varlen draft-query rows
# into a padded [bs, max_seq_len, H, D] buffer (clone-first semantics;
# rows past seq_lens_q[b] keep the base bytes). The SGLang baseline
# launches a dense 3D grid where short sequences early-exit - instead
# this kernel gives every ACTUAL source row exactly one program: flat
# index j maps to (b, row_in_b) by a scalar scan over cu_seqlens (bs is
# small), so no program ever exits without work and the clone covers
# the untouched tail.

import torch
import triton
import triton.language as tl


@triton.jit
def _pad_rows_kernel(
    q,
    cu,
    out,
    total,
    bs,
    row_elems,
    q_s0,
    o_s1,
    o_s2,
    cu_s0,
    BLOCK_C: tl.constexpr,
):
    j = tl.program_id(0).to(tl.int64)
    if j < total:
        # find the batch owning flat source row j (cu is non-decreasing)
        b = 0
        for i in range(0, bs):
            cu_i = tl.load(cu + i * cu_s0)
            if j >= cu_i:
                b = i
        beg = tl.load(cu + b * cu_s0).to(tl.int64)
        src = j * q_s0
        dst = (j - beg) * o_s1 + b * o_s2
        cols = tl.arange(0, BLOCK_C).to(tl.int64)
        for c0 in range(0, row_elems, BLOCK_C):
            cc = c0 + cols
            m = cc < row_elems
            v = tl.load(q + src + cc, mask=m, other=0)
            tl.store(out + dst + cc, v, mask=m)


def pad_draft_extend_query(q, padded_q, seq_lens_q, cu_seqlens_q):
    total = q.shape[0]
    q = q if q.is_contiguous() else q.contiguous()
    # a contiguous clone: value-identical to the reference clone while
    # keeping the kernel's flat row addressing valid for any base layout
    out = padded_q.clone(memory_format=torch.contiguous_format)
    if total:
        bs = cu_seqlens_q.shape[0] - 1
        row_elems = q.shape[-2] * q.shape[-1]
        _pad_rows_kernel[(total,)](
            q,
            cu_seqlens_q,
            out,
            total,
            bs,
            row_elems,
            q.stride(0),
            out.stride(1),
            out.stride(0),
            cu_seqlens_q.stride(0),
            BLOCK_C=1024,
            num_warps=4,
        )
    return out


__all__ = ["pad_draft_extend_query"]
