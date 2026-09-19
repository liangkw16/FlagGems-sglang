# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Inverse of pad_draft_extend_query: gather accepted rows into a ragged
# [total, H, D] tensor. Wrapper stages the row map via torch searchsorted
# (T49 precedent, no host sync); kernel is a pure 2D-grid gather copied
# verbatim from the empirically-passing minimal form on this backend
# (multiple semantically-identical variants miscompile to no-ops here).

import torch
import triton
import triton.language as tl


@triton.jit
def _unpad(
    raw_out, src_row, out, rows, rs0, os0,
    ROW_SPAN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        src = tl.load(src_row + row) * rs0
        dst = row.to(tl.int64) * os0
        for base in range(
            tl.program_id(1) * BLOCK,
            ROW_SPAN,
            tl.num_programs(1) * BLOCK,
        ):
            offs = base + tl.arange(0, BLOCK)
            m = offs < ROW_SPAN
            v = tl.load(raw_out + src + offs, m, other=0)
            tl.store(out + dst + offs, v, m)


def unpad_draft_extend_output(raw_out, cu_seqlens_q, seq_lens_q, sum_seq_lens_q):
    assert raw_out.ndim == 4
    bs, token_per_batch, heads, dim = raw_out.shape
    assert seq_lens_q.shape == (bs,) and cu_seqlens_q.shape == (bs + 1,)
    assert seq_lens_q.dtype == cu_seqlens_q.dtype == torch.int32
    assert raw_out.dtype in (torch.float16, torch.bfloat16)
    assert raw_out.is_contiguous()
    out = torch.empty(
        (sum_seq_lens_q, heads, dim),
        dtype=raw_out.dtype,
        device=raw_out.device,
    )
    if bs and token_per_batch and out.numel():
        total = int(sum_seq_lens_q)
        ar = torch.arange(total, device=raw_out.device)
        cu64 = cu_seqlens_q.to(torch.int64)
        seg = torch.searchsorted(cu_seqlens_q[1:], ar, right=True).long()
        tok = ar - cu64[seg]
        src_row = (seg * token_per_batch + tok).to(torch.int32)
        row_span = heads * dim
        _unpad[(min(total, 65535), min(max(1, triton.cdiv(row_span, 1024)), 255))](
            raw_out,
            src_row,
            out,
            total,
            raw_out.stride(0),
            out.stride(0),
            ROW_SPAN=row_span,
            BLOCK=1024,
        )
    return out


__all__ = ["unpad_draft_extend_output"]
