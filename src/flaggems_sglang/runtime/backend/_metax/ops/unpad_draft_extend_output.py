# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Metax vendor for unpad_draft_extend_output: num_warps 8 (the
# relu2/T86 positive precedents; our muxi reads 100-113 vs the
# leader's 245).

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_rows(
    raw, src_row, out, rows, span, rs0, os0,
    BLOCK: tl.constexpr,
):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        src = tl.load(src_row + row) * rs0
        dst = row.to(tl.int64) * os0
        # row_span stays a RUNTIME arg: the constexpr form of this
        # loop/mask bound empirically miscompiles to a no-op on the
        # proxy stack while the runtime form is correct (inverted from
        # the static-unroll lesson - bounds tied to program_id axes).
        for base in range(
            tl.program_id(1) * BLOCK,
            span,
            tl.num_programs(1) * BLOCK,
        ):
            offs = base + tl.arange(0, BLOCK)
            m = offs < span
            v = tl.load(raw + src + offs, m, other=0)
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
        _copy_rows[(min(total, 65535), min(max(1, triton.cdiv(row_span, 1024)), 255))](
            raw_out,
            src_row,
            out,
            total,
            row_span,
            # src_row is already the flat (b*tpb+t) row id, so the
            # source row stride is the token-row span, NOT stride(0)
            # (which counts tpb rows and double-flattens).
            row_span,
            out.stride(0),
            BLOCK=1024,
            num_warps=8,
        )
    return out


__all__ = ["unpad_draft_extend_output"]
