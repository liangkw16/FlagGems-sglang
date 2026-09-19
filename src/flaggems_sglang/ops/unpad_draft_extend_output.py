# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Inverse of pad_draft_extend_query: gather the accepted rows of the
# padded [bs, token_per_batch, H, D] attention output into a ragged
# [total_tokens, H, D] tensor. Reference is a Python loop; one program
# per accepted token row copies its H*D span (flat 1D, i64 only in
# addressing).

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["batch", "row_span"])
def _unpad(
    raw_out, lens, cum, out, batch, row_span, rs0, os0,
    TPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # 1D grid over (batch, token) pairs; token blocks via grid axis 1
    # keeps the grid inside Enflame's 255 y-limit. TPS is constexpr so
    # the decode and the single combined guard fold cheaply.
    pid = tl.program_id(0)
    seg = pid // TPS
    tok = pid % TPS
    s = tl.load(lens + seg)
    beg = tl.load(cum + seg)
    active = (seg < batch) & (tok < s)
    src = tl.where(active, pid, 0).to(tl.int64) * rs0
    dst = tl.where(active, beg + tok, 0).to(tl.int64) * os0
    for base in range(
        tl.program_id(1) * BLOCK,
        row_span,
        tl.num_programs(1) * BLOCK,
    ):
        offs = base + tl.arange(0, BLOCK)
        m = (offs < row_span) & active
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
        row_span = heads * dim
        splits = min(max(1, triton.cdiv(row_span, 1024)), 255)
        _unpad[(min(bs * token_per_batch, 65535), splits)](
            raw_out,
            seq_lens_q,
            cu_seqlens_q,
            out,
            bs,
            row_span,
            raw_out.stride(0),
            out.stride(0),
            TPS=max(1, token_per_batch),
            BLOCK=1024,
        )
    return out


__all__ = ["unpad_draft_extend_output"]
