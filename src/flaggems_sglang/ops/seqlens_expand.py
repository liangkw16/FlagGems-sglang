# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang fd32226 kernels/ops/attention/pad.py.

import torch
import triton
import triton.language as tl


@triton.jit
def _seqlens_expand(
    extend,
    seq,
    offsets,
    out,
    es,
    ss,
    os_,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    qo = tl.load(extend + pid * es)
    kv = tl.load(seq + pid * ss)
    # Clamp keeps DP-padded/idle rows safe for uint32 downstream readers
    # (a negative length would read as ~4e9 tokens).
    start = kv - qo + 1
    offs = tl.arange(0, BLOCK)
    mask = offs < qo
    base = tl.load(offsets + pid * os_)
    values = tl.maximum(start + offs, 0)
    tl.store(out + (base + offs).to(tl.int64), values, mask=mask)


def seqlens_expand(extend_seq_lens, seq_lens, total_len, max_q_len):
    assert extend_seq_lens.ndim == seq_lens.ndim == 1
    n = extend_seq_lens.numel()
    assert seq_lens.numel() == n
    assert extend_seq_lens.dtype == seq_lens.dtype == torch.int32
    out = torch.empty(
        total_len, dtype=torch.int32, device=extend_seq_lens.device
    )
    if n and total_len:
        offsets = torch.zeros(
            n + 1, dtype=torch.int32, device=extend_seq_lens.device
        )
        torch.cumsum(extend_seq_lens, dim=0, out=offsets[1:])
        _seqlens_expand[(n,)](
            extend_seq_lens,
            seq_lens,
            offsets,
            out,
            extend_seq_lens.stride(0),
            seq_lens.stride(0),
            offsets.stride(0),
            BLOCK=triton.next_power_of_2(max(1, max_q_len)),
        )
    return out


__all__ = ["seqlens_expand"]
