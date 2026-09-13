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
    qo_len,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles = tl.cdiv(qo_len, BLOCK)
    for tile in range(tl.program_id(1), tiles, tl.num_programs(1)):
        qo = tl.load(extend + pid * es)
        kv = tl.load(seq + pid * ss)
        # Clamp keeps DP-padded/idle rows safe for uint32 downstream
        # readers (a negative length would read as ~4e9 tokens).
        start = kv - qo + 1
        offs = tile * BLOCK + tl.arange(0, BLOCK)
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
        # E1: the S0 whole-request form runs one program per request,
        # underfilling wide grids when requests are few but q_len is
        # large (the per-chip gap to the leader is uniform). Tile the
        # request dimension like the kv_indices splits: fixed 1024-lane
        # blocks, one program per (request, tile).
        # grid.y clamped to the Enflame hardware limit with a
        # grid-stride over the remaining tiles.
        block = 1024
        tiles = min(triton.cdiv(max(1, max_q_len), block), 255)
        _seqlens_expand[(n, tiles)](
            extend_seq_lens,
            seq_lens,
            offsets,
            out,
            extend_seq_lens.stride(0),
            seq_lens.stride(0),
            offsets.stride(0),
            max_q_len,
            BLOCK=block,
        )
    return out


__all__ = ["seqlens_expand"]
