# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang fd32226 ops/attention/pad.py (task contract source).

import torch
import triton
import triton.language as tl


@triton.jit
def _seqlens_expand(
    extend,
    seq,
    out,
    es,
    ss,
    qo_len,
    tasks,
    tiles: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # E5: tile-first flat work mapping. Adjacent programs walk one
    # request's contiguous output tiles (work = request * tiles + tile)
    # instead of interleaving requests across the fastest axis of the
    # old (n, tiles) 2D launch. The 1D grid is capped with a
    # grid-stride so every backend's launch limits stay safe.
    for work in range(tl.program_id(0), tasks, tl.num_programs(0)):
        pid = work // tiles
        tile = work % tiles
        # Exclusive prefix sum of extend[0:pid], accumulated in
        # fixed-width masked chunks: no constexpr depends on the
        # request count, so one compiled variant serves every shape.
        # E4: the prefix load multiplies the element stride (es) like
        # every other extend access below.
        acc = tl.zeros([BLOCK_N], dtype=tl.int32)
        for s0 in range(0, pid, BLOCK_N):
            ridx = s0 + tl.arange(0, BLOCK_N)
            acc += tl.load(extend + ridx * es, mask=ridx < pid, other=0)
        base = tl.sum(acc)
        qo = tl.load(extend + pid * es)
        kv = tl.load(seq + pid * ss)
        # Clamp keeps DP-padded/idle rows safe for uint32 downstream
        # readers (a negative length would read as ~4e9 tokens).
        start = kv - qo + 1
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        mask = offs < qo
        values = tl.maximum(start + offs, 0)
        tl.store(out + (base + offs).to(tl.int64), values, mask=mask)


@triton.jit
def _seqlens_prefix(
    extend,
    prefix,
    n,
    es,
    BLOCK: tl.constexpr,
):
    # E4 large-batch path: one program turns the per-request prefix
    # accumulation (quadratic in the fused kernel) into a single chunked
    # scan. prefix[i] = sum(extend[0:i]) for i in [0, n); the carry is a
    # scalar so the loop body is the proven load/cumsum/store form
    # (tl.cumsum is platform-proven in this repo's batch-2 cumsum ops).
    carry = tl.full((), 0, tl.int32)
    for s0 in range(0, n, BLOCK):
        ridx = s0 + tl.arange(0, BLOCK)
        m = ridx < n
        v = tl.load(extend + ridx * es, m, other=0)
        incl = tl.cumsum(v, axis=0)
        tl.store(prefix + ridx, carry + incl - v, m)
        carry += tl.sum(v, axis=0)


@triton.jit
def _seqlens_expand_p(
    extend,
    seq,
    prefix,
    out,
    es,
    ss,
    qo_len,
    tasks,
    tiles: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # E5 flat work mapping, scan-path variant (base from prefix[i]).
    for work in range(tl.program_id(0), tasks, tl.num_programs(0)):
        pid = work // tiles
        tile = work % tiles
        base = tl.load(prefix + pid)
        qo = tl.load(extend + pid * es)
        kv = tl.load(seq + pid * ss)
        start = kv - qo + 1
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        mask = offs < qo
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
        # E6: tiles is constexpr so the per-program work decode
        # (work // tiles, work % tiles) folds to shifts/multiplies
        # instead of runtime integer division on every backend.
        block = 1024
        tiles = max(1, triton.cdiv(max_q_len, block))
        tasks = n * tiles
        grid = (min(tasks, 65535),)
        # E4 threshold: small batches keep the fused single-launch
        # kernel (per-program prefix accumulation is cheap there);
        # large batches pay one scan launch to materialize prefix[i]
        # once (haiguang/muxi/card_a +21% on the platform).
        if n <= 1024:
            _seqlens_expand[grid](
                extend_seq_lens,
                seq_lens,
                out,
                extend_seq_lens.stride(0),
                seq_lens.stride(0),
                max_q_len,
                tasks,
                tiles,
                BLOCK=block,
                BLOCK_N=block,
            )
        else:
            prefix = torch.empty(
                n, dtype=torch.int32, device=extend_seq_lens.device
            )
            _seqlens_prefix[(1,)](
                extend_seq_lens,
                prefix,
                n,
                extend_seq_lens.stride(0),
                BLOCK=block,
            )
            _seqlens_expand_p[grid](
                extend_seq_lens,
                seq_lens,
                prefix,
                out,
                extend_seq_lens.stride(0),
                seq_lens.stride(0),
                max_q_len,
                tasks,
                tiles,
                BLOCK=block,
            )
    return out


__all__ = ["seqlens_expand"]
