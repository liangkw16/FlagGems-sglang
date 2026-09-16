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
    BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    # Exclusive prefix sum of extend[0:pid], accumulated in fixed-width
    # masked chunks: no constexpr depends on the request count, so one
    # compiled variant serves every shape (and the separate zeros +
    # torch.cumsum launches disappear - the whole op is one launch).
    # E4: the prefix load now multiplies the element stride (es) like
    # every other extend access below - a strided view used to read the
    # wrong elements here while the current-row load stayed correct.
    acc = tl.zeros([BLOCK_N], dtype=tl.int32)
    for s0 in range(0, pid, BLOCK_N):
        ridx = s0 + tl.arange(0, BLOCK_N)
        acc += tl.load(extend + ridx * es, mask=ridx < pid, other=0)
    base = tl.sum(acc).to(tl.int64)
    qo = tl.load(extend + pid * es)
    kv = tl.load(seq + pid * ss)
    # The hint only chooses the launch grid; actual rows bound the work.
    tiles = tl.cdiv(qo.to(tl.int64), BLOCK)
    for tile in range(tl.program_id(1), tiles, tl.num_programs(1)):
        # Clamp keeps DP-padded/idle rows safe for uint32 downstream
        # readers (a negative length would read as ~4e9 tokens).
        start = kv - qo + 1
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        mask = offs < qo
        values = tl.maximum(start + offs.to(tl.int32), 0)
        tl.store(out + base + offs, values, mask=mask)


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
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = tl.load(prefix + pid).to(tl.int64)
    qo = tl.load(extend + pid * es)
    kv = tl.load(seq + pid * ss)
    tiles = tl.cdiv(qo.to(tl.int64), BLOCK)
    for tile in range(tl.program_id(1), tiles, tl.num_programs(1)):
        start = kv - qo + 1
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        mask = offs < qo
        values = tl.maximum(start + offs.to(tl.int32), 0)
        tl.store(out + base + offs, values, mask=mask)


@triton.jit
def _seqlens_prefix_wide(
    extend,
    prefix,
    n,
    es,
    BLOCK: tl.constexpr,
):
    # Rare metadata domain: physical output offsets cannot wrap at int32.
    carry = tl.full((), 0, tl.int64)
    for s0 in range(0, tl.cast(n, tl.int64), BLOCK):
        ridx = s0 + tl.arange(0, BLOCK).to(tl.int64)
        mask = ridx < n
        values = tl.load(extend + ridx * es, mask=mask, other=0).to(tl.int64)
        inclusive = tl.cumsum(values, axis=0)
        tl.store(prefix + ridx, carry + inclusive - values, mask=mask)
        carry += tl.sum(values, axis=0)


@triton.jit
def _seqlens_expand_p_safe(
    extend,
    seq,
    prefix,
    out,
    es,
    ss,
    qo_len,
    n,
    BLOCK: tl.constexpr,
):
    # The capped fallback grid must cover every request and output tile.
    for pid in range(
        tl.program_id(0).to(tl.int64), tl.cast(n, tl.int64), tl.num_programs(0)
    ):
        base = tl.load(prefix + pid).to(tl.int64)
        qo = tl.load(extend + pid * es)
        kv = tl.load(seq + pid * ss)
        tiles = tl.cdiv(qo.to(tl.int64), BLOCK)
        for tile in range(tl.program_id(1), tiles, tl.num_programs(1)):
            start = kv - qo + 1
            offs = tile * BLOCK + tl.arange(0, BLOCK)
            mask = offs < qo
            # Values follow the reference's int32 wrap before clamp.
            values = tl.maximum(start + offs.to(tl.int32), 0)
            offset = base + offs
            tl.store(out + offset, values, mask=mask)


@triton.jit
def _seqlens_expand_grouped(
    extend,
    seq,
    prefix,
    out,
    es,
    ss,
    n,
    groups,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.arange(0, GROUP)
    lanes = tl.arange(0, BLOCK).to(tl.int64)
    for group in range(
        tl.program_id(0).to(tl.int64),
        tl.cast(groups, tl.int64),
        tl.num_programs(0),
    ):
        req = group * GROUP + rows.to(tl.int64)
        valid = req < n
        qo = tl.load(extend + req * es, valid, other=0)
        kv = tl.load(seq + req * ss, valid, other=0)
        # Physical positions never wrap; values intentionally wrap int32.
        base = tl.load(prefix + req, valid, other=0).to(tl.int64)
        start = kv - qo + 1
        # max_q_len is only a dispatch hint. Actual lengths bound all writes.
        longest = tl.max(qo, axis=0).to(tl.int64)
        for p0 in range(0, longest, BLOCK):
            pos = p0 + lanes
            mask = valid[:, None] & (pos[None, :] < qo[:, None])
            values = tl.maximum(start[:, None] + pos[None, :].to(tl.int32), 0)
            tl.store(out + base[:, None] + pos[None, :], values, mask)


def seqlens_expand(extend_seq_lens, seq_lens, total_len, max_q_len):
    assert extend_seq_lens.ndim == seq_lens.ndim == 1
    n = extend_seq_lens.numel()
    assert seq_lens.numel() == n
    assert extend_seq_lens.dtype == seq_lens.dtype == torch.int32
    out = torch.empty(
        total_len, dtype=torch.int32, device=extend_seq_lens.device
    )
    if n and total_len:
        block = 1024
        tiles = min(triton.cdiv(max(1, max_q_len), block), 255)
        es, ss = extend_seq_lens.stride(0), seq_lens.stride(0)
        int_max = 2**31 - 1
        wide_prefix = (
            total_len > int_max
            or n > (int_max // block) * block
            or (n - 1) * max(es, ss) > int_max
            or max_q_len > int_max - (block - 1)
        )
        safe_grid = (min(n, 65535), min(tiles, 65535 // min(n, 65535)))
        safe_fallback = wide_prefix or n * tiles > 65535
        if n <= 1024 and not safe_fallback:
            # E4 small-N layout with actual per-request loop bounds.
            _seqlens_expand[(n, tiles)](
                extend_seq_lens,
                seq_lens,
                out,
                es,
                ss,
                max_q_len,
                BLOCK=block,
                BLOCK_N=block,
            )
        else:
            prefix = torch.empty(
                n,
                dtype=torch.int64 if wide_prefix else torch.int32,
                device=extend_seq_lens.device,
            )
            if wide_prefix:
                _seqlens_prefix_wide[(1,)](
                    extend_seq_lens,
                    prefix,
                    n,
                    es,
                    BLOCK=block,
                )
            else:
                _seqlens_prefix[(1,)](
                    extend_seq_lens,
                    prefix,
                    n,
                    es,
                    BLOCK=block,
                )
            if n > 1024 and max_q_len <= 32:
                # ponytail: four-row groups serialize to their longest row;
                # revisit the layout only if target measurements justify it.
                groups = triton.cdiv(n, 4)
                _seqlens_expand_grouped[(min(groups, 65535),)](
                    extend_seq_lens,
                    seq_lens,
                    prefix,
                    out,
                    es,
                    ss,
                    n,
                    groups,
                    GROUP=4,
                    BLOCK=32,
                    num_warps=4,
                )
            elif safe_fallback:
                # Existing E9 safety path, independent of grouped scheduling.
                _seqlens_expand_p_safe[safe_grid](
                    extend_seq_lens,
                    seq_lens,
                    prefix,
                    out,
                    es,
                    ss,
                    max_q_len,
                    n,
                    BLOCK=block,
                )
            else:
                # E4 safe-grid layout with actual per-request loop bounds.
                _seqlens_expand_p[(n, tiles)](
                    extend_seq_lens,
                    seq_lens,
                    prefix,
                    out,
                    es,
                    ss,
                    max_q_len,
                    BLOCK=block,
                )
    return out


__all__ = ["seqlens_expand"]
