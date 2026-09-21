# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# e13: the no-atomic three-launch core (shared with the enflame vendor,
# see that file for the full design note). The e12 generic proved the
# launch-count thesis (163 -> 445 with atomics on the CUDA-like chips);
# this port keeps the three-launch shape with zero atomics - the Ascend
# stack faults atomic_add asynchronously (the e5-e9 chain) - and keeps
# the Ascend-proven launch geometry: full-width grids, no num_warps pin.

import torch
import triton
import triton.language as tl

_BLOCK = 32
_ROWS = 8
@triton.jit(
    do_not_specialize=["n", "num_routed", "epd", "num_blocks", "hist_rows"]
)
def _mabs_hist_fill(
    flat,
    counts,
    sorted_ids,
    n,
    num_routed,
    epd,
    num_blocks,
    hist_rows,
    buf_numel,
    sentinel,
    BLOCK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < hist_rows:
        lanes = tl.arange(0, BLOCK_E)
        lm = lanes < epd
        idx = tl.arange(0, BLOCK)
        for block in range(pid, num_blocks, hist_rows):
            offs = block * BLOCK + idx
            e = tl.load(flat + offs, offs < n, other=-1).to(tl.int32)
            row = tl.sum(
                ((e[:, None] == lanes[None, :]) & (offs[:, None] < n)).to(
                    tl.int32
                ),
                axis=0,
            )
            tl.store(
                counts + block * epd + lanes, row, lm
            )
    else:
        fbase = (pid - hist_rows) * BLOCK_F
        foffs = fbase + tl.arange(0, BLOCK_F)
        fm = foffs < buf_numel
        fill = tl.zeros((BLOCK_F,), dtype=tl.int32) + sentinel
        tl.store(sorted_ids + foffs, fill, fm)


@triton.jit(do_not_specialize=["epd", "num_blocks", "num_routed"])
def _mabs_scan(
    counts,
    prefix,
    base,
    nblk,
    npost_ptr,
    eids,
    epd,
    num_blocks,
    num_routed,
    block_size,
    BLOCK_E: tl.constexpr,
    ROWS: tl.constexpr,
):
    lanes = tl.arange(0, BLOCK_E)
    lm = lanes < num_routed
    rows = tl.arange(0, ROWS)
    totals = tl.zeros((BLOCK_E,), dtype=tl.int32)
    for r0 in range(0, num_blocks, ROWS):
        r = r0 + rows
        rm = (r[:, None] < num_blocks) & lm[None, :]
        tile = tl.load(
            counts + r[:, None] * epd + lanes[None, :], rm, other=0
        )
        # exclusive prefix per row along the block axis plus the carry
        # from every earlier tile
        pref = totals[None, :] + (
            tl.cumsum(tile, axis=0) - tile
        )
        tl.store(
            prefix + r[:, None] * epd + lanes[None, :], pref, rm
        )
        totals += tl.sum(tile, axis=0)
    aligned = ((totals + block_size - 1) // block_size) * block_size
    starts = tl.cumsum(aligned, 0) - aligned
    total = tl.sum(aligned, 0)
    tl.store(base + lanes, starts, lm)
    tl.store(nblk + lanes, aligned // block_size, lm)
    tl.store(npost_ptr, total)
    nblk_v = aligned // block_size
    start_blk = starts // block_size
    max_nb = tl.max(nblk_v, 0)
    for i in range(0, max_nb):
        m = lm & (i < nblk_v)
        tl.store(eids + start_blk + i, lanes.to(tl.int32), m)


@triton.jit(do_not_specialize=["n", "num_routed", "epd", "num_blocks"])
def _mabs_place(
    flat,
    prefix,
    base,
    sorted_ids,
    n,
    num_routed,
    epd,
    num_blocks,
    BLOCK: tl.constexpr,
):
    idx = tl.arange(0, BLOCK)
    for block in range(tl.program_id(0), num_blocks, tl.num_programs(0)):
        offs = block * BLOCK + idx
        m = offs < n
        e = tl.load(flat + offs, m, other=-1).to(tl.int32)
        hit = m & (e >= 0) & (e < num_routed)
        # in-block rank: how many earlier lanes of this block share my
        # expert (deterministic - original flat order within the block)
        rank = tl.sum(
            ((e[:, None] == e[None, :]) & (idx[None, :] < idx[:, None])).to(
                tl.int32
            ),
            axis=1,
        )
        block_e = block * epd
        pref = tl.load(prefix + block_e + e, hit, other=0)
        b = tl.load(base + e, hit, other=0)
        tl.store(sorted_ids + b + pref + rank, offs, hit)


def moe_align_block_size(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_pad,
    cumsum_buffer,
    pad_sorted_token_ids,
):
    assert topk_ids.ndim == 2
    flat = topk_ids.reshape(-1)
    n = flat.numel()
    num_routed = num_experts - 1
    device = flat.device
    buf_numel = sorted_token_ids.numel()
    block_e = triton.next_power_of_2(max(1, num_routed))
    if n:
        num_blocks = triton.cdiv(n, _BLOCK)
        epd = block_e
        tables = torch.empty(
            2 * num_blocks * epd, dtype=torch.int32, device=device
        )
        counts, prefix = (
            tables[: num_blocks * epd],
            tables[num_blocks * epd :],
        )
        aux = torch.empty(2 * epd, dtype=torch.int32, device=device)
        base, nblk = aux[:epd], aux[epd:]
        hist_rows = min(num_blocks, 256)
        fill_programs = triton.cdiv(buf_numel, 2048)
        grid1 = hist_rows + min(fill_programs, 512)
        _mabs_hist_fill[(grid1,)](
            flat,
            counts,
            sorted_token_ids,
            n,
            num_routed,
            epd,
            num_blocks,
            hist_rows,
            buf_numel,
            n,
            BLOCK=_BLOCK,
            BLOCK_E=block_e,
            BLOCK_F=2048,
        )
        _mabs_scan[(1,)](
            counts,
            prefix,
            base,
            nblk,
            num_tokens_post_pad,
            expert_ids,
            epd,
            num_blocks,
            num_routed,
            block_size,
            BLOCK_E=block_e,
            ROWS=_ROWS,
        )
        _mabs_place[(min(num_blocks, 2048),)](
            flat,
            prefix,
            base,
            sorted_token_ids,
            n,
            num_routed,
            epd,
            num_blocks,
            BLOCK=_BLOCK,
        )
    else:
        # empty input stays Triton-only: the blanket fill runs with no
        # histogram programs and the scan walks zero rows
        epd = block_e
        aux = torch.empty(2 * epd, dtype=torch.int32, device=device)
        base, nblk = aux[:epd], aux[epd:]
        fill_programs = triton.cdiv(buf_numel, 2048)
        _mabs_hist_fill[(fill_programs,)](
            flat,
            base,
            sorted_token_ids,
            0,
            num_routed,
            epd,
            0,
            0,
            buf_numel,
            0,
            BLOCK=_BLOCK,
            BLOCK_E=block_e,
            BLOCK_F=2048,
        )
        _mabs_scan[(1,)](
            base,
            base,
            base,
            nblk,
            num_tokens_post_pad,
            expert_ids,
            epd,
            0,
            num_routed,
            block_size,
            BLOCK_E=block_e,
            ROWS=_ROWS,
        )
    return sorted_token_ids, expert_ids, num_tokens_post_pad


__all__ = ["moe_align_block_size"]
