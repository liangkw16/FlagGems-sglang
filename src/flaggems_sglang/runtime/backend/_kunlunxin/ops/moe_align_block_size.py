# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Kunlun vendor for moe_align_block_size, vectorized round. The s0-era
# scalar triple loop (O(E*numel) single program) read 2.70x while the
# field reads 29-33x on this chip. XPU bans the generic kernel's
# ingredients (atomic_rmw, tl.cumsum/tt.scan, tensor-indexed gather),
# so this form keeps every banned construct out by design:
#   fill: the caller buffer gets the sentinel blanket from one memset
#     op before any placement (the reference itself prefills);
#   kernel 1 counts: 128-lane tiles, each program one-hot sums its
#     tiles against chunked expert lanes ([128, 64] compare matrices -
#     elementwise + tl.sum reduce, both XPU-legal) and writes its
#     private per-expert row (no atomics);
#   kernel 2a starts (one program): per-expert totals straight from
#     cnt, exclusive padded starts via a [64, 64] triangular matrix per
#     chunk instead of tt.scan, npost, and the end mark;
#   kernel 2b bases (one program): recomputes the starts in registers
#     (loads only cnt), writes each tile program's write base, and
#     fills expert_ids with scalar run stores; the split exists because
#     a same-launch read-after-write on starts returned pre-kernel
#     bytes under the empty-initialized buffer (load hoisting), which
#     zeros-initialization masked during bring-up;
#   kernel 3 places: each tile program computes the within-tile rank of
#     every lane with a [128, 128] value-compare triangular reduce (no
#     scan) and, per expert chunk, scatters matches to
#     tile_base + rank. The scatter store with computed per-lane
#     offsets is the one construct without prior XPU evidence - it is
#     this round's probe; masked non-matching lanes never store.

import torch
import triton
import triton.language as tl

_E_CHUNK_MAX = 64
_TILE = 128


@triton.jit(do_not_specialize=["numel", "num_routed", "buf_numel"])
def _moe_counts(
    flat,
    cnt,  # [programs, BLOCK_E]
    numel,
    num_routed,
    buf_numel,
    BLOCK_E: tl.constexpr,
    E_CHUNK: tl.constexpr,
    TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for e0 in range(0, BLOCK_E, E_CHUNK):
        lanes = e0 + tl.arange(0, E_CHUNK)
        acc = tl.zeros((E_CHUNK,), dtype=tl.int32)
        for base in range(pid * TILE, numel, nprog * TILE):
            offs = base + tl.arange(0, TILE)
            m = offs < numel
            v = tl.load(flat + offs, m, other=-1)
            hit = (v[:, None] == lanes[None, :]) & m[:, None]
            acc += tl.sum(hit.to(tl.int32), 0)
        tl.store(cnt + pid * BLOCK_E + lanes, acc)


@triton.jit(do_not_specialize=["nprog", "numel", "num_routed",
                               "block_size", "buf_numel"])
def _moe_scan_starts(
    cnt,       # [programs, BLOCK_E] per-program partial counts
    starts,    # [BLOCK_E] padded exclusive expert starts (+ end mark)
    npost_ptr,
    nprog,
    numel,
    num_routed,
    block_size,
    BLOCK_E: tl.constexpr,
    E_CHUNK: tl.constexpr,
):
    # totals come straight from cnt; nothing this kernel writes is read
    # back within the same launch (the read-after-write hazard on
    # starts is why the fused single-program scan was split)
    run = tl.zeros((), dtype=tl.int32)
    for e0 in range(0, BLOCK_E, E_CHUNK):
        lanes = e0 + tl.arange(0, E_CHUNK)
        c = tl.zeros((E_CHUNK,), dtype=tl.int32)
        for p in range(0, nprog):
            c += tl.load(cnt + p * BLOCK_E + lanes)
        c = tl.where(lanes < num_routed, c, 0)
        aligned = ((c + block_size - 1) // block_size) * block_size
        tri = (lanes[None, :] < lanes[:, None]).to(tl.int32)
        pre = tl.sum(tri * aligned[None, :], 1)
        tl.store(starts + lanes, run + pre)
        run += tl.sum(aligned, 0)
    tl.store(npost_ptr, run)
    # num_routed <= BLOCK_E - 1 always, so this slot exists
    tl.store(starts + num_routed, run)


@triton.jit(do_not_specialize=["nprog", "numel", "num_routed",
                               "block_size", "buf_numel"])
def _moe_scan_bases(
    cnt,       # [programs, BLOCK_E]
    starts,    # written by _moe_scan_starts (previous launch)
    tilebase,  # [programs, BLOCK_E] write bases per tile program
    eids,
    nprog,
    numel,
    num_routed,
    block_size,
    BLOCK_E: tl.constexpr,
    E_CHUNK: tl.constexpr,
):
    # recompute the exclusive padded starts in registers (loads only
    # cnt); starts is read-only here, written by the previous kernel
    run = tl.zeros((), dtype=tl.int32)
    for e0 in range(0, BLOCK_E, E_CHUNK):
        lanes = e0 + tl.arange(0, E_CHUNK)
        c = tl.zeros((E_CHUNK,), dtype=tl.int32)
        for p in range(0, nprog):
            c += tl.load(cnt + p * BLOCK_E + lanes)
        c = tl.where(lanes < num_routed, c, 0)
        aligned = ((c + block_size - 1) // block_size) * block_size
        tri = (lanes[None, :] < lanes[:, None]).to(tl.int32)
        pre = tl.sum(tri * aligned[None, :], 1)
        st = run + pre
        run += tl.sum(aligned, 0)
        carry = tl.zeros((E_CHUNK,), dtype=tl.int32)
        for p in range(0, nprog):
            tl.store(tilebase + p * BLOCK_E + lanes, st + carry)
            carry += tl.load(cnt + p * BLOCK_E + lanes)
    # expert_ids variable runs, scalar stores; starts read-only here
    for e in range(0, num_routed):
        s = tl.load(starts + e)
        nb = (tl.load(starts + e + 1) - s) // block_size
        for b in range(0, nb):
            tl.store(eids + s // block_size + b, e)


@triton.jit(do_not_specialize=["numel", "num_routed"])
def _moe_place(
    flat,
    tilebase,  # [programs, BLOCK_E]
    sorted_ids,
    numel,
    num_routed,
    BLOCK_E: tl.constexpr,
    E_CHUNK: tl.constexpr,
    TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for base in range(pid * TILE, numel, nprog * TILE):
        offs = base + tl.arange(0, TILE)
        m = offs < numel
        v = tl.load(flat + offs, m, other=-1)
        # within-tile rank of each lane among prior lanes holding the
        # same value: [TILE, TILE] triangular compare, row-reduced
        tri = offs[None, :] < offs[:, None]
        rank = tl.sum(
            ((v[None, :] == v[:, None]) & tri).to(tl.int32), 1
        )
        for e0 in range(0, BLOCK_E, E_CHUNK):
            lanes = e0 + tl.arange(0, E_CHUNK)
            lm = lanes < num_routed
            hit = (v[:, None] == lanes[None, :]) & m[:, None]
            tb = tl.load(tilebase + pid * BLOCK_E + lanes)
            dest = tb[None, :] + rank[:, None]
            tl.store(
                sorted_ids + dest,
                tl.broadcast_to(
                    offs.to(tl.int32)[:, None], (TILE, E_CHUNK)
                ),
                hit & lm[None, :],
            )
            add = tl.sum(hit.to(tl.int32), 0)
            tl.store(tilebase + pid * BLOCK_E + lanes, tb + add)


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
    flat = topk_ids.reshape(-1)
    numel = flat.numel()
    num_routed = num_experts - 1
    device = flat.device
    buf_numel = sorted_token_ids.numel()
    BLOCK_E = triton.next_power_of_2(max(2, num_experts))
    E_CHUNK = min(_E_CHUNK_MAX, BLOCK_E)
    # the sentinel blanket precedes placement so placed values survive;
    # the reference prefills the whole buffer unconditionally
    # (torch.full_like ignores no flag), so the blanket does too
    if buf_numel:
        sorted_token_ids.fill_(numel)
    if numel:
        nprog = min(triton.cdiv(numel, _TILE), 1024)
        # one allocation carries all three scratch buffers (T77 e6
        # lesson: per-call allocations are priced extremely high on
        # kunlun); the slices are contiguous views
        scratch = torch.empty(
            2 * nprog * BLOCK_E + BLOCK_E, dtype=torch.int32,
            device=device,
        )
        cnt = scratch[: nprog * BLOCK_E]
        tilebase = scratch[
            nprog * BLOCK_E : 2 * nprog * BLOCK_E
        ]
        starts = scratch[2 * nprog * BLOCK_E :]
        _moe_counts[(nprog,)](
            flat, cnt, numel, num_routed, buf_numel,
            BLOCK_E=BLOCK_E, E_CHUNK=E_CHUNK, TILE=_TILE,
        )
        _moe_scan_starts[(1,)](
            cnt, starts, num_tokens_post_pad,
            nprog, numel, num_routed, block_size,
            BLOCK_E=BLOCK_E, E_CHUNK=E_CHUNK,
        )
        _moe_scan_bases[(1,)](
            cnt, starts, tilebase, expert_ids,
            nprog, numel, num_routed, block_size,
            BLOCK_E=BLOCK_E, E_CHUNK=E_CHUNK,
        )
        _moe_place[(nprog,)](
            flat, tilebase, sorted_token_ids, numel, num_routed,
            BLOCK_E=BLOCK_E, E_CHUNK=E_CHUNK, TILE=_TILE,
        )
    else:
        num_tokens_post_pad.fill_(0)
    return sorted_token_ids, expert_ids, num_tokens_post_pad


__all__ = ["moe_align_block_size"]
