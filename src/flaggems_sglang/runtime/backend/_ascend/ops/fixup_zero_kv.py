# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# T80 e29: the per-row 2D broadcast stores (i64 t[:,None]*OS0 + vv
# pointer grid) lower poorly on AscendVector; this rewrite goes 1D-flat
# per (row, v-chunk) item with int32 offset vectors and int64 scalar
# bases only, at a wide BLOCK_V (the T92 ascend vendor proved 1D wide
# bf16 stores fit the 192KB UB budget). The 48-worker grid stride and
# the zero-KV early exit stay.

import torch
import triton
import triton.language as tl

_BLOCK_V = 8192
_BLOCK_T_WIDE = 8
_BLOCK_V_WIDE = 512
# ponytail: fixed core-scale cap avoids a costly per-call device query;
# revisit the cap only if target-chip timing shows idle cores.
_MAX_WORKERS = 48


@triton.jit
def _fixup_zero_kv(
    out,
    lse,
    lens,
    cum,
    BATCH: tl.constexpr,
    SLOTS: tl.constexpr,
    OS0: tl.constexpr,
    LS0: tl.constexpr,
    KS0: tl.constexpr,
    CS0: tl.constexpr,
    HV: tl.constexpr,
    NH: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Each virtual item owns one (segment, row*v-chunk) flat slot; the
    # int64 pipeline lives only in the scalar row bases, every lane
    # vector is int32.
    # max(...,1) keeps the lse row write alive when HV == 0 (no value
    # chunks); the out store is fully masked in that case
    chunks = tl.cdiv(HV, BLOCK_V) if HV > 0 else 1
    for item in range(tl.program_id(0), BATCH * SLOTS, tl.num_programs(0)):
        seg = item // SLOTS
        slot = item % SLOTS
        if tl.load(lens + seg * KS0) == 0:
            beg = tl.load(cum + seg * CS0).to(tl.int64)
            end = tl.load(cum + (seg + 1) * CS0).to(tl.int64)
            rows = (end - beg).to(tl.int32)
        # int32 lane offsets are guarded by the wrapper (HV < 2^31);
        # the wide fallback kernel carries the i64 pipeline
            v = tl.arange(0, BLOCK_V)
            h = tl.arange(0, BLOCK_H)
            hm = h < NH
            zeros = tl.zeros((BLOCK_V,), dtype=out.dtype.element_ty)
            ninf = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
            for work in range(slot, rows * chunks, SLOTS):
                row = work // chunks
                vchunk = work % chunks
                off = vchunk * BLOCK_V + v
                base = (beg + row) * OS0
                tl.store(out + base + off, zeros, off < HV)
                if vchunk == 0:
                    tl.store(lse + (beg + row) * LS0 + h, ninf, hm)


@triton.jit
def _fixup_zero_kv_wide(
    out,
    lse,
    lens,
    cum,
    BATCH: tl.constexpr,
    SLOTS: tl.constexpr,
    OS0: tl.constexpr,
    LS0: tl.constexpr,
    KS0: tl.constexpr,
    CS0: tl.constexpr,
    HV: tl.constexpr,
    NH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # i64 pipeline for rows wide enough to overflow the int32 lane
    # domain (the wrapper dispatches on HV); pre-e29 2D form
    for item in range(tl.program_id(0), BATCH * SLOTS, tl.num_programs(0)):
        seg = item // SLOTS
        slot = item % SLOTS
        if tl.load(lens + seg * KS0) == 0:
            beg = tl.load(cum + seg * CS0).to(tl.int64)
            end = tl.load(cum + (seg + 1) * CS0).to(tl.int64)
            tiles = tl.cdiv(end - beg, BLOCK_T)
            v = tl.arange(0, BLOCK_V).to(tl.int64)
            h = tl.arange(0, BLOCK_H).to(tl.int64)
            hm = h < NH
            zeros = tl.zeros((BLOCK_T, BLOCK_V), dtype=out.dtype.element_ty)
            ninf = tl.full((BLOCK_T, BLOCK_H), float("-inf"), dtype=tl.float32)
            for tile in range(slot, tiles, SLOTS):
                t = beg + tile * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
                tm = t < end
                for v0 in tl.static_range(0, HV, BLOCK_V):
                    vv = v0 + v[None, :]
                    tl.store(
                        out + t[:, None] * OS0 + vv,
                        zeros,
                        tm[:, None] & (vv < HV),
                    )
                tl.store(
                    lse + t[:, None] * LS0 + h[None, :],
                    ninf,
                    tm[:, None] & hm[None, :],
                )


def fixup_zero_kv(out, lse, kv_lens, cum_seq_lens, max_seq_len):
    assert out.ndim == 3 and lse.ndim == 2
    total_tokens, num_heads, v_head_dim = out.shape
    assert lse.shape == (total_tokens, num_heads)
    assert out.dtype in (torch.float16, torch.bfloat16)
    assert lse.dtype == torch.float32
    assert out.stride(2) == 1 and out.stride(1) == v_head_dim
    assert lse.stride(1) == 1
    batch = kv_lens.numel()
    assert cum_seq_lens.numel() == batch + 1
    assert kv_lens.dtype == cum_seq_lens.dtype == torch.int32
    if batch and total_tokens:
        workers = _MAX_WORKERS
        # one slot per ~(rows*chunks) work quantum: enough fanout to
        # split long zero segments, bounded by the core-scale grid
        rows_per_seg = max(1, total_tokens // batch)
        chunks = triton.cdiv(num_heads * v_head_dim, _BLOCK_V)
        # slots = per-segment work quanta capped at the worker grid:
        # small segments keep full fanout (no floor-to-one collapse),
        # long segments split across the whole core-scale grid
        slots = min(workers, max(1, rows_per_seg * max(chunks, 1)))
        grid = (min(workers, batch * slots),)
        hv = num_heads * v_head_dim
        if hv < 2**31:
            _fixup_zero_kv[grid](
                out,
                lse,
                kv_lens,
                cum_seq_lens,
                BATCH=batch,
                SLOTS=slots,
                OS0=out.stride(0),
                LS0=lse.stride(0),
                KS0=kv_lens.stride(0),
                CS0=cum_seq_lens.stride(0),
                HV=hv,
                NH=num_heads,
                BLOCK_V=_BLOCK_V,
                BLOCK_H=triton.next_power_of_2(max(1, num_heads)),
            )
        else:
            _fixup_zero_kv_wide[grid](
                out,
                lse,
                kv_lens,
                cum_seq_lens,
                BATCH=batch,
                SLOTS=slots,
                OS0=out.stride(0),
                LS0=lse.stride(0),
                KS0=kv_lens.stride(0),
                CS0=cum_seq_lens.stride(0),
                HV=hv,
                NH=num_heads,
                BLOCK_T=_BLOCK_T_WIDE,
                BLOCK_V=_BLOCK_V_WIDE,
                BLOCK_H=triton.next_power_of_2(max(1, num_heads)),
            )
    return out, lse


__all__ = ["fixup_zero_kv"]
