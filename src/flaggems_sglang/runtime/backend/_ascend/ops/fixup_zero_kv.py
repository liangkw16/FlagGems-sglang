# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# T80 e22: keep e16's per-row stores, but distribute segment/tile slots
# through one physical-VectorCore-sized worker grid. The earlier e21
# (batch, 48) launch still started up to batch*48 programs.

import torch
import triton
import triton.language as tl

_BLOCK_T = 8
_BLOCK_V = 512
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
    BLOCK_T: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Each virtual item owns one (segment, tile slot). Fixed workers
    # revisit items in a grid stride; slots split one long zero segment
    # across cores without launching batch*SLOTS physical programs.
    for item in range(tl.program_id(0), BATCH * SLOTS, tl.num_programs(0)):
        seg = item // SLOTS
        slot = item % SLOTS
        if tl.load(lens + seg * KS0) == 0:
            beg = tl.load(cum + seg * CS0).to(tl.int64)
            end = tl.load(cum + (seg + 1) * CS0).to(tl.int64)
            full_tiles = (end - beg) // BLOCK_T
            v = tl.arange(0, BLOCK_V).to(tl.int64)
            h = tl.arange(0, BLOCK_H).to(tl.int64)
            hm = h < NH
            zeros = tl.zeros((BLOCK_T, BLOCK_V), dtype=out.dtype.element_ty)
            ninf = tl.full((BLOCK_T, BLOCK_H), float("-inf"), dtype=tl.float32)
            for tile in range(slot, full_tiles, SLOTS):
                t = beg + tile * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
                for v0 in tl.static_range(0, HV, BLOCK_V):
                    vv = v0 + v[None, :]
                    if v0 + BLOCK_V <= HV:
                        tl.store(out + t[:, None] * OS0 + vv, zeros)
                    else:
                        tl.store(out + t[:, None] * OS0 + vv, zeros, vv < HV)
                if NH == BLOCK_H:
                    tl.store(lse + t[:, None] * LS0 + h[None, :], ninf)
                else:
                    tl.store(lse + t[:, None] * LS0 + h[None, :], ninf, hm)
            # Only one slot owns the partial tail; full tiles need no
            # token mask or per-element comparison on Vector Cores.
            if full_tiles * BLOCK_T < end - beg and slot == full_tiles % SLOTS:
                t = beg + full_tiles * BLOCK_T + tl.arange(0, BLOCK_T).to(
                    tl.int64
                )
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
        slots = min(
            workers, max(1, triton.cdiv(total_tokens, batch * _BLOCK_T))
        )
        grid = (min(workers, batch * slots),)
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
            HV=num_heads * v_head_dim,
            NH=num_heads,
            BLOCK_T=_BLOCK_T,
            BLOCK_V=_BLOCK_V,
            BLOCK_H=triton.next_power_of_2(max(1, num_heads)),
        )
    return out, lse


__all__ = ["fixup_zero_kv"]
