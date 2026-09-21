# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# fixup_zero_kv: the e8 in-place core in the XPU-safe flat form
# (1D [BLOCK_V] stores only - the [BLOCK_T,1]+[1,BLOCK_H] pointer
# broadcast hit the XPU LLVM packing bug at e1/e2).

import torch
import triton
import triton.language as tl


@triton.jit
def _fixup_zero_kv(
    out,
    lse,
    lens,
    cum,
    batch,
    ot,
    os0,
    ls0,
    HV: tl.constexpr,
    NH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 1D grid: pid -> (segment, token-tile). Programs of healthy
    # segments exit immediately; the tile loop strides by the host's ot
    # estimate so an understated max_seq_len still covers every token.
    seg = tl.program_id(0) // ot
    if seg < batch:
        zero = tl.load(lens + seg) == 0
        if zero:
            beg = tl.load(cum + seg).to(tl.int64)
            end = tl.load(cum + seg + 1).to(tl.int64)
            tiles = tl.cdiv(end - beg, BLOCK_T)
            v = tl.arange(0, BLOCK_V).to(tl.int64)
            h = tl.arange(0, BLOCK_H).to(tl.int64)
            hm = h < NH
            zeros = tl.zeros(
                (BLOCK_T, BLOCK_V), dtype=out.dtype.element_ty
            )
            ninf = tl.full(
                (BLOCK_T, BLOCK_H), float("-inf"), dtype=tl.float32
            )
            for tile in range(tl.program_id(0) % ot, tiles, ot):
                t = beg + tile * BLOCK_T + tl.arange(0, BLOCK_T).to(
                    tl.int64
                )
                tm = t < end
                # HV is constexpr (the T81-e5 lesson): the inner sweep
                # unrolls and the lane mask folds away whenever BLOCK_V
                # divides the row width (96*128 = 24*512 does).
                for v0 in tl.static_range(0, HV, BLOCK_V):
                    vv = v0 + v[None, :]
                    m = tm[:, None] & (vv < HV)
                    tl.store(out + t[:, None] * os0 + vv, zeros, m)
                tl.store(
                    lse + t[:, None] * ls0 + h[None, :], ninf,
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
        hv, nh = num_heads * v_head_dim, num_heads
        block_t = 8
        span = max_seq_len if isinstance(max_seq_len, int) else total_tokens
        ot = max(1, triton.cdiv(min(span, total_tokens), block_t))
        _fixup_zero_kv[(batch * ot,)](
            out,
            lse,
            kv_lens,
            cum_seq_lens,
            batch,
            ot,
            out.stride(0),
            lse.stride(0),
            HV=hv,
            NH=nh,
            BLOCK_T=block_t,
            BLOCK_V=512,
            BLOCK_H=triton.next_power_of_2(max(1, nh)),
        )
    return out, lse


__all__ = ["fixup_zero_kv"]
