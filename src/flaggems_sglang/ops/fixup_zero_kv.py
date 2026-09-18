# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 74338e9 kernels/ops/attention/fixup_zero_kv.py and
# jit/csrc/attention/fixup_zero_kv.cuh (2D early-exit grid, vectorised
# fills). The reference host-syncs on nonzero().tolist(); this variant
# decides everything on-device in one launch over a 1D grid. E3 fuses
# the clone into the same kernel: the output tensors are allocated
# empty and every element is written exactly once - zero-KV segments
# get constants (0 / -inf), untouched segments are copied from the
# inputs - cutting wrapper traffic from (2+z)D to (2-z)D and removing
# both clone launches. The lse stores now carry the head mask (the
# e1/e2 bytes could overrun a non-power-of-two head row).

import torch
import triton
import triton.language as tl


@triton.jit
def _fixup_zero_kv(
    out,
    lse,
    src_out,
    src_lse,
    lens,
    cum,
    total,
    os0,
    ls0,
    sos0,
    sls0,
    batch,
    ot,
    HV: tl.constexpr,
    NH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 1D grid: pid -> (segment, token-tile). Every output element has
    # exactly one writer; the tile loop strides by the host's ot
    # estimate so an understated max_seq_len still covers every token.
    seg = tl.program_id(0) // ot
    if seg < batch:
        zero = tl.load(lens + seg) == 0
        beg = tl.load(cum + seg).to(tl.int64)
        end = tl.load(cum + seg + 1).to(tl.int64)
        tokens = end - beg
        tiles = tl.cdiv(tokens, BLOCK_T)
        v = tl.arange(0, BLOCK_V).to(tl.int64)
        h = tl.arange(0, BLOCK_H).to(tl.int64)
        hm = h < NH
        zeros = tl.zeros((BLOCK_T, BLOCK_V), dtype=out.dtype.element_ty)
        ninf = tl.full(
            (BLOCK_T, BLOCK_H), float("-inf"), dtype=tl.float32
        )
        for tile in range(tl.program_id(0) % ot, tiles, ot):
            t = beg + tile * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
            tm = t < end
            # HV is constexpr (the T81-e5 lesson): the inner sweep
            # unrolls and the lane mask folds away whenever BLOCK_V
            # divides the row width (96*128 = 24*512 does).
            for v0 in tl.static_range(0, HV, BLOCK_V):
                vv = v0 + v[None, :]
                m = tm[:, None] & (vv < HV)
                if zero:
                    tl.store(out + t[:, None] * os0 + vv, zeros, m)
                else:
                    value = tl.load(
                        src_out + t[:, None] * sos0 + vv, m, other=0
                    )
                    tl.store(out + t[:, None] * os0 + vv, value, m)
            m2 = tm[:, None] & hm[None, :]
            if zero:
                tl.store(lse + t[:, None] * ls0 + h[None, :], ninf, m2)
            else:
                value = tl.load(
                    src_lse + t[:, None] * sls0 + h[None, :], m2, other=0
                )
                tl.store(lse + t[:, None] * ls0 + h[None, :], value, m2)
        # Tokens outside the cum boundaries keep the input bytes (the
        # reference clones them untouched): segment 0's programs copy
        # the head gap [0, cum[0]), the last segment's copy the tail
        # gap [cum[batch], total). Zero-iteration loops when covered.
        if seg == 0:
            for tile in range(
                tl.program_id(0) % ot, tl.cdiv(beg, BLOCK_T), ot
            ):
                t = tile * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
                tm = t < beg
                for v0 in tl.static_range(0, HV, BLOCK_V):
                    vv = v0 + v[None, :]
                    m = tm[:, None] & (vv < HV)
                    value = tl.load(
                        src_out + t[:, None] * sos0 + vv, m, other=0
                    )
                    tl.store(out + t[:, None] * os0 + vv, value, m)
                m2 = tm[:, None] & hm[None, :]
                value = tl.load(
                    src_lse + t[:, None] * sls0 + h[None, :], m2, other=0
                )
                tl.store(lse + t[:, None] * ls0 + h[None, :], value, m2)
        if seg == batch - 1:
            for tile in range(
                tl.program_id(0) % ot,
                tl.cdiv(total - end, BLOCK_T),
                ot,
            ):
                t = end + tile * BLOCK_T + tl.arange(0, BLOCK_T).to(
                    tl.int64
                )
                tm = t < total
                for v0 in tl.static_range(0, HV, BLOCK_V):
                    vv = v0 + v[None, :]
                    m = tm[:, None] & (vv < HV)
                    value = tl.load(
                        src_out + t[:, None] * sos0 + vv, m, other=0
                    )
                    tl.store(out + t[:, None] * os0 + vv, value, m)
                m2 = tm[:, None] & hm[None, :]
                value = tl.load(
                    src_lse + t[:, None] * sls0 + h[None, :], m2, other=0
                )
                tl.store(lse + t[:, None] * ls0 + h[None, :], value, m2)


def fixup_zero_kv(out, lse, kv_lens, cum_seq_lens, max_seq_len):
    assert out.ndim == 3 and lse.ndim == 2
    total_tokens, num_heads, v_head_dim = out.shape
    assert lse.shape == (total_tokens, num_heads)
    assert out.dtype in (torch.float16, torch.bfloat16)
    assert lse.dtype == torch.float32
    # Rows are written as flat spans, so the inner dims must be
    # contiguous (row-major after the token stride).
    assert out.stride(2) == 1 and out.stride(1) == v_head_dim
    assert lse.stride(1) == 1
    batch = kv_lens.numel()
    assert cum_seq_lens.numel() == batch + 1
    assert kv_lens.dtype == cum_seq_lens.dtype == torch.int32
    out_fixed = torch.empty_like(out)
    lse_fixed = torch.empty_like(lse)
    if batch and total_tokens:
        hv, nh = num_heads * v_head_dim, num_heads
        block_t = 8
        # max_seq_len only sizes the launch (advisory); the in-kernel
        # tile stride keeps coverage if it understates the real spans.
        span = max_seq_len if isinstance(max_seq_len, int) else total_tokens
        ot = max(1, triton.cdiv(min(span, total_tokens), block_t))
        _fixup_zero_kv[(batch * ot,)](
            out_fixed,
            lse_fixed,
            out,
            lse,
            kv_lens,
            cum_seq_lens,
            total_tokens,
            out_fixed.stride(0),
            lse_fixed.stride(0),
            out.stride(0),
            lse.stride(0),
            batch,
            ot,
            HV=hv,
            NH=nh,
            BLOCK_T=block_t,
            BLOCK_V=512,
            BLOCK_H=triton.next_power_of_2(max(1, nh)),
        )
    else:
        # batch == 0 with tokens present (or empty tensors): the
        # reference clones everything, so nothing stays uninitialised.
        out_fixed.copy_(out)
        lse_fixed.copy_(lse)
    return out_fixed, lse_fixed


__all__ = ["fixup_zero_kv"]
