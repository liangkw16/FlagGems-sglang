# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Enflame vendor for fixup_zero_kv: the e1-proven clone+patch
# structure (燧原 23.3 on submission 17224) - the e3 fused-copy
# generic regressed GCU 40% (14.1 on 17314), so this chip keeps the
# two-launch clone form, with the lse head mask added (the e1/e2
# latent overrun fix rides along).

import torch
import triton
import triton.language as tl


@triton.jit
def _fixup_zero_kv(
    out,
    lse,
    lens,
    cum,
    hv,
    nh,
    os0,
    ls0,
    batch,
    ot,
    BLOCK_T: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 1D grid: pid -> (segment, token-tile). Programs of non-zero-KV
    # segments exit on the scalar guard; the tile loop strides by the
    # host's ot estimate so an understated max_seq_len still covers
    # every token (correctness never trusts the advisory bound).
    seg = tl.program_id(0) // ot
    if seg < batch:
        if tl.load(lens + seg) == 0:
            beg = tl.load(cum + seg).to(tl.int64)
            end = tl.load(cum + seg + 1).to(tl.int64)
            tokens = end - beg
            tiles = tl.cdiv(tokens, BLOCK_T)
            v = tl.arange(0, BLOCK_V).to(tl.int64)
            h = tl.arange(0, BLOCK_H)
            zeros = tl.zeros((BLOCK_T, BLOCK_V), dtype=out.dtype.element_ty)
            ninf = tl.full(
                (BLOCK_T, BLOCK_H), float("-inf"), dtype=tl.float32
            )
            for tile in range(tl.program_id(0) % ot, tiles, ot):
                t = beg + tile * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
                tm = (t < end)[:, None]
                for v0 in range(0, hv, BLOCK_V):
                    vv = v0 + v[None, :]
                    tl.store(
                        out + t[:, None] * os0 + vv,
                        zeros,
                        tm & (vv < hv),
                    )
                tl.store(
                    lse + t[:, None] * ls0 + h[None, :],
                    ninf,
                    tm & (h[None, :] < nh),
                )


def fixup_zero_kv(out, lse, kv_lens, cum_seq_lens, max_seq_len):
    assert out.ndim == 3 and lse.ndim == 2
    total_tokens, num_heads, v_head_dim = out.shape
    assert lse.shape == (total_tokens, num_heads)
    assert out.dtype in (torch.float16, torch.bfloat16)
    assert lse.dtype == torch.float32
    # The kernel zeroes each token row as one flat span, so the inner
    # two dims must be contiguous (row-major after the token stride).
    assert out.stride(2) == 1 and out.stride(1) == v_head_dim
    assert lse.stride(1) == 1
    batch = kv_lens.numel()
    assert cum_seq_lens.numel() == batch + 1
    assert kv_lens.dtype == cum_seq_lens.dtype == torch.int32
    out_fixed, lse_fixed = out.clone(), lse.clone()
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
            kv_lens,
            cum_seq_lens,
            hv,
            nh,
            out_fixed.stride(0),
            lse_fixed.stride(0),
            batch,
            ot,
            BLOCK_T=block_t,
            BLOCK_V=512,
            BLOCK_H=triton.next_power_of_2(max(1, nh)),
        )
    return out_fixed, lse_fixed


__all__ = ["fixup_zero_kv"]
