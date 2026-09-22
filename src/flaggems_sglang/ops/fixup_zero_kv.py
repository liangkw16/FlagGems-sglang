# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 74338e9 kernels/ops/attention/fixup_zero_kv.py.
# The reference host-syncs on nonzero().tolist(); this variant decides
# everything on-device. E8 landed the fix in place (out/lse are the
# operator's outputs; only zero-KV rows are written, healthy segments
# exit immediately). E17 flattens the writes: with the row-major
# layout a zero-KV segment's out rows form ONE contiguous span
# [beg*HV, end*HV) and its lse rows another [beg*NH, end*NH), so each
# work item streams flat CHUNK-wide stores over its span - no 2D
# tiles, no per-token indexing, perfectly coalesced on every backend
# (the 2D-tile form read 111x on GCU against the field's 193x band).
# Work items are (segment, out-chunk | lse-chunk) pairs, strided by
# block index across the capped grid.

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=["batch", "ot_out", "ot_lse", "hv", "nh", "items"]
)
def _fixup_zero_kv(
    out,
    lse,
    lens,
    cum,
    batch,
    ot_out,
    ot_lse,
    hv,
    nh,
    items,
    CHUNK: tl.constexpr,
):
    per = ot_out + ot_lse
    for it in range(tl.program_id(0), items, tl.num_programs(0)):
        seg = it // per
        sub = it % per
        if tl.load(lens + seg) == 0:
            beg = tl.load(cum + seg)
            end = tl.load(cum + seg + 1)
            if sub < ot_out:
                base = beg * hv
                total = (end - beg) * hv
                for c in range(sub, tl.cdiv(total, CHUNK), ot_out):
                    offs = c * CHUNK + tl.arange(0, CHUNK)
                    zeros = tl.zeros((CHUNK,), dtype=out.dtype.element_ty)
                    tl.store(out + base + offs, zeros, offs < total)
            else:
                s2 = sub - ot_out
                base = beg * nh
                total = (end - beg) * nh
                for c in range(s2, tl.cdiv(total, CHUNK), ot_lse):
                    offs = c * CHUNK + tl.arange(0, CHUNK)
                    ninf = tl.full(
                        (CHUNK,), float("-inf"), dtype=tl.float32
                    )
                    tl.store(lse + base + offs, ninf, offs < total)


def fixup_zero_kv(out, lse, kv_lens, cum_seq_lens, max_seq_len):
    assert out.ndim == 3 and lse.ndim == 2
    total_tokens, num_heads, v_head_dim = out.shape
    assert lse.shape == (total_tokens, num_heads)
    assert out.dtype in (torch.float16, torch.bfloat16)
    assert lse.dtype == torch.float32
    assert out.stride(2) == 1 and out.stride(1) == v_head_dim
    assert lse.stride(1) == 1
    assert out.numel() < 2**31 and lse.numel() < 2**31
    batch = kv_lens.numel()
    assert cum_seq_lens.numel() == batch + 1
    assert kv_lens.dtype == cum_seq_lens.dtype == torch.int32
    hv, nh = num_heads * v_head_dim, num_heads
    if batch and total_tokens:
        chunk = 2048
        span = max_seq_len if isinstance(max_seq_len, int) else total_tokens
        ot_out = max(1, (span * hv + chunk - 1) // chunk)
        ot_lse = max(1, (span * nh + chunk - 1) // chunk)
        items = batch * (ot_out + ot_lse)
        _fixup_zero_kv[(min(items, 2048),)](
            out,
            lse,
            kv_lens,
            cum_seq_lens,
            batch,
            ot_out,
            ot_lse,
            hv,
            nh,
            items,
            CHUNK=chunk,
        )
    return out, lse


__all__ = ["fixup_zero_kv"]
