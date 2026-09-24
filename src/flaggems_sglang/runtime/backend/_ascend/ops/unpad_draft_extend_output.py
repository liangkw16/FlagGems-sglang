# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Ascend vendor for unpad_draft_extend_output, e22 candidate: revert the
# ascend bytes to the e19r proven form (submission 19541 water band
# 378-507; ZIP e19r-f9abadc ascend member == git f9abadc, (bs, tiles)
# grid + BLOCK=16384 + int64 offs/elems integer mask, the e14 lineage)
# and apply exactly ONE variable on top: drop the other=0 prefill of the
# masked load. chip-rulesets.md:25 - "masked load 的 other 预填会串行化
# MTE2": this pure copy kernel has carried other=0 in every round since
# e14, and load-side MTE2 serialisation is the prime suspect for the
# huawei 442-507 band vs the 金狐狸/CosmosMind 684-792 band (per-chip
# gap decomposition, climb-loop.json s2t1op092: huawei 684.79 vs 377.9
# is 56% of the total avg gap). Numerics: masked-out lanes now hold
# undef instead of 0, and the store carries the same mask m, so those
# lanes never reach memory - output bytes are identical (NVIDIA proxy
# verifies). Nothing from the e21r failure surface is inherited: no
# unmasked main loop, no fp32 tail, no persistent rotation, no warps16
# (e21r 20313 failed test[3] 1520/164352 with those variables present);
# the live vector set is a subset of e19r's already-compiled form, so
# UB demand is monotonically non-increasing vs the 192KB budget the
# e20/e21 int32+fp32 bytes overflowed.

import torch
import triton
import triton.language as tl


@triton.jit
def _unpad(
    raw_out, lens, cum, out, span, tpb, lstride, cstride,
    BLOCK: tl.constexpr,
):
    seg = tl.program_id(0)
    tile = tl.program_id(1)
    n = tl.load(lens + seg.to(tl.int64) * lstride)
    beg = tl.load(cum + seg.to(tl.int64) * cstride)
    # segment bases stay int64 scalars (raw_out can exceed 2^31 elements)
    # but the per-lane offset pipeline runs int32: a 16384-lane int64
    # vector is 128KB of slow i64 ALU/UB on AscendVector, and the
    # in-segment offsets are bounded by token_per_batch*span << 2^31
    src = seg.to(tl.int64) * tpb * span
    dst = (beg.to(tl.int64) * span)
    elems32 = n * span
    for base in range(tile * BLOCK, elems32, tl.num_programs(1) * BLOCK):
        offs = base + tl.arange(0, BLOCK)
        m = offs < elems32
        v = tl.load(raw_out + src + offs, m)
        tl.store(out + dst + offs, v, m)


@triton.jit
def _unpad_wide(
    raw_out, lens, cum, out, span, tpb, lstride, cstride,
    BLOCK: tl.constexpr,
):
    # int64 pipeline for segments whose element count can reach 2^31
    # (the wrapper dispatches on token_per_batch*span); numerically the
    # pre-e24 bytes
    seg = tl.program_id(0)
    tile = tl.program_id(1)
    n = tl.load(lens + seg.to(tl.int64) * lstride)
    beg = tl.load(cum + seg.to(tl.int64) * cstride)
    src = seg.to(tl.int64) * tpb * span
    dst = (beg.to(tl.int64) * span)
    elems = n.to(tl.int64) * span
    for base in range(
        tile.to(tl.int64) * BLOCK, elems, tl.num_programs(1).to(tl.int64) * BLOCK
    ):
        offs = base + tl.arange(0, BLOCK)
        m = offs < elems
        v = tl.load(raw_out + src + offs, m)
        tl.store(out + dst + offs, v, m)


def unpad_draft_extend_output(raw_out, cu_seqlens_q, seq_lens_q, sum_seq_lens_q):
    assert raw_out.ndim == 4
    bs, token_per_batch, heads, dim = raw_out.shape
    assert seq_lens_q.shape == (bs,) and cu_seqlens_q.shape == (bs + 1,)
    assert seq_lens_q.dtype == cu_seqlens_q.dtype == torch.int32
    assert raw_out.dtype in (torch.float16, torch.bfloat16)
    assert raw_out.is_contiguous()
    out = torch.empty(
        (sum_seq_lens_q, heads, dim),
        dtype=raw_out.dtype,
        device=raw_out.device,
    )
    span = heads * dim
    if bs and token_per_batch and out.numel():
        tiles = min(max(1, (token_per_batch * span + 16383) // 16384), 255)
        # int32 offsets only when every in-segment element count fits
        # the signed domain (n <= token_per_batch bounds elems by
        # token_per_batch*span); otherwise the int64 pipeline
        kernel = (
            _unpad
            if token_per_batch * span < 2**31
            else _unpad_wide
        )
        kernel[(bs, tiles)](
            raw_out,
            seq_lens_q,
            cu_seqlens_q,
            out,
            span,
            token_per_batch,
            seq_lens_q.stride(0),
            cu_seqlens_q.stride(0),
            BLOCK=16384,
            num_warps=16,
        )
    return out


__all__ = ["unpad_draft_extend_output"]
