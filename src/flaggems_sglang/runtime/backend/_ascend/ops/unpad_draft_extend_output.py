# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Ascend vendor for unpad_draft_extend_output, e21r candidate: the e20/e21
# int32+fp32 ruleset bytes died on UB capacity, not on the ruleset itself -
# submission 20162's raw_result huawei failed_cases[0] reads "ub overflow,
# requires 3145984 bits while 1572864 bits available" (384KB demand vs the
# 192KB Unified Buffer of chip-rulesets.md), and the ~196,640B excess is the
# fp32 offs conversion + fp32 mask materialised at BLOCK=16384 width
# (16384*4B*3-class extra buffers; e14's same-BLOCK form with an integer
# mask compiled and read 442-507, so the fp32 vectors are the only new
# large buffers). Fix: stop materialising wide fp32 vectors at all -
#   - the main loop streams whole BLOCK=16384 tiles UNMASKED (the width
#     peak of the ladder 2048->129.6 / 8192->315.6 / 16384->442 /
#     32768->378 is kept); every touched element lies in
#     [src, src+full_end) which is inside the accepted prefix, and the
#     prefix is inside the raw_out segment / the segment's out rows by
#     contract, so an unmasked whole block is memory-safe (structure
#     precedent: T40 E16 unmasked main + masked last sub-block, huawei
#     +130% - hot path carries zero compares);
#   - a single small BLOCK_TAIL=2048 masked loop drains the remainder,
#     where the fp32 compare operands are loop-local quantities bounded
#     by BLOCK (< 16384 << 2**24) and thus always exactly representable
#     in fp32 - the e20 2**24-domain boundary bug cannot trigger, so the
#     scalar guard branch is deleted (Vector CMP has no int path; Vector
#     ADD has no int64 - addressing stays int32 behind the numel<2**31
#     assert).
# Each loop's live vector set is a subset of e14's already-compiled form
# (int mask + other-filled load at the same width), so UB demand is
# monotonically non-increasing. Kept verbatim from the submitted e21
# (commit 395c6d7f): the official capped grid-stride persistent rotation
# grid=(min(bs, CAP),) with CAP=64 pre-registered (the T77-e5 verified
# shape; a 32/40/48 sweep is the only follow-up if this lands positive
# but under the gate). Each program rotates over segments
# p, p+num_programs, ... (vector_operator.md: "keep the launch close
# to the number of physical Vector Cores and let each program process
# multiple tiles in an inner loop"; "GPU-style small tiles with very
# large grids often cause repeated dispatch overhead on NPUs"), reads
# the segment's lens/cum pair exactly once per visit, and the inner
# loops walk only the real accepted length elems - the padded tail
# beyond n tokens spawns zero idle tiles.

import torch
import triton
import triton.language as tl

_BLOCK = 16384
_BLOCK_TAIL = 2048
_NUM_WARPS = 16
_MAX_PROGRAMS = 64  # pre-registered CAP; T77-e5 capped-request form


@triton.jit
def _unpad(
    raw_out, lens, cum, out, span, tpb, lstride, cstride, bs,
    BLOCK: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
):
    for seg in range(tl.program_id(0), bs, tl.num_programs(0)):
        n = tl.load(lens + seg * lstride)
        beg = tl.load(cum + seg * cstride)
        src = seg * tpb * span
        dst = beg * span
        elems = n * span
        full_end = (elems // BLOCK) * BLOCK
        # unmasked main loop: whole BLOCK tiles strictly inside the
        # accepted prefix; no fp32 offs/mask vector at BLOCK width, which
        # is what overflowed the 192KB UB in e20/e21 (submission 20162)
        for base in range(0, full_end, BLOCK):
            offs = base + tl.arange(0, BLOCK)
            v = tl.load(raw_out + src + offs)
            tl.store(out + dst + offs, v)
        # masked tail drain: offs and rem are loop-local and bounded by
        # BLOCK (< 2**24), so the fp32 compares stay exact - the Vector
        # CMP ruleset path without the e20 boundary bug
        rem = elems - full_end
        for base in range(0, rem, BLOCK_TAIL):
            offs = base + tl.arange(0, BLOCK_TAIL)
            m = offs.to(tl.float32) < rem.to(tl.float32)
            v = tl.load(raw_out + src + full_end + offs, m)
            tl.store(out + dst + full_end + offs, v, m)


def unpad_draft_extend_output(raw_out, cu_seqlens_q, seq_lens_q, sum_seq_lens_q):
    assert raw_out.ndim == 4
    bs, token_per_batch, heads, dim = raw_out.shape
    assert seq_lens_q.shape == (bs,) and cu_seqlens_q.shape == (bs + 1,)
    assert seq_lens_q.dtype == cu_seqlens_q.dtype == torch.int32
    assert raw_out.dtype in (torch.float16, torch.bfloat16)
    assert raw_out.is_contiguous()
    assert raw_out.numel() < 2**31  # int32 element offsets
    out = torch.empty(
        (sum_seq_lens_q, heads, dim),
        dtype=raw_out.dtype,
        device=raw_out.device,
    )
    span = heads * dim
    if bs and token_per_batch and out.numel():
        _unpad[(min(bs, _MAX_PROGRAMS),)](
            raw_out,
            seq_lens_q,
            cu_seqlens_q,
            out,
            span,
            token_per_batch,
            seq_lens_q.stride(0),
            cu_seqlens_q.stride(0),
            bs,
            BLOCK=_BLOCK,
            BLOCK_TAIL=_BLOCK_TAIL,
            num_warps=_NUM_WARPS,
        )
    return out


__all__ = ["unpad_draft_extend_output"]
