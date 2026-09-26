# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""AMD vendor for Task 97 fused_sigmoid_mul (e4: two-segment flat hot path).

Hypothesis (preregistered in ledger fused_sigmoid_mul.md e4): card_b
currently runs the generic kernel, and every one of its tiles pays
three structural costs - the flat offsets are built in i64
(``tl.program_id(0).to(tl.int64) * BLOCK +
tl.arange(0, BLOCK).to(tl.int64)``, src/flaggems_sglang/ops/
fused_sigmoid_mul.py:31), AMD's 64-bit integer address arithmetic
lowers to paired 32-bit VALU ops (architecture-level: the vector ALU
has no native 64-bit integer path, so each i64 add/mul roughly
doubles the integer-instruction count of the addressing pipeline),
and both loads carry ``mask=..., other=0``
(fused_sigmoid_mul.py:45-46), predicating every lane and pre-filling
masked-out lanes. The 金狐狸 breakaway on card_b reads as an
exclusive structure, not a parameter axis: climb-loop s2t1op097
(snapshot 2026-09-26T09:42) card_b 金狐狸 4.1558 (#1) vs ours 3.1582
(#6), gap -0.9976 - the largest single-chip gap on this task - while
the runner-up cgzhou sits at 3.216 and every other team clusters at
3.07-3.17.

This is a port of the e3 structure (platform-validated on THIS op's
huawei: 1.311 -> 1.674 = +28%, submission 21516; ledger
fused_sigmoid_mul.md e3). The flat contiguous hot path (attn AND gate
contiguous - the platform shape) becomes a two-segment grid:
full-tile programs take a ``pid < n_full`` scalar branch (uniform per
program) through a mask-free/other-free 2-load-1-store body; the
single tail program runs a masked load/store with no ``other`` (undef
lanes never reach memory - the store carries the same mask). Offsets
stay int32 strictly inside the exact no-overflow domain: BLOCK=16384
divides 2^31, so for numel <= 2^31 no computed offset exceeds
INT32_MAX (at exactly 2^31 the grid has no tail program and max offs
== INT32_MAX), while numel = 2^31 + 1 would wrap the tail base to
-2^31 - the e3 r2 review finding; past the bound the host routes to
the i64-offset cold twin carrying the generic cast form. The strided
arm keeps the generic kernel bytes (BLOCK=2048/w8, i64 offs, masked
``other=0``) - only the flat contiguous path differs from what card_b
runs today, so any strided-input regression is attributable to this
file's seam alone.

Launch tier is preregistered as UNDECIDED: these initial bytes ship
BLOCK=16384/num_warps=8, and the NVIDIA proxy pre-screen sweeps
8192/16384 x w4/w8 before firing (single-line module-constant
change). The NVIDIA proxy is a disaster gate only - performance does
not extrapolate to card_b (session-mining-retrospective.md:59: card_b
builds its fallback via _amd; §4.1: T24 proxy 5x -> 燧原 -24.5%).

Family and negative evidence, disclosed with weights: same-concept
AMD bytes are armed-unfired (T92 e27 load-side de-mask confined to
_amd/_metax, no platform verdict yet - its reading should calibrate
this candidate alongside ours); T92 E23 (both masks dropped on
_ascend, int64 + 2D grid-stride + per-base guard) read huawei -28.8%
and rolled back, while this file uses int32 + 1D
one-tile-per-program + a single kernel-level branch, and its exact
structural twin is the platform-validated e3 huawei winner (+28%).
Wider family: T40 E16 zero-compare hot path huawei +130%, T39 e10
whole-block scalar skip +246% (evidence carried in
_ascend/ops/add_constant.py).

Preregistered gates (ledger fused_sigmoid_mul.md e4): card_b >= 3.4
keep / >= 3.7 axis-confirmed; card_b < 3.158 or any numeric failure
deletes this file and card_b reverts to the generic bytes. Pure
Triton both arms, host-side static dispatch, no try/except, no
module-level mutable containers. Not compiled on AMD hardware
locally; the platform evaluator is the arbiter.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 16384
_NUM_WARPS = 8
# exact no-overflow domain of the int32 hot path: BLOCK=16384 divides
# 2^31 (131072 * 16384 = 2147483648), so numel <= 2^31 never computes
# an offset above INT32_MAX, while numel = 2^31 + 1 wraps the tail
# base to -2^31 (all lanes then pass `offs < numel` -> OOB)
_INT32_NUMEL_MAX = 2**31


def _flat_cold(numel):
    """Host dispatch: route flat inputs past the int32 hot path's
    exact no-overflow domain to the i64-offset cold twin."""
    return numel > _INT32_NUMEL_MAX


@triton.jit
def _fused_sigmoid_mul_two_segment(
    attn,
    gate,
    out,
    numel,
    n_full,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if pid < n_full:
        # hot path: full tile, 2 loads + 1 store, no mask, no other
        # -> no per-lane predication, no masked-lane prefill, and the
        # addressing pipeline stays pure int32; every lane is
        # in-bounds by construction
        a = tl.load(attn + offs).to(tl.float32)
        g = tl.load(gate + offs).to(tl.float32)
        tl.store(out + offs, (a * tl.sigmoid(g)).to(out.dtype.element_ty))
    else:
        # single tail program: masked load without `other` (undef
        # lanes are never stored - the store carries the same mask)
        m = offs < numel
        a = tl.load(attn + offs, mask=m).to(tl.float32)
        g = tl.load(gate + offs, mask=m).to(tl.float32)
        tl.store(
            out + offs, (a * tl.sigmoid(g)).to(out.dtype.element_ty), mask=m
        )


@triton.jit
def _fused_sigmoid_mul_two_segment_i64(
    attn,
    gate,
    out,
    numel,
    n_full,
    BLOCK: tl.constexpr,
):
    # cold twin for numel > _INT32_NUMEL_MAX: same two-segment body,
    # offsets computed in i64 (the generic cast form)
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    if pid < n_full:
        a = tl.load(attn + offs).to(tl.float32)
        g = tl.load(gate + offs).to(tl.float32)
        tl.store(out + offs, (a * tl.sigmoid(g)).to(out.dtype.element_ty))
    else:
        m = offs < numel
        a = tl.load(attn + offs, mask=m).to(tl.float32)
        g = tl.load(gate + offs, mask=m).to(tl.float32)
        tl.store(
            out + offs, (a * tl.sigmoid(g)).to(out.dtype.element_ty), mask=m
        )


@triton.jit
def _fused_sigmoid_mul_strided(
    attn,
    gate,
    out,
    numel,
    hidden,
    gate_d,
    a_s0,
    a_s1,
    g_s0,
    g_s1,
    g_s2,
    ATTN_CONT: tl.constexpr,
    GATE_CONT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # strided arm: the generic kernel body (i64 offs, masked
    # other=0) - card_b runs these bytes for strided inputs today,
    # kept unchanged so the flat rewrite is the only delta
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < numel
    a_ptr = attn + offs
    g_ptr = gate + offs
    if not (ATTN_CONT and GATE_CONT):
        # stride math in i64: large strides or numel can overflow i32
        row = offs // hidden
        rem = offs - row * hidden
        if not ATTN_CONT:
            a_ptr = attn + row * a_s0 + rem * a_s1
        if not GATE_CONT:
            head = rem // gate_d
            dim = rem - head * gate_d
            g_ptr = gate + row * g_s0 + head * g_s1 + dim * g_s2
    a = tl.load(a_ptr, mask=mask, other=0).to(tl.float32)
    g = tl.load(g_ptr, mask=mask, other=0).to(tl.float32)
    tl.store(out + offs, (a * tl.sigmoid(g)).to(out.dtype.element_ty), mask=mask)


def fused_sigmoid_mul(attn_output, gate):
    assert attn_output.numel() == gate.numel()
    numel = attn_output.numel()
    out = torch.empty(
        attn_output.shape, dtype=attn_output.dtype, device=attn_output.device
    )
    if numel:
        attn_cont = attn_output.is_contiguous()
        gate_cont = gate.is_contiguous()
        if attn_cont and gate_cont:
            # flat hot path: two-segment no-mask grid (host-side
            # static dispatch; both arms are Triton kernels). Past
            # the int32 domain the i64 cold twin takes over.
            n_full = numel // _BLOCK
            grid = (n_full + (1 if numel % _BLOCK else 0),)
            kernel = (
                _fused_sigmoid_mul_two_segment_i64
                if _flat_cold(numel)
                else _fused_sigmoid_mul_two_segment
            )
            kernel[grid](
                attn_output,
                gate,
                out,
                numel,
                n_full,
                BLOCK=_BLOCK,
                num_warps=_NUM_WARPS,
            )
            return out
        hidden = attn_output.shape[-1] if attn_output.dim() > 1 else numel
        if gate.dim() == 3:
            gate_d = gate.shape[-1]
            g_s0, g_s1, g_s2 = gate.stride(0), gate.stride(1), gate.stride(2)
        else:
            gate_d = 1
            g_s0 = gate.stride(0) if gate.dim() > 1 else 1
            g_s1 = gate.stride(-1)
            g_s2 = 0
        _fused_sigmoid_mul_strided[(triton.cdiv(numel, 2048),)](
            attn_output,
            gate,
            out,
            numel,
            hidden,
            gate_d,
            attn_output.stride(0) if attn_output.dim() > 1 else 0,
            attn_output.stride(-1),
            g_s0,
            g_s1,
            g_s2,
            ATTN_CONT=attn_cont,
            GATE_CONT=gate_cont,
            BLOCK=2048,
            num_warps=8,
        )
    return out


__all__ = ["fused_sigmoid_mul"]
