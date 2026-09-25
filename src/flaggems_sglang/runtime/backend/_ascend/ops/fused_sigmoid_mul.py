# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Ascend vendor for Task 97 fused_sigmoid_mul (two-segment no-mask hot path).

The persistent grid-stride bytes this file's strided arm keeps (git
5cfdb654, blob SHA-256
ab88bc50632db1b20f842849892881a3875a79ae49790b1572a519aae873ef8c)
carry two documented Ascend pathologies on EVERY tile of the hot
loop: `m = offs < numel` is a vector compare that Ascend lowers to
scalar code (chip-rulesets.md:36), and `tl.load(..., other=0)`
pre-fills the masked lanes and serializes MTE2 (chip-rulesets.md:39).
Platform readings pinned the remaining gap as structural, not a
launch-parameter axis: the thin flat map read 1.022 (submission
b2db48ca), persistent-izing lifted huawei to 1.311 (submission
9ab3c5c4) while the 48-vs-512 CTA cap was measured neutral, so the
CTA-count axis is exhausted; leaders show huawei 2.0+ is reachable
on this op (金狐狸 2.185, GuanghuLab 2.013).

This rewrite splits the flat contiguous hot path (attn AND gate
contiguous - the platform shape) into a two-segment grid: full-tile
programs take a `pid < n_full` scalar branch (uniform per program)
through a mask-free/other-free 2-load-1-store body, and the single
tail program runs a masked load/store with no `other` (undef lanes
never reach memory - the store carries the same mask). BLOCK=16384 /
num_warps=16 per the chip-rulesets.md:40 ladder. The int32 hot
variant is dispatched only inside its exact no-overflow domain:
BLOCK=16384 divides 2^31, so for numel <= 2^31 no computed offset
exceeds INT32_MAX (at exactly 2^31 the grid has no tail program and
max offs == INT32_MAX), while at numel = 2^31 + 1 the tail base
wraps to -2^31 and every lane passes `offs < numel` (review r2
finding, reproduced by a local int32 wraparound simulation). Above
the bound the host routes to an i64-offset cold twin carrying the
same cast form as the generic/kunlunxin kernels - the flat path in
this file was previously the library's only unguarded one. The
strided arm keeps the proven persistent bytes unchanged (1024-area
tiles, 48 CTAs, w4); that arm's flat addressing remains int32 as
before (pre-existing, safety beyond 2^31 not verifiable locally;
triggering needs a single >=4GiB bf16 input, which competition
shapes do not reach - disclosure, not a launch gate).

Family evidence for the two-segment shape (as carried in the T93 E3
armed docstring, _ascend/ops/add_constant.py): T40 E16 hot path with
zero per-tile compare -> huawei +130%; T39 e10 uniform scalar branch
skipping whole blocks -> huawei +246%. Closest same-concept precedent
is NEGATIVE and weighs equally: T92 E23 (whole-block unmasked
load/store + masked tail) passed 8/8 but read huawei -28.8% and was
rolled back; its bytes carried int64 addressing on a 2D grid-stride
loop with a per-base guard, while this file uses int32 addressing, a
1D one-tile-per-program grid and a single kernel-level branch. The
family evidence does not decompose which factor drove E23's
regression, so the huawei sign is genuinely open; T93 E3 (same
family, 1 load/1 store) is armed-unfired and its platform verdict
should calibrate this candidate's launch. Preregistered gates
(ledger fused_sigmoid_mul.md e3): numeric failure or huawei < 1.311
rolls _ascend back to the 5cfdb654 bytes; huawei >= 1.6 keeps,
>= 1.9 confirms the axis; kunlunxin < 1.0 is a water-level re-roll
(chip-rulesets.md:58, +/-10-35% swing), not attributed to the 2048
restore shipping in the same package. Not compiled on Ascend
hardware locally; the platform evaluator is the arbiter.
"""

import torch
import triton
import triton.language as tl

_TILE = 1024
_PERSISTENT = 48
_BLOCK = 16384
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
        # -> no vector-compare lowering, no MTE2 prefill; every lane
        # is in-bounds by construction
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
    # offsets computed in i64 (the generic/kunlunxin cast form)
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
def _fused_sigmoid_mul_persistent(
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
    TILE: tl.constexpr,
):
    for base in range(
        tl.program_id(0) * TILE, numel, tl.num_programs(0) * TILE
    ):
        offs = base + tl.arange(0, TILE)
        mask = offs < numel
        a_ptr = attn + offs
        g_ptr = gate + offs
        if not (ATTN_CONT and GATE_CONT):
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
        tl.store(
            out + offs,
            (a * tl.sigmoid(g)).to(out.dtype.element_ty),
            mask=mask,
        )


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
                num_warps=16,
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
        grid = (min(triton.cdiv(numel, _TILE), _PERSISTENT),)
        _fused_sigmoid_mul_persistent[grid](
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
            TILE=_TILE,
            num_warps=4,
        )
    return out


__all__ = ["fused_sigmoid_mul"]
