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

"""Ascend vendor for Task 97 fused_sigmoid_mul.

Our generic thin-program flat map read 1.022x on the platform's
Ascend evaluator; today's T93 e1 measurement showed the PR#70
persistent grid-stride shape (1024-area tiles, num_warps=4, small
persistent grid) doubles this op family on that chip (0.131 ->
0.282). Math is unchanged (fp32 tl.sigmoid). Not compiled on Ascend
hardware locally; the platform evaluator is the arbiter.
"""

import torch
import triton
import triton.language as tl

_TILE = 1024
_PERSISTENT = 512


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
