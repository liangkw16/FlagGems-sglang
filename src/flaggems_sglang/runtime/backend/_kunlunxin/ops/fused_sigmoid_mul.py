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

"""Kunlunxin vendor for Task 97 fused_sigmoid_mul.

Our generic 2048-lane flat map read 1.166x on kunlunxin; today's
T93 e1 platform measurement showed the same flat elementwise shape at
16384 lanes reads +55% on this chip (0.442 -> 0.685), matching PR#56's
validated width. Math is unchanged (fp32 tl.sigmoid, the exact
reference formula). The strided-gate decomposition is preserved for
non-contiguous inputs. Not compiled on kunlunxin hardware locally.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 2048


@triton.jit
def _fused_sigmoid_mul_wide(
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
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(
        tl.int64
    )
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
        hidden = attn_output.shape[-1] if attn_output.dim() > 1 else numel
        if gate.dim() == 3:
            gate_d = gate.shape[-1]
            g_s0, g_s1, g_s2 = gate.stride(0), gate.stride(1), gate.stride(2)
        else:
            gate_d = 1
            g_s0 = gate.stride(0) if gate.dim() > 1 else 1
            g_s1 = gate.stride(-1)
            g_s2 = 0
        _fused_sigmoid_mul_wide[(triton.cdiv(numel, _BLOCK),)](
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
            BLOCK=_BLOCK,
            num_warps=8,
        )
    return out


__all__ = ["fused_sigmoid_mul"]
