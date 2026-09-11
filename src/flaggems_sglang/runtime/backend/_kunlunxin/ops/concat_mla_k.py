# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Kunlunxin vendor: every concat_mla_k submission so far (s0, s0r, s0r2
# and the BH=16 round) died in the Kunlun evaluator's XMLIR stack with
# "服务线程卡死自动恢复" -- four deterministic crashes across two
# kernel tunings, which moves this from the intermittent crash family to
# a form-specific compiler fault. The shared shape of every crash is the
# generic's 2D [BH, BN]/[BH, BR] tile with two segmented stores off one
# base expression. This vendor flattens to the one-program-per-row shape
# that Kunlun passes elsewhere on this platform (l2norm row blocks,
# kv_indices row loops): each program copies one output row with a
# single padded 1D range, two masked loads and two masked stores over
# disjoint ranges, and grid-stride over rows. Shape scalars stay
# unspecialized per the Kunlun recompile-storm guidance from the
# FlagGems permute_copy vendor.

import torch
import triton
import triton.language as tl

_MAX_GRID = 65535


@triton.jit(do_not_specialize=["rows", "heads", "nd", "rd"])
def _concat_mla_k_rows(
    nope,
    rope,
    out,
    rows,
    heads,
    nd,
    rd,
    ns0,
    ns1,
    ns2,
    rs0,
    rs2,
    os1,
    os2,
    D_PAD: tl.constexpr,
):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        row64 = row.to(tl.int64)
        token = row64 // heads
        head = row64 % heads
        d = tl.arange(0, D_PAD).to(tl.int64)
        nm = d < nd
        rm = (d >= nd) & (d < nd + rd)
        no = tl.load(nope + token * ns0 + head * ns1 + d * ns2, nm, other=0)
        ro = tl.load(rope + token * rs0 + (d - nd) * rs2, rm, other=0)
        # row = token * heads + head, so the output row stride is the
        # head-dim stride (out.stride(1)), not the token stride.
        tl.store(out + row64 * os1 + d * os2, no, nm)
        tl.store(out + row64 * os1 + d * os2, ro, rm)


def concat_mla_k(k, k_nope, k_rope):
    assert k.ndim == k_nope.ndim == k_rope.ndim == 3
    tokens, heads, dim = k.shape
    nd, rd = k_nope.shape[2], k_rope.shape[2]
    assert k_nope.shape[:2] == (tokens, heads)
    assert k_rope.shape[:2] == (tokens, 1) and nd + rd == dim
    assert k.dtype == k_nope.dtype == k_rope.dtype == torch.bfloat16
    out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if out.numel():
        rows = tokens * heads
        _concat_mla_k_rows[(min(rows, _MAX_GRID),)](
            k_nope,
            k_rope,
            out,
            rows,
            heads,
            nd,
            rd,
            *k_nope.stride(),
            k_rope.stride(0),
            k_rope.stride(2),
            out.stride(1),
            out.stride(2),
            D_PAD=triton.next_power_of_2(max(1, dim)),
        )
    return out


__all__ = ["concat_mla_k"]
