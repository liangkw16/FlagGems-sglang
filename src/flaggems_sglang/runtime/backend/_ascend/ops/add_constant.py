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

"""Ascend vendor for Task 93 add_constant (two-segment no-mask hot path).

The persistent bytes this file replaces (git 5cfdb654, member SHA-256
7b3ee54ba624ad0d7bac434ea3b2a3d3fa68986a4ac17d903c01aadaa0997040)
carried two documented Ascend pathologies on EVERY tile of the hot
loop: `m = offs < numel` is an int32 vector compare that Ascend Vector
CMP lowers to scalar code (chip-rulesets.md:36), and `tl.load(...,
other=0)` pre-fills the masked lanes and serializes MTE2
(chip-rulesets.md:39). The platform readings pinned the cost as
structural, not a parameter axis: grid cap 48 vs 512 was measured
neutral (ledger e1/e2, add_constant.md:8) while huawei sits at 0.2472
against wangteam 0.9878 / OpeGoodn 0.9860.

Family evidence for the two-segment shape this file adopts:
- T40 E16: hot path with zero per-tile compare, only the last
  sub-block masked -> huawei 0.74->1.70 (+130%, experiments
  README.md:984-990, "int32/64 Vector CMP 标量退化 = elementwise 头号杀手");
- T39 e10: uniform scalar branch skipping whole blocks -> huawei
  +246% (README.md:735);
- T92 e22 (current huawei-proven bytes): dropping `other=` from the
  masked load kept numerics exact because the store carries the same
  mask, so undef lanes never reach memory; same form as upstream
  FlagGems pointwise_dynamic, whose generated loads are
  `tl.load(..., mask=mask)` without `other`.
- T92 e21r warns that its combined form (unmasked main loop + fp32
  tail + persistent rotation) compiled on huawei but failed numerics
  (test[3] 1520/164352). This file keeps the flat one-tile-per-program
  geometry, no fp32 anywhere, and a single scalar branch, so none of
  that failure surface is inherited; the preregistered numeric gate
  rolls _ascend back to the 5cfdb654 bytes on any failure.
- The closest same-concept precedent is NEGATIVE and weighs equally:
  T92 E23 (ledger unpad_draft_extend_output.md, "E23 平台终态
  (20457)") ran whole-block unmasked load/store plus a masked tail on
  the same huawei Ascend, passed 8/8 correct, yet huawei read 316.2016
  vs e22's 444.1506 (-28.8%, net -92.0072 across eight chips), hit its
  preregistered avg gate and was rolled back (commit 47dc1382).
  Differences that plausibly matter here: E23 carried int64 addressing
  on a (bs,tiles) 2D grid-stride loop with a per-base scalar guard,
  while this file uses int32 addressing, a 1D one-tile-per-program grid
  and a single kernel-level pid branch. The family evidence does not
  decompose which factor drove E23's regression, so the huawei sign of
  this candidate is genuinely open; the preregistered gates bound the
  downside (see ledger add_constant.md e3): numeric failure, huawei
  <0.5, or avg <= 0.841 (e1 team best) rolls _ascend back to the
  5cfdb654 bytes; huawei >=0.9 swaps the team best.

Structure: grid = numel // BLOCK full-tile programs plus exactly one
tail program when numel % BLOCK != 0. A kernel-internal scalar branch
on `pid < n_full` (uniform per program; both arms are Triton compute,
no PyTorch fallback) sends full tiles down a mask-free/other-free
load-store and the single tail program down a masked load with no
`other`. BLOCK=16384 / num_warps=16 is the preregistered pick from the
T92 e22 huawei-proven bytes (chip-rulesets.md:40 ladder: >=4096 -> 16
warps for compute kernels); int32 addressing stays in-domain
(chip-rulesets.md:36). CONSTANT stays a JIT fold. Not compiled on
Ascend hardware locally; the platform evaluator is the arbiter.
"""

import torch
import triton
import triton.language as tl

_TILE = 16384


@triton.jit
def _add_constant_two_segment(
    src,
    out,
    numel,
    n_full,
    CONSTANT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if pid < n_full:
        # hot path: full tile, no mask, no other -> no Vector CMP,
        # no MTE2 prefill; every lane is in-bounds by construction
        value = tl.load(src + offs)
        tl.store(out + offs, value + CONSTANT)
    else:
        # single tail program: masked load without `other` (undef
        # lanes are never stored - the store carries the same mask)
        m = offs < numel
        value = tl.load(src + offs, mask=m)
        tl.store(out + offs, value + CONSTANT, mask=m)


def add_constant(src, constant):
    assert src.dtype == torch.int32
    assert src.dim() == 1 and src.is_contiguous()
    numel = src.numel()
    out = torch.empty_like(src)
    if numel:
        n_full = numel // _TILE
        grid = (n_full + (1 if numel % _TILE else 0),)
        _add_constant_two_segment[grid](
            src, out, numel, n_full, constant, BLOCK=_TILE, num_warps=16
        )
    return out


__all__ = ["add_constant"]
