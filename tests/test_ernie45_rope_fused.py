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

import importlib.util
import unittest
from pathlib import Path

import torch

from tests._op_variants import load_operator_modules

MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "ernie45_rope_fused.py"
)
SPEC = importlib.util.spec_from_file_location(
    "ernie45_rope_fused_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

TOL = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1.5e-2, 1.5e-2),
}


def _apply_rope(x, n_h, head_size, rotary_dim, cos, sin):
    num_tokens = x.shape[0]
    half_rd = rotary_dim // 2
    x = x.view(num_tokens, n_h, head_size).clone()
    x1 = x[..., :half_rd].float()
    x2 = x[..., half_rd:rotary_dim].float()
    cos_e = cos.unsqueeze(1)
    sin_e = sin.unsqueeze(1)
    new1 = x1 * cos_e - x2 * sin_e
    new2 = x2 * cos_e + x1 * sin_e
    out = torch.cat(
        [new1.to(x.dtype), new2.to(x.dtype), x[..., rotary_dim:]], dim=-1
    )
    return out.view(num_tokens, n_h * head_size)


def reference(
    q, k, cos_sin_cache, positions, mrope_section, head_size, rotary_dim
):
    num_tokens, n_q_dim = q.shape
    n_k_dim = k.shape[1]
    n_qh = n_q_dim // head_size
    n_kh = n_k_dim // head_size
    half_rd = rotary_dim // 2
    section_h, section_w, section_t = mrope_section
    section_hw = section_h + section_w
    tpos = positions[0].long()
    hpos = positions[1].long()
    wpos = positions[2].long()
    ridx = torch.arange(half_rd, device=q.device)
    use_hw = (ridx < section_hw).unsqueeze(0)
    use_h = ((ridx % 2) == 0).unsqueeze(0)
    pos_hw = torch.where(use_h, hpos.unsqueeze(1), wpos.unsqueeze(1))
    pos = torch.where(use_hw, pos_hw, tpos.unsqueeze(1))
    col = ridx.unsqueeze(0).expand(num_tokens, half_rd)
    cos = cos_sin_cache[pos, col].float()
    sin = cos_sin_cache[pos, col + half_rd].float()
    q_out = _apply_rope(q, n_qh, head_size, rotary_dim, cos, sin)
    k_out = _apply_rope(k, n_kh, head_size, rotary_dim, cos, sin)
    return q_out, k_out


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class Ernie45RopeFusedTest(unittest.TestCase):
    def _check(self, q, k, cache, pos, sec, hs, rd):
        q_snap, k_snap = q.clone(), k.clone()
        aq, ak = MODULE.ernie45_rope_fused(q, k, cache, pos, sec, hs, rd)
        eq, ek = reference(q, k, cache, pos, sec, hs, rd)
        for name, got, exp, dt in (
            ("q", aq, eq, q.dtype),
            ("k", ak, ek, k.dtype),
        ):
            self.assertEqual(got.shape, exp.shape, name)
            self.assertEqual(got.dtype, exp.dtype, name)
            atol, rtol = TOL[dt]
            torch.testing.assert_close(got, exp, atol=atol, rtol=rtol)
        torch.testing.assert_close(q, q_snap)
        torch.testing.assert_close(k, k_snap)

    def test_dtypes(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                T, hs, rd = 32, 128, 64
                sec = [16, 16, 0]  # h+w+t = 32 = rd//2
                q = torch.randn(T, 8 * hs, device="cuda", dtype=dtype)
                k = torch.randn(T, 2 * hs, device="cuda", dtype=dtype)
                cache = torch.randn(1024, rd, device="cuda", dtype=dtype)
                pos = torch.randint(0, 1000, (3, T), device="cuda")
                self._check(q, k, cache, pos, sec, hs, rd)

    def test_sections(self):
        for rd, sec in (
            (64, [16, 16, 0]),
            (64, [8, 8, 16]),
            (128, [32, 32, 0]),
            (128, [16, 16, 32]),
        ):
            with self.subTest(rd=rd, sec=sec):
                T, hs = 16, 128
                q = torch.randn(T, 8 * hs, device="cuda")
                k = torch.randn(T, 2 * hs, device="cuda")
                cache = torch.randn(1024, rd, device="cuda")
                pos = torch.randint(0, 1000, (3, T), device="cuda")
                self._check(q, k, cache, pos, sec, hs, rd)

    def test_gqa_and_tail(self):
        # rotary_dim < head_size (tail pass-through)
        for hs, rd in ((128, 64), (128, 128), (64, 32)):
            with self.subTest(hs=hs, rd=rd):
                T = 8
                sec_sum = rd // 2
                sec = [
                    sec_sum // 3,
                    sec_sum // 3,
                    sec_sum - 2 * (sec_sum // 3),
                ]
                # Ensure h==w
                sec[1] = sec[0]
                sec[2] = sec_sum - 2 * sec[0]
                q = torch.randn(T, 8 * hs, device="cuda")
                k = torch.randn(T, 2 * hs, device="cuda")
                cache = torch.randn(256, rd, device="cuda")
                pos = torch.randint(0, 200, (3, T), device="cuda")
                self._check(q, k, cache, pos, sec, hs, rd)

    def test_empty(self):
        q = torch.randn(0, 1024, device="cuda")
        k = torch.randn(0, 256, device="cuda")
        cache = torch.randn(100, 64, device="cuda")
        pos = torch.zeros(3, 0, dtype=torch.int64, device="cuda")
        aq, ak = MODULE.ernie45_rope_fused(
            q, k, cache, pos, [16, 16, 0], 128, 64
        )
        self.assertEqual(aq.shape, (0, 1024))
        self.assertEqual(ak.shape, (0, 256))


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class Ernie45RopeVariantsTest(unittest.TestCase):
    """Core matrix across every backend variant (generic + vendors)."""

    MODULES = load_operator_modules("ernie45_rope_fused")

    def test_variants_match_reference(self):
        # (T, head_size, rotary_dim, section, dtype): covers no-tail and
        # t-section splits, a non-128 head size, and a bf16 pass so the
        # per-pair vendor indexing is checked beyond the fp32 defaults.
        cases = (
            (16, 128, 64, [16, 16, 0], torch.float32),
            (16, 128, 128, [16, 16, 32], torch.float32),
            (8, 64, 32, [8, 8, 0], torch.float32),
            (5, 128, 64, [6, 6, 20], torch.bfloat16),
        )
        for T, hs, rd, sec, dtype in cases:
            q = torch.randn(T, 8 * hs, device="cuda", dtype=dtype)
            k = torch.randn(T, 2 * hs, device="cuda", dtype=dtype)
            cache = torch.randn(256, rd, device="cuda", dtype=dtype)
            pos = torch.randint(0, 200, (3, T), device="cuda")
            eq, ek = reference(q, k, cache, pos, sec, hs, rd)
            atol, rtol = TOL[dtype]
            for name, module in self.MODULES:
                with self.subTest(module=name, rd=rd, hs=hs, dtype=dtype):
                    aq, ak = module.ernie45_rope_fused(
                        q, k, cache, pos, sec, hs, rd
                    )
                    torch.testing.assert_close(aq, eq, atol=atol, rtol=rtol)
                    torch.testing.assert_close(ak, ek, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
