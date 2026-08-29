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

MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "rotary_embedding.py"
)
SPEC = importlib.util.spec_from_file_location(
    "rotary_embedding_module", MODULE_PATH
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def reference(x, cos, sin, interleaved):
    xf = x.float()
    x1 = xf[..., 0::2]
    x2 = xf[..., 1::2]
    c = cos.float().reshape(cos.shape[0], 1, -1)
    s = sin.float().reshape(sin.shape[0], 1, -1)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    out = torch.stack([o1, o2], dim=-1).reshape(xf.shape)
    return out.to(x.dtype)


class TestRotaryEmbedding(unittest.TestCase):
    TOL = {
        torch.float32: dict(rtol=1e-5, atol=1e-5),
        torch.float16: dict(rtol=1e-2, atol=1e-2),
        torch.bfloat16: dict(rtol=1.5e-2, atol=1.5e-2),
    }

    def _run(self, T, H, D, dtype, interleaved=False):
        torch.manual_seed(T * 1000 + H * 10 + D)
        x = torch.randn(T, H, D, dtype=dtype, device="cuda")
        cos = torch.randn(T, D // 2, dtype=dtype, device="cuda")
        sin = torch.randn(T, D // 2, dtype=dtype, device="cuda")
        out = MOD.rotary_embedding(x, cos, sin, interleaved)
        ref = reference(x, cos, sin, interleaved)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)
        torch.testing.assert_close(out.float(), ref.float(), **self.TOL[dtype])

    def test_matrix(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for T, H, D in [
                (1, 1, 8),
                (4, 8, 64),
                (16, 32, 128),
                (256, 8, 128),
                (4096, 32, 128),
                (128, 4, 192),
                (3, 5, 96),
                (65536, 1, 64),
            ]:
                with self.subTest(dtype=dtype, T=T, H=H, D=D):
                    self._run(T, H, D, dtype)

    def test_fp32_tables(self):
        T, H, D = 32, 8, 128
        torch.manual_seed(1)
        x = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda")
        cos = torch.randn(T, D // 2, dtype=torch.float32, device="cuda")
        sin = torch.randn(T, D // 2, dtype=torch.float32, device="cuda")
        out = MOD.rotary_embedding(x, cos, sin, False)
        ref = reference(x, cos, sin, False)
        torch.testing.assert_close(
            out.float(), ref.float(), rtol=1.5e-2, atol=1.5e-2
        )

    def test_noncontiguous(self):
        T, H, D = 8, 4, 64
        torch.manual_seed(2)
        base = torch.randn(T, H, 2 * D, dtype=torch.float16, device="cuda")
        x = base[:, :, :D]
        cos = torch.randn(T, D // 2, dtype=torch.float16, device="cuda")
        sin = torch.randn(T, D // 2, dtype=torch.float16, device="cuda")
        out = MOD.rotary_embedding(x, cos, sin, False)
        ref = reference(x.contiguous(), cos, sin, False)
        torch.testing.assert_close(
            out.float(), ref.float(), rtol=1e-2, atol=1e-2
        )

    def test_input_invariance(self):
        x = torch.randn(16, 8, 128, dtype=torch.bfloat16, device="cuda")
        cos = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
        sin = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
        xc, cc, sc = x.clone(), cos.clone(), sin.clone()
        MOD.rotary_embedding(x, cos, sin, False)
        self.assertTrue(torch.equal(x, xc))
        self.assertTrue(torch.equal(cos, cc))
        self.assertTrue(torch.equal(sin, sc))

    def test_empty(self):
        x = torch.empty(0, 8, 64, dtype=torch.float16, device="cuda")
        cos = torch.empty(0, 32, dtype=torch.float16, device="cuda")
        sin = torch.empty(0, 32, dtype=torch.float16, device="cuda")
        out = MOD.rotary_embedding(x, cos, sin, False)
        self.assertEqual(out.shape, (0, 8, 64))


if __name__ == "__main__":
    unittest.main()
