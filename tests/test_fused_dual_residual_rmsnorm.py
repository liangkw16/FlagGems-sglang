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
    / "fused_dual_residual_rmsnorm.py"
)
SPEC = importlib.util.spec_from_file_location(
    "fused_dual_residual_rmsnorm_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

TOL = {torch.float32: (1e-4, 1e-4), torch.float16: (1e-2, 1e-2),
       torch.bfloat16: (1.5e-2, 1.5e-2)}


def _rmsnorm32(v32, w32, eps):
    rms = torch.sqrt((v32 * v32).mean(dim=-1, keepdim=True) + eps)
    return v32 / rms * w32


def reference(x, residual, weight1, weight2, eps):
    mid = residual + _rmsnorm32(
        x.float(), weight1.float(), eps
    ).to(residual.dtype)
    out = _rmsnorm32(mid.float(), weight2.float(), eps).to(x.dtype)
    return out, mid


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class FusedDualResidualRmsnormTest(unittest.TestCase):
    def _check(self, x, residual, w1, w2, eps=1e-5):
        actual_out, actual_mid = MODULE.fused_dual_residual_rmsnorm(
            x, residual, w1, w2, eps
        )
        exp_out, exp_mid = reference(x, residual, w1, w2, eps)
        for name, got, exp, dt in (
            ("out", actual_out, exp_out, x.dtype),
            ("mid", actual_mid, exp_mid, residual.dtype),
        ):
            self.assertEqual(got.shape, exp.shape, name)
            self.assertEqual(got.dtype, exp.dtype, name)
            atol, rtol = TOL[dt]
            torch.testing.assert_close(got, exp, atol=atol, rtol=rtol)

    def test_same_dtype(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                T, D = 32, 512
                x = torch.randn(T, D, device="cuda", dtype=dtype)
                r = torch.randn(T, D, device="cuda", dtype=dtype)
                w1 = torch.randn(D, device="cuda", dtype=dtype)
                w2 = torch.randn(D, device="cuda", dtype=dtype)
                self._check(x, r, w1, w2)

    def test_mixed_dtype(self):
        # x fp32, residual bf16 (different cast targets)
        x = torch.randn(16, 256, device="cuda", dtype=torch.float32)
        r = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randn(256, device="cuda", dtype=torch.float32)
        w2 = torch.randn(256, device="cuda", dtype=torch.float32)
        self._check(x, r, w1, w2)

    def test_shapes(self):
        for T, D in ((1, 16), (100, 1023), (5, 4096), (1, 8192)):
            with self.subTest(T=T, D=D):
                x = torch.randn(T, D, device="cuda")
                r = torch.randn(T, D, device="cuda")
                w1 = torch.randn(D, device="cuda")
                w2 = torch.randn(D, device="cuda")
                self._check(x, r, w1, w2)

    def test_empty(self):
        x = torch.randn(0, 128, device="cuda")
        r = torch.randn(0, 128, device="cuda")
        w1 = torch.randn(128, device="cuda")
        w2 = torch.randn(128, device="cuda")
        out, mid = MODULE.fused_dual_residual_rmsnorm(x, r, w1, w2, 1e-5)
        self.assertEqual(out.shape, (0, 128))
        self.assertEqual(mid.shape, (0, 128))


if __name__ == "__main__":
    unittest.main()
