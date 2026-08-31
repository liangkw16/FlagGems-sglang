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
    / "per_token_quant_int8.py"
)
SPEC = importlib.util.spec_from_file_location(
    "per_token_quant_int8_module", MODULE_PATH
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

_EPS = 1e-10


def group_reference(x, group_size, dtype=torch.int8):
    iinfo = torch.iinfo(dtype)
    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = (
        x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=_EPS).to(torch.float32)
    )
    x_s = amax / iinfo.max
    x_q = (x_ / x_s).clamp(min=iinfo.min, max=iinfo.max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))
    return x_q, x_s


def reference(x):
    return group_reference(x, x.shape[-1])


class TestPerTokenQuantInt8(unittest.TestCase):
    def _check(self, x):
        x_q, x_s = MOD.per_token_quant_int8(x)
        ref_q, ref_s = reference(x)
        self.assertEqual(x_q.dtype, torch.int8)
        self.assertEqual(x_s.dtype, torch.float32)
        self.assertEqual(x_q.shape, x.shape)
        self.assertEqual(x_s.shape, x.shape[:-1] + (1,))
        self.assertTrue(torch.equal(x_q, ref_q))
        torch.testing.assert_close(x_s, ref_s, rtol=1e-6, atol=1e-8)

    def test_matrix(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for rows, n in [
                (256, 512),
                (1024, 2560),
                (8192, 512),
                (65536, 256),
                (129, 1025),
                (1, 128),
                (64, 96),
                # e3 row-pack tail boundaries: rows not divisible by the
                # packed ROWS_TILE (16/8/4/2 for these N) and the 1024/2048
                # dispatch boundary
                (1023, 256),
                (77, 100),
                (33, 1024),
                (5, 2048),
            ]:
                torch.manual_seed(rows + n)
                x = torch.randn(rows, n, dtype=dtype, device="cuda") * 3
                with self.subTest(dtype=dtype, rows=rows, n=n):
                    self._check(x)

    def test_trunc_boundary(self):
        x = torch.tensor(
            [[0.9, -0.9, 1.6, -1.6, 127.6, -200.0, 0.0, 255.0]],
            dtype=torch.float32,
            device="cuda",
        )
        x_q, x_s = MOD.per_token_quant_int8(x)
        ref_q, ref_s = reference(x)
        torch.testing.assert_close(x_q, ref_q, rtol=0, atol=0)

    def test_zero_row(self):
        x = torch.zeros(4, 128, dtype=torch.float32, device="cuda")
        x_q, x_s = MOD.per_token_quant_int8(x)
        ref_q, ref_s = reference(x)
        torch.testing.assert_close(x_q, ref_q, rtol=0, atol=0)
        torch.testing.assert_close(x_s, ref_s, rtol=1e-6, atol=1e-12)

    def test_multi_dim_and_noncontig(self):
        torch.manual_seed(7)
        base = torch.randn(2, 3, 256, dtype=torch.float16, device="cuda")
        with self.subTest("multi-dim"):
            self._check(base)
        sliced = base[:, :, :128]
        x_q, x_s = MOD.per_token_quant_int8(sliced)
        ref_q, ref_s = reference(sliced.contiguous())
        torch.testing.assert_close(x_q, ref_q, rtol=0, atol=0)

    def test_invariance_and_empty(self):
        x = torch.randn(128, 256, dtype=torch.float16, device="cuda")
        xc = x.clone()
        MOD.per_token_quant_int8(x)
        self.assertTrue(torch.equal(x, xc))
        e = torch.empty(0, 128, dtype=torch.float16, device="cuda")
        x_q, x_s = MOD.per_token_quant_int8(e)
        self.assertEqual(x_q.shape, (0, 128))
        self.assertEqual(x_s.shape, (0, 1))


if __name__ == "__main__":
    unittest.main()
