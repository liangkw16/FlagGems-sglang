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
    / "fused_rmsnorm.py"
)
SPEC = importlib.util.spec_from_file_location(
    "fused_rmsnorm_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _reference(x, weight, eps):
    x32 = x.float()
    rms = torch.sqrt((x32 * x32).mean(dim=-1, keepdim=True) + eps)
    return ((x32 / rms) * weight.float()).to(x.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class FusedRmsnormTest(unittest.TestCase):
    def test_contiguous_decode_shapes_match_reference(self):
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        for dtype, tolerance in tolerances.items():
            for rows, hidden_size in (
                (1, 512),
                (4, 4096),
                (4, 5120),
                (4, 8192),
            ):
                with self.subTest(
                    dtype=dtype, rows=rows, hidden_size=hidden_size
                ):
                    x = (
                        torch.linspace(
                            -3.0,
                            3.0,
                            rows * hidden_size,
                            device="cuda",
                        )
                        .reshape(rows, hidden_size)
                        .to(dtype)
                    )
                    if rows == 1:
                        x.zero_()
                    weight = torch.linspace(
                        0.5, 1.5, hidden_size, device="cuda"
                    ).to(dtype)

                    actual = MODULE.fused_rmsnorm(x, weight, 1e-5)
                    expected = _reference(x, weight, 1e-5)

                    torch.testing.assert_close(
                        actual, expected, atol=tolerance, rtol=tolerance
                    )

    def test_public_api_matches_reference(self):
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        for dtype, tolerance in tolerances.items():
            for hidden_size in (513, 8193):
                with self.subTest(dtype=dtype, hidden_size=hidden_size):
                    x_storage = torch.randn(
                        (3, hidden_size * 2), device="cuda", dtype=dtype
                    )
                    weight_storage = torch.randn(
                        hidden_size * 2, device="cuda", dtype=dtype
                    )
                    x = x_storage[:, ::2]
                    weight = weight_storage[1::2]
                    x_before = x.clone()
                    weight_before = weight.clone()

                    actual = MODULE.fused_rmsnorm(x, weight, 1e-6)
                    expected = _reference(x, weight, 1e-6)

                    self.assertEqual(actual.shape, x.shape)
                    self.assertEqual(actual.dtype, x.dtype)
                    torch.testing.assert_close(
                        actual, expected, atol=tolerance, rtol=tolerance
                    )
                    torch.testing.assert_close(x, x_before, atol=0.0, rtol=0.0)
                    torch.testing.assert_close(
                        weight, weight_before, atol=0.0, rtol=0.0
                    )

    def test_empty_input_preserves_contract(self):
        x = torch.empty((0, 513), device="cuda", dtype=torch.float16)
        weight = torch.ones(513, device="cuda", dtype=torch.float16)

        actual = MODULE.fused_rmsnorm(x, weight, 1e-6)

        self.assertEqual(actual.shape, x.shape)
        self.assertEqual(actual.dtype, x.dtype)
        self.assertEqual(actual.numel(), 0)


if __name__ == "__main__":
    unittest.main()
