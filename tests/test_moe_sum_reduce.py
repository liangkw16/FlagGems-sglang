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
    / "moe_sum_reduce.py"
)
SPEC = importlib.util.spec_from_file_location(
    "moe_sum_reduce_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def reference(input, routed_scaling_factor):
    return input.float().sum(dim=1).mul(routed_scaling_factor).to(input.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class MoeSumReduceTest(unittest.TestCase):
    def test_dtypes_and_hidden_tail_match_reference(self):
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        generator = torch.Generator(device="cuda").manual_seed(20260824)

        for dtype, tolerance in tolerances.items():
            with self.subTest(dtype=dtype):
                input = torch.randn(
                    (3, 8, 257),
                    device="cuda",
                    dtype=dtype,
                    generator=generator,
                )
                original = input.clone()

                actual = MODULE.moe_sum_reduce(input, 0.75)
                expected = reference(input, 0.75)

                self.assertEqual(actual.shape, (3, 257))
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(input, original, atol=0, rtol=0)
                torch.testing.assert_close(
                    actual, expected, atol=tolerance, rtol=tolerance
                )

    def test_noncontiguous_input_uses_all_strides(self):
        base = torch.arange(
            4 * 8 * 38, device="cuda", dtype=torch.float32
        ).reshape(4, 8, 38)
        input = base[::2, 1::2, 1::2]
        self.assertFalse(input.is_contiguous())

        actual = MODULE.moe_sum_reduce(input, -0.125)
        expected = reference(input, -0.125)

        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_empty_dimensions_and_zero_topk_match_reference(self):
        shapes = ((0, 4, 17), (3, 4, 0), (2, 0, 17))

        for shape in shapes:
            with self.subTest(shape=shape):
                input = torch.empty(shape, device="cuda", dtype=torch.float16)

                actual = MODULE.moe_sum_reduce(input, 2.0)
                expected = reference(input, 2.0)

                self.assertEqual(actual.shape, (shape[0], shape[2]))
                self.assertEqual(actual.dtype, input.dtype)
                torch.testing.assert_close(
                    actual, expected, atol=1e-2, rtol=1e-2
                )


if __name__ == "__main__":
    unittest.main()
