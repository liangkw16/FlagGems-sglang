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
KUNLUN_MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "runtime"
    / "backend"
    / "_kunlunxin"
    / "ops"
    / "moe_sum_reduce.py"
)
ASCEND_MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "runtime"
    / "backend"
    / "_ascend"
    / "ops"
    / "moe_sum_reduce.py"
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_module("moe_sum_reduce_module", MODULE_PATH)
KUNLUN_MODULE = _load_module(
    "moe_sum_reduce_kunlunxin_module", KUNLUN_MODULE_PATH
)
ASCEND_MODULE = _load_module(
    "moe_sum_reduce_ascend_module", ASCEND_MODULE_PATH
)


def reference(input, routed_scaling_factor):
    return input.float().sum(dim=1).mul(routed_scaling_factor).to(input.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class MoeSumReduceTest(unittest.TestCase):
    def test_block_boundaries_all_dtypes(self):
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        for dtype, tolerance in tolerances.items():
            for hidden_dim in (255, 256, 257, 511, 512, 513):
                with self.subTest(dtype=dtype, hidden_dim=hidden_dim):
                    input = (
                        torch.linspace(
                            -2,
                            2,
                            2 * 3 * hidden_dim,
                            device="cuda",
                        )
                        .reshape(2, 3, hidden_dim)
                        .to(dtype)
                    )
                    original = input.clone()

                    actual = MODULE.moe_sum_reduce(input, 1.25)
                    expected = reference(input, 1.25)

                    torch.testing.assert_close(
                        input, original, atol=0.0, rtol=0.0
                    )
                    torch.testing.assert_close(
                        actual, expected, atol=tolerance, rtol=tolerance
                    )

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

    def test_vendors_cover_platform_failure_scale(self):
        num_tokens, top_k, hidden_dim = 4096, 8, 7168
        hidden_blocks = (hidden_dim + 255) // 256
        self.assertEqual(num_tokens * hidden_blocks, 114688)
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        generator = torch.Generator(device="cuda").manual_seed(20260824)

        for dtype, tolerance in tolerances.items():
            with self.subTest(dtype=dtype):
                input = torch.randn(
                    (num_tokens, top_k, hidden_dim),
                    device="cuda",
                    dtype=dtype,
                    generator=generator,
                )
                original = input.clone()
                expected = reference(input, 0.75)

                for name, module in (
                    ("generic", MODULE),
                    ("kunlunxin", KUNLUN_MODULE),
                    ("ascend", ASCEND_MODULE),
                ):
                    with self.subTest(module=name):
                        actual = module.moe_sum_reduce(input, 0.75)

                        self.assertEqual(
                            actual.shape, (num_tokens, hidden_dim)
                        )
                        self.assertEqual(actual.dtype, dtype)
                        torch.testing.assert_close(
                            actual, expected, atol=tolerance, rtol=tolerance
                        )

                torch.testing.assert_close(input, original, atol=0.0, rtol=0.0)

    def test_kunlun_vendor_block1024_boundaries(self):
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        generator = torch.Generator(device="cuda").manual_seed(20260824)

        for dtype, tolerance in tolerances.items():
            for hidden_dim in (1023, 1024, 1025, 2049):
                with self.subTest(dtype=dtype, hidden_dim=hidden_dim):
                    input = torch.randn(
                        (5, 4, hidden_dim),
                        device="cuda",
                        dtype=dtype,
                        generator=generator,
                    )
                    expected = reference(input, 0.75)

                    actual = KUNLUN_MODULE.moe_sum_reduce(input, 0.75)

                    self.assertEqual(actual.shape, (5, hidden_dim))
                    self.assertEqual(actual.dtype, dtype)
                    torch.testing.assert_close(
                        actual, expected, atol=tolerance, rtol=tolerance
                    )


if __name__ == "__main__":
    unittest.main()
