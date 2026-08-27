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
from unittest import mock

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
METAX_MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "runtime"
    / "backend"
    / "_metax"
    / "ops"
    / "moe_sum_reduce.py"
)
AMD_MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "runtime"
    / "backend"
    / "_amd"
    / "ops"
    / "moe_sum_reduce.py"
)
ENFLAME_MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "runtime"
    / "backend"
    / "_enflame"
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
METAX_MODULE = _load_module("moe_sum_reduce_metax_module", METAX_MODULE_PATH)
AMD_MODULE = _load_module("moe_sum_reduce_amd_module", AMD_MODULE_PATH)
ENFLAME_MODULE = _load_module(
    "moe_sum_reduce_enflame_module", ENFLAME_MODULE_PATH
)


def reference(input, routed_scaling_factor):
    return input.float().sum(dim=1).mul(routed_scaling_factor).to(input.dtype)


class MetaxLaunchPolicyTest(unittest.TestCase):
    def test_metax_routes_only_official_qwen_shapes_to_large_launch(self):
        class FakeKernel:
            def __init__(self):
                self.calls = []

            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    self.calls.append((grid, kwargs))

                return launch

        fake_kernel = FakeKernel()
        cases = (
            (torch.empty((2, 8, 2048)), (2, 2), 1024, 8, None),
            (torch.empty((2, 8, 4096)), (2, 4), 1024, 8, None),
            (torch.empty((2, 4, 2048)), (2, 8), 256, 4, 1),
            (torch.empty((2, 8, 2049)), (2, 9), 256, 4, 1),
            (
                torch.empty((2, 8, 4096))[:, :, ::2],
                (2, 8),
                256,
                4,
                1,
            ),
        )

        with mock.patch.object(
            METAX_MODULE, "_moe_sum_reduce_kernel", fake_kernel
        ):
            for (
                input,
                expected_grid,
                block_size,
                num_warps,
                num_stages,
            ) in cases:
                METAX_MODULE.moe_sum_reduce(input, 0.75)
                grid, kwargs = fake_kernel.calls[-1]
                self.assertEqual(grid, expected_grid)
                self.assertEqual(kwargs["BLOCK_SIZE"], block_size)
                self.assertEqual(kwargs["num_warps"], num_warps)
                self.assertEqual(kwargs.get("num_stages"), num_stages)


class AmdLaunchPolicyTest(unittest.TestCase):
    def test_amd_uses_official_configs_and_config_driven_grid(self):
        configs = AMD_MODULE._moe_sum_reduce_kernel.configs
        self.assertEqual(
            AMD_MODULE._moe_sum_reduce_kernel.keys,
            ["hidden_size", "topk"],
        )
        self.assertEqual(
            [
                (config.kwargs["BLOCK_SIZE"], config.num_warps)
                for config in configs
            ],
            [(128, 2), (256, 4), (512, 8), (1024, 8)],
        )

        class FakeKernel:
            def __init__(self):
                self.grids = []
                self.kwargs = None

            def __getitem__(self, grid):
                self.grids = [
                    grid({"BLOCK_SIZE": block_size})
                    for block_size in (128, 256, 512, 1024)
                ]

                def launch(*args, **kwargs):
                    self.kwargs = kwargs

                return launch

        fake_kernel = FakeKernel()
        with mock.patch.object(
            AMD_MODULE, "_moe_sum_reduce_kernel", fake_kernel
        ):
            AMD_MODULE.moe_sum_reduce(torch.empty((2, 3, 1025)), 0.75)

        self.assertEqual(
            fake_kernel.grids,
            [(2, 9), (2, 5), (2, 3), (2, 2)],
        )
        self.assertNotIn("BLOCK_SIZE", fake_kernel.kwargs)
        self.assertNotIn("num_warps", fake_kernel.kwargs)
        self.assertNotIn("num_stages", fake_kernel.kwargs)
        self.assertEqual(fake_kernel.kwargs["topk"], 3)


class AscendLaunchPolicyTest(unittest.TestCase):
    def test_ascend_uses_safe_official_dual_level_configs(self):
        configs = ASCEND_MODULE._moe_sum_reduce_kernel.configs
        self.assertEqual(
            ASCEND_MODULE._moe_sum_reduce_kernel.keys,
            ["num_tokens", "hidden_size", "topk"],
        )
        self.assertEqual(
            [
                (
                    config.kwargs["BLOCK_SIZE"],
                    config.kwargs["BLOCK_SIZE_SUB"],
                    config.num_warps,
                    config.num_stages,
                )
                for config in configs
            ],
            [
                (512, 256, 4, 2),
                (512, 512, 4, 2),
                (1024, 256, 4, 2),
                (1024, 512, 4, 2),
                (1024, 1024, 4, 2),
                (2048, 256, 4, 2),
                (2048, 512, 4, 2),
                (2048, 1024, 4, 2),
            ],
        )

        class FakeKernel:
            def __init__(self):
                self.grids = []
                self.kwargs = None

            def __getitem__(self, grid):
                self.grids = [
                    grid({"BLOCK_SIZE": block_size})
                    for block_size in (512, 1024, 2048)
                ]

                def launch(*args, **kwargs):
                    self.kwargs = kwargs

                return launch

        fake_kernel = FakeKernel()
        with mock.patch.object(
            ASCEND_MODULE, "_moe_sum_reduce_kernel", fake_kernel
        ):
            ASCEND_MODULE.moe_sum_reduce(torch.empty((2, 3, 1025)), 0.75)

        self.assertEqual(
            fake_kernel.grids,
            [(6,), (4,), (2,)],
        )
        self.assertNotIn("BLOCK_SIZE", fake_kernel.kwargs)
        self.assertNotIn("BLOCK_SIZE_SUB", fake_kernel.kwargs)
        self.assertNotIn("num_warps", fake_kernel.kwargs)
        self.assertNotIn("num_stages", fake_kernel.kwargs)
        self.assertEqual(fake_kernel.kwargs["topk"], 3)

    def test_ascend_uses_all_safe_max_shape_programs(self):
        class FakeKernel:
            def __init__(self):
                self.grids = []

            def __getitem__(self, grid):
                self.grids = [
                    grid({"BLOCK_SIZE": block_size})
                    for block_size in (512, 1024, 2048)
                ]

                def launch(*args, **kwargs):
                    return None

                return launch

        fake_kernel = FakeKernel()
        with mock.patch.object(
            ASCEND_MODULE, "_moe_sum_reduce_kernel", fake_kernel
        ):
            ASCEND_MODULE.moe_sum_reduce(torch.empty((4096, 8, 7168)), 0.75)

        self.assertEqual(fake_kernel.grids, [(57344,), (28672,), (16384,)])

    def test_ascend_caps_above_official_grid_limit(self):
        class FakeKernel:
            def __init__(self):
                self.grids = []

            def __getitem__(self, grid):
                self.grids = [
                    grid({"BLOCK_SIZE": block_size})
                    for block_size in (512, 1024, 2048)
                ]

                def launch(*args, **kwargs):
                    return None

                return launch

        fake_kernel = FakeKernel()
        with mock.patch.object(
            ASCEND_MODULE, "_moe_sum_reduce_kernel", fake_kernel
        ):
            ASCEND_MODULE.moe_sum_reduce(torch.empty((65536, 1, 17)), 0.75)

        self.assertEqual(fake_kernel.grids, [(65535,), (65535,), (65535,)])


class EnflameLaunchPolicyTest(unittest.TestCase):
    def test_enflame_uses_sixteenk_tile_fixed_launch(self):
        class FakeKernel:
            def __init__(self):
                self.calls = []

            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    self.calls.append((grid, kwargs))

                return launch

        fake_kernel = FakeKernel()
        with mock.patch.object(
            ENFLAME_MODULE, "_moe_sum_reduce_kernel", fake_kernel
        ):
            ENFLAME_MODULE.moe_sum_reduce(torch.empty((2, 3, 16385)), 0.75)

        grid, kwargs = fake_kernel.calls[-1]
        self.assertEqual(grid, (2, 2))
        self.assertEqual(kwargs["BLOCK_SIZE"], 16384)
        self.assertEqual(kwargs["num_warps"], 8)
        self.assertEqual(kwargs["num_stages"], 1)
        self.assertEqual(kwargs["TOP_K"], 3)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class MoeSumReduceTest(unittest.TestCase):
    def test_amd_autotune_all_dtypes_noncontiguous(self):
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        generator = torch.Generator(device="cuda").manual_seed(20260827)

        for dtype, tolerance in tolerances.items():
            with self.subTest(dtype=dtype):
                base = torch.randn(
                    (4, 3, 2050),
                    device="cuda",
                    dtype=dtype,
                    generator=generator,
                )
                input = base[:, :, 1::2]
                self.assertFalse(input.is_contiguous())
                original = input.clone()

                actual = AMD_MODULE.moe_sum_reduce(input, -0.125)
                expected = reference(input, -0.125)

                self.assertEqual(actual.shape, (4, 1025))
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(
                    actual, expected, atol=tolerance, rtol=tolerance
                )
                torch.testing.assert_close(input, original, atol=0.0, rtol=0.0)

    def test_ascend_autotune_all_dtypes_noncontiguous(self):
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        generator = torch.Generator(device="cuda").manual_seed(20260827)

        for dtype, tolerance in tolerances.items():
            with self.subTest(dtype=dtype):
                base = torch.randn(
                    (4, 3, 2050),
                    device="cuda",
                    dtype=dtype,
                    generator=generator,
                )
                input = base[:, :, 1::2]
                self.assertFalse(input.is_contiguous())
                original = input.clone()

                actual = ASCEND_MODULE.moe_sum_reduce(input, -0.125)
                expected = reference(input, -0.125)

                self.assertEqual(actual.shape, (4, 1025))
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(
                    actual, expected, atol=tolerance, rtol=tolerance
                )
                torch.testing.assert_close(input, original, atol=0.0, rtol=0.0)

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

                    expected = reference(input, 1.25)

                    for module in (MODULE, ASCEND_MODULE, ENFLAME_MODULE):
                        actual = module.moe_sum_reduce(input, 1.25)
                        torch.testing.assert_close(
                            actual, expected, atol=tolerance, rtol=tolerance
                        )
                    torch.testing.assert_close(
                        input, original, atol=0.0, rtol=0.0
                    )

    def test_enflame_sixteenk_block_boundaries_all_dtypes(self):
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        for dtype, tolerance in tolerances.items():
            for hidden_dim in (16383, 16384, 16385):
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

                    expected = reference(input, 1.25)

                    actual = ENFLAME_MODULE.moe_sum_reduce(input, 1.25)
                    self.assertEqual(actual.shape, (2, hidden_dim))
                    self.assertEqual(actual.dtype, dtype)
                    torch.testing.assert_close(
                        actual, expected, atol=tolerance, rtol=tolerance
                    )
                    torch.testing.assert_close(
                        input, original, atol=0.0, rtol=0.0
                    )

    def test_ascend_outer_block_boundaries_all_dtypes(self):
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        generator = torch.Generator(device="cuda").manual_seed(20260827)

        for dtype, tolerance in tolerances.items():
            for hidden_size in (1023, 1024, 1025, 2047, 2048, 2049):
                with self.subTest(dtype=dtype, hidden_size=hidden_size):
                    input = torch.randn(
                        (2, 3, hidden_size),
                        device="cuda",
                        dtype=dtype,
                        generator=generator,
                    )
                    original = input.clone()

                    actual = ASCEND_MODULE.moe_sum_reduce(input, 1.25)
                    expected = reference(input, 1.25)

                    torch.testing.assert_close(
                        actual, expected, atol=tolerance, rtol=tolerance
                    )
                    torch.testing.assert_close(
                        input, original, atol=0.0, rtol=0.0
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

                expected = reference(input, 2.0)

                for module in (MODULE, ASCEND_MODULE, ENFLAME_MODULE):
                    actual = module.moe_sum_reduce(input, 2.0)
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
                    ("enflame", ENFLAME_MODULE),
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

    def test_metax_fast_and_fallback_routes_match_reference(self):
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        generator = torch.Generator(device="cuda").manual_seed(20260827)

        for dtype, tolerance in tolerances.items():
            inputs = (
                torch.randn(
                    (2, 8, 2048),
                    device="cuda",
                    dtype=dtype,
                    generator=generator,
                ),
                torch.randn(
                    (2, 8, 4096),
                    device="cuda",
                    dtype=dtype,
                    generator=generator,
                ),
                torch.randn(
                    (2, 8, 4096),
                    device="cuda",
                    dtype=dtype,
                    generator=generator,
                )[:, :, ::2],
            )
            for input in inputs:
                with self.subTest(dtype=dtype, strides=input.stride()):
                    actual = METAX_MODULE.moe_sum_reduce(input, 0.75)
                    expected = reference(input, 0.75)
                    torch.testing.assert_close(
                        actual, expected, atol=tolerance, rtol=tolerance
                    )


if __name__ == "__main__":
    unittest.main()
