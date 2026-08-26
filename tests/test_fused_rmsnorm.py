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

KUNLUN_MODULE_PATH = (
    MODULE_PATH.parents[1]
    / "runtime"
    / "backend"
    / "_kunlunxin"
    / "ops"
    / "fused_rmsnorm.py"
)

NVIDIA_MODULE_PATH = (
    MODULE_PATH.parents[1]
    / "runtime"
    / "backend"
    / "_nvidia"
    / "ops"
    / "fused_rmsnorm.py"
)


def _reference(x, weight, eps):
    x32 = x.float()
    rms = torch.sqrt((x32 * x32).mean(dim=-1, keepdim=True) + eps)
    return ((x32 / rms) * weight.float()).to(x.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class FusedRmsnormTest(unittest.TestCase):
    def test_kunlun_multirow_and_boundaries_match_reference(self):
        spec = importlib.util.spec_from_file_location(
            "fused_rmsnorm_kunlunxin_module", KUNLUN_MODULE_PATH
        )
        if spec is None or spec.loader is None:
            self.fail(f"cannot load {KUNLUN_MODULE_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        generator = torch.Generator(device="cuda").manual_seed(20260824)
        for dtype, tolerance in tolerances.items():
            for rows, hidden_size, noncontiguous in (
                (4096, 256, False),
                (4097, 256, False),
                (4097, 128, False),
                (4095, 256, False),
                (4096, 257, False),
                (4096, 256, True),
            ):
                with self.subTest(
                    dtype=dtype,
                    rows=rows,
                    hidden_size=hidden_size,
                    noncontiguous=noncontiguous,
                ):
                    if noncontiguous:
                        x = torch.randn(
                            rows,
                            hidden_size * 2,
                            device="cuda",
                            dtype=dtype,
                            generator=generator,
                        )[:, ::2]
                        weight = torch.randn(
                            hidden_size * 2,
                            device="cuda",
                            dtype=dtype,
                            generator=generator,
                        )[::2]
                    else:
                        x = torch.randn(
                            rows,
                            hidden_size,
                            device="cuda",
                            dtype=dtype,
                            generator=generator,
                        )
                        weight = torch.randn(
                            hidden_size,
                            device="cuda",
                            dtype=dtype,
                            generator=generator,
                        )
                    x_before = x.clone()
                    weight_before = weight.clone()

                    actual = module.fused_rmsnorm(x, weight, 1e-6)
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

    def test_nvidia_dynamic_warps_match_mapping_and_reference(self):
        spec = importlib.util.spec_from_file_location(
            "fused_rmsnorm_nvidia_module", NVIDIA_MODULE_PATH
        )
        if spec is None or spec.loader is None:
            self.fail(f"cannot load {NVIDIA_MODULE_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        launches = []

        class KernelProbe:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    launches.append((grid, kwargs))

                return launch

        kernel = module._fused_rmsnorm_kernel
        module._fused_rmsnorm_kernel = KernelProbe()
        try:
            for hidden_size, expected_warps in (
                (512, 4),
                (1024, 4),
                (1536, 8),
                (2048, 8),
                (3072, 16),
                (4096, 16),
                (5120, 32),
                (8192, 32),
                (8193, 32),
            ):
                x = torch.empty(
                    1, hidden_size, device="cuda", dtype=torch.float16
                )
                weight = torch.empty(
                    hidden_size, device="cuda", dtype=torch.float16
                )
                module.fused_rmsnorm(x, weight, 1e-6)
                grid, kwargs = launches[-1]
                self.assertEqual(grid, (1,))
                self.assertEqual(kwargs["num_warps"], expected_warps)
                self.assertEqual(kwargs["num_stages"], 1)
        finally:
            module._fused_rmsnorm_kernel = kernel

        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        generator = torch.Generator(device="cuda").manual_seed(20260827)
        for dtype, tolerance in tolerances.items():
            for rows, hidden_size in ((7, 512), (3, 3072), (2, 8193)):
                with self.subTest(
                    dtype=dtype,
                    rows=rows,
                    hidden_size=hidden_size,
                ):
                    x = torch.randn(
                        rows,
                        hidden_size,
                        device="cuda",
                        dtype=dtype,
                        generator=generator,
                    )
                    weight = torch.randn(
                        hidden_size,
                        device="cuda",
                        dtype=dtype,
                        generator=generator,
                    )
                    x_before = x.clone()
                    weight_before = weight.clone()

                    actual = module.fused_rmsnorm(x, weight, 1e-6)
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
