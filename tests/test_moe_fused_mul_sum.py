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
    / "moe_fused_mul_sum.py"
)

VENDOR_PATHS = {
    vendor: Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "runtime"
    / "backend"
    / f"_{vendor}"
    / "ops"
    / "moe_fused_mul_sum.py"
    for vendor in ("kunlunxin", "enflame")
    if (
        Path(__file__).parents[1]
        / "src"
        / "flaggems_sglang"
        / "runtime"
        / "backend"
        / f"_{vendor}"
        / "ops"
        / "moe_fused_mul_sum.py"
    ).exists()
}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_module("moe_fused_mul_sum_module", MODULE_PATH)

VENDOR_MODULES = {
    name: _load_module(f"moe_fused_mul_sum_{name}", path)
    for name, path in VENDOR_PATHS.items()
}

TOLERANCES = {
    torch.float16: 1e-2,
    torch.bfloat16: 1.5e-2,
    torch.float32: 1e-4,
}


def reference(
    inputs,
    topk_weights,
    topk_ids=None,
    expert_map=None,
    routed_scaling_factor=None,
    is_ep=False,
):
    scale = 1.0 if routed_scaling_factor is None else routed_scaling_factor
    w = topk_weights.float() * scale

    if expert_map is not None:
        valid = expert_map[topk_ids.long()] >= 0
        w = w * valid.to(w.dtype)
    elif is_ep:
        valid = topk_ids >= 0
        w = w * valid.to(w.dtype)

    weighted = inputs.float() * w.unsqueeze(-1)
    out = weighted.sum(dim=1)
    return out.to(inputs.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class MoeFusedMulSumTest(unittest.TestCase):
    def test_all_dtypes_plain_sum_match_reference(self):
        generator = torch.Generator(device="cuda").manual_seed(20260829)

        for dtype, tolerance in TOLERANCES.items():
            with self.subTest(dtype=dtype):
                inputs = torch.randn(
                    (3, 8, 257),
                    device="cuda",
                    dtype=dtype,
                    generator=generator,
                )
                topk_weights = torch.rand(
                    (3, 8),
                    device="cuda",
                    dtype=torch.float32,
                    generator=generator,
                )
                original = inputs.clone()

                actual = MODULE.moe_fused_mul_sum(
                    inputs, topk_weights, routed_scaling_factor=0.75
                )
                expected = reference(
                    inputs, topk_weights, routed_scaling_factor=0.75
                )

                self.assertEqual(actual.shape, (3, 257))
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(
                    actual, expected, atol=tolerance, rtol=tolerance
                )
                torch.testing.assert_close(
                    inputs, original, atol=0.0, rtol=0.0
                )

    def test_scale_none_matches_reference(self):
        inputs = torch.randn((5, 4, 512), device="cuda", dtype=torch.float32)
        topk_weights = torch.rand((5, 4), device="cuda")

        actual = MODULE.moe_fused_mul_sum(inputs, topk_weights)
        expected = reference(inputs, topk_weights)

        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_expert_map_masking_drops_invalid_experts(self):
        generator = torch.Generator(device="cuda").manual_seed(20260829)
        inputs = torch.randn(
            (7, 4, 1025),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        topk_weights = torch.rand((7, 4), device="cuda", generator=generator)
        # experts 1 and 3 are not owned by this EP rank
        expert_map = torch.tensor(
            [0, -1, 2, -1], device="cuda", dtype=torch.int32
        )
        topk_ids = torch.randint(
            0, 4, (7, 4), device="cuda", generator=generator, dtype=torch.int32
        )

        actual = MODULE.moe_fused_mul_sum(
            inputs, topk_weights, topk_ids, expert_map, 2.5
        )
        expected = reference(inputs, topk_weights, topk_ids, expert_map, 2.5)

        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    def test_is_ep_masking_drops_negative_ids(self):
        generator = torch.Generator(device="cuda").manual_seed(20260829)
        inputs = torch.randn(
            (6, 3, 511),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        topk_weights = torch.rand((6, 3), device="cuda", generator=generator)
        topk_ids = torch.randint(
            -2,
            8,
            (6, 3),
            device="cuda",
            generator=generator,
            dtype=torch.int32,
        )

        actual = MODULE.moe_fused_mul_sum(
            inputs, topk_weights, topk_ids, None, None, True
        )
        expected = reference(inputs, topk_weights, topk_ids, None, None, True)

        torch.testing.assert_close(actual, expected, atol=1.5e-2, rtol=1.5e-2)

    def test_vendor_matrix_heavy_ep_drop_and_plain_parity(self):
        # e4 zero-weight slot skip: dropped slots contribute exactly nothing,
        # so results stay bit-compatible with the reference while their input
        # slabs are never fetched; covers all shipped kernel variants
        generator = torch.Generator(device="cuda").manual_seed(20260831)
        modules = {"generic": MODULE, **VENDOR_MODULES}
        for name, module in modules.items():
            for dtype, atol in TOLERANCES.items():
                with self.subTest(name=name, dtype=dtype):
                    inputs = torch.randn(
                        (33, 8, 2048),
                        device="cuda",
                        dtype=dtype,
                        generator=generator,
                    )
                    topk_weights = torch.rand(
                        (33, 8), device="cuda", generator=generator
                    )
                    drop = (
                        torch.rand((33, 8), device="cuda", generator=generator)
                        < 0.875
                    )
                    ids = torch.randint(
                        0,
                        16,
                        (33, 8),
                        device="cuda",
                        generator=generator,
                        dtype=torch.int32,
                    )
                    topk_ids = torch.where(
                        drop,
                        torch.tensor(-1, device="cuda", dtype=torch.int32),
                        ids,
                    )
                    actual = module.moe_fused_mul_sum(
                        inputs, topk_weights, topk_ids, None, 2.5, True
                    )
                    expected = reference(
                        inputs, topk_weights, topk_ids, None, 2.5, True
                    )
                    torch.testing.assert_close(
                        actual, expected, atol=atol, rtol=atol
                    )
                    actual_plain = module.moe_fused_mul_sum(
                        inputs, topk_weights
                    )
                    expected_plain = reference(inputs, topk_weights)
                    torch.testing.assert_close(
                        actual_plain, expected_plain, atol=atol, rtol=atol
                    )
            with self.subTest(name=name, case="expert_map_heavy_drop"):
                expert_map = torch.tensor(
                    [0] + [-1] * 15, device="cuda", dtype=torch.int32
                )
                inputs = torch.randn(
                    (17, 8, 1024),
                    device="cuda",
                    dtype=torch.float16,
                    generator=generator,
                )
                topk_weights = torch.rand(
                    (17, 8), device="cuda", generator=generator
                )
                topk_ids = torch.randint(
                    0,
                    16,
                    (17, 8),
                    device="cuda",
                    generator=generator,
                    dtype=torch.int32,
                )
                actual = module.moe_fused_mul_sum(
                    inputs, topk_weights, topk_ids, expert_map, 2.5
                )
                expected = reference(
                    inputs, topk_weights, topk_ids, expert_map, 2.5
                )
                torch.testing.assert_close(
                    actual, expected, atol=1e-2, rtol=1e-2
                )

    def test_noncontiguous_input_uses_all_strides(self):
        base = torch.arange(
            4 * 8 * 38, device="cuda", dtype=torch.float32
        ).reshape(4, 8, 38)
        inputs = base[::2, 1::2, 1::2]
        topk_weights = torch.rand((2, 4), device="cuda")
        self.assertFalse(inputs.is_contiguous())

        actual = MODULE.moe_fused_mul_sum(
            inputs, topk_weights, routed_scaling_factor=-0.125
        )
        expected = reference(
            inputs, topk_weights, routed_scaling_factor=-0.125
        )

        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_block_boundaries_all_dtypes(self):
        for dtype, tolerance in TOLERANCES.items():
            for hidden_dim in (255, 256, 257, 511, 512, 513):
                with self.subTest(dtype=dtype, hidden_dim=hidden_dim):
                    inputs = (
                        torch.linspace(
                            -2, 2, 2 * 3 * hidden_dim, device="cuda"
                        )
                        .reshape(2, 3, hidden_dim)
                        .to(dtype)
                    )
                    topk_weights = torch.rand((2, 3), device="cuda")
                    expected = reference(
                        inputs, topk_weights, routed_scaling_factor=1.25
                    )

                    actual = MODULE.moe_fused_mul_sum(
                        inputs, topk_weights, routed_scaling_factor=1.25
                    )
                    torch.testing.assert_close(
                        actual, expected, atol=tolerance, rtol=tolerance
                    )

    def test_empty_dimensions_and_zero_topk_match_reference(self):
        shapes = ((0, 4, 17), (3, 4, 0), (2, 0, 17))

        for shape in shapes:
            with self.subTest(shape=shape):
                inputs = torch.empty(shape, device="cuda", dtype=torch.float16)
                topk_weights = torch.rand((shape[0], shape[1]), device="cuda")

                expected = reference(
                    inputs, topk_weights, routed_scaling_factor=2.0
                )

                actual = MODULE.moe_fused_mul_sum(
                    inputs, topk_weights, routed_scaling_factor=2.0
                )
                self.assertEqual(actual.shape, (shape[0], shape[2]))
                self.assertEqual(actual.dtype, inputs.dtype)
                torch.testing.assert_close(
                    actual, expected, atol=1e-2, rtol=1e-2
                )

    def test_platform_scale_qwen_moe_shape(self):
        num_tokens, top_k, hidden_dim = 4096, 8, 7168
        generator = torch.Generator(device="cuda").manual_seed(20260829)

        for dtype, tolerance in TOLERANCES.items():
            with self.subTest(dtype=dtype):
                inputs = torch.randn(
                    (num_tokens, top_k, hidden_dim),
                    device="cuda",
                    dtype=dtype,
                    generator=generator,
                )
                topk_weights = torch.rand(
                    (num_tokens, top_k), device="cuda", generator=generator
                )
                original = inputs.clone()
                expected = reference(
                    inputs, topk_weights, routed_scaling_factor=0.75
                )

                actual = MODULE.moe_fused_mul_sum(
                    inputs, topk_weights, routed_scaling_factor=0.75
                )

                self.assertEqual(actual.shape, (num_tokens, hidden_dim))
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(
                    actual, expected, atol=tolerance, rtol=tolerance
                )
                torch.testing.assert_close(
                    inputs, original, atol=0.0, rtol=0.0
                )


if __name__ == "__main__":
    unittest.main()
