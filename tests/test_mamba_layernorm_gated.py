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
    / "mamba_layernorm_gated.py"
)
SPEC = importlib.util.spec_from_file_location(
    "mamba_layernorm_gated_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _reference(
    x,
    weight,
    bias,
    eps,
    z=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=True,
):
    rows, hidden_size = x.shape
    if group_size is None:
        group_size = hidden_size
    group_count = hidden_size // group_size
    output_dtype = x.dtype

    x_float = x.float().view(rows, group_count, group_size)
    z_float = (
        z.float().view(rows, group_count, group_size)
        if z is not None
        else None
    )
    if z_float is not None and not norm_before_gate:
        x_float = x_float * z_float * torch.sigmoid(z_float)

    if is_rms_norm:
        variance = (x_float * x_float).mean(dim=-1, keepdim=True)
    else:
        mean = x_float.mean(dim=-1, keepdim=True)
        x_float = x_float - mean
        variance = (x_float * x_float).mean(dim=-1, keepdim=True)

    normalized = x_float * torch.rsqrt(variance + eps)
    output = normalized * weight.float().view(group_count, group_size)
    if bias is not None:
        output = output + bias.float().view(group_count, group_size)
    if z_float is not None and norm_before_gate:
        output = output * z_float * torch.sigmoid(z_float)
    return output.reshape(rows, hidden_size).to(output_dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class MambaLayernormGatedTest(unittest.TestCase):
    def test_public_api_branches_match_reference(self):
        torch.manual_seed(20260824)
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        cases = (
            ("rms_plain", None, False, False, True, True),
            ("ln_group_post_gate", 37, True, True, True, False),
            ("rms_group_pre_gate", 37, True, True, False, True),
            ("ln_pre_gate", None, False, True, False, False),
        )
        rows, hidden_size = 3, 259

        for dtype, tolerance in tolerances.items():
            for (
                name,
                group_size,
                with_bias,
                with_z,
                norm_before_gate,
                is_rms_norm,
            ) in cases:
                with self.subTest(dtype=dtype, case=name):
                    x_storage = torch.randn(
                        (rows, hidden_size * 2), device="cuda", dtype=dtype
                    )
                    weight_storage = torch.randn(
                        hidden_size * 2, device="cuda", dtype=dtype
                    )
                    bias_storage = torch.randn(
                        hidden_size * 2, device="cuda", dtype=dtype
                    )
                    z_storage = torch.randn(
                        (rows, hidden_size * 2), device="cuda", dtype=dtype
                    )
                    x = x_storage[:, ::2]
                    weight = weight_storage[::2]
                    bias = bias_storage[1::2] if with_bias else None
                    z = z_storage[:, 1::2] if with_z else None
                    x_before = x.clone()
                    weight_before = weight.clone()
                    bias_before = bias.clone() if bias is not None else None
                    z_before = z.clone() if z is not None else None

                    actual = MODULE.mamba_layernorm_gated(
                        x,
                        weight,
                        bias,
                        1e-6,
                        z=z,
                        group_size=group_size,
                        norm_before_gate=norm_before_gate,
                        is_rms_norm=is_rms_norm,
                    )
                    expected = _reference(
                        x,
                        weight,
                        bias,
                        1e-6,
                        z=z,
                        group_size=group_size,
                        norm_before_gate=norm_before_gate,
                        is_rms_norm=is_rms_norm,
                    )

                    self.assertFalse(x.is_contiguous())
                    self.assertEqual(actual.shape, x.shape)
                    self.assertEqual(actual.dtype, x.dtype)
                    torch.testing.assert_close(
                        actual, expected, atol=tolerance, rtol=tolerance
                    )
                    torch.testing.assert_close(x, x_before, atol=0.0, rtol=0.0)
                    torch.testing.assert_close(
                        weight, weight_before, atol=0.0, rtol=0.0
                    )
                    if bias is not None:
                        torch.testing.assert_close(
                            bias, bias_before, atol=0.0, rtol=0.0
                        )
                    if z is not None:
                        torch.testing.assert_close(
                            z, z_before, atol=0.0, rtol=0.0
                        )

    def test_empty_input_preserves_contract(self):
        x = torch.empty((0, 259), device="cuda", dtype=torch.float16)
        weight = torch.ones(259, device="cuda", dtype=torch.float16)

        actual = MODULE.mamba_layernorm_gated(
            x, weight, None, 1e-6, group_size=37
        )

        self.assertEqual(actual.shape, x.shape)
        self.assertEqual(actual.dtype, x.dtype)
        self.assertEqual(actual.numel(), 0)

    def test_group_boundaries_match_reference(self):
        torch.manual_seed(20260825)
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }

        for dtype, tolerance in tolerances.items():
            for group_size in (1, 255, 256, 257, 511, 512, 513):
                with self.subTest(dtype=dtype, group_size=group_size):
                    hidden_size = group_size * 2
                    x = torch.randn(
                        (2, hidden_size), device="cuda", dtype=dtype
                    )
                    weight = torch.randn(
                        hidden_size, device="cuda", dtype=dtype
                    )
                    bias = torch.randn(hidden_size, device="cuda", dtype=dtype)
                    z = torch.randn_like(x)

                    actual = MODULE.mamba_layernorm_gated(
                        x,
                        weight,
                        bias,
                        1e-6,
                        z=z,
                        group_size=group_size,
                        norm_before_gate=False,
                        is_rms_norm=False,
                    )
                    expected = _reference(
                        x,
                        weight,
                        bias,
                        1e-6,
                        z=z,
                        group_size=group_size,
                        norm_before_gate=False,
                        is_rms_norm=False,
                    )

                    torch.testing.assert_close(
                        actual, expected, atol=tolerance, rtol=tolerance
                    )


if __name__ == "__main__":
    unittest.main()
