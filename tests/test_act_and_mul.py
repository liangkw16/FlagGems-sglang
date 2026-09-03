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
import torch.nn.functional as F

MODULE_PATH = (
    Path(__file__).parents[1] / "src" / "flaggems_sglang" / "ops" / "act_and_mul.py"
)
SPEC = importlib.util.spec_from_file_location("act_and_mul_module", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

TOLERANCES = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1.5e-2, 1.5e-2),
}


def reference(gateup_output, activation="silu", swiglu_limit=None):
    hidden_size = gateup_output.shape[1]
    half = hidden_size // 2
    gate = gateup_output[:, :half].float()
    up = gateup_output[:, half:].float()

    if swiglu_limit is not None:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)

    if activation == "silu":
        act = F.silu(gate)
    elif activation == "gelu":
        act = F.gelu(gate, approximate="tanh")
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    out = (act.to(gateup_output.dtype) * up.to(gateup_output.dtype)).to(
        gateup_output.dtype
    )
    return out


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class ActAndMulTest(unittest.TestCase):
    def _check(
        self,
        gateup_output,
        activation="silu",
        swiglu_limit=None,
        equal_nan=False,
    ):
        actual = MODULE.act_and_mul(
            gateup_output, activation=activation, swiglu_limit=swiglu_limit
        )
        expected = reference(
            gateup_output, activation=activation, swiglu_limit=swiglu_limit
        )
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        atol, rtol = TOLERANCES[gateup_output.dtype]
        torch.testing.assert_close(
            actual, expected, atol=atol, rtol=rtol, equal_nan=equal_nan
        )
        return actual

    def test_contiguous_dtypes_and_activations(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for activation in ("silu", "gelu"):
                with self.subTest(dtype=dtype, activation=activation):
                    x = torch.randn(64, 512, device="cuda", dtype=dtype) * 3.0
                    self._check(x, activation=activation)

    def test_half_width_and_row_boundaries(self):
        for rows, d in (
            (1, 1),
            (1, 7),
            (3, 63),
            (5, 64),
            (7, 65),
            (2, 511),
            (2, 512),
            (2, 513),
            (129, 1023),
            (129, 1024),
            (129, 1025),
            (4, 4096),
        ):
            with self.subTest(rows=rows, d=d):
                x = torch.randn(rows, 2 * d, device="cuda") * 2.0
                self._check(x)

    def test_contract_is_2d_only(self):
        # The task statement defines gateup_output as [M, 2H]; its
        # reference slices dim 1 ([:, :half]) and has no defined
        # semantics for higher-rank inputs, so the covered contract is
        # strictly 2D rows. Exercise a tall and a wide 2D case instead.
        self._check(torch.randn(4097, 8192, device="cuda"))
        self._check(torch.randn(1, 2, device="cuda"))

    def test_non_contiguous_input(self):
        base = torch.randn(32, 2048, device="cuda")
        x = base[:, ::2]
        self.assertFalse(x.is_contiguous())
        self._check(x)

    def test_input_not_modified(self):
        x = torch.randn(16, 1024, device="cuda", dtype=torch.float16)
        snapshot = x.clone()
        self._check(x)
        torch.testing.assert_close(x, snapshot)

    def test_empty_rows_and_zero_width(self):
        out = MODULE.act_and_mul(torch.randn(0, 64, device="cuda"))
        self.assertEqual(out.shape, (0, 32))
        self._check(torch.randn(0, 64, device="cuda"))
        out = MODULE.act_and_mul(torch.randn(8, 0, device="cuda"))
        self.assertEqual(out.shape, (8, 0))

    def test_special_values_match_reference(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for activation in ("silu", "gelu"):
                with self.subTest(dtype=dtype, activation=activation):
                    gate = torch.tensor(
                        [
                            [float("-inf"), -1e4, -92.0, -90.0, -8.0, -0.0],
                            [0.0, 1e-8, 0.5, 1.0, 8.0, 90.0],
                            [92.0, 1e4, float("inf"), float("nan"), 3.0, -3.0],
                        ],
                        device="cuda",
                        dtype=dtype,
                    )
                    up = torch.tensor(
                        [
                            [float("-inf"), -1e4, -92.0, -90.0, -8.0, -0.0],
                            [0.0, 1e-8, 0.5, 1.0, 8.0, 90.0],
                            [92.0, 1e4, float("inf"), float("nan"), 3.0, -3.0],
                        ],
                        device="cuda",
                        dtype=dtype,
                    )
                    x = torch.cat([gate, up], dim=1)
                    # silu(-inf) is NaN in both implementations; the
                    # comparison must tolerate matching NaNs.
                    self._check(x, activation=activation, equal_nan=True)

    def test_swiglu_limit_clamping(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for activation in ("silu", "gelu"):
                for limit in (0.5, 7.0, 1e4):
                    with self.subTest(dtype=dtype, activation=activation, limit=limit):
                        x = torch.randn(33, 1026, device="cuda", dtype=dtype) * 20.0
                        self._check(x, activation=activation, swiglu_limit=limit)

    def test_swiglu_limit_zero_still_clamps(self):
        # 0.0 is falsy but must behave as a real limit, not as None.
        x = torch.randn(16, 512, device="cuda") * 3.0
        out = MODULE.act_and_mul(x, swiglu_limit=0.0)
        self.assertTrue(torch.all(out == 0))

    def test_asymmetric_clamp_semantics(self):
        # gate is clamped on max only, up symmetrically; probe both sides.
        x = torch.zeros(1, 8, device="cuda")
        x[0, 0] = -50.0  # gate below -limit stays untouched
        x[0, 1] = 50.0  # gate above limit clamps down
        x[0, 2] = 0.25  # gate inside window
        x[0, 4] = -50.0  # up below -limit clamps up
        x[0, 5] = 50.0  # up above limit clamps down
        x[0, 6] = 0.25  # up inside window
        self._check(x, swiglu_limit=1.0)

    def test_unsupported_activation_raises(self):
        x = torch.randn(4, 64, device="cuda")
        with self.assertRaises(ValueError):
            MODULE.act_and_mul(x, activation="relu")
        with self.assertRaises(ValueError):
            reference(x, activation="relu")


if __name__ == "__main__":
    unittest.main()
