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
    / "gelu_and_mul.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gelu_and_mul_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

TOLERANCES = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1.5e-2, 1.5e-2),
}


def reference(hidden_states):
    import torch.nn.functional as F

    d = hidden_states.shape[-1] // 2
    x1, x3 = hidden_states[..., :d], hidden_states[..., d:]
    out = F.gelu(x1.float(), approximate="none") * x3.float()
    return out.to(hidden_states.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class GeluAndMulTest(unittest.TestCase):
    def _check(self, hidden_states, equal_nan=False):
        actual = MODULE.gelu_and_mul(hidden_states)
        expected = reference(hidden_states)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        atol, rtol = TOLERANCES[hidden_states.dtype]
        torch.testing.assert_close(
            actual, expected, atol=atol, rtol=rtol, equal_nan=equal_nan
        )
        return actual

    def test_contiguous_dtypes_match_reference(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.randn(64, 512, device="cuda", dtype=dtype) * 3.0
                self._check(x)

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

    def test_multiple_leading_dims(self):
        x = torch.randn(2, 3, 5, 256, device="cuda")
        self._check(x)

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
        out = MODULE.gelu_and_mul(torch.randn(0, 64, device="cuda"))
        self.assertEqual(out.shape, (0, 32))
        self._check(torch.randn(0, 64, device="cuda"))
        out = MODULE.gelu_and_mul(torch.randn(8, 0, device="cuda"))
        self.assertEqual(out.shape, (8, 0))

    def test_special_values_match_reference(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                gate = torch.tensor(
                    [
                        float("-inf"),
                        -1e4,
                        -8.0,
                        -0.0,
                        0.0,
                        8.0,
                        1e4,
                        float("inf"),
                        float("nan"),
                    ],
                    device="cuda",
                    dtype=torch.float32,
                ).to(dtype)
                up = torch.tensor(
                    [-2.0, 3.0, 0.5, 0.0, -0.0, -1.5, 0.25, 0.0, 2.0],
                    device="cuda",
                    dtype=torch.float32,
                ).to(dtype)
                x = torch.stack([gate, up], dim=-1).reshape(1, -1)
                self._check(x, equal_nan=True)

    def test_grid_stride_fold_path(self):
        rows = 65536
        d = 1024
        x = (torch.randn(rows, 2 * d, device="cuda") * 2.0).to(torch.float16)
        self.assertGreater(rows * d, 65535 * 1024)
        self._check(x)


if __name__ == "__main__":
    unittest.main()
