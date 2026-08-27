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
    / "interleaved_rope.py"
)
SPEC = importlib.util.spec_from_file_location(
    "interleaved_rope_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def reference(x, mrope_section):
    _, S, D = x.shape
    d = torch.arange(D, device=x.device)
    cond_a = (d % 3 == 1) & (d < mrope_section[1] * 3)
    cond_b = (d % 3 == 2) & (d < mrope_section[2] * 3)

    out = x[0].clone()
    out[:, cond_a] = x[1][:, cond_a]
    out[:, cond_b] = x[2][:, cond_b]
    return out


def sections_for(dim):
    third = dim // 3
    return [third, 0, 0]


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class InterleavedRopeTest(unittest.TestCase):
    def _check(self, x, mrope_section):
        actual = MODULE.interleaved_rope(x, mrope_section)
        expected = reference(x, mrope_section)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        torch.testing.assert_close(
            actual, expected, atol=0.0, rtol=0.0, equal_nan=True
        )
        return actual

    def test_dtypes_match_reference(self):
        for dtype in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int64,
        ):
            with self.subTest(dtype=dtype):
                x = torch.randn(3, 17, 96, device="cuda").to(dtype)
                self._check(x, [16, 8, 8])

    def test_shape_and_section_boundaries(self):
        cases = [
            (5, 8, [1, 1, 0]),
            (5, 8, [1, 0, 1]),
            (5, 8, [2, 0, 0]),
            (5, 8, [0, 1, 1]),
            (7, 9, [1, 1, 1]),
            (7, 9, [0, 2, 1]),
            (3, 16, [3, 1, 1]),
            (2, 63, [11, 5, 5]),
            (2, 64, [21, 0, 0]),
            (2, 65, [20, 1, 0]),
            (2, 511, [100, 70, 0]),
            (2, 512, [170, 0, 0]),
            (2, 513, [100, 71, 0]),
            (129, 1023, [200, 100, 41]),
            (129, 1024, [241, 100, 0]),
            (129, 1025, [141, 200, 0]),
            (4, 4096, [1000, 180, 185]),
            (1, 192, [16, 24, 24]),
        ]
        for seq_len, dim, section in cases:
            with self.subTest(seq_len=seq_len, dim=dim, section=section):
                assert sum(section) == dim // 3
                x = torch.randn(3, seq_len, dim, device="cuda")
                self._check(x, section)

    def test_boundary_columns_exact(self):
        dim = 96
        x = torch.arange(3 * 4 * dim, device="cuda", dtype=torch.float32)
        x = x.reshape(3, 4, dim)
        section = [16, 8, 8]
        out = self._check(x, section)
        for d in range(dim):
            if d % 3 == 1 and d < 24:
                expected_stream = 1
            elif d % 3 == 2 and d < 24:
                expected_stream = 2
            else:
                expected_stream = 0
            torch.testing.assert_close(
                out[:, d], x[expected_stream][:, d], atol=0.0, rtol=0.0
            )

    def test_non_contiguous_input(self):
        base = torch.randn(3, 8, 512, device="cuda")
        x = base[:, :, ::2]
        self.assertFalse(x.is_contiguous())
        self.assertEqual(x.shape, (3, 8, 256))
        self._check(x, [85, 0, 0])

    def test_input_not_modified(self):
        x = torch.randn(3, 16, 64, device="cuda", dtype=torch.float16)
        snapshot = x.clone()
        self._check(x, [21, 0, 0])
        torch.testing.assert_close(x, snapshot)

    def test_empty_seq(self):
        x = torch.randn(3, 0, 64, device="cuda")
        out = MODULE.interleaved_rope(x, [21, 0, 0])
        self.assertEqual(out.shape, (0, 64))

    def test_grid_stride_fold_path(self):
        seq_len = 65536
        dim = 1024
        x = (torch.randn(3, seq_len, dim, device="cuda") * 2.0).to(
            torch.float16
        )
        self.assertGreater(seq_len * dim, 65535 * 1024)
        self._check(x, [241, 100, 0])


if __name__ == "__main__":
    unittest.main()
