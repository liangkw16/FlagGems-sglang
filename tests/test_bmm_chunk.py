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
import math
import unittest
from pathlib import Path

import torch

MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "bmm_chunk.py"
)
SPEC = importlib.util.spec_from_file_location("bmm_chunk_module", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def reference(a, b, chunk_size, causal=False):
    batch, seqlen, ngroups, k = a.shape
    nchunks = math.ceil(seqlen / chunk_size)
    a_chunks = a.reshape(batch, nchunks, chunk_size, ngroups, k).float()
    b_chunks = b.reshape(batch, nchunks, chunk_size, ngroups, k).float()
    return torch.einsum("bcigk,bcjgk->bcgij", a_chunks, b_chunks)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class BmmChunkTest(unittest.TestCase):
    def test_dtypes_non_power_shapes_and_causal_match_reference(self):
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        generator = torch.Generator(device="cuda").manual_seed(20260824)

        for dtype, tolerance in tolerances.items():
            with self.subTest(dtype=dtype):
                a = torch.randn(
                    (2, 14, 3, 13),
                    device="cuda",
                    dtype=dtype,
                    generator=generator,
                )
                b = torch.randn(
                    (2, 14, 3, 13),
                    device="cuda",
                    dtype=dtype,
                    generator=generator,
                )
                a_before, b_before = a.clone(), b.clone()
                expected = reference(a, b, 7)

                outputs = (
                    MODULE.bmm_chunk(a, b, 7, causal=False),
                    MODULE.bmm_chunk(a, b, 7, causal=True),
                )

                for actual in outputs:
                    self.assertEqual(actual.shape, (2, 2, 3, 7, 7))
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(
                        actual,
                        expected,
                        atol=tolerance,
                        rtol=tolerance,
                    )
                torch.testing.assert_close(outputs[0], outputs[1])
                torch.testing.assert_close(a, a_before, atol=0, rtol=0)
                torch.testing.assert_close(b, b_before, atol=0, rtol=0)

    def test_noncontiguous_inputs_use_all_strides(self):
        a_base = torch.randn((2, 18, 6, 22), device="cuda")
        a = a_base[:, 1::2, ::2, 1::2]
        b_base = torch.randn((11, 3, 9, 2), device="cuda")
        b = b_base.permute(3, 2, 1, 0)
        self.assertEqual(a.shape, (2, 9, 3, 11))
        self.assertEqual(b.shape, a.shape)
        self.assertFalse(a.is_contiguous())
        self.assertFalse(b.is_contiguous())
        self.assertNotEqual(a.stride(), b.stride())
        a_before, b_before = a.clone(), b.clone()

        actual = MODULE.bmm_chunk(a, b, 3)
        expected = reference(a, b, 3)

        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(a, a_before, atol=0, rtol=0)
        torch.testing.assert_close(b, b_before, atol=0, rtol=0)

    def test_mixed_dtypes_keep_fp32_path(self):
        generator = torch.Generator(device="cuda").manual_seed(20260826)
        shape = (1, 66, 2, 65)
        a = torch.randn(
            shape,
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        b = torch.randn(
            shape,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )

        actual = MODULE.bmm_chunk(a, b, 33)
        expected = reference(a, b, 33)

        torch.testing.assert_close(actual, expected, atol=1.5e-2, rtol=1.5e-2)

    def test_tile_and_k_block_boundaries(self):
        generator = torch.Generator(device="cuda").manual_seed(20260825)
        cases = (
            (31, 17),
            (32, 17),
            (33, 17),
            (17, 31),
            (17, 32),
            (17, 33),
        )

        for chunk_size, k_size in cases:
            with self.subTest(chunk_size=chunk_size, k_size=k_size):
                shape = (1, 2 * chunk_size, 3, k_size)
                a = torch.randn(
                    shape,
                    device="cuda",
                    dtype=torch.float32,
                    generator=generator,
                )
                b = torch.randn(
                    shape,
                    device="cuda",
                    dtype=torch.float32,
                    generator=generator,
                )

                actual = MODULE.bmm_chunk(a, b, chunk_size)
                expected = reference(a, b, chunk_size)

                torch.testing.assert_close(
                    actual, expected, atol=1e-4, rtol=1e-4
                )

    def test_empty_sequence_returns_empty_fp32(self):
        a = torch.empty((2, 0, 3, 5), device="cuda", dtype=torch.float16)
        b = torch.empty_like(a)

        actual = MODULE.bmm_chunk(a, b, 3)
        expected = reference(a, b, 3)

        self.assertEqual(actual.shape, (2, 0, 3, 3, 3))
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected)

    def test_seqlen_must_be_divisible_by_chunk_size(self):
        a = torch.randn((1, 10, 2, 5), device="cuda")
        b = torch.randn_like(a)

        with self.assertRaisesRegex(ValueError, "divisible"):
            MODULE.bmm_chunk(a, b, 4)


if __name__ == "__main__":
    unittest.main()
