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
    / "chunk_local_cumsum_vector.py"
)
SPEC = importlib.util.spec_from_file_location(
    "chunk_local_cumsum_vector_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def reference(g, chunk_size, reverse=False, scale=None):
    batch, seqlen, nheads, state = g.shape
    chunks = seqlen // chunk_size
    values = g.float().view(batch, chunks, chunk_size, nheads, state)
    if reverse:
        values = values.flip(2)
    output = values.cumsum(2)
    if scale is not None:
        output = output * scale
    if reverse:
        output = output.flip(2)
    return output.reshape(batch, seqlen, nheads, state)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class ChunkLocalCumsumVectorTest(unittest.TestCase):
    def test_matches_reference(self):
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        generator = torch.Generator(device="cuda").manual_seed(20260824)
        cases = ((2, 30, 3, 5, 5), (1, 256, 8, 16, 64))
        for dtype, tolerance in tolerances.items():
            for shape in cases:
                batch, seqlen, nheads, state, chunk_size = shape
                for reverse in (False, True):
                    for scale in (None, -0.25, 2.0):
                        with self.subTest(
                            dtype=dtype,
                            shape=shape,
                            reverse=reverse,
                            scale=scale,
                        ):
                            base = torch.randn(
                                batch,
                                seqlen,
                                nheads,
                                state * 2,
                                generator=generator,
                                device="cuda",
                                dtype=dtype,
                            )
                            g = base[..., ::2]
                            original = g.clone()

                            actual = MODULE.chunk_local_cumsum_vector(
                                g, chunk_size, reverse, scale
                            )
                            expected = reference(g, chunk_size, reverse, scale)

                            self.assertEqual(actual.dtype, torch.float32)
                            self.assertEqual(actual.shape, g.shape)
                            torch.testing.assert_close(
                                actual,
                                expected,
                                atol=tolerance,
                                rtol=tolerance,
                            )
                            torch.testing.assert_close(
                                g, original, atol=0.0, rtol=0.0
                            )

    def test_empty_input(self):
        g = torch.empty((2, 0, 3, 5), device="cuda")

        actual = MODULE.chunk_local_cumsum_vector(g, 5)

        self.assertEqual(actual.shape, g.shape)
        self.assertEqual(actual.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
