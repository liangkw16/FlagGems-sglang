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
import torch.nn.functional as F

MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "chunk_cumsum.py"
)
SPEC = importlib.util.spec_from_file_location(
    "chunk_cumsum_competition_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def reference(dt, a, chunk_size, dt_bias=None, dt_softplus=False):
    batch, seqlen, nheads = dt.shape
    nchunks = math.ceil(seqlen / chunk_size)
    dt_f = dt.float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float()
    if dt_softplus:
        dt_f = torch.where(dt_f <= 20.0, F.softplus(dt_f), dt_f)
    dt_f = dt_f.clamp(min=0.0)
    dt_out = (
        dt_f.reshape(batch, nchunks, chunk_size, nheads)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    return dt_out, (dt_out * a.float()[None, :, None, None]).cumsum(-1)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class ChunkCumsumTest(unittest.TestCase):
    def test_matches_reference(self):
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        generator = torch.Generator(device="cuda").manual_seed(20260824)
        cases = ((2, 30, 7, 5), (1, 256, 32, 64), (3, 96, 9, 16))
        for dtype, tolerance in tolerances.items():
            for batch, seqlen, nheads, chunk_size in cases:
                for has_bias, softplus in ((False, False), (True, True)):
                    with self.subTest(
                        dtype=dtype,
                        shape=(batch, seqlen, nheads),
                        chunk_size=chunk_size,
                        has_bias=has_bias,
                        softplus=softplus,
                    ):
                        dt = torch.randn(
                            batch,
                            seqlen,
                            nheads * 2,
                            generator=generator,
                            device="cuda",
                            dtype=dtype,
                        )[:, :, ::2]
                        a = -torch.rand(
                            nheads, generator=generator, device="cuda"
                        )
                        bias = (
                            torch.randn(
                                nheads, generator=generator, device="cuda"
                            )
                            if has_bias
                            else None
                        )
                        original = dt.clone()

                        actual = MODULE.chunk_cumsum(
                            dt, a, chunk_size, bias, softplus
                        )
                        expected = reference(dt, a, chunk_size, bias, softplus)

                        self.assertEqual(len(actual), 2)
                        for got, want in zip(actual, expected):
                            self.assertEqual(got.dtype, torch.float32)
                            self.assertEqual(got.shape, want.shape)
                            torch.testing.assert_close(
                                got, want, atol=tolerance, rtol=tolerance
                            )
                        torch.testing.assert_close(
                            dt, original, atol=0.0, rtol=0.0
                        )


if __name__ == "__main__":
    unittest.main()
