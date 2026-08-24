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

BACKEND_ROOT = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "runtime"
    / "backend"
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_module(
    "chunk_cumsum_competition_module",
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "chunk_cumsum.py",
)
ASCEND_MODULE = _load_module(
    "chunk_cumsum_ascend_module",
    BACKEND_ROOT / "_ascend" / "ops" / "chunk_cumsum.py",
)
KUNLUN_MODULE = _load_module(
    "chunk_cumsum_kunlunxin_module",
    BACKEND_ROOT / "_kunlunxin" / "ops" / "chunk_cumsum.py",
)


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


def padded_reference(dt, a, chunk_size, dt_bias=None, dt_softplus=False):
    batch, seqlen, nheads = dt.shape
    nchunks = math.ceil(seqlen / chunk_size)
    dt_f = dt.float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float()
    if dt_softplus:
        dt_f = torch.where(dt_f <= 20.0, F.softplus(dt_f), dt_f)
    dt_f = dt_f.clamp(min=0.0)
    padded = torch.zeros(
        batch,
        nchunks * chunk_size,
        nheads,
        device=dt.device,
        dtype=torch.float32,
    )
    padded[:, :seqlen] = dt_f
    dt_out = (
        padded.reshape(batch, nchunks, chunk_size, nheads)
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
                for has_bias, softplus in (
                    (False, False),
                    (False, True),
                    (True, False),
                    (True, True),
                ):
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
                            nheads * 2, generator=generator, device="cuda"
                        )[::2]
                        bias = (
                            torch.randn(
                                nheads * 2,
                                generator=generator,
                                device="cuda",
                            )[::2]
                            if has_bias
                            else None
                        )
                        original = dt.clone()
                        original_a = a.clone()
                        original_bias = (
                            bias.clone() if bias is not None else None
                        )

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
                        torch.testing.assert_close(
                            a, original_a, atol=0.0, rtol=0.0
                        )
                        if bias is not None:
                            torch.testing.assert_close(
                                bias, original_bias, atol=0.0, rtol=0.0
                            )

    def test_partial_chunk_matches_fixed_upstream(self):
        generator = torch.Generator(device="cuda").manual_seed(10)
        dt = torch.randn(
            1,
            5,
            6,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )[:, :, ::2]
        a = -torch.rand(6, generator=generator, device="cuda")[::2]
        bias = torch.rand(6, generator=generator, device="cuda")[::2]

        actual = MODULE.chunk_cumsum(dt, a, 4, bias, True)
        expected = padded_reference(dt, a, 4, bias, True)

        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            actual[0][:, :, -1, 1:],
            torch.zeros_like(actual[0][:, :, -1, 1:]),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            actual[1][:, :, -1, 1:],
            actual[1][:, :, -1, :1].expand_as(actual[1][:, :, -1, 1:]),
            atol=0.0,
            rtol=0.0,
        )

    def test_softplus_threshold(self):
        dt = torch.tensor(
            [-30.0, -1.0, 0.0, 20.0, 21.0],
            device="cuda",
            dtype=torch.float32,
        ).reshape(1, 5, 1)
        a = torch.tensor([-0.5], device="cuda")

        actual = MODULE.chunk_cumsum(dt, a, 5, dt_softplus=True)
        expected = reference(dt, a, 5, dt_softplus=True)

        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)

    def test_empty_inputs(self):
        for shape in ((0, 8, 3), (1, 0, 3), (1, 8, 0)):
            with self.subTest(shape=shape):
                dt = torch.empty(shape, device="cuda", dtype=torch.float32)
                a = torch.empty(shape[2], device="cuda", dtype=torch.float32)

                actual = MODULE.chunk_cumsum(dt, a, 4)
                expected = reference(dt, a, 4)

                for got, want in zip(actual, expected):
                    self.assertEqual(got.shape, want.shape)
                    self.assertEqual(got.dtype, torch.float32)
                    self.assertEqual(got.numel(), 0)

    def test_vendors_cover_folded_grid(self):
        torch.manual_seed(20260824)
        batch, seqlen, nheads, chunk_size = 2, 16384, 96, 64
        nchunks = (seqlen + chunk_size - 1) // chunk_size
        block_h = 4
        total = ((nheads + block_h - 1) // block_h) * nchunks * batch
        self.assertGreater(total, 4096)

        for dtype, tolerance in (
            (torch.float32, 1e-4),
            (torch.float16, 1e-2),
        ):
            for softplus in (False, True):
                with self.subTest(dtype=dtype, softplus=softplus):
                    dt = torch.randn(
                        (batch, seqlen, nheads),
                        device="cuda",
                        dtype=dtype,
                    )
                    a = torch.randn((nheads,), device="cuda", dtype=dtype)
                    bias = torch.randn((nheads,), device="cuda", dtype=dtype)
                    expected = reference(
                        dt,
                        a,
                        chunk_size,
                        dt_bias=bias,
                        dt_softplus=softplus,
                    )
                    for name, module in (
                        ("generic", MODULE),
                        ("ascend", ASCEND_MODULE),
                        ("kunlunxin", KUNLUN_MODULE),
                    ):
                        with self.subTest(module=name):
                            actual = module.chunk_cumsum(
                                dt,
                                a,
                                chunk_size,
                                dt_bias=bias,
                                dt_softplus=softplus,
                            )
                            for out, exp in zip(actual, expected):
                                torch.testing.assert_close(
                                    out, exp, atol=tolerance, rtol=tolerance
                                )


if __name__ == "__main__":
    unittest.main()
