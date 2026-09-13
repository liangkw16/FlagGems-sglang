# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("residual_gate_add")

TOL = {
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1e-2),
    torch.float32: (1e-6, 1e-6),
}


def reference(residual, update, gate):
    product = (update.float() * gate.float()).to(residual.dtype)
    return (residual.float() + product.float()).to(residual.dtype)


def make_case(shape=(9, 128), dtype=torch.bfloat16, broadcast=False):
    r = torch.randn(shape, dtype=dtype, device="cuda")
    u = torch.randn(shape, dtype=dtype, device="cuda")
    d = shape[-1]
    g = (
        torch.randn((1,) * (len(shape) - 1) + (d,), dtype=dtype, device="cuda")
        if broadcast
        else torch.randn(shape, dtype=dtype, device="cuda")
    )
    return r, u, g


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class ResidualGateAddTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        snapshots = [a.clone() for a in args]
        r = args[0]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.residual_gate_add(*args)
                self.assertEqual(actual.dtype, r.dtype)
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=TOL[r.dtype][0],
                    atol=TOL[r.dtype][1],
                )
                for a, s in zip(args, snapshots):
                    torch.testing.assert_close(
                        a, s, rtol=0, atol=0, equal_nan=True
                    )

    def test_dtypes_and_broadcast(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for shape in ((4, 128), (3, 5, 64), (1, 1025), (2049, 8)):
                for broadcast in (False, True):
                    with self.subTest(
                        dtype=dtype, shape=shape, broadcast=broadcast
                    ):
                        self.check(make_case(shape, dtype, broadcast))

    def test_double_rounding(self):
        # fp16 products that actually round: magnitudes above fp16
        # precision must round to dtype BEFORE the add - compare bitwise
        # against the eager chain, not loosely.
        r = torch.full((8, 64), 0.1, dtype=torch.float16, device="cuda")
        u = torch.full((8, 64), 300.0, dtype=torch.float16, device="cuda")
        g = torch.full((8, 64), 300.0, dtype=torch.float16, device="cuda")
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.residual_gate_add(r, u, g)
                expected = reference(r, u, g)
                self.assertTrue(torch.equal(actual, expected))

    def test_cross_tile_broadcast_and_stride(self):
        # E4 regression: 97 rows exceeds the 24-program cap (second
        # grid-stride round); width 1025 is non-pow2 and 4100 crosses
        # the 4096 block boundary; the broadcast gate must index by
        # column on the 2D path.
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                shape = (97, 1025)
                r = torch.randn(shape, dtype=dtype, device="cuda")
                u = torch.randn(shape, dtype=dtype, device="cuda")
                gb = torch.randn((1, 1025), dtype=dtype, device="cuda")
                self.check((r, u, gb))
                gs = torch.randn(shape, dtype=dtype, device="cuda")
                self.check((r, u, gs))
        for d in (4095, 4096, 4097):
            with self.subTest(d=d):
                r = torch.randn(33, d, dtype=torch.float32, device="cuda")
                u = torch.randn(33, d, dtype=torch.float32, device="cuda")
                g = torch.randn(33, d, dtype=torch.float32, device="cuda")
                self.check((r, u, g))

    def test_double_rounding_precision(self):
        # fp16 1+2^-10 squared rounds; -(1+2^-9) + that rounds to exactly
        # 0 - skipping the intermediate rounding yields 2^-20 instead.
        r = torch.full(
            (8, 64), -(1 + 2**-9), dtype=torch.float16, device="cuda"
        )
        u = torch.full((8, 64), 1 + 2**-10, dtype=torch.float16, device="cuda")
        g = torch.full((8, 64), 1 + 2**-10, dtype=torch.float16, device="cuda")
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.residual_gate_add(r, u, g)
                expected = reference(r, u, g)
                torch.testing.assert_close(
                    actual, expected, rtol=1e-3, atol=1e-3
                )
                self.assertLessEqual(actual.abs().max().item(), 2**-19)

    def test_empty(self):
        self.check(make_case(shape=(0, 8)))


RELEASE_REQUIRED_TESTS = [
    "ResidualGateAddTest.test_dtypes_and_broadcast",
    "ResidualGateAddTest.test_double_rounding",
    "ResidualGateAddTest.test_cross_tile_broadcast_and_stride",
    "ResidualGateAddTest.test_double_rounding_precision",
    "ResidualGateAddTest.test_empty",
]


if __name__ == "__main__":
    unittest.main()
