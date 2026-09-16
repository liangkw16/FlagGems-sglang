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

    def test_rounding_cancellation_contract(self):
        # Third-round review counterexamples: the exact products exceed
        # the dtype mantissa and the residual cancels the rounded
        # product to zero in eager. Without enable_fp_fusion=False the
        # kernel skipped the intermediate rounding and EXCEEDED the
        # per-dtype tolerance on all three dtypes (measured on the
        # proxy). These now require the platform criterion to hold.
        cases = [
            (torch.float16, -4104.0, 64.0625, 1e-3),
            (torch.bfloat16, -1040.0, 32.25, 1e-2),
            (torch.float32, -64.03125, 8.001953125, 1e-6),
        ]
        for dt, r0, u0, atol in cases:
            for rows in (2, 5):
                with self.subTest(dtype=dt):
                    r = torch.full((rows, 64), r0, dtype=dt, device="cuda")
                    u = torch.full((rows, 64), u0, dtype=dt, device="cuda")
                    gb = torch.full((1, 64), u0, dtype=dt, device="cuda")
                    for name, module in MODULES:
                        with self.subTest(module=name):
                            for g in (u, gb):
                                actual = module.residual_gate_add(r, u, g)
                                expected = reference(r, u, g)
                                torch.testing.assert_close(
                                    actual, expected, rtol=1e-3, atol=atol
                                )

    def test_grouped_row_and_dispatch_boundaries(self):
        shapes = (
            (3, 64),
            (4, 63),
            (5, 65),
            (23, 256),
            (24, 256),
            (25, 256),
            (95, 1023),
            (96, 1024),
            (97, 1025),
        )
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for shape in shapes:
                with self.subTest(dtype=dtype, shape=shape):
                    self.check(make_case(shape, dtype, True))

    def test_empty(self):
        self.check(make_case(shape=(0, 8)))

    def test_persistent_broadcast_boundaries(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for rows in (24, 25):
                for width in (4095, 4096, 4097):
                    with self.subTest(dtype=dtype, rows=rows, width=width):
                        self.check(make_case((rows, width), dtype, True))
            self.check(make_case((97, 8193), dtype, True))

    def test_broadcast_alias_and_fresh_output(self):
        # Input aliases are legal: only the freshly allocated output is written.
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            r = torch.randn((97, 4097), dtype=dtype, device="cuda")
            for update in (r, r.clone()):
                for gate in (r[:1], update[:1]):
                    args = (r, update, gate)
                    self.check(args)
                    for name, module in MODULES:
                        with self.subTest(dtype=dtype, module=name):
                            out = module.residual_gate_add(*args)
                            self.assertEqual(out.shape, r.shape)
                            self.assertTrue(out.is_contiguous())
                            for value in args:
                                self.assertNotEqual(
                                    out.data_ptr(), value.data_ptr()
                                )

    def test_persistent_broadcast_special_values(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            r, u, g = make_case((97, 4097), dtype, True)
            special = torch.tensor(
                [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1.0],
                dtype=dtype,
                device="cuda",
            )
            for value in (r, u, g):
                value.reshape(-1)[: special.numel()] = special
            expected = reference(r, u, g)
            snapshots = [value.clone() for value in (r, u, g)]
            for name, module in MODULES:
                with self.subTest(dtype=dtype, module=name):
                    actual = module.residual_gate_add(r, u, g)
                    torch.testing.assert_close(
                        actual,
                        expected,
                        rtol=TOL[dtype][0],
                        atol=TOL[dtype][1],
                        equal_nan=True,
                    )
                    for value, before in zip((r, u, g), snapshots):
                        torch.testing.assert_close(
                            value, before, rtol=0, atol=0, equal_nan=True
                        )


RELEASE_REQUIRED_TESTS = [
    "ResidualGateAddTest.test_grouped_row_and_dispatch_boundaries",
    "ResidualGateAddTest.test_dtypes_and_broadcast",
    "ResidualGateAddTest.test_double_rounding",
    "ResidualGateAddTest.test_cross_tile_broadcast_and_stride",
    "ResidualGateAddTest.test_double_rounding_precision",
    "ResidualGateAddTest.test_rounding_cancellation_contract",
    "ResidualGateAddTest.test_empty",
    "ResidualGateAddTest.test_persistent_broadcast_boundaries",
    "ResidualGateAddTest.test_broadcast_alias_and_fresh_output",
    "ResidualGateAddTest.test_persistent_broadcast_special_values",
]


if __name__ == "__main__":
    unittest.main()
