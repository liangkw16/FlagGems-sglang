# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("sigmoid_gate_mul")

TOL = {
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1e-2),
    torch.float32: (1e-6, 1e-6),
}


def reference(x, gate):
    return (x.float() * torch.sigmoid(gate.float())).to(x.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class SigmoidGateMulTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        snapshots = [args[0].clone(), args[1].clone()]
        x = args[0]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.sigmoid_gate_mul(*args)
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=TOL[x.dtype][0],
                    atol=TOL[x.dtype][1],
                    equal_nan=True,
                )
                torch.testing.assert_close(
                    args[0], snapshots[0], rtol=0, atol=0, equal_nan=True
                )
                torch.testing.assert_close(
                    args[1], snapshots[1], rtol=0, atol=0
                )

    def test_dtypes_shapes(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for shape in ((4, 128), (3, 5, 64), (1,), (2049,), (8193, 33)):
                with self.subTest(dtype=dtype, shape=shape):
                    self.check(
                        (
                            torch.randn(shape, dtype=dtype, device="cuda"),
                            torch.randn(shape, dtype=dtype, device="cuda"),
                        )
                    )

    def test_special_values(self):
        for dtype in (torch.bfloat16, torch.float32):
            x = torch.randn(9, 65, dtype=dtype, device="cuda")
            g = torch.randn(9, 65, dtype=dtype, device="cuda")
            x[:, 0] = float("nan")
            g[:, 1] = float("inf")
            g[:, 2] = -float("inf")
            self.check((x, g))

    def test_empty(self):
        self.check(
            (
                torch.empty(0, 8, dtype=torch.float32, device="cuda"),
                torch.empty(0, 8, dtype=torch.float32, device="cuda"),
            )
        )


RELEASE_REQUIRED_TESTS = [
    "SigmoidGateMulTest.test_dtypes_shapes",
    "SigmoidGateMulTest.test_special_values",
    "SigmoidGateMulTest.test_empty",
]


if __name__ == "__main__":
    unittest.main()
