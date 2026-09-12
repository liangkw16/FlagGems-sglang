# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch
import torch.nn.functional as F

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("gelu_tanh_and_mul")

TOL = {
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1e-2),
    torch.float32: (1e-5, 1e-5),
}


def reference(input):
    d = input.shape[-1] // 2
    x1, x3 = input[..., :d].float(), input[..., d:].float()
    return (F.gelu(x1, approximate="tanh") * x3).to(input.dtype)


def make_case(shape=(37, 512), dtype=torch.bfloat16):
    return torch.randn(shape, dtype=dtype, device="cuda") * 3


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class GeluTanhAndMulTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        snapshot = args[0].clone()
        x = args[0]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.gelu_tanh_and_mul(*args)
                self.assertEqual(actual.dtype, x.dtype)
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=TOL[x.dtype][0],
                    atol=TOL[x.dtype][1],
                    equal_nan=True,
                )
                torch.testing.assert_close(
                    x, snapshot, rtol=0, atol=0, equal_nan=True
                )

    def test_dtypes_and_widths(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for shape in ((4, 128), (3, 5, 1026), (1, 2), (2049, 64)):
                with self.subTest(dtype=dtype, shape=shape):
                    self.check((make_case(shape, dtype),))

    def test_wide_range(self):
        # large |x| drives the tanh argument to saturation
        x = torch.randn(9, 512, dtype=torch.float32, device="cuda") * 60
        self.check((x,))
        x2 = torch.randn(9, 512, dtype=torch.bfloat16, device="cuda") * 40
        self.check((x2,))

    def test_special_values(self):
        x = torch.randn(9, 64, dtype=torch.float32, device="cuda")
        x[:, 0] = float("nan")
        x[:, 1] = float("inf")
        x[:, 2] = -float("inf")
        self.check((x,))

    def test_empty(self):
        self.check((torch.empty(0, 8, dtype=torch.float32, device="cuda"),))


RELEASE_REQUIRED_TESTS = [
    "GeluTanhAndMulTest.test_dtypes_and_widths",
    "GeluTanhAndMulTest.test_wide_range",
    "GeluTanhAndMulTest.test_special_values",
    "GeluTanhAndMulTest.test_empty",
]


if __name__ == "__main__":
    unittest.main()
