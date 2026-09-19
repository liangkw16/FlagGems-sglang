# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("relu2")


def reference(input):
    x = torch.relu(input.float())
    return (x * x).to(input.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class Relu2Test(unittest.TestCase):
    def check(self, x):
        want = reference(x)
        for name, module in MODULES:
            with self.subTest(module=name):
                got = module.relu2(x)
                torch.testing.assert_close(
                    got.float(), want.float(), rtol=2e-2, atol=2e-2,
                    equal_nan=(torch.isnan(want).any().item()),
                )

    def test_shapes_and_specials(self):
        for shape in ((1, 1), (7, 1024), (513, 5120), (2049, 3584)):
            with self.subTest(shape=shape):
                self.check(
                    torch.randn(shape, dtype=torch.bfloat16, device="cuda")
                )
        x = torch.randn(64, 64, dtype=torch.float16, device="cuda")
        self.check(x)
        x = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
        x[0, :6] = torch.tensor(
            [0.0, -0.0, -5.0, 1e-40, 3e38, float("nan")], device="cuda"
        )
        self.check(x)


RELEASE_REQUIRED_TESTS = [
    "Relu2Test.test_shapes_and_specials",
]


if __name__ == "__main__":
    unittest.main()
