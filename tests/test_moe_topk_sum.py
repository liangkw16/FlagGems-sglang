# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("moe_topk_sum")


def reference(x, out):
    return x.float().sum(dim=1).to(out.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class MoeTopkSumTest(unittest.TestCase):
    def check(self, x):
        out = torch.empty(
            x.shape[0], x.shape[2], dtype=x.dtype, device=x.device
        )
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.moe_topk_sum(x, out.clone())
                torch.testing.assert_close(
                    actual, reference(x, out), rtol=2e-2, atol=2e-2
                )

    def test_shapes_and_topk(self):
        for rows, topk, hdim in (
            (1, 1, 1),
            (7, 4, 1024),
            (128, 8, 5120),
            (513, 3, 1025),
            (2049, 16, 3584),
        ):
            with self.subTest(shape=(rows, topk, hdim)):
                self.check(
                    torch.randn(
                        rows, topk, hdim, dtype=torch.bfloat16,
                        device="cuda",
                    )
                )

    def test_cancellation(self):
        x = torch.randn(64, 8, 2048, dtype=torch.bfloat16, device="cuda")
        x[:, 1] = -x[:, 0]
        self.check(x)


RELEASE_REQUIRED_TESTS = [
    "MoeTopkSumTest.test_shapes_and_topk",
    "MoeTopkSumTest.test_cancellation",
]


if __name__ == "__main__":
    unittest.main()
