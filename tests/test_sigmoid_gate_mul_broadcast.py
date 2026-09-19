# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("sigmoid_gate_mul_broadcast")


def reference(x, gate):
    g = torch.sigmoid(gate.reshape(-1, 1).float())
    return (x.float() * g).to(x.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class SGMBTest(unittest.TestCase):
    def check(self, x, gate):
        want = reference(x, gate)
        for name, module in MODULES:
            with self.subTest(module=name):
                got = module.sigmoid_gate_mul_broadcast(x, gate)
                torch.testing.assert_close(
                    got.float(), want.float(), rtol=2e-2, atol=2e-2
                )

    def test_shapes_and_saturation(self):
        for n, d in ((1, 1), (7, 1024), (513, 5120), (2049, 1023)):
            with self.subTest(shape=(n, d)):
                x = torch.randn(n, d, dtype=torch.bfloat16, device="cuda")
                gate = torch.randn(n, 1, dtype=torch.bfloat16, device="cuda") * 3
                self.check(x, gate)
        x = torch.randn(32, 2048, dtype=torch.float16, device="cuda")
        gate = torch.tensor(
            [0.0, -100.0, 100.0] * 10 + [0.0, 0.0], dtype=torch.float16, device="cuda"
        ).reshape(32, 1)
        self.check(x, gate)


RELEASE_REQUIRED_TESTS = [
    "SGMBTest.test_shapes_and_saturation",
]


if __name__ == "__main__":
    unittest.main()
