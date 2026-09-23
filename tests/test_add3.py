# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("add3")


def reference(a, b, c):
    return (a + b) + c


def bits(tensor):
    return tensor.contiguous().view(torch.uint8)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class Add3Test(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        snapshots = [x.clone() for x in args]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.add3(*args)
                torch.testing.assert_close(
                    bits(actual), bits(expected), rtol=0, atol=0
                )
                for value, before in zip(args, snapshots):
                    torch.testing.assert_close(
                        bits(value), bits(before), rtol=0, atol=0
                    )

    def test_double_rounding_contract(self):
        # a=1, b=1/256, c=-1: the first add rounds to exactly 1.0
        # (RTNE picks the even neighbour), so the pair yields 0 while a
        # single-rounding fp32 accumulation would yield 1/256. The
        # fused kernel must reproduce the unfused pair bit-for-bit.
        args = [
            torch.tensor(values, dtype=torch.bfloat16, device="cuda")
            for values in (
                [1.0] * 16,
                [2.0**-8] * 16,
                [-1.0] * 16,
            )
        ]
        self.check(args)
        self.assertEqual(reference(*args)[0].item(), 0.0)

    def test_sizes_and_boundaries(self):
        for numel in (
            16,
            1024,
            1023 * 16 + 16,
            12 * 16384 - 16,
            12 * 16384,
            12 * 16384 + 16,
            1 << 20,
            (1 << 20) + 16,
        ):
            with self.subTest(numel=numel):
                self.check(
                    [
                        torch.randn(numel, dtype=torch.bfloat16, device="cuda")
                        for _ in range(3)
                    ]
                )

    def test_special_values_and_cancellation(self):
        # numel must stay a multiple of 16 (the task contract), so the
        # special lanes are padded with ordinary values.
        special = torch.tensor(
            [
                float("nan"),
                float("inf"),
                -float("inf"),
                -0.0,
                0.0,
                1.0,
                2.0,
                -2.0,
            ]
            * 2,
            dtype=torch.bfloat16,
            device="cuda",
        )
        base = torch.randn(16, dtype=torch.bfloat16, device="cuda")
        self.check([special, -special, base])
        # (+inf) + (+inf) and (large) + (-large) round-trip the pair.
        big = torch.full((64,), 3.0e38, dtype=torch.bfloat16, device="cuda")
        self.check([big, big, -big])
        self.check([big, -big, big])


RELEASE_REQUIRED_TESTS = [
    "Add3Test.test_double_rounding_contract",
    "Add3Test.test_sizes_and_boundaries",
    "Add3Test.test_special_values_and_cancellation",
]


if __name__ == "__main__":
    unittest.main()
