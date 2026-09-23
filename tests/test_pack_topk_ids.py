# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("pack_topk_ids")


def reference(topk_ids, topk_weights):
    weight_bits = (
        topk_weights.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        & 0xFFFF
    )
    return (topk_ids.to(torch.int32) << 16) | weight_bits


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class PackTopkIdsTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.pack_topk_ids(*args)
                torch.testing.assert_close(
                    actual, expected, rtol=0, atol=0
                )

    def test_sizes_and_bit_patterns(self):
        for shape in ((1, 8), (128, 8), (4096, 16), (1, 1)):
            with self.subTest(shape=shape):
                ids = torch.randint(
                    0, 1 << 15, shape, dtype=torch.int32, device="cuda"
                )
                w = torch.randn(shape, dtype=torch.float32, device="cuda")
                self.check((ids, w))

    def test_special_weight_bits(self):
        ids = torch.zeros(16, dtype=torch.int32, device="cuda")
        w = torch.tensor(
            [0.0, -0.0, 1.0, -1.0, 1e-40, 3e38, float("inf"),
             float("-inf"), float("nan")] + [0.5] * 7,
            dtype=torch.float32,
            device="cuda",
        )
        self.check((ids, w))

    def test_rounding_carry_bits(self):
        # exact f32 bit patterns pinning the integer round-to-nearest
        # -even form to torch's bfloat16 cast: ties with even/odd high
        # half lsb, the round-up that carries into the exponent, and a
        # tie inside the subnormal band
        patterns = [
            0x40008000,  # tie, lsb 0 -> round down (0x4000)
            0x40018000,  # tie, lsb 1 -> round up (0x4002)
            0x3F7FFFFF,  # just below 1.0 -> carries up to 0x3F80
            0xBF7FFFFF,  # negative mirror
            0x00008000,  # subnormal-band tie
            0x00017FFF,  # subnormal-band near tie
            0x3F800001,  # just above 1.0 -> truncates to 0x3F80
        ]
        ids = torch.arange(
            len(patterns), dtype=torch.int32, device="cuda"
        )
        signed = [
            p - (1 << 32) if p >= (1 << 31) else p for p in patterns
        ]
        w = (
            torch.tensor(signed, dtype=torch.int32, device="cuda")
            .view(torch.float32)
        )
        self.check((ids, w))


RELEASE_REQUIRED_TESTS = [
    "PackTopkIdsTest.test_sizes_and_bit_patterns",
    "PackTopkIdsTest.test_special_weight_bits",
    "PackTopkIdsTest.test_rounding_carry_bits",
]


if __name__ == "__main__":
    unittest.main()
