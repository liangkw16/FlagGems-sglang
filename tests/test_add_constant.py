# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("add_constant")


def reference(src, constant):
    return src + constant


def bits(tensor):
    return tensor.contiguous().view(torch.uint8)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class AddConstantTest(unittest.TestCase):
    def check(self, src, constant):
        expected = reference(src, constant)
        snapshot = src.clone()
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.add_constant(src, constant)
                torch.testing.assert_close(
                    bits(actual), bits(expected), rtol=0, atol=0
                )
                self.assertIsNot(actual, src)
                torch.testing.assert_close(
                    bits(src), bits(snapshot), rtol=0, atol=0
                )

    def make(self, numel, *, start=-1000, step=7):
        values = [(start + step * i + 2**31) % 2**32 - 2**31 for i in range(numel)]
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    def test_tiny_and_tail_boundaries(self):
        for numel in (1, 2, 3, 15, 16, 17, 1023, 1024, 1025, 4095, 4096, 4097):
            with self.subTest(numel=numel):
                self.check(self.make(numel), 5)

    def test_large_crossing_persistent_cap(self):
        for numel in (2**20 + 1, 3 * 2**20 + 7):
            with self.subTest(numel=numel):
                self.check(self.make(numel), -12345)

    def test_constant_values(self):
        src = self.make(257)
        for constant in (0, 1, -1, 255, 2**20, -(2**20), 2**31 - 1, -(2**31)):
            with self.subTest(constant=constant):
                self.check(src, constant)

    def test_int32_wraparound_matches_reference(self):
        # saturate the top of the range: src + c must wrap modulo 2**32
        # exactly like the torch reference, not clamp.
        src = torch.full((64,), 2**31 - 1, dtype=torch.int32, device="cuda")
        self.check(src, 1)
        low = torch.full((64,), -(2**31), dtype=torch.int32, device="cuda")
        self.check(low, -1)

    def test_two_segment_block_boundaries(self):
        # regression for the two-segment no-mask hot path (_ascend,
        # BLOCK=16384): numel at B-1/B/B+1 and 2B-1/2B/2B+1 straddles
        # the scalar branch, and exact multiples (32768, 65536) run a
        # grid with no tail program at all - pure unmasked tiles.
        for numel in (16383, 16384, 16385, 32767, 32768, 32769, 65536):
            with self.subTest(numel=numel):
                self.check(self.make(numel), 7)


RELEASE_REQUIRED_TESTS = [
    "AddConstantTest.test_tiny_and_tail_boundaries",
    "AddConstantTest.test_large_crossing_persistent_cap",
    "AddConstantTest.test_constant_values",
    "AddConstantTest.test_int32_wraparound_matches_reference",
    "AddConstantTest.test_two_segment_block_boundaries",
]

if __name__ == "__main__":
    unittest.main()
