# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("clamp_position")


def reference(seq_lens):
    return (seq_lens - 1).clamp(min=0)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class ClampPositionTest(unittest.TestCase):
    def check(self, x):
        before = x.clone()
        expected = reference(x)
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.clamp_position(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x, before, rtol=0, atol=0)
                if x.numel():
                    self.assertNotEqual(actual.data_ptr(), x.data_ptr())

    def test_integer_limits(self):
        for dtype in (torch.int32, torch.int64):
            limits = torch.iinfo(dtype)
            values = [limits.min, limits.min + 1, -2, -1, 0, 1, 2, limits.max]
            if dtype == torch.int64:
                values += [2**32, 2**40 + 1]
            with self.subTest(dtype=dtype):
                self.check(torch.tensor(values, device="cuda", dtype=dtype))

    def test_tile_boundaries_and_empty(self):
        for dtype in (torch.int32, torch.int64):
            for n in (0, 1, 255, 256, 257, 1025, 32768, 65537):
                with self.subTest(dtype=dtype, n=n):
                    self.check(torch.arange(n, device="cuda", dtype=dtype))

    def test_strided_and_repeated_calls(self):
        for dtype in (torch.int32, torch.int64):
            storage = torch.arange(1027, device="cuda", dtype=dtype)
            x = storage[1::2]
            self.check(x)
            x.fill_(1)
            self.check(x)
            x.zero_()
            self.check(x)


RELEASE_REQUIRED_TESTS = [
    "ClampPositionTest.test_integer_limits",
    "ClampPositionTest.test_tile_boundaries_and_empty",
    "ClampPositionTest.test_strided_and_repeated_calls",
]


if __name__ == "__main__":
    unittest.main()
