# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("tma_align_input_scale")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class TmaAlignInputScaleTest(unittest.TestCase):
    def check(self, m, k):
        x = torch.randn(m, k, dtype=torch.float32, device="cuda")
        for name, module in MODULES:
            with self.subTest(module=name, m=m, k=k):
                out = module.tma_align_input_scale(x)
                torch.testing.assert_close(out, x, rtol=0, atol=0)
                # column-major layout: stride(0) == 1 on the padded buffer
                self.assertEqual(out.stride(0), 1)
                self.assertEqual(out.shape, (m, k))

    def test_shapes_and_padding(self):
        for m, k in ((1, 8), (4, 16), (5, 33), (64, 128), (33, 4096), (7, 64)):
            with self.subTest(m=m, k=k):
                self.check(m, k)

    def test_column_major_input(self):
        base = torch.randn(4, 9, dtype=torch.float32, device="cuda")
        x = base.t()  # [9, 4] non-contiguous
        for name, module in MODULES:
            with self.subTest(module=name):
                out = module.tma_align_input_scale(x)
                torch.testing.assert_close(out, x, rtol=0, atol=0)
                self.assertEqual(out.stride(0), 1)

    def test_values_exact_and_padded_zero(self):
        x = torch.randn(9, 17, dtype=torch.float32, device="cuda")
        for name, module in MODULES:
            with self.subTest(module=name):
                out = module.tma_align_input_scale(x)
                torch.testing.assert_close(out, x, rtol=0, atol=0)


RELEASE_REQUIRED_TESTS = [
    "TmaAlignInputScaleTest.test_shapes_and_padding",
    "TmaAlignInputScaleTest.test_column_major_input",
    "TmaAlignInputScaleTest.test_values_exact_and_padded_zero",
]

if __name__ == "__main__":
    unittest.main()
