# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch
import torch.nn.functional as F

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("group_norm_silu")

TOL = {
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1.5e-2, 1.5e-2),
    torch.float32: (1e-4, 1e-4),
}


def reference(x, weight, bias, num_groups, eps):
    y = F.group_norm(x.float(), num_groups, weight.float(), bias.float(), eps)
    return F.silu(y).to(x.dtype)


def make_case(
    shape=(4, 32, 8, 8), num_groups=8, dtype=torch.bfloat16, off=0.0
):
    x = torch.randn(shape, dtype=dtype, device="cuda") + off
    c = shape[1]
    w = torch.randn(c, dtype=dtype, device="cuda")
    b = torch.randn(c, dtype=dtype, device="cuda")
    return x, w, b, num_groups, 1e-5


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class GroupNormSiluTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        snapshots = [args[0].clone(), args[1].clone(), args[2].clone()]
        x = args[0]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.group_norm_silu(*args)
                self.assertEqual(actual.dtype, x.dtype)
                self.assertEqual(actual.shape, x.shape)
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=TOL[x.dtype][0],
                    atol=TOL[x.dtype][1],
                )
                for a, s in zip(args[:3], snapshots):
                    torch.testing.assert_close(
                        a, s, rtol=0, atol=0, equal_nan=True
                    )

    def test_dtypes_and_shapes(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for shape, g in (
                ((4, 32, 8, 8), 8),
                ((2, 640, 4, 4), 32),
                (
                    (
                        3,
                        8,
                    ),
                    4,
                ),
                ((5, 12, 1, 1, 7), 6),
                ((7, 16, 130), 2),
            ):
                with self.subTest(dtype=dtype, shape=shape, groups=g):
                    self.check(make_case(shape, g, dtype))

    def test_large_mean_small_var(self):
        # E[x^2] - mean^2 cancellation guard (fp32 1e-4 tolerance):
        # a realistic biased-activation range (mean 100, unit spread).
        # Extreme 1e6-range inputs would also diverge between two fp32
        # summation orders in the reference itself.
        x, w, b, g, eps = make_case((4, 64, 16, 16), 8, torch.float32)
        x = x + 100.0
        self.check((x, w, b, g, eps))

    def test_groups_edges(self):
        self.check(make_case((3, 8, 5), 1))
        self.check(make_case((3, 8, 5), 8))

    def test_resident_tile_boundaries(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for spatial in (2047, 2048, 2049):
                with self.subTest(dtype=dtype, group_elements=spatial):
                    self.check(make_case((2, 2, spatial), 2, dtype))
            # Three channels per group: padded channel lanes must not
            # read the next group's input or affine parameters.
            for spatial in (511, 513):
                with self.subTest(
                    dtype=dtype, group_channels=3, spatial=spatial
                ):
                    self.check(make_case((2, 6, spatial), 2, dtype))
        self.check(make_case((2, 6, 257), 2, torch.float32, off=100.0))

    def test_empty(self):
        self.check(make_case((0, 8, 4), 4))


RELEASE_REQUIRED_TESTS = [
    "GroupNormSiluTest.test_dtypes_and_shapes",
    "GroupNormSiluTest.test_large_mean_small_var",
    "GroupNormSiluTest.test_groups_edges",
    "GroupNormSiluTest.test_resident_tile_boundaries",
    "GroupNormSiluTest.test_empty",
]


if __name__ == "__main__":
    unittest.main()
