# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("fill_padded_rows")


def reference(x, num_token_non_padded, fill_value):
    out = x.clone()
    n = int(num_token_non_padded)
    if n < out.shape[0]:
        out[n:] = fill_value
    return out


def make_case(
    rows=9,
    cols=1025,
    dtype=torch.bfloat16,
    n_valid=None,
    fill_value=-1.5,
    strided=False,
    n_dtype=torch.int32,
):
    x = torch.randn(rows, cols, dtype=dtype, device="cuda")
    if strided:
        storage = torch.randn(rows * 2 + 1, cols, dtype=dtype, device="cuda")
        x = storage[1::2]
    n = torch.tensor(
        rows if n_valid is None else n_valid, dtype=n_dtype, device="cuda"
    )
    return x, n, fill_value


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class FillPaddedRowsTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        snapshots = [args[0].clone(), args[1].clone()]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.fill_padded_rows(*args)
                torch.testing.assert_close(
                    actual, expected, rtol=0, atol=0, equal_nan=True
                )
                torch.testing.assert_close(
                    args[0], snapshots[0], rtol=0, atol=0, equal_nan=True
                )
                torch.testing.assert_close(
                    args[1], snapshots[1], rtol=0, atol=0, equal_nan=True
                )
                if args[0].numel():
                    self.assertNotEqual(actual.data_ptr(), args[0].data_ptr())

    def test_dtypes_and_fill_values(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for fill in (0.0, -1.5, 7):
                with self.subTest(dtype=dtype, fill=fill):
                    self.check(make_case(dtype=dtype, fill_value=fill))

    def test_boundary_counts(self):
        for n_valid in (0, 1, 4, 8, 9, 14):
            with self.subTest(n_valid=n_valid):
                self.check(make_case(n_valid=n_valid))
        for n_dtype in (torch.int32, torch.int64):
            with self.subTest(n_dtype=n_dtype):
                self.check(make_case(n_valid=4, n_dtype=n_dtype))

    def test_strides_and_widths(self):
        for cols in (1, 511, 512, 513, 4096, 7168):
            with self.subTest(cols=cols):
                self.check(make_case(cols=cols, n_valid=3))
        self.check(make_case(cols=1025, strided=True, n_valid=4))
        self.check(make_case(rows=8193, cols=257, n_valid=8192))

    def test_empty(self):
        self.check(make_case(rows=0, cols=513))
        self.check(make_case(rows=4, cols=0, n_valid=2))


RELEASE_REQUIRED_TESTS = [
    "FillPaddedRowsTest.test_dtypes_and_fill_values",
    "FillPaddedRowsTest.test_boundary_counts",
    "FillPaddedRowsTest.test_strides_and_widths",
    "FillPaddedRowsTest.test_empty",
]


if __name__ == "__main__":
    unittest.main()
