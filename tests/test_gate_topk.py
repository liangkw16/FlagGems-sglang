# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("gate_topk")

TOL = {
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1e-2),
    torch.float32: (1e-6, 1e-6),
}


def reference(x, k):
    values, indices = torch.topk(x, k, dim=-1, sorted=True)
    return values, indices.to(torch.int32)


def make_case(
    rows=37,
    cols=100,
    k=8,
    dtype=torch.bfloat16,
    fill=None,
):
    if fill is None:
        x = torch.randn(rows, cols, dtype=dtype, device="cuda")
    else:
        x = torch.full((rows, cols), fill, dtype=dtype, device="cuda")
        x[:, 0] = 1.5
        x[:, 1] = 1.5
    return x, k


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class GateTopkTest(unittest.TestCase):
    def check(self, args):
        expected_values, _ = reference(*args)
        # torch.topk tie order is NOT the contract on CUDA (measured: it
        # returns the larger column first on bf16 ties about half the
        # time). The task text defines tie-break = smaller column index,
        # so derive the expected indices from a stable descending argsort.
        x = args[0]
        k = args[1]
        order = torch.argsort(x.float(), dim=-1, descending=True, stable=True)[
            :, :k
        ]
        expected_indices = order.to(torch.int32)
        snapshot = x.clone()
        for name, module in MODULES:
            with self.subTest(module=name):
                values, indices = module.gate_topk(*args)
                self.assertEqual(values.dtype, x.dtype)
                self.assertEqual(indices.dtype, torch.int32)
                torch.testing.assert_close(
                    values,
                    expected_values,
                    rtol=TOL[x.dtype][0],
                    atol=TOL[x.dtype][1],
                    equal_nan=True,
                )
                torch.testing.assert_close(
                    indices, expected_indices, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    x, snapshot, rtol=0, atol=0, equal_nan=True
                )

    def check_ties(self, args, fill):
        # torch.topk tie order is not guaranteed; verify the contract
        # directly: every row is [1.5, 1.5, fill, fill, ...], so the
        # descending top-k with smaller-index tie-break is exactly
        # [1.5, 1.5, fill...] over columns [0, 1, 2, ..., k-1].
        x, k = args
        rows = x.shape[0]
        for name, module in MODULES:
            with self.subTest(module=name):
                values, indices = module.gate_topk(*args)
                head = min(k, 2)
                expected_values = torch.full(
                    (rows, k), fill, dtype=x.dtype, device="cuda"
                )
                expected_values[:, :head] = 1.5
                expected_indices = torch.arange(
                    k, dtype=torch.int32, device="cuda"
                ).expand(rows, k)
                torch.testing.assert_close(
                    values, expected_values, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    indices, expected_indices, rtol=0, atol=0
                )

    def test_dtypes_and_k(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for k in (1, 3, 8, 32):
                with self.subTest(dtype=dtype, k=k):
                    self.check(make_case(dtype=dtype, k=k, cols=100))

    def test_column_tails(self):
        for cols in (1, 8, 31, 32, 33, 64, 96, 1024):
            with self.subTest(cols=cols):
                self.check(make_case(cols=cols, k=min(4, cols)))

    def test_rows_and_grid_edges(self):
        for rows in (1, 32, 33, 8193):
            with self.subTest(rows=rows):
                self.check(make_case(rows=rows, cols=64, k=8))

    def test_ties(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                self.check_ties(
                    make_case(rows=17, cols=64, k=5, dtype=dtype, fill=0.25),
                    fill=0.25,
                )
                self.check_ties(
                    make_case(rows=5, cols=33, k=32, dtype=dtype, fill=-2.0),
                    fill=-2.0,
                )

    def test_special_values(self):
        x = torch.randn(9, 64, dtype=torch.float32, device="cuda")
        x[:, :3] = torch.tensor(
            [float("inf"), -float("inf"), float("nan")], device="cuda"
        )
        self.check((x, 8))


RELEASE_REQUIRED_TESTS = [
    "GateTopkTest.test_dtypes_and_k",
    "GateTopkTest.test_column_tails",
    "GateTopkTest.test_rows_and_grid_edges",
    "GateTopkTest.test_ties",
    "GateTopkTest.test_special_values",
]


if __name__ == "__main__":
    unittest.main()
