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

    def test_signed_zero_stable_indices_and_value_bits(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for cols, k in ((2, 1), (2, 2), (33, 32)):
                x = torch.zeros((2, cols), dtype=dtype, device="cuda")
                x[0, ::2] = -0.0
                x[1, 1::2] = -0.0
                expected_indices = torch.arange(
                    k, dtype=torch.int32, device="cuda"
                ).expand(2, k)
                expected_values = x[:, :k].contiguous()
                snapshot = x.clone()
                for name, module in MODULES:
                    with self.subTest(
                        module=name, dtype=dtype, cols=cols, k=k
                    ):
                        values, indices = module.gate_topk(x, k)
                        self.assertEqual(values.dtype, dtype)
                        self.assertEqual(indices.dtype, torch.int32)
                        torch.testing.assert_close(
                            indices, expected_indices, rtol=0, atol=0
                        )
                        self.assertTrue(
                            torch.equal(
                                values.contiguous().view(torch.uint8),
                                expected_values.view(torch.uint8),
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                x.view(torch.uint8), snapshot.view(torch.uint8)
                            )
                        )

    def test_packed_index_width_boundary(self):
        # At 65505 columns, rounding to a 32-column tile gives N_PAD=65536.
        # Its column-zero tag needs 17 bits and must not overlap value bits.
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for cols in (65504, 65505, 65536, 65537):
                x = torch.full((1, cols), -1.0, dtype=dtype, device="cuda")
                x[0, 0] = 1.0
                snapshot = x.clone()
                for name, module in MODULES:
                    with self.subTest(module=name, dtype=dtype, cols=cols):
                        values, indices = module.gate_topk(x, 1)
                        self.assertEqual(values.dtype, dtype)
                        self.assertEqual(indices.dtype, torch.int32)
                        torch.testing.assert_close(
                            indices, torch.zeros_like(indices), rtol=0, atol=0
                        )
                        self.assertTrue(
                            torch.equal(
                                values.view(torch.uint8),
                                x[:, :1].contiguous().view(torch.uint8),
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                x.view(torch.uint8), snapshot.view(torch.uint8)
                            )
                        )
        # Wide keys must still keep stable ties and indices above 65535.
        x = torch.full((1, 65537), -1.0, dtype=torch.float32, device="cuda")
        x[0, 0] = x[0, 1] = 1.0
        x[0, -1] = 2.0
        for name, module in MODULES:
            with self.subTest(module=name, wide_ties=True):
                values, indices = module.gate_topk(x, 3)
                torch.testing.assert_close(
                    indices,
                    torch.tensor(
                        [[65536, 0, 1]], dtype=torch.int32, device="cuda"
                    ),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    values, x[:, [65536, 0, 1]], rtol=0, atol=0
                )

    def test_nan_sign_payload_padding_and_stable_indices(self):
        # Sort all NaNs as one numeric class, then by smaller column.
        # Compare values as bytes so canonicalizing the sorting key cannot
        # silently canonicalize a selected NaN sign or payload in the output.
        formats = (
            (torch.float16, torch.int16, 16, 0x7E01, 0x7E02, 0x7C00, 0xBC00),
            (torch.bfloat16, torch.int16, 16, 0x7FC1, 0x7FC2, 0x7F80, 0xBF80),
            (
                torch.float32,
                torch.int32,
                32,
                0x7FC00001,
                0x7FC00002,
                0x7F800000,
                0xBF800000,
            ),
        )
        for dtype, int_dtype, width, nan1, nan2, inf, minus_one in formats:
            sign = 1 << (width - 1)
            negative_nan1 = nan1 | sign
            negative_nan2 = nan2 | sign
            mixed = [negative_nan2, inf, nan1, inf | sign, nan2, negative_nan1]
            cases = [
                ("singleton", [negative_nan1], 1, [0]),
                ("mixed_k1", mixed, 1, [0]),
                ("mixed_k3", mixed, 3, [0, 2, 4]),
            ]
            for cols, nan_columns in ((31, (0, 15, 30)), (33, (0, 30, 32))):
                raw = [minus_one] * cols
                raw[1] = inf
                raw[2] = inf | sign
                for col, payload in zip(
                    nan_columns, (negative_nan2, nan1, nan2)
                ):
                    raw[col] = payload
                cases.append((f"tail{cols}_k1", raw, 1, [nan_columns[0]]))
                cases.append((f"tail{cols}_k3", raw, 3, list(nan_columns)))
            for case_name, raw, k, expected_columns in cases:
                signed_raw = [
                    value - (1 << width) if value & sign else value
                    for value in raw
                ]
                x = torch.tensor(
                    [signed_raw], dtype=int_dtype, device="cuda"
                ).view(dtype)
                snapshot = x.clone()
                expected_indices = torch.tensor(
                    [expected_columns], dtype=torch.int32, device="cuda"
                )
                expected_values = x[:, expected_columns].contiguous()
                for name, module in MODULES:
                    with self.subTest(
                        module=name, dtype=dtype, case=case_name
                    ):
                        values, indices = module.gate_topk(x, k)
                        self.assertEqual(values.dtype, dtype)
                        self.assertEqual(indices.dtype, torch.int32)
                        torch.testing.assert_close(
                            indices, expected_indices, rtol=0, atol=0
                        )
                        self.assertTrue(
                            torch.equal(
                                values.contiguous().view(torch.uint8),
                                expected_values.view(torch.uint8),
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                x.view(torch.uint8), snapshot.view(torch.uint8)
                            )
                        )

        # Reloading selected values must never launch for an impossible k.
        # Match the existing vendor precondition, including empty rows/cols.
        for rows, cols, k in ((1, 0, 1), (0, 0, 1), (1, 1, 2), (0, 1, 2)):
            x = torch.empty((rows, cols), dtype=torch.float32, device="cuda")
            for name, module in MODULES:
                with self.subTest(
                    module=name, invalid_shape=(rows, cols), k=k
                ):
                    with self.assertRaises(AssertionError):
                        module.gate_topk(x, k)


RELEASE_REQUIRED_TESTS = [
    "GateTopkTest.test_dtypes_and_k",
    "GateTopkTest.test_column_tails",
    "GateTopkTest.test_rows_and_grid_edges",
    "GateTopkTest.test_ties",
    "GateTopkTest.test_special_values",
    "GateTopkTest.test_signed_zero_stable_indices_and_value_bits",
    "GateTopkTest.test_packed_index_width_boundary",
    "GateTopkTest.test_nan_sign_payload_padding_and_stable_indices",
]


if __name__ == "__main__":
    unittest.main()
