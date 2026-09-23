# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("indexed_scale_shift")


def reference(x, shift, scale, indices):
    idx = indices.long()
    sh = shift[idx].float()
    sc = scale[idx].float()
    one_plus = (1.0 + sc).to(torch.bfloat16).float()
    scaled = (x.float() * one_plus).to(torch.bfloat16).float()
    return (scaled + sh).to(x.dtype)


def bits(tensor):
    return tensor.contiguous().view(torch.uint8)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class IndexedScaleShiftTest(unittest.TestCase):
    def check(self, args, exact=False):
        expected = reference(*args)
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.indexed_scale_shift(*args)
                if exact:
                    torch.testing.assert_close(
                        bits(actual), bits(expected), rtol=0, atol=0
                    )
                else:
                    torch.testing.assert_close(
                        actual.float(),
                        expected.float(),
                        rtol=2e-2,
                        atol=2e-2,
                    )

    def test_shapes_and_variants(self):
        for rows, hdim, variants in (
            (1, 1, 1),
            (7, 1024, 5),
            (513, 5120, 256),
            (2049, 1023, 3),
        ):
            with self.subTest(shape=(rows, hdim, variants)):
                x = torch.randn(
                    rows, hdim, dtype=torch.bfloat16, device="cuda"
                )
                shift = torch.randn(
                    variants, hdim, dtype=torch.bfloat16, device="cuda"
                )
                scale = (
                    torch.randn(
                        variants, hdim, dtype=torch.bfloat16, device="cuda"
                    )
                    * 0.1
                )
                idx = torch.randint(
                    0, variants, (rows,), dtype=torch.int64, device="cuda"
                )
                self.check((x, shift, scale, idx))

    def test_double_round_boundary(self):
        # 1 + scale lands exactly on a bf16 halfway point, where the
        # explicit intermediate round is the contract. Near 1.0 the bf16
        # ULP is 2**-7, so 2**-8 ties 1.0 (even mantissa) against
        # 1 + 2**-7 (odd) and 3*2**-8 ties 1 + 2**-7 (odd) against
        # 1 + 2**-6 (even); below 1.0 the ULP halves, so -2**-9 ties
        # 1 - 2**-8 against 1.0. Each tie must resolve round-to-nearest
        # -even exactly like the reference (with x=1 and shift=0 the
        # output pins the one_plus round bit-for-bit).
        for scale_value in (2.0**-8, 3 * 2.0**-8, -(2.0**-9)):
            with self.subTest(scale=scale_value):
                x = torch.ones(16, 16, dtype=torch.bfloat16, device="cuda")
                scale = torch.full(
                    (1, 16), scale_value, dtype=torch.bfloat16, device="cuda"
                )
                shift = torch.zeros(1, 16, dtype=torch.bfloat16, device="cuda")
                idx = torch.zeros(16, dtype=torch.int32, device="cuda")
                self.check((x, shift, scale, idx), exact=True)

    def test_each_bf16_boundary(self):
        # Each tuple puts a different eager intermediate on a bf16 tie.
        for x_value, scale_value, shift_value in (
            (129 / 64, 1 / 256, 0),
            (129 / 64, 1 / 2, -1 / 128),
            (2, 0, 1 / 128),
            (2, 1 / 256, -2),
            (-2, 1 / 256, 2),
            (129 / 64, 1 / 2, -3),
            (-129 / 64, 1 / 2, 3),
        ):
            with self.subTest(values=(x_value, scale_value, shift_value)):
                x = torch.full(
                    (4, 17), x_value, dtype=torch.bfloat16, device="cuda"
                )
                scale = torch.full(
                    (2, 17),
                    scale_value,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                shift = torch.full(
                    (2, 17),
                    shift_value,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                idx = torch.ones(4, dtype=torch.int32, device="cuda")
                self.check((x, shift, scale, idx), exact=True)

    def test_sparse_table_and_second_tile_tail(self):
        x = torch.full((4, 1025), 2, dtype=torch.bfloat16, device="cuda")
        scale = torch.zeros(3, 1025, dtype=torch.bfloat16, device="cuda")
        shift = torch.zeros_like(scale)
        scale[1].fill_(0.5)
        scale[2].fill_(1)
        shift[1].fill_(-0.125)
        shift[2].fill_(0.25)
        idx = torch.tensor([0, 1, 2, 1], dtype=torch.int32, device="cuda")
        self.check((x, shift, scale, idx), exact=True)

        # A large logical table with two selected rows must not be materialized.
        scale = torch.full(
            (1, 17), 0.5, dtype=torch.bfloat16, device="cuda"
        ).expand(1_000_000, 17)
        shift = torch.zeros(1, 17, dtype=torch.bfloat16, device="cuda").expand(
            1_000_000, 17
        )
        idx = torch.tensor([0, 999_999], dtype=torch.int32, device="cuda")
        self.check((x[:2, :17], shift, scale, idx), exact=True)

    def test_row_grid_boundary(self):
        # Different values per row expose a missed or duplicated second pass.
        for rows, hdim in ((2047, 17), (2048, 17), (2049, 17), (3072, 4096)):
            with self.subTest(shape=(rows, hdim)):
                values = (torch.arange(rows, device="cuda") % 8).to(
                    torch.bfloat16
                )
                x = values[:, None].expand(rows, hdim).contiguous()
                scale = torch.zeros(
                    3, hdim, dtype=torch.bfloat16, device="cuda"
                )
                shift = torch.zeros_like(scale)
                scale[1].fill_(0.5)
                scale[2].fill_(1)
                shift[1].fill_(-0.125)
                shift[2].fill_(0.25)
                idx = (torch.arange(rows, device="cuda") % 3).to(torch.int32)
                self.check((x, shift, scale, idx), exact=True)

    def test_special_values(self):
        x = torch.tensor(
            [[float("inf"), float("-inf"), float("nan")]],
            dtype=torch.bfloat16,
            device="cuda",
        )
        shift = torch.zeros(1, 3, dtype=torch.bfloat16, device="cuda")
        scale = torch.zeros_like(shift)
        idx = torch.zeros(1, dtype=torch.int32, device="cuda")
        expected = reference(x, shift, scale, idx)
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.indexed_scale_shift(x, shift, scale, idx)
                torch.testing.assert_close(
                    bits(actual[:, :2]), bits(expected[:, :2]), rtol=0, atol=0
                )
                self.assertTrue(torch.isnan(actual[0, 2]))

    def test_int32_indices(self):
        x = torch.randn(33, 2048, dtype=torch.bfloat16, device="cuda")
        shift = torch.randn(7, 2048, dtype=torch.bfloat16, device="cuda")
        scale = torch.randn(7, 2048, dtype=torch.bfloat16, device="cuda")
        idx = torch.randint(0, 7, (33,), dtype=torch.int32, device="cuda")
        self.check((x, shift, scale, idx))


RELEASE_REQUIRED_TESTS = [
    "IndexedScaleShiftTest.test_shapes_and_variants",
    "IndexedScaleShiftTest.test_double_round_boundary",
    "IndexedScaleShiftTest.test_each_bf16_boundary",
    "IndexedScaleShiftTest.test_sparse_table_and_second_tile_tail",
    "IndexedScaleShiftTest.test_row_grid_boundary",
    "IndexedScaleShiftTest.test_special_values",
    "IndexedScaleShiftTest.test_int32_indices",
]


if __name__ == "__main__":
    unittest.main()
